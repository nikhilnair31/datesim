from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import pytesseract
import requests
import sqlite3
import openai
import base64
import time
import os
import io

# Set your API key here
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')

# Path to the Tesseract executable, update this if Tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Set db path
db_path = 'data/data.db'
# UI
curr_username = None
uploaded_files = None
# Vars
max_tokens = 512
temperature = 0.0
img_num = 1
max_messages = 1
# Models
ocr_to_conv_model_name = "gpt-4-turbo-preview"
chat_model_name = "openchat/openchat-3.5-1210"
# System Prompts
ocr_to_conv_system_message = """
You are an OCR to conversation formatter. You will receive the OCR text of a WhatsApp conversation and are to structure it in the format below. The OCR text will contain weird text artifacts but just focus on what could be part of the conversation. 
user: <content>
person: <content> 
If the conversation cannot be determined return "-"
"""
username_system_prompt = """
You are a person chatting with another on a dating app. Only reply for yourself.
Given below is an example of your texting style.
NEVER break character or mention that you're an AI model.
"""

def run_script_on_db(sql_script, data = None, action_type="insert"):
    """
    Executes a SQL script on the database with optional data.

    Parameters:
    - sql_script: The SQL command to execute.
    - data: Optional data to pass with the SQL command.
    - action_type: The type of action to perform ('insert' or 'check').

    Returns:
    - The result of the fetchall() if action_type is 'check', otherwise None.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        if data:
            c.execute(sql_script, data)
        else:
            c.execute(sql_script)

        # Handle actions accordingly
        if action_type == "check":
            result = c.fetchall()
            return result
        elif action_type == "insert":
            conn.commit()
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        # Optionally, raise an exception or handle it as needed
    
    finally:
        # Ensure the connection is closed even if an error occurs
        if conn:
            conn.close()
def update_or_insert_data(username, formatted_texts):
    # Check if the username exists and usernames_simmed is empty
    print(f"Checking existing username...")
    query = "SELECT DISTINCT username FROM info WHERE username = ?"
    data = (username,)
    existing_user = run_script_on_db( query, data, action_type="check" )
    existing_user = existing_user[0][0]
    print(f"Username checked!: {existing_user}")

    if existing_user:
        print(f"Updating existing record...")
        query = "UPDATE info SET conv_style = ?, ran_sim = ?, usernames_simmed = ? WHERE username = ?"
        data = (str(formatted_texts), False, "[]", username)
        run_script_on_db( query, data, action_type="insert" )
        print(f"Updated existing record!: {data}")
    else:
        print(f"Inserting new record...")
        query = "INSERT INTO info (username, conv_style, ran_sim, usernames_simmed) VALUES (?, ?, ?, ?)"
        data = (username, str(formatted_texts), False, "[]")
        run_script_on_db( query, data, action_type="insert" )
def initialize_db():
    print('Creating DB...')
    run_script_on_db(
        '''
        CREATE TABLE IF NOT EXISTS info 
        (username TEXT, conv_style TEXT, ran_sim BOOLEAN, usernames_simmed TEXT)
        '''
    )
    print('DB created!\n')
    
def make_openai_call(model_name, system_message, user_message):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}',
    }
    messages = [
        {
            'role': 'system', 
            'content': system_message
        },
        {
            'role': 'user', 
            'content': user_message
        }
    ]
    data = {
        'messages': messages,
        'model': model_name,
        'max_tokens': max_tokens,
        "temperature": temperature,
        'seed': 48
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', json=data, headers=headers)
    result_text = response.json()['choices'][0]['message']['content']

    return result_text
def make_together_call(model_name, messages_list):
    payload = {
        "model": model_name,
        "messages": messages_list,
        "max_tokens": 256,
        "temperature": 0.9
    }

    model_name = payload["model"]
    if "gpt" in model_name:
        url = "https://api.openai.com/v1/chat/completions"
        api_key = openai_api_key
    else:
        url = "https://api.together.xyz/v1/chat/completions"
        api_key = together_api_key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    # print(response.json())
    response_text = response.json()['choices'][0]['message']['content']
    # print(response_text)

    return response_text

def negate_image(image_file):
    try:
        # Convert Streamlit UploadedFile to bytes and open with PIL
        img_bytes = io.BytesIO(image_file.getvalue())
        with Image.open(img_bytes) as img:
            # Process image as before
            grayscale_img = img.convert('L')
            inverted_img = Image.eval(grayscale_img, lambda x: 255 - x)
            # Since we can't save to a file path directly, save to bytes
            img_byte_arr = io.BytesIO()
            inverted_img.save(img_byte_arr, format=img.format)
            # Return bytes-like object
            img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print("Error:", e)
        return None
def extract_text_from_image_tesseract(image_bytes):
    try:
        # Use bytes object directly with pytesseract
        img = Image.open(io.BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def pull_conv():
    global curr_username, uploaded_files, img_num

    if uploaded_files and len(uploaded_files) == img_num:
        with st.spinner('Processing...'):
            ocr_text_list = []
            formatted_texts = []

            for uploaded_file in uploaded_files:
                inverted_image_bytes = negate_image(uploaded_file)
                ocr_text = extract_text_from_image_tesseract(inverted_image_bytes)
                ocr_text_list.append(ocr_text)
                formatted_ocr_text = make_openai_call(
                    ocr_to_conv_model_name,
                    ocr_to_conv_system_message, 
                    ocr_text
                )
                formatted_texts.append(formatted_ocr_text)
                
            print(f"ocr_text_list\n{ocr_text_list}")
            print(f"formatted_texts\n{formatted_texts}")
            
            if curr_username and formatted_texts:
                update_or_insert_data(curr_username, formatted_texts)
                st.success("Data inserted/updated successfully!")
    else:
        st.error(f"Please upload exactly {img_num} images.")
def run_simulation():
    """
    Simulate a conversation where the output of one thread feeds into the next.

    Parameters:
    - curr_username: The current user initiating the conversation.
    - unsimmed_username: The other user with whom the conversation is simulated.
    - system_message_1: The system message for the first thread.
    - system_message_2: The system message for the second thread.
    - make_openai_call: Function to call the OpenAI API for simulating conversation.
    - total_messages: Total number of messages to be generated in the conversation.
    """
    global curr_username
    
    print(f"curr_username: {curr_username}\n")

    # Fetch a list of unsimmed usernames
    print(f"Fetching all usernames...")
    query = "SELECT DISTINCT username FROM info WHERE username != ?"
    data = (curr_username,)
    list_of_all_usernames = run_script_on_db( query, data, "check" )
    list_of_all_usernames = [username[0] for username in list_of_all_usernames]
    print(f"Fetched all usernames!\n{list_of_all_usernames}\n")

    # Fetch a list of simmed usernames
    print(f"Fetching simmed usernames...")
    query = "SELECT usernames_simmed FROM info WHERE username = ?"
    data = (curr_username,)
    list_of_simmed_usernames = run_script_on_db( query, data, "check" )
    list_of_simmed_usernames = [username[0] for username in list_of_simmed_usernames]
    if list_of_simmed_usernames[0] != '':
        list_of_simmed_usernames = eval(list_of_simmed_usernames[0] )
    else:
        list_of_simmed_usernames = []
    print(f"Fetched simmed usernames!\n{list_of_simmed_usernames}\n")

    # Calculate the list of unsimmed usernames
    list_of_unsimmed_usernames = [username for username in list_of_all_usernames if username not in list_of_simmed_usernames]
    print(f"List of unsimmed usernames...\n{list_of_unsimmed_usernames}\n")

    for unsimmed_username in list_of_unsimmed_usernames:
        print(f"Unsimmed username: {unsimmed_username}\n")
        
        # Initialize conversation history for both threads
        conversation_history = {
            curr_username: [],
            unsimmed_username: []
        }
        print(f"Conversation history\n{conversation_history}\n")
        
        # Fetch the current username's conversation style
        print(f"Fetching current username conversation style...")
        query = "SELECT conv_style FROM info WHERE username = ?"
        data = (str(curr_username),)
        curr_username_conv_style = run_script_on_db( query, data, "check" )
        curr_username_conv_style = [username[0] for username in curr_username_conv_style]
        print(f"Fetched current username conversation style\n{curr_username_conv_style}\n")
        conversation_history[curr_username].append({"role": "system", "content": username_system_prompt + curr_username_conv_style[0]})

        # Fetch the unsimmed username's conversation style
        print(f"Fetching unsimmed username conversation style...")
        query = "SELECT conv_style FROM info WHERE username = ?"
        data = (unsimmed_username,)
        unsimmed_username_conv_style = run_script_on_db( query, data, "check" )
        unsimmed_username_conv_style = [username[0] for username in unsimmed_username_conv_style]
        print(f"Fetched unsimmed username conversation style\n{unsimmed_username_conv_style}\n")
        conversation_history[unsimmed_username].append({"role": "system", "content": username_system_prompt + unsimmed_username_conv_style[0]})

        # Initialize conversation with an empty string or a starting message
        current_message = "hi"

        # Simulate conversations, generating up to 10 messages
        for i in range(max_messages):
            # First thread: Current user's perspective
            conversation_history[curr_username].append({"role": "user", "content": current_message})
            response_1 = make_together_call(
                chat_model_name,
                conversation_history[curr_username]
            )
            conversation_history[curr_username].append({"role": "assistant", "content": response_1})
            print(f"Thread 1 - Message {i+1}: {response_1}")

            # Use the response from the first thread as input for the second thread
            conversation_history[unsimmed_username].append({"role": "assistant", "content": current_message})
            response_2 = make_together_call(
                chat_model_name,
                conversation_history[unsimmed_username]
            )
            conversation_history[unsimmed_username].append({"role": "user", "content": response_2})
            print(f"Thread 2 - Message {i+1}: {response_2}")

            # Update the current message for the next iteration
            current_message = response_2

        # For illustration, print the simulated conversations
        print(f"Conversation from {curr_username}'s perspective:\n", conversation_history[curr_username])
        print(f"Conversation from {unsimmed_username}'s perspective:\n", conversation_history[unsimmed_username])

        list_of_simmed_usernames.append(unsimmed_username)

        # Update ran_sim status for the unsimmed_username to True
        print(f"Updating ran_sim status...")
        query = "UPDATE info SET ran_sim = ? WHERE username = ?"
        data = (True, curr_username)
        run_script_on_db( query, data, "insert" )
        print(f"Updated ran_sim status!")

        print(f"Updating usernames_simmed list...")
        query = "UPDATE info SET usernames_simmed = ? WHERE username = ?"
        data = (str(list_of_simmed_usernames), curr_username)
        run_script_on_db( query, data, "insert" )
        print(f"Updated usernames_simmed list!\n")

        st.success("Chats Simulated")

def ui():
    global curr_username, uploaded_files, img_num

    # Set the sidebar title
    st.sidebar.title("datesim")

    # User
    st.sidebar.title("User")
    # Create a text input in the sidebar for the username
    curr_username = st.sidebar.text_input("Add username")
    
    # Training
    st.sidebar.title("Training")
    # Create a number input in the sidebar for img_num
    img_num = st.sidebar.number_input("Number of Images", min_value=1, max_value=5, value=1, step=1)
    # Add a file uploader in the sidebar for uploading files
    uploaded_files = st.sidebar.file_uploader("Choose training images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    # Chat Simulation
    st.sidebar.title("Chat Simulation")
    # Create a number input in the sidebar for max_messages
    max_messages = st.sidebar.number_input("Max Messages", min_value=1, max_value=100, value=1, step=1)
    # Create a number input in the sidebar for max_tokens
    max_tokens = st.sidebar.slider("Max Tokens", min_value=64, max_value=512, value=256, step=64)
    # Create a slider in the sidebar for temperature
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

def main():
    # Training
    st.title("Training")
    if st.button('Process Images'):
        pull_conv()
    
    # Chat Simulation
    st.title("Chat Simulation")
    if st.button('Run Chat Simulation'):
        run_simulation()

if __name__ == "__main__":
    initialize_db()
    ui()
    main()