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
# Vars
max_tokens = 512
temperature = 0.0
img_num = 1
max_messages = 5
curr_username = None
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

def db_action(db_action_text, db_complete_text, sql_script, data, action_type="insert"):
    print(db_action_text)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if action_type == "insert":
        if data:
            c.execute(sql_script, data)
        else:
            c.execute(sql_script)
    elif action_type == "check":
        c.execute(sql_script, data)
        result = c.fetchone()
        return result
    conn.commit()
    conn.close()

    print(db_complete_text)
    st.success(db_complete_text)
def update_or_insert_data(username, formatted_texts):
    # Check if the username exists and usernames_simmed is empty
    existing_user = db_action(
        'Checking existing username...',
        "Username checked!",
        "SELECT * FROM info WHERE username = ?",
        (username,),
        action_type="check"
    )

    if existing_user:
        if existing_user[3] == "[]":  # Assuming the usernames_simmed column is the 4th column
            # Update existing record
            db_action(
                'Updating existing record...',
                "Updated record!",
                "UPDATE info SET conv_style = ?, ran_sim = ?, usernames_simmed = ? WHERE username = ?",
                (str(formatted_texts), False, "[]", username),
                action_type="insert"
            )
        else:
            # Insert new record since usernames_simmed is not empty
            db_action(
                'Inserting new record...',
                "Inserted record!",
                "INSERT INTO info (username, conv_style, ran_sim, usernames_simmed) VALUES (?, ?, ?, ?)",
                (username, str(formatted_texts), False, "[]"),
                action_type="insert"
            )
    else:
        # Insert new record since username does not exist
        db_action(
            'Inserting new record...',
            "Inserted record!",
            "INSERT INTO info (username, conv_style, ran_sim, usernames_simmed) VALUES (?, ?, ?, ?)",
            (username, str(formatted_texts), False, "[]"),
            action_type="insert"
        )
def initialize_db():
    db_action(
        'Creating DB...',
        "DB created!",
        '''
        CREATE TABLE IF NOT EXISTS info 
        (username TEXT, conv_style TEXT, ran_sim BOOLEAN, usernames_simmed TEXT)
        ''',
        None
    )
    
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
    print(response.json())

    return response.json()['choices'][0]['message']['content']

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
    list_of_unsimmed_usernames = db_action(
        "Fetching unsimmed usernames...",
        "Fetched unsimmed usernames!",
        "SELECT username FROM info WHERE ran_sim = 0 AND username != ?",
        (curr_username,),
        "check"
    )
    # list_of_unsimmed_usernames = [row[0] for row in list_of_unsimmed_usernames]
    print(f"list_of_unsimmed_usernames\n{list_of_unsimmed_usernames}\n")

    # Fetch a list of simmed usernames
    list_of_simmed_usernames = db_action(
        "Fetching simmed usernames...",
        "Fetched simmed usernames!",
        "SELECT usernames_simmed FROM info WHERE username != ?",
        (curr_username,),
        "check"
    )
    list_of_simmed_usernames = [row[0] for row in list_of_simmed_usernames]
    print(f"list_of_simmed_usernames\n{list_of_simmed_usernames}\n")

    for unsimmed_username in list_of_unsimmed_usernames:
        print(f"unsimmed_username\n{unsimmed_username}\n")
        
        # Initialize conversation history for both threads
        conversation_history = {
            curr_username: [],
            unsimmed_username: []
        }
        print(f"conversation_history\n{conversation_history}\n")
        
        # Fetch the current username's conversation style
        curr_username_conv_style = db_action(
            "Fetching current username conversation style...",
            "Fetched current username conversation style!",
            "SELECT conv_style FROM info WHERE username = ?",
            (str(curr_username),),
            "check"
        )
        print(f"curr_username_conv_style\n{curr_username_conv_style}\n")
        conversation_history[curr_username].append({"role": "system", "content": username_system_prompt + curr_username_conv_style[0]})

        # Fetch the unsimmed username's conversation style
        unsimmed_username_conv_style = db_action(
            "Fetching unsimmed username conversation style...",
            "Fetched current username conversation style!",
            "SELECT conv_style FROM info WHERE username = ?",
            (unsimmed_username,),
            "check"
        )
        print(f"unsimmed_username_conv_style\n{unsimmed_username_conv_style}\n")
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

        # Update ran_sim status for the unsimmed_username to True
        db_action(
            "Updating ran_sim status...",
            "Updated ran_sim status!",
            "UPDATE info SET ran_sim = ? WHERE username = ?",
            (True, curr_username),
            action_type="insert"
        )
        db_action(
            "Updating usernames_simmed list...",
            "Updated usernames_simmed list!\n",
            "UPDATE info SET usernames_simmed = ? WHERE username = ?",
            (unsimmed_username, curr_username),
            action_type="insert"
        )

def main():
    global curr_username

    initialize_db()

    st.title("RIZZARENA")

    curr_username = st.text_input("Add username")

    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    if st.button('Process Images'):
        if uploaded_files and len(uploaded_files) == img_num:
            with st.spinner('Processing...'):
                extracted_texts = []
                formatted_texts = []

                for uploaded_file in uploaded_files:
                    inverted_image_bytes = negate_image(uploaded_file)
                    ocr_text = extract_text_from_image_tesseract(inverted_image_bytes)
                    extracted_texts.append(ocr_text)
                    formatted_ocr_text = make_openai_call(
                        ocr_to_conv_model_name,
                        ocr_to_conv_system_message, 
                        ocr_text
                    )
                    formatted_texts.append(formatted_ocr_text)
                    
                print(f"extracted_texts\n{extracted_texts}")
                print(f"formatted_texts\n{formatted_texts}")
                
                if curr_username and formatted_texts:
                    update_or_insert_data(curr_username, formatted_texts)
                    st.success("Data inserted/updated successfully!")
        else:
            st.error(f"Please upload exactly {img_num} images.")

    if st.button('Run Simulation'):
        run_simulation()

if __name__ == "__main__":
    main()