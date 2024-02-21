import openai
import requests
import base64
import time
import os

# Set your OpenAI API key here
openai_api_key = os.getenv('OPENAI_API_KEY')
system_prompt = """
You are a smart OCR system. You will be provided screenshots of the user's WhatsApp conversations. 
Extract and structure the conversations in the format below:
user: <content>
person: <content>
"""
detail = "high"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4v_api(base64_enc_img1, base64_enc_img2, base64_enc_img3):
    print('Calling GPT4V API...')
    
    start_time = time.time()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_enc_img1}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_enc_img2}",
                            "detail": detail
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_enc_img3}",
                            "detail": detail
                        }
                    },
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    elapsed_time = time.time() - start_time
    print(f'Received GPT4V response in {elapsed_time:.2f} seconds.')
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f'Error: {response.status_code}')
        print(response.json())
        return "Error in calling GPT API."

def main():
    base64_encoded_image1 = encode_image_to_base64(r'data\conv1.jpeg')
    base64_encoded_image2 = encode_image_to_base64(r'data\conv2.jpeg')
    base64_encoded_image3 = encode_image_to_base64(r'data\conv3.jpeg')
    gpt_response = call_gpt4v_api(
        base64_encoded_image1,
        base64_encoded_image2,
        base64_encoded_image3
    )
    print(f"gpt_response\n{gpt_response}")

if __name__ == "__main__":
    main()