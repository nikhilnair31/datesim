import os
import subprocess
import pytesseract
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Set your OpenAI API key here
img_paths = [
    r'data\conv1.jpeg',
    r'data\conv2.jpeg',
    r'data\conv3.jpeg'
]

# Path to the Tesseract executable, update this if Tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def negate_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert image to grayscale
            grayscale_img = img.convert('L')
            # Invert the grayscale image
            inverted_img = Image.eval(grayscale_img, lambda x: 255 - x)
            # Create output path by appending "_inv" before the file extension
            output_img_path = f"{os.path.splitext(image_path)[0]}_inv{os.path.splitext(image_path)[1]}"
            # Save the inverted image
            inverted_img.save(output_img_path)
        return output_img_path
    except Exception as e:
        print("Error:", e)
        return None

def extract_text_from_image_tesseract(image_path):
    try:
        with Image.open(image_path) as img:
            extracted_text = pytesseract.image_to_string(img)
            return extracted_text
    except Exception as e:
        print("Error:", e)
        return None

def main():
    for img_path in img_paths:
        # Negate the image
        print(f"img_path: {img_path}")
        output_img_path = negate_image(img_path)
        print(f"output_img_path: {output_img_path}")
        base64_encoded_image = extract_text_from_image_tesseract(img_path)
        print(f"base64_encoded_image\n{base64_encoded_image}\n")

if __name__ == "__main__":
    main()