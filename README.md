# datesim

This repository contains code for a tool named datesim, which processes images containing text (such as screenshots of conversations) and simulates conversations based on the extracted text using AI models.

## Features

- **Image Processing**: Upload images containing text (e.g., screenshots of WhatsApp conversations).
- **Text Extraction**: Extract text from the uploaded images using Tesseract OCR (Optical Character Recognition).
- **Conversation Simulation**: Simulate conversations based on the extracted text using AI models from OpenAI.
- **Database Integration**: Store formatted conversation data in a SQLite database.

## Prerequisites

- Python 3.11
- Tesseract OCR installed
- OpenAI API key
- Together API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your-username/datesim.git
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    - Create a `.env` file in the project directory.
    - Add your OpenAI API key and Together API key in the `.env` file:

        ```
        OPENAI_API_KEY=your-openai-api-key
        TOGETHER_API_KEY=your-together-api-key
        ```

4. Update the path to the Tesseract executable (`pytesseract.pytesseract.tesseract_cmd`) if necessary.

5. Run the application:

    ```
    streamlit run app.py
    ```

## Usage

1. Launch the application.
2. Enter your username.
3. Upload images containing text.
4. Click "Process Images" to extract text and simulate conversations.
5. Then, click "Run Simulation" to simulate conversations.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository