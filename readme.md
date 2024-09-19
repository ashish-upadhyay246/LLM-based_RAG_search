# LLM-based RAG Search

This project consists of a Flask backend and a Streamlit frontend that work together to scrape web data, generate embeddings, and produce responses using Google's Gemini. The backend handles data processing and the frontend provides a user interface for input and output. The entered query is passed on to backend where it is converted into a google search link. Option to select the required number of sites to be scraped is provided on the frontend to control the level of information gathered (default it 10 sites). The sites are then scraped, the scraped text is converted to chunks, which is further converted into embeddings and then indexed. The query is converted to an embedding and then the chunks with highest similarity are fed to the Gemini API along with the query to produce the response which is then displayed on the frontend.

## Project Structure

- `flask_app.py`: The Flask backend that processes requests and interacts with the `modules.py` file.
- `streamlit_app.py`: The Streamlit frontend that provides a user interface for input and displays results.
- `modules.py`: Contains functions for web scraping, text processing, embedding, and response generation.

## Prerequisites

- Python 3.9 or above

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ashish-upadhyay246/LLM-based_RAG_search.git
    cd genai-assignment
    ```
    or download it.

2. **Create a virtual environment:**

    Using `venv`

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

    Using `conda`

    ```bash
    conda create --name project_env python=3.8
    conda activate project_env
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory and add your API key:

    ```env
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Running the Application

1. **Start the Flask backend:**

    ```bash
    python flask_app.py
    ```

    The Flask server will run at `http://localhost:5000`.

2. **Start the Streamlit frontend:**

    ```bash
    streamlit run streamlit_app.py
    ```

    The Streamlit app will open in your default web browser.
