# RAG Chatbot with ChromaDB and OpenAI

This project loads CSV data into a Chroma vector database, retrieves the most relevant rows for a user query, and uses OpenAI to generate a final answer in a Streamlit chat UI.

## Features
- Reads rows from a CSV file
- Stores the rows as text in ChromaDB
- Generates embeddings using OpenAI embeddings
- Uses recent chat history together with retrieved context
- Runs as a Streamlit web app

## Project structure
- `chatbot-RAG-expans.py` - main app
- `requirements.txt` - Python dependencies
- `data/` - folder containing the CSV file used for the knowledge base
- `chroma_db/` - local ChromaDB storage generated at runtime

## Requirements
- Python 3.10+
- OpenAI API key

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the same folder as the script and add your API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. Make sure your CSV file exists in the `data` directory. The app expects a file like:

```text
data/rag_sample_qas_from_kis.csv
```

## Run the app

```bash
streamlit run chatbot-RAG-expans.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Notes
- Do not commit your `.env` file.
- ChromaDB data is stored in the local `chroma_db/` folder.
- This app uses the OpenAI chat model and embedding model configured in the code.

## Dependencies included in requirements.txt
- `openai`
- `chromadb`
- `python-dotenv`
- `streamlit`
- `requests`

## Suggested `.gitignore`

```gitignore
.env
.venv/
__pycache__/
chroma_db/
*.pyc
```
