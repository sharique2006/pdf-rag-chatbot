# PDF RAG Chatbot (OpenAI + LangChain + FAISS)

A runnable beginner-friendly Retrieval-Augmented Generation (RAG) chatbot. It reads one PDF, splits its text into chunks, creates OpenAI embeddings, stores them in a local FAISS vector index, retrieves relevant passages for each question, and asks an OpenAI chat model to answer using those passages.

## Project structure

```text
rag_pdf_chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── rag.py
├── data/
│   └── document.pdf       # Add your PDF here
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10–3.12 recommended
- An OpenAI API key with API billing/credits enabled
- One text-based PDF document

## Run on macOS / Linux

Open a terminal in this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace `your_openai_api_key_here` with your real key. Do not share or commit the key.

Put your PDF at `data/document.pdf`, then start the chatbot:

```bash
streamlit run app/main.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Run on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`, add your key, and place your PDF at `data\document.pdf`. Then run:

```powershell
streamlit run app/main.py
```

If PowerShell blocks activation, see the Python venv documentation or use Command Prompt with `.venv\Scripts\activate.bat`.

## How the RAG pipeline works

1. **Load:** `PyPDFLoader` extracts text and page metadata from `data/document.pdf`.
2. **Chunk:** `RecursiveCharacterTextSplitter` splits long pages into overlapping chunks.
3. **Embed:** `text-embedding-3-small` converts each chunk into a vector using the OpenAI API.
4. **Index:** FAISS stores the vectors locally in `vector_store/`.
5. **Retrieve:** each question searches for the four most similar chunks.
6. **Generate:** `gpt-4o-mini` answers using the retrieved context and is instructed not to invent information.
7. **Cite:** the prompt asks the model to include page numbers; the UI also shows the retrieved passages and pages.

## Notes and troubleshooting

- The first index build makes embedding API calls and may incur OpenAI API charges. Each question also uses the chat API.
- ChatGPT subscriptions and OpenAI API billing are separate.
- This starter app rebuilds the index when the app process starts; keep the PDF unchanged while using it. If you replace the PDF, restart the app and rebuild the index.
- Scanned/image-only PDFs may need OCR before this app can extract their text.
- If you see an authentication error, verify `OPENAI_API_KEY` in `.env` and confirm API access/billing.
- If package installation fails, use Python 3.11 in a fresh virtual environment.
- Do not upload confidential company documents to an external API unless your organization's policy permits it.
