from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError

from rag import answer_question, get_vector_store

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF")
st.caption("RAG pipeline: PDF → chunks → embeddings → FAISS retrieval → OpenAI answer")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Copy .env.example to .env and add your API key.")
    st.stop()

pdf_path = ROOT_DIR / "data" / "document.pdf"
if not pdf_path.exists():
    st.warning("Add your PDF as `data/document.pdf`, then restart the app.")
    st.stop()

with st.sidebar:
    st.header("Document")
    st.write("Using: `data/document.pdf`")
    st.write("The app builds a local FAISS index when first needed.")
    if st.button("Build / rebuild index"):
        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                get_vector_store.clear()
                get_vector_store()
            st.success("Index built successfully.")
        except Exception as exc:
            st.error(f"Could not build index: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved PDF passages"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Passage {i} — page {source['page']}**")
                    st.write(source["content"])

question = st.chat_input("Ask a question about your PDF...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                answer, sources = answer_question(question)
            st.markdown(answer)
            with st.expander("Retrieved PDF passages"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Passage {i} — page {source['page']}**")
                    st.write(source["content"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except FileNotFoundError as exc:
            st.error(str(exc))
        except OpenAIError as exc:
            st.error(f"OpenAI API error: {exc}")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
