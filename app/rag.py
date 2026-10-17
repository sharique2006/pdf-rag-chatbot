from pathlib import Path
from functools import lru_cache
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "data" / "document.pdf"
INDEX_DIR = ROOT_DIR / "vector_store"


def build_vector_store():
    """Load the configured PDF, split it into chunks, embed chunks, and build a FAISS index."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Put one PDF document at data/document.pdf and restart the app."
        )

    pages = PyPDFLoader(str(PDF_PATH)).load()
    if not pages or not any(page.page_content.strip() for page in pages):
        raise ValueError("The PDF has no extractable text. If it is scanned, run OCR first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


@lru_cache(maxsize=1)
def get_vector_store():
    """Build the index once per app process."""
    return build_vector_store()


def answer_question(question: str) -> tuple[str, List[dict]]:
    vector_store = get_vector_store()
    matches = vector_store.similarity_search_with_score(question, k=4)

    if not matches:
        return "I couldn't find relevant information in the PDF.", []

    context_parts = []
    sources = []
    for doc, score in matches:
        page_num = doc.metadata.get("page", None)
        page_label = str(page_num + 1) if isinstance(page_num, int) else "unknown"
        context_parts.append(f"[Page {page_label}]\n{doc.page_content}")
        sources.append({
            "page": page_label,
            "content": doc.page_content[:500],
            "distance": round(float(score), 4),
        })

    context = "\n\n---\n\n".join(context_parts)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful PDF question-answering assistant. Answer using only the "
         "provided PDF context. If the context does not contain the answer, say you "
         "could not find the answer in the document. Do not invent facts. Cite page "
         "numbers in your answer when possible."),
        ("human", "PDF context:\n{context}\n\nQuestion: {question}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources
