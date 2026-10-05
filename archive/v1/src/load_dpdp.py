"""Build ChromaDB knowledge base from DPDP Act and Rules PDFs."""

import os
import chromadb
from chromadb.utils import embedding_functions
import pypdf

DB_PATH = "../data/dpdp_db"
PDF_FILES = [
    "../data/dpdp_act_rules.pdf",
    "../data/dpdp_act_summary.pdf",
]


def extract_pages(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append({"text": text, "page": i + 1, "source": pdf_path})
    return chunks


def build_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection("dpdp_act")
    except Exception:
        pass

    collection = client.create_collection("dpdp_act", embedding_function=ef)

    all_chunks = []
    for pdf in PDF_FILES:
        if not os.path.isfile(pdf):
            raise FileNotFoundError(f"PDF not found: {pdf}")
        chunks = extract_pages(pdf)
        all_chunks.extend(chunks)
        print(f"Extracted {len(chunks)} pages from {os.path.basename(pdf)}")

    collection.add(
        documents=[c["text"] for c in all_chunks],
        ids=[f"{os.path.basename(c['source'])}_p{c['page']}" for c in all_chunks],
        metadatas=[{"page": c["page"], "source": c["source"]} for c in all_chunks],
    )
    print(f"Loaded {len(all_chunks)} pages into ChromaDB at {DB_PATH}")


if __name__ == "__main__":
    build_db()
