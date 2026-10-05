import hashlib
import json
import os
import re

from prompts import TOP_K


class KnowledgeBase:
    """DPDP Act / Rules passages in a persistent ChromaDB collection (all-MiniLM-L6-v2, cosine)."""

    def __init__(self, pdfs, db_dir, chunk_chars=900, overlap=150):
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        sig = hashlib.md5(json.dumps(sorted(os.path.basename(p) + str(os.path.getsize(p)) for p in pdfs)
                                     + [chunk_chars, overlap]).encode()).hexdigest()[:8]
        client = chromadb.PersistentClient(path=db_dir)
        self.col = client.get_or_create_collection(f"dpdp_{sig}", embedding_function=ef,
                                                   metadata={"hnsw:space": "cosine"})
        if self.col.count() == 0:
            self._index(pdfs, chunk_chars, overlap)
        self.sources = sorted({m["source"] for m in self.col.get(include=["metadatas"])["metadatas"]})
        print(f"[kb] {self.col.count()} passages from {self.sources}", flush=True)

    def _index(self, pdfs, chunk_chars, overlap):
        from pypdf import PdfReader

        ids, docs, metas = [], [], []
        for path in pdfs:
            name = os.path.basename(path)
            for pno, page in enumerate(PdfReader(path).pages, 1):
                text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
                start, k = 0, 0
                while start < len(text):
                    chunk = text[start:start + chunk_chars].strip()
                    if len(chunk) > 40:
                        ids.append(f"{name}_p{pno}_{k}")
                        docs.append(chunk)
                        metas.append({"source": name, "page": pno})
                        k += 1
                    start += chunk_chars - overlap
        for i in range(0, len(ids), 200):
            self.col.add(ids=ids[i:i + 200], documents=docs[i:i + 200], metadatas=metas[i:i + 200])

    def retrieve(self, text, k=TOP_K):
        res = self.col.query(query_texts=[text], n_results=k)
        return "\n\n---\n\n".join(res["documents"][0])
