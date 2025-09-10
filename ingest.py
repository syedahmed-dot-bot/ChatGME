# app/ingest_v2.py  (you can name it ingest.py and overwrite)
import os, re, json, uuid, sys, traceback
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT.parent / "data").resolve()
VEC_DIR  = (ROOT.parent / "vector" / "db_faiss_bge").resolve()   # <- change if your folder is "vectorstore"
VEC_DIR.mkdir(parents=True, exist_ok=True)

def dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text

def load_pdf_pages(pdf_path: Path):
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()                  # returns a list (may be empty)
        return pages
    except Exception as e:
        print(f" Failed to load {pdf_path.name}: {e}")
        traceback.print_exc()
        return []

def main():
    print(f" DATA_DIR: {DATA_DIR}")
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(" No PDF files found in data/. Add PDFs and run again.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    total_pages = 0
    all_chunks = []

    for pdf in pdfs:
        print(f"\n Loading: {pdf.name}")
        pages = load_pdf_pages(pdf)
        n_pages = len(pages)
        print(f"   → pages loaded: {n_pages}")
        total_pages += n_pages

        if n_pages == 0:
            print(f"  Skipping {pdf.name}: 0 pages.")
            continue

        # clean text + normalize metadata (force 1-based page numbers)
        for i, d in enumerate(pages):
            d.page_content = dehyphenate(d.page_content or "")
            md = d.metadata or {}
            md["source"] = pdf.name
            # fallback to enumerate index if loader didn’t set 'page'
            raw_page = md.get("page", md.get("page_number", i))
            try:
                md["page"] = int(raw_page) + 1
            except Exception:
                md["page"] = i + 1
            d.metadata = md

        # split in one call (no indexing into empty lists)
        chunks = splitter.split_documents(pages)
        print(f"   → chunks created: {len(chunks)}")

        # add stable IDs & ensure metadata keys exist
        for j, d in enumerate(chunks):
            md = d.metadata or {}
            md.setdefault("source", pdf.name)
            md.setdefault("page", 1)
            md["chunk_id"] = f'{md["source"]}::p{md["page"]}::c{j}-{uuid.uuid4().hex[:6]}'
            d.metadata = md

        all_chunks.extend(chunks)

    if not all_chunks:
        print(" No chunks produced. Likely all PDFs failed to load or were empty.")
        sys.exit(1)

    print(f"\n Total PDFs: {len(pdfs)} | Total pages: {total_pages} | Total chunks: {len(all_chunks)}")

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
    )

    print(" Embedding & building FAISS index…")
    db = FAISS.from_documents(all_chunks, emb)
    db.save_local(str(VEC_DIR))
    print(f" Saved FAISS to: {VEC_DIR}")

    # sanity sample
    print("\n Sample chunks:")
    for d in all_chunks[:3]:
        src = d.metadata.get("source", "?")
        pg  = d.metadata.get("page", "?")
        print(f" • {src} p{pg}: {d.page_content[:160].replace('\\n',' ')}")

if __name__ == "__main__":
    main()
