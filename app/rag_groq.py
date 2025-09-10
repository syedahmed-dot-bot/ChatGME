# app/rag_mistral.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever


from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parent
# If your folder is "vectorstore", change "vector" -> "vectorstore"
VEC_DIR = (ROOT.parent / "vector" / "db_faiss_bge").resolve()
BM25_JSONL = (ROOT.parent / "bm25_store" / "corpus.jsonl").resolve()

def load_faiss():
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(str(VEC_DIR), emb, allow_dangerous_deserialization=True)

def load_bm25():
    if not BM25_JSONL.exists():
        return None
    texts, metas = [], []
    with open(BM25_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append(rec["metadata"])
    bm = BM25Retriever.from_texts(texts=texts, metadatas=metas)
    bm.k = 6
    return bm


def build_retriever():
    # FAISS (MMR)
    vec = load_faiss().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.2},
    )

    # BM25 (if available)
    bm = load_bm25()
    if bm:
        # favor BM25 slightly for exact terms (AF, ANA, etc.)
        hybrid = EnsembleRetriever(retrievers=[bm, vec], weights=[0.6, 0.4])
    else:
        hybrid = vec

    # Optional cross-encoder reranker (keep top 5)
    base = hybrid
    try:
        rer = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)
        base = ContextualCompressionRetriever(base_compressor=rer, base_retriever=hybrid)
    except Exception:
        pass  # fall back to `hybrid` if cross-encoder not installed

    # --- Multi-query wrapper (no returns above this line!) ---
    mq = MultiQueryRetriever.from_llm(
        retriever=base,
        llm=get_groq(),  # reuse your Mistral client
        prompt=ChatPromptTemplate.from_template(
            "Generate 4 diverse search queries (synonyms, abbreviations, related terms) "
            "for the user's medical question.\nUser question: {question}"
        ),
        include_original=True,
    )

    return mq


def get_groq():
    api_key = os.getenv("GROQ_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_AI_API_KEY is not set in .env")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.5,
        max_tokens=256,
        timeout=60,
    )

def build_rag_chain(llm=None):
    if llm is None:
        llm = get_groq()

    # Keep the answer clean; we’ll print SOURCES programmatically
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful medical QA assistant. Use ONLY the provided context. "
         "Answer concisely for clinicians. If the context doesn’t contain the answer, say 'I don’t know'. "
         "Do NOT guess page numbers or add citations inline."),
        ("human",
         "Context:\n{context}\n\nQuestion:\n{input}\n\n"
         "Answer succinctly based only on the context.")
    ])

    # Include page/file in the stuffed docs for human-readable context, but don’t ask model to cite
    document_prompt = PromptTemplate.from_template("Page {page} — {source}\n\n{page_content}\n")

    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
        document_variable_name="context",
    )

    retriever = build_retriever()
    return create_retrieval_chain(retriever, doc_chain)
