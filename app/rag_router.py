# app/rag_router.py
import os, json, re
from dotenv import load_dotenv

load_dotenv()

from typing import List, Dict, Tuple, Any
import numpy as np

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema import Document, HumanMessage, AIMessage

from rag_mistral import build_retriever, get_mistral
from rag_groq import get_groq
from langchain_huggingface import HuggingFaceEmbeddings


# ------------------------- Prompts -------------------------

SYSTEM = (
    "You are a careful medical QA assistant. Use ONLY the retrieved context. "
    "Answer concisely for clinicians. If the context doesn't contain the answer, say 'I don't know'. "
    "Do not add made-up citations or page numbers."
)

HUMAN = (
    "Context:\n{context}\n\n"
    "Question:\n{input}\n\n"
    "Provide a concise, factual answer based only on the context."
)

# Answering prompt (includes chat history)
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", HUMAN),
])

# How each document is injected for the stuff chain
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "Page {page} — {source}\n\n{page_content}\n"
)

# Query rewrite / condensation for history-aware retrieval
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's question into a standalone, context-aware medical search query. "
     "Use correct terminology, synonyms, and common abbreviations. Be concise."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# ------------------------- Utilities -------------------------

_EMB = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={'normalize_embeddings': True, "batch_size": 64},
)

STOPWORDS = set("""
a an the and or of for to with on in at from by into over under as is are was were be been being this that those these it its their his her your my our
""".split())

def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    if isinstance(x, dict):
        for key in ("page_content", "text", "answer", "output_text", "result", "content"):
            v = x.get(key)
            if isinstance(v, str):
                return v
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

def _normalize_answer(out: Any) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for key in ("text", "answer", "output_text", "result"):
            if key in out:
                return _to_text(out[key])
        return _to_text(out)
    return _to_text(out)

def _tokenize(txt: str) -> List[str]:
    s = _to_text(txt)
    return [t.lower() for t in re.findall(r"[A-Za-z]+", s)]

def _keyword_overlap(answer: str, docs: List[Document]) -> float:
    ans_tokens = [t for t in _tokenize(answer) if t not in STOPWORDS]
    if not ans_tokens:
        return 0.0
    ctx_tokens = set()
    for d in docs:
        txt = _to_text(getattr(d, "page_content", ""))
        ctx_tokens.update([t for t in _tokenize(txt) if t not in STOPWORDS])
    overlap = sum(1 for t in ans_tokens if t in ctx_tokens)
    return overlap / max(1, len(ans_tokens))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embedding_faithfulness(answer: str, docs: List[Document]) -> float:
    a_vec = np.array(_EMB.embed_query(_to_text(answer)), dtype=np.float32)
    best = 0.0
    for d in docs:
        txt = _to_text(getattr(d, "page_content", ""))[:2000]
        v = np.array(_EMB.embed_query(txt), dtype=np.float32)
        best = max(best, _cosine(a_vec, v))
    return best

def score_answer(answer: str, docs: List[Document]) -> float:
    ans = _to_text(answer).strip()
    if not ans:
        return 0.0
    penalty = 0.0
    if "i don't know" in ans.lower():
        penalty += 0.35
    if len(ans) < 25:
        penalty += 0.15
    faith = _embedding_faithfulness(ans, docs)
    overlap = _keyword_overlap(ans, docs)
    score = (0.65 * faith) + (0.35 * overlap) - penalty
    return max(0.0, score)

def _ensure_chat_messages(history) -> List:
    """Convert simple history into LangChain messages; keep last ~6 turns for brevity."""
    msgs = []
    if not history:
        return msgs
    for turn in history[-6:]:
        # support dict {"role": "...", "content": "..."} or (user, ai) tuples
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            role = turn["role"]
            content = _to_text(turn["content"])
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))
        elif isinstance(turn, tuple) and len(turn) == 2:
            user, ai = turn
            msgs.append(HumanMessage(content=_to_text(user)))
            msgs.append(AIMessage(content=_to_text(ai)))
    return msgs

def _build_doc_chain(llm):
    return create_stuff_documents_chain(
        llm=llm,
        prompt=ANSWER_PROMPT,
        document_prompt=DOCUMENT_PROMPT,
        document_variable_name="context",
    )


# ------------------------- Router (Groq + Mistral) -------------------------

def answer_with_both_llms(question: str, chat_history=None) -> Dict:
    """
    Return the best-scored answer among Groq and Mistral.
    Output: {"model": "...", "answer": "...", "score": float, "loser_score": float, "context": [Document,...]}
    """
    chat_msgs = _ensure_chat_messages(chat_history)

    # --- History-aware retrieval (use Groq for query rewrite to avoid Mistral 429 issues) ---
    base_retriever = build_retriever()
    try:
        llm_rewrite = get_groq()  # fast & usually available
    except Exception:
        llm_rewrite = get_mistral()  # fallback if Groq key missing

    hist_retriever = create_history_aware_retriever(
        llm=llm_rewrite,
        retriever=base_retriever,
        prompt=CONDENSE_PROMPT,
    )

    docs = hist_retriever.invoke({"input": question, "chat_history": chat_msgs})
    if not isinstance(docs, list):
        docs = docs.get("context") or docs.get("documents") or []

    # --- Build chains for both cloud models ---
    groq_ans, s_groq = None, -1.0
    mis_ans,  s_mis  = None, -1.0

    # Groq
    try:
        groq_chain = _build_doc_chain(get_groq())
        groq_out = groq_chain.invoke({"input": question, "context": docs, "chat_history": chat_msgs})
        groq_ans = _normalize_answer(groq_out)
        s_groq = score_answer(groq_ans, docs)
    except Exception:
        pass  # ignore, other model may succeed

    # Mistral (can 429 during peak; catch & continue)
    try:
        mistral_chain = _build_doc_chain(get_mistral())
        mis_out = mistral_chain.invoke({"input": question, "context": docs, "chat_history": chat_msgs})
        mis_ans = _normalize_answer(mis_out)
        s_mis = score_answer(mis_ans, docs)
    except Exception:
        pass

    # --- Pick best ---
    candidates: List[Tuple[str, str, float]] = []
    if groq_ans is not None:
        candidates.append(("groq", groq_ans, s_groq))
    if mis_ans is not None:
        candidates.append(("mistral", mis_ans, s_mis))

    if not candidates:
        return {
            "model": "none",
            "answer": "I don't know based on the provided context.",
            "score": 0.0,
            "loser_score": 0.0,
            "context": docs,
        }

    # highest score wins
    model, answer, score = max(candidates, key=lambda x: x[2])
    others = [s for (m, a, s) in candidates if m != model]
    loser_score = max(others) if others else 0.0

    return {"model": model, "answer": answer, "score": score, "loser_score": loser_score, "context": docs}
