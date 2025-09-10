# app/streamlit_app.py
import streamlit as st
from rag_router import answer_with_both_llms

st.set_page_config(page_title="ChatGME",  layout="wide")

st.title("ChatGME")
st.caption("Retrieval-augmented; routes between GROQ and MISTRAL based on grounded scoring.")

if "chat" not in st.session_state:
    # store as list of {"role": "user"/"assistant", "content": "...", "model": "..."}
    st.session_state.chat = []

# Show history
for turn in st.session_state.chat:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['content']}")
    else:
        model = turn.get("model", "?").upper()
        score = turn.get("score", 0.0)
        st.markdown(f"**Assistant ({model}, score={score:.3f}):** {turn['content']}")
        with st.expander("Sources"):
            for i, d in enumerate(turn.get("context", []), 1):
                meta = d.metadata or {}
                st.write(f"[{i}] {meta.get('source','?')} (p{meta.get('page','?')})")
    st.markdown("---")

q = st.chat_input("Ask a follow-up…")
if q:
    # append user msg
    st.session_state.chat.append({"role": "user", "content": q})

    # build light history in router-friendly shape
    hist = []
    for t in st.session_state.chat[:-1]:
        hist.append({"role": t["role"], "content": t["content"]})

    with st.spinner("Thinking..."):
        out = answer_with_both_llms(q, chat_history=hist)

    # append assistant msg
    st.session_state.chat.append({
        "role": "assistant",
        "content": out["answer"],
        "model": out["model"],
        "score": out["score"],
        "context": out.get("context", []),
    })

    st.rerun()
