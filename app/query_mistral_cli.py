# app/query_mistral_cli.py
from rag_mistral import build_rag_chain

if __name__ == "__main__":
    rag = build_rag_chain()

    while True:
        try:
            q = input("\nYour question (or 'exit'): ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            out = rag.invoke({"input": q, "question": q})
            print("\n=== ANSWER ===\n", out.get("answer",""))

            ctx = out.get("context") or out.get("source_documents") or out.get("input_documents") or []
            print("\n=== SOURCES ===")
            if not ctx:
                print("(no documents returned)")
            else:
                for i, d in enumerate(ctx, 1):
                    meta = getattr(d, "metadata", {}) or {}
                    print(f"[{i}] {meta.get('source','?')} (p{meta.get('page','?')})")
        except KeyboardInterrupt:
            break
