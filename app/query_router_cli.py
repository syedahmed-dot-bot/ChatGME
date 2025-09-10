# app/query_router_cli.py
from rag_router import answer_with_both_llms

def _s(x) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return repr(x)

if __name__ == "__main__":
    while True:
        try:
            q = input("\nYour question (or 'exit'): ").strip()
            if q.lower() in {"exit", "quit"}:
                break

            out = answer_with_both_llms(q)

            print(f"\n=== SELECTED MODEL ===  {out['model']}  (score={out['score']:.3f}, other={out['loser_score']:.3f})")
            print("\n=== ANSWER ===\n", _s(out.get("answer", "")))

            ctx = out.get("context", [])
            print("\n=== SOURCES ===")
            if not ctx:
                print("(no documents returned)")
            else:
                for i, d in enumerate(ctx, 1):
                    meta    = getattr(d, "metadata", {}) or {}
                    source  = _s(meta.get("source", "?"))
                    page    = _s(meta.get("page", "?"))
                    snippet = _s(getattr(d, "page_content", ""))[:160]
                    print(f"[{i}] {source} (p{page}): {snippet}...")
        except KeyboardInterrupt:
            break 
        
