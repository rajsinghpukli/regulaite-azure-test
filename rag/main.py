from __future__ import annotations
import json
from rag.pipeline import ask

if __name__ == "__main__":
    q = input("Ask: ").strip()
    ans = ask(q, user_id="dev", history=[])
    print(json.dumps(ans.model_dump(), indent=2, ensure_ascii=False))
