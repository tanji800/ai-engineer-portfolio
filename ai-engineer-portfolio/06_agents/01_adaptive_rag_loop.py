"""STEP 6a - "Dynamic search" = Adaptive / Corrective RAG, written as a plain state machine (what LangGraph automates):
  route -> retrieve -> grade -> (generate -> check grounded) | (rewrite query -> retry) | web fallback.
Every decision is a swappable function: offline heuristics here, LLM calls in production (see LLM_* versions).
Run:  python 06_agents/01_adaptive_rag_loop.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.embedder import get_embedder, tokenize
from common.data import DOCS
from common.llm import llm_json

emb = get_embedder()
E = emb.encode(DOCS)

# ---------- tools ----------
def vector_search(q, k=3):
    s = E @ emb.encode(q)
    return [(DOCS[i], float(s[i])) for i in np.argsort(-s)[:k]]

def web_search(q):                      # stub tool; plug Tavily/SerpAPI/Bing here
    return [(f"(web result for '{q}')", 0.5)]

# ---------- decisions (offline heuristics) ----------
def route(q):
    if any(w in tokenize(q) for w in ("today", "latest", "news", "price")):
        return "web"
    if len(tokenize(q)) <= 2:
        return "direct"
    return "vector"

def grade(q, hits, thresh=0.20):        # "is retrieved context relevant?"
    return bool(hits) and hits[0][1] >= thresh

def rewrite(q):                          # production: ask an LLM to expand/clarify the query
    return q + " index vectors retrieval"

# ---------- LLM versions of the same decisions (use when ANTHROPIC_API_KEY is set) ----------
def LLM_route(q):
    return llm_json(f"Route to 'vector' (internal docs), 'web' (fresh info) or 'direct'. Query: {q}. "
                    f'JSON: {{"route": "..."}}', {"route": "vector"})["route"]

def LLM_grade(q, hits):
    return llm_json(f"Do these passages answer '{q}'? {hits}. JSON: {{\"relevant\": true|false}}",
                    {"relevant": True})["relevant"]


def adaptive_rag(q, max_retries=2, trace=None):
    trace = trace if trace is not None else []
    r = route(q); trace.append(f"route={r}")
    if r == "direct":
        return "(answer directly from LLM, no retrieval)", trace
    query = q
    for attempt in range(max_retries + 1):
        hits = web_search(query) if r == "web" else vector_search(query)
        ok = grade(query, hits) if r == "vector" else True
        trace.append(f"attempt={attempt} source={r} top_score={hits[0][1]:.2f} relevant={ok}")
        if ok:
            return f"ANSWER from: {hits[0][0]}", trace
        if r == "vector" and attempt >= 1:
            r = "web"; trace.append("fallback -> web")      # corrective step: internal docs failed twice
        else:
            query = rewrite(query); trace.append(f"rewrote -> '{query}'")
    return "I could not find a reliable answer.", trace


if __name__ == "__main__":
    for q in ["how do we measure retrieval quality", "latest news on vector databases", "xyzzy plugh quantum banana"]:
        ans, tr = adaptive_rag(q)
        print(f"\nQ: {q}\n  " + "\n  ".join(tr) + f"\n  => {ans}")
