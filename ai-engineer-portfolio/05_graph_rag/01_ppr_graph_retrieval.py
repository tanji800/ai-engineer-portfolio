"""STEP 5a - Why graphs help multi-hop questions: HippoRAG-style Personalized PageRank, no LLM needed.
Question: "Where is the supplier of the recalled X200 part headquartered?"
  hop 1: X200 recall -> Acme Corp   hop 2: Kestrel supplies X200   hop 3: Kestrel HQ -> Dayton, Ohio
Plain vector search finds the recall passage but usually misses the HQ passage (shares no words with the question).
Run:  python 05_graph_rag/01_ppr_graph_retrieval.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
import numpy as np
from common.embedder import get_embedder

PASSAGES = [
    "Acme Corp recalled the X200 brake part in 2023.",
    "Kestrel Components supplies the X200 brake part to Acme Corp.",
    "Kestrel Components is headquartered in Dayton, Ohio.",
    "Bolt Industries is headquartered in Austin, Texas.",
    "Bolt Industries supplies fasteners to Zenith Motors.",
]
ENTITIES = ["X200", "Acme Corp", "Kestrel Components", "Dayton", "Ohio",
            "Bolt Industries", "Austin", "Texas", "Zenith Motors", "fasteners"]


def ents(text):                                    # toy NER; in production use an LLM / spaCy
    return [e for e in ENTITIES if e.lower() in text.lower()]


G = nx.Graph()
for p in PASSAGES:                                 # entities co-occurring in a passage get linked
    es = ents(p)
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            G.add_edge(es[i], es[j])

question = "Where is the supplier of the recalled X200 part headquartered?"
seeds = ents(question)
ppr = nx.pagerank(G, alpha=0.7, personalization={s: 1.0 for s in seeds})   # alpha = damping (stay-near-seeds knob)
passage_score = [sum(ppr.get(e, 0) for e in ents(p)) for p in PASSAGES]

print("seed entities:", seeds)
print("\nGRAPH (PPR) top-3:")
for i in np.argsort(passage_score)[::-1][:3]:
    print(f"  {passage_score[i]:.3f}  {PASSAGES[i]}")

emb = get_embedder()
E = emb.encode(PASSAGES); s = E @ emb.encode(question)
print("\nDENSE top-3:")
for i in np.argsort(-s)[:3]:
    print(f"  {s[i]:.3f}  {PASSAGES[i]}")
