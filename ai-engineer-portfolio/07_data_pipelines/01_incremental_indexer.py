"""STEP 7 - Data-engineering core: incremental indexing (embed only the delta) + deletes + model-version safety.
Re-embedding a whole corpus on every run is slow and costly. Content-hash each chunk; re-embed only what changed.
Run:  python 07_data_pipelines/01_incremental_indexer.py
Production equivalents: CocoIndex, Airflow/Dagster + CDC, Kafka consumers upserting into pgvector/Qdrant.
"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.embedder import get_embedder

MODEL_VERSION = "hash-512-v1"       # include in the hash: changing the embedding model MUST re-embed everything


class IncrementalIndex:
    def __init__(self):
        self.emb = get_embedder()
        self.store = {}              # doc_id -> (content_hash, vector)   (a vector DB / pgvector table in real life)
        self.embed_calls = 0

    @staticmethod
    def _h(text):
        return hashlib.sha256((MODEL_VERSION + "|" + text).encode()).hexdigest()

    def sync(self, source: dict):
        added = updated = deleted = skipped = 0
        for did, text in source.items():
            h = self._h(text)
            if did not in self.store:
                added += 1
            elif self.store[did][0] != h:
                updated += 1
            else:
                skipped += 1
                continue
            self.store[did] = (h, self.emb.encode(text)); self.embed_calls += 1
        for did in set(self.store) - set(source):           # deletes: easy to forget, causes stale answers
            del self.store[did]; deleted += 1
        return dict(added=added, updated=updated, deleted=deleted, skipped=skipped, embed_calls=self.embed_calls)


if __name__ == "__main__":
    idx = IncrementalIndex()
    src = {"a": "HNSW is a graph index", "b": "IVF uses clusters", "c": "BM25 is lexical"}
    print("run 1:", idx.sync(src))
    src["b"] = "IVF uses k-means clusters"; del src["c"]; src["d"] = "pgvector lives in postgres"
    print("run 2:", idx.sync(src))
    print("run 3:", idx.sync(src), "<- nothing changed => zero embedding calls")
