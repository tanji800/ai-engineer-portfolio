import importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_rrf_prefers_docs_ranked_well_in_both_lists():
    m = load("04_hybrid_search/01_bm25_dense_rrf_eval.py")
    assert m.rrf([[1, 2, 3], [2, 1, 4]])[:2] in ([1, 2], [2, 1])
    assert m.rrf([[9, 1], [8, 1]])[0] == 1          # doc present in both lists wins

def test_ivf_full_probe_matches_exact():
    m = load("02_vector_search/01_flat_and_ivf.py")
    X = m.make_data(n=2000, d=16, clusters=10)
    ivf = m.IVF(nlist=8); ivf.train_add(X)
    q = X[5]
    assert set(ivf.search(q, 5, nprobe=8)) == set(m.flat_search(X, q, 5))

def test_incremental_only_embeds_delta():
    m = load("07_data_pipelines/01_incremental_indexer.py")
    idx = m.IncrementalIndex()
    idx.sync({"a": "x", "b": "y"})
    r = idx.sync({"a": "x", "b": "y2", "c": "z"})
    assert (r["skipped"], r["updated"], r["added"]) == (1, 1, 1)

def test_agent_calculator_blocks_unsafe_input():
    m = load("06_agents/02_tool_calling_agent.py")
    assert m.calculator("2*3") == "6"
    try:
        m.calculator("__import__('os')"); assert False
    except Exception:
        pass
