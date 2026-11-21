"""STEP 5b - Production-style graph RAG with LightRAG (https://github.com/HKUDS/LightRAG).
Needs: pip install lightrag-hku + an OpenAI key (its default example). The API changes between versions -
if this errors, copy the current Quick Start from the README and keep the loop over query modes.

Experiment to run and write up in your notes: same questions, 5 modes -> which wins for which question type?
  naive  = plain vector RAG      local  = entity-centric       global = relationship/theme-centric
  hybrid = local+global          mix    = graph + vector
"""
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

QUESTIONS = ["What are the top themes?", "How are the main characters related?"]


async def main():
    rag = LightRAG(working_dir="./rag_storage",
                   embedding_func=openai_embed,
                   llm_model_func=gpt_4o_mini_complete)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.ainsert(open("book.txt").read())          # any long text file
    for q in QUESTIONS:
        for mode in ("naive", "local", "global", "hybrid", "mix"):
            print(f"\n[{mode}] {q}\n", await rag.aquery(q, param=QueryParam(mode=mode)))

asyncio.run(main())
