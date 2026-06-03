"""Direct Corpus Interaction (DCI): agentic retrieval over a raw text corpus.

Implements the approach from "Beyond Semantic Similarity: Rethinking Retrieval
for Agentic Search via Direct Corpus Interaction" (arXiv 2605.05242): an LLM
agent searches the raw corpus with terminal-style tools (ripgrep, file reads)
instead of an embedding index, iterating to combine lexical clues across
documents. No vector store, no offline indexing.
"""
