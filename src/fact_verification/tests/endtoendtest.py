import pandas as pd

from fact_verification.retrieval.bm25 import BM25Retriever
from fact_verification.retrieval.dpr import DPRRetriever
from fact_verification.retrieval.hybrid import HybridRetriever, HybridConfig

from fact_verification.reranking.cross_encoder import CrossEncoderReranker

candidates = pd.DataFrame([
    {
        "candidate_id": "p0",
        "source_url": "source-0",
        "text": "Gustave Eiffel was born in Dijon on 15 December 1832.",
    },
    {
        "candidate_id": "p1",
        "source_url": "source-1",
        "text": "The Eiffel Tower was constructed for the 1889 Exposition Universelle.",
    },
    {
        "candidate_id": "p2",
        "source_url": "source-2",
        "text": "Marie Curie was born in Warsaw in 1867.",
    },
    {
        "candidate_id": "p3",
        "source_url": "source-3",
        "text": "Paris is the capital of France.",
    },
])

query = "Gustave Eiffel was born in Dijon."

bm25 = BM25Retriever()
dpr = DPRRetriever()
hybrid = HybridRetriever(lexical_retriever=bm25, dense_retriever=dpr, config=HybridConfig(method="rrf"))
reranker = CrossEncoderReranker()

bm25_results = bm25.retrieve(query=query, candidates=candidates, k=4)
dpr_results = dpr.retrieve(query=query, candidates=candidates, k=4)
hybrid_results = hybrid.fuse(lexical_results=bm25_results, dense_results=dpr_results, k=4)
reranked_results = reranker.rerank(query=query, candidates=hybrid_results, k=4)

print("BM25")
print(bm25_results)

print("DPR")
print(dpr_results)

print("Hybrid")
print(hybrid_results)

print("Hybrid + Cross-Encoder")
print(reranked_results)