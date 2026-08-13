import numpy as np
import pandas as pd

from fact_verification.retrieval.dpr import DPRRetriever


candidates = pd.DataFrame([
    {
        "candidate_id": "p0",
        "source_url": "source-a",
        "text": "Gustave Eiffel was born in Dijon in 1832.",
    },
    {
        "candidate_id": "p1",
        "source_url": "source-b",
        "text": "The Eiffel Tower was completed in Paris in 1889.",
    },
    {
        "candidate_id": "p2",
        "source_url": "source-c",
        "text": "Marie Curie was born in Warsaw.",
    },
])


retriever = DPRRetriever()

results = retriever.retrieve(query="Gustave Eiffel was born in Dijon.", candidates=candidates,k=3)

print(results[["rank", "candidate_id", "score", "text"]])
print("finished")