import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np

class MaintenanceVectorStore:
    def __init__(
        self,
        index_path="rag/maintenance.index",
        json_path="data/maintenance_logs.json"
    ):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(json_path, "r") as f:
            self.logs = json.load(f)

    def retrieve(self, query_text, top_k=3):
        query_embedding = self.model.encode(
            [query_text], convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.logs[idx])

        return results
