import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def embed_maintenance_logs(
    json_path="data/maintenance_logs.json",
    index_path="rag/maintenance.index"
):
    # Load maintenance logs
    with open(json_path, "r") as f:
        logs = json.load(f)

    # Convert each record to text (important for retrieval quality)
    documents = []
    for log in logs:
        text = (
            f"Equipment: {log['equipment_type']}. "
            f"Thermal pattern: {log['thermal_pattern']}. "
            f"Observed temperature: {log['observed_temperature']}. "
            f"Failure mode: {log['failure_mode']}. "
            f"Root cause: {log['root_cause']}. "
            f"Action taken: {log['action_taken']}. "
            f"Downtime hours: {log['downtime_hours']}. "
            f"Repair cost USD: {log['repair_cost_usd']}."
        )
        documents.append(text)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = model.encode(documents, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    print("Maintenance logs embedded and FAISS index created.")
    return documents

if __name__ == "__main__":
    embed_maintenance_logs()
