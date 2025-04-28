from models.mongo import datasets_collection, clustering_collection

def get_dataset_info(dataset_id: str) -> dict | None:
    """Fetch basic dataset info (name, size, date)."""
    doc = datasets_collection.find_one({"_id": dataset_id})
    if not doc:
        return None
    meta = doc.get("metadata", {})
    return {
        "file_name": doc.get("file_name"),
        "upload_date": doc.get("upload_date"),
        "size_in_bytes": meta.get("file_size"),
        "n_features": meta.get("nbr_col"),
    }

def get_all_algorithms_metrics(dataset_id: str) -> list:
    """Fetch all clustering results and their metrics."""
    cursor = clustering_collection.find({"dataset_id": dataset_id})
    results = []
    for doc in cursor:
        results.append({
            "algorithm": doc.get("method"),
            "parameters": doc.get("parameters"),
            "metrics": doc.get("metrics"),
            "runtime_sec": doc.get("runtime_sec"),
            "created_at": doc.get("created_at"),
            "scatter_path": doc.get("results", {}).get("scatter_path")
        })
    return results
