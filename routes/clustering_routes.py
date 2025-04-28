from services.clustering_service import run_kmeans, save_clustering_result, compute_elbow_kmeans, compute_elbow_kmedoids
from flask import Blueprint, request, jsonify
import time, numpy as np
import pandas as pd
from models.mongo import datasets_collection

clustering_bp = Blueprint('cluster', __name__)

# ──── ROUTES ─────────────────────────────────────────

@clustering_bp.route('/cluster/kmeans/<dataset_id>', methods=['POST'])
def cluster_kmeans(dataset_id):
    body = request.get_json()
    columns = body.get("columns")          
    k = int(body.get("k", 3))

    X, used_cols = load_dataset(dataset_id, columns)
    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    start_time = time.perf_counter()

    result = run_kmeans(X, k)

    save_clustering_result(
        dataset_id   = dataset_id,
        method       = "kmeans",
        parameters   = {"n_clusters": k, "columns": used_cols},
        X            = X,
        labels       = result["labels"],
        extra_results={
            "centers": result["centers"],
            "silhouette_values": result["silhouette_values"]
        },
        sse          = result["sse"],
        ssc          = result["ssc"],
        start_time   = start_time            # ✅ FIXED: comma was missing
    )
    return jsonify(result)


@clustering_bp.route("/cluster/elbow/kmeans/<dataset_id>", methods=["POST"])
def elbow_kmeans(dataset_id):
    body = request.get_json(silent=True) or {}
    k_min = int(body.get("k_min", 1))
    k_max = int(body.get("k_max", 10))
    columns = body.get("columns") 

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404
    if k_min < 1 or k_max < k_min:
        return jsonify({"error": "Invalid k range"}), 400

    sse = compute_elbow_kmeans(X, k_min, k_max)
    return jsonify({
        "k"      : list(range(k_min, k_max + 1)),
        "sse"    : sse,
        "columns": used_cols
    })


@clustering_bp.route("/cluster/elbow/kmedoids/<dataset_id>", methods=["POST"])
def elbow_kmedoids(dataset_id):
    body = request.get_json(silent=True) or {}
    k_min = int(body.get("k_min", 1))
    k_max = int(body.get("k_max", 10))
    metric = body.get("metric", "euclidean")
    columns = body.get("columns") 

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404
    if k_min < 1 or k_max < k_min:
        return jsonify({"error": "Invalid k range"}), 400

    sse = compute_elbow_kmedoids(X, k_min, k_max, metric)
    return jsonify({
        "k"      : list(range(k_min, k_max + 1)),
        "sse"    : sse,
        "metric" : metric,
        "columns": used_cols
    })

# ──── HELPERS ─────────────────────────────────────────

def load_dataset(dataset_id: str, columns: list[str] | None = None):
    """
    Fetch the *normalized* data for a given dataset id, optionally
    selecting only specific columns.

    Returns
    -------
    X : np.ndarray      (shape: n_samples × n_features)
    used_columns : list[str]
    """
    doc = datasets_collection.find_one({"_id": dataset_id})
    if not doc or not doc.get("normalized_data"):
        return None, None

    df = pd.DataFrame(doc["normalized_data"])

    if columns:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in dataset: {missing}")
        df = df[columns]

    X = df.to_numpy(dtype=float)
    return X, df.columns.tolist()
