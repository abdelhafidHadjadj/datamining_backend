# clustering_routes.py

from services.clustering_service import (
    run_kmeans, run_kmedoids, run_diana, run_agnes, run_dbscan,
    compute_elbow_kmeans, compute_elbow_kmedoids, generate_visualization
)
from flask import Blueprint, request, jsonify, send_from_directory
import uuid
import time
import numpy as np
import pandas as pd
from models.mongo import datasets_collection, clustering_collection
from datetime import datetime

clustering_bp = Blueprint('cluster', __name__)

# ───────────────────────── HELPERS ─────────────────────────

def load_dataset(dataset_id: str, columns: list[str] | None = None):
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

def safe_jsonify(obj):
    if isinstance(obj, dict):
        return {k: safe_jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_jsonify(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ───────────────────────── ROUTES ─────────────────────────

@clustering_bp.route('/cluster/kmeans/<dataset_id>', methods=['POST'])
def cluster_kmeans(dataset_id):
    body = request.get_json()
    columns = body.get("columns")
    k = int(body.get("k", 3))

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    start_time = time.perf_counter()
    result = run_kmeans(X, k)
    scatter_filename = generate_visualization(X, result["labels"], dataset_id, "kmeans")
    scatter_path = f"/uploads/{scatter_filename}"

    prepared_result = {
        "method": "kmeans",
        "dataset_id": dataset_id,
        "parameters": {
            "n_clusters": k,
            "columns": used_cols
        },
        "metrics": {
            "sse": result["sse"],
            "ssc": result["ssc"],
            "silhouette_score": result["silhouette_score"]
        },
        "results": {
            "labels": result["labels"].tolist(),
            "centers": result["centers"].tolist(),
            "cluster_sizes": {str(c): int((result["labels"] == c).sum()) for c in set(result["labels"])},
            "scatter_path": scatter_path,
            "silhouette_values": result["silhouette_values"].tolist()
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4),
        "created_at": datetime.utcnow().isoformat()
    }
    return jsonify(safe_jsonify(prepared_result))

@clustering_bp.route('/cluster/kmedoids/<dataset_id>', methods=['POST'])
def cluster_kmedoids(dataset_id):
    body = request.get_json()
    columns = body.get("columns")
    k = int(body.get("k", 3))
    metric = body.get("metric", "euclidean")

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    start_time = time.perf_counter()
    result = run_kmedoids(X, k, metric)
    scatter_filename = generate_visualization(X, result["labels"], dataset_id, "kmedoids")
    scatter_path = f"/uploads/{scatter_filename}"

    prepared_result = {
        "method": "kmedoids",
        "dataset_id": dataset_id,
        "parameters": {
            "n_clusters": k,
            "columns": used_cols,
            "metric": metric
        },
        "metrics": {
            "sse": result["sse"],
            "ssc": result["ssc"],
            "silhouette_score": result["silhouette_score"]
        },
        "results": {
            "labels": result["labels"].tolist(),
            "centers": result["centers"].tolist(),
            "cluster_sizes": {str(c): int((result["labels"] == c).sum()) for c in set(result["labels"])},
            "scatter_path": scatter_path,
            "silhouette_values": result["silhouette_values"].tolist()
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4),
        "created_at": datetime.utcnow().isoformat()
    }
    return jsonify(safe_jsonify(prepared_result))

@clustering_bp.route('/cluster/diana/<dataset_id>', methods=['POST'])
def cluster_diana(dataset_id):
    body = request.get_json()
    columns = body.get("columns")
    k = int(body.get("k", 3))

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    start_time = time.perf_counter()
    result = run_diana(X, k)
    scatter_filename = generate_visualization(X, result["labels"], dataset_id, "diana")
    scatter_path = f"/uploads/{scatter_filename}"

    prepared_result = {
        "method": "diana",
        "dataset_id": dataset_id,
        "parameters": {
            "n_clusters": k,
            "columns": used_cols
        },
        "metrics": {
            "silhouette_score": result["silhouette_score"]
        },
        "results": {
            "labels": result["labels"].tolist(),
            "cluster_sizes": {str(c): int((result["labels"] == c).sum()) for c in set(result["labels"])},
            "scatter_path": scatter_path,
            "silhouette_values": result["silhouette_values"].tolist()
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4),
        "created_at": datetime.utcnow().isoformat()
    }
    return jsonify(safe_jsonify(prepared_result))

@clustering_bp.route('/cluster/agnes/<dataset_id>', methods=['POST'])
def cluster_agnes(dataset_id):
    body = request.get_json()
    columns = body.get("columns")
    k = int(body.get("k", 3))

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    start_time = time.perf_counter()
    result = run_agnes(X, k)
    scatter_filename = generate_visualization(X, result["labels"], dataset_id, "agnes")
    scatter_path = f"/uploads/{scatter_filename}"

    prepared_result = {
        "method": "agnes",
        "dataset_id": dataset_id,
        "parameters": {
            "n_clusters": k,
            "columns": used_cols
        },
        "metrics": {
            "silhouette_score": result["silhouette_score"]
        },
        "results": {
            "labels": result["labels"].tolist(),
            "cluster_sizes": {str(c): int((result["labels"] == c).sum()) for c in set(result["labels"])},
            "scatter_path": scatter_path,
            "silhouette_values": result["silhouette_values"].tolist()
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4),
        "created_at": datetime.utcnow().isoformat()
    }
    return jsonify(safe_jsonify(prepared_result))

@clustering_bp.route('/cluster/dbscan/<dataset_id>', methods=['POST'])
def cluster_dbscan(dataset_id):
    body = request.get_json()
    columns = body.get("columns")
    eps = float(body.get("eps", 0.5))
    min_samples = int(body.get("min_samples", 5))

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    start_time = time.perf_counter()
    result = run_dbscan(X, eps, min_samples)
    scatter_filename = generate_visualization(X, result["labels"], dataset_id, "dbscan")
    scatter_path = f"/uploads/{scatter_filename}"

    prepared_result = {
        "method": "dbscan",
        "dataset_id": dataset_id,
        "parameters": {
            "eps": eps,
            "min_samples": min_samples,
            "columns": used_cols
        },
        "metrics": {
            "silhouette_score": result["silhouette_score"]
        },
        "results": {
            "labels": result["labels"].tolist(),
            "cluster_sizes": {str(c): int((result["labels"] == c).sum()) for c in set(result["labels"])},
            "scatter_path": scatter_path,
            "silhouette_values": result["silhouette_values"].tolist()
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4),
        "created_at": datetime.utcnow().isoformat()
    }
    return jsonify(safe_jsonify(prepared_result))

# ─────────────────────── SAVE & UPLOADS ───────────────────────

@clustering_bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@clustering_bp.route('/cluster/save/<dataset_id>', methods=['POST'])
def save_result(dataset_id):
    body = request.get_json()
    if not body:
        return jsonify({"error": "No clustering data provided"}), 400

    body["_id"] = str(uuid.uuid4())
    body["dataset_id"] = dataset_id
    body["created_at"] = datetime.utcnow()

    clustering_collection.insert_one(body)
    return jsonify({"status": "saved", "id": body["_id"]})

@clustering_bp.route('/cluster/elbow/kmeans/<dataset_id>', methods=['POST'])
def elbow_kmeans(dataset_id):
    body = request.get_json(silent=True) or {}
    k_min = int(body.get("k_min", 2))
    k_max = int(body.get("k_max", 10))
    columns = body.get("columns")

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    filename = compute_elbow_kmeans(X, dataset_id, k_min, k_max)
    image_url = f"http://127.0.0.1:5000/uploads/{filename}"

    return jsonify({"image_url": image_url, "columns": used_cols})

@clustering_bp.route('/cluster/elbow/kmedoids/<dataset_id>', methods=['POST'])
def elbow_kmedoids(dataset_id):
    body = request.get_json(silent=True) or {}
    k_min = int(body.get("k_min", 2))
    k_max = int(body.get("k_max", 10))
    metric = body.get("metric", "euclidean")
    columns = body.get("columns")

    try:
        X, used_cols = load_dataset(dataset_id, columns)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if X is None:
        return jsonify({"error": "Dataset not found or not normalized"}), 404

    filename = compute_elbow_kmedoids(X, dataset_id, k_min, k_max, metric)
    image_url = f"http://127.0.0.1:5000/uploads/{filename}"

    return jsonify({"image_url": image_url, "columns": used_cols, "metric": metric})
