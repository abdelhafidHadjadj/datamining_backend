# ✅ UPDATED: services/clustering_service.py

import uuid, time, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples
)
from scipy.spatial.distance import cdist
from datetime import datetime
from models.mongo import clustering_collection

UPLOAD_FOLDER = 'uploads'

# ───────────────────────
# Generic Metrics Helper
# ───────────────────────
def _base_metrics(X, labels, sse=None, ssc=None):
    metrics = {"sse": sse, "ssc": ssc}
    if len(set(labels)) > 1:
        metrics.update({
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabaz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels)
        })
    return {k: v for k, v in metrics.items() if v is not None}

# ───────────────────────
# Save Clustering Result
# ───────────────────────
def save_clustering_result(dataset_id, method, parameters, X, labels,
                           extra_results=None, sse=None, ssc=None, start_time=None):
    doc = {
        "_id": str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "method": method,
        "parameters": parameters,
        "metrics": _base_metrics(X, labels, sse, ssc),
        "results": {
            "labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
            "cluster_sizes": {str(c): int((labels == c).sum()) for c in set(labels)},
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4) if start_time else None,
        "created_at": datetime.utcnow()
    }

    if extra_results:
        safe_extra = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in extra_results.items()}
        doc["results"].update(safe_extra)

    scatter_filename = generate_visualization(X, labels, dataset_id, method)
    doc["results"]["scatter_url"] = f"/uploads/{scatter_filename}"

    clustering_collection.insert_one(doc)
    return doc

# ───────────────────────
# Elbow (KMeans, KMedoids)
# ───────────────────────
def compute_elbow_kmeans(X, dataset_id, k_min=2, k_max=10):
    sse = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
        sse.append(km.inertia_)
    plt.figure()
    plt.plot(ks, sse, marker='o')
    plt.title('Elbow Method (KMeans)')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.grid(True)
    filename = f"elbow_kmeans_{dataset_id}.png"
    plt.savefig(os.path.join(UPLOAD_FOLDER, filename))
    plt.close()
    return filename

def compute_elbow_kmedoids(X, dataset_id, k_min=2, k_max=10, metric="euclidean"):
    sse = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMedoids(n_clusters=k, metric=metric, method='pam', random_state=42).fit(X)
        medoids = X[km.medoid_indices_]
        dist2 = np.min(cdist(X, medoids, metric=metric)**2, axis=1)
        sse.append(float(dist2.sum()))
    plt.figure()
    plt.plot(ks, sse, marker='o')
    plt.title('Elbow Method (KMedoids)')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.grid(True)
    filename = f"elbow_kmedoids_{dataset_id}.png"
    plt.savefig(os.path.join(UPLOAD_FOLDER, filename))
    plt.close()
    return filename

# ───────────────────────
# Algorithms Runners
# ───────────────────────
def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
    sse = float(model.inertia_)
    normX = X / np.linalg.norm(X, axis=1, keepdims=True)
    normCenters = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    cos_sim = (normX @ normCenters.T)[np.arange(X.shape[0]), labels]
    ssc = float(np.sum(1 - cos_sim))
    sil_score, sil_values = silhouette_infos(X, labels)
    return {"labels": labels, "centers": centers, "sse": sse, "ssc": ssc,
            "silhouette_score": sil_score, "silhouette_values": sil_values}

def run_kmedoids(X, k, metric="euclidean"):
    actual_metric = "cityblock" if metric == "manhattan" else metric
    model = KMedoids(n_clusters=k, metric=metric, method='pam', random_state=42)
    labels = model.fit_predict(X)
    medoids = model.cluster_centers_
    dist2 = np.min(cdist(X, medoids, metric=actual_metric)**2, axis=1)
    sse = float(dist2.sum())
    normX = X / np.linalg.norm(X, axis=1, keepdims=True)
    normMedoids = medoids / np.linalg.norm(medoids, axis=1, keepdims=True)
    cos_sim = (normX @ normMedoids.T)[np.arange(X.shape[0]), labels]
    ssc = float(np.sum(1 - cos_sim))
    sil_score, sil_values = silhouette_infos(X, labels)
    return {"labels": labels, "centers": medoids, "sse": sse, "ssc": ssc,
            "silhouette_score": sil_score, "silhouette_values": sil_values}

def run_diana(X, k, linkage_method="complete"):
    model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=linkage_method)
    labels = model.fit_predict(X)
    sse = calculate_sse(X, labels)
    sil_score, sil_values = silhouette_infos(X, labels)
    return {"labels": labels, "centers": None, "sse": sse, "ssc": None,
            "silhouette_score": sil_score, "silhouette_values": sil_values}

def run_agnes(X, k, linkage_method="ward"):
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X)
    sse = calculate_sse(X, labels)
    sil_score, sil_values = silhouette_infos(X, labels)
    return {"labels": labels, "centers": None, "sse": sse, "ssc": None,
            "silhouette_score": sil_score, "silhouette_values": sil_values}


def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 is noise

    if n_clusters >= 2:
        # Filter out noise points (-1) for silhouette calculation
        mask = labels != -1
        sil_values = silhouette_samples(X[mask], labels[mask], metric="euclidean")
        sil_score_value = silhouette_score(X[mask], labels[mask], metric="euclidean")
    else:
        sil_values = np.zeros(len(X))
        sil_score_value = None

    return {
        "labels": labels,
        "centers": None,
        "sse": None,
        "ssc": None,
        "silhouette_score": sil_score_value,
        "silhouette_values": sil_values
    }



def calculate_sse(X, labels):
    """
    Calcule le SSE (Sum of Squared Errors) pour n'importe quel clustering.

    Args:
        X: array-like, shape (n_samples, n_features)
        labels: array-like, shape (n_samples,)
    
    Returns:
        SSE (float)
    """
    sse = 0
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            continue  # Ignorer les bruités pour DBSCAN
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        sse += np.sum(distances ** 2)

    return sse



# ───────────────────────
# Plot Generators
# ───────────────────────
def generate_visualization(X, labels, dataset_id, method):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if method in ["agnes", "diana"]:
        # Compute linkage for hierarchical clustering
        linkage_method = 'ward' if method == 'agnes' else 'complete'
        linked = linkage(X, method=linkage_method)

        dendrogram(
            linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=False,   # hide labels for clarity
            no_labels=True,           # no cluttered sample names
            color_threshold=0         # auto-color branches
        )
        plt.title(f"Hierarchical Dendrogram ({method.upper()})")
        plt.xlabel('Samples')
        plt.ylabel('Distance')

    else:
        if X.shape[1] >= 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=40)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Clustering Scatter Plot ({method.upper()})')
        else:
            plt.scatter(X[:, 0], np.zeros(len(X)), c=labels, cmap='tab10', s=40)
            plt.title(f'Clustering Scatter Plot ({method.upper()})')

    # Clean background
    ax.set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.grid(True)

    filename = f"scatter_{method}_{dataset_id}_{uuid.uuid4().hex}.png"
    path = os.path.join(UPLOAD_FOLDER, filename)
    plt.savefig(path, bbox_inches='tight')  # Save tightly
    plt.close()

    return filename

# ───────────────────────
# Helper (Silhouette Info)
# ───────────────────────
def silhouette_infos(X, labels):
    if len(set(labels)) > 1:
        sil_values = silhouette_samples(X, labels, metric="euclidean")
        sil_score_value = silhouette_score(X, labels, metric="euclidean")
        return sil_score_value, sil_values
    else:
        return None, np.zeros(len(X))




def silhouette_score_dbscan(X, labels):
    """
    Calcule le Silhouette Score pour un clustering DBSCAN
    en ignorant les points bruités (label = -1).

    Args:
        X: array-like, shape (n_samples, n_features)
        labels: array-like, shape (n_samples,)
    
    Returns:
        silhouette score (float) ou None si pas calculable
    """
    mask = labels != -1  # Filtrer les bruités
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    # Vérifier qu'il y a au moins 2 clusters
    if len(set(labels_filtered)) > 1:
        score = silhouette_score(X_filtered, labels_filtered)
        return score
    else:
        return None
