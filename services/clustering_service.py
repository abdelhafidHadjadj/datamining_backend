# services/clustering_service.py (add at bottom)
import uuid, time
from datetime import datetime
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from models.mongo import clustering_collection
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids              # pip install scikit-extra
from scipy.spatial.distance import cdist                # for K-Medoids SSE
import numpy as np




def _base_metrics(X, labels, sse=None, ssc=None):
    """Compute all generic metrics that exist for this algorithm."""
    metrics = {
        "sse"             : sse,
        "ssc"             : ssc,
        "silhouette_avg"  : silhouette_score(X, labels) if len(set(labels)) > 1 else None,
        "calinski_harabaz": calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else None,
        "davies_bouldin"  : davies_bouldin_score(X, labels)    if len(set(labels)) > 1 else None,
    }
    return {k: v for k, v in metrics.items() if v is not None}

def save_clustering_result(dataset_id, method,
                           parameters,          # dict
                           X, labels,           # ndarray + list/ndarray
                           extra_results=None,  # dict specific to the algo
                           sse=None, ssc=None,  # optional pre-computed scores
                           start_time=None):
    """
    Generic saver: plugs into every algorithm’s route.
    """
    doc = {
        "_id"       : str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "method"    : method,
        "parameters": parameters,
        "metrics"   : _base_metrics(X, labels, sse, ssc),
        "results"   : {
            "labels"           : labels.tolist() if hasattr(labels, "tolist") else list(labels),
            "cluster_sizes"    : {str(c): int((labels == c).sum()) for c in set(labels)},
        },
        "runtime_sec": round(time.perf_counter() - start_time, 4) if start_time else None,
        "created_at" : datetime.utcnow()
    }
    if extra_results:
        doc["results"].update(extra_results)

    clustering_collection.insert_one(doc)
    return doc   # handy if caller wants to send it back to the UI



# ────────────────────────────  ELBOW HELPERS  ────────────────────────────


def compute_elbow_kmeans(X: np.ndarray, k_min: int = 1, k_max: int = 10):
    """Returns a list of SSE (inertia) for k = k_min … k_max (inclusive)."""
    sse = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        sse.append(float(km.inertia_))                  # inertia_ = SSE
    return sse

def compute_elbow_kmedoids(X: np.ndarray,
                           k_min: int = 1,
                           k_max: int = 10,
                           metric: str = "euclidean"):
    """Returns SSE for K-Medoids; we compute it manually."""
    sse = []
    for k in range(k_min, k_max + 1):
        km = KMedoids(n_clusters=k,
                      metric=metric,
                      method="pam",
                      random_state=42)
        km.fit(X)
        medoids = X[km.medoid_indices_]                # true representatives
        # Squared Euclidean distance to nearest medoid
        dist2 = np.min(cdist(X, medoids, metric=metric)**2, axis=1)
        sse.append(float(dist2.sum()))
    return sse



def run_kmeans(X: np.ndarray, k: int):
    """
    Run K-Means clustering on given data.

    Parameters
    ----------
    X : np.ndarray
        The input data (n_samples, n_features).
    k : int
        Number of clusters to form.

    Returns
    -------
    result : dict
        {
            "labels": np.ndarray,
            "centers": np.ndarray,
            "sse": float,
            "ssc": float,
            "silhouette_values": np.ndarray
        }
    """
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    # SSE
    sse = float(model.inertia_)

    # SSC (Sum of Squared Cosine distances)
    normX = X / np.linalg.norm(X, axis=1, keepdims=True)
    normCenters = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    cos_sim = (normX @ normCenters.T)[np.arange(X.shape[0]), labels]
    ssc = float(np.sum(1 - cos_sim))

    # Silhouette per sample
    if len(set(labels)) > 1:  # silhouette needs at least 2 clusters
        sil_values = silhouette_score(X, labels, sample_size=len(X), random_state=42, metric="euclidean")
        from sklearn.metrics import silhouette_samples
        sil_samples = silhouette_samples(X, labels, metric="euclidean")
    else:
        sil_values = None
        sil_samples = np.zeros(len(X))

    return {
        "labels": labels,
        "centers": centers,
        "sse": sse,
        "ssc": ssc,
        "silhouette_score": sil_values,
        "silhouette_values": sil_samples
    }
