import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from collections import defaultdict
import json
from bson import ObjectId
from datetime import datetime
def clean_dataframe(df, drop_columns=None, fillna_config=None):
    false_values = ["??", "N/A", "NA", "missing", "undefined", "null", "None", "", "?"]
    df.replace(false_values, np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"[WARN] Échec de conversion de la colonne '{col}' : {e}")

    analysis_info = []
    total_rows = len(df)

    print("=== ANALYSE AVANT TRAITEMENT ===")
    for col in df.columns:
        missing_indices = df[df[col].isna()].index.tolist()
        info = {
            "column": col,
            "dtype": df[col].dtype,
            "total_rows": total_rows,
            "missing_count": df[col].isna().sum(),
            "missing_indices": missing_indices
        }
        analysis_info.append(info)
        print(f"[INFO] {col} - Type: {info['dtype']}, Total: {info['total_rows']}, Missing: {info['missing_count']}")

    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    if fillna_config:
        for col, method in fillna_config.items():
            if col not in df.columns:
                print(f"[WARN] Colonne '{col}' absente, ignorée.")
                continue
            try:
                if method == 'mean':
                    val = df[col].mean()
                elif method == 'median':
                    val = df[col].median()
                elif method == 'mode':
                    mode_series = df[col].mode()
                    val = mode_series[0] if not mode_series.empty else None
                elif method == 'min':
                    val = df[col].min()
                elif method == 'max':
                    val = df[col].max()
                else:
                    val = method  # default value

                df[col].fillna(val, inplace=True)
                print(f"[DEBUG] Colonne {col} remplie par {method}: {val}")
            except Exception as e:
                print(f"[ERROR] Erreur lors du remplissage de '{col}' : {e}")

    return df, analysis_info


def generate_metadata(df: pd.DataFrame, file_size) -> dict:
    metadata = {
        "file_size": file_size,
        "columns": df.columns.tolist(),
        "nbr_col": len(df.columns),
        "missing_values": {},
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }

    for col in df.columns:
        missing_positions = df[df[col].isnull()].index.tolist()
        if missing_positions:
            metadata["missing_values"][col] = {
                "count": len(missing_positions),
                "position": missing_positions
            }

    return metadata


def convert_np(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64)):
        return float(value)
    elif pd.isna(value):
        return None
    return value



def normalize_zscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def normalize_minmax(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def compute_metrics(data, labels):
   

    unique_labels = list(set(labels))
    print("⛏️ Types dans compute_metrics:")
    print(data.dtypes)
    print("Shape:", data.shape)
    print("Type des labels:", type(labels), "Exemple:", labels[:5])

    if len(unique_labels) <= 1 or (set(unique_labels) == {-1}):
        return {
            "silhouette_score": None,
            "intra_class_distance": None,
            "inter_class_distance": None
        }

    try:
        silhouette = silhouette_score(data, labels)
    except Exception as e:
        print("Erreur silhouette_score:", e)
        silhouette = None

    # Intra-class distance
    clusters = defaultdict(list)
    for point, label in zip(data.values, labels):  # ✅ fix ici
        if label != -1:
            clusters[label].append(point)

    intra_dists = []
    centroids = []

    for points in clusters.values():
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)
        distances = np.linalg.norm(points - centroid, axis=1)
        intra_dists.extend(distances)

    intra_class = np.mean(intra_dists) if intra_dists else None

    # Inter-class distance
    centroids = np.array(centroids)
    if len(centroids) > 1:
        pairwise_dists = cdist(centroids, centroids)
        upper_triangle = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
        inter_class = np.mean(upper_triangle)
    else:
        inter_class = None

    return {
        "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
        "intra_class_distance": round(intra_class, 4) if intra_class is not None else None,
        "inter_class_distance": round(inter_class, 4) if inter_class is not None else None
    }




def convert_to_serializable(obj):
    """Convertit des objets numpy en types Python natifs compatibles JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Conversion en liste
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj  # Si c'est un objet compatible JSON, retourne-le tel quel


def clean_for_json(obj):
    """Recursively clean data for JSON serialization."""
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj