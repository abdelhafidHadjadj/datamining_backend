# utils/utils.py
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bson import ObjectId
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

FALSE_VALUES = [
    "??",
    "N/A",
    "NA",
    "missing",
    "undefined",
    "null",
    "None",
    "",
    "?",
]


# ───────────────────────── helpers ─────────────────────────
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip().lower() for c in df2.columns]
    return df2


def profile_dataframe(df: pd.DataFrame) -> Dict:
    df = normalize_cols(df)
    prof = {}
    for col in df.columns:
        s = df[col]
        prof[col] = {
            "dtype": str(s.dtype),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "skew": float(s.skew()) if s.dtype.kind in "if" else None,
            "top": s.value_counts(dropna=True).head(3).to_dict(),
        }
    return prof


def clean_dataframe(
    df: pd.DataFrame, rules: Dict[str, Dict]
) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_cols(df)
    df.replace(FALSE_VALUES, np.nan, inplace=True)

    for col in df.select_dtypes(include="object"):
        df[col] = pd.to_numeric(df[col], errors="ignore")

    to_drop, imputed = [], []
    for col, spec in rules.items():
        if col not in df.columns:
            continue
        if spec.get("action") == "drop":
            to_drop.append(col)
            continue

        method = spec.get("method", "").lower()
        if method:
            if method == "mean":
                val = df[col].mean()
            elif method == "median":
                val = df[col].median()
            elif method == "mode":
                val = (
                    df[col].mode(dropna=True).iloc[0]
                    if not df[col].mode().empty
                    else np.nan
                )
            elif method == "min":
                val = df[col].min()
            elif method == "max":
                val = df[col].max()
            else:
                val = spec["method"]
            df[col] = df[col].fillna(val)
            imputed.append(col)

    if to_drop:
        df.drop(columns=to_drop, inplace=True)

    return df, imputed


def generate_metadata(df: pd.DataFrame, file_size) -> dict:
    metadata = {
        "file_size": file_size,
        "columns": df.columns.tolist(),
        "nbr_col": len(df.columns),
        "missing_values": {},
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    for col in df.columns:
        missing_positions = df[df[col].isnull()].index.tolist()
        if missing_positions:
            metadata["missing_values"][col] = {
                "count": len(missing_positions),
                "position": missing_positions,
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
    if len(unique_labels) <= 1 or (set(unique_labels) == {-1}):
        return {
            "silhouette_score": None,
            "intra_class_distance": None,
            "inter_class_distance": None,
        }

    try:
        silhouette = silhouette_score(data, labels)
    except Exception:
        silhouette = None

    clusters = defaultdict(list)
    for point, label in zip(data.values, labels):
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

    centroids = np.array(centroids)
    if len(centroids) > 1:
        pairwise_dists = cdist(centroids, centroids)
        upper_triangle = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
        inter_class = np.mean(upper_triangle)
    else:
        inter_class = None

    return {
        "silhouette_score": round(silhouette, 4)
        if silhouette is not None
        else None,
        "intra_class_distance": round(intra_class, 4)
        if intra_class is not None
        else None,
        "inter_class_distance": round(inter_class, 4)
        if inter_class is not None
        else None,
    }


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def clean_for_json(obj):
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
