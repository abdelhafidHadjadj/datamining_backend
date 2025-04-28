import uuid
import pandas as pd
from datetime import datetime
from bson import ObjectId
import numpy as np
from models.mongo import datasets_collection
from utils.utils import convert_np, FALSE_VALUES


# ─── Helpers ──────────────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.utcnow()


# ─── Sauvegarder un dataset brut ──────────────────────────────────────────
def save_dataset(file_name: str, df: pd.DataFrame, meta: dict) -> dict:
    doc = {
        "_id": str(uuid.uuid4()),
        "file_name": file_name,
        "raw_data": df.applymap(convert_np).to_dict("records"),
        "cleaned_data": None,
        "normalized_data": None,
        "metadata": meta,
        "preprocessing_steps": [],
        "upload_date": _now()
    }
    datasets_collection.insert_one(doc)
    return doc


# ─── Récupérer un dataset par ID ──────────────────────────────────────────
def get_dataset(ds_id: str) -> dict | None:
    return datasets_collection.find_one({"_id": ds_id})


# ─── Mettre à jour un dataset (champ + étape) ─────────────────────────────
def update_dataset(ds_id: str, field: str, data, step: str) -> None:
    datasets_collection.update_one(
        {"_id": ds_id},
        {
            "$set": {field: data},
            "$push": {"preprocessing_steps": step}
        }
    )


def clean_dataframe(
    df: pd.DataFrame,
    drop_columns: list[str] = None,
    fillna_config: dict[str, str] = None,
) -> pd.DataFrame:
    df.replace(FALSE_VALUES, np.nan, inplace=True)

    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors="ignore")

    if fillna_config:
        for col, method in fillna_config.items():
            if col not in df.columns:
                continue
            if method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
            elif method == "min":
                df[col] = df[col].fillna(df[col].min())
            elif method == "max":
                df[col] = df[col].fillna(df[col].max())
            else:
                df[col] = df[col].fillna(method)
    
    return df


def normalize_dataframe(df: pd.DataFrame, columns: list[str], method: str = "minmax") -> pd.DataFrame:
    if method not in ["minmax", "zscore", "robust"]:
        raise ValueError(f"Unsupported normalization method: {method}")

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue  # Skip invalid or non-numeric columns

        if method == "minmax":
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val) if min_val != max_val else 0.0

        elif method == "zscore":
            mean, std = df[col].mean(), df[col].std()
            df[col] = (df[col] - mean) / std if std != 0 else 0.0

        elif method == "robust":
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            df[col] = (df[col] - median) / iqr if iqr != 0 else 0.0

    return df