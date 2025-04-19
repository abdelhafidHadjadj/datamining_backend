import pandas as pd
import numpy as np
import uuid
from utils.utils import clean_dataframe, convert_np, normalize_minmax, normalize_zscore
from models.mongo import datasets_collection
from datetime import datetime



def save_dataset(file_name: str, df: pd.DataFrame, metadata):
    # Nettoyage des données pour stockage MongoDB
    raw_data = df.applymap(convert_np).to_dict(orient='records')
    dataset_doc = {
        "_id": str(uuid.uuid4()),
        "file_name": file_name,
        "raw_data": raw_data,
        "cleaned_data": None,
        "metadata": metadata,
        "preprocessing_steps": [],
        "upload_date": datetime.now()
    }

    datasets_collection.insert_one(dataset_doc)
    return dataset_doc

def get_dataset(dataset_id):
    return datasets_collection.find_one({"_id": dataset_id})


def clean_dataset(dataset_id, processing_params):
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    # Utiliser les données déjà traitées si elles existent
    base_data = dataset.get("cleaned_data") or dataset.get("raw_data") or dataset.get("raw_preview")
    df = pd.DataFrame(base_data)

    drop_columns = processing_params.get("drop_columns", [])
    fillna_config = processing_params.get("fill_missing", {})

    # Nettoyage uniquement sur les colonnes non ignorées
    columns_to_clean = [col for col in df.columns if col not in drop_columns]
    df_clean_target = df[columns_to_clean]

    cleaned_df, _ = clean_dataframe(df_clean_target, drop_columns=[], fillna_config=fillna_config)

    # Réinsérer les colonnes ignorées sans modification
    for col in drop_columns:
        if col in df.columns:
            cleaned_df[col] = df[col]

    # Réorganiser dans l'ordre original
    cleaned_df = cleaned_df[df.columns]

    # Ajouter l'étape de nettoyage dans les steps
    steps = dataset.get("preprocessing_steps", [])
    steps.append({
        "step": "cleaning",
        "columns": fillna_config,
        "ignored_columns": drop_columns,
        "method": "statistical",
        "cleaned_at": datetime.now()

    })

    # Mettre à jour le dataset avec les nouvelles données nettoyées
    update_dataset(dataset_id, cleaned_df, preprocessing_steps=steps)

    # Générer une prévisualisation
    preview = cleaned_df.head(10).replace({np.nan: None}).to_dict(orient="records")
    preview = [{k: convert_np(v) for k, v in row.items()} for row in preview]

    return cleaned_df, preview


def update_dataset(dataset_id, cleaned_df: pd.DataFrame = None, preprocessing_steps: list = None, normalized_df: pd.DataFrame = None):
    update_fields = {}

    if cleaned_df is not None:
        cleaned_data = cleaned_df.applymap(convert_np).to_dict(orient='records')
        update_fields["cleaned_data"] = cleaned_data

    if normalized_df is not None:
        normalized_data = normalized_df.applymap(convert_np).to_dict(orient='records')
        update_fields["normalized_data"] = normalized_data

    if preprocessing_steps is not None:
        update_fields["preprocessing_steps"] = preprocessing_steps

    datasets_collection.update_one(
        {"_id": dataset_id},
        {"$set": update_fields}
    )


def normalize_dataset(dataset_id, method="zscore", selected_columns=None):
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    # On part toujours de processed_data (données nettoyées)
    base_data = dataset.get("cleaned_data") or dataset.get("raw_data")
    df = pd.DataFrame(base_data)

    if not selected_columns:
        raise ValueError("No columns selected for normalization")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [col for col in selected_columns if col in numeric_cols]

    if not target_cols:
        raise ValueError("Selected columns are not numeric or do not exist")

    normalized_df = df.copy()

    if method == "zscore":
        for col in target_cols:
            mean = df[col].mean()
            std = df[col].std()
            normalized_df[col] = (df[col] - mean) / std if std != 0 else 0
    elif method == "minmax":
        for col in target_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            normalized_df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0
    else:
        raise ValueError("Unsupported normalization method")

    # Ajouter l'étape à l'historique
    steps = dataset.get("preprocessing_steps", [])
    steps.append({
        "step": "normalization",
        "method": method,
        "columns": target_cols,
        "normalize_at": datetime.now()
    })

    # Enregistrer dans normalized_data uniquement
    update_dataset(dataset_id, normalized_df=normalized_df, preprocessing_steps=steps)

    # Retourner un aperçu
    preview = normalized_df.head(10).replace({np.nan: None}).to_dict(orient="records")
    preview = [{k: convert_np(v) for k, v in row.items()} for row in preview]

    return normalized_df, preview
