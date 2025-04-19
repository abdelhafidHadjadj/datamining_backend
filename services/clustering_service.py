from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import uuid
import json
from utils.utils import compute_metrics, convert_to_serializable
from models.mongo import datasets_collection, clustering_results_collection
def perform_clustering(dataset_id, method, n_clusters, eps, min_samples, columns=None):
    # Convertir dataset_id en str (assure la compatibilité avec MongoDB)
    dataset_id = str(dataset_id)
    print("Requête MongoDB avec l'ID:", dataset_id)

    # Récupération du dataset depuis MongoDB
    dataset = datasets_collection.find_one({"_id": dataset_id})
    if not dataset:
        raise ValueError("Dataset not found")

    # Chargement des données nettoyées ou normalisées
    df = pd.DataFrame(dataset.get("normalized_data") or dataset.get("cleaned_data"))
    if df.empty:
        raise ValueError("Dataset is empty")

    # Si des colonnes spécifiques sont fournies, filtrer le DataFrame
    if columns:
        try:
            df = df[columns]
        except KeyError as e:
            raise ValueError(f"Invalid column names: {e}")

    print("col selected: ", df)
    # Forcer la conversion de toutes les colonnes spécifiées en numérique
    if columns:
        df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    # Vérification que des colonnes numériques sont disponibles
    if numeric_df.empty:
        raise ValueError("No numeric columns available for clustering after conversion.")

    # Option : drop les lignes NaN ou les remplir selon besoin
    numeric_df = numeric_df.dropna()  # ou .fillna(0) si tu préfères

    print("Dtypes après conversion :", numeric_df.dtypes)
    print("Aperçu des données numériques :", numeric_df.head())

    # Choix du modèle
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters)
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "agnes":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    elif method == "diana":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
    elif method == "kmedoids":
        model = KMedoids(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}. Supported methods: kmeans, dbscan, agnes, diana, kmedoids.")

    # Calcul des clusters
    labels = model.fit_predict(numeric_df)
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    if method == "dbscan":
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Calcul des métriques
    metrics = compute_metrics(numeric_df, labels)

    # Debug infos
    print("Labels:", labels)
    print("Metrics:", metrics)

    # Nettoyage des colonnes (si dictionnaire)
    if isinstance(columns, dict):
        columns = list(columns.keys())
    elif not isinstance(columns, list):
        columns = [columns]

    # S’assurer que toutes les colonnes sont des chaînes de caractères
    columns = [str(col) for col in columns]

    # Préparation des données à insérer
    result = {
        "_id": str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "method": method,
        "parameters": {
            "n_clusters": n_clusters,
            "eps": eps,
            "min_samples": min_samples,
            "columns": columns if isinstance(columns, list) else [columns]
        },
        "labels": labels,
        "metrics": metrics,  # Stocké comme JSON string
        "created_at": datetime.now()
    }

    print("Structure à insérer :", result)

    # Insertion dans la base MongoDB
    clustering_results_collection.insert_one(result)

    # Génération d’un aperçu des clusters


    return result
