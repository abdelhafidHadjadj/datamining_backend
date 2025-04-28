from flask import Blueprint, request, jsonify
import pandas as pd
import io
import traceback
from bson.objectid import ObjectId
from scipy.io import arff
from datetime import datetime

from services.dataset_service import save_dataset, get_dataset, update_dataset, clean_dataframe, normalize_dataframe
from models.mongo import datasets_collection
from utils.utils import (
    FALSE_VALUES,
    generate_metadata,
    clean_for_json,
    convert_np,
)

dataset_bp = Blueprint('dataset', __name__)

uploaded_files = {}  # temporaire, pour debug

# ─────────────────────────── UPLOAD ─────────────────────────────
@dataset_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = file.filename
        file_content = file.read()
        file_size = len(file_content)
        ext = filename.split('.')[-1].lower()

        # Chargement du fichier en DataFrame
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_content), na_values=FALSE_VALUES, on_bad_lines='skip')

        elif ext == 'arff':
            arff_str = file_content.decode('utf-8')
            arff_buffer = io.StringIO(arff_str)
            data, meta = arff.loadarff(arff_buffer)
            df = pd.DataFrame(data)

            for col in df.columns:
                if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
                    df[col] = df[col].str.decode('utf-8')
            df.replace(FALSE_VALUES, pd.NA, inplace=True)

        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Nettoyage des types
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(float)
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(int)

        metadata = generate_metadata(df, file_size)
        dataset = save_dataset(filename, df, metadata)

        uploaded_files['file'] = io.BytesIO(file_content)  # optionnel

        return jsonify({
            'message': 'File uploaded and saved successfully.',
            'data': clean_for_json(dataset)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ─────────────────────────── GET ALL ─────────────────────────────
@dataset_bp.route('/datasets', methods=['GET'])
def get_all_datasets():
    try:
        datasets = list(datasets_collection.find())
        return jsonify([clean_for_json(dataset) for dataset in datasets])
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ─────────────────────────── GET BY ID ─────────────────────────────
@dataset_bp.route('/datasets/<dataset_id>', methods=['GET'])
def get_dataset_by_id(dataset_id):
    try:
        dataset = datasets_collection.find_one({"_id": str(dataset_id)})
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        return jsonify(clean_for_json(dataset))
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ─────────────────────────── CLEAN ─────────────────────────────
@dataset_bp.route("/datasets/clean/<ds_id>", methods=["POST"])
def clean_ds(ds_id):
    try:
        body = request.get_json(force=True)
        ds = get_dataset(ds_id)
        if not ds:
            return jsonify({"error": "Dataset not found"}), 404

        # 1️⃣  Re-create a DataFrame from the raw data
        df = pd.DataFrame(ds["raw_data"])

        # 2️⃣  Apply the cleaning logic
        df_clean = clean_dataframe(
            df,
            drop_columns=body.get("drop_columns"),
            fillna_config=body.get("fillna")
        )

        # 3️⃣  Convert to pure-Python types so Mongo + JSON can store/return it
        cleaned_records = df_clean.applymap(convert_np).to_dict("records")

        # 4️⃣  Persist the cleaned version in MongoDB
        update_dataset(
            ds_id,
            "cleaned_data",
            cleaned_records,
            "cleaning"
        )

        # 5️⃣  Return both a status message **and** the cleaned dataset
        return jsonify({
            "message": "Dataset cleaned successfully.",
            "dataset_id": ds_id,
            "row_count": len(cleaned_records),
            "cleaned_data": cleaned_records        # ← what you asked for
        }), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@dataset_bp.route('/datasets/normalize/<dataset_id>', methods=['POST'])
def normalize_dataset(dataset_id):
    try:
        # Validate input JSON
        body = request.get_json()
        if not body:
            return jsonify({"error": "Missing request body"}), 400

        # Here, body is expected to be: { column_name: method_name, ... }
        if not isinstance(body, dict) or not all(isinstance(v, str) for v in body.values()):
            return jsonify({"error": "Invalid format. Expected { column: method }"}), 400

        selected_columns = list(body.keys())
        method_by_column = body  # this is your dict {column: method}

        # Fetch dataset
        dataset = get_dataset(dataset_id)
        if not dataset or not dataset.get('cleaned_data'):
            return jsonify({"error": "Dataset not found or not cleaned"}), 404

        df = pd.DataFrame(dataset['cleaned_data'])

        # Normalize selected columns individually
        normalized_df = df.copy()
        for col in selected_columns:
            if col not in df.columns:
                return jsonify({"error": f"Column '{col}' not found in dataset"}), 400

            normalized_df[col] = normalize_column(df[col], method_by_column[col])

        # Save normalized data
        update_dataset(
            ds_id=dataset_id,
            field="normalized_data",
            data=normalized_df.applymap(convert_np).to_dict("records"),
            step="Normalization (custom methods)"
        )

        return jsonify({
            "message": "Dataset normalized successfully.",
            "preview": normalized_df.applymap(convert_np).to_dict("records")
        }), 200

    except Exception as e:
        print("[ERROR]", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def normalize_column(series, method):
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'robust':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return (series - q1) / (q3 - q1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
