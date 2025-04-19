from flask import Blueprint, request, jsonify
import pandas as pd
from services.dataset_service import *
import io
from utils.utils import generate_metadata, clean_dataframe, clean_for_json
from scipy.io import arff
import traceback
from bson.objectid import ObjectId

dataset_bp = Blueprint('dataset', __name__)
# Dictionnaire pour stocker les fichiers temporairement (juste pour l'exemple)
# En production, il faut utiliser un stockage plus approprié comme un S3, une base de données, etc.

uploaded_files = {}



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

        df_raw = None
        df = None
        false_values = ["??", "N/A", "NA", "missing", "undefined", "null", "None", "", "?"]

        if ext == 'csv':
            buffer = io.BytesIO(file_content)
            df_raw = pd.read_csv(io.BytesIO(file_content), on_bad_lines='skip')
            df = pd.read_csv(buffer, na_values=false_values, on_bad_lines='skip')

        elif ext == 'arff':
            # Décode les bytes en string UTF-8
            arff_str = file_content.decode('utf-8')
            arff_buffer = io.StringIO(arff_str)
            data, meta = arff.loadarff(arff_buffer)
            df_raw = pd.DataFrame(data)
            df = df_raw.copy()

            # Décodage des colonnes de type byte
            for col in df.columns:
                if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
                    df[col] = df[col].str.decode('utf-8')

        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Nettoyage des types
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(float)
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(int)

        metadata = generate_metadata(df, file_size)
        dataset = save_dataset(filename, df, metadata)

        uploaded_files['file'] = io.BytesIO(file_content)

        return jsonify({
            'message': 'File uploaded and saved successfully.',
            'data': dataset
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Route pour traiter le fichier
@dataset_bp.route('/process/clean/<dataset_id>', methods=['POST'])
def clean_dataset_route(dataset_id):
    try:
        processing_params = request.json.get("processing_params", {})
        cleaned_df, preview = clean_dataset(dataset_id, processing_params)

        return jsonify({
            "message": "Dataset cleaned successfully.",
            "preview": preview,
            "cleaned_data": cleaned_df.replace({np.nan: None}).to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@dataset_bp.route('/datasets', methods=['GET'])
def get_all_datasets():
    try:
        datasets = list(datasets_collection.find())
        cleaned_datasets = [clean_for_json(dataset) for dataset in datasets]
        return jsonify(cleaned_datasets)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@dataset_bp.route('/datasets/<dataset_id>', methods=['GET'])
def get_dataset_by_id(dataset_id):
    try:
        dataset = datasets_collection.find_one({"_id": str(dataset_id)})

        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        def clean(obj):
            if isinstance(obj, list):
                return [clean(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, float) and pd.isna(obj):
                return None
            elif isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        return jsonify(clean(dataset))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@dataset_bp.route('/process/normalize/<dataset_id>', methods=['POST'])
def normalize_dataset_route(dataset_id):
    try:
        json_data = request.json or {}
        method = json_data.get("method", "zscore")
        selected_columns = json_data.get("columns", [])

        normalized_df, preview = normalize_dataset(dataset_id, method, selected_columns)

        return jsonify({
            "message": f"Dataset normalized using {method} on selected columns.",
            "preview": preview,
            "normalized_data": normalized_df.replace({np.nan: None}).to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
