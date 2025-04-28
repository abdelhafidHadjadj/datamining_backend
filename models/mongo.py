from config import get_db

db = get_db()
datasets_collection = db["datasets"]
clustering_collection = db["clusters"]
