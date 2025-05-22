import json
from pathlib import Path
from typing import Dict

class DataCollectionService:
    def save_data_from_file(payload):
        file_path = Path(payload.train_file)

        if not file_path.exists() or not file_path.is_file():
            return {"status": "error", "message": f"File not found: {file_path}"}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, record in enumerate(data):
                return {
                "status": "success",
                "data_size": len(data),
                "message": "Data loaded and validated successfully.",
                "data":data
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
