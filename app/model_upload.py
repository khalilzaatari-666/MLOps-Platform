import os
from typing import Dict, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from ultralytics import YOLO
from app.models import ModelModel
from dotenv import load_dotenv 

load_dotenv()
ALLOWED_MODELS = os.getenv("ALLOWED_MODELS")

MODELS_DIR = "pretrained_models"
# Ensure models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

upload_progress: Dict[str, Dict[str, Union[str, int, float]]] = {}

class UploadProgress:
    def __init__(self, upload_id: str, filename: str, total_size: int) -> None:
        """
        Initialize the upload progress tracker.
        :param upload_id: Unique identifier for the upload
        :param filename: Name of the file being uploaded
        :param total_size: Total size of the file in bytes
        """
        self.upload_id = upload_id
        self.filename = filename
        self.total_size = total_size
        self.uploaded_size = 0
        self.status = "uploading"
        self.created_at = datetime.utcnow()

    def update_progress(self, chunk_size: int):
        self.uploaded_size += chunk_size
        progress_percent = (self.uploaded_size / self.total_size) * 100
        upload_progress[self.upload_id] = {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "total_size": self.total_size,
            "uploaded_size": self.uploaded_size,
            "progress_percent": round(progress_percent,2),
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }   

def extract_model_metadata(file_path: str, model_name: str) -> dict:
    """Extract metadata from YOLO model file"""
    try:
        # Load YOLO model
        model = YOLO(str(file_path))
        
        # Extract class names
        class_names = list(getattr(model, 'names', {}).values())
        
        # Determine model type based on filename
        model_type = "detect"  # Default
        if "-seg" in model_name: 
            model_type = "segment"
        elif "-pose" in model_name: 
            model_type = "pose"
        
        # Extract device info
        device = str(model.device) if hasattr(model, 'device') else 'cpu'
        
        # Extract input size
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'args') and isinstance(model.model.args, dict):
                input_size = model.model.args.get("imgsz", 640)
            else:
                input_size = 640
        except AttributeError:
            input_size = 640
        
        return {
            'model_type': model_type,
            'input_size': input_size,
            'class_names': class_names,
            'device': device
        }
        
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
        # Return default metadata if extraction fails
        model_type = "detect"
        if "-seg" in model_name: 
            model_type = "segment"
        elif "-pose" in model_name: 
            model_type = "pose"
            
        return {
            'model_type': model_type,
            'input_size': 640,
            'class_names': [],
            'device': 'cpu'
        }

async def save_file(
    file_contents: bytes,
    upload_id: str,
    model_path: str,
    model_name: str,
    group: Optional[str],
    db: Session
):
    try:
        upload_progress[upload_id]["status"] = "uploading"
        total_size = len(file_contents)
        upload_progress[upload_id]["total_size"] = total_size
        
        # Write file to disk
        with open(model_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            for i in range(0, total_size, chunk_size):
                chunk = file_contents[i:i+chunk_size]
                f.write(chunk)
                upload_progress[upload_id]["uploaded_size"] += len(chunk)
                upload_progress[upload_id]["progress_percent"] = (
                    upload_progress[upload_id]["uploaded_size"] / total_size * 100
                )
        
        # Extract metadata after file is saved
        model_metadata = extract_model_metadata(file_path=model_path, model_name=model_name)
        if not model_metadata:
            raise ValueError("Failed to extract model metadata")
        
        # Check if model exists again (race condition protection)
        existing_model = db.query(ModelModel).filter(ModelModel.name == model_name).first()
        
        if existing_model:
            # Update existing model
            existing_model.model_path = model_path
            existing_model.group = group if group else existing_model.group
            existing_model.model_type = model_metadata['model_type']
            existing_model.input_size = model_metadata['input_size']
            existing_model.class_names = model_metadata['class_names']
            existing_model.device = model_metadata['device']
            existing_model.is_active = True
            existing_model.last_used = datetime.now()
            model_id = existing_model.id
        else:
            # Create new model
            new_model = ModelModel(
                name=model_name,
                group=group,
                model_type=model_metadata['model_type'],
                model_path=model_path,
                input_size=model_metadata['input_size'],
                class_names=model_metadata['class_names'],
                device=model_metadata['device'],
                is_active=True,
                last_used=datetime.now()
            )
            db.add(new_model)
            model_id = new_model.id
        
        db.commit()
        upload_progress[upload_id]["status"] = "completed"
        print(f"Model {model_name} successfully uploaded and metadata added to database.")
        
    except Exception as e:
        upload_progress[upload_id]["status"] = "failed"
        upload_progress[upload_id]["error"] = str(e)
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Don't raise as this is a background task
        logger.error(f"Failed to process model upload: {str(e)}")