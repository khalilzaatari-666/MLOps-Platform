import os
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from app.models import ModelModel
from fastapi import UploadFile
from ultralytics import YOLO

MODELS_DIR = "pretrained_models"

async def upload_model(
    model_file: UploadFile,
    group: str = None,
    db: Session = None
):
    """
    Uploads a YOLO model file, saves it locally, and records its metadata in the DB.
    """
    try:
        # Ensure the models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Define the full save path
        save_path = os.path.join(MODELS_DIR, model_file.filename)

        # Save the uploaded file to the file system
        with open(save_path, "wb") as f:
            contents = await model_file.read()
            f.write(contents)

        # Load the YOLO model
        model = YOLO(str(save_path))

        # Extract model name without extension
        model_name = Path(model_file.filename).stem

        # Check if model already exists
        existing_model = db.query(ModelModel).filter_by(name=model_name).first()
        if existing_model:
            return {'message': 'Model already exists'}

        # Extract class names
        class_names = list(getattr(model, 'names', {}).values())

        # Determine model type
        model_type = "detect"
        if "-seg" in model_name:
            model_type = "segment"
        elif "-pose" in model_name:
            model_type = "pose"

        # Extract input size
        input_size = model.model.args.get("imgsz", 640)  # type: ignore

        # Device info
        device = str(getattr(model, 'device', 'cpu'))

        # Create database model instance
        db_model = ModelModel(
            name=model_name,
            group=group,
            model_type=model_type,
            model_path=str(save_path),
            input_size=input_size,
            class_names=class_names,
            device=device,
            last_used=datetime.now()
        )

        # Add and commit
        db.add(db_model)
        db.commit()
        db.refresh(db_model)

        return {"message": "Model uploaded and registered", "model_id": db_model.id}

    except Exception as e:
        return {"error": str(e)}