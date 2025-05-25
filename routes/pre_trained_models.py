from datetime import datetime
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File, BackgroundTasks
from core.dependencies import get_db
from sqlalchemy.orm import Session
from app.models import ModelModel
from app.model_upload import extract_model_metadata, save_file, upload_progress
import os
import uuid
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "pretrained_models")

@router.post("/upload-model")
async def upload_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    group: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a .pt model file with progress tracking"""
    
    upload_id = str(uuid.uuid4())
    
    if not file.filename or not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Only .pt files are allowed")
    
    model_name = Path(file.filename).stem
    existing_model = db.query(ModelModel).filter(ModelModel.name == model_name).first()
    
    # Create model directory
    model_dir = MODEL_STORAGE_PATH
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, file.filename)

    # Create a copy of the file contents to use in background task
    file_contents = await file.read()
    await file.seek(0)  # Rewind for potential reuse
    
    # Start background task with the file contents
    background_tasks.add_task(
        save_file,
        file_contents=file_contents,
        upload_id=upload_id,
        model_path=model_path,
        model_name=model_name,
        group=group,
        db=db
    )
    
    return JSONResponse({
        "upload_id": upload_id,
        "message": "Model uploaded successfully",
        "filename": file.filename
    })

@router.get("/list-models")
async def list_models(db: Session = Depends(get_db)):
    """List all models from database"""
    try:
        models = db.query(ModelModel).filter(ModelModel.is_active == True).all()
        
        models_data = []
        for model in models:
            # Check if file still exists
            file_exists = os.path.exists(model.model_path) if model.model_path else False #type: ignore
            
            models_data.append({
                "id": model.id,
                "name": model.name,
                "group": model.group,
                "model_type": model.model_type,
                "model_path": model.model_path,
                "input_size": model.input_size,
                "class_names": model.class_names,
                "device": model.device,
                "is_active": model.is_active,
                "last_used": model.last_used.isoformat() if model.last_used is not None else None,
                "file_exists": file_exists,
                "file_size": os.path.getsize(model.model_path) if file_exists else 0  #type: ignore
            })
        
        return {"models": models_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/deactivate-model/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model from database and filesystem"""
    try:
        # Find model in database
        model = db.query(ModelModel).filter(ModelModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        setattr(model, "is_active", False)  # Mark as inactive
        db.commit()
        
        return {"message": f"Model {model.name} deactivated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-model/{model_id}")
async def update_model(
    model_id: int,
    group: Optional[str] = None,
    device: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update model metadata"""
    try:
        model = db.query(ModelModel).filter(ModelModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Update fields if provided
        if group is not None:
            model.group = group     #type: ignore
        if device is not None:
            model.device = device   #type: ignore
        
        db.commit()
        return {"message": f"Model {model.name} updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))