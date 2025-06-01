from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from core.dependencies import get_db
from sqlalchemy.orm import Session
from app.models import ModelModel
from app.model_upload import upload_model
import os
from typing import Optional
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "pretrained_models")

@router.post("/upload-model")
async def upload_model_endpoint(
    model_file: UploadFile = File(...),
    group: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    if not model_file.filename.endswith(".pt"):
        raise HTTPException(
            status_code=400,
            detail="Only .pt files are supported"
        )
    
    result = await upload_model(model_file, group, db)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    if result.get("message") == "Model already exists":
        raise HTTPException(status_code=400, detail="Model already exists")
    
    return result


@router.get("/list-models")
async def list_models(db: Session = Depends(get_db)):
    """List all models from database"""
    try:
        models = db.query(ModelModel).all()
        
        models_data = []
        for model in models:
            # Check if file still exists
            file_exists = os.path.exists(model.model_path) if model.model_path else False
            
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
async def deactivate_model(model_id: int, db: Session = Depends(get_db)):
    """Deactivate model"""
    try:
        # Find model in database
        model = db.query(ModelModel).filter(ModelModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not model.is_active:
            raise HTTPException(status_code=400, detail="Model already inactive")
        
        setattr(model, "is_active", False)  # Mark as inactive
        db.commit()
        
        return {"message": f"Model {model.name} deactivated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put('/activate-model/{model_id}')
async def activate_model(model_id: int, db: Session = Depends(get_db)):
    """Activate model"""
    try:
        # Find model in database
        model = db.query(ModelModel).filter(ModelModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if model.is_active:
            raise HTTPException(status_code=400, detail="Model already active")
        
        setattr(model, "is_active", True)  # Mark as active
        db.commit()
        
        return {"message": f"Model {model.name} activated successfully"}
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