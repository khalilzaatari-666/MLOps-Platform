from typing import List
import os
import uuid
import requests
from app import models
from app.annotation_service import auto_annotate, process_validated_annotations
from app.database import SessionLocal
from fastapi import FastAPI, File, HTTPException, Depends, UploadFile
from app import crud, schemas
from app.crud import API_USERS
from app.model_service import register_existing_models, list_models
from app.models import ModelModel, UserModel
from app.schemas import ModelResponse
from sqlalchemy.orm import Session
from app.schemas import (
    UserResponse
)
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

HEADERS = {
    "Authorization": "Bearer UVmQ2Y66zBeKGG3A_RAhHm8JJU3JvTqYIzcnUDl0DTQ",
    "Content-Type": "application/json"
}
API_USERS = "https://api.pcs-agri.com/users"
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        register_existing_models(db=db)
    finally:
        db.close()

@app.post("/datasets/", response_model=schemas.DatasetResponse)
async def create_dataset(
    dataset_request: schemas.CreateDatasetRequest, 
    db: Session = Depends(get_db)
):
    dataset = crud.create_dataset(
        db=db,
        model=dataset_request.model,
        start_date=dataset_request.start_date,
        end_date=dataset_request.end_date,
        user_ids=dataset_request.user_ids
    )
    return dataset

@app.post("/list_users/", response_model=List[schemas.UserResponse])
async def list_users(db: Session = Depends(get_db)):
    # Make the API request to fetch users
    try:
        response = requests.post(API_USERS, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response JSON
        response_data = response.json()
        users_list = response_data.get("All users", [])
        
        # Add the fetched users to the database if they don't already exist
        for user_data in users_list:
            existing_user = db.query(models.UserModel).filter(
                models.UserModel.id == user_data['id']
            ).first()
            
            if not existing_user:
                new_user = models.UserModel(
                    id=user_data['id'],
                    full_name=user_data['full_name'],
                    company_name=user_data['company_name']
                )
                db.add(new_user)
        
        db.commit()
        users_from_db = crud.list_users(db=db)
        return users_from_db
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users from external API: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/users/", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    """
    Endpoint to list all users stored in the database.
    """
    try:
        users = db.query(UserModel).all()
        return [
            {
                "id": user.id,
                "full_name": user.full_name,
                "company_name": user.company_name,
            }
            for user in users
        ]
    except Exception as e:
        # Log the error
        print(f"Error in get_users: {str(e)}")
        # Re-raise or return a custom error
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/images/{dataset_id}", response_model=List[schemas.ImageResponse])
async def images(dataset_id: int, db: Session = Depends(get_db)):
    images = crud.list_images(dataset_id=dataset_id, db=db)  # List images for the provided dataset ID
    return images

@app.get("/datasets/", response_model=List[schemas.DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    datasets = crud.list_datasets(db=db)  # List all datasets from the database
    return datasets

@app.get("/models/", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    models = db.query(ModelModel).all()
    
    # Ensure each model's class_names is properly formatted
    for model in models:
        if isinstance(model.class_names, dict):
            model.class_names = list(model.class_names.values())
    
    return models

@app.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(ModelModel).filter(ModelModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Ensure class_names is properly formatted
    if isinstance(model.class_names, dict):
        model.class_names = list(model.class_names.values())
    
    return model

@app.get("/annotate/{dataset_id}/{model_id}")
def annotate_images(
    dataset_id: int, 
    model_id: int, 
    db: Session = Depends(get_db)
):
    auto_annotate(dataset_id=dataset_id, model_id=model_id, db=db)
    return {"message": "Auto-annotation completed successfully."}

@app.post("/datasets/{dataset_id}/process-validated-annotations")
async def process_annotations_endpoint(
    dataset_id: str, 
    annotations_zip: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    validated_path = await process_validated_annotations(dataset_id, annotations_zip, db)
    return {"message": "Validated annotations processed successfully", "path": validated_path}