import io
from pathlib import Path
from typing import List, Literal
import os
import zipfile
import requests
from app import models
from app.annotation_service import auto_annotate, process_validated_annotations
from app.database import SessionLocal
from fastapi import APIRouter, FastAPI, File, HTTPException, Depends, UploadFile
from fastapi.responses import StreamingResponse
from app import crud, schemas
from app.crud import API_USERS
from app.model_service import prepare_yolo_dataset_by_id, register_existing_models
from app.models import DatasetModel, ModelModel, UserModel
from app.schemas import ModelResponse
from sqlalchemy.orm import Session
from app.schemas import UserResponse

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

@app.get("/datasets/{dataset_id}/download/{folder_type}")
async def download_dataset_folder(
    dataset_id: str,
    folder_type: Literal["raw", "auto_annotated", "labels", "validated"],
    db: Session = Depends(get_db)
):
    """
    Download a specific folder from a dataset as a zip file.
    
    Args:
        dataset_id: The ID of the dataset
        folder_type: Type of folder to download (raw, auto_annotated, labels, validated)
        db: Database session
        
    Returns:
        A streaming response with the zipped folder content
    """
    # Check if dataset exists
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create the folder path based on the folder type
    dataset_path = os.path.join("datasets", dataset.name)
    folder_path = os.path.join(dataset_path, folder_type)
    
    # Check if the folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=404, 
            detail=f"The {folder_type} folder for dataset '{dataset.name}' does not exist"
        )
    
    # Check if the folder is empty
    if not os.listdir(folder_path):
        raise HTTPException(
            status_code=404,
            detail=f"The {folder_type} folder for dataset '{dataset.name}' is empty"
        )
    
    # Create a zip file in memory
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path to maintain directory structure in the zip
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname)
    
    # Seek to the beginning of the BytesIO object
    zip_io.seek(0)
    
    # Prepare the response with appropriate headers
    filename = f"{dataset.name}_{folder_type}.zip"
    
    return StreamingResponse(
        zip_io,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@app.post("/yolo-split/{dataset_id}")
def create_yolo_split(
    dataset_id: int, 
    db: Session = Depends(get_db)
):
    try:
        yaml_path = prepare_yolo_dataset_by_id(
            db=db,
            dataset_id=dataset_id,
            overwrite=True  # Set to True for testing
        )
        
        # Count files in each split to verify
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        dataset_path = os.path.join("datasets", dataset.name)
        output_dir = os.path.join(dataset_path, "yolo_splits")
        
        file_counts = {}
        for split in ['train', 'val', 'test']:
            img_count = len(list(Path(os.path.join(output_dir, split, 'images')).glob('*.*')))
            label_count = len(list(Path(os.path.join(output_dir, split, 'labels')).glob('*.txt')))
            file_counts[split] = {'images': img_count, 'labels': label_count}
            
        return {
            "status": "success",
            "yaml_path": yaml_path,
            "file_counts": file_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))