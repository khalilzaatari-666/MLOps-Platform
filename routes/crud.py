from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import requests
from app import crud, models, schemas
from core.dependencies import get_db
from app.schemas import DatasetStatus, UserResponse, ModelResponse, DeployedModelResponse, DatasetResponse
from app.models import UserModel, ModelModel
from dotenv import load_dotenv
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()
API_USERS = os.getenv("API_USERS")
API_AUTH_KEY = os.getenv("API_AUTH_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_AUTH_KEY}",
    "Content-Type": "application/json"
}

@router.post("/datasets", response_model=schemas.DatasetResponse)
async def create_dataset(
    dataset_request: schemas.CreateDatasetRequest, 
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Creating dataset with model={dataset_request.model}, "
                    f"start_date={dataset_request.start_date}, "
                    f"end_date={dataset_request.end_date}, "
                    f"user_ids={dataset_request.user_ids}")

        dataset = crud.create_dataset(
            db=db,
            model=dataset_request.model,
            start_date=dataset_request.start_date,
            end_date=dataset_request.end_date,
            user_ids=dataset_request.user_ids
        )

        logger.info(f"Dataset created successfully: {dataset['name']} (ID: {dataset['id']})")
        return dataset

    except HTTPException as http_exc:
        logger.error(f"HTTP error: {http_exc.detail}")
        raise http_exc

    except Exception as exc:
        logger.exception("Unexpected error occurred while creating dataset.")
        raise HTTPException(status_code=500, detail="Internal server error while creating dataset.")


@router.get("/datasets/{status}", response_model=List[schemas.DatasetResponse])
async def get_dataset_by_status(
    status: str,
    db: Session = Depends(get_db)
):
    if status.lower() not in ["raw", "auto_annotated", "validated"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    dataset = crud.get_dataset_by_status(
        db=db,
        status=DatasetStatus(status)
    )
    return dataset
    
@router.get("/users", response_model=List[UserResponse])
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

@router.get("/images/{dataset_id}", response_model=List[schemas.ImageResponse])
async def images(dataset_id: int, db: Session = Depends(get_db)):
    images = crud.list_images(dataset_id=dataset_id, db=db)  # List images for the provided dataset ID
    return images

@router.get("/datasets", response_model=List[schemas.DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    datasets = crud.list_datasets(db=db)  # List all datasets from the database
    return datasets

@router.get("/models", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    models = db.query(ModelModel).all()
    
    # Ensure each model's class_names is properly formatted
    formatted_models = []
    for model in models:
        if isinstance(model.class_names, dict):
            model_dict = {
                'id': model.id,
                'name': model.name,
                'class_names': list(model.class_names.values()),
                'created_at': model.created_at
            }
            formatted_models.append(model_dict)
        else:
            formatted_models.append(model)
    
    return formatted_models

@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(ModelModel).filter(ModelModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create a new dictionary with formatted data
    model_data = {
        'id': model.id,
        'name': model.name,
        'class_names': list(model.class_names.values()) if isinstance(model.class_names, dict) else model.class_names,
        'created_at': model.created_at
    }
    
    return model_data

@router.get("/deployed_models", response_model=List[DeployedModelResponse])
def list_deployed_models(db: Session = Depends(get_db)):
    deployed_models = crud.list_deployed_models(db=db)
    return deployed_models

@router.get("/deployed_models/{model_id}", response_model=DeployedModelResponse)
def get_deployed_model(model_id: int, db: Session = Depends(get_db)):
    deployed_model = crud.get_deployed_model(model_id=model_id, db=db)
    if not deployed_model:
        raise HTTPException(status_code=404, detail="Deployed model not found")
    return deployed_model