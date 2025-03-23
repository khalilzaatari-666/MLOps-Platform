import json
from typing import List

import requests
from app import models
from app.database import SessionLocal
from fastapi import FastAPI, HTTPException, Depends
from app import crud, schemas
from app.crud import API_USERS, create_dataset
from app.models import DatasetModel, ImageModel, UserModel
from sqlalchemy.orm import Session
from app.schemas import DatasetResponse, UserResponse, ImageResponse

app = FastAPI()

HEADERS = {
    "Authorization": "Bearer UVmQ2Y66zBeKGG3A_RAhHm8JJU3JvTqYIzcnUDl0DTQ",
    "Content-Type": "application/json"
}
API_USERS = "https://api.pcs-agri.com/users"

def get_db():
    db = SessionLocal()
    try:
        yield db
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

# Endpoint to list all users
@app.get("/users/", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    """
    Endpoint to list all users stored in the database.
    """
    return await list_users(db)

async def list_users(db: Session):
    """
    This function lists all users stored in the database.
    """
    users = db.query(UserModel).all()
    return [
        {
            "id": user.id,
            "full_name": user.full_name,
            "company_name": user.company_name,
        }
        for user in users
    ]

@app.get("/images/{dataset_id}", response_model=List[schemas.ImageResponse])
async def images(dataset_id: int, db: Session = Depends(get_db)):
    images = crud.list_images(dataset_id=dataset_id, db=db)  # List images for the provided dataset ID
    return images

@app.get("/datasets/", response_model=List[schemas.DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    datasets = crud.list_datasets(db=db)  # List all datasets from the database
    return datasets
