from datetime import datetime
from operator import and_
from pathlib import Path
import re
from fastapi import HTTPException
import json
import logging
from typing import Any, Dict, Literal
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import requests
from app.models import DatasetModel, ImageModel, UserModel, dataset_images, DeployedModel
from app.schemas import DatasetStatus
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

ALLOWED_MODELS = os.getenv("ALLOWED_MODELS")
API_AUTH_KEY = os.getenv("API_AUTH_KEY")
API_IMAGES_INFO = os.getenv("API_IMAGES_INFO")
API_DOWNLOAD_IMAGE = os.getenv("API_DOWNLOAD_IMAGE")
DATASET_STORAGE_PATH = os.getenv("DATASET_STORAGE_PATH", "./datasets")  
    

def create_dataset(db: Session, model: str, start_date: str, end_date: str, user_ids: list) -> Dict[str, Any]:
    """
    This function fetches images from the API, creates a dataset from them,
    associates them with the dataset, and then downloads the images to a local directory.
    """
    # Validate model before making the API request
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Choose from {', '.join(ALLOWED_MODELS)}.")

    images_info = fetch_images(db, model, start_date, end_date, user_ids)
    if not images_info:
        raise HTTPException(
            status_code=404,
            detail=f"No images found for model '{model}' and clients '{user_ids}' between {start_date} and {end_date}."
        )
    
    created_at = datetime.now()
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    
    # Remove special characters and spaces
    clean_model = re.sub(r'[^a-zA-Z0-9]', '_', model.strip().lower())

    if not images_info:
        # Return a valid response with default values
        return {
            "name": f"{clean_model}_dataset_{start}_{end}",
            "start_date": start_date,  # Keep as string
            "end_date": end_date,      # Keep as string
            "model": model,
            "id": -1, # Use a placeholder ID for empty datasets
            "created_at": created_at.isoformat(),
            "status": DatasetStatus.RAW
        }

    dataset_name = f"{clean_model}_dataset_{start}_{end}"  # Generate the name

    existing_dataset = db.query(DatasetModel).filter(DatasetModel.name == dataset_name).first()
    if existing_dataset:
        # Dataset with the same parameters already exists, return it
        print(f"Dataset already exists: {existing_dataset.name}")
        return {
            "name": existing_dataset.name,
            "start_date": existing_dataset.start_date,  # Convert to string
            "end_date": existing_dataset.end_date,      # Convert to string
            "model": existing_dataset.model,
            "id": existing_dataset.id,
            "created_at": existing_dataset.created_at.isoformat(),
            "status": existing_dataset.status
        }

    new_dataset = DatasetModel(
        name=dataset_name,
        start_date=start_date,
        end_date=end_date,
        model=model,
        created_at=created_at,
        status=DatasetStatus.RAW
    )

    # Associate users with dataset
    users = db.query(UserModel).filter(UserModel.id.in_(user_ids)).all() if user_ids else db.query(UserModel).all()
    new_dataset.users.extend(users)

    # Associate images with dataset
    for image_info in images_info:
        image_name = image_info.get("image_name")  # Ensure the key matches the API response
        if not image_name:
            print(f"Skipping invalid image info: {image_info}")
            continue

        image = db.query(ImageModel).filter(ImageModel.filename == image_name).first()
        if not image:
            image = ImageModel(filename=image_name)
            db.add(image)  # Add the image to the session immediately
            db.commit()    # Commit the session to persist the image
            db.refresh(image)  # Refresh to get the ID of the newly added image
        new_dataset.images.append(image)

    db.add(new_dataset)  # Add the new dataset to the session
    db.commit()  # Commit to persist the dataset
    db.refresh(new_dataset)  # Refresh to get the ID of the newly added dataset

    # Create dataset storage directory
    dataset_path = os.path.join(DATASET_STORAGE_PATH, dataset_name)
    raw_images_path = os.path.join(dataset_path, "raw")  # Create 'raw' subfolder
    os.makedirs(raw_images_path, exist_ok=True)

    # Debug: Print the dataset and raw folder paths
    print(f"Dataset folder created at: {dataset_path}")
    print(f"Raw images folder created at: {raw_images_path}")

    # Create metadata.json
    metadata = {
        "created_at": created_at.isoformat(),  # ISO format timestamp
        "user_ids": user_ids,                  # List of user IDs
        "image_count": len(new_dataset.images),  # Number of images in the dataset
        "models": model,                       # Model(s) used for the dataset
        "start_date": start_date,              # Start date of the dataset
        "end_date": end_date                   # End date of the dataset
    }

    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {metadata_path}")

    # Download images
    for image in new_dataset.images:
        image_name = image.filename
        print(f"Downloading image: {image_name}")  # Debug: Print the image name

        # Construct the image URL
        headers = {"Authorization": f"Bearer {API_AUTH_KEY}"}

        # Download the image
        response = requests.get(API_DOWNLOAD_IMAGE, params={"image_name": image_name}, headers=headers, stream=True)
        image_path = os.path.join(raw_images_path, image_name)
        if response.status_code == 200:
            with open(image_path, "wb") as f:  
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {image_name}")
        else:
            print(f"Failed to download {image_name}: {response.text}")

    return {
        "id": new_dataset.id,  # Use the auto-generated ID from the database
        "name": new_dataset.name,
        "start_date": new_dataset.start_date,  # Convert to string
        "end_date": new_dataset.end_date,      # Convert to string
        "model": new_dataset.model,
        "created_at": new_dataset.created_at.isoformat(),
        "status": new_dataset.status
    }


def fetch_images(db: Session, model: str, start_date: str, end_date: str, user_ids: list):
    """
    Fetches images from an external API based on the given parameters.
    """
    # Validate model before making the API request
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Choose from {', '.join(ALLOWED_MODELS)}.")
    
    # Fetch users from the database based on user_ids (if passed)
    if user_ids:
        users_list = db.query(UserModel).filter(UserModel.id.in_(user_ids)).all()  # Filter by passed user_ids
    else:
        users_list = db.query(UserModel).all()  # Fetch all users if no user_ids are passed

    # Raise an exception if no users are found
    if not users_list:
        raise HTTPException(status_code=404,detail="No users found in the database.")

    # Extract user_ids from the users list
    extracted_user_ids = [user.id for user in users_list]  # Assuming 'user_id' is the field name
    
    # If no user_ids are found, raise an exception
    if not extracted_user_ids:
        raise HTTPException(status_code=404, detail="Users not existing.")
    
    params = {
            "models": model               ### param
        }
    # Prepare the payload for the API request
    payload = {
        "startDate": start_date,  # Pass as string
        "endDate": end_date,      # Pass as string
        "clientId": user_ids        # Pass as list of user IDs   !!!!!!!!! clientId
    }
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {API_AUTH_KEY}"
    }

    response = requests.post(API_IMAGES_INFO, json=payload, headers=headers, params=params)   # !!!!!

    if response.status_code == 200:
        response_data = response.json()
        images_list = response_data.get("NameImages", [])  # Ensure the key matches the API response
        return images_list  # Expected to return a list of image metadata
    else:
        print(f"Error fetching images: {response.status_code} {response.text}")
        return []

def list_users(db:Session):
    return db.query(UserModel).all()

def list_datasets(db: Session):
    """
    This function lists all datasets stored in the database.
    """
    datasets = db.query(DatasetModel).all()
    return [
        {
            "id": dataset.id,
            "name": dataset.name,
            "start_date": dataset.start_date,  # datetime.date object
            "end_date": dataset.end_date,      # datetime.date object
            "model": dataset.model,
            "created_at": dataset.created_at.isoformat(),
            "users": [user.id for user in dataset.users],  # List of user IDs
            "images": [image.id for image in dataset.images],  # List of image IDs
            "status": dataset.status
        }
        for dataset in datasets
    ]

def get_dataset_by_status(db: Session, status: DatasetStatus): 
    """
    Get all datasets by status
    """
    datasets = db.query(DatasetModel).filter(DatasetModel.status == status).all()
    return datasets


def list_images(dataset_id: int, db: Session):
    """
    This function lists all images associated with a specific dataset.
    """
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}
    
    return dataset.images

ImageType = Literal['raw', 'auto_annotated', 'validated']

def get_dataset_image(
    db: Session,
    dataset_id: int,
    image_id: int,  # Now using database image ID
    image_type: ImageType,
):
    """
    Returns an image file with database validation
    
    Args:
        db: Database session
        dataset_id: ID of the dataset
        image_id: Database ID of the image
        image_type: One of ['raw', 'auto_annotated', 'validated']
        user_id: Optional user ID for access validation
    
    Returns:
        FileResponse with the image file
    
    Raises:
        HTTPException 404 if image/dataset not found
        HTTPException 403 if user doesn't have access
    """
    try:
        # Verify dataset exists
        dataset = db.query(DatasetModel).filter_by(id=dataset_id).first()
        if not dataset:
            logger.error(f"Dataset not found: {dataset_id}")
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get image metadata from database
        image = db.query(ImageModel).filter_by(id=image_id).first()
        if not image:
            logger.error(f"Image not found in DB: {image_id}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Verify image exists and belongs to dataset using the association table
        association_exists = db.query(dataset_images).filter(
            and_(
                dataset_images.c.dataset_id == dataset_id,
                dataset_images.c.image_id == image_id
            )
        ).first()
        
        if not association_exists:
            raise HTTPException(
                status_code=404,
                detail="Image not found in specified dataset"
            )
        
        # Construct the exact path: datasets/dataset.name/{image_type}/filename
        image_path = Path("datasets") / dataset.name / image_type / image.filename
        logger.info(f"Looking for image at: {image_path}")
        
        # Check if file exists
        if not image_path.exists():
            logger.error(f"Image file not found at: {image_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Image file not found at: {image_path}"
            )
        
        # Determine media type from extension
        ext = image_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        
        return FileResponse(
            image_path,
            media_type=media_types.get(ext, 'image/jpeg'),
            filename=image_path.name
        )

    except Exception as e:
        logger.error(f"Error in get_dataset_image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
def list_deployed_models(db: Session):
    """
    List all deployed models from the database.
    """
    deployed_models = db.query(DeployedModel).all()
    return [
        {
            "id": model.id,
            "dataset_id": model.dataset_id,
            "model_id": model.model_id,
            "path": model.minio_path,
            "deployment_date": model.deployment_date.isoformat(),
            "score": model.score,
            "status": model.status,
        }
        for model in deployed_models
    ]

def get_deployed_model(db: Session, model_id: int):
    """
    Get a specific deployed model by its ID.
    """
    deployed_model = db.query(DeployedModel).filter(DeployedModel.id == model_id).first()
    if not deployed_model:
        raise HTTPException(status_code=404, detail="Deployed model not found")
    
    return {
        "id": deployed_model.id,
        "dataset_id": deployed_model.dataset_id,
        "model_id": deployed_model.model_id,
        "path": deployed_model.minio_path,
        "deployment_date": deployed_model.deployment_date.isoformat(),
        "score": deployed_model.score,
        "status": deployed_model.status,
    }