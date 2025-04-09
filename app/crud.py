from datetime import date, datetime
from fastapi import HTTPException
import json
from typing import Any, Dict
from sqlalchemy.orm import Session
import os
import requests
from app.models import DatasetModel, ImageModel, UserModel

API_IMAGES_INFO = "https://api.pcs-agri.com/images_info"
API_DOWNLOAD_IMAGE = "http://api.pcs-agri.com/download-image"
API_USERS = "http://api.pcs-agri.com/users"
DATASET_STORAGE_PATH = "./datasets"
API_AUTH_KEY="UVmQ2Y66zBeKGG3A_RAhHm8JJU3JvTqYIzcnUDl0DTQ"
HEADERS = {
    "Authorization": "Bearer {API}",
    "Content-Type": "application/json"
}

ALLOWED_MODELS = [
    "melon, pasteque, concombre, courgette, pg_cucurbit, artichaut",
    "tomate, aubergine, poivron",
    "poireau",
    "radis, choux de bruxelles",
    "haricot",
    "salad"
]
    

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
        # Return a valid response with default values
        return {
            "name": f"{model.capitalize()} Dataset from {start_date} to {end_date}",
            "start_date": start_date,  # Keep as string
            "end_date": end_date,      # Keep as string
            "model": model,
            "id": -1 , # Use a placeholder ID for empty datasets
            "created_at": created_at.isoformat()
        }

    created_at = datetime.now()
    dataset_name = f"{model.capitalize()} Dataset from {start_date} to {end_date}"  # Generate the name

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
            "created_at": existing_dataset.created_at.isoformat()
        }

    new_dataset = DatasetModel(
        name=dataset_name,
        start_date=start_date,
        end_date=end_date,
        model=model,
        created_at=created_at
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
            db.add(image)
            db.commit()
            db.refresh(image)
        new_dataset.images.append(image)

    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

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
        "created_at": new_dataset.created_at.isoformat()
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

    # Debug: Print the payload
    #print("Payload:", payload)

    # Make the API request using POST
    response = requests.post(API_IMAGES_INFO, json=payload, headers=headers, params=params)   # !!!!!

    if response.status_code == 200:
        response_data = response.json()
        images_list = response_data.get("NameImages", [])  # Ensure the key matches the API response
        return images_list  # Expected to return a list of image metadata
    else:
        print(f"Error fetching images: {response.status_code} {response.text}")
        return []

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
        }
        for dataset in datasets
    ]

def list_images(dataset_id: int, db: Session):
    """
    This function lists all images associated with a specific dataset.
    """
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}
    
    return dataset.images
