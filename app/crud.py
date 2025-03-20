from datetime import date, datetime
from http.client import HTTPException
from sqlalchemy.orm import Session
import os
import requests
from app.database import SessionLocal
from app.models import DatasetModel, ImageModel, UserModel
from app.schemas import DatasetBase, ImageBase, ImageResponse, UserBase, UserResponse  # Assuming models are defined in models.py

API_IMAGES_INFO = "https://api.example.com/images_info"
API_DOWNLOAD_IMAGE = "http://api.pcs-agri.com/download_image"
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
    

def create_dataset(db: Session, model: str, start_date: str, end_date: str, user_ids: list):
    """
    This function fetches images from the API, creates a dataset from them,
    associates them with the dataset, and then downloads the images to a local directory.
    """
    # Validate model before making the API request
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Choose from {', '.join(ALLOWED_MODELS)}.")

    images_info = fetch_images(db, model, start_date, end_date, user_ids)
    print("Fetched Images Info:", images_info)  # Debug: Print fetched images

    if not images_info:
        # Return a valid response with default values
        return {
            "name": f"{model.capitalize()} Dataset from {start_date} to {end_date}",
            "start_date": start_date,  # Keep as string
            "end_date": end_date,      # Keep as string
            "model": model,
            "id": -1  # Use a placeholder ID for empty datasets
        }

    created_at = datetime.now()
    dataset_name = f"{model.capitalize()} Dataset from {start_date} to {end_date}"  # Generate the name

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

    # Download images
    for image in new_dataset.images:
        image_name = image.filename
        print(f"Downloading image: {image_name}")  # Debug: Print the image name

        # Construct the image URL
        image_url = f"{API_DOWNLOAD_IMAGE}?image_name={image_name}"
        headers = {"Authorization": f"Bearer {API_AUTH_KEY}"}

        # Download the image
        response = requests.get(image_url, headers=headers, stream=True)

        if response.status_code == 200:
            # Save the image to the raw folder
            image_path = os.path.join(raw_images_path, image_name)
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {image_name} to {image_path}")  # Debug: Print download success
        else:
            print(f"Failed to download {image_name}: {response.text}")  # Debug: Print download failure

    return {
        "name": dataset_name,
        "start_date": start_date,  # Keep as string
        "end_date": end_date,      # Keep as string
        "model": model,
        "id": new_dataset.id  # Use the auto-generated ID from the database
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

    # Extract user_ids from the users list
    extracted_user_ids = [user.id for user in users_list]  # Assuming 'user_id' is the field name
    
    # If no user_ids are found, raise an exception
    if not extracted_user_ids:
        raise HTTPException(status_code=404, detail="No users found in the database.")
    
    # Prepare the payload for the API request
    payload = {
        "models": model,
        "startDate": start_date,  # Pass as string
        "endDate": end_date,      # Pass as string
        "users": extracted_user_ids  # Pass as list of user IDs
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_AUTH_KEY}"
    }

    # Debug: Print the payload
    print("Payload:", payload)

    # Make the API request using POST
    response = requests.post(API_IMAGES_INFO, json=payload, headers=headers)
    
    # Debug: Print the response status code and data
    print("Response Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body:", response.text)

    if response.status_code == 200:
        response_data = response.json()
        images_list = response_data.get("NameImages", [])  # Ensure the key matches the API response
        return images_list  # Expected to return a list of image metadata
    else:
        print(f"Error fetching images: {response.status_code} {response.text}")
        return []


def list_users(db: Session):
    """
    This function lists all users stored in the database.
    """
    users = db.query(UserModel).all()
    return users

def list_datasets(db: Session):
    """
    This function lists all datasets stored in the database.
    """
    datasets = db.query(DatasetModel).all()
    return datasets

def list_images(dataset_id: int, db: Session):
    """
    This function lists all images associated with a specific dataset.
    """
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}
    
    return dataset.images