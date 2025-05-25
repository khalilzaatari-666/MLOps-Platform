import io
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import os
import zipfile
from app.models import DatasetModel
from sqlalchemy.orm import Session
from typing import Literal
from core.dependencies import get_db
from app.crud import get_dataset_image


router = APIRouter()

# Endpoint to download locally stored files
@router.get("/datasets/{dataset_id}/download/{folder_type}")
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
    dataset_path = os.path.join("datasets", str(dataset.name))
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
    
    # Prepare the response with routerropriate headers
    filename = f"{dataset.name}_{folder_type}.zip"
    
    return StreamingResponse(
        zip_io,
        media_type="routerlication/zip",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

# Endpoint to download a specific image from a dataset
@router.get("/datasets/{dataset_id}/{image_type}/{image_id}")
async def get_image(
    dataset_id: int,
    image_id: int,
    image_type: Literal["raw", "auto_annotated", "validated"] = "raw",
    db: Session = Depends(get_db),
):
    """
    Get an image from a dataset with database validation
    
    Parameters:
    - dataset_id: ID of the dataset
    - image_id: Database ID of the image
    - image_type: Type of image (raw, auto_annotated, validated)
    """
    return get_dataset_image(
        db=db,
        dataset_id=dataset_id,
        image_id=image_id,
        image_type=image_type,
    )