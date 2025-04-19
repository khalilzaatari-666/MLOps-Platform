from datetime import datetime
import os
import shutil
import tempfile
from typing import OrderedDict
import zipfile
import cv2
from fastapi import HTTPException, UploadFile
from requests import Session
from ultralytics import YOLO
from app.models import BoundingBox, DatasetModel, ImageModel, ModelModel
from app.schemas import DatasetStatus

def auto_annotate(dataset_id : str, model_id : str, db: Session) -> str:
    """
    This function performs auto-annotation on the dataset using the specified YOLO model.

    Args:
        dataset_id: The ID of the dataset
        model_id: The ID of the used model
        db: Database session
    
    """
    # Fetch the dataset and model from the database
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    model = db.query(ModelModel).filter(ModelModel.id == model_id).first()

    if not dataset or not model:
        raise HTTPException(status_code=404, detail="Dataset or Model not found")
    
    #Get dataset path
    dataset_path = os.path.join("datasets" , dataset.name)
    raw_images = os.path.join(dataset_path , "raw")

    # Load the YOLO model
    try:
        yolo_model = YOLO(model.model_path)
        model.last_used = datetime.utcnow()  # Update last used timestamp
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

    # Track unique classes
    unique_classes = set()

    #Create save directory "auto_annotated"
    images_output = os.path.join(dataset_path, "auto_annotated")
    labels_output = os.path.join(dataset_path, "labels")

    # Remove existing directories if they exist
    for folder in [images_output, labels_output]:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # Recursively delete folder and contents
    
    # Create fresh directories
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    if not os.path.exists(images_output):
        os.makedirs(images_output)

    if not os.path.exists(labels_output):
        os.makedirs(labels_output)

    # Perform auto-annotation
    for image_file in os.listdir(raw_images):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(raw_images, image_file)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        
        image = db.query(ImageModel).filter(ImageModel.filename == image_file).first()
        
        # Perform inference
        results = yolo_model.predict(image_path)

        # Save visualization
        vis_path = os.path.join(images_output, image_file)
        results[0].save(filename=vis_path)

        # Save labels and track classes
        label_path = os.path.join(labels_output, os.path.splitext(image_file)[0] + '.txt')
        with open(label_path, 'w') as f:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = yolo_model.names[cls_id]
                unique_classes.add(cls_name)
                
                # Write YOLO-format label (class_id x_center y_center width height)
                xywhn = box.xywhn[0].tolist()
                f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in xywhn)}\n")

                db.add(BoundingBox(
                    model_id=model_id,
                    image_id=image.id,
                    created_at=datetime.utcnow(),
                    class_id=int(box.cls),
                    class_name=yolo_model.names[int(box.cls)],
                    x_center=float(box.xywhn[0][0]),
                    y_center=float(box.xywhn[0][1]),
                    width=float(box.xywhn[0][2]),
                    height=float(box.xywhn[0][3]),
                    confidence=float(box.conf),
                    validation_status="auto"
                ))

                db.commit()

        # Save class names (sorted by class ID)
        detected_classes = sorted(list(unique_classes))
        class_names_path = os.path.join(labels_output, "labels.txt")
        with open(class_names_path, 'w') as f:
            f.write("\n".join(detected_classes))
        
        dataset.status = DatasetStatus.AUTO_ANNOTATED
        db.commit()

    print(f"Auto-annotation completed for dataset {dataset.name} using model {model.name}.")


async def process_validated_annotations(dataset_id: str, annotations_zip: UploadFile, db: Session) -> str:
    """
    This function extracted the validated annotations and stores them in the labels folder of the dataset
    
    Args:
        dataset_id: The ID of the dataset
        annotations_zip: Uploaded zip file containing validated YOLO annotations
        db: Database session
        
    Returns:
        Path to the validated annotations folder
    """
    # Fetch the dataset from the database
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get dataset paths
    dataset_path = os.path.join("datasets", dataset.name)
    labels_path = os.path.join(dataset_path, "labels")

    # Ensure dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset.name} does not exist.")

    # Remove old labels folder if it exists
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)

    os.makedirs(labels_path, exist_ok=True)

    # Save the uploaded zip temporarily
    temp_zip_path = os.path.join(dataset_path, "temp_annotations.zip")
    with open(temp_zip_path, "wb") as f:
        content = await annotations_zip.read()
        f.write(content)

    # Extract ZIP into labels folder
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(labels_path)

    # Clean up temporary zip file
    os.remove(temp_zip_path)

    dataset.status = DatasetStatus.VALIDATED    
    db.commit()

    return f"Annotations successfully extracted to {labels_path}"