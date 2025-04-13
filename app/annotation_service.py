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

    print(f"Auto-annotation completed for dataset {dataset.name} using model {model.name}.")


async def process_validated_annotations(dataset_id: str, annotations_zip: UploadFile, db: Session) -> str:
    """
    This function processes manually validated annotations from an uploaded zip file
    and creates visualizations with the validated bounding boxes.
    
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
    raw_images_path = os.path.join(dataset_path, "raw")
    labels_path = os.path.join(dataset_path, "labels")
    
    # Create validated directory
    validated_path = os.path.join(dataset_path, "validated")
    if os.path.exists(validated_path):
        shutil.rmtree(validated_path)  # Remove existing directory
    os.makedirs(validated_path, exist_ok=True)
    
    # Create temporary directory for zip handling
    temp_dir = os.path.join(dataset_path, "temp_validated")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create temporary file for the uploaded zip
    temp_zip_path = os.path.join(temp_dir, "validated_annotations.zip")
    
    try:
        # Save the uploaded file to a temporary location
        with open(temp_zip_path, "wb") as buffer:
            content = await annotations_zip.read()
            buffer.write(content)
        
        # Extract zip file
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Get class mapping from original labels.txt
        class_map = {}
        labels_txt_path = os.path.join(labels_path, "labels.txt")
        if os.path.exists(labels_txt_path):
            with open(labels_txt_path, 'r') as f:
                classes = f.read().splitlines()
                for i, class_name in enumerate(classes):
                    class_map[i] = class_name
        else:
            print("Warning: No labels.txt found. Class names might be missing.")
        
        # Process each image in the raw dataset
        for image_file in os.listdir(raw_images_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Get corresponding annotation file name (without extension)
            base_name = os.path.splitext(image_file)[0]
            annotation_file = f"{base_name}.txt"
            annotation_path = os.path.join(temp_dir, annotation_file)
            
            # Skip if no annotation file exists for this image
            if not os.path.exists(annotation_path):
                print(f"No validated annotation found for {image_file}")
                continue
            
            # Load the image
            image_path = os.path.join(raw_images_path, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                continue
                
            height, width = img.shape[:2]
            
            # Get the image from the database
            image = db.query(ImageModel).filter(ImageModel.filename == image_file).first()
            if not image:
                print(f"Image {image_file} not found in database")
                continue
                
            # Clear existing validated bounding boxes for this image
            db.query(BoundingBox).filter(
                BoundingBox.image_id == image.id,
                BoundingBox.validation_status == "validated"
            ).delete()
            
            # Read and draw bounding boxes
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid annotation format in {annotation_file}: {line}")
                    continue
                    
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
                
                # Convert normalized YOLO format to pixel coordinates
                x1 = int((x_center - box_width/2) * width)
                y1 = int((y_center - box_height/2) * height)
                x2 = int((x_center + box_width/2) * width)
                y2 = int((y_center + box_height/2) * height)
                
                # Ensure coordinates are within image boundaries
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1))
                y2 = max(0, min(y2, height-1))
                
                # Get class name if available
                class_name = class_map.get(cls_id, f"class_{cls_id}")
                
                # Generate a consistent color for this class
                color = (
                    (cls_id * 123) % 255,
                    (cls_id * 85) % 255,
                    (cls_id * 37) % 255
                )
                
                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)
                
                # Add class label
                label = f"{class_name}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save to database
                db.add(BoundingBox(
                    image_id=image.id,
                    created_at=datetime.utcnow(),
                    class_id=cls_id,
                    class_name=class_name,
                    x_center=x_center,
                    y_center=y_center,
                    width=box_width,
                    height=box_height,
                    confidence=1.0,  # Manual validation implies high confidence
                    validation_status="validated"
                ))
            
            # Save the annotated image
            validated_image_path = os.path.join(validated_path, image_file)
            cv2.imwrite(validated_image_path, img)
            
        # Commit all database changes
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process validated annotations: {str(e)}")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print(f"Validated annotations processed for dataset {dataset.name}.")
    return validated_path