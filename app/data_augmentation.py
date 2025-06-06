import glob
import random
from tqdm import tqdm
from app.models import DatasetModel
from app.schemas import TransformerType
from app.schemas import DatasetStatus
from core.dependencies import get_db
import albumentations as A
import string
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
import os
import cv2
from typing import List
import numpy as np
import random
import glob

def get_transformer_configs():
    """
    Define all available transformers with fixed parameters
    """
    return {
        TransformerType.VERTICAL_FLIP: A.VerticalFlip(p=1),
        TransformerType.HORIZONTAL_FLIP: A.HorizontalFlip(p=1),
        TransformerType.TRANSPOSE: A.Transpose(p=1.0),
        TransformerType.CENTER_CROP: A.CenterCrop(p=1.0, height=800, width=800)
    }


def save_augmented_image_bbox(path_image: str, path_bbox: str, augmented_boxes: List, augmented_image: np.ndarray, img_name: str, transformer_name: str):
    """Save augmented image and bbox annotations with augmented prefix"""
    # Create directories if they don't exist
    os.makedirs(path_image, exist_ok=True)
    os.makedirs(path_bbox, exist_ok=True)
    
    # Generate cipher for unique naming
    CIPHER_LENGTH = 10
    cipher = ''.join(random.choices(string.ascii_letters + string.digits, k=CIPHER_LENGTH))
    
    # Convert image format and save with "augmented_" prefix
    image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    output_image_path = f"{path_image}/augmented_{transformer_name}_{img_name}_{cipher}.jpg"
    cv2.imwrite(output_image_path, image)
    
    # Save bbox annotations with "augmented_" prefix
    output_bbox_path = f"{path_bbox}/augmented_{transformer_name}_{img_name}_{cipher}.txt"
    with open(output_bbox_path, "w") as file:
        for bbox in augmented_boxes:
            class_label = int(0.0)  # Default class label
            bbox_str = ' '.join([str(class_label)] + [f'{coord:.6f}' for coord in bbox])
            file.write(bbox_str + "\n")
    
    print(f'saved: {output_image_path}')
    return output_image_path

def read_bbox(path: str) -> np.ndarray:
    """Read bounding box annotations from file"""
    if not os.path.exists(path):
        return np.array([])
    
    with open(path, "r") as file:
        annotations = file.readlines()
    
    if not annotations:
        return np.array([])
    
    bboxes = np.array([list(map(float, annotation.strip().split())) for annotation in annotations])
    
    # Fix floating point precision issues by clipping bbox coordinates to valid range [0.0, 1.0]
    if len(bboxes) > 0:
        # For YOLO format: [class, x_center, y_center, width, height]
        # Coordinates should be in range [0.0, 1.0]
        bboxes[:, 1:] = np.clip(bboxes[:, 1:], 0.0, 1.0)
        
        # Additional validation: ensure width and height don't exceed 1.0
        # x_center + width/2 should not exceed 1.0, same for y_center + height/2
        for i in range(len(bboxes)):
            x_center, y_center, width, height = bboxes[i, 1:5]
            
            # Ensure bbox doesn't go outside image boundaries
            if x_center + width/2 > 1.0:
                width = 2 * (1.0 - x_center)
            if y_center + height/2 > 1.0:
                height = 2 * (1.0 - y_center)
            if x_center - width/2 < 0.0:
                width = 2 * x_center
            if y_center - height/2 < 0.0:
                height = 2 * y_center
                
            bboxes[i, 3] = width
            bboxes[i, 4] = height
    
    return bboxes

def apply_augmentation(dataset_id: str, transformer_types: List[TransformerType], db: Session = Depends(get_db)):
    """Apply single data augmentation transformer to a dataset"""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Define paths
    base_path = f"datasets/{dataset.name}"
    images_path = f"{base_path}/raw"
    labels_path = f"{base_path}/labels"
    output_images_path = f"{base_path}/augmented_data/images"
    output_labels_path = f"{base_path}/augmented_data/labels"
    
    # Validate dataset exists
    if not os.path.exists(images_path):
        raise HTTPException(status_code=404, detail=f"Dataset images path not found: {images_path}")
    
    if not os.path.exists(labels_path):
        raise HTTPException(status_code=404, detail=f"Dataset labels path not found: {labels_path}")
    
    # Create output directories
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
        image_files.extend(glob.glob(os.path.join(images_path, ext.upper())))
    
    if not image_files:
        raise HTTPException(status_code=404, detail=f"No images found in {images_path}")
    
    # Get corresponding label files
    label_files = []
    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_path, f"{img_name}.txt")
        label_files.append(label_path)
    
    print("Copying original images and labels...")
    
    # First, copy all original images and labels to augmented_data folder
    for img_path, label_path in tqdm(zip(image_files, label_files), desc="Copying originals"):
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Copy original image
        original_img_dest = os.path.join(output_images_path, f"original_{img_name}")
        import shutil
        shutil.copy2(img_path, original_img_dest)
        
        # Copy original label (if exists)
        if os.path.exists(label_path):
            original_label_dest = os.path.join(output_labels_path, f"original_{label_name}")
            shutil.copy2(label_path, original_label_dest)
    
    # Get transformer configuration
    transformer_configs = get_transformer_configs()
    for transformer_type in transformer_types:
        if transformer_type not in transformer_configs:
            raise HTTPException(status_code=400, detail=f"Transformer type {transformer_type.value} not found")
    
    print(f"Applying {len(transformer_types)} transformation(s): {[t.value for t in transformer_types]}")
    
    total_processed = 0
    image_index = 1
    
    # Process each image for augmentation
    for img_path, label_path in tqdm(zip(image_files, label_files), desc="Processing augmentations"):
        
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'image ==========================================> {image_index}')
        print(f'path_image {img_name}')
        image_index += 1
        
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read bounding boxes
            bboxes = read_bbox(label_path)
            for transformer_type in transformer_types:
                transformer = transformer_configs[transformer_type]
                print(f"  Applying {transformer_type.value} to {img_name}")
            
                if len(bboxes) == 0:
                    # No annotations, apply transformation to image only
                    transform = A.Compose([transformer])
                    transformed = transform(image=image)
                    augmented_image = transformed['image']
                    augmented_boxes = []
                else:
                    # Apply transformation with bounding boxes
                    boxes = bboxes[:, 1:]        # Only boxes
                    class_labels = bboxes[:, 0]  # Only labels
                    
                    transform = A.Compose(
                        [transformer], 
                        bbox_params=A.BboxParams(
                            format='yolo',
                            min_visibility=0.6,
                            label_fields=['class_labels']
                        )
                    )
                    
                    transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)
                    augmented_image = transformed['image']
                    augmented_boxes = transformed['bboxes']
                
                # Save augmented image and labels with "augmented_" prefix
                save_augmented_image_bbox(
                    path_image=output_images_path,
                    path_bbox=output_labels_path,
                    augmented_boxes=augmented_boxes,
                    augmented_image=augmented_image,
                    img_name=img_name,
                    transformer_name=transformer_type.value
                )
                
                total_processed += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    dataset.count = len(image_files) + total_processed
    print(f"Copying completed. Original images: {len(image_files)}")
    print(f"Augmentation completed. Augmented images: {total_processed}")
    print(f"Total images in augmented_data: {len(image_files) + total_processed}")
    
    return 'Congratulations, your data is augmented successfully'
