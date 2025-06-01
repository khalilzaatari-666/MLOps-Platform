from datetime import datetime
import json
import logging
import math
from pathlib import Path
from typing import Optional
from fastapi import HTTPException
from ultralytics import YOLO
import os
from sqlalchemy.orm import Session
from app.models import DatasetModel, ModelModel
import shutil
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent

def register_existing_models(db: Session, models_dir: str = "pretrained_models"):
    model_files = Path(models_dir).glob("*.pt")
    
    for model_path in model_files:
        model_name = model_path.stem
        
        if db.query(ModelModel).filter(ModelModel.name == model_name).first():
            continue
            
        model = YOLO(str(model_path))

        class_names = list(getattr(model, 'names', {}).values())
        
        model_type = "detect"
        if "-seg" in model_name: model_type = "segment"
        if "-pose" in model_name: model_type = "pose"
        
        # Extract device info if needed
        device = str(model.device) if hasattr(model, 'device') else 'cpu'
        
        db_model = ModelModel(
            name=model_name,
            model_type=model_type,
            model_path=str(model_path),
            input_size=model.model.args.get("imgsz", 640), # type: ignore
            class_names=class_names,
            device=device,  # Add this line
            last_used=datetime.now()
        )
        db.add(db_model)
    
    db.commit()

def prepare_yolo_dataset_by_id(
    db: Session,
    dataset_id: int,
    split_ratios: Optional[dict] = None,
    random_state: int = 42,
    overwrite=True
) -> str:
    """
    Prepare YOLO dataset by splitting into train/val/test sets inside the dataset folder
    
    Args:
        db: Database session
        dataset_id: ID of the dataset in database
        split_ratios: Dict with 'train', 'val', 'test' ratios (must sum to 1)
        random_state: Random seed for reproducibility
        overwrite: Whether to overwrite existing splits
    
    Returns:
        Path to the generated data.yaml file
    """

    # Default split ratios if not provided
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    # Validate split ratios
    if not math.isclose(sum(split_ratios.values()), 1.0, rel_tol=1e-3):
        raise ValueError("Split ratios must sum to 1.0")
    
    if not all(k in split_ratios for k in ['train', 'val', 'test']):
        raise ValueError("Split ratios must contain 'train', 'val', and 'test' keys")

    # Get dataset from database
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset with ID {dataset_id} not found")
    
    logging.info(f"Preparing dataset {dataset_id}: {dataset.name}")
    
    # Construct dataset path from name
    # All paths are relative to PROJECT_ROOT
    dataset_path = PROJECT_ROOT / "datasets" / dataset.name / "augmented_data"
    if not os.path.exists(str(dataset_path)):
        raise ValueError(f"Dataset folder not found at: {dataset_path}")
    
    yolo_dir = PROJECT_ROOT / "datasets" / dataset.name / "yolo_splits"

    # Check if raw folder exists
    images_folder = os.path.join(str(dataset_path), "images")
    if not os.path.exists(images_folder):
        raise ValueError(f"images folder not found at: {images_folder}")

    # Create output directory inside the dataset folder
    if os.path.exists(yolo_dir):
        if overwrite:
            shutil.rmtree(yolo_dir)
        else:
            # Return existing yaml if splits already exist
            existing_yaml = os.path.join(yolo_dir, "data.yaml")
            if os.path.exists(existing_yaml):
                return existing_yaml
    
    os.makedirs(yolo_dir, exist_ok=True)

    # Create directory structure
    (yolo_dir / "train/images").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "train/labels").mkdir(parents=True, exist_ok=True)  
    (yolo_dir / "val/images").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "val/labels").mkdir(parents=True, exist_ok=True)    
    (yolo_dir / "test/images").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "test/labels").mkdir(parents=True, exist_ok=True) 

    # Get list of image files directly from images folder
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(images_folder).glob(f'*{ext}')))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_folder}")
    
    # Extract image stems for splitting
    image_stems = [f.stem for f in image_files]

    # Split dataset according to custom ratios
    val_test_ratio = split_ratios['val'] + split_ratios['test']
    train_stems, val_test_stems = train_test_split(image_stems, test_size=val_test_ratio, random_state=random_state)

    # Adjust test ratio within the val+test subset
    test_ratio = split_ratios['test'] / val_test_ratio if val_test_ratio > 0 else 0
    val_stems, test_stems = train_test_split(val_test_stems, test_size=test_ratio, random_state=random_state) if val_test_stems else ([], [])

    # Map of stems to their split destination
    stem_to_split = {}
    for stem in train_stems:
        stem_to_split[stem] = 'train'
    for stem in val_stems:
        stem_to_split[stem] = 'val'
    for stem in test_stems:
        stem_to_split[stem] = 'test'

    # Copy files from raw to appropriate splits
    labels_dir = os.path.join(str(dataset_path), "labels")
    has_labels = os.path.exists(labels_dir)
    
    for img_file in image_files:
        stem = img_file.stem
        split = stem_to_split.get(stem)
        if not split:
            continue  # Skip if not in any split (shouldn't happen)
        
        # Copy image
        dest_img = os.path.join(yolo_dir, split, 'images', img_file.name)
        shutil.copy(img_file, dest_img)
        
        # Copy label if it exists
        if has_labels:
            label_file = os.path.join(labels_dir, f"{stem}.txt")
            if os.path.exists(label_file):
                dest_label = os.path.join(yolo_dir, split, 'labels', f"{stem}.txt")
                shutil.copy(label_file, dest_label)

    # Create data.yaml file with the requested structure
    yaml_content = f"""
        # Train/val/test sets: specify directories, *.txt files, or lists
        train: train/images
        val: val/images
        test: test/images

        nc: 1

        names:
            0: 1
        """

    yaml_path = yolo_dir / "data.yaml"
    with open(str(yaml_path), 'w') as f:
        f.write(yaml_content)

    # Return path to the data.yaml file
    return str(yaml_path)


