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
import yaml
from sklearn.model_selection import train_test_split

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
            input_size=model.model.args.get("imgsz", 640),
            class_names=class_names,
            device=device,  # Add this line
            last_used=datetime.now()
        )
        db.add(db_model)
    
    db.commit()

def prepare_yolo_dataset_by_id(
    db: Session,
    dataset_id: int,
    split_ratios: dict = None,
    random_state: int = 42,
    overwrite: bool = False, 
    **kwargs
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
    dataset_path = os.path.join("datasets", dataset.name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset folder not found at: {dataset_path}")

    # Check if raw folder exists
    raw_folder = os.path.join(dataset_path, "raw")
    if not os.path.exists(raw_folder):
        raise ValueError(f"Raw folder not found at: {raw_folder}")

    # Create output directory inside the dataset folder
    output_dir = os.path.join(dataset_path, "yolo_splits")
    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            # Return existing yaml if splits already exist
            existing_yaml = os.path.join(output_dir, "data.yaml")
            if os.path.exists(existing_yaml):
                return existing_yaml
    
    os.makedirs(output_dir, exist_ok=True)

    # Create split directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        #os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

    # Get list of image files directly from raw folder
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(raw_folder).glob(f'*{ext}')))
    
    if not image_files:
        raise ValueError(f"No image files found in {raw_folder}")
    
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
    #labels_dir = os.path.join(dataset_path, "labels")
    #has_labels = os.path.exists(labels_dir)
    
    for img_file in image_files:
        stem = img_file.stem
        split = stem_to_split.get(stem)
        if not split:
            continue  # Skip if not in any split (shouldn't happen)
        
        # Copy image
        dest_img = os.path.join(output_dir, split, 'images', img_file.name)
        shutil.copy(img_file, dest_img)
        
        # Copy label if it exists
        #if has_labels:
        #    label_file = os.path.join(labels_dir, f"{stem}.txt")
        #    if os.path.exists(label_file):
        #        dest_label = os.path.join(output_dir, split, 'labels', f"{stem}.txt")
        #        shutil.copy(label_file, dest_label)

    # Create data.yaml file with the requested structure
    data = {
        'train': f"datasets/{dataset.name}/yolo_splits/train",
        'val': f"datasets/{dataset.name}/yolo_splits/val",
        'test': f"datasets/{dataset.name}/yolo_splits/test",
        'nc': 1,
        'names': {0: "1"}
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    # Push the yaml path to XCom for other tasks to use
    kwargs['ti'].xcom_push(key='yaml_path', value=yaml_path)

    # Return path to the data.yaml file
    return yaml_path


def train_yolo_model(dataset_id, params, param_idx, **kwargs):
    """
    Train YOLO model with specific parameters
    
    Args:
        dataset_id: ID of the dataset
        dataset_name: Name of the dataset
        params: Training parameters
        param_idx: Index of the parameter set
    
    Returns:
        Path to the trained model
    """
    ti = kwargs['ti']
    run_id = kwargs['run_id']
    
    # Get yaml path from XCom
    yaml_path = ti.xcom_pull(task_ids='prepare_dataset', key='yaml_path')
    
    # Create model directory
    model_dir = os.path.join("models", f"dataset_{dataset_id}", run_id, f"params_{param_idx}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Log parameters
    logging.info(f"Training model for dataset {dataset_id} with parameters: {params}")
    
    try:
        # Initialize model (yolov8n.pt is the default small model)
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=yaml_path,
            project=model_dir,
            name='train',
            **params
        )
        
        # Get the best model path
        best_model_path = os.path.join(model_dir, 'train', 'weights', 'best.pt')
        
        # Save metrics to a file for later comparison
        metrics = {
            'map50': float(results.maps50) if hasattr(results, 'maps50') else 0,
            'map75': float(results.maps75) if hasattr(results, 'maps75') else 0,
            'map50_95': float(results.maps50_95) if hasattr(results, 'maps50_95') else 0,
            'params': params
        }
        
        metrics_path = os.path.join(model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        # Push results to XCom
        ti.xcom_push(key=f'model_path_{param_idx}', value=best_model_path)
        ti.xcom_push(key=f'metrics_{param_idx}', value=metrics)
        
        return best_model_path
    
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def evaluate_and_select_best_model(dataset_id, **kwargs):
    """
    Evaluate all trained models and select the best one
    
    Args:
        dataset_id: ID of the dataset
        dataset_name: Name of the dataset
    
    Returns:
        Path to the best model
    """
    ti = kwargs['ti']
    run_id = kwargs['run_id']
    
    # Base directory for all models of this dataset
    models_dir = os.path.join("models", f"dataset_{dataset_id}", run_id)
    
    # Find all metrics files
    metrics_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file == 'metrics.json':
                metrics_files.append(os.path.join(root, file))
    
    # Load all metrics
    best_map50 = -1
    best_model_path = None
    best_params = None
    all_metrics = []
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
            
            # Check if this is the best model so far
            if metrics['map50'] > best_map50:
                best_map50 = metrics['map50']
                param_dir = os.path.dirname(metrics_file)
                best_model_path = os.path.join(param_dir, 'train', 'weights', 'best.pt')
                best_params = metrics['params']
    
    if best_model_path:
        # Copy the best model to a special location
        best_model_dest = os.path.join(models_dir, 'best_model.pt')
        shutil.copy(best_model_path, best_model_dest)
        
        # Save best model info
        best_model_info = {
            'map50': best_map50,
            'model_path': best_model_dest,
            'params': best_params
        }
        
        best_info_path = os.path.join(models_dir, 'best_model_info.json')
        with open(best_info_path, 'w') as f:
            json.dump(best_model_info, f)
        
        logging.info(f"Best model selected: {best_model_dest} with mAP50: {best_map50}")
        
        # Push to XCom
        ti.xcom_push(key='best_model_path', value=best_model_dest)
        ti.xcom_push(key='best_model_map50', value=best_map50)
        ti.xcom_push(key='best_model_params', value=best_params)
        
        return best_model_dest
    else:
        logging.error("No models were successfully trained")
        return None