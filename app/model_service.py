from datetime import datetime
import math
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
import os
from sqlalchemy.orm import Session
from app.models import DatasetModel, ModelModel
import shutil
import yaml
from sklearn.model_selection import train_test_split

def register_existing_models(db: Session, models_dir: str = "models"):
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

def list_models(db: Session):
    """
    List all registered YOLO models
    """
    models = db.query(ModelModel).all()
    return models

def prepare_yolo_dataset_by_id(
    db: Session,
    dataset_id: int,
    base_datasets_dir: str = "datasets",
    split_ratios: dict = None,
    random_state: int = 42,
    class_names: Optional[list] = None,
    overwrite: bool = False
) -> str:
    """
    Prepare YOLO dataset by splitting into train/val/test sets inside the dataset folder
    
    Args:
        db: Database session
        dataset_id: ID of the dataset in database
        split_ratios: Dict with 'train', 'val', 'test' ratios (must sum to 1)
        random_state: Random seed for reproducibility
        class_names: Optional list of class names (if not stored in DB)
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
    
    # Construct dataset path from name
    dataset_path = os.path.join(base_datasets_dir, dataset.name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset folder not found at: {dataset_path}")

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

    # Create split directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

    # Get list of image files
    image_files = list(Path(os.path.join(dataset_path, 'raw')).glob('*.*'))
    image_stems = [f.stem for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    # Split dataset according to custom ratios
    val_test_ratio = split_ratios['val'] + split_ratios['test']
    train, val_test = train_test_split(image_stems, test_size=val_test_ratio, random_state=random_state)

    # Adjust test ratio within the val+test subset
    test_ratio = split_ratios['test'] / val_test_ratio
    val, test = train_test_split(val_test, test_size=test_ratio, random_state=random_state)

    # Function to copy files
    def copy_files(filenames, split):
        for stem in filenames:
            img_ext = next((e for e in ['.jpg', '.jpeg', '.png'] if os.path.exists(os.path.join(dataset_path, 'images', f"{stem}{e}"))), None)
            if img_ext:
                shutil.copy(
                    os.path.join(dataset_path, 'images', f"{stem}{img_ext}"),
                    os.path.join(output_dir, split, 'images', f"{stem}{img_ext}")
                )
                label_file = os.path.join(dataset_path, 'labels', f"{stem}.txt")
                if os.path.exists(label_file):
                    shutil.copy(
                        label_file,
                        os.path.join(output_dir, split, 'labels', f"{stem}.txt"))
    
    # Copy files to respective splits
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')

    # Get class names
    if class_names is None:
        class_names = dataset.class_names or ["object"]

    # Create data.yaml file
    data = {
        'train': f"yolo_splits/train",
        'val': f"yolo_splits/val",
        'test': f"yolo_splits/test",
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    # Return path to the data.yaml file
    return yaml_path

