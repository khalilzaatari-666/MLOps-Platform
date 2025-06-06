from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from celery import Celery
import uuid
import pymysql
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO
from app.database import SessionLocal
from app.model_service import prepare_yolo_dataset_by_id
from app.models import DatasetModel, TestTask, TrainingInstance, TrainingTask
from app.schemas import (TestingStatus, TrainingStatus)
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
CELERY_BROKER = os.getenv("CELERY_BROKER_URL")
CELERY_BACKEND = os.getenv("CELERY_BACKEND_URL")

app = Celery('tasks',
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND)

METRIC_MAPPING = {
    'accuracy': 'metrics/mAP50(B)',
    'precision': 'metrics/precision(B)',
    'recall': 'metrics/recall(B)'
}

PROJECT_ROOT = Path(__file__).parent.parent

def compute_params_hash(params: Dict[str, Any]) -> str:
    import hashlib
    import json
    from datetime import datetime
    
    # Add timestamp to ensure uniqueness
    unique_params = params.copy()
    unique_params["_unique"] = datetime.utcnow().isoformat()  # Or use uuid.uuid4()
    
    return hashlib.sha256(
        json.dumps(unique_params, sort_keys=True).encode()
    ).hexdigest()

# Database utility functions
def get_training_task(db: Session, task_id: str):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise ValueError(f"Training task {task_id} not found")
    return task

def create_training_task(db: Session, dataset_id: int, params: Dict[str, Any], training_instance_id: int ,queue_position: Optional[int] = None) -> TrainingTask:
    
    task_id = str(uuid.uuid4())
    
    # Determine initial status based on queue position
    status = TrainingStatus.PENDING
    if queue_position and queue_position > 0:
        status = TrainingStatus.QUEUED
    
    task = TrainingTask(
        id=task_id,
        dataset_id=dataset_id,
        status=status,
        params=params,
        params_hash=compute_params_hash(params),
        queue_position=queue_position,
        start_date=datetime.utcnow()
    )

    #Fetch the instance and establish relationship
    instance = db.query(TrainingInstance).get(training_instance_id)
    task.instances.append(instance)
    
    db.add(task)
    db.commit()
    db.refresh(task)
    return task

def update_training_task(db: Session, task_id: str, updates: Dict[str, Any]):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise ValueError(f"Training task {task_id} not found")
    
    # If params are being updated, update the hash too
    if 'params' in updates:
        updates['params_hash'] = compute_params_hash(updates['params'])
    
    for key, value in updates.items():
        setattr(task, key, value)
    
    if 'status' in updates and updates['status'] == TrainingStatus.COMPLETED:
        setattr(task, 'end_date', datetime.utcnow())
    
    db.commit()
    db.refresh(task)
    return task

def get_next_task_in_queue(db: Session, dataset_id: Optional[int] = None):
    """Get the next task in queue that is ready to process"""
    query = db.query(TrainingTask).filter(TrainingTask.status == TrainingStatus.QUEUED)
    
    if dataset_id:
        query = query.filter(TrainingTask.dataset_id == dataset_id)
    
    # Order by queue position
    next_task = query.order_by(TrainingTask.queue_position).first()
    return next_task

def advance_queue(db: Session, dataset_id: Optional[int] = None):
    """Move the next task in the queue to running status"""
    next_task = get_next_task_in_queue(db, dataset_id)
    if next_task:
        update_training_task(db, str(next_task.id), {'status': TrainingStatus.PENDING})
        prepare_dataset_task.delay(next_task.dataset_id, next_task.id, next_task.split_ratios or {"train": 0.7, "val": 0.2, "test": 0.1})
        return next_task
    return None

# Get correct path to data.yaml for a dataset
def get_dataset_yaml_path(dataset_id: int) -> str:
    """Return the path to data.yaml for a dataset"""
    db = SessionLocal()
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    yaml_path = os.path.join(PROJECT_ROOT, "datasets", dataset.name, "yolo_splits", "data.yaml") # type: ignore
    return yaml_path

@app.task(bind=True, name="prepare_dataset")
def prepare_dataset_task(self, dataset_id: int, task_id: str, split_ratios: Dict):
    """
    Celery task for dataset preparation
    """
    from celery.exceptions import SoftTimeLimitExceeded
    logger.info(f"Starting task prepare_dataset_task for dataset {dataset_id}, task_id {task_id}")
    db = SessionLocal()
    
    try:
        logger.info(f"Updating task status to PREPARING")
        update_training_task(db, task_id, {'status': TrainingStatus.PREPARING})
        
        logger.info(f"Beginning dataset preparation")
        start_time = time.time()
        
        yaml_path = prepare_yolo_dataset_by_id(
            db=db,
            dataset_id=dataset_id,
            split_ratios=split_ratios or {"train": 0.7, "val": 0.2, "test": 0.1},
            overwrite=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset preparation completed in {elapsed_time:.2f} seconds")
        
        logger.info(f"Updating task status to PREPARED and queueing training task")
        update_training_task(db, task_id, {
            'dataset_path': yaml_path,
            'status': TrainingStatus.PREPARED
        })

        logger.info("Queueing training task")
        train_model_task.delay(
            dataset_id=dataset_id,
            task_id=task_id,
            yaml_path=str(yaml_path)
        )
        
        return {
            'status': 'SUCCESS',
            'yaml_path': str(yaml_path),
            'dataset_id': dataset_id,
            'task_id': task_id
        }
    except SoftTimeLimitExceeded:
        logger.error(f"Dataset preparation timed out after 5 minutes")
        update_training_task(db, task_id, {
            'status': TrainingStatus.FAILED,
            'error': f"Dataset preparation timed out after 5 minutes"
        })
        advance_queue(db, dataset_id)
        raise
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}", exc_info=True)
        update_training_task(db, task_id, {
            'status': TrainingStatus.FAILED,
            'error': f"Dataset preparation failed: {str(e)}"
        })
    finally:
        db.close()

@app.task(bind=True, name="train_model")
def train_model_task(self, dataset_id: int, task_id: str, yaml_path: str):
    """Train a single model with the task's hyperparameters"""
    db = SessionLocal()
   
    try:
        task = get_training_task(db, task_id)
        if task.status != TrainingStatus.PREPARED:
            raise ValueError("Dataset not prepared for task")
       
        logger.info(f"Starting training for task {task_id}")
        update_training_task(db, task_id, {'status': TrainingStatus.IN_PROGRESS})
        os.chdir(PROJECT_ROOT)
       
        # Get device preference from task parameters
        use_gpu = task.params.get('use_gpu', True)
        device = 0 if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        start_time = datetime.utcnow()
        model = YOLO(f"{PROJECT_ROOT}/experiments/{task.dataset_id}/yolov8n.pt")
       
        # Set up callback to capture metrics after each epoch
        def on_fit_epoch_end(trainer):
            if hasattr(trainer, 'validator'):  # Ensure validation ran
                metrics = trainer.metrics
                if metrics:
                    metrics_dict = {
                        'metrics/mAP50(B)': float(metrics.get('metrics/mAP50(B)', 0)),
                        'metrics/mAP50-95(B)': float(metrics.get('metrics/mAP50-95(B)', 0)),
                        'metrics/precision(B)': float(metrics.get('metrics/precision(B)', 0)),
                        'metrics/recall(B)': float(metrics.get('metrics/recall(B)', 0)),
                        'train/box_loss': float(metrics.get('train/box_loss', 0)),
                        'val/box_loss': float(metrics.get('val/box_loss', 0))
                    }
                    
        # Register callback directly
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        results = model.train(
            data=yaml_path,
            imgsz=task.params.get('imgsz', 640),
            epochs=task.params.get('epochs', 100),
            batch=task.params.get('batch', 16),
            lr0=task.params.get('lr0', 0.01),
            lrf=task.params.get('lrf', 0.1),
            momentum=task.params.get('momentum', 0.937),
            weight_decay=task.params.get('weight_decay', 0.0005),
            warmup_epochs=task.params.get('warmup_epochs', 3.0),
            warmup_momentum=task.params.get('warmup_momentum', 0.8),
            box=task.params.get('box', 7.5),
            cls=task.params.get('cls', 0.5),
            dfl=task.params.get('dfl', 1.5),
            project=f"{PROJECT_ROOT}/runs/runs_{task.dataset_id}",
            name=f"train_{task_id[:8]}",
            exist_ok=True,
            device=device,
            val=True
        )
        
        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()
       
        update_training_task(db, task_id, {
            'status': TrainingStatus.COMPLETED,
            'results': results.results_dict if results else {},
            'model_path': results.save_dir if results else None,
        })

        final_metrics = {
            'metrics/mAP50(B)': results.results_dict.get('metrics/mAP50', 0) if results and hasattr(results, 'results_dict') else 0,
            'metrics/mAP50-95(B)': results.results_dict.get('metrics/mAP50-95', 0) if results and hasattr(results, 'results_dict') else 0,
            'metrics/precision(B)': results.results_dict.get('metrics/precision', 0) if results and hasattr(results, 'results_dict') else 0,
            'metrics/recall(B)': results.results_dict.get('metrics/recall', 0) if results and hasattr(results, 'results_dict') else 0,
            'train/box_loss': results.results_dict.get('train/box_loss', 0) if results and hasattr(results, 'results_dict') else 0,
            'val/box_loss': results.results_dict.get('val/box_loss', 0) if results and hasattr(results, 'results_dict') else 0,
            'inference_speed': results.speed.get('inference', 0) if results and hasattr(results, 'speed') else 0  # ms per image
        }
       
        return {
            'status': 'SUCCESS',
            'task_id': task_id,
            'dataset_id': dataset_id,
            'results': final_metrics,
            'model_path': str(results.save_dir) if results else None,
            "duration_seconds": duration_seconds
        }
    except Exception as e:
        logger.error(f"Training failed for task {task_id}: {str(e)}")
        update_training_task(db, task_id, {
            'status': TrainingStatus.FAILED,
            'error': str(e),
            'end_date': datetime.utcnow()
        })
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        db.close()
        advance_queue(db, task.dataset_id) # type: ignore

def create_testing_task(db: Session, dataset_id: int, testing_instance_id: int, training_task_id: int, queue_position: int):
    """Create a new testing task"""
    task = TestTask(
        dataset_id=dataset_id,
        testing_instance_id=testing_instance_id,
        training_task_id=training_task_id,
        queue_position=queue_position
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task

def get_testing_task(db: Session, task_id: str):
    """Get a testing task by ID"""
    return db.query(TestTask).filter(TestTask.id == task_id).first()

def update_testing_task(db: Session, task_id: str, updates: dict):
    """Update a testing task"""
    task = db.query(TestTask).filter(TestTask.id == task_id).first()
    if task:
        for key, value in updates.items():
            setattr(task, key, value)
        db.commit()
        db.refresh(task)
    return task

def get_completed_training_tasks(db: Session, dataset_id: int):
    """Get all completed training tasks for the most recent training instance of a dataset"""
    # First get the most recent training instance for this dataset
    last_instance = db.query(TrainingInstance).filter(
        TrainingInstance.dataset_id == dataset_id
    ).order_by(
        TrainingInstance.created_at.desc()
    ).first()

    if not last_instance:
        return []

    # Then get all completed tasks associated with this instance
    return db.query(TrainingTask).filter(
        TrainingTask.id.in_([task.id for task in last_instance.tasks]),
        TrainingTask.status == TrainingStatus.COMPLETED
    ).all()


@app.task(bind=True, name='test_model')
def run_inference_task(self, dataset_id: int, training_task_id: int, test_task_id: int, use_gpu: bool):
    """
    Celery task to run inference on a trained YOLO model
    """
    from ultralytics import YOLO
    import os
    
    db = SessionLocal()
    try:
        # Update task status to running
        update_testing_task(db, str(test_task_id), {
            "status": TestingStatus.IN_PROGRESS,
            "started_at": datetime.utcnow()
        })
        
        # Get the training task to access the trained model
        training_task = db.query(TrainingTask).filter(TrainingTask.id == training_task_id).first()
        if not training_task:
            raise Exception(f"Training task {training_task_id} not found")
        
        if not training_task.model_path:
            raise Exception(f"No model path found for training task {training_task_id}")
        
        # Get dataset info
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            raise Exception(f"Dataset {dataset_id} not found")
        
        # Load the trained YOLO model
        logger.info(f"Loading YOLO model from {training_task.model_path}")
        model_path = f"{training_task.model_path}/weights/best.pt"
        model = YOLO(model_path)
        
        # Run YOLO validation/inference
        logger.info(f"Running YOLO validation for testing task {test_task_id}")

        data_yaml = f"datasets/{dataset.name}/yolo_splits/data.yaml"
        device = 0 if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Use YOLO's built-in validation
        results = model.val(
            data=data_yaml,  # Path to dataset.yaml or test images folder
            save=True,
            save_txt=True,
            save_json=True,
            project=f"test_results/dataset_{dataset_id}",
            name=f"task_{test_task_id}",
            exist_ok=True,
            split='test',
            device=device
        )
        
        # Extract metrics from YOLO results
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        
        # Get key metrics
        map50 = float(metrics.get('metrics/mAP50(B)', 0.0))
        map50_95 = float(metrics.get('metrics/mAP50-95(B)', 0.0))
        precision = float(metrics.get('metrics/precision(B)', 0.0))
        recall = float(metrics.get('metrics/recall(B)', 0.0))

        # Get per-class metrics if available
        class_metrics = {}
        if hasattr(results, 'ap_class_index') and hasattr(results, 'ap'):
            for i, class_idx in enumerate(results.ap_class_index):
                class_name = model.names[class_idx] if class_idx < len(model.names) else f"class_{class_idx}"
                class_metrics[class_name] = {
                    'ap50': float(results.ap[i, 0]) if len(results.ap.shape) > 1 else 0.0,
                    'ap50_95': float(results.ap[i, :].mean()) if len(results.ap.shape) > 1 else 0.0
                }
        
        # Update task with results
        update_testing_task(db, str(test_task_id), {
            "status": TestingStatus.COMPLETED,
            "completed_at": datetime.utcnow(),
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "class_metrics": class_metrics,
            "results_summary": metrics
        })
        
        logger.info(f"Testing task {test_task_id} completed successfully - mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")
        
        # Check if there's a next task in the queue
        next_task = db.query(TestTask).filter(
            TestTask.dataset_id == dataset_id,
            TestTask.status == TestingStatus.PENDING
        ).order_by(TestTask.queue_position).first()
        
        if next_task:
            logger.info(f"Starting next testing task {next_task.id}")
            run_inference_task.delay(
                dataset_id=dataset_id,
                training_task_id=next_task.training_task_id,
                test_task_id=next_task.id,
                use_gpu=use_gpu
            )
            update_testing_task(db, str(next_task.id), {"status": TestingStatus.PENDING})
        
    except Exception as e:
        logger.error(f"Testing task {test_task_id} failed: {str(e)}")
        update_testing_task(db, str(test_task_id), {
            "status": TestingStatus.FAILED,
            "completed_at": datetime.utcnow()
        })
        raise
    finally:
        db.close()
