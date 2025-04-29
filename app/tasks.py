from datetime import datetime
import hashlib
import json
import logging
import os

from prometheus_client import start_http_server
os.environ['MLFLOW_TRACKING_URI'] = ''
os.environ['MLFLOW_DISABLE'] = 'true'

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
from app.models import BestInstanceModel, BestModel, DatasetModel, TestTask, TrainingInstance, TrainingTask
from app.schemas import (ModelSelectionConfig, TestTaskCreate, TrainingStatus)
from core.settings import PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('tasks',
    broker='pyamqp://guest:guest@localhost:5672//',
    backend='redis://localhost:6379/1')

METRIC_MAPPING = {
    'accuracy': 'metrics/mAP50(B)',
    'precision': 'metrics/precision(B)',
    'recall': 'metrics/recall(B)'
}

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
        task.end_date = datetime.utcnow()
    
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
        update_training_task(db, next_task.id, {'status': TrainingStatus.PENDING})
        prepare_dataset_task.delay(next_task.dataset_id, next_task.id, next_task.split_ratios or {"train": 0.7, "val": 0.2, "test": 0.1})
        return next_task
    return None

# API Endpoint to get test task status
def get_test_task_status(test_task_id: str):
    """Get status and results of a test task"""
    db = SessionLocal()
    try:
        test_task = db.query(TestTask).filter(TestTask.id == test_task_id).first()
        
        if not test_task:
            return {'status': 'NOT_FOUND', 'test_task_id': test_task_id}
        
        return {
            'status': 'SUCCESS',
            'test_task': {
                'id': test_task.id,
                'dataset_id': test_task.dataset_id,
                'model_path': test_task.model_path,
                'status': test_task.status,
                'start_date': test_task.start_date.isoformat(),
                'end_date': test_task.end_date.isoformat() if test_task.end_date else None,
                'results': test_task.results,
                'error': test_task.error
            }
        }
    finally:
        db.close()

# Get correct path to data.yaml for a dataset
def get_dataset_yaml_path(dataset_id: int) -> str:
    """Return the path to data.yaml for a dataset"""
    db = SessionLocal()
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    yaml_path = os.path.join(PROJECT_ROOT, "datasets", dataset.name, "yolo_splits", "data.yaml")
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
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}", exc_info=True)
        update_training_task(db, task_id, {
            'status': TrainingStatus.FAILED,
            'error': f"Dataset preparation failed: {str(e)}"
        })
    except SoftTimeLimitExceeded:
        logger.error(f"Dataset preparation timed out after 5 minutes")
        update_training_task(db, task_id, {
            'status': TrainingStatus.FAILED,
            'error': f"Dataset preparation timed out after 5 minutes"
        })
        advance_queue(db, dataset_id)
        raise
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
        model = YOLO(f"experiments/{task.dataset_id}/yolov8n.pt")
       
        # Set up callback to capture metrics after each epoch
        def on_fit_epoch_end(trainer):
            if hasattr(trainer, 'validator'):  # Ensure validation ran
                metrics = trainer.metrics
                if metrics:
                    metrics_dict = {
                        'metrics/mAP50(B)': float(metrics.get('metrics/mAP50(B)', 0)),
                        'metrics/mAP50-95(B)': float(metrics.get('metrics/mAP50-95(B)', 0)),
                        'metrics/precision(B)': float(metrics.get('metrics/precision(B)', 0)),
                        'metrics/recall(B)': float(metrics.get('metrics/recall(B)', 0))
                    }
                    
        # Register callback directly
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        results = model.train(
            data=yaml_path,
            epochs=task.params.get('epochs', 100),
            batch=task.params.get('batch_size', 16),
            lr0=task.params.get('lr0', 0.01),
            lrf=task.params.get('lrf', 0.1),
            momentum=task.params.get('momentum', 0.937),
            weight_decay=task.params.get('weight_decay', 0.0005),
            warmup_epochs=task.params.get('warmup_epochs', 3.0),
            warmup_momentum=task.params.get('warmup_momentum', 0.8),
            box=task.params.get('box', 7.5),
            cls=task.params.get('cls', 0.5),
            dfl=task.params.get('dfl', 1.5),
            project=f"runs_{task.dataset_id}",
            name=f"train_{task_id[:8]}",
            exist_ok=True,
            device=device,
            val=True
        )
        
        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()
       
        update_training_task(db, task_id, {
            'status': TrainingStatus.COMPLETED,
            'results': results.results_dict,
            'model_path': results.save_dir,
        })

        final_metrics = {
            'metrics/mAP50(B)': results.results_dict.get('metrics/mAP50', 0),
            'metrics/mAP50-95(B)': results.results_dict.get('metrics/mAP50-95', 0),
            'metrics/precision(B)': results.results_dict.get('metrics/precision', 0),
            'metrics/recall(B)': results.results_dict.get('metrics/recall', 0),
            'inference_speed': results.speed.get('inference', 0)  # ms per image
        }
       
        return {
            'status': 'SUCCESS',
            'task_id': task_id,
            'dataset_id': dataset_id,
            'results': results.results_dict,
            'model_path': str(results.save_dir),
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
        advance_queue(db, task.dataset_id)


def test_model_task(dataset_id: int):
    """Test a trained model on a dataset"""
    db = SessionLocal()
    
    try:
        last_best_model = db.query(BestInstanceModel).order_by(BestInstanceModel.instance_id.desc()).first()
        if not last_best_model:
            raise ValueError("No available model to test")
        
        if last_best_model.dataset_id == dataset_id:
            raise ValueError("Model already trained on this dataset")
        
        # Prepare dataset paths
        try:
            data_yaml = prepare_yolo_dataset_by_id(
                db,
                dataset_id,
                split_ratios={"train": 0.2, "val": 0.2, "test": 0.6},
                overwrite=True
            )
            if not Path(data_yaml).exists():
                raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")
        except Exception as e:
            raise RuntimeError(f"Dataset preparation failed: {str(e)}")

        # Validate model path
        model_file = Path(last_best_model.model_path) / "weights" / "best.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found at: {model_file}")
        
        # Record start time
        start_time = datetime.utcnow()
        
        # Load model
        try:
            logger.info(f"Loading model from {model_file}")
            model = YOLO(str(model_file))
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        # Run validation
        try:
            logger.info(f"Starting validation on dataset ID: {last_best_model.dataset_id}")
            results = model.val(data=data_yaml, split='test')
            
            # Record end time
            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()
            
            if not hasattr(results, 'results_dict'):
                raise RuntimeError("Validation did not return expected results format")
            
            # Extract and format metrics
            metrics = {
                'precision': results.results_dict.get('metrics/precision(B)'),
                'recall': results.results_dict.get('metrics/recall(B)'),
                'map50': results.results_dict.get('metrics/mAP50(B)'),
                'map': results.results_dict.get('metrics/mAP50-95(B)'),
                'inference_speed': results.speed.get('inference'),
                'test_timestamp': datetime.utcnow().isoformat(),
                'model_id': last_best_model.instance_id,
                'model_path': str(model_file),
                'dataset_id': last_best_model.dataset_id,
                'trained_on_dataset': dataset_id,
                'start_date': start_time.isoformat(),
                'end_date': end_time.isoformat(),
                'duration_seconds': duration_seconds
            }
            
            logger.info(
                f"Test completed. mAP50: {metrics['map50']:.3f}, "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, "
                f"Duration: {metrics['duration_seconds']:.2f} seconds"
            )
        
            test_task = TestTask(
                id=str(uuid.uuid4()),
                dataset_id=last_best_model.dataset_id,
                dataset_tested_on=dataset_id,
                model_path=str(model_file),
                status=TrainingStatus.COMPLETED,
                start_date=start_time,
                end_date=end_time,
                results=metrics
            )
            db.add(test_task)
            db.commit()
            db.refresh(test_task)
            
            return {
                **metrics,
                "test_task_id": test_task.id 
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model testing failed: {str(e)}")
        
    finally:
        db.close()

