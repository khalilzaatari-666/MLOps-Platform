from datetime import datetime
import hashlib
import json
import logging
import os
import time
from celery import Celery
import uuid
import mlflow
import pymysql
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO
from app.database import SessionLocal
from app.model_service import prepare_yolo_dataset_by_id
from app.models import BestModel, DatasetModel, TestTask, TrainingInstance, TrainingTask
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
    """Create consistent hash of params JSON"""
    params_str = json.dumps(params, sort_keys=True)  # Sort keys for consistency
    return hashlib.sha256(params_str.encode()).hexdigest()

# Database utility functions
def get_training_task(db: Session, task_id: str):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise ValueError(f"Training task {task_id} not found")
    return task

def create_training_task(db: Session, dataset_id: int, params: Dict[str, Any], training_instance_id: int ,queue_position: Optional[int] = None) -> TrainingTask:
    # Check if task with same dataset_id and params already exists
    params_hash = compute_params_hash(params)
    existing_task = db.query(TrainingTask).filter(
        TrainingTask.dataset_id == dataset_id,
        TrainingTask.params_hash == params_hash
    ).first()
    
    if existing_task:
        raise ValueError(f"Training task with these parameters already exists (ID: {existing_task.id})")
    
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
        params_hash=params_hash,
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

# API Endpoint to get best model info
def get_best_model_info(dataset_id: int):
    """
    API function to get information about the best model for a dataset
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        Dict with best model info or None if not found
    """
    db = SessionLocal()
    try:
        best_model = db.query(BestModel).filter(BestModel.dataset_id == dataset_id).first()
        
        if not best_model:
            return {'status': 'NOT_FOUND', 'dataset_id': dataset_id}
        
        return {
            'status': 'SUCCESS',
            'dataset_id': dataset_id,
            'best_model_id': best_model.id,
            'best_model_task_id': best_model.task_id,
            'model_path': best_model.model_path,
            'score': best_model.score,
            'model_info': best_model.model_info,
            'created_at': best_model.created_at.isoformat()
        }
    finally:
        db.close()

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
    return f"datasets/{dataset.name}/yolo_splits/data.yaml"


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
        
        # MLflow setup 
        mlflow.set_tracking_uri("mysql+pymysql://root:root@localhost/mlflow_tracking")
        experiment_name = f"YOLO_Training_{dataset_id}"
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Starting training for task {task_id}")
        update_training_task(db, task_id, {'status': TrainingStatus.IN_PROGRESS})

        os.chdir(PROJECT_ROOT)
        
        # Get device preference from task parameters
        use_gpu = task.params.get('use_gpu', True)
        device = 0 if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")

        with mlflow.start_run(run_name=f"task_{task_id[:8]}") as run:
            # Log all parameters
            mlflow.pytorch.autolog()
            mlflow.log_params(task.params)
            mlflow.set_tag("celery_task_id", self.request.id)
            mlflow.set_tag("device", str(device))
        
            model = YOLO(f"experiments/{experiment_name}/yolov8n.pt")
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
                device=device
            )

            print(results.results_dict)
            
            update_training_task(db, task_id, {
                'status': TrainingStatus.COMPLETED,
                'results': results.results_dict,
                'model_path': results.save_dir,
                'mlflow_run_id': run.info.run_id
            })
        
        return {
            'status': 'SUCCESS',
            'task_id': task_id,
            'dataset_id': dataset_id,
            'results': results.results_dict,
            'model_path': str(results.save_dir),
            "mlflow_run_id": run.info.run_id
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
        db.close()
        advance_queue(db, task.dataset_id)

@app.task(bind=True, name="select_best_model")
def select_best_model_task(selection_config: dict):
    """Select the best model based on a specific metric"""
    db = SessionLocal()
    try:
        config = ModelSelectionConfig(**selection_config)
        
        tasks = db.query(TrainingTask).filter(
            TrainingTask.dataset_id == config.dataset_id,
            TrainingTask.status == TrainingStatus.COMPLETED
        ).all()

        if not tasks:
            raise ValueError(f"No completed training tasks found for dataset {config.dataset_id}")

        best_task = max(tasks, key=lambda t: t.results.get(config.yolo_metric, -1))
        score = best_task.results.get(config.yolo_metric, 0)
        
        existing_best = db.query(BestModel).filter(
            BestModel.dataset_id == config.dataset_id
        ).first()
        
        if existing_best:
            existing_best.task_id = best_task.id
            existing_best.model_path = best_task.model_path
            existing_best.score = score
            existing_best.model_info = {
                'params': best_task.params,
                'params_hash': best_task.params_hash,
                'metric': config.selection_metric,
                'score': score
            }
            existing_best.created_at = datetime.utcnow()
        else:
            best_model = BestModel(
                dataset_id=config.dataset_id,
                task_id=best_task.id,
                model_path=best_task.model_path,
                score=score,
                model_info={
                    'params': best_task.params,
                    'params_hash': best_task.params_hash,
                    'metric': config.selection_metric,
                    'score': score
                }
            )
            db.add(best_model)
        
        db.commit()
        
        return {
            'status': 'SUCCESS',
            'best_model': {
                'task_id': best_task.id,
                'dataset_id': config.dataset_id,
                'model_path': best_task.model_path,
                'score': score,
                'metric': config.selection_metric
            }
        }
    except Exception as e:
        logger.error(f"Failed to select best model: {str(e)}")
        raise
    finally:
        db.close()

@app.task(bind=True, name="test_model")
def test_model_task(test_config: dict):
    """Test a trained model on a dataset"""
    db = SessionLocal()
    task_id = str(uuid.uuid4())
    
    try:
        config = TestTaskCreate(**test_config)
        
        task = TestTask(
            id=task_id,
            dataset_id=config.dataset_id,
            model_path=config.model_path,
            status=TrainingStatus.IN_PROGRESS,
            start_date=datetime.utcnow()
        )
        db.add(task)
        db.commit()

        data_yaml = get_dataset_yaml_path(config.dataset_id)

        model = YOLO(f"{config.model_path}/weights/best.pt")
        results = model.val(data=data_yaml)

        task.status = TrainingStatus.COMPLETED
        task.end_date = datetime.utcnow()
        task.results = results.results_dict
        db.commit()

        return task.to_dict()
    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")
        if 'task' in locals():
            task.status = TrainingStatus.FAILED
            task.error = str(e)
            task.end_date = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()