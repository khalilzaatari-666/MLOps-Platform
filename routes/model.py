from fastapi import APIRouter, Depends, HTTPException, logger
from sqlalchemy.orm import Session
import os
from typing import List, Literal
from core.dependencies import get_db
from app.model_service import prepare_yolo_dataset_by_id, register_existing_models
from app.models import BestModel, DatasetModel, TestTask, TrainingTask
from app.schemas import VALID_METRICS, ModelSelectionConfig, TestTaskCreate, TrainModelRequest, TrainingResponse, TrainingStatus, TrainingStatusResponse
from app.tasks import create_training_task, get_training_task, prepare_dataset_task, select_best_model_task, test_model_task, update_training_task
from pathlib import Path


router = APIRouter()

# Endpoint to prepare YOLO dataset splits
@router.post("/datasets/{dataset_id}/prepare")
def prepare_dataset(
    dataset_id: int, 
    db: Session = Depends(get_db)
):
    try:
        yaml_path = prepare_yolo_dataset_by_id(
            db=db,
            dataset_id=dataset_id,
            overwrite=True  # Set to True for testing
        )
        
        # Count files in each split to verify
        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        dataset_path = os.path.join("datasets", dataset.name)
        output_dir = os.path.join(dataset_path, "yolo_splits")
        
        file_counts = {}
        for split in ['train', 'val', 'test']:
            img_count = len(list(Path(os.path.join(output_dir, split, 'images')).glob('*.*')))
            file_counts[split] = {'images': img_count}
            
        return {
            "status": "success",
            "yaml_path": yaml_path,
            "file_counts": file_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/start_training", response_model=TrainingResponse)
def start_training(
    request: TrainModelRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint to start training models with different hyperparameters
    on a specific dataset.
    """
    task_ids = []
    
    try:
        # Create training tasks for each set of hyperparameters
        for i, params in enumerate(request.params_list):
            task = create_training_task(
                db=db,
                dataset_id=request.dataset_id,
                params=params,
                queue_position=i
            )
            task_ids.routerend(task.id)
        
        # Start the first task in the queue
        if task_ids:
            first_task = get_training_task(db, task_ids[0])
            prepare_dataset_task.delay(request.dataset_id, first_task.id, request.split_ratios)
            update_training_task(db, first_task.id, {"status": TrainingStatus.PENDING})
        
        return TrainingResponse(
            task_ids=task_ids,
            status="success",
            message=f"Successfully queued {len(task_ids)} training tasks"
        )
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{dataset_id}", response_model=TrainingStatusResponse)
def get_training_status(dataset_id: int, db: Session = Depends(get_db)):
    """Get status of all training tasks for a dataset"""
    try:
        # Get all training tasks for this dataset
        tasks = db.query(TrainingTask).filter(
            TrainingTask.dataset_id == dataset_id
        ).all()
        
        if not tasks:
            return TrainingStatusResponse(
                dataset_id=dataset_id,
                status="NOT_FOUND",
                message="No training tasks found for this dataset"
            )
        
        # Calculate overall status
        all_completed = all(task.status == TrainingStatus.COMPLETED for task in tasks)
        any_failed = any(task.status == TrainingStatus.FAILED for task in tasks)
        any_running = any(task.status in [TrainingStatus.IN_PROGRESS, TrainingStatus.PREPARING] for task in tasks)
        
        if all_completed:
            status = TrainingStatus.COMPLETED
        elif any_failed and not any_running:
            status = TrainingStatus.FAILED
        elif any_running:
            status = TrainingStatus.IN_PROGRESS
        else:
            status = TrainingStatus.QUEUED
        
        # Get best model if exists
        best_model = db.query(BestModel).filter(
            BestModel.dataset_id == dataset_id
        ).first()
        
        # Calculate progress
        completed_count = sum(1 for task in tasks if task.status == TrainingStatus.COMPLETED)
        progress = completed_count / len(tasks) if tasks else 0
        
        # Get start and end dates
        start_dates = [task.start_date for task in tasks if task.start_date]
        end_dates = [task.end_date for task in tasks if task.end_date]
        
        start_date = min(start_dates) if start_dates else None
        end_date = max(end_dates) if end_dates and all_completed else None
        
        # Consolidate subtasks info
        subtasks = {
            task.id: {
                "status": task.status,
                "params": task.params,
                "results": task.results if task.status == TrainingStatus.COMPLETED else None,
                "error": task.error if task.status == TrainingStatus.FAILED else None,
                "queue_position": task.queue_position
            } for task in tasks
        }
        
        return TrainingStatusResponse(
            dataset_id=dataset_id,
            status=status,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
            progress=progress,
            subtasks=subtasks,
            best_model=best_model.model_info if best_model else None
        )
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/select-best-model", response_model=dict)
def select_best_model(
    config: ModelSelectionConfig,
    db: Session = Depends(get_db)
):
    """Endpoint to select the best model based on a specific metric"""
    try:
        # Validate that the selection metric is valid
        if config.selection_metric not in VALID_METRICS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric: {config.selection_metric}. Valid options: {', '.join(VALID_METRICS)}"
            )
        
        # Validate that there are completed training tasks
        completed_tasks = db.query(TrainingTask).filter(
            TrainingTask.dataset_id == config.dataset_id,
            TrainingTask.status == TrainingStatus.COMPLETED
        ).all()
        
        if not completed_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No completed training tasks found for dataset {config.dataset_id}"
            )
        
        # Start the selection task
        task = select_best_model_task.delay(config.dict())
        
        return {
            "status": "success",
            "message": "Best model selection started",
            "task_id": task.id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select best model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/best-model/{dataset_id}")
def get_best_model_info(dataset_id: int, db: Session = Depends(get_db)):
    """Endpoint to get information about the best model for a dataset"""
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
    except Exception as e:
        logger.error(f"Failed to get best model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-model", response_model=dict)
def test_model(config: TestTaskCreate, db: Session = Depends(get_db)):
    """Endpoint to test a trained model on a dataset"""
    try:
        task = test_model_task.delay(config.dict())
        
        return {
            "status": "success",
            "message": "Model testing started",
            "test_task_id": task.id
        }
    except Exception as e:
        logger.error(f"Failed to start model testing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-status/{test_task_id}")
def get_test_task_status(test_task_id: str, db: Session = Depends(get_db)):
    """Endpoint to get status and results of a test task"""
    try:
        test_task = db.query(TestTask).filter(TestTask.id == test_task_id).first()
        
        if not test_task:
            raise HTTPException(status_code=404, detail=f"Test task {test_task_id} not found")
        
        return {
            'status': 'SUCCESS',
            'test_task': {
                'id': test_task.id,
                'dataset_id': test_task.dataset_id,
                'model_path': test_task.model_path,
                'status': test_task.status,
                'start_date': test_task.start_date.isoformat() if test_task.start_date else None,
                'end_date': test_task.end_date.isoformat() if test_task.end_date else None,
                'results': test_task.results,
                'error': test_task.error
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))