from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, logger
import mlflow
import pandas as pd
from sqlalchemy.orm import Session
from core.dependencies import get_db
from app.model_service import prepare_yolo_dataset_by_id
from app.models import BestInstanceModel, BestModel, DatasetModel, TestTask, TrainingInstance, TrainingTask
from app.schemas import METRIC_MAPPING, ModelSelectionConfig, TestTaskCreate, TrainModelRequest, TrainingResponse, TrainingStatus, TrainingStatusResponse
from app.tasks import create_training_task, get_training_task, prepare_dataset_task, test_model_task, update_training_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            overwrite=False  # Set to True for testing
        )
            
        return {
            "status": "success",
            "yaml_path": yaml_path
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
    on a specific dataset. Deletes any existing tasks for this dataset first.
    """
    task_ids = []
    
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri("mysql+pymysql://root:root@localhost/mlflow_tracking")
        experiment_name = f"YOLO_Training_{request.dataset_id}"
        mlflow.set_experiment(experiment_name)
        
        # Delete all existing tasks for this dataset
        #existing_tasks = db.query(TrainingTask).filter(
        #    TrainingTask.dataset_id == request.dataset_id
        #).all()
        
        #for task in existing_tasks:
        #    db.delete(task)
        #db.commit()
        #logger.info(f"Deleted {len(existing_tasks)} existing tasks for dataset {request.dataset_id}")

        # Create a new training instance
        training_instance = TrainingInstance(dataset_id=request.dataset_id)
        db.add(training_instance)
        db.flush()  # This gets the auto-generated ID

        # Create new training tasks for each set of hyperparameters
        for i, params in enumerate(request.params_list):
            params['use_gpu'] = request.use_gpu
            task = create_training_task(
                db=db,
                dataset_id=request.dataset_id,
                training_instance_id=training_instance.id,
                params=params,
                queue_position=i
            )
            task_ids.append(task.id)

            # Log initial params to MLflow
            with mlflow.start_run(run_name=f"task_{task.id[:8]}"):
                mlflow.log_params(params)
                mlflow.set_tag("dataset_id", str(request.dataset_id))
                mlflow.set_tag("use_gpu", str(request.use_gpu))
                mlflow.set_tag("task_id", str(task.id))
                mlflow.set_tag("queue_position", str(i))
                mlflow.set_tag("training_instance_id", str(training_instance.id))

            logger.info(f"Created training task {task.id} for dataset {request.dataset_id} with queue position {i}")

        # Start the first task in the queue if tasks were created
        if task_ids:
            first_task = get_training_task(db, task_ids[0])
            prepare_dataset_task.delay(
                dataset_id=request.dataset_id,
                task_id=first_task.id,
                split_ratios=request.split_ratios
            )
            update_training_task(db, first_task.id, {"status": TrainingStatus.PENDING})
            logger.info(f"Started first training task {first_task.id} for dataset {request.dataset_id}")

        return TrainingResponse(
            task_ids=task_ids,
            status="success",
            message=f"Successfully queued {len(task_ids)} training tasks",
            training_instance_id=training_instance.id
        )
        
    except Exception as e:
        db.rollback()  # Rollback in case of error
        logger.error(f"Failed to start training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.get("/status/{dataset_id}", response_model=TrainingStatusResponse)
def get_training_status(dataset_id: int, db: Session = Depends(get_db)):
    """Get status of all training tasks for a dataset"""
    try:
        # Step 1: Get latest training instance for the dataset
        latest_instance = db.query(TrainingInstance).filter(
            TrainingInstance.dataset_id == dataset_id
        ).order_by(TrainingInstance.created_at.desc()).first()
        
        if not latest_instance:
            return TrainingStatusResponse(
                dataset_id=dataset_id,
                status="NOT_FOUND",
                message="No training instance found for this dataset"
            )
        
        # Step 2: Get all tasks associated with that instance
        tasks = latest_instance.tasks

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
       
        # Calculate overall progress
        completed_count = sum(1 for task in tasks if task.status == TrainingStatus.COMPLETED)
        progress = completed_count / len(tasks) if tasks else 0
       
        # Get start and end dates
        start_dates = [task.start_date for task in tasks if task.start_date]
        end_dates = [task.end_date for task in tasks if task.end_date]
       
        start_date = min(start_dates) if start_dates else None
        end_date = max(end_dates) if end_dates and all_completed else None
       
        # Consolidate subtasks info with individual progress
        subtasks = {
            task.id: {
                "status": task.status,
                "params": task.params,
                "results": task.results if task.status == TrainingStatus.COMPLETED else None,
                "error": task.error if task.status == TrainingStatus.FAILED else None,
                "queue_position": task.queue_position,
                "progress": task.progress if hasattr(task, 'progress') else 0.0,  # Individual task progress
                "start_date": task.start_date.isoformat() if task.start_date else None,
                "end_date": task.end_date.isoformat() if task.end_date else None
            } for task in tasks
        }
       
        return TrainingStatusResponse(
            dataset_id=dataset_id,
            status=status,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
            progress=progress,
            subtasks=subtasks,
        )
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/training-task/{task_id}", response_model=dict)
def get_training_task_status(task_id: str, db: Session = Depends(get_db)):
    """Get status of a specific training task"""
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Training task {task_id} not found")
        if task.status == TrainingStatus.IN_PROGRESS:
            short_id = task_id.split('-')[0]
            results_path = f"runs_{task.dataset_id}/train_{short_id}/results.csv"

            results_df = pd.read_csv(results_path)
            current_epoch = len(results_df)
            total_epochs = task.params.get("epochs", 0)
            progress = min(round((current_epoch / total_epochs)*100) , 100) if total_epochs > 0 else 0.0
            
            # Get current progress and status
            return {
                "status": "success",
                "task": {
                    "id": task.id,
                    "dataset_id": task.dataset_id,
                    "status": task.status,
                    "params": task.params,
                    "progress": progress,
                    "start_date": task.start_date.isoformat() if task.start_date else None,
                    "end_date": task.end_date.isoformat() if task.end_date else None,
                    "results": task.results if task.status == TrainingStatus.COMPLETED else None,
                    "error": task.error if task.status == TrainingStatus.FAILED else None,
                    "queue_position": task.queue_position
                }
            }
        else:
            return {
                "status": "success",
                "task": {
                    "id": task.id,
                    "dataset_id": task.dataset_id,
                    "status": task.status,
                    "params": task.params,
                    "progress": 100.0 if task.status == TrainingStatus.COMPLETED else 0.0,
                    "start_date": task.start_date.isoformat() if task.start_date else None,
                    "end_date": task.end_date.isoformat() if task.end_date else None,
                    "results": task.results if task.status == TrainingStatus.COMPLETED else None,
                    "error": task.error if task.status == TrainingStatus.FAILED else None,
                    "queue_position": task.queue_position
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/select-best-model", response_model=dict)
def select_best_model(
    config: ModelSelectionConfig,
    db: Session = Depends(get_db)
):
    """Endpoint to select the best model for a specific training instance"""
    try:
        # Get the latest training instance for this dataset
        latest_instance = db.query(TrainingInstance).filter(
            TrainingInstance.dataset_id == config.dataset_id
        ).order_by(TrainingInstance.created_at.desc()).first()
        
        if not latest_instance:
            raise HTTPException(
                status_code=404,
                detail=f"No training instances found for dataset {config.dataset_id}"
            )
        
        # Find completed tasks associated with this training instance
        completed_tasks = []
        for task in latest_instance.tasks:
            task_obj = db.query(TrainingTask).filter(
                TrainingTask.id == task.id,
                TrainingTask.status == TrainingStatus.COMPLETED
            ).first()
            if task_obj:
                completed_tasks.append(task_obj)
        
        if not completed_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No completed training tasks found for the latest training instance of dataset {config.dataset_id}"
            )
        
        # Get the appropriate metric name from the mapping
        if config.selection_metric not in METRIC_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric: {config.selection_metric}"
            )
        # Use the mapped YOLO metric via the model property
        yolo_metric = METRIC_MAPPING[config.selection_metric]
            
        # Select the best model based on the metric
        best_task = max(completed_tasks, key=lambda t: t.results.get(yolo_metric, -1))
        score = best_task.results.get(yolo_metric, 0)
        
        # Check for existing best model for this instance
        existing_best = db.query(BestInstanceModel).filter(
            BestInstanceModel.instance_id == latest_instance.id
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
            existing_best.updated_at = datetime.utcnow()
            model_result = existing_best
        else:
            best_instance_model = BestInstanceModel(
                instance_id=latest_instance.id,
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
            db.add(best_instance_model)
            model_result = best_instance_model
        
        db.commit()

        training_task = db.query(TrainingTask).filter(
            TrainingTask.id == best_task.id
        ).first()

        task_params = training_task.params
        
        return {
            'status': 'success',
            'best_model': {
                'id': model_result.id if hasattr(model_result, 'id') else None,
                'task_id': best_task.id,
                'instance_id': latest_instance.id,
                'dataset_id': config.dataset_id,
                'model_path': best_task.model_path,
                'params': task_params,
                'score': score,
                'metric': config.selection_metric
            }
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