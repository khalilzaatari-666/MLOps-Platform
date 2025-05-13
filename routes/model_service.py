from datetime import datetime
import logging
from typing import Dict
from fastapi import APIRouter, Depends, HTTPException, logger, status
from fastapi.responses import JSONResponse
import pandas as pd
from sqlalchemy.orm import Session
from app.model_deployment import deploy_model_to_minio
from core.dependencies import get_db
from app.model_service import prepare_yolo_dataset_by_id
from app.models import BestInstanceModel, DatasetModel, DeployedModel, TestTask, TrainingInstance, TrainingTask
from app.schemas import METRIC_MAPPING, ModelSelectionConfig, TrainModelRequest, TrainingResponse, TrainingStatus, TrainingStatusResponse
from app.tasks import create_training_task, get_training_task, prepare_dataset_task, test_model_task, update_training_task
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent

# Endpoint to prepare YOLO dataset splits
@router.post("/datasets/{dataset_id}/prepare")
def prepare_dataset(
    dataset_id: int, 
    db: Session = Depends(get_db),
    split_ratios: dict = None
):
    try:
        yaml_path = prepare_yolo_dataset_by_id(
            db=db,
            dataset_id=dataset_id,
            split_ratios=split_ratios,
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
    """Get status of a specific training task with current metrics"""
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
       
        if not task:
            raise HTTPException(status_code=404, detail=f"Training task {task_id} not found")
        
        total_epochs = task.params.get("epochs", 0)
        # Prepare the base response
        response = {
            "status": "success",
            "task": {
                "id": task.id,
                "dataset_id": task.dataset_id,
                "status": task.status,
                "params": task.params,
                "start_date": task.start_date.isoformat() if task.start_date else None,
                "end_date": task.end_date.isoformat() if task.end_date else None,
                "error": task.error if task.status == TrainingStatus.FAILED else None,
                "queue_position": task.queue_position
            }
        }
        
        # Handle metrics based on task status
        if task.status == TrainingStatus.IN_PROGRESS:
            short_id = task_id.split('-')[0]
            results_path = f"{PROJECT_ROOT}/runs/runs_{task.dataset_id}/train_{short_id}/results.csv"
            
            try:
                # Read the CSV with the training metrics
                results_df = pd.read_csv(results_path)
                current_epoch = len(results_df)
                progress = min(round((current_epoch / total_epochs)*100), 100) if total_epochs > 0 else 0.0
                
                # Get the latest metrics (last row of the CSV)
                if not results_df.empty:
                    latest_metrics = results_df.iloc[-1].to_dict()
                    # Clean up the metrics for JSON serialization
                    metrics = {
                        "epoch": current_epoch,
                        "metrics/mAP50(B)": float(latest_metrics.get("metrics/mAP50(B)", 0)),
                        "metrics/mAP50-95(B)": float(latest_metrics.get("metrics/mAP50-95(B)", 0)),
                        "metrics/precision(B)": float(latest_metrics.get("metrics/precision(B)", 0)),
                        "metrics/recall(B)": float(latest_metrics.get("metrics/recall(B)", 0))
                    }
                    
                    # Get historical metrics for plotting
                    metrics_history = []
                    for _, row in results_df.iterrows():
                        metrics_history.append({
                            "epoch": int(row.get("epoch", 0)),
                            "metrics/mAP50(B)": float(row.get("metrics/mAP50(B)", 0)),
                            "metrics/mAP50-95(B)": float(row.get("metrics/mAP50-95(B)", 0)),
                            "metrics/precision(B)": float(row.get("metrics/precision(B)", 0)),
                            "metrics/recall(B)": float(row.get("metrics/recall(B)", 0))
                        })
                    
                    response["task"]["current_metrics"] = metrics
                    response["task"]["metrics_history"] = metrics_history
                else:
                    response["task"]["current_metrics"] = {}
                    response["task"]["metrics_history"] = []
                
                response["task"]["progress"] = progress
                response["task"]["current_epoch"] = current_epoch
                response["task"]["total_epochs"] = total_epochs
                
            except FileNotFoundError:
                # If the results file doesn't exist yet
                response["task"]["progress"] = 0
                response["task"]["current_metrics"] = {}
                response["task"]["metrics_history"] = []
                response["task"]["current_epoch"] = 0
                response["task"]["total_epochs"] = total_epochs
        
        elif task.status == TrainingStatus.COMPLETED:
            # For completed tasks, use the stored results
            response["task"]["progress"] = 100.0
            response["task"]["results"] = task.results
            
            # If results contains metrics information, extract it
            if task.results and isinstance(task.results, dict):
                response["task"]["current_metrics"] = {
                    "metrics/mAP50(B)": task.results.get("metrics/mAP50", 0),
                    "metrics/mAP50-95(B)": task.results.get("metrics/mAP50-95", 0),
                    "metrics/precision(B)": task.results.get("metrics/precision", 0),
                    "metrics/recall(B)": task.results.get("metrics/recall", 0)
                }
        else:
            # For other statuses (QUEUED, FAILED, etc.)
            response["task"]["progress"] = 0.0
        
        return response
       
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
        best_model = (db.query(BestInstanceModel).filter(BestInstanceModel.dataset_id == dataset_id).order_by(BestInstanceModel.id.desc()) .first())
        
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

@router.post("/test-model/{dataset_id}", response_model=Dict, status_code=status.HTTP_200_OK)
async def test_model(dataset_id: int, use_gpu: bool):
    """
    Endpoint to test a trained model on a specified dataset
    
    Parameters:
    - dataset_id: ID of the dataset to test on
    
    Returns:
    - Test metrics including precision, recall, mAP scores
    """
    try:
        test_results = test_model_task(dataset_id, use_gpu)
        
        return JSONResponse(
            content=test_results,
            status_code=status.HTTP_200_OK
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Testing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during testing"
        )

@router.post("/deploy_model", response_model=dict)
async def deploy_model(db: Session = Depends(get_db)):
    """
    Endpoint to deploy the best model to PCS-AGRI's MINIO object store.
    """ 
    try: 
        # Get the latest best model
        lastest_best_model = db.query(BestInstanceModel).order_by(BestInstanceModel.created_at.desc()).first()
        if not lastest_best_model:
            raise HTTPException(status_code=404, detail="No best model found")
        
        # Get the model path
        model_folder = lastest_best_model.model_path
        model_path = f"{model_folder}/weights/best.pt"
        if not model_path:
            raise HTTPException(status_code=404, detail="Model path not found")
        
        # Create a name for the model folder in the object store
        model_dataset = db.query(DatasetModel).filter(DatasetModel.id == lastest_best_model.dataset_id).first()
        if not model_dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        model_seed = model_dataset.model
        current_time = datetime.now().strftime("%Y%m%d")
        model_name = f"{model_seed}_{current_time}"
        model_folder_name = f"models-trackseeds/models-test/model_{model_name}"
        destination_path = f"{model_folder_name}/best.pt"

        deploy_results =  deploy_model_to_minio(model_path, destination_path)

        if deploy_results.get("status") == "success":
            new_deployed_model = DeployedModel(
                model_id = lastest_best_model.id,
                dataset_id = lastest_best_model.dataset_id,
                minio_path = deploy_results.get("destination_file"),
                deployment_date = datetime.now(),
                score = lastest_best_model.score,
                model_info = lastest_best_model.model_info,
                status = "active"
            )
        
        db.add(new_deployed_model)
        db.commit()
        db.refresh(new_deployed_model)

        deploy_results["deployed_model_id"] = new_deployed_model.id

        return JSONResponse(
            content=deploy_results,
            status_code=status.HTTP_200_OK
        )
    
    except Exception as e:
        logger.error(f"Failed to deploy model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error while deploying the model"
        )

    
