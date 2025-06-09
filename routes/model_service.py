from datetime import datetime
import logging
from typing import Dict, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
from sqlalchemy.orm import Session
from app.model_deployment import deploy_model_to_minio
from core.dependencies import get_db
from app.model_service import prepare_yolo_dataset_by_id
from app.models import BestInstanceModel, DatasetModel, DeployedModel, TestTask, TestingInstance, TrainingInstance, TrainingTask, training_instance_task_association
from app.schemas import METRIC_MAPPING, ModelSelectionConfig, TestModelRequest, TestingResponse, TestingStatus, TrainModelRequest, TrainingResponse, TrainingStatus, TrainingStatusResponse
from app.tasks import create_testing_task, create_training_task, get_completed_training_tasks, get_testing_task, get_training_task, prepare_dataset_task, run_inference_task, update_testing_task, update_training_task
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
    split_ratios: Optional[dict] = None
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
            update_training_task(db, str(first_task.id), {"status": TrainingStatus.PENDING})
            logger.info(f"Started first training task {first_task.id} for dataset {request.dataset_id}")

        return TrainingResponse(
            task_ids=task_ids,
            status="success",
            message=f"Successfully queued {len(task_ids)} training tasks"
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
                "error": task.error if task.status == str(TrainingStatus.FAILED) else None,
                "queue_position": task.queue_position,
                "progress": task.progress if hasattr(task, 'progress') else 0.0,  # Individual task progress
                "start_date": task.start_date.isoformat() if getattr(task, "start_date", None) else None,
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
                "start_date": task.start_date.isoformat() if getattr(task, "start_date", None) else None,
                "end_date": task.end_date.isoformat() if getattr(task, "end_date", None) else None,
                "error": task.error if str(task.status) == str(TrainingStatus.FAILED) else None,
                "queue_position": task.queue_position
            }
        }
        
        # Handle metrics based on task status
        if str(task.status) == str(TrainingStatus.IN_PROGRESS):
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
                        "metrics/recall(B)": float(latest_metrics.get("metrics/recall(B)", 0)),
                        'train/box_loss': float(latest_metrics.get('train/box_loss', 0)),
                        'val/box_loss': float(latest_metrics.get('val/box_loss', 0)),
                    }
                    
                    # Get historical metrics for plotting
                    metrics_history = []
                    for _, row in results_df.iterrows():
                        metrics_history.append({
                            "epoch": int(row.get("epoch", 0)),
                            "metrics/mAP50(B)": float(row.get("metrics/mAP50(B)", 0)),
                            "metrics/mAP50-95(B)": float(row.get("metrics/mAP50-95(B)", 0)),
                            "metrics/precision(B)": float(row.get("metrics/precision(B)", 0)),
                            "metrics/recall(B)": float(row.get("metrics/recall(B)", 0)),
                            "train/box_loss": float(row.get("train/box_loss", 0)),
                            "val/box_loss": float(row.get("val/box_loss", 0))
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
        
        elif str(task.status) == str(TrainingStatus.COMPLETED):
            # For completed tasks, use the stored results
            response["task"]["progress"] = 100.0
            response["task"]["results"] = task.results
            
            # If results contains metrics information, extract it
            if task.results is not None and isinstance(task.results, dict):
                response["task"]["current_metrics"] = {
                    "metrics/mAP50(B)": task.results.get("metrics/mAP50", 0),
                    "metrics/mAP50-95(B)": task.results.get("metrics/mAP50-95", 0),
                    "metrics/precision(B)": task.results.get("metrics/precision", 0),
                    "metrics/recall(B)": task.results.get("metrics/recall", 0),
                    "train/box_loss": task.results.get("train/box_loss", 0),
                    "val/box_loss": task.results.get("val/box_loss", 0)
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
    
@router.post("/start_testing", response_model=TestingResponse)
def start_testing(
    request: TestModelRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint to start testing all trained models for a dataset.
    Runs inference on each completed model using the test split.
    """
    test_task_ids = []
    try:
        # Get all completed training tasks for this dataset
        completed_training_tasks = get_completed_training_tasks(db, request.dataset_id)
        if not completed_training_tasks:
            return TestingResponse(
                test_task_ids=[],
                status="error",
                message=f"No completed training tasks found for dataset {request.dataset_id}"
            )
        
        # Create a new testing instance
        testing_instance = TestingInstance(dataset_id=request.dataset_id)
        db.add(testing_instance)
        db.flush()  # This gets the auto-generated ID
        
        # Create testing tasks for each completed training task
        for i, training_task in enumerate(completed_training_tasks):
            test_task = create_testing_task(
                db=db,
                dataset_id=request.dataset_id,
                testing_instance_id=testing_instance.id,
                training_task_id=training_task.id,
                queue_position=i
            )
            test_task_ids.append(test_task.id)
            logger.info(f"Created testing task {test_task.id} for training task {training_task.id} (queue position: {i})")
        
        # Commit all the testing tasks
        db.commit()
        
        # Start the first testing task if any were created
        if test_task_ids:
            first_test_task = get_testing_task(db, test_task_ids[0])
            logger.info(f"Starting first testing task {first_test_task.id} with queue position {first_test_task.queue_position}")
            
            run_inference_task.delay(
                dataset_id=request.dataset_id,
                training_task_id=first_test_task.training_task_id,
                test_task_id=first_test_task.id,
                use_gpu=request.useGpu
            )
            
            logger.info(f"Queued first testing task {first_test_task.id}")
        
        return TestingResponse(
            test_task_ids=test_task_ids,
            status="success",
            message=f"Successfully queued {len(test_task_ids)} testing tasks"
        )
        
    except Exception as e:
        logger.error(f"Error starting testing for dataset {request.dataset_id}: {str(e)}")
        return TestingResponse(
            test_task_ids=[],
            status="error",
            message=f"Failed to start testing: {str(e)}"
        )

# Additional endpoint to get testing results from latest testing instance
@router.get("/testing_results/{dataset_id}")
def get_testing_results(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get comprehensive testing results for the latest testing instance of a dataset.
    Returns progress statistics and individual task details.
    """
    # Get the most recent testing instance for this dataset
    latest_testing_instance = db.query(TestingInstance).filter(
        TestingInstance.dataset_id == dataset_id
    ).order_by(TestingInstance.created_at.desc()).first()

    if not latest_testing_instance:
        return {
            "dataset_id": dataset_id,
            "testing_instance_id": None,
            "testing_instance_created_at": None,
            "progress": {
                "total_tasks": 0,
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "progress_percentage": 0.0
            },
            "tasks": [],
            "message": "No testing instances found for this dataset"
        }

    # Refresh to ensure we get the latest state
    db.refresh(latest_testing_instance)

    # Get all testing tasks for this instance
    test_tasks = db.query(TestTask).filter(
        TestTask.testing_instance_id == latest_testing_instance.id
    ).all()

    # Calculate progress statistics
    status_counts = {
        "PENDING": 0,
        "IN_PROGRESS": 0,
        "COMPLETED": 0,
        "FAILED": 0
    }

    for task in test_tasks:
        status_counts[task.status] += 1

    total_tasks = len(test_tasks)
    progress_percentage = (
        (status_counts["COMPLETED"] / total_tasks * 100) 
        if total_tasks > 0 else 0.0
    )

    # Build detailed task results
    tasks = []
    for task in test_tasks:
        # Get associated training task details
        training_task = db.query(TrainingTask).filter(
            TrainingTask.id == task.training_task_id
        ).first()

        tasks.append({
            "test_task_id": str(task.id),
            "training_task_id": str(task.training_task_id),
            "queue_position": task.queue_position,
            "status": task.status,
            "hyperparameters": training_task.params if training_task else None,
            "model_path": training_task.model_path if training_task else None,
            "map50": task.map50,
            "map50_95": task.map50_95,
            "precision": task.precision,
            "recall": task.recall,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        })

    return {
        "dataset_id": dataset_id,
        "testing_instance_id": latest_testing_instance.id,
        "testing_instance_created_at": latest_testing_instance.created_at.isoformat(),
        "progress": {
            "total_tasks": total_tasks,
            "pending": status_counts["PENDING"],
            "in_progress": status_counts["IN_PROGRESS"],
            "completed": status_counts["COMPLETED"],
            "failed": status_counts["FAILED"],
            "progress_percentage": round(progress_percentage, 2)
        },
        "tasks": tasks
    }

# Endpoint to get individual testing task results
@router.get("/test_task/{test_task_id}/results")
def get_test_task_results(test_task_id: UUID, db: Session = Depends(get_db)):
    """Get detailed results of a specific testing task"""
    test_task_id_str = str(test_task_id)
    test_task = db.query(TestTask).filter(TestTask.id == test_task_id_str).first()
    
    if not test_task:
        raise HTTPException(status_code=404, detail="Testing task not found")
    
    # Get associated training task info (regardless of status)
    training_task = db.query(TrainingTask).filter(
        TrainingTask.id == test_task.training_task_id
    ).first()
    
    # Base response structure
    response = {
        "test_task_id": test_task.id,
        "training_task_id": test_task.training_task_id,
        "dataset_id": test_task.dataset_id,
        "testing_instance_id": test_task.testing_instance_id,
        "status": test_task.status,
        "hyperparameters": training_task.params if training_task else None,
        "model_path": training_task.model_path if training_task else None,
        "started_at": test_task.started_at,
        "completed_at": test_task.completed_at,
        "duration_seconds": (
            (test_task.completed_at - test_task.started_at).total_seconds() 
            if test_task.completed_at and test_task.started_at 
            else None
        ),
        "metrics": {
            "map50": test_task.map50,
            "map50_95": test_task.map50_95,
            "precision": test_task.precision,
            "recall": test_task.recall
        }
    }
    
    return response

@router.post("/select-best-model", response_model=dict)
def select_best_model(
    config: ModelSelectionConfig,
    db: Session = Depends(get_db)
):
    """Select best model based on test task metrics"""
    try:
        # Get the latest testing instance
        latest_test_instance = db.query(TestingInstance).filter(
            TestingInstance.dataset_id == config.dataset_id
        ).order_by(TestingInstance.created_at.desc()).first()
        
        if not latest_test_instance:
            raise HTTPException(
                status_code=404,
                detail=f"No testing instances found for dataset {config.dataset_id}"
            )
        
        # Get completed test tasks with metrics
        completed_test_tasks = db.query(TestTask).filter(
            TestTask.testing_instance_id == latest_test_instance.id,
            TestTask.status == TestingStatus.COMPLETED,
            TestTask.map50.isnot(None)  # Ensure we have metrics
        ).all()
        
        if not completed_test_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No completed test tasks with metrics found for dataset {config.dataset_id}"
            )

        # Metric mapping to test task fields
        METRIC_MAPPING = {
            'map50': 'map50',
            'map50_95': 'map50_95',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        if config.selection_metric not in METRIC_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric. Choose from: {list(METRIC_MAPPING.keys())}"
            )
        
        metric_field = METRIC_MAPPING[config.selection_metric]
        
        # Find task with best metric value
        best_test_task = max(
            completed_test_tasks,
            key=lambda task: getattr(task, metric_field, -1)
        )
        best_score = getattr(best_test_task, metric_field)
        
        # Get associated training task
        training_task = db.query(TrainingTask).filter(
            TrainingTask.id == best_test_task.training_task_id
        ).first()
        
        if not training_task:
            raise HTTPException(
                status_code=404,
                detail="Associated training task not found"
            )
        
        training_instance = db.query(TrainingInstance).join(
            training_instance_task_association,
            TrainingInstance.id == training_instance_task_association.c.training_instance_id
        ).filter(
            training_instance_task_association.c.training_task_id == training_task.id
        ).first()

        
        # Update or create best model record
        existing_best = db.query(BestInstanceModel).filter(
            BestInstanceModel.task_id == training_task.id
        ).first()
        
        if existing_best:
            # Update existing
            existing_best.task_id = str(training_task.id)
            existing_best.instance_id = training_instance.id
            existing_best.model_path = training_task.model_path
            existing_best.score = best_score
            existing_best.model_info = {
                'params': training_task.params,
                'test_task_id': best_test_task.id,
                'metric': config.selection_metric,
                'score': best_score,
                'test_metrics': {
                    'map50': best_test_task.map50,
                    'map50_95': best_test_task.map50_95,
                    'precision': best_test_task.precision,
                    'recall': best_test_task.recall
                }
            }
            existing_best.updated_at = datetime.utcnow()
        else:
            # Create new
            best_model = BestInstanceModel(
                instance_id = training_instance.id,
                dataset_id=config.dataset_id,
                task_id=training_task.id,
                model_path=training_task.model_path,
                score=best_score,
                model_info={
                    'params': training_task.params,
                    'test_task_id': best_test_task.id,
                    'metric': config.selection_metric,
                    'score': best_score,
                    'test_metrics': {
                        'map50': best_test_task.map50,
                        'map50_95': best_test_task.map50_95,
                        'precision': best_test_task.precision,
                        'recall': best_test_task.recall
                    }
                }
            )
            db.add(best_model)
        
        db.commit()

        return {
            'status': 'success',
            'best_model': {
                'dataset_id': config.dataset_id,
                'training_task_id': training_task.id,
                'test_task_id': best_test_task.id,
                'model_path': training_task.model_path,
                'metric': config.selection_metric,
                'params': training_task.params,
                'score': best_score,
                'test_metrics': {
                    'map50': best_test_task.map50,
                    'map50_95': best_test_task.map50_95,
                    'precision': best_test_task.precision,
                    'recall': best_test_task.recall
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Model selection failed: {str(e)}")
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

        if deploy_results is not None and deploy_results.get("status") == "success":
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
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error while deploying the model: deploy_model_to_minio returned None or failed"
            )
    
    except Exception as e:
        logger.error(f"Failed to deploy model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error while deploying the model"
        )

    
