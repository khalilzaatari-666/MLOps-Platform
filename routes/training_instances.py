from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from app.models import DatasetModel, TrainingInstance, TrainingTask, training_instance_task_association
from app.schemas import METRIC_MAPPING, TrainingStatus, TrainingTaskSchema
from core.dependencies import get_db

router = APIRouter(prefix="/training-tasks", tags=["training-tasks"])

@router.get("/last-instance/{dataset_id}", response_model=List[Dict])
def get_last_instance_tasks(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Get all tasks from the latest training instance for a dataset"""
    # Get latest instance
    latest_instance = db.query(TrainingInstance).filter(
        TrainingInstance.dataset_id == dataset_id
    ).order_by(TrainingInstance.created_at.desc()).first()
    
    if not latest_instance:
        raise HTTPException(404, detail="No training instances found")
    
    # Get associated tasks
    tasks = db.query(TrainingTask).join(
        training_instance_task_association
    ).filter(
        training_instance_task_association.c.training_instance_id == latest_instance.id
    ).all()
    
    return [task.to_dict() for task in tasks]

@router.get("/last-instance", response_model=List[Dict])
def get_latest_instance_tasks(
    db: Session = Depends(get_db)
):
    """
    Get all training tasks from the most recent training instance
    """
    # Get the latest training instance
    latest_instance = db.query(TrainingInstance).order_by(
        TrainingInstance.created_at.desc()
    ).first()
    
    if not latest_instance:
        raise HTTPException(
            status_code=404,
            detail="No training instances found"
        )
    
    # Get all tasks associated with this instance
    tasks = db.query(TrainingTask).join(
        training_instance_task_association
    ).filter(
        training_instance_task_association.c.training_instance_id == latest_instance.id
    ).all()
    
    return [task.to_dict() for task in tasks]

@router.get("/last-instance-info", response_model=Dict)
def get_latest_instance_info(
    db:Session = Depends(get_db)
):
    """
    Get informations about the last training instance
    """
    latest_instance = db.query(TrainingInstance).order_by(TrainingInstance.created_at.desc()).first()

    dataset = db.query(DatasetModel).filter(DatasetModel.id == latest_instance.dataset_id).first()

    if not latest_instance:
        raise HTTPException(status_code=404, detail="No training instances found")

    instance_data = {
        "id": latest_instance.id,
        "created_at": latest_instance.created_at.isoformat(),
        "dataset_id": latest_instance.dataset_id,
        "dataset_name": dataset.name,
        "dataset_group": dataset.model,
    }

    return instance_data

@router.get("/all", response_model=List[Dict])
def get_all_training_tasks(
    db: Session = Depends(get_db)
):
    """Get all training tasks sorted by end_date"""
    tasks = db.query(TrainingTask).order_by(
        TrainingTask.end_date.desc()  # Using existing end_date field
    ).all()
    
    return [task.to_dict() for task in tasks]