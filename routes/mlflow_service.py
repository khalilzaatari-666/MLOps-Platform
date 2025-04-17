from celery import Task
from fastapi import APIRouter
from mlflow import MlflowClient
from celery.app.control import Control
from app.database import SessionLocal
from app.models import TrainingTask
from app.tasks import app


client = MlflowClient()
router = APIRouter()

@router.get("/active_runs/{dataset_id}")
async def get_active_runs(dataset_id: str):
    """
    Get active runs for a given experiment ID.
    """
    # Fetch active runs from MLflow
    experiment_name = f"YOLO_Training_{dataset_id}"
    experiment = client.get_experiment_by_name(experiment_name)
    # If the experiment exists, fetch active runs
    if experiment:
        experiment_id = experiment.experiment_id
        active_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'RUNNING'"
        )
    else:
        active_runs = []
        print(f"No experiment found with name: {experiment_name}")

    return [run.to_dictionary() for run in active_runs]

@router.get("/revoke_all_known_tasks")
def revoke_all_known_tasks():
    db = SessionLocal()
    tasks = db.query(TrainingTask).filter(TrainingTask.status.in_(["PENDING", "PREPARING", "PREPARED" ,"QUEUD"])).all()

    for task in tasks:
        if task.celery_id:
            try:
                app.control.revoke(task.celery_id, terminate=True, signal="SIGKILL")
                task.status = "REVOKED"
                db.add(task)
                print(f"Revoked tasks successfully!")
            except Exception as e:
                print(f"Failed to revoke task {task.id}: {e}")
    db.commit()
    db.close()