from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from core.dependencies import get_db
from app.annotation_service import auto_annotate, process_validated_annotations
router = APIRouter()

# Auto annotation endpoint
@router.post("/annotate/{dataset_id}/{model_id}/{use_gpu}")
def annotate_images(
    dataset_id: int, 
    model_id: int, 
    use_gpu: bool,
    db: Session = Depends(get_db)
):
    auto_annotate(dataset_id=dataset_id, model_id=model_id, use_gpu=use_gpu, db=db)
    return {"message": "Auto-annotation completed successfully."}

# Annotate the images after bounding boxes manual validation
@router.post("/datasets/{dataset_id}/replace_labels")
async def process_annotations(
    dataset_id: int, 
    annotations_zip: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    result = await process_validated_annotations(dataset_id, annotations_zip, db)
    return {"message": "Validated annotations processed successfully"}