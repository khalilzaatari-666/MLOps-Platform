from app.data_augmentation import apply_augmentation
from app.schemas import AugmentationRequest, DatasetStatus
from fastapi import APIRouter, Depends, HTTPException
from core.dependencies import get_db
from sqlalchemy.orm import Session
from app.models import DatasetModel

router = APIRouter()

@router.post("/augment-dataset")
async def augment_dataset(request: AugmentationRequest, db: Session = Depends(get_db)):
    """
    Apply data augmentation to a dataset
    
    - **dataset_id**: ID/name of the dataset
    - **transformers**: List of transformers to apply (vertical_flip, horizontal_flip, transpose, center_crop)
    """
    dataset = db.query(DatasetModel).filter(DatasetModel.id == request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    print("received dataset id: " , request.dataset_id)
    print("received transformers: " , request.transformers)
    # Verify if dataset has labels
    if (dataset.status == DatasetStatus.RAW):
        raise HTTPException(
            status_code=400,
            detail="Dataset must be AUTO_ANNOTATED or VALIDATED or AUGMENTED"
        )

    try:
        result = apply_augmentation(
            dataset_id=str(request.dataset_id),
            transformer_types=request.transformers,
            db=db
        )
        dataset.status = DatasetStatus.AUGMENTED
        db.commit()
        return {"message": result}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/available-transformers")
async def get_available_transformers():
    """Get list of available transformers"""
    return {
        "transformers": [
            {
                "name": "vertical_flip",
                "description": "Flip image vertically"
            },
            {
                "name": "horizontal_flip", 
                "description": "Flip image horizontally"
            },
            {
                "name": "transpose",
                "description": "Transpose image (swap width and height)"
            },
            {
                "name": "center_crop",
                "description": "Crop image from center (800x800)"
            }
        ]
    }
