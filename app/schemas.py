from pydantic import ConfigDict
import enum
from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional
from ultralytics import YOLO

METRIC_MAPPING = {
    'accuracy': 'metrics/mAP50(B)',
    'precision': 'metrics/precision(B)',
    'recall': 'metrics/recall(B)'
}

# Pydantic schema for User
class UserBase(BaseModel):
    full_name: str
    company_name: str

class UserResponse(BaseModel):
    id: int
    full_name: str
    company_name: str

    class Config:
        from_attributes = True

# If you need a separate schema that includes datasets
class UserWithDatasetsResponse(UserBase):
    id: int
    datasets: List[int] = []  # List of dataset IDs associated with the user

    class Config:
        from_attributes = True

# Pydantic schema for Image
class ImageBase(BaseModel):
    filename: str

class ImageResponse(ImageBase):
    id: int

    class Config:
        from_attributes = True

# Pydantic schema for Dataset
class DatasetBase(BaseModel):
    name: str
    start_date: str
    end_date: str
    model: str
    created_at: datetime
    users: List[int]  # List of user IDs
    images: List[int]  # List of image IDs

class DatasetResponse(BaseModel):
    id: int
    name: str
    start_date: date  # Use date instead of str
    end_date: date    # Use date instead of str
    model: str
    created_at: date
    users_id: List[int] = Field(alias="users")
    status: str
    class Config:
        from_attributes = True

class DatasetStatus(str, enum.Enum):
    RAW = "RAW"  
    AUTO_ANNOTATED = "AUTO_ANNOTATED"
    VALIDATED = "VALIDATED"

class CreateDatasetRequest(BaseModel):
    model: str = Field(..., description="The model to use for the dataset")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    user_ids: List[int] = []

class ModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    model_path: str
    input_size: int
    class_names: List[str]
    device: str
    is_active: bool
    last_used: datetime

    @field_validator('class_names', mode='before')
    @classmethod
    def convert_class_names(cls, v):
        if isinstance(v, dict):
            return list(v.values())  # Convert {'0':'class0'} to ['class0']
        if isinstance(v, str):
            return [v]  # Convert string to single-item list
        if not isinstance(v, list):
            raise ValueError("class_names must be a list, dict, or str")
        return v
    
    class Config:
        from_attributes = True


class HyperparameterConfig(BaseModel):
    lr0: float = 0.01
    lrf: float = 0.1
    epochs: int = 100,
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

class TrainingStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TrainModelRequest(BaseModel):
    dataset_id: int
    params_list: List[Dict[str, Any]]
    split_ratios: Optional[Dict[str, float]] = {"train": 0.7, "val": 0.3}
    use_gpu: bool = False

class TrainingResponse(BaseModel):
    task_ids: List[str]
    status: str
    message: str

class TrainingStatusResponse(BaseModel):
    dataset_id: int
    status: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    progress: Optional[float] = None
    subtasks: Dict[str, Any] = {}
    best_model: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TrainingTaskSchema(BaseModel):
    id: str
    dataset_id: int
    status: TrainingStatus
    params: Dict[str, Any]
    params_hash: str
    results: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    queue_position: Optional[int] = None
    dataset_path: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    error: Optional[str] = None
    split_ratios: Optional[Dict[str, float]] = None
    
    model_config = ConfigDict(
        from_attributes=True,  # Replaces orm_mode=True
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )

class HyperparameterConfig(BaseModel):
    epochs: int = Field(100, ge=10, description="Must be ≥10")
    batch_size: int = Field(16, ge=1, description="Batch size for training")
    lr0: float = Field(0.01, gt=0, description="Initial learning rate")
    lrf: float = Field(0.1, gt=0, description="Final learning rate factor")
    momentum: float = Field(0.937, ge=0, le=1, description="SGD momentum/Adam beta1")
    weight_decay: float = Field(0.0005, ge=0, description="Optimizer weight decay")
    warmup_epochs: float = Field(3.0, ge=0, description="Warmup epochs")
    warmup_momentum: float = Field(0.8, ge=0, le=1, description="Warmup momentum")
    box: float = Field(7.5, ge=0, description="Box loss gain")
    cls: float = Field(0.5, ge=0, description="Class loss gain")
    dfl: float = Field(1.5, ge=0, description="DFL loss gain")

class TrainingTaskCreate(BaseModel):
    dataset_id: int
    hyperparams: HyperparameterConfig

class ModelSelectionConfig(BaseModel):
    dataset_id: int
    selection_metric: Literal['accuracy', 'precision', 'recall']
    instance_id: Optional[int] = None

    @property
    def yolo_metric(self) -> str:
        return METRIC_MAPPING[self.selection_metric]

class TestTaskCreate(BaseModel):
    dataset_id: int