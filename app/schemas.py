# app/schemas.py
from pydantic import BaseModel, Field, field_serializer, field_validator, validator
from datetime import date, datetime
from typing import List, Optional
from ultralytics import YOLO

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
    created_at: str
    users: List[int]  # List of user IDs
    images: List[int] # List of image IDs

class CreateDatasetRequest(BaseModel):
    model: str
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