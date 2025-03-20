# app/schemas.py
from pydantic import BaseModel, Field, field_serializer
from datetime import date
from typing import List, Optional

# Pydantic schema for User
class UserBase(BaseModel):
    full_name: str
    company_name: str

class UserResponse(BaseModel):
    id: int
    full_name: str
    company_name: str

    class Config:
        orm_mode = True

# If you need a separate schema that includes datasets
class UserWithDatasetsResponse(UserBase):
    id: int
    datasets: List[int] = []  # List of dataset IDs associated with the user

    class Config:
        orm_mode = True

# Pydantic schema for Image
class ImageBase(BaseModel):
    filename: str

class ImageResponse(ImageBase):
    id: int

    class Config:
        orm_mode = True

# Pydantic schema for Dataset
class DatasetBase(BaseModel):
    name: str
    start_date: str
    end_date: str
    model: str
    created_at: Optional[date] = None

class DatasetResponse(BaseModel):
    name: str
    start_date: str  # Use str instead of date
    end_date: str    # Use str instead of date
    model: str
    id: int

class CreateDatasetRequest(BaseModel):
    model: str
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    user_ids: List[int] = []