from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, ForeignKey, Table, Date, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from ultralytics import YOLO
from app.database import Base

# Association Table (Datasets ↔ Images)
dataset_images = Table(
    "dataset_images",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("image_id", Integer, ForeignKey("images.id"), primary_key=True),
)

# Association Table (Users ↔ Datasets)
user_datasets = Table(
    'user_datasets', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('dataset_id', Integer, ForeignKey('datasets.id'), primary_key=True)
)

class UserModel(Base):
    "Client model for storing clients metadata."
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255))
    company_name = Column(String(255))
    
    # Many-to-many relationship with datasets
    datasets = relationship("DatasetModel", secondary=user_datasets, back_populates="users")

class DatasetModel(Base):
    "Dataset model for storing dataset metadata."
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    start_date = Column(String(255))
    end_date = Column(String(255))
    model = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Many-to-many relationship with users
    users = relationship("UserModel", secondary=user_datasets, back_populates="datasets")
    
    # Many-to-many relationship with images
    images = relationship("ImageModel", secondary=dataset_images, back_populates="datasets")

class ImageModel(Base):
    "Image model for storing image metadata."
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    
    # Many-to-many relationship with datasets
    datasets = relationship("DatasetModel", secondary=dataset_images, back_populates="images")
    bounding_boxes = relationship("BoundingBox", back_populates="image", cascade="all, delete-orphan")

class ModelModel(Base):
    """Tracks metadata for pre-existing local YOLO models"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    model_type = Column(String(50))
    model_path = Column(String(512))
    input_size = Column(Integer)
    class_names = Column(JSON(none_as_null=True), nullable=False, server_default='[]')
    device = Column(String(20), default='cpu')
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime)

class BoundingBox(Base):
    __tablename__ = "bounding_boxes"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('models.id'), index=True)  # Only reference, no back-populate
    image_id = Column(Integer, ForeignKey('images.id'), index=True)
    class_id = Column(Integer)
    class_name = Column(String(100))
    x_center = Column(Float)  # Normalized (0-1)
    y_center = Column(Float)  # Normalized (0-1)
    width = Column(Float)     # Normalized (0-1)
    height = Column(Float)    # Normalized (0-1)
    confidence = Column(Float)
    validation_status = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to Image (one-way to Model)
    image = relationship("ImageModel", back_populates="bounding_boxes")