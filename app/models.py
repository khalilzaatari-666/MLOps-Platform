from datetime import datetime
import hashlib
import json
import uuid
from pydantic import Field
from sqlalchemy import Column, Integer, Float, String, ForeignKey, Table, Boolean, DateTime, JSON, Enum, UniqueConstraint
from sqlalchemy.orm import relationship
from ultralytics import YOLO
from app.database import Base
from app.schemas import TrainingStatus, DatasetStatus

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

# Association table for many-to-many
training_instance_task_association = Table(
    'training_instance_task_association',
    Base.metadata,
    Column('training_instance_id', Integer, ForeignKey('training_instances.id'), primary_key=True),
    Column('training_task_id', String(36), ForeignKey('training_tasks.id'), primary_key=True)
)

# Model for storing metadata of users
class UserModel(Base):
    "Client model for storing clients metadata."
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255))
    company_name = Column(String(255))
    
    # Many-to-many relationship with datasets
    datasets = relationship("DatasetModel", secondary=user_datasets, back_populates="users")

# Model for storing metadata of datasets
class DatasetModel(Base):
    "Dataset model for storing dataset metadata."
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    start_date = Column(String(255))
    end_date = Column(String(255))
    model = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.RAW)
    
    # Many-to-many relationship with users
    users = relationship("UserModel", secondary=user_datasets, back_populates="datasets")
    # Many-to-many relationship with images
    images = relationship("ImageModel", secondary=dataset_images, back_populates="datasets")
    # One-to-many relationship with training tasks
    test_tasks = relationship("TestTask", back_populates="dataset")
    # One-to-many relationship with training tasks
    best_model = relationship("BestModel", back_populates="dataset", uselist=False)
    # One-to-many relationship with training tasks
    training_tasks = relationship("TrainingTask", back_populates="dataset")
    # One-to-many relationship with training instances
    training_instances = relationship("TrainingInstance", back_populates="dataset")


# Model for storing metadata of images
class ImageModel(Base):
    "Image model for storing image metadata."
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    
    # Many-to-many relationship with datasets
    datasets = relationship("DatasetModel", secondary=dataset_images, back_populates="images")
    bounding_boxes = relationship("BoundingBox", back_populates="image", cascade="all, delete-orphan")


# Model for storing metadata of pre-existing local YOLO models
class ModelModel(Base):
    """Tracks metadata for pre-existing local YOLO models"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    model_type = Column(String(50))
    model_path = Column(String(512))
    input_size = Column(Integer)
    class_names = Column(JSON(none_as_null=True), nullable=False)
    device = Column(String(20), default='cpu')
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime)

    def __init__(self, **kwargs):
        if 'class_names' not in kwargs:
            kwargs['class_names'] = []
        super().__init__(**kwargs)

# Bounding Box model
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

class TrainingInstance(Base):
    __tablename__ = "training_instances"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    
    tasks = relationship("TrainingTask", 
                        secondary=training_instance_task_association,
                        back_populates="instances")
    dataset = relationship("DatasetModel", back_populates="training_instances")

class TrainingTask(Base):
    __tablename__ = "training_tasks"
    __table_args__ = (
        UniqueConstraint('dataset_id', 'params_hash', name='uq_dataset_id_params_hash'),
    )
   
    id = Column(String(36), primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    status = Column(Enum(TrainingStatus))
    params = Column(JSON)
    params_hash = Column(String(64), nullable=False)
    results = Column(JSON)
    model_path = Column(String(255))  # Reasonable path length
    queue_position = Column(Integer)
    dataset_path = Column(String(255))  # Reasonable path length
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    error = Column(String(1024))
    split_ratios = Column(JSON , nullable=True)  # Store split ratios for train/val/test

    # Relationship to DatasetModel
    dataset = relationship("DatasetModel", back_populates="training_tasks")

    # Relationship to TrainingInstance
    instances = relationship("TrainingInstance",
                            secondary=training_instance_task_association,
                            back_populates="tasks")

    @staticmethod
    def compute_params_hash(params):
        """Create consistent hash of params JSON"""
        # Add timestamp to ensure uniqueness
        unique_params = params.copy()
        unique_params["_unique"] = datetime.utcnow().isoformat()  # Or use uuid.uuid4()
        
        return hashlib.sha256(
            json.dumps(unique_params, sort_keys=True).encode()
        ).hexdigest()
    
    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "params": self.params,
            "results": self.results,
            "model_path": self.model_path,
            "queue_position": self.queue_position,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "error": self.error
        }
    
class BestInstanceModel(Base):
    __tablename__ = "best_instance_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    instance_id = Column(Integer, ForeignKey("training_instances.id"), unique=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    task_id = Column(String(36), ForeignKey("training_tasks.id"))
    model_path = Column(String(255))
    score = Column(Float)
    model_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define relationships
    instance = relationship("TrainingInstance")
    dataset = relationship("DatasetModel")
    task = relationship("TrainingTask")
    
class TestTask(Base):
    __tablename__ = "test_tasks"
   
    id = Column(String(36), primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    dataset_tested_on = Column(Integer)
    model_path = Column(String(255))  # Reasonable path length
    status = Column(Enum(TrainingStatus))
    results = Column(JSON)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    error = Column(String(500))

    # Relationship to DatasetModel
    dataset = relationship("DatasetModel", back_populates="test_tasks")
    
    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "model_path": self.model_path,
            "status": self.status,
            "results": self.results,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "error": self
        }


# Add BestModel table
class BestModel(Base):
    __tablename__ = "best_models"
   
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), unique=True)
    task_id = Column(String(36), ForeignKey("training_tasks.id"))
    model_path = Column(String(255), nullable=False)
    model_info = Column(JSON, nullable=True)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
   
    # Relationships
    dataset = relationship("DatasetModel", back_populates="best_model")
    training_task = relationship("TrainingTask")

