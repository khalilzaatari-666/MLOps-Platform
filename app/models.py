from sqlalchemy import Column, Integer, String, ForeignKey, Table, Date, DateTime, func
from sqlalchemy.orm import relationship
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
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    company_name = Column(String)
    
    # Define the relationship using string references
    datasets = relationship("DatasetModel", secondary=user_datasets, back_populates="users")

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    start_date = Column(Date)
    end_date = Column(Date)
    model = Column(String)
    created_at = Column(Date)
    
    # Define the relationship using string references
    users = relationship("UserModel", secondary=user_datasets, back_populates="datasets")
    images = relationship("ImageModel", secondary=dataset_images, back_populates="dataset")

class ImageModel(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    
    # Define the relationship
    dataset = relationship("DatasetModel", secondary=dataset_images , back_populates="images")