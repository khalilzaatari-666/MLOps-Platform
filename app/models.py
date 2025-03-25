from sqlalchemy import Column, Integer, String, ForeignKey, Table, Date
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
    full_name = Column(String(255))
    company_name = Column(String(255))
    
    # Many-to-many relationship with datasets
    datasets = relationship("DatasetModel", secondary=user_datasets, back_populates="users")

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    start_date = Column(String(255))
    end_date = Column(String(255))
    model = Column(String(255))
    created_at = Column(Date)
    
    # Many-to-many relationship with users
    users = relationship("UserModel", secondary=user_datasets, back_populates="datasets")
    
    # Many-to-many relationship with images
    images = relationship("ImageModel", secondary=dataset_images, back_populates="datasets")

class ImageModel(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    
    # Many-to-many relationship with datasets
    datasets = relationship("DatasetModel", secondary=dataset_images, back_populates="images")