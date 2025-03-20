from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import date
from app.crud import fetch_images  # Import your fetch_images function
from app.models import Base, UserModel  # Import your models

# Database setup (replace with your actual database URL)
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/pcsagri"  # Example SQLite database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables (if they don't exist)
Base.metadata.create_all(bind=engine)

# Create a database session
db = SessionLocal()

# Test the fetch_images function
images_info = fetch_images(
    db=db,
    model="tomate, aubergine, poivron",
    start_date="2025-03-16",
    end_date="2025-03-17",
    user_ids=[]
)
print("Fetched Images Info:", images_info)