from app.database import engine
from app.models import Base  # Import your Base class that has all models

# Create all tables
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")

