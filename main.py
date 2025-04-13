from fastapi import FastAPI
from app.database import SessionLocal
from app.model_service import register_existing_models
from routes import annotation, crud, file_handling, model
 
app = FastAPI()

app.include_router(annotation.router, prefix="", tags=["annotation"])
app.include_router(crud.router, prefix="", tags=["crud"])
app.include_router(file_handling.router, prefix="", tags=["file_handling"])
app.include_router(model.router, prefix="", tags=["model"])

@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        register_existing_models(db=db)
    finally:
        db.close()

