import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.responses import Response
from app.database import SessionLocal, Base, engine
from app.model_service import register_existing_models
from routes import annotation, crud, file_handling, training_instances, model_service

# Initialize FastAPI app
app = FastAPI(title="MLOps Platform API")

logger = logging.getLogger("uvicorn")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creating tables from models
@app.on_event("startup")
async def startup():
    """Create database tables on startup"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Start a database session and register pre-trained models
        db = SessionLocal()
        try:
            register_existing_models(db=db)
            logger.info("Tables created, and pre-trained models registered.")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

app.include_router(annotation.router, prefix="", tags=["annotation"])
app.include_router(crud.router, prefix="", tags=["crud"])
app.include_router(file_handling.router, prefix="", tags=["file_handling"])
app.include_router(model_service.router, prefix="", tags=["model"])
app.include_router(training_instances.router, prefix="", tags=["training-instances"])


