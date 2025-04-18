import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from app.metrics import metrics_exporter
from routes import model
from app.database import engine, Base
from app.database import SessionLocal
from app.model_service import register_existing_models
from routes import annotation, crud, file_handling, mlflow_service, training_instances

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create all tables
Base.metadata.create_all(bind=engine)
logger.info("Tables created successfully!")

# Initialize FastAPI app
app = FastAPI(title="MLOps Platform API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start metrics server
metrics_exporter.start_server()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/dashboard")
async def dashboard():
    """Redirect to the Grafana dashboard"""
    return RedirectResponse(url="http://localhost:3000/d/ml-metrics/ml-training-metrics?orgId=1")

@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Endpoint for exporting Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

app.include_router(annotation.router, prefix="", tags=["annotation"])
app.include_router(crud.router, prefix="", tags=["crud"])
app.include_router(file_handling.router, prefix="", tags=["file_handling"])
app.include_router(model.router, prefix="", tags=["model"])
app.include_router(mlflow_service.router, prefix="", tags=["mlflow_service"])
app.include_router(training_instances.router, prefix="", tags=["training-instances"])

@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        register_existing_models(db=db)
    finally:
        db.close()

