# MLOps FastAPI Project

## Project Overview
AI Models Optimization MLOps Platform for PCS-AGRI

## Setup Instructions
1. Clone this repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the FastAPI server: `uvicorn app.main:app --reload`

## Project Structure
- `data/`: Contains all data used in the project
  - `raw/`: Raw, immutable data
  - `processed/`: Processed data ready for modeling
  - `external/`: Data from external sources
- `models/`: Saved model files
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `src/`: Source code for model development
  - `data/`: Scripts for data processing
  - `features/`: Scripts for feature engineering
  - `models/`: Scripts for model training and evaluation
  - `visualization/`: Scripts for creating visualizations
  - `pipeline/`: Scripts for the ML pipeline
- `app/`: FastAPI application
  - `api/`: API endpoints
  - `core/`: Core functionality
  - `schemas/`: Pydantic models for request/response validation
  - `services/`: Business logic and services
- `config/`: Configuration files
- `docs/`: Documentation
- `tests/`: Unit and integration tests

## API Documentation
When the server is running, you can access:
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

