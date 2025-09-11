"""
Main FastAPI application for the Infrastructure Health Predictor.
Started as a simple prototype, grew into something more complex.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# TODO: clean up these imports, some might not be needed
from src.models.predictor import Predictor
from src.data.collector import MetricCollector
from src.utils.logger import get_logger

# hack: global state for now, should refactor to dependency injection
predictor_instance = None
collector_instance = None

logger = get_logger(__name__)

app = FastAPI(
    title="Infrastructure Health Predictor API",
    description="API for predicting infrastructure issues before they happen",
    version="0.1.0",
    # debug=True  # commented out for production
)

# CORS middleware - TODO: make origins configurable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # too permissive, should fix
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionRequest(BaseModel):
    cluster_id: str
    time_range: str = "1h"
    metrics: Optional[list] = None

class PredictionResponse(BaseModel):
    cluster_id: str
    timestamp: datetime
    predictions: Dict[str, float]
    confidence: float
    # metadata: Dict[str, Any]  # TODO: add metadata field

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    # uptime: float  # TODO: add uptime tracking

# Dependency functions
def get_predictor():
    """Get or create predictor instance."""
    global predictor_instance
    if predictor_instance is None:
        # lazy loading - might want to change this
        logger.info("Loading predictor model...")
        predictor_instance = Predictor()
        # TODO: handle model loading errors better
    return predictor_instance

def get_collector():
    """Get or create collector instance."""
    global collector_instance
    if collector_instance is None:
        collector_instance = MetricCollector()
    return collector_instance

# Routes
@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Infrastructure Health Predictor API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check(
    predictor: Predictor = Depends(get_predictor)
) -> HealthResponse:
    """Health check endpoint."""
    # TODO: add more health checks (database, redis, etc.)
    model_loaded = predictor.is_model_loaded() if predictor else False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version="0.1.0",
        model_loaded=model_loaded
    )

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    predictor: Predictor = Depends(get_predictor),
    collector: MetricCollector = Depends(get_collector)
) -> PredictionResponse:
    """
    Make predictions for a given cluster.
    
    This endpoint collects recent metrics and runs them through
    the LSTM model to predict potential issues.
    """
    try:
        # Collect metrics
        logger.info(f"Collecting metrics for cluster {request.cluster_id}")
        metrics_data = collector.collect_metrics(
            cluster_id=request.cluster_id,
            time_range=request.time_range
        )
        
        if not metrics_data:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for cluster {request.cluster_id}"
            )
        
        # Make prediction
        # TODO: add caching for repeated requests
        predictions, confidence = predictor.predict(metrics_data)
        
        return PredictionResponse(
            cluster_id=request.cluster_id,
            timestamp=datetime.now(),
            predictions=predictions,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # debug print - should remove in production
        # print(f"Error details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/clusters")
async def list_clusters(
    collector: MetricCollector = Depends(get_collector)
):
    """List available clusters that have metrics."""
    # TODO: cache this response
    clusters = collector.get_available_clusters()
    return {"clusters": clusters}

@app.get("/metrics/{cluster_id}")
async def get_metrics(
    cluster_id: str,
    collector: MetricCollector = Depends(get_collector)
):
    """Get recent metrics for a cluster (debug endpoint)."""
    metrics = collector.get_recent_metrics(cluster_id)
    return {
        "cluster_id": cluster_id,
        "metrics": metrics[:10]  # limit for response size
    }

# TODO: add endpoint for model retraining
# TODO: add endpoint for alert configuration
# TODO: add websocket for real-time updates

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )