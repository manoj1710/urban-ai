from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urbanflux_ai.services.freshness_service import FreshnessService
from urbanflux_ai.services.spoilage_service import SpoilageService
from urbanflux_ai.services.priority_service import PriorityService
from urbanflux_ai.services.route_service import RouteService
import uvicorn
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="UrbanFlux AI Engine", description="AI Backend for Logistics Optimization")

# Load Services
freshness_service = FreshnessService()
spoilage_service = SpoilageService()
priority_service = PriorityService()
route_service = RouteService()

@app.on_event("startup")
async def startup_event():
    logger.info("UrbanFlux AI Engine starting up...")
    logger.info(f"Freshness model loaded: {freshness_service.model is not None}")
    logger.info(f"Spoilage model loaded: {spoilage_service.model is not None}")
    logger.info(f"Priority model loaded: {priority_service.model is not None}")

# --- Pydantic Models for Validation ---

class FreshnessRequest(BaseModel):
    produced_date: str
    expiry_date: str
    storage_type: str
    quality_grade: str

class SpoilageRequest(BaseModel):
    freshness: float
    delay_hours: float
    temperature: float
    congestion: str

class PriorityRequest(BaseModel):
    spoilage_risk: str
    customer_demand: float
    distance_km: float

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "urbanflux-ai",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/ai/freshness")
def predict_freshness(data: FreshnessRequest):
    try:
        result = freshness_service.predict(
            data.produced_date, 
            data.expiry_date, 
            data.storage_type, 
            data.quality_grade
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/spoilage-risk")
def predict_spoilage(data: SpoilageRequest):
    try:
        result = spoilage_service.predict(
            data.freshness,
            data.delay_hours,
            data.temperature,
            data.congestion
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/priority-score")
def predict_priority(data: PriorityRequest):
    try:
        result = priority_service.predict(
            data.spoilage_risk,
            data.customer_demand,
            data.distance_km
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/route-analysis")
def analyze_route():
    try:
        result = route_service.analyze()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
