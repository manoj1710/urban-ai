import joblib
import pandas as pd
import logging
from datetime import datetime
from urbanflux_ai.utils.constants import FRESHNESS_MODEL_PATH

logger = logging.getLogger(__name__)

class FreshnessService:
    def __init__(self):
        try:
            self.model = joblib.load(FRESHNESS_MODEL_PATH)
            logger.info(f"Freshness model loaded successfully from {FRESHNESS_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Freshness model not loaded: {e}")
            self.model = None
    
    def predict(self, produced_date_str, expiry_date_str, storage_type, quality_grade):
        if not self.model:
            return {"error": "Model not loaded"}

        produced_date = datetime.strptime(produced_date_str, "%Y-%m-%d")
        now = datetime.now()
        
        days_in_storage = (now - produced_date).days
        
        input_data = pd.DataFrame([{
            'days_in_storage': days_in_storage,
            'storage_type': storage_type,
            'quality_grade': quality_grade
        }])
        
        prediction = self.model.predict(input_data)[0]
        prediction = float(prediction)  # Convert numpy type to Python float
        prediction = max(0, min(100, prediction))
        
        confidence = 0.95 if 0 <= days_in_storage <= 20 else 0.8
        
        return {
            "freshness": round(prediction, 1),
            "confidence": confidence
        }
