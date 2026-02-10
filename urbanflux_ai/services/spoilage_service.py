import joblib
import pandas as pd
import numpy as np
import logging
from urbanflux_ai.utils.constants import SPOILAGE_MODEL_PATH

logger = logging.getLogger(__name__)

class SpoilageService:
    def __init__(self):
        try:
            self.model = joblib.load(SPOILAGE_MODEL_PATH)
            logger.info(f"Spoilage model loaded successfully from {SPOILAGE_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Spoilage model not loaded: {e}")
            self.model = None
    
    def predict(self, freshness, delay_hours, temperature, congestion):
        if not self.model:
            return {"error": "Model not loaded"}

        input_data = pd.DataFrame([{
            'current_freshness': freshness,
            'delay_factor': 1.0 + (delay_hours * 0.2), # Approximate mapping
            'temperature': temperature,
            'congestion_level': congestion
        }])
        
        prediction = self.model.predict(input_data)[0]
        probs = self.model.predict_proba(input_data)
        confidence = float(np.max(probs))  # Convert numpy type to Python float
        
        return {
            "risk_level": str(prediction),  # Ensure string type
            "risk_score": round(confidence, 2)
        }
