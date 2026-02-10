import joblib
import pandas as pd
import logging
from urbanflux_ai.utils.constants import PRIORITY_MODEL_PATH

logger = logging.getLogger(__name__)

class PriorityService:
    def __init__(self):
        try:
            self.model = joblib.load(PRIORITY_MODEL_PATH)
            logger.info(f"Priority model loaded successfully from {PRIORITY_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Priority model not loaded: {e}")
            self.model = None
    
    def predict(self, spoilage_risk, demand, distance):
        if not self.model:
            return {"error": "Model not loaded"}

        input_data = pd.DataFrame([{
            'spoilage_risk': spoilage_risk,
            'demand_score': demand,
            'distance_km': distance
        }])
        
        score = self.model.predict(input_data)[0]
        score = float(score)  # Convert numpy type to Python float
        score = max(0, min(10, score))
        
        reasons = []
        if demand > 80:
            reasons.append("High customer demand")
        if spoilage_risk == "High":
            reasons.append("High spoilage risk")
        if distance < 50:
            reasons.append("Short distance")
            
        reason_str = ", ".join(reasons) if reasons else "Balanced factors"
        
        return {
            "priority_score": round(score, 1),
            "confidence": 0.89,
            "reason": reason_str
        }
