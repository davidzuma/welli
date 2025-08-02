"""
Churn prediction using behavioral classifier - requires trained model
"""

import numpy as np
from typing import Dict, Any, List
import logging
from dotenv import load_dotenv
from src.utils.model_loader import ModelLoader
from src.utils.feature_prep import FeaturePreparator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ChurnPredictor:
    """
    Predicts user churn risk using behavioral features - requires trained model
    """
    
    def __init__(self):
        self.model_loader = ModelLoader(base_path="ml_models/churn_classification")
        self.feature_prep = FeaturePreparator()
        self.churn_model = None
        self.churn_scaler = None
        self.feature_names = [
            "days_since_signup",
            "total_sessions", 
            "avg_session_duration",
            "streak_length",
            "last_login_days_ago",
            "content_completion_rate",
            "notification_response_rate",
            "goal_progress_percentage"
        ]
        
        # Load and validate churn model
        self._load_model()
    
    def _load_model(self):
        """Load trained churn prediction model and scaler - required, no fallback"""
        # Load trained churn model - required
        self.churn_model = self.model_loader.load_joblib_model("churn_model.joblib")
        
        if not self.churn_model:
            raise Exception("Churn model not found. Please ensure churn_model.joblib exists in ml_models/churn_classification/ directory")
        
        # Load churn scaler - required
        self.churn_scaler = self.model_loader.load_joblib_model("churn_scaler.joblib")
        
        if not self.churn_scaler:
            raise Exception("Churn scaler not found. Please ensure churn_scaler.joblib exists in ml_models/churn_classification/ directory")
        
        logger.info("Successfully loaded churn prediction model and scaler")
    
    def _get_risk_factors(self, features: np.ndarray, churn_prob: float) -> List[str]:
        """Identify key risk factors contributing to churn probability"""
        factors = []
        
        days_since_signup = features[0, 0]
        total_sessions = features[0, 1]
        last_login_days_ago = features[0, 4]
        completion_rate = features[0, 5]
        notification_response = features[0, 6]
        goal_progress = features[0, 7]
        
        if last_login_days_ago > 7:
            factors.append("Extended period without app usage")
        elif last_login_days_ago > 3:
            factors.append("Several days since last login")
        
        if total_sessions < 3 and days_since_signup > 7:
            factors.append("Low session count relative to signup time")
        
        if completion_rate < 0.3:
            factors.append("Low content completion rate")
        
        if notification_response < 0.2:
            factors.append("Poor notification engagement")
        
        if goal_progress < 0.2:
            factors.append("Limited progress toward wellness goals")
        
        if not factors and churn_prob > 0.5:
            factors.append("General engagement decline patterns")
        
        return factors
    
    def _get_intervention_recommendation(self, risk_level: str, factors: List[str]) -> str:
        """Get intervention recommendation based on risk level and factors"""
        if risk_level == "high":
            if "Extended period without app usage" in factors:
                return "Send personalized re-engagement campaign with wellness goal reminder"
            elif "Low content completion rate" in factors:
                return "Offer shorter, easier content options with immediate rewards"
            else:
                return "Provide one-on-one check-in with personalized motivation"
        
        elif risk_level == "medium":
            if "notification engagement" in " ".join(factors).lower():
                return "Optimize notification timing and personalize message content"
            else:
                return "Introduce streak-building challenges with social elements"
        
        else:  # low risk
            return "Continue current engagement patterns with occasional check-ins"
    
    def predict_churn(self, user_data: Any) -> Dict[str, Any]:
        """
        Predict churn probability for a user - requires trained model
        
        Args:
            user_data: ChurnPredictionRequest object with user metrics
            
        Returns:
            Dictionary with churn prediction and recommendations
        """
        if not self.churn_model:
            raise Exception("Churn model not loaded")
        
        if not self.churn_scaler:
            raise Exception("Churn scaler not loaded")
        
        try:
            # Convert Pydantic model to dict
            if hasattr(user_data, 'dict'):
                user_dict = user_data.dict()
            else:
                user_dict = user_data
            
            # Prepare features
            features = self.feature_prep.prepare_churn_features(user_dict)
            
            # Get churn probability using trained model
            churn_prob = float(self.churn_model.predict_proba(features)[0][1])
            
            # Determine risk level
            if churn_prob >= 0.7:
                risk_level = "high"
            elif churn_prob >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Get risk factors and intervention
            risk_factors = self._get_risk_factors(features, churn_prob)
            intervention = self._get_intervention_recommendation(risk_level, risk_factors)
            
            return {
                "user_id": user_dict["user_id"],
                "churn_probability": round(churn_prob, 3),
                "risk_level": risk_level,
                "recommended_intervention": intervention,
                "factors_contributing_to_risk": risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error in churn prediction: {str(e)}")
            raise Exception(f"Churn prediction failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if the churn predictor is ready to use"""
        return self.churn_model is not None and self.churn_scaler is not None
