"""
Feature preparation utilities for ML models
"""

import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeaturePreparator:
    """Utility class for preparing features for ML models"""
    
    def __init__(self):
        self.time_encoding = {
            "morning": 0,
            "afternoon": 1, 
            "evening": 2
        }
    
    def prepare_clustering_features(self, user_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for user clustering
        
        Args:
            user_data: Dictionary containing user behavior data
            
        Returns:
            numpy array of features ready for clustering
        """
        try:
            features = [
                user_data.get("session_count", 0),
                user_data.get("avg_session_duration", 0.0),
                user_data.get("streak_length", 0),
                self.time_encoding.get(user_data.get("preferred_time_of_day", "morning"), 0),
                user_data.get("content_engagement_rate", 0.0),
                user_data.get("notification_response_rate", 0.0)
            ]
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error preparing clustering features: {str(e)}")
            return np.zeros((1, 6))
    
    def prepare_churn_features(self, user_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for churn prediction
        
        Args:
            user_data: Dictionary containing user behavior data
            
        Returns:
            numpy array of features ready for churn prediction
        """
        try:
            features = [
                user_data.get("days_since_signup", 0),
                user_data.get("total_sessions", 0),
                user_data.get("avg_session_duration", 0.0),
                user_data.get("streak_length", 0),
                user_data.get("last_login_days_ago", 0),
                user_data.get("content_completion_rate", 0.0),
                user_data.get("notification_response_rate", 0.0),
                user_data.get("goal_progress_percentage", 0.0)
            ]
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error preparing churn features: {str(e)}")
            return np.zeros((1, 8))
    
    def normalize_similarity_scores(self, scores: List[float]) -> List[float]:
        """Normalize similarity scores to 0-1 range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching"""
        # Simple keyword extraction - in production, use more sophisticated NLP
        import re
        
        # Clean and split text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'me', 'my', 'myself', 'we', 'our'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Return top 10 keywords
