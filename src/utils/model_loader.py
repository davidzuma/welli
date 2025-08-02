"""
Utilities for loading and managing ML models
"""

import pickle
import joblib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility class for loading different types of models and data"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def load_pickle_model(self, model_path: str) -> Any:
        """Load a pickle model file"""
        full_path = self.base_path / model_path
        if not full_path.exists():
            logger.warning(f"Model file not found: {full_path}")
            return None
        
        try:
            with open(full_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded pickle model from {full_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading pickle model {full_path}: {str(e)}")
            return None
    
    def load_joblib_model(self, model_path: str) -> Any:
        """Load a joblib model file"""
        full_path = self.base_path / model_path
        if not full_path.exists():
            logger.warning(f"Model file not found: {full_path}")
            return None
            
        try:
            model = joblib.load(full_path)
            logger.info(f"Loaded joblib model from {full_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading joblib model {full_path}: {str(e)}")
            return None
    
    def load_json_data(self, data_path: str) -> Optional[Dict]:
        """Load JSON data file"""
        full_path = self.base_path / data_path
        if not full_path.exists():
            logger.warning(f"Data file not found: {full_path}")
            return None
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON data from {full_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON data {full_path}: {str(e)}")
            return None
    
    def save_json_data(self, data: Dict, data_path: str) -> bool:
        """Save data to JSON file"""
        full_path = self.base_path / data_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON data to {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON data {full_path}: {str(e)}")
            return False
    
    def model_exists(self, model_path: str) -> bool:
        """Check if model file exists"""
        return (self.base_path / model_path).exists()


def create_dummy_models():
    """Create dummy models and data for testing"""
    loader = ModelLoader()
    
    # Create dummy content catalog
    content_catalog = {
        "content": [
            {
                "id": "meditation_001",
                "title": "5-Minute Morning Meditation",
                "description": "Start your day with mindfulness and calm breathing exercises",
                "category": "meditation",
                "tags": ["morning", "breathing", "mindfulness", "calm"]
            },
            {
                "id": "exercise_001", 
                "title": "Quick Cardio Workout",
                "description": "High-energy 10-minute cardio session to boost your heart rate",
                "category": "exercise",
                "tags": ["cardio", "energy", "fitness", "quick"]
            },
            {
                "id": "nutrition_001",
                "title": "Healthy Breakfast Ideas",
                "description": "Nutritious breakfast recipes to fuel your morning",
                "category": "nutrition", 
                "tags": ["breakfast", "healthy", "nutrition", "recipes"]
            },
            {
                "id": "sleep_001",
                "title": "Better Sleep Hygiene",
                "description": "Tips and techniques for improving your sleep quality",
                "category": "sleep",
                "tags": ["sleep", "rest", "hygiene", "quality"]
            },
            {
                "id": "stress_001",
                "title": "Stress Relief Techniques",
                "description": "Practical methods to manage and reduce daily stress",
                "category": "mental_health",
                "tags": ["stress", "relief", "mental", "techniques"]
            }
        ]
    }
    
    # Save content catalog
    loader.save_json_data(content_catalog, "content_catalog.json")
    
    # Create dummy cluster model info
    cluster_info = {
        "clusters": {
            0: {
                "name": "Routine Builders",
                "description": "Users who prefer structured, consistent wellness routines"
            },
            1: {
                "name": "Explorers", 
                "description": "Users who like trying different activities and approaches"
            },
            2: {
                "name": "Casual Check-ins",
                "description": "Users who engage sporadically but consistently"
            }
        },
        "feature_names": [
            "session_count", "avg_session_duration", "streak_length",
            "preferred_time_encoded", "content_engagement_rate", "notification_response_rate"
        ]
    }
    
    loader.save_json_data(cluster_info, "cluster_info.json")
    
    logger.info("Created dummy models and data files")


# Initialize dummy data if files don't exist
def ensure_dummy_data():
    """Ensure dummy data exists for testing"""
    loader = ModelLoader()
    if not loader.model_exists("content_catalog.json"):
        create_dummy_models()
