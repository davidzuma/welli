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


