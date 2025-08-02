"""
User behavioral clustering using KMeans model - requires trained model
"""

import numpy as np
from typing import Dict, Any
import logging
from dotenv import load_dotenv
from src.utils.model_loader import ModelLoader
from src.utils.feature_prep import FeaturePreparator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class UserClusterer:
    """
    Clusters users into behavioral segments using early activity data - requires trained model
    """
    
    def __init__(self):
        self.model_loader = ModelLoader(base_path="ml_models/clustering")
        self.feature_prep = FeaturePreparator()
        self.kmeans_model = None
        self.clustering_scaler = None
        self.cluster_info = None
        
        # Load and validate models
        self._load_models()
    
    def _load_models(self):
        """Load KMeans model, scaler, and cluster information - required, no fallback"""
        # Load trained KMeans model - required
        self.kmeans_model = self.model_loader.load_joblib_model("kmeans_model.joblib")
        
        if not self.kmeans_model:
            raise Exception("KMeans model not found. Please ensure kmeans_model.joblib exists in ml_models/clustering/ directory")
        
        # Load clustering scaler - required
        self.clustering_scaler = self.model_loader.load_joblib_model("clustering_scaler.joblib")
        
        if not self.clustering_scaler:
            raise Exception("Clustering scaler not found. Please ensure clustering_scaler.joblib exists in ml_models/clustering/ directory")
        
        # Load cluster information - required
        self.cluster_info = self.model_loader.load_json_data("cluster_info.json")
        
        if not self.cluster_info:
            raise Exception("Cluster info not found. Please ensure cluster_info.json exists in ml_models/clustering/ directory")
        
        logger.info("Successfully loaded KMeans model, scaler, and cluster info")
    
    def cluster_user(self, user_data: Any) -> Dict[str, Any]:
        """
        Cluster user into behavioral segment - requires trained model
        
        Args:
            user_data: UserBehaviorData object with user metrics
            
        Returns:
            Dictionary with cluster assignment and metadata
        """
        if not self.kmeans_model:
            raise Exception("KMeans model not loaded")
        
        if not self.clustering_scaler:
            raise Exception("Clustering scaler not loaded")
        
        if not self.cluster_info:
            raise Exception("Cluster info not loaded")
        
        try:
            # Convert Pydantic model to dict
            if hasattr(user_data, 'dict'):
                user_dict = user_data.dict()
            else:
                user_dict = user_data
            
            # Prepare features
            features = self.feature_prep.prepare_clustering_features(user_dict)
            
            # Get cluster prediction using trained model
            cluster_id = int(self.kmeans_model.predict(features)[0])
            
            # Calculate confidence as distance to cluster center
            distances = self.kmeans_model.transform(features)[0]
            min_distance = min(distances)
            max_distance = max(distances) if max(distances) > 0 else 1.0
            confidence = 1.0 - (min_distance / max_distance)
            
            # Get cluster information
            cluster_data = self.cluster_info["clusters"].get(str(cluster_id), {
                "name": f"Cluster {cluster_id}",
                "description": "Unknown cluster"
            })
            
            return {
                "user_id": user_dict["user_id"],
                "cluster_id": cluster_id,
                "cluster_name": cluster_data["name"],
                "cluster_description": cluster_data["description"],
                "confidence_score": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in user clustering: {str(e)}")
            raise Exception(f"User clustering failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if the clusterer is ready to use"""
        return (self.kmeans_model is not None and 
                self.clustering_scaler is not None and
                self.cluster_info is not None)
