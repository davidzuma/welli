"""
FastAPI routes for the Welli retention engine
"""

from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import (
    GoalMatchRequest, GoalMatchResponse,
    UserBehaviorData, ClusterResponse,
    ChurnPredictionRequest, ChurnPredictionResponse,
    DailyPlanRequest, DailyPlanResponse,
    ErrorResponse
)
from src.models.content_matcher import ContentMatcher
from src.models.clustering import UserClusterer
from src.models.churn_model import ChurnPredictor
from src.models.micro_coach import MicroCoach
from src.utils.model_loader import ModelLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize models (these will be loaded lazily)
content_matcher = None
user_clusterer = None 
churn_predictor = None
micro_coach = None

def get_content_matcher():
    """Lazy loading of content matcher"""
    global content_matcher
    if content_matcher is None:
        content_matcher = ContentMatcher()
    return content_matcher

def get_user_clusterer():
    """Lazy loading of user clusterer"""
    global user_clusterer
    if user_clusterer is None:
        user_clusterer = UserClusterer()
    return user_clusterer

def get_churn_predictor():
    """Lazy loading of churn predictor"""
    global churn_predictor
    if churn_predictor is None:
        churn_predictor = ChurnPredictor()
    return churn_predictor

def get_micro_coach():
    """Lazy loading of micro coach"""
    global micro_coach
    if micro_coach is None:
        micro_coach = MicroCoach()
    return micro_coach

@router.post("/match-goal", response_model=GoalMatchResponse)
async def match_goal(
    request: GoalMatchRequest,
    matcher: ContentMatcher = Depends(get_content_matcher)
):
    """
    Match user's goal to semantically similar content using sentence embeddings
    """
    try:
        logger.info(f"Matching goal: {request.goal}")
        result = matcher.match_goal_to_content(
            goal=request.goal,
            limit=request.limit
        )
        return result
    except Exception as e:
        logger.error(f"Error in goal matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Goal matching failed: {str(e)}")

@router.post("/cluster-user", response_model=ClusterResponse)
async def cluster_user(
    user_data: UserBehaviorData,
    clusterer: UserClusterer = Depends(get_user_clusterer)
):
    """
    Cluster user into behavioral segment using early activity data
    """
    try:
        logger.info(f"Clustering user: {user_data.user_id}")
        result = clusterer.cluster_user(user_data)
        return result
    except Exception as e:
        logger.error(f"Error in user clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User clustering failed: {str(e)}")

@router.post("/predict-churn", response_model=ChurnPredictionResponse)
async def predict_churn(
    request: ChurnPredictionRequest,
    predictor: ChurnPredictor = Depends(get_churn_predictor)
):
    """
    Predict churn risk using behavioral classifier
    """
    try:
        logger.info(f"Predicting churn for user: {request.user_id}")
        result = predictor.predict_churn(request)
        return result
    except Exception as e:
        logger.error(f"Error in churn prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Churn prediction failed: {str(e)}")

@router.post("/daily-plan", response_model=DailyPlanResponse)
async def generate_daily_plan(
    request: DailyPlanRequest,
    coach: MicroCoach = Depends(get_micro_coach)
):
    """
    Generate personalized daily wellness plan using OpenAI
    """
    try:
        logger.info(f"Generating daily plan for user: {request.user_id}")
        result = await coach.generate_daily_plan(request)
        return result
    except Exception as e:
        logger.error(f"Error in daily plan generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Daily plan generation failed: {str(e)}")

# Health check for models
@router.get("/models/health")
async def models_health_check():
    """Check if all models are loaded and functioning"""
    health_status = {
        "content_matcher": False,
        "user_clusterer": False,
        "churn_predictor": False,
        "micro_coach": False
    }
    
    try:
        # Test each model
        matcher = get_content_matcher()
        health_status["content_matcher"] = matcher.is_ready()
        
        clusterer = get_user_clusterer()
        health_status["user_clusterer"] = clusterer.is_ready()
        
        predictor = get_churn_predictor()
        health_status["churn_predictor"] = predictor.is_ready()
        
        coach = get_micro_coach()
        health_status["micro_coach"] = coach.is_ready()
        
    except Exception as e:
        logger.error(f"Error in model health check: {str(e)}")
    
    all_healthy = all(health_status.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "models": health_status
    }
