"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Goal Matching Schemas
class GoalMatchRequest(BaseModel):
    goal: str = Field(..., description="User's wellness goal in natural language")
    limit: Optional[int] = Field(5, description="Number of content recommendations to return")

class ContentItem(BaseModel):
    id: str
    title: str
    description: str
    category: str
    similarity_score: float

class GoalMatchResponse(BaseModel):
    user_goal: str
    matched_content: List[ContentItem]
    total_results: int

# User Clustering Schemas
class UserBehaviorData(BaseModel):
    user_id: str
    session_count: int = Field(..., description="Number of app sessions")
    avg_session_duration: float = Field(..., description="Average session duration in minutes")
    streak_length: int = Field(..., description="Current streak in days")
    preferred_time_of_day: str = Field(..., description="morning, afternoon, evening")
    content_engagement_rate: float = Field(..., description="Percentage of content completed")
    notification_response_rate: float = Field(..., description="Percentage of notifications acted upon")

class ClusterResponse(BaseModel):
    user_id: str
    cluster_id: int
    cluster_name: str
    cluster_description: str
    confidence_score: float

# Churn Prediction Schemas
class ChurnPredictionRequest(BaseModel):
    user_id: str
    days_since_signup: int
    total_sessions: int
    avg_session_duration: float
    streak_length: int
    last_login_days_ago: int
    content_completion_rate: float
    notification_response_rate: float
    goal_progress_percentage: float

class ChurnPredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_level: str  # "low", "medium", "high"
    recommended_intervention: str
    factors_contributing_to_risk: List[str]

# Daily Plan Schemas
class DailyPlanRequest(BaseModel):
    user_id: str
    goal: str = Field(..., description="User's main wellness goal")
    current_streak: int = Field(..., description="Current streak in days")
    recent_activities: List[str] = Field(..., description="List of recent activities completed")
    available_time_minutes: Optional[int] = Field(15, description="Available time for wellness activities")
    preferred_time: Optional[str] = Field("morning", description="Preferred time of day")
    mood: Optional[str] = Field(None, description="Current mood if provided")

class DailyPlanItem(BaseModel):
    activity: str
    duration_minutes: int
    description: str
    category: str

class DailyPlanResponse(BaseModel):
    user_id: str
    plan_date: str
    motivational_message: str
    daily_items: List[DailyPlanItem]
    estimated_total_time: int
    follow_up_time: str

# Error Response Schema
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
