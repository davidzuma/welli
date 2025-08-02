"""
Micro-coach using OpenAI's GPT-4o-mini for daily wellness planning - requires OpenAI API key
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MicroCoach:
    """
    AI-powered micro-coach for generating personalized daily wellness plans
    """
    
    def __init__(self):
        self.openai_client = None
        self.model_name = "gpt-4o-mini"
        
        # Initialize OpenAI client
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client - strict mode, no fallbacks"""
        try:
            import openai
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required but not found in environment variables")
            
            # Remove quotes if present
            api_key = api_key.strip("'\"")
            
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the micro-coach"""
        return """You are a supportive wellness micro-coach. Your role is to create personalized, 
        achievable daily wellness plans that help users build sustainable habits.

        Guidelines:
        - Keep plans simple with 1-2 activities maximum
        - Focus on small wins and building momentum
        - Be encouraging and supportive in tone
        - Consider the user's available time and current streak
        - Adapt recommendations based on recent activities
        - Include specific, actionable items with time estimates
        - Provide a motivational message
        
        Respond ONLY with valid JSON in this exact format:
        {
            "motivational_message": "Brief, encouraging message",
            "daily_items": [
                {
                    "activity": "Specific activity name",
                    "duration_minutes": 10,
                    "description": "Clear description of what to do",
                    "category": "meditation|exercise|nutrition|sleep|mental_health"
                }
            ],
            "follow_up_time": "evening"
        }"""
    
    def _create_user_prompt(self, user_data: Dict[str, Any]) -> str:
        """Create user-specific prompt"""
        goal = user_data.get("goal", "improve overall wellness")
        streak = user_data.get("current_streak", 0)
        recent_activities = user_data.get("recent_activities", [])
        available_time = user_data.get("available_time_minutes", 15)
        preferred_time = user_data.get("preferred_time", "morning")
        mood = user_data.get("mood")
        
        prompt = f"""Create a personalized wellness plan for today.

        User Context:
        - Goal: {goal}
        - Current streak: {streak} days
        - Recent activities: {', '.join(recent_activities) if recent_activities else 'None'}
        - Available time: {available_time} minutes
        - Preferred time: {preferred_time}"""
        
        if mood:
            prompt += f"\n- Current mood: {mood}"
        
        prompt += f"""
        
        Create a plan that:
        - Fits within {available_time} minutes
        - Builds on their {streak}-day streak
        - Complements recent activities: {recent_activities}
        - Aligns with their goal: {goal}
        """
        
        return prompt
    
    async def _call_openai(self, user_prompt: str) -> Dict[str, Any]:
        """Make async call to OpenAI API - strict mode"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse OpenAI response as JSON: {content}")
            raise Exception("Invalid JSON response from OpenAI")
    
    async def generate_daily_plan(self, user_data: Any) -> Dict[str, Any]:
        """
        Generate personalized daily wellness plan using OpenAI - strict mode
        
        Args:
            user_data: DailyPlanRequest object with user preferences and history
            
        Returns:
            Dictionary with daily plan and motivational content
        """
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
            
        # Convert Pydantic model to dict
        if hasattr(user_data, 'dict'):
            user_dict = user_data.dict()
        else:
            user_dict = user_data
        
        # Generate plan using OpenAI
        user_prompt = self._create_user_prompt(user_dict)
        plan_data = await self._call_openai(user_prompt)
        
        # Calculate total time
        total_time = sum(item.get("duration_minutes", 0) for item in plan_data.get("daily_items", []))
        
        # Add metadata
        today = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "user_id": user_dict["user_id"],
            "plan_date": today,
            "motivational_message": plan_data.get("motivational_message", "Have a great wellness day!"),
            "daily_items": plan_data.get("daily_items", []),
            "estimated_total_time": total_time,
            "follow_up_time": plan_data.get("follow_up_time", "evening")
        }
    
    def is_ready(self) -> bool:
        """Check if the micro-coach is ready to use"""
        return self.openai_client is not None
