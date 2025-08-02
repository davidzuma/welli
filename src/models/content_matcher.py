"""
Content matching using FAISS vector database with OpenAI embeddings for goal-to-content semantic matching
"""

import numpy as np
from typing import List, Dict, Any
import logging
import os
import json
from dotenv import load_dotenv
from src.utils.model_loader import ModelLoader, ensure_dummy_data

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class ContentMatcher:
    """
    Simple content matcher using OpenAI text-embedding-3-small + FAISS for semantic search
    """
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.openai_client = None
        self.faiss_index = None
        self.content_data = []
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
        self.embedding_model = "text-embedding-3-small"  # Can be changed to text-embedding-3-large
        
        # Ensure dummy data exists
        ensure_dummy_data()
        
        # Initialize everything
        self._setup_openai()
        self._load_content_and_build_index()
    
    def _setup_openai(self):
        """Setup OpenAI client - required, no fallback"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required but not found in environment variables")
            
            # Remove quotes if present
            api_key = api_key.strip("'\"")
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _setup_faiss_index(self):
        """Setup FAISS index - required, no fallback"""
        try:
            import faiss
            
            # Use L2 distance (Euclidean) - works well for normalized embeddings
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            logger.info(f"FAISS index created with dimension {self.embedding_dimension}")
            
        except ImportError:
            raise ImportError("FAISS library is required. Install with: pip install faiss-cpu")
        except Exception as e:
            raise Exception(f"Failed to setup FAISS index: {str(e)}")
    
    
    def _load_content_and_build_index(self):
        """Load content catalog and build FAISS index with embeddings"""
        # Load content catalog
        content_catalog = self.model_loader.load_json_data("content_catalog.json")
        if not content_catalog:
            raise Exception("Failed to load content catalog")
        
        self.content_data = content_catalog["content"]
        logger.info(f"Loaded {len(self.content_data)} content items")
        
        # Setup FAISS and build embeddings index
        self._setup_faiss_index()
        self._build_embeddings_index()
    
    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {str(e)}")
            raise
    
    def _build_embeddings_index(self):
        """Build FAISS index with content embeddings"""
        if not self.faiss_index or not self.content_data:
            raise Exception("FAISS index or content data not available")
        
        try:
            # Prepare texts for embedding
            content_texts = []
            for item in self.content_data:
                # Combine title, description, and tags for embedding
                text = f"{item['title']} {item['description']} {' '.join(item.get('tags', []))}"
                content_texts.append(text)
            
            # Get embeddings from OpenAI
            embeddings = self._get_openai_embeddings(content_texts)
            
            if embeddings:
                # Convert to numpy array and add to FAISS index
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.faiss_index.add(embeddings_array)
                logger.info(f"Built FAISS index with {len(embeddings)} content embeddings")
            else:
                raise Exception("Failed to get embeddings from OpenAI")
                
        except Exception as e:
            logger.error(f"Error building embeddings index: {str(e)}")
            raise
    
    def match_goal_to_content(self, goal: str, limit: int = 5) -> Dict[str, Any]:
        """
        Match user goal to semantically similar content using FAISS and OpenAI embeddings
        
        Args:
            goal: User's wellness goal in natural language
            limit: Number of content items to return
            
        Returns:
            Dictionary with matched content and metadata
        """
        if not self.content_data:
            raise Exception("No content data available")
        
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
            
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            raise Exception("FAISS index not ready")
        
        try:
            # Get goal embedding
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[goal]
            )
            goal_embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(
                goal_embedding.reshape(1, -1), 
                min(limit, self.faiss_index.ntotal)
            )
            
            # Convert distances to similarities (lower distance = higher similarity)
            matched_content = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.content_data):
                    item = self.content_data[idx]
                    # Convert L2 distance to similarity score (0-1 range)
                    similarity = 1.0 / (1.0 + distance)
                    
                    matched_content.append({
                        "id": item["id"],
                        "title": item["title"],
                        "description": item["description"],
                        "category": item["category"],
                        "similarity_score": float(similarity)
                    })
            
            logger.info(f"Found {len(matched_content)} matches using FAISS + OpenAI")
            
            return {
                "user_goal": goal,
                "matched_content": matched_content,
                "total_results": len(matched_content)
            }
            
        except Exception as e:
            logger.error(f"Error in goal matching: {str(e)}")
            raise Exception(f"Goal matching failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if the content matcher is ready to use"""
        return (len(self.content_data) > 0 and 
                self.openai_client is not None and
                self.faiss_index is not None and 
                self.faiss_index.ntotal > 0)
    
    def get_index_status(self) -> Dict[str, Any]:
        """Get status information about the FAISS index and embeddings"""
        return {
            "content_items_loaded": len(self.content_data),
            "openai_client_ready": self.openai_client is not None,
            "faiss_index_ready": self.faiss_index is not None,
            "embeddings_count": self.faiss_index.ntotal if self.faiss_index else 0,
            "fully_ready": self.is_ready()
        }
