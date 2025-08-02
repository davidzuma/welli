# Welli - Digital Wellness Assistant

AI-powered retention engine for digital wellness apps. Provides personalized content matching, user clustering, churn prediction, and daily wellness planning.

## Architecture

- FastAPI backend with modular ML components:
  - Goal-to-content matching (FAISS + OpenAI embeddings)
  - Behavioral clustering (KMeans)
  - Churn prediction (Logistic Regression)
  - Micro-coach (OpenAI GPT-4o-mini)

## Quick Start

### Prerequisites

- Python 3.11+ (for local development) OR Docker
- OpenAI API key

### Installation

```bash
git clone https://github.com/davidzuma/welli.git
cd welli
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

### Running

#### Option 1: Docker (Recommended)

```bash
# 1. Clone and setup
git clone https://github.com/davidzuma/welli.git
cd welli
cp .env.example .env
# Add your OpenAI API key to .env file

# 2. Build and run with Docker
docker build -t welli-api .
docker run -d -p 8000:8000 --env-file .env --name welli-api welli-api

# 3. Check if it's running
docker ps
docker logs welli-api

# 4. Stop when done
docker stop welli-api && docker rm welli-api
```

#### Option 2: Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access the API:**

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API Endpoints

All endpoints return JSON responses. See interactive documentation at `/docs` for detailed schemas.

### Core Endpoints

- **`POST /api/v1/match-goal`**: Match user goals to relevant content using AI embeddings
- **`POST /api/v1/cluster-user`**: Classify users into behavioral segments
- **`POST /api/v1/predict-churn`**: Predict user churn risk and recommend interventions
- **`POST /api/v1/daily-plan`**: Generate personalized daily wellness plans

### Health & Info

- **`GET /`**: API information and available endpoints
- **`GET /health`**: Service health check
- **`GET /docs`**: Interactive API documentation (Swagger UI)

### Example Usage

```bash
# Match a goal to content
curl -X POST http://localhost:8000/api/v1/match-goal \
  -H "Content-Type: application/json" \
  -d '{"goal": "I want to reduce stress and anxiety", "limit": 3}'

# Predict churn risk
curl -X POST http://localhost:8000/api/v1/predict-churn \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "days_since_signup": 30,
    "total_sessions": 25,
    "avg_session_duration": 15.5,
    "last_login_days_ago": 2,
    "content_engagement_rate": 0.8,
    "notification_response_rate": 0.6,
    "streak_length": 7,
    "content_completion_rate": 0.75,
    "goal_progress_percentage": 65.0
  }'
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
LOG_LEVEL=INFO
MODEL_BASE_PATH=data/
```

### Docker Configuration

The Docker container:

- Uses Python 3.11-slim base image
- Installs all dependencies automatically
- Loads ML models and data files
- Exposes port 8000
- Reads environment variables from `.env` file

## Project Structure

```
welli/
├── src/
│   ├── api/           # FastAPI routes and schemas
│   ├── models/        # ML models (content matching, clustering, churn, coaching)
│   ├── utils/         # Utilities (feature prep, model loading)
│   └── main.py        # FastAPI application entry point
├── data/              # ML model files and content catalog
├── ml_models/         # Trained model artifacts
├── notebooks/         # Jupyter notebooks for development
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker container configuration
├── .env.example       # Environment variables template
└── README.md
```

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run with reload for development
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Development

```bash
# Build and run for development
docker build -t welli-api .
docker run -p 8000:8000 --env-file .env -v $(pwd):/app welli-api

# View logs
docker logs -f welli-api
```

## Production Deployment

### Docker Production

```bash
# Build production image
docker build -t welli-api:latest .

# Run in production mode
docker run -d \
  --name welli-api \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  welli-api:latest

# Health check
curl http://localhost:8000/health
```

### Docker Compose (Optional)

Create a `docker-compose.yml` for easier management:

```yaml
version: '3.8'
services:
  welli-api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with: `docker-compose up -d`

## License

MIT License. See [LICENSE](LICENSE).
