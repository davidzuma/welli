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
- Python 3.8+
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
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

## API Endpoints

- `/api/v1/match-goal`: Match user goals to content
- `/api/v1/cluster-user`: Cluster users by behavior
- `/api/v1/predict-churn`: Predict churn risk
- `/api/v1/daily-plan`: Generate daily wellness plan

## Configuration

Set environment variables in `.env`:
```bash
OPENAI_API_KEY=your_api_key_here
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Project Structure

```
welli/
├── src/
│   ├── api/
│   ├── models/
│   ├── utils/
│   └── main.py
├── data/
├── requirements.txt
```

## License

MIT License. See [LICENSE](LICENSE).
