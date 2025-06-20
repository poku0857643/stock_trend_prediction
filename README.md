
## 🧱 Component Architecture

```
        ┌────────────┐      REST API      ┌──────────────┐
        │  Frontend  ├───────────────────►│  trend-api   │
        └────────────┘                    └──────────────┘
              │                                │
              │ REST API                       │ calls model + scaler
              ▼                                ▼
        ┌────────────┐      REST API      ┌──────────────┐
        │  strategy  ├───────────────────►│  strategy-api│
        │   UI       │                    └──────────────┘
        └────────────┘                          │
              ▲                                 ▼
              │                    ┌──────────────────────┐
              └───────────────────►│ Embedding + Generator│
                                   └──────────────────────┘
                                                │
                                                ▼
                                   ┌──────────────────────┐
                                   │   Feedback Logging   │
                                   └──────────────────────┘
                                                │
                                                ▼
                                   ┌──────────────────────┐
                                   │     Feedback DB      │
                                   └──────────────────────┘
```

- `frontend`: Interactive web UI
- `trend-api`: Loads LSTM model, performs predictions
- `strategy-api`: Uses trend predictions + LLM to generate strategy
- `shared/`: Handles embedding extraction and reusable utilities



# Strategy Generation and Feedback System
This subsystem complements the trend-prediction pipeline by generating actionable investment strategies and closing the loop with feedback-driven optimization.

### 🔹 Key Components
| Component                     | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| **FastAPI Gateway**          | Routes requests between frontend and microservices (strategy, feedback, DB) |
| **Strategy Generation Service** | Uses LLMs and market context to propose tailored strategies                 |
| **ML Inference Engine**      | Predictive backend using LSTM and transformer models                        |
| **Feedback Logging Service** | Captures user feedback, ratings, and logs it in structured format           |
| **Strategy Database**        | Stores generated strategies, prompts, results, and metadata                 |
| **Feedback Database**        | Logs qualitative/quantitative feedback for long-term improvement            |
| **Monitoring Agent**         | Collects metrics and logs for Prometheus/Grafana dashboard                  |

# 📊 Trend Prediction & Strategy Generator System

This project is a modular, containerized system that provides:
- 📈 **Trend Prediction API**: Predicts stock price trends using an LSTM-based model.
- 🤖 **Strategy Generation API**: Leverages document embeddings and LLMs (e.g., OpenAI) to generate investment strategies.
- 🌐 **Frontend UI**: Elegant HTML/CSS/JS interface for interacting with both APIs.

---

## 🔧 Architecture Overview

```
TrendStrategySystem/
├── trend-api/              # LSTM prediction service
├── strategy-api/           # LLM-based strategy generation
├── shared/                 # Common components (e.g., extractors, config)
├── frontend/               # Web interface
├── db/                     # Database init files
├── .github/workflows/      # CI/CD pipelines
├── docker-compose.yml      # For local orchestration
└── .env                    # Environment variables
```

---

## 🚀 Local Development

### 1. Set up `.env`
```
DB_USER=postgres
DB_PASS=secret
DB_NAME=trends
DB_HOST=cloudsql
OPENAI_API_KEY=your-openai-key
```

### 2. Run with Docker Compose
```bash
docker-compose up --build
```

- `http://localhost:8000/trend_predict` → Trend API health check
- `http://localhost:8000/generate_strategies` → Strategy generation endpoint
- `http://localhost:8000/` → Frontend UI

---

## 📈 Trend Prediction API

### Endpoints
- `GET /trend_predict?ticker=AAPL`
- `POST /trend_predict_batch`
- `GET /health` → Checks trend model, PDF loader, and cache

### Example Output
```json
{
  "ticker": "AAPL",
  "predicted_next_close": 189.21,
  "confidence": 0.85
}
```

---

## 🧠 Strategy Generator API

### Input (POST `/generate_strategies`)
```json
{
  "local_folder": "path/to/pdfs",
  "online_folder": null,
  "cloud_folder": null,
  "strategy_prompt": "What should our investment strategy be for Q3?",
  "tickers": ["AAPL"],
  "use_internal_trend_api": true,
  "trend_prediction": 189.21
}
```

### Output
Returns structured strategy suggestions, prompt metadata, and source stats.

---

## 💻 Frontend

A polished HTML/JS interface is provided at:
```
frontend/
├── index.html
├── style.css
└── app.js
```
Interacts with both APIs, styled like Apple/Google interfaces.

---

## ☁️ Cloud SQL Integration

- GCP Cloud SQL connected via proxy in Docker Compose
- Mount `/cloudsql` for Cloud Run/GKE deployment

---

## ⚙️ CI/CD

- GitHub Actions build Docker images and push to container registry
- Runs unit tests and handles GCP deployment jobs

---

## 🛠️ Future Enhancements

- Feedback loop for strategy effectiveness
- User login + strategy history
- API rate limiting and caching
- Prometheus/Grafana monitoring

---

## 🧪 Testing

Unit tests can be placed under each service:
```
trend-api/tests/
strategy-api/tests/
```