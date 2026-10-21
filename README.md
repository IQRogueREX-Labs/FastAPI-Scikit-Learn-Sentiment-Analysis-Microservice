# 📊 REX-Sentiment-Engine | Real-Time Sentiment Inference Microservice

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License MIT">
  <img src="https://img.shields.io/badge/Deploy-Streamlit%20Cloud%20%7C%20Docker%20%7C%20AWS-blue?style=for-the-badge" alt="Deploy">
</p>

---

## 🚀 Overview

**REX-Sentiment-Engine** is a lightweight, production-ready microservice that delivers real-time binary sentiment classification (Positive/Negative) via a RESTful API. Built on **FastAPI** for asynchronous high-throughput serving and **scikit-learn** for zero-dependency ML inference, it eliminates the overhead of heavy deep-learning frameworks while maintaining sub-millisecond latency. The service initializes an in-memory TF-IDF vectorizer (unigrams + bigrams) coupled with a Logistic Regression classifier trained on a curated seed corpus, making it ideal for edge deployments, CI/CD pipeline quality gates, or rapid prototyping where GPU resources are unavailable.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML-Powered Inference** | TF-IDF (1-2 grams) + Logistic Regression for instant sentiment scoring |
| 🎛️ **RESTful API** | `/predict` (POST) & `/health` (GET) with auto-generated OpenAPI/Swagger docs |
| 📊 **Confidence Calibration** | Returns probability scores rounded to 4 decimals for threshold tuning |
| ⭐ **Pydantic Validation** | Strict input schemas with `min_length=1` and example payloads |
| 📈 **Observability Ready** | Structured JSON responses including `tokens_analyzed` for monitoring |
| 🔍 **Test Coverage** | Pytest suite covering health checks, positive & negative prediction paths |
| 🎨 **Zero-Config Deploy** | Single-file `main.py` runs via `uvicorn` or Docker with no external model artifacts |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **API Framework** | FastAPI 0.110+ |
| **ASGI Server** | Uvicorn 0.28+ |
| **ML Pipeline** | scikit-learn 1.4+ (TF-IDF, LogisticRegression) |
| **Data Handling** | NumPy 1.26+, Pydantic 2.6+ |
| **Testing** | Pytest, FastAPI TestClient |
| **Containerization** | Docker (multi-stage) |

---

## 📂 Project Structure

```bash
FastAPI-Scikit-Learn-Sentiment-Analysis-Microservice/
├── LICENSE
├── main.py                 # FastAPI app + ML pipeline + entrypoint
├── requirements.txt        # Pinned dependencies
├── test_service.py         # Pytest suite (health + prediction)
└── auto_readme.py          # (Internal) README generator script
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/ChinmayVChatradamath/FastAPI-Scikit-Learn-Sentiment-Analysis-Microservice.git
cd FastAPI-Scikit-Learn-Sentiment-Analysis-Microservice

# 2. Create & activate virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
# Development mode with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production mode (single worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**API Docs:** `http://localhost:8000/docs` (Swagger UI)  
**Health Check:** `curl http://localhost:8000/health`  
**Predict:** `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "The service was fast and super reliable!"}'`

---

## 📊 Core Architecture / Model Details

| Component | Specification |
|-----------|---------------|
| **Vectorizer** | `TfidfVectorizer(ngram_range=(1,2), stop_words="english")` |
| **Classifier** | `LogisticRegression(max_iter=100, solver="lbfgs")` |
| **Training Corpus** | 6 samples (3 positive, 3 negative) — in-memory, no persistence |
| **Labels** | `1 = POSITIVE`, `0 = NEGATIVE` |
| **Output Schema** | `SentimentResponse(sentiment: str, confidence: float, tokens_analyzed: int)` |

> **Note:** The model trains on startup in <50 ms. For production, persist `vectorizer` and `classifier` via `joblib` and load at runtime to support larger corpora.

---

## 🧠 How It Works

1. **Startup** — FastAPI lifespan initializes `TfidfVectorizer` and fits `LogisticRegression` on the hardcoded seed corpus.
2. **Request Ingestion** — Client POSTs JSON `{ "text": "..." }` to `/predict`; Pydantic validates `min_length=1`.
3. **Vectorization** — Input text transformed via fitted TF-IDF vocabulary (unigrams + bigrams).
4. **Inference** — Logistic Regression outputs class label + class probabilities.
5. **Confidence Extraction** — `max(probabilities)` → `confidence` (float, 4 decimals).
6. **Response Assembly** — Returns `{ "sentiment": "POSITIVE|NEGATIVE", "confidence": 0.XXXX, "tokens_analyzed": N }`.
7. **Health Endpoint** — `/health` returns service metadata for load balancer probes.

---

## 🎛️ Controls / API Reference

### `GET /health`
```json
{
  "status": "healthy",
  "service": "REX-Sentiment-Engine",
  "version": "1.0.0"
}
```

### `POST /predict`
**Request**
```json
{ "text": "Absolutely love the new dashboard!" }
```

**Response (200)**
```json
{
  "sentiment": "POSITIVE",
  "confidence": 0.9876,
  "tokens_analyzed": 6
}
```

**Error (500)**
```json
{ "detail": "Internal server error message" }
```

---

## 🚀 Deployment

### Docker (Multi-Stage)
```dockerfile
# Dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY main.py .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
docker build -t rex-sentiment-engine .
docker run -d -p 8000:8000 rex-sentiment-engine
```

### Streamlit Cloud / Render / Fly.io
- Push to GitHub → Connect repo → Set **Build Command**: `pip install -r requirements.txt`  
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### AWS (ECS Fargate / Lambda + API Gateway)
- Package as Docker image → Push to ECR → Define Task Definition (CPU: 256, Memory: 512 MB)  
- Attach Application Load Balancer with `/health` target group.

---

## 🔮 Future Improvements

- [ ] **Model Persistence** — Serialize vectorizer & classifier with `joblib` for larger datasets
- [ ] **Multi-Class Support** — Extend to Neutral / Mixed sentiments with expanded corpus
- [ ] **Batch Endpoint** — `/predict/batch` accepting `List[str]` for throughput optimization
- [ ] **Prometheus Metrics** — `/metrics` exposing latency histograms & prediction counters
- [ ] **CI/CD Pipeline** — GitHub Actions: lint → test → docker build → deploy to staging
- [ ] **Input Sanitization** — HTML/Unicode normalization & length caps for DoS protection
- [ ] **A/B Testing Framework** — Canary routing between model versions via header-based switching

---

## 👨‍💻 Author
**Chinmay V Chatradamath**

---

## ⭐ Support

If this project helps you, please consider giving it a ⭐ on GitHub — it fuels future enhancements!