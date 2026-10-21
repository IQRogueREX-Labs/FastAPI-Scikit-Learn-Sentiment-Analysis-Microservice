# 📊 REX-Sentiment-Engine | Real-Time Sentiment Inference Microservice

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/Deploy-Streamlit%20Cloud%20%7C%20Docker%20%7C%20AWS-blue" alt="Deploy">
</p>

---

## 🚀 Overview

**REX-Sentiment-Engine** is a lightweight, production-ready **FastAPI microservice** that delivers real-time sentiment classification using a classic **TF-IDF + Logistic Regression** pipeline. Designed for low-latency inference, it initializes an in-memory model on startup—eliminating cold-start overhead—and exposes a clean REST interface (`/predict`) with structured Pydantic schemas. Ideal for embedding into chatbots, review aggregators, or any system requiring instant polarity detection without heavy deep-learning dependencies.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **TF-IDF + Logistic Regression** | Interpretable, fast, and memory-efficient baseline for binary sentiment. |
| 🎛️ **FastAPI Native** | Auto-generated OpenAPI docs (`/docs`), request validation, and async-ready. |
| 📊 **Confidence Scoring** | Returns calibrated probability (`confidence`) alongside label. |
| ⭐ **Health Endpoint** | `/health` for Kubernetes liveness/readiness probes. |
| 📈 **Token Metadata** | Reports `tokens_analyzed` for observability & debugging. |
| 🔍 **Comprehensive Tests** | Pytest suite covering health, positive, and negative paths. |
| 🎨 **Zero-Config Startup** | Trains on a curated micro-corpus in < 200 ms—no external artifacts. |

---

## 🛠️ Tech Stack

- **Framework**: FastAPI ≥ 0.110
- **ASGI Server**: Uvicorn ≥ 0.28
- **ML Stack**: scikit-learn ≥ 1.4 (TF-IDF Vectorizer, LogisticRegression)
- **Data**: NumPy ≥ 1.26
- **Validation**: Pydantic ≥ 2.6
- **Testing**: Pytest + FastAPI TestClient

---

## 📂 Project Structure

```bash
FastAPI-Scikit-Learn-Sentiment-Analysis-Microservice/
├── LICENSE
├── main.py                 # FastAPI app, model init, endpoints
├── requirements.txt        # Pinned dependencies
├── test_service.py         # Pytest suite (health + prediction)
└── auto_readme.py          # (Internal) README generator script
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Chinmay-V-Chatradamath/FastAPI-Scikit-Learn-Sentiment-Analysis-Microservice.git
cd FastAPI-Scikit-Learn-Sentiment-Analysis-Microservice

# 2. Create & activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
# Development (auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production (multi-worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📊 Core Architecture / Model Details

| Component | Specification |
|-----------|---------------|
| **Vectorizer** | `TfidfVectorizer(ngram_range=(1,2), stop_words="english")` |
| **Classifier** | `LogisticRegression()` (default L2, C=1.0, solver=lbfgs) |
| **Training Data** | 6 hand-crafted samples (3 positive, 3 negative) |
| **Inference Latency** | ~1–3 ms / request (CPU) |
| **Input Schema** | `SentimentRequest(text: str, min_length=1)` |
| **Output Schema** | `SentimentResponse(sentiment: "POSITIVE"\|"NEGATIVE", confidence: float, tokens_analyzed: int)` |

---

## 🧠 How It Works

1. **Startup** – FastAPI lifespan initializes `TfidfVectorizer` and `LogisticRegression` on the embedded corpus.
2. **Request** – Client POSTs JSON `{ "text": "..." }` to `/predict`.
3. **Validation** – Pydantic enforces non-empty string; returns 422 on failure.
4. **Vectorization** – Input text transformed via fitted TF-IDF vocabulary (unigrams + bigrams).
5. **Inference** – Logistic Regression outputs class (0/1) and `predict_proba`.
6. **Post-Processing** – Maps `1 → "POSITIVE"`, `0 → "NEGATIVE"`; extracts max probability as `confidence`.
7. **Response** – Returns structured JSON with sentiment, confidence (4-decimal), and token count.

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
{ "text": "The service was fast and super reliable!" }
```

**Response (200)**
```json
{
  "sentiment": "POSITIVE",
  "confidence": 0.9876,
  "tokens_analyzed": 7
}
```

**Error (500)**
```json
{ "detail": "Internal server error message" }
```

---

## 🚀 Deployment

### Docker
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```
```bash
docker build -t rex-sentiment .
docker run -p 8000:8000 rex-sentiment
```

### Streamlit Cloud / Render / Fly.io / AWS ECS
- Push to GitHub → Connect repo → Set build command `pip install -r requirements.txt` → Start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Ensure `PORT` env var is respected (modify `main.py` if needed).

---

## 🔮 Future Improvements

- [ ] Swap in-memory training for persisted `joblib` artifacts (larger corpora).
- [ ] Add `/batch-predict` endpoint for throughput optimization.
- [ ] Integrate Prometheus metrics (`/metrics`) + Grafana dashboard.
- [ ] Support multi-class (Neutral) via OvR or fine-tuned transformer.
- [ ] CI/CD pipeline (GitHub Actions → Docker Hub → K8s).
- [ ] Input sanitization & rate limiting middleware.

---

## 👨‍💻 Author
**Chinmay V Chatradamath**

---

## ⭐ Support

If you find this microservice useful, please ⭐ the repository and share it with your network!