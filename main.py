from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI(
    title="REX-Sentiment-Engine",
    description="Real-time sentiment inference service powered by TF-IDF and Logistic Regression.",
    version="1.0.0"
)

# Training corpus for quick in-memory initialization
X_train = [
    "I absolutely love this product, it works wonders!",
    "Fantastic experience, highly recommended to everyone.",
    "Great customer support and lightning fast delivery.",
    "Terrible build quality, it broke within two days.",
    "Worst purchase I have ever made, complete waste of money.",
    "Very disappointed with the performance, would not recommend."
]
y_train = [1, 1, 1, 0, 0, 0]  # 1: Positive, 0: Negative

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, example="The service was fast and super reliable!")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    tokens_analyzed: int

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "REX-Sentiment-Engine", "version": "1.0.0"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(payload: SentimentRequest):
    try:
        vec = vectorizer.transform([payload.text])
        prediction = classifier.predict(vec)[0]
        probabilities = classifier.predict_proba(vec)[0]
        confidence = float(np.max(probabilities))
        
        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        token_count = len(payload.text.split())
        
        return SentimentResponse(
            sentiment=label,
            confidence=round(confidence, 4),
            tokens_analyzed=token_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
