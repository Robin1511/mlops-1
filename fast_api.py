import fastapi
import uvicorn
from pydantic import BaseModel
import joblib
from transformers import pipeline

app = fastapi.FastAPI()

model = joblib.load("regression.joblib")
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Level 4 - BERT + Regression API", "version": "2.0", "endpoints": ["/predict", "/sentiment"]}

@app.post("/predict")
def predict(size: int, nb_rooms: int, garden: bool):
    garden_int = 1 if garden else 0
    y_pred = model.predict([[size, nb_rooms, garden_int]])
    return {"y_pred": float(y_pred[0])}

@app.post("/sentiment")
def predict_sentiment(input_data: TextInput):
    result = sentiment_model(input_data.text)
    return {"sentiment": result[0]["label"], "confidence": result[0]["score"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6670)
