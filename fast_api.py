import fastapi
import uvicorn
from pydantic import BaseModel
import joblib

app = fastapi.FastAPI()

model = joblib.load("regression.joblib")

@app.get("/")
def read_root():
    return {"message": "Bonjour, lancez le serveur avec : 'uvicorn fast_api:app --reload'"}

@app.post("/predict")
def predict(size: int, nb_rooms: int, garden: bool):
    garden_int = 1 if garden else 0
    y_pred = model.predict([[size, nb_rooms, garden_int]])
    return {"y_pred": float(y_pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6670)
