import fastapi
import uvicorn
from pydantic import BaseModel
import joblib

app = fastapi.FastAPI()

model = joblib.load("regression.joblib")

@app.get("/")
def read_root():
    return {"message": "Bonjour, lancez le serveur avec : 'uvicorn fast_api:app --reload'"}


class HouseFeatures(BaseModel):
    size: int
    nb_rooms: int
    garden: bool

@app.post("/predict")
def predict(features: HouseFeatures):
    garden_int = 1 if features.garden else 0
    y_pred = model.predict([[features.size, features.nb_rooms, garden_int]])
    return {"y_pred": float(y_pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
