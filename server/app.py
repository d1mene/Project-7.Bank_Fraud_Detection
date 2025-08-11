from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
import datetime
from collections import deque

from pipe_layers import MissingValuesPreprocessor, BankMonthsInputer, FeatureEngineering

app = FastAPI(title="Live-Fraud-Monitor")
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="templates"), name="static")

with open("final_pipeline.pkl", "rb") as f:
    MODEL = pickle.load(f)

CLASSES = getattr(MODEL, "classes_", ["normal", "fraud"])
EVENTS: deque = deque(maxlen=100)

class Prediction(BaseModel):
    ts: str
    proba: float
    label: str

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Прогнозируем сразу для всего датафрейма
    proba_matrix = MODEL.predict_proba(df)
    labels = MODEL.classes_[proba_matrix.argmax(axis=1)]

    results = []
    for i, (label, proba_vec) in enumerate(zip(labels, proba_matrix)):
        idx = proba_vec.argmax()
        evt = Prediction(
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            proba=float(proba_vec[idx]),
            label=str(label)
        )
        EVENTS.append(evt)
        results.append(evt)

    return results

@app.get("/events")
async def events():
    return list(EVENTS)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)