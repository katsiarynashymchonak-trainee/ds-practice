import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('../../models/sales_model.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    ID: int
    item_cnt_month: float


class Prediction(BaseModel):
    ID: int
    item_cnt_month: float
    pred_value: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.get('/info')
def info():
    return (
        "The model trained on the Predict Future Sales dataset from "
        "Kaggle is designed to forecast monthly item sales for individual stores. "
        "It leverages historical transactional data, item and shop metadata, and temporal "
        "patterns to predict future demand.")


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    form_dict = vars(form)
    df = pd.DataFrame([form_dict])
    y = model['model'].predict(df)

    return {
        'ID': form.ID,
        'item_cnt_month': form.item_cnt_month,
        'pred_value': y[0]
    }
