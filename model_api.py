import dill
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from pipe.src.future_sales_inno_ds_project.config import FEATURE_COLS
from pipe.src.future_sales_inno_ds_project.data_preparation.data_preprocessor import DataPreprocessor
from pipe.src.future_sales_inno_ds_project.feature_engineering.feature_engineer import FeatureEngineer
from pipe.src.future_sales_inno_ds_project.data_preparation.data_loader import DataLoader

# Initialize FastAPI app
app = FastAPI()

# Load trained model and metadata
try:
    with open('models/sales_model.pkl', 'rb') as file:
        model = dill.load(file)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load reference metadata for feature engineering
try:
    loader = DataLoader()
    items = loader.load_items()
    categories = loader.load_categories()
    shops = loader.load_shops()
except Exception as e:
    raise RuntimeError(f"Failed to load metadata: {e}")

# Define input schema for prediction
class Form(BaseModel):
    ID: int
    date_block_num: int
    shop_id: int
    item_id: int
    item_cnt_month: Optional[float] = Field(default=None, description="True label (optional, used for output)")

# Define output schema for prediction response
class Prediction(BaseModel):
    ID: int
    item_cnt_month: Optional[float]
    pred_value: float

# Health check endpoint
@app.get('/status')
def status():
    return "I'm OK"

# Return model metadata
@app.get('/version')
def version():
    return model.get('metadata', 'No metadata available')

# Provide model description
@app.get('/info')
def info():
    return (
        "The model trained on the Predict Future Sales dataset from Kaggle is designed to forecast "
        "monthly item sales for individual stores. It leverages historical transactional data, item "
        "and shop metadata, and temporal patterns to predict future demand."
    )

# Prediction endpoint
@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    try:
        # Convert input to DataFrame, excluding unset optional fields
        input_data = form.dict(exclude_unset=True)
        input_df = pd.DataFrame([input_data])

        # Ensure item_cnt_month column exists (even if it's NaN)
        if 'item_cnt_month' not in input_df.columns:
            input_df['item_cnt_month'] = pd.NA

        # Apply feature engineering
        enriched_df = FeatureEngineer.add_features(
            data=input_df,
            items=items,
            categories=categories,
            shops=shops
        )

        # Cast types
        enriched_df = DataPreprocessor.cast_types(enriched_df)

        # Check for missing values in required features
        missing_features = enriched_df[FEATURE_COLS].isnull().sum()
        if missing_features.any():
            raise HTTPException(
                status_code=422,
                detail=f"Missing values in features: {missing_features[missing_features > 0].to_dict()}"
            )

        # Select features
        features = enriched_df[FEATURE_COLS]

        # Run prediction
        y_pred = model['model'].predict(features)

        # Return formatted response
        return Prediction(
            ID=form.ID,
            item_cnt_month=form.item_cnt_month,
            pred_value=float(y_pred[0])
        )

    except HTTPException as http_exc:
        raise http_exc
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required feature: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e) or 'No additional details'}")
