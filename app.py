"""
FastAPI Backend — Dating App Match Outcome Predictor
Run with: uvicorn app:app --reload --port 8000
"""

import pickle
import __main__
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from custom_transformers import MultiTagBinarizer

__main__.MultiTagBinarizer = MultiTagBinarizer

# ── Load artifacts ─────────────────────────────────────────────────────────────
with open("preprocessing_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

with open("trained_models.pkl", "rb") as f:
    raw = pickle.load(f)

preprocessor  = art["preprocessor"]      # ColumnTransformer (fitted)
label_encoder = art["label_encoder"]     # LabelEncoder

# XGBoost uses full feature space (not PCA)
xgb_model = raw["models"]["XGBoost"]["model"]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Dating App Match Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema — all features the model expects ───────────────────────────
class PredictRequest(BaseModel):
    # Shown in UI form
    gender:              str   = Field("Male",        description="Gender")
    location_type:       str   = Field("Urban",       description="Location type")
    income_bracket:      str   = Field("Middle",      description="Income bracket")
    education_level:     str   = Field("Bachelor's",  description="Education level")
    app_usage_time_min:  int   = Field(60,    ge=0,   description="Daily app usage (minutes)")
    swipe_right_ratio:   float = Field(0.5,   ge=0, le=1, description="Swipe-right ratio (0–1)")
    profile_pics_count:  int   = Field(5,     ge=1,   description="Number of profile photos")
    bio_length:          int   = Field(150,   ge=0,   description="Bio character count")
    message_sent_count:  int   = Field(30,    ge=0,   description="Messages sent")
    swipe_time_of_day:   str   = Field("Evening",     description="Most active swipe period")

    # Hidden — filled with sensible defaults
    sexual_orientation:  str   = Field("Straight")
    interest_tags:       str   = Field("Music,Travel,Sports")
    likes_received:      int   = Field(50,    ge=0)
    mutual_matches:      int   = Field(10,    ge=0)
    emoji_usage_rate:    float = Field(0.3,   ge=0, le=1)
    last_active_hour:    int   = Field(20,    ge=0, le=23)


@app.get("/")
def root():
    return {"status": "ok", "message": "Dating App Match Predictor API"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.model_dump()])

        # Drop leaky label columns if accidentally present
        df = df.drop(columns=["app_usage_time_label", "swipe_right_label"],
                     errors="ignore")

        # Transform using fitted ColumnTransformer
        X = preprocessor.transform(df)

        # Predict with XGBoost
        pred_int = xgb_model.predict(X)
        label    = label_encoder.inverse_transform(pred_int)[0]

        # Probabilities for top-3
        proba      = xgb_model.predict_proba(X)[0]
        top3_idx   = np.argsort(proba)[::-1][:3]
        top3       = [
            {"outcome": label_encoder.inverse_transform([i])[0],
             "probability": round(float(proba[i]) * 100, 1)}
            for i in top3_idx
        ]

        return {
            "prediction": label,
            "confidence": round(float(proba[pred_int[0]]) * 100, 1),
            "top3": top3,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options")
def options():
    """Return valid dropdown values for the frontend form."""
    return {
        "gender":            ["Male", "Female", "Non-binary", "Other"],
        "location_type":     ["Urban", "Suburban", "Rural"],
        "income_bracket":    ["Very Low", "Low", "Lower-Middle", "Middle",
                              "Upper-Middle", "High", "Very High"],
        "education_level":   ["No Formal Education", "High School", "Diploma",
                              "Associate's", "Bachelor's", "MBA",
                              "Master's", "PhD", "Postdoc"],
        "swipe_time_of_day": ["Early Morning", "Morning", "Afternoon",
                              "Evening", "Night", "After Midnight"],
    }
