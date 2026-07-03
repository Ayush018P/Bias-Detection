from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import sys
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

# Add the project root to sys.path so we can import internal modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.preprocessor import preprocess_data
from models.trainer import train_and_evaluate
from bias.detector import discover_subgroups

app = FastAPI(title="SABPF API", version="1.0")

# Enable CORS for the future React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# GLOBAL STATE (Temporary for transition from Streamlit)
# ---------------------------------------------------------
# In a real SaaS product, you would use PostgreSQL to save the user's config,
# and S3 or local disk to save the uploaded CSV.
app_state = {
    "raw_data": None,
    "pp_data": None, # (X_tr, X_te, y_tr, y_te, X_proc)
    "models": {},
    "eval_results": {},
    "bias_results": None,
}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "SABPF API is running!"}

@app.post("/api/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    prot_cols: str = Form(...)  # Expected as a comma-separated string
):
    """
    Endpoint to upload a CSV file and store it in the API's memory.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in CSV.")
            
        prot_cols_list = [c.strip() for c in prot_cols.split(",")]
        for col in prot_cols_list:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Protected column '{col}' not found in CSV.")

        # Binarize target if necessary
        y_raw = df[target_col]
        if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
            y = (y_raw == y_raw.unique()[0]).astype(int)
        else: 
            y = y_raw.astype(int)
            
        X = df.drop(columns=[target_col])
        
        # Save to global state
        app_state["raw_data"] = {
            "X": X.reset_index(drop=True),
            "y": y.reset_index(drop=True),
            "protected_cols": prot_cols_list,
            "feature_names": list(X.columns),
            "dataset_name": file.filename,
            "task_type": "binary_classification"
        }
        
        return {
            "message": "Dataset uploaded successfully",
            "rows": len(df),
            "features": len(X.columns),
            "protected_attributes": len(prot_cols_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
def preprocess():
    """
    Preprocesses the data currently stored in memory.
    """
    raw_data = app_state.get("raw_data")
    if not raw_data:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
        
    try:
        X_tr, X_te, y_tr, y_te, X_proc = preprocess_data(raw_data['X'], raw_data['y'])
        app_state["pp_data"] = (X_tr, X_te, y_tr, y_te, X_proc)
        
        return {"message": "Data preprocessed successfully", "train_size": len(X_tr), "test_size": len(X_te)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TrainRequest(BaseModel):
    models: List[str] # e.g. ["Logistic Regression", "XGBoost"]

@app.post("/api/train-models")
def train_models(request: TrainRequest):
    """
    Trains the specified models on the preprocessed data.
    """
    if not app_state.get("pp_data"):
        raise HTTPException(status_code=400, detail="Data not preprocessed yet.")
        
    X_tr, X_te, y_tr, y_te, X_proc = app_state["pp_data"]
    dataset_name = app_state["raw_data"]["dataset_name"]
    
    try:
        models, eval_objs = train_and_evaluate(dataset_name, X_tr, X_te, y_tr, y_te, request.models)
        app_state["models"] = models
        app_state["eval_results"] = eval_objs
        
        # Prepare a clean response without non-serializable objects (like scikit-learn models)
        summary = {name: metrics["Accuracy"] for name, metrics in eval_objs.items()}
        return {"message": "Models trained successfully", "accuracies": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AuditRequest(BaseModel):
    model_name: str
    threshold_dpd: float = 0.10

@app.post("/api/run-audit")
def run_audit(request: AuditRequest):
    """
    Runs the bias detection audit using a trained model.
    """
    if not app_state.get("models") or request.model_name not in app_state["models"]:
        raise HTTPException(status_code=400, detail="Model not found or not trained.")
        
    X_tr, X_te, y_tr, y_te, X_proc = app_state["pp_data"]
    raw_data = app_state["raw_data"]
    model = app_state["models"][request.model_name]
    
    try:
        y_pred = model.predict(X_te)
        
        # Use raw unencoded X_te for interpretability where possible
        X_te_raw = raw_data['X'].loc[X_te.index]
        
        res_df = discover_subgroups(
            X_te_raw, y_te, y_pred, model, raw_data['protected_cols'], request.threshold_dpd
        )
        
        app_state["bias_results"] = res_df
        
        # Return as JSON
        if res_df.empty:
            return {"message": "Audit complete", "findings": []}
            
        # Convert DataFrame to list of dicts for JSON serialization
        # NaN values need to be handled, but Pandas to_dict('records') usually works ok
        # For strict JSON, we replace NaN with None
        clean_df = res_df.replace({pd.NA: None, float('nan'): None})
        return {"message": "Audit complete", "findings": clean_df.to_dict(orient="records")}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
