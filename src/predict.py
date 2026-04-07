import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Path Setup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "compressed_xgb_model.joblib"
FEAT_PATH = BASE_DIR / "models" / "features.joblib"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"
REMEDY_PATH = BASE_DIR / "data" / "Final_Remedies.csv"
CRITICAL_PATH = BASE_DIR / "data" / "CRITICAL_DISEASES.txt"

# ---------------------------------------------------------
# 1. GLOBAL LOADING (Optimized for performance)
# ---------------------------------------------------------
CRITICAL_DISEASES = []

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    
    # Load Remedies CSV
    remedy_df = pd.read_csv(REMEDY_PATH)
    remedy_df.columns = remedy_df.columns.str.strip().str.lower()
    
    # Setup lookup keys
    disease_col = "diseases" if "diseases" in remedy_df.columns else "disease_name"
    remedy_df[disease_col] = remedy_df[disease_col].astype(str).str.lower().str.strip()
    remedy_df['match_key'] = remedy_df[disease_col].str.replace("_", " ")
    
    # Setup remedy column
    remedy_col = "ayurvedic_remedy" if "ayurvedic_remedy" in remedy_df.columns else remedy_df.columns[-1]

    # Load Critical List
    if CRITICAL_PATH.exists():
        with open(CRITICAL_PATH, "r") as f:
            CRITICAL_DISEASES = [line.strip().lower() for line in f if line.strip()]
    
    GLOBAL_LOAD_SUCCESS = True
except Exception as e:
    print(f"Startup Error: {e}")
    GLOBAL_LOAD_SUCCESS = False

# ---------------------------------------------------------
# 2. DIAGNOSIS FUNCTION
# ---------------------------------------------------------
def get_full_diagnosis(user_symptoms_dict):
    if not GLOBAL_LOAD_SUCCESS:
        return {"success": False, "Error": "Server configuration error."}

    # 1. Prepare Input & Track Severity
    input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features, dtype="float32")
    match_count = 0
    max_severity = 0 
    
    for symptom, level in user_symptoms_dict.items():
        clean_symptom = str(symptom).lower().strip()
        try:
            level = float(level)
            if level > 0 and clean_symptom in input_data.columns:
                input_data.loc[0, clean_symptom] = level
                match_count += 1
                if level > max_severity: max_severity = level
        except: continue

    if match_count == 0:
        return {"success": False, "Error": "No symptoms matched."}

    # 2. Prediction
    probs = model.predict_proba(input_data)[0]
    top_idx = np.argmax(probs)
    prediction = label_encoder.inverse_transform([top_idx])[0]
    norm_pred = prediction.lower().strip().replace("_", " ")

    # 3. TRIAGE LOGIC (Matching Image 1 & Image 2)
    # ---------------------------------------------------------
    severity_label = ""
    instruction = ""
    ayurvedic_remedy = ""

    # Check against the external Critical Diseases list
    is_critical = any(crit in norm_pred for crit in CRITICAL_DISEASES)

    # CASE 1: RED (Critical or Level 3)
    if is_critical or max_severity >= 3.0:
        severity_label = "🔴 Medical Emergency"
        instruction = "Stop! Do not self-treat. Consult a doctor immediately."
        ayurvedic_remedy = "Not applicable. Please seek immediate medical care."

    # CASE 2: YELLOW (Moderate / Level 2) -> Matches your 2nd Image
    elif max_severity >= 2.0:
        severity_label = "🟡 Ayurvedic Treatment"
        instruction = "Use these remedies to balance your Doshas."
        
        # CSV Lookup
        match = remedy_df[remedy_df['match_key'] == norm_pred]
        if not match.empty:
            ayurvedic_remedy = str(match.iloc[0][remedy_col])
        else:
            # Fallback if specific remedy isn't in CSV
            ayurvedic_remedy = "Consult an Ayurvedic specialist for a personalized plan."

    # CASE 3: GREEN (Mild / Level 1)
    else:
        severity_label = "🟢 Low / Mild"
        instruction = "Symptoms appear mild. Rest and hydration are advised."
        ayurvedic_remedy = "Standard herbal teas and rest are sufficient."

    # 4. FINAL OUTPUT FORMAT (Strictly following screenshot keys)
    return {
        "success": True,
        "prediction": {
            "disease": prediction,
            "severity_level": severity_label,
            "instruction": instruction,
            "ayurvedic_remedy": ayurvedic_remedy
        }
    }