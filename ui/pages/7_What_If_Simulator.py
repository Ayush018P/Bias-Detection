import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar

st.set_page_config(page_title="What-If Simulator", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

st.header("🔮 What-If Simulator")
st.markdown("Instantly test how your model reacts to individual profiles. Flip a protected attribute (like Gender or Race) to see if the AI changes its mind!")

if "data" not in st.session_state or not st.session_state.data:
    st.warning("Please load a dataset from the sidebar to begin.")
    st.stop()

if 'models' not in st.session_state or not st.session_state.models:
    st.warning("Please train at least one model in the 'Model Training' tab first.")
    st.stop()

raw = st.session_state.data
X_tr, X_te, y_tr, y_te, X_proc = st.session_state.pp_data

model_choice = st.selectbox("Select Model to Test", list(st.session_state.models.keys()))
model = st.session_state.models[model_choice]

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Profile Builder")
    st.markdown("Enter the details of a hypothetical person.")
    
    user_inputs = {}
    
    # We build the form dynamically based on the raw dataset
    with st.form("what_if_form"):
        for col in raw['X'].columns:
            if pd.api.types.is_numeric_dtype(raw['X'][col]):
                # Numerical input
                min_val = float(raw['X'][col].min())
                max_val = float(raw['X'][col].max())
                mean_val = float(raw['X'][col].median())
                user_inputs[col] = st.number_input(f"{col} (Range: {min_val:.1f} - {max_val:.1f})", value=mean_val)
            else:
                # Categorical input
                unique_options = raw['X'][col].dropna().unique().tolist()
                user_inputs[col] = st.selectbox(f"{col}", unique_options)
                
        submit = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

with col2:
    st.subheader("🤖 AI Decision")
    
    if submit:
        # We need to map the user's raw inputs to the processed integer values the model expects
        processed_inputs = {}
        for col in raw['X'].columns:
            raw_val = user_inputs[col]
            
            if pd.api.types.is_numeric_dtype(raw['X'][col]):
                # Numerics go straight through
                processed_inputs[col] = raw_val
            else:
                # For categoricals, find what integer it maps to in X_proc
                try:
                    # Find a row where the raw data equals the user's input, and grab its processed value
                    match_idx = raw['X'][raw['X'][col] == raw_val].index[0]
                    processed_inputs[col] = X_proc.loc[match_idx, col]
                except IndexError:
                    # Fallback if somehow not found (shouldn't happen with selectbox)
                    processed_inputs[col] = -1
                    
        # Create a 1-row DataFrame for the model
        input_df = pd.DataFrame([processed_inputs])
        
        # Make sure column order exactly matches what the model was trained on
        input_df = input_df[X_proc.columns]
        
        with st.spinner("Analyzing profile..."):
            pred = model.predict(input_df)[0]
            try:
                prob = model.predict_proba(input_df)[0][1] * 100
            except:
                prob = None
                
        st.markdown("### Prediction Result")
        if pred == 1:
            st.success(f"### ✅ APPROVED (Positive Class)")
        else:
            st.error(f"### ❌ REJECTED (Negative Class)")
            
        if prob is not None:
            st.metric("Confidence Score", f"{prob:.1f}%")
            
        st.divider()
        st.markdown("### The 'What-If' Test")
        st.markdown("Try changing one of the **Protected Attributes** on the left (e.g., Race, Gender, Age).")
        st.markdown("*If the prediction flips from Approved to Rejected just because you changed their demographic, you have mathematically proven bias in the model!*")
