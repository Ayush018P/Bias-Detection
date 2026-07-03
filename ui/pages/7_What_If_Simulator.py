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

st.header("🕵️‍♂️ What-If Simulator (Counterfactual Analysis)")
st.markdown("Prove bias by selecting a real applicant and changing their demographics. If changing a protected attribute (like Race or Gender) changes the AI's decision, you have proven individual bias.")

if "data" not in st.session_state or not st.session_state.data:
    st.warning("Please load a dataset from the sidebar to begin.")
    st.stop()

if 'models' not in st.session_state or not st.session_state.models:
    st.warning("Train models first in the 'Model Training' tab.")
    st.stop()

raw = st.session_state.data
X_tr, X_te, y_tr, y_te, X_proc = st.session_state.pp_data

# Ensure we have protected columns
if not raw['protected_cols']:
    st.error("No protected attributes selected when dataset was loaded.")
    st.stop()

# 1. Select a Model
model_choice = st.selectbox("Active AI Model", list(st.session_state.models.keys()))
model = st.session_state.models[model_choice]

st.divider()

# 2. Select an Applicant
X_te_raw = raw['X'].loc[X_te.index]
applicant_indices = list(range(len(X_te_raw)))

# Let user pick an applicant by integer index
selected_idx = st.number_input("Select an Applicant (Index 0 to {})".format(len(applicant_indices)-1), min_value=0, max_value=len(applicant_indices)-1, value=0)

# Extract raw and preprocessed data for this applicant
real_idx = X_te.index[selected_idx]
raw_applicant = X_te_raw.loc[real_idx]
num_applicant = X_te.loc[real_idx]

# Run original prediction
orig_pred = model.predict([num_applicant])[0]
orig_prob = None
if hasattr(model, "predict_proba"):
    orig_prob = model.predict_proba([num_applicant])[0][1]

# Visual outcome mapper
def outcome_badge(pred_val):
    if pred_val == 1:
        return "✅ **Positive Outcome** (1)"
    return "❌ **Negative Outcome** (0)"

# 3. Display Profile Card
st.subheader("👤 Applicant Profile")
col1, col2 = st.columns([2, 1])

with col1:
    with st.expander("View Full Raw Profile", expanded=True):
        st.dataframe(pd.DataFrame(raw_applicant).T, use_container_width=True)

with col2:
    st.info("### Original AI Decision")
    st.markdown(outcome_badge(orig_pred))
    if orig_prob is not None:
         st.caption(f"Confidence (Probability of 1): {orig_prob:.2f}")

st.divider()

# 4. Counterfactual Tweak
st.subheader("🔄 Counterfactual Tweak")
st.markdown("Change a protected attribute below to see if the AI changes its mind.")

tweak_col = st.selectbox("Protected Attribute to change:", raw['protected_cols'])

# Get unique raw values for this column from the full raw dataset
unique_raw_vals = raw['X'][tweak_col].unique()

# Dropdown for the new value
original_val = raw_applicant[tweak_col]
new_raw_val = st.selectbox("New Value:", unique_raw_vals, index=list(unique_raw_vals).index(original_val))

if new_raw_val != original_val:
    # We must find the numerical encoding for this new raw value
    # We do a lookup: find any row in the FULL raw dataset that has this new value, and get its preprocessed numerical value
    lookup_row_idx = raw['X'][raw['X'][tweak_col] == new_raw_val].index[0]
    new_num_val = X_proc.loc[lookup_row_idx, tweak_col]
    
    # Create the counterfactual numerical row
    cf_num_applicant = num_applicant.copy()
    cf_num_applicant[tweak_col] = new_num_val
    
    # Predict
    cf_pred = model.predict([cf_num_applicant])[0]
    cf_prob = None
    if hasattr(model, "predict_proba"):
        cf_prob = model.predict_proba([cf_num_applicant])[0][1]
        
    st.markdown("### The Counterfactual Reality")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**If Applicant was:** `{new_raw_val}`")
        st.markdown(outcome_badge(cf_pred))
        if cf_prob is not None:
            st.caption(f"Confidence (Probability of 1): {cf_prob:.2f}")
            
    with c2:
        if cf_pred != orig_pred:
            st.error("🚨 **BIAS DETECTED!** 🚨\nThe AI completely flipped its decision purely based on this protected attribute!")
        else:
            st.success("✅ **No Change.**\nThe AI's decision remained the same despite the demographic change.")
            
else:
    st.caption("Change the value above to run a counterfactual prediction.")
