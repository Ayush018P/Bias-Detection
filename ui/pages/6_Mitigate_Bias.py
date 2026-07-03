import streamlit as st
import pandas as pd
import sys
import io
import joblib
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar
from bias.mitigator import mitigate_data, mitigate_model

st.set_page_config(page_title="Mitigate Bias", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

st.header("🛠️ Bias Mitigation")
st.markdown("Use advanced algorithms to mathematically remove bias from your data or your model.")

if "data" not in st.session_state or not st.session_state.data:
    st.warning("Please load a dataset from the sidebar to begin.")
    st.stop()

raw = st.session_state.data
X_tr, X_te, y_tr, y_te, X_proc = st.session_state.pp_data

if not raw['protected_cols']:
    st.error("No protected attributes selected when dataset was loaded.")
    st.stop()

tab1, tab2 = st.tabs(["Data-Level (Clean the Dataset)", "Model-Level (Fix the Model)"])

with tab1:
    st.subheader("Correlation Remover")
    st.markdown("Removes the linear correlation between features and the protected attributes, creating a 'clean' dataset.")
    
    mitigate_cols = st.multiselect("Protected attributes to mitigate", raw['protected_cols'], default=raw['protected_cols'])
    
    if st.button("Generate De-biased Dataset", type="primary"):
        if not mitigate_cols:
            st.error("Select at least one protected attribute.")
        else:
            with st.spinner("Applying Correlation Remover..."):
                # CorrelationRemover requires numerical data, so we use X_proc (the ordinally encoded dataset)
                clean_X = mitigate_data(X_proc, mitigate_cols)
                
                # Recombine with Y for download
                df_out = clean_X.copy()
                # Use standard target column name or infer
                df_out['target_label'] = raw['y']
                
                csv = df_out.to_csv(index=False).encode('utf-8')
                
            st.success("Data successfully de-biased!")
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name=f"debiased_{raw['dataset_name']}.csv",
                mime="text/csv",
            )

with tab2:
    st.subheader("Threshold Optimizer")
    st.markdown("Adjusts the decision thresholds of an existing model to enforce Demographic Parity or Equalized Odds.")
    
    if 'models' not in st.session_state or not st.session_state.models:
        st.warning("Train models first in the 'Model Training' tab.")
    else:
        model_choice = st.selectbox("Select Model to Optimize", list(st.session_state.models.keys()))
        prot_attr = st.selectbox("Optimize for Protected Attribute", raw['protected_cols'])
        constraint = st.radio("Fairness Constraint", ["demographic_parity", "equalized_odds"])
        
        if st.button("Optimize Model", type="primary"):
            with st.spinner(f"Optimizing {model_choice} thresholds..."):
                model = st.session_state.models[model_choice]
                
                # We fit the optimizer on the test set here (or a holdout validation set) to learn thresholds
                # Using X_te (preprocessed) because the base estimator needs it to predict
                # We extract the raw sensitive features from X_te_raw
                X_te_raw = raw['X'].loc[X_te.index]
                sensitive_series = X_te_raw[prot_attr]
                
                optimized_model = mitigate_model(model, X_te, y_te, sensitive_series, constraint)
                
                model_buffer = io.BytesIO()
                joblib.dump(optimized_model, model_buffer)
                
            st.success("Model successfully optimized!")
            st.download_button(
                label="Download Fair Model (.joblib)",
                data=model_buffer.getvalue(),
                file_name=f"fair_{model_choice}_{prot_attr}.joblib",
                mime="application/octet-stream",
            )
