import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar
from models.trainer import train_and_evaluate, get_models, load_trained_models

st.set_page_config(page_title="Model Training", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

st.header("🧠 Model Training & Evaluation")

if "data" not in st.session_state or not st.session_state.data:
    st.warning("Please load a dataset from the sidebar to begin.")
    st.stop()

if 'models' not in st.session_state: st.session_state.models = {}
if 'eval_results' not in st.session_state: st.session_state.eval_results = {}

X_tr, X_te, y_tr, y_te, X_proc = st.session_state.pp_data

model_names = list(get_models().keys())
selected_models = st.multiselect("Select Models to Train", model_names, default=model_names)

col1, col2 = st.columns(2)
with col1:
    if st.button("Train Selected Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Select at least one model.")
        else:
            with st.spinner("Training models"):
                models, eval_objs = train_and_evaluate(st.session_state.data['dataset_name'], X_tr, X_te, y_tr, y_te, selected_models)
                st.session_state.models = models
                st.session_state.eval_results = eval_objs
            st.success("Training Complete!")
            
with col2:
    if st.button("Load Pre-trained Models", use_container_width=True):
        if not selected_models:
            st.error("Select at least one model to load.")
        else:
            with st.spinner("Loading models from disk..."):
                loaded_models, loaded_evals = load_trained_models(st.session_state.data['dataset_name'], X_te, y_te, selected_models)
                if not loaded_models:
                    st.warning("No pre-trained models found for this dataset. Please train them first.")
                else:
                    st.session_state.models.update(loaded_models)
                    st.session_state.eval_results.update(loaded_evals)
                    st.success(f"Loaded {len(loaded_models)} pre-trained model(s)!")
                    
if st.session_state.models:
    st.markdown(f"**Currently Active Models:** {', '.join(st.session_state.models.keys())}")
        
if st.session_state.eval_results:
    st.subheader("Results Table")
    res_df = pd.DataFrame([v for k, v in st.session_state.eval_results.items()])
    display_cols = [c for c in res_df.columns if c != 'roc_curve']
    st.dataframe(res_df[display_cols].style.background_gradient(cmap='viridis', subset=['Accuracy', 'AUC-ROC']))
    
    st.subheader("ROC Curves")
    fig = go.Figure()
    for m_name, res in st.session_state.eval_results.items():
        if res.get('roc_curve'):
            fig.add_trace(go.Scatter(x=res['roc_curve']['fpr'], y=res['roc_curve']['tpr'], name=f"{m_name} (AUC={res['AUC-ROC']:.3f})", mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'), name="Random"))
    st.plotly_chart(fig, use_container_width=True)
