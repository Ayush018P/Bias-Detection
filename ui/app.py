"""
ui/app.py
Main Streamlit Application for SABPF.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import io
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

load_dotenv()

# Modify sys path so we can import from internal modules when run from root
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.loaders import DATASET_LOADERS, load_dataset
from data.preprocessor import preprocess_data
from models.trainer import train_and_evaluate, get_models, load_trained_models
from models.evaluator import evaluate_model
from bias.detector import discover_subgroups
from reports.generator import export_pdf, export_json
from bias.explainer import generate_eli5_summary
from bias.ai_assistant import SABPFAssistant

# Force reload to pick up multi-provider changes
import importlib
import bias.ai_assistant
import bias.explainer
importlib.reload(bias.ai_assistant)
importlib.reload(bias.explainer)

st.set_page_config(page_title="SABPF Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Session State ---
if 'data' not in st.session_state: st.session_state.data = None
if 'pp_data' not in st.session_state: st.session_state.pp_data = None  # X_train, etc
if 'models' not in st.session_state: st.session_state.models = {}
if 'eval_results' not in st.session_state: st.session_state.eval_results = {}
if 'bias_results' not in st.session_state: st.session_state.bias_results = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- Sidebar ---
with st.sidebar:
    st.title("Bais-Detection")
    st.markdown("**Scalable Automated Bias Prioritization Framework**")
    st.divider()
    
    dataset_opts = ["Upload Custom CSV"] + list(DATASET_LOADERS.keys())
    selected_dataset = st.selectbox("Select Dataset", dataset_opts, index=1)
    
    is_custom = (selected_dataset == "Upload Custom CSV")
    uploaded_file = None
    target_col = None
    prot_cols = []
    
    if is_custom:
        uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
        if uploaded_file is not None:
             try:
                 df_temp = pd.read_csv(uploaded_file)
                 target_col = st.selectbox("Select Target Label (To Predict)", df_temp.columns)
                 prot_cols = st.multiselect("Select Protected Attributes (To Audit)", df_temp.columns)
             except Exception as e:
                 st.error(f"Error parsing CSV: {e}")
                 
    # Action hooks
    btn_disabled = is_custom and (uploaded_file is None or not target_col or not prot_cols)
    
    if st.button("Load & Preprocess Data", use_container_width=True, type="primary", disabled=btn_disabled):
        with st.spinner("Loading dataset..."):
            if is_custom:
                # Reset pointer because pandas already consumed the buffer in preview
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                y_raw = df[target_col]
                
                # Auto-binarize target for the fairness engine
                if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
                    y = (y_raw == y_raw.unique()[0]).astype(int)
                else: 
                    y = y_raw.astype(int)
                    
                X = df.drop(columns=[target_col])
                raw_data = {
                    "X": X.reset_index(drop=True),
                    "y": y.reset_index(drop=True),
                    "protected_cols": prot_cols,
                    "feature_names": list(X.columns),
                    "dataset_name": uploaded_file.name,
                    "task_type": "binary_classification"
                }
            else:
                raw_data = load_dataset(selected_dataset)
                
            st.session_state.data = raw_data
            # Preprocess
            X_tr, X_te, y_tr, y_te, X_proc = preprocess_data(raw_data['X'], raw_data['y'])
            st.session_state.pp_data = (X_tr, X_te, y_tr, y_te, X_proc)
        st.success("Loaded!")
        
    st.divider()
    st.markdown("### Quick Stats")
    if st.session_state.data:
        st.caption(f"Dataset: **{st.session_state.data['dataset_name']}**")
        if st.session_state.bias_results is not None:
            df_b = st.session_state.bias_results
            if not df_b.empty and 'Priority' in df_b.columns:
                st.caption(f"Subgroups Analyzed: **{len(df_b)}**")
                crit_count = len(df_b[df_b['Priority'] == 'Critical'])
                if crit_count > 0:
                    st.error(f" CRITICAL Findings: {crit_count}")
                else:
                    st.success(f" CRITICAL Findings: 0")
                    
                total_bss = df_b['BSS'].sum()
                st.caption(f"Total BSS: **{total_bss:.2f}**")
                
                # Global Bias Percentage (Max DPD)
                max_bias_pct = df_b['DPD'].abs().max() * 100
                st.metric("Dataset Bias Index", f"{max_bias_pct:.1f}%")
                
                # AI Status
                gemini_key = os.getenv("GEMINI_API_KEY")
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    st.success(" ")
                elif gemini_key:
                    st.info("🤖 AI via Gemini ready")
                else:
                    st.warning("⚠️ No AI API Key found in .env")
            else:
                st.caption("Subgroups Analyzed: **0**")
                st.success(" No Findings / All Subgroups <30 size")
                
# --- Page Routing ---
page = st.radio("Navigation", ["Dataset Explorer", "Model Training", "Bias Analysis", "Intersectional Explorer", "Statistical Report", "Gov Report Export", "SABPF AI Assistant"], horizontal=True)
st.divider()

if not st.session_state.data:
    st.warning(" Please load a dataset from the sidebar to begin.")
    st.stop()

raw = st.session_state.data
X_tr, X_te, y_tr, y_te, X_proc = st.session_state.pp_data

if page == "Dataset Explorer":
    st.header(" Dataset Explorer")
    
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    row1_c1.metric("Rows", raw['X'].shape[0])
    row1_c2.metric("Features", raw['X'].shape[1])
    row1_c3.metric("Protected Attributes", len(raw['protected_cols']))
    
    st.subheader("Class Balance")
    fig1 = px.pie(names=raw['y'].value_counts().index, values=raw['y'].value_counts().values, hole=0.4, title="Target Class Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Protected Attribute Distributions")
    for col in raw['protected_cols']:
        if col in raw['X'].columns:
            vc = raw['X'][col].value_counts().reset_index()
            vc.columns = [col, 'Count']
            fig2 = px.bar(vc, x=col, y='Count', title=f"Distribution of {col}")
            st.plotly_chart(fig2, use_container_width=True)

elif page == "Model Training":
    st.header(" Model Training & Evaluation")
    
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

elif page == "Bias Analysis":
    st.header(" Core Bias Analysis")
    
    if not st.session_state.models:
        st.warning("Train models first in the 'Model Training' tab.")
    else:
        col1, col2, col3 = st.columns(3)
        model_choice = col1.selectbox("Select Model to Audit", list(st.session_state.models.keys()))
        threshold_dpd = col2.slider("DPD Significance Threshold", 0.05, 0.30, 0.10, 0.01)
        min_bias_pct = col3.slider("Filter by Min Bias (%)", 0, 100, 0, 5)
        
        if st.button("Run Comprehensive Bias Audit", type="primary"):
            with st.spinner("Discovering subgroups & computing stats..."):
                model = st.session_state.models[model_choice]
                y_pred = model.predict(X_te)
                
                # Use raw unencoded X_te for interpretability where possible, but we need it matched to test indices
                # X_proc has int encoded. We'll use raw for names, but filter to test indices.
                X_te_raw = raw['X'].loc[X_te.index]
                
                res_df = discover_subgroups(X_te_raw, y_te, y_pred, model, raw['protected_cols'], threshold_dpd)
                st.session_state.bias_results = res_df
            st.success("Audit complete.")
            
        if st.session_state.bias_results is not None and not st.session_state.bias_results.empty:
            df_b = st.session_state.bias_results
            
            c1, c2, c3 = st.columns(3)
            max_bias_val = df_b['DPD'].abs().max() * 100
            c1.metric("Highest Bias Index", f"{max_bias_val:.1f}%")
            c2.metric("Worst EOD", f"{df_b['EOD'].max():.3f}")
            cr_count = len(df_b[df_b['Priority'] == 'Critical'])
            c3.metric("Critical Findings", cr_count, delta="!" if cr_count > 0 else "")
            
            st.subheader("Ranked Disparities (By Severity)")
            
            # Prepare percentage column
            df_b['DPD (%)'] = (df_b['DPD'] * 100).round(2)
            
            # Apply Filter
            filtered_df = df_b[df_b['DPD'].abs() * 100 >= min_bias_pct].copy()
            
            # Styling func
            def color_tier(val):
                color = 'green'
                if val == 'Critical': color = 'red'
                elif val == 'High': color = 'orange'
                elif val == 'Medium': color = 'lightblue'
                return f'color: {color}; font-weight: bold'
                
            display_df = filtered_df[['Rank', 'Type', 'Subgroup Name', 'n', 'DPD (%)', 'DIR', 'EOD', 'p_val_corrected', 'BSS', 'Priority']].copy()
            st.dataframe(display_df.style.map(color_tier, subset=['Priority'])
                                       .background_gradient(cmap='Reds', subset=['BSS']), use_container_width=True)
            
            st.subheader("Severity Heatmap")
            if len(df_b) > 1:
                fig = px.treemap(df_b, path=[px.Constant("All Subgroups"), 'Type', 'Subgroup Name'], values='BSS',
                                color='Priority', color_discrete_map={'Critical':'darkred', 'High':'orange', 'Medium':'lightblue', 'Low':'gray'},
                                title="Bias Severity Composition")
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Summary Toggle
                if st.button("Summarize", type="primary"):
                    with st.spinner("Extracting insights..."):
                        summary = generate_eli5_summary(df_b)
                    st.info(summary)
            else:
                st.info("Add a valid 'Gemini API Key' in the sidebar to unlock AI plain-English summaries.")

elif page == "Intersectional Explorer":
    st.header(" Intersectional Bias")
    if st.session_state.bias_results is None:
         st.info("Run the Bias Audit first.")
    else:
        df_b = st.session_state.bias_results
        if df_b.empty or 'Type' not in df_b.columns:
            st.warning("No intersectional subgroups large enough (n>=30) were found.")
        else:
            df_int = df_b[df_b["Type"] == "Intersectional"]
            if df_int.empty:
                st.warning("No intersectional subgroups large enough (n>=30) were found.")
            else:
                st.subheader("Top 10 Intersectional Biases")
                top10 = df_int.head(10)
                fig = px.bar(top10, x="Subgroup Name", y="BSS", color="Priority",
                             title="Intersectional BSS Rankings")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

elif page == "Statistical Report":
    st.header(" Statistical Validations")
    if st.session_state.bias_results is None:
         st.info("Run the Bias Audit first.")
    else:
        df_b = st.session_state.bias_results
        if df_b.empty or 'p_val_raw' not in df_b.columns:
             st.warning("No valid bias subgroups discovered to generate statistics on.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("P-values (Corrected vs Raw)")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df_b['p_val_raw'], name="Raw"))
                fig.add_trace(go.Histogram(x=df_b['p_val_corrected'], name="FDR Corrected"))
                fig.update_layout(barmode='overlay')
                fig.update_traces(opacity=0.75)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Effect Size (Cohen's h)")
                fig2 = px.histogram(df_b, x="Effect_Size_h", nbins=20)
                st.plotly_chart(fig2, use_container_width=True)
                
            st.subheader("Detailed Statistical Matrix")
            st.dataframe(df_b[['Subgroup Name', 'n', 'DPD', 'CI_lower', 'CI_upper', 'p_val_raw', 'p_val_corrected', 'Effect_Size_h']], use_container_width=True)

elif page == "Gov Report Export":
    st.header(" Compliance & Governance Export")
    if st.session_state.bias_results is None or not st.session_state.models:
         st.info("Train models and run Bias Audit first to generate reports.")
    elif st.session_state.bias_results.empty or 'Priority' not in st.session_state.bias_results.columns:
         st.warning("No bias subgroups were discovered, so no governance export can be generated.")
    else:
        st.markdown("Export your findings for compliance logging.")
        
        # We assume the user selected a model in the Bias tab, we'll grab that or defaulting to the first trained one
        model_name = list(st.session_state.models.keys())[0] 
        metrics = st.session_state.eval_results[model_name]
        
        if st.button("Prepare Artifacts"):
            os.makedirs(project_root / "reports" / "out", exist_ok=True)
            pdf_path = project_root / "reports" / "out" / "sabpf_report.pdf"
            json_path = project_root / "reports" / "out" / "sabpf_report.json"
            
            export_json(st.session_state.bias_results, model_name, selected_dataset, str(json_path))
            export_pdf(st.session_state.bias_results, metrics, model_name, selected_dataset, str(pdf_path))
            
            st.success("Artifacts Generated!")
            
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(label="Download PDF Report", data=pdf_file, file_name="sabpf_governance.pdf", mime="application/pdf")
                
            with open(json_path, "rb") as jf:
                st.download_button(label="Download JSON Export", data=jf, file_name="sabpf_governance.json", mime="application/json")

elif page == "SABPF AI Assistant":
    st.header("SABPF AI Assistant")
    st.markdown("Ask anything about the project methodology, your current dataset, or any problems you're facing.")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not (gemini_key or groq_key):
        st.warning("No AI API keys found in .env (GEMINI_API_KEY or GROQ_API_KEY). Please add at least one to use the AI Assistant.")
    else:
        # Prefer Groq if available (due to user's current Gemini restrictions)
        provider = "groq" if groq_key else "gemini"
        st.caption(f"Using **{provider.upper()}** as AI Provider")
        
        # Initialize assistant
        assistant = SABPFAssistant(provider=provider)
        
        # Display chat messages from history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What would you like to know about this project or your results?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response from assistant
                    # We pass the bias results if available for context-aware answers
                    response = assistant.get_response(
                        user_query=prompt,
                        bias_results=st.session_state.bias_results,
                        chat_history=st.session_state.chat_history[:-1]  # excludes current prompt which is passed separately
                    )
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

