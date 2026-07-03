import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import json
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar
from bias.detector import discover_subgroups
from bias.explainer import generate_eli5_summary
from db.database import SessionLocal
from db.models import AuditHistory

st.set_page_config(page_title="Bias Analysis", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

st.header("⚖️ Core Bias Analysis")

if "data" not in st.session_state or not st.session_state.data:
    st.warning("Please load a dataset from the sidebar to begin.")
    st.stop()

if 'models' not in st.session_state or not st.session_state.models:
    st.warning("Train models first in the 'Model Training' tab.")
    st.stop()
    
raw = st.session_state.data
X_tr, X_te, y_tr, y_te, X_proc = st.session_state.pp_data

col1, col2, col3 = st.columns(3)
model_choice = col1.selectbox("Select Model to Audit", list(st.session_state.models.keys()))
threshold_dpd = col2.slider("DPD Significance Threshold", 0.05, 0.30, 0.10, 0.01)
min_bias_pct = col3.slider("Filter by Min Bias (%)", 0, 100, 0, 5)

if st.button("Run Comprehensive Bias Audit", type="primary"):
    with st.spinner("Discovering subgroups & computing stats..."):
        model = st.session_state.models[model_choice]
        y_pred = model.predict(X_te)
        
        X_te_raw = raw['X'].loc[X_te.index]
        
        res_df = discover_subgroups(X_te_raw, y_te, y_pred, model, raw['protected_cols'], threshold_dpd)
        st.session_state.bias_results = res_df
        
        # --- SAVE TO DATABASE ---
        if not res_df.empty:
            cr_count = len(res_df[res_df['Priority'] == 'Critical'])
            total_bss = res_df['BSS'].sum()
            max_bias_pct = res_df['DPD'].abs().max() * 100
            clean_df = res_df.replace({pd.NA: None, float('nan'): None})
            
            db = SessionLocal()
            audit_record = AuditHistory(
                user_id=st.session_state["user_id"],
                dataset_name=raw['dataset_name'],
                model_name=model_choice,
                critical_findings_count=cr_count,
                total_bss=total_bss,
                max_bias_pct=max_bias_pct,
                results_json=clean_df.to_json(orient="records")
            )
            db.add(audit_record)
            db.commit()
            db.close()
            st.toast("Audit saved to History!")
            
    st.success("Audit complete.")

if "bias_results" in st.session_state and st.session_state.bias_results is not None and not st.session_state.bias_results.empty:
    df_b = st.session_state.bias_results
    
    c1, c2, c3, c4, c5 = st.columns(5)
    max_bias_val = df_b['DPD'].abs().max() * 100
    c1.metric("Selection Gap (DPD)", f"{max_bias_val:.1f}%", help="How much less likely this group is to be approved compared to average.")
    c2.metric("Accuracy Bias (EOD)", f"{df_b['EOD'].max():.3f}", help="Does the model make more errors for this group compared to others?")
    
    # New metrics (if available in results, fallback to 0)
    fnr_max = df_b['FNR_Disparity'].abs().max() if 'FNR_Disparity' in df_b.columns else 0
    ppv_max = df_b['PPV_Gap'].abs().max() if 'PPV_Gap' in df_b.columns else 0
    
    c3.metric("Unfair Rejection (FNR)", f"{fnr_max:.3f}", help="Are qualified people in this group unfairly rejected more often?")
    c4.metric("Precision Gap (PPV)", f"{ppv_max:.3f}", help="Difference in how often the model's positive predictions are actually correct for this group.")
    
    cr_count = len(df_b[df_b['Priority'] == 'Critical'])
    c5.metric("Critical Findings", cr_count, delta="!" if cr_count > 0 else "", help="Number of subgroups requiring immediate mitigation.")
    
    st.subheader("Ranked Disparities (By Severity)")
    df_b['DPD (%)'] = (df_b['DPD'] * 100).round(2)
    filtered_df = df_b[df_b['DPD'].abs() * 100 >= min_bias_pct].copy()
    
    def color_tier(val):
        color = 'green'
        if val == 'Critical': color = 'red'
        elif val == 'High': color = 'orange'
        elif val == 'Medium': color = 'lightblue'
        return f'color: {color}; font-weight: bold'
        
    # Ensure new columns exist
    for col in ['FNR_Disparity', 'PPV_Gap', 'Calibration_Gap']:
        if col not in filtered_df.columns:
            filtered_df[col] = 0.0
            
    display_df = filtered_df[['Rank', 'Type', 'Subgroup Name', 'n', 'DPD (%)', 'DIR', 'EOD', 'FNR_Disparity', 'PPV_Gap', 'Priority']].copy()
    
    # Rename columns to Plain English with Acronyms
    display_df = display_df.rename(columns={
        "DPD (%)": "Selection Gap (DPD %)",
        "DIR": "Impact Ratio (DIR)",
        "EOD": "Accuracy Bias (EOD)",
        "FNR_Disparity": "Unfair Rejection (FNR Gap)",
        "PPV_Gap": "Precision Gap (PPV Gap)",
    })
    
    st.dataframe(display_df.style.map(color_tier, subset=['Priority']), use_container_width=True)
    
    st.subheader("Severity Heatmap")
    if len(df_b) > 1:
        fig = px.treemap(df_b, path=[px.Constant("All Subgroups"), 'Type', 'Subgroup Name'], values='BSS',
                        color='Priority', color_discrete_map={'Critical':'darkred', 'High':'orange', 'Medium':'lightblue', 'Low':'gray'},
                        title="Bias Severity Composition")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Summarize with AI", type="primary"):
            with st.spinner("Extracting insights..."):
                summary = generate_eli5_summary(df_b)
            st.info(summary)
