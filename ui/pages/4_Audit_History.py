import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar
from db.database import SessionLocal
from db.models import AuditHistory

st.set_page_config(page_title="Audit History", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

st.header("🗄️ Audit History")
st.markdown("Review your previous bias audits stored in the database.")

db = SessionLocal()
history = db.query(AuditHistory).filter(AuditHistory.user_id == st.session_state["user_id"]).order_by(AuditHistory.created_at.desc()).all()
db.close()

if not history:
    st.info("You haven't run any bias audits yet. Go to 'Bias Analysis' to run one!")
else:
    # Prepare data for table
    data = []
    for h in history:
        data.append({
            "Date": h.created_at.strftime("%Y-%m-%d %H:%M"),
            "Dataset": h.dataset_name,
            "Model": h.model_name,
            "Critical Findings": h.critical_findings_count,
            "Total Severity (BSS)": round(h.total_bss, 2),
            "Max Bias %": f"{h.max_bias_pct:.1f}%",
            "Legal Risk Level": h.legal_risk_level if h.legal_risk_level else "Unknown"
        })
        
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
