import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar

st.set_page_config(page_title="Dataset Explorer", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

st.header("📊 Dataset Explorer")

if "data" not in st.session_state or not st.session_state.data:
    st.warning("Please load a dataset from the sidebar to begin.")
    st.stop()

raw = st.session_state.data

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
