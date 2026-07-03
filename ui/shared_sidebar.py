import streamlit as st
import pandas as pd
from data.loaders import DATASET_LOADERS, load_dataset
from data.preprocessor import preprocess_data

def render_sidebar():
    if not st.session_state.get("authenticated"):
        return

    with st.sidebar:
        st.title("SABPF Enterprise")
        st.markdown(f"User: **{st.session_state.get('username')}**")
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
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    y_raw = df[target_col]
                    
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
                X_tr, X_te, y_tr, y_te, X_proc = preprocess_data(raw_data['X'], raw_data['y'])
                st.session_state.pp_data = (X_tr, X_te, y_tr, y_te, X_proc)
            st.success("Loaded!")
            
        st.divider()
        st.markdown("### Status")
        if "data" in st.session_state and st.session_state.data:
            st.caption(f"Active Dataset: **{st.session_state.data['dataset_name']}**")
        else:
            st.caption("No dataset loaded.")
