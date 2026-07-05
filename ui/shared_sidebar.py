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
                     available_prot_cols = [c for c in df_temp.columns if c != target_col]
                     prot_cols = st.multiselect("Select Protected Attributes (To Audit)", available_prot_cols)
                 except Exception as e:
                     st.error(f"Error parsing CSV: {e}")
                     
        # Validation warnings
        invalid_target = False
        if is_custom and uploaded_file is not None and target_col and prot_cols:
            uploaded_file.seek(0)
            df_temp = pd.read_csv(uploaded_file)
            
            # Check Target Label
            target_unique = df_temp[target_col].nunique()
            if target_unique > 2:
                invalid_target = True
                st.error(f"**Invalid Target Label:** `{target_col}` has {target_unique} unique values. This tool REQUIRES a binary outcome (exactly 2 unique values, like Yes/No or 1/0). You cannot proceed with this column.")
                
            # Check Protected Attributes
            for p_col in prot_cols:
                p_unique = df_temp[p_col].nunique()
                if p_unique > 50:
                    st.warning(f"**Protected Attribute Warning:** `{p_col}` has {p_unique} unique values. Protected attributes must be categorical (e.g., Race, Gender, City). Continuous numbers like Prices or IDs will cause the bias detector to find 0 subgroups.")
                     
        # Action hooks
        btn_disabled = is_custom and (uploaded_file is None or not target_col or not prot_cols or invalid_target)
        
        if st.button("Load & Preprocess Data", use_container_width=True, type="primary", disabled=btn_disabled):
            with st.spinner("Loading dataset..."):
                try:
                    if is_custom:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file)
                        y_raw = df[target_col]
                        
                        if pd.api.types.is_numeric_dtype(y_raw):
                            y = y_raw.astype(int)
                        else: 
                            # This safely handles object, category, boolean, and pandas string dtypes
                            y = (y_raw == y_raw.unique()[0]).astype(int)
                            
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
                        
                    # Calculate new pp_data FIRST before updating any session state
                    X_tr, X_te, y_tr, y_te, X_proc = preprocess_data(raw_data['X'], raw_data['y'])
                    
                    # Only update session state if preprocessing succeeds
                    st.session_state.data = raw_data
                    st.session_state.pp_data = (X_tr, X_te, y_tr, y_te, X_proc)
                    
                    # Clear any old models trained on a different dataset!
                    st.session_state.models = {}
                    st.session_state.eval_results = {}
                    st.session_state.bias_results = None
                    
                    st.success("Loaded!")
                except Exception as e:
                    st.error(f"Failed to process dataset: {str(e)}")
            
        st.divider()
        st.markdown("### Status")
        if "data" in st.session_state and st.session_state.data:
            st.caption(f"Active Dataset: **{st.session_state.data['dataset_name']}**")
        else:
            st.caption("No dataset loaded.")
