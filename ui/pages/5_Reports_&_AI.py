import streamlit as st
import os
import sys
from pathlib import Path

# Fix sys path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ui.shared_sidebar import render_sidebar
from reports.generator import export_pdf, export_json
from bias.ai_assistant import SABPFAssistant

st.set_page_config(page_title="Reports & AI", layout="wide")

if not st.session_state.get("authenticated"):
    st.error("Please log in from the Home page first.")
    st.stop()

render_sidebar()

tab1, tab2 = st.tabs(["Governance Reports", "AI Assistant"])

with tab1:
    st.header("📄 Compliance & Governance Export")
    if "bias_results" not in st.session_state or st.session_state.bias_results is None or not st.session_state.models:
         st.info("Train models and run a Bias Audit first to generate reports.")
    elif st.session_state.bias_results.empty or 'Priority' not in st.session_state.bias_results.columns:
         st.warning("No bias subgroups were discovered, so no governance export can be generated.")
    else:
        st.markdown("Export your findings for compliance logging.")
        
        model_name = list(st.session_state.models.keys())[0] 
        metrics = st.session_state.eval_results[model_name]
        selected_dataset = st.session_state.data['dataset_name']
        
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

with tab2:
    st.header("🤖 SABPF AI Assistant")
    st.markdown("Ask anything about the project methodology, your current dataset, or any problems you're facing.")

    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not (gemini_key or groq_key):
        st.warning("No AI API keys found in .env (GEMINI_API_KEY or GROQ_API_KEY). Please add at least one to use the AI Assistant.")
    else:
        provider = "groq" if groq_key else "gemini"
        active_key = groq_key if groq_key else gemini_key
        st.caption(f"Using **{provider.upper()}** as AI Provider")
        
        assistant = SABPFAssistant(api_key=active_key, provider=provider)
        
        if 'chat_history' not in st.session_state: 
            st.session_state.chat_history = []
            
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What would you like to know about this project or your results?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    bias_res = st.session_state.get("bias_results")
                    response = assistant.get_response(
                        user_query=prompt,
                        bias_results=bias_res,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                    st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
