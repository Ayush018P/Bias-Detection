"""
bias/explainer.py
Contains AI connectivity logic to translate statistical bias metrics into ELI5 (Explain Like I'm Five) English.
Now uses the SABPFAssistant for robust, multi-provider support (Gemini + Groq).
"""

import pandas as pd
import logging
from .ai_assistant import SABPFAssistant

logger = logging.getLogger(__name__)

def generate_eli5_summary(df_b: pd.DataFrame) -> str:
    """
    Sends the top bias disparities to an AI Assistant to retrieve an ELI5 summary.
    Automatically handles provider selection (Gemini or Groq).
    """
    if df_b.empty:
        return "No bias subgroups were discovered, so no explanation is needed."

    # Use the assistant's built-in multi-provider detection
    # We prefer Groq if available as it currently bypasses Gemini's 404/403 issues
    import os
    provider = "groq" if os.getenv("GROQ_API_KEY") else "gemini"
    
    assistant = SABPFAssistant(provider=provider)
    
    # Construct a specific prompt for the explainer
    top_findings = df_b.head(5).copy()
    data_str = top_findings[['Subgroup Name', 'Type', 'DPD', 'EOD', 'p_val_corrected', 'BSS', 'Priority']].to_markdown(index=False)
    
    prompt = f"""
    You are an expert AI fairness auditor. Translate these top statistical bias findings into a plain-English executive summary.
    
    Metrics:
    - DPD (Demographic Parity): Disparity in selection.
    - EOD (Equal Opportunity): Disparity in accuracy.
    - BSS: Overall risk ranking.
    
    Top Findings:
    {data_str}
    
    Instructions:
    1. Write a 2-paragraph "Explain Like I'm Five" (ELI5) summary.
    2. Suggest 1 gentle next-step (e.g. reviewing data collection, reweighing training data).
    3. Use Markdown.
    """
    
    return assistant.get_response(user_query=prompt)
