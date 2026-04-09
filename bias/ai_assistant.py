"""
bias/ai_assistant.py
Multi-provider AI Assistant for SABPF.
Supports Google Gemini and Groq (Llama-3).
"""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Try to import providers
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

load_dotenv()

class SABPFAssistant:
    def __init__(self, api_key: str = None, provider: str = "gemini"):
        """
        Initialize the assistant with a specific provider and key.
        If no key is provided, it attempts to load from environment.
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY" if self.provider == "gemini" else "GROQ_API_KEY")
        self.model_id = os.getenv("GEMINI_MODEL" if self.provider == "gemini" else "GROQ_MODEL", 
                                 "gemini-1.5-flash" if self.provider == "gemini" else "llama-3.3-70b-versatile")
        
        self.client = None
        if self.provider == "gemini" and HAS_GEMINI and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        elif self.provider == "groq" and HAS_GROQ and self.api_key:
            self.client = Groq(api_key=self.api_key)
            
        self.project_context = self._load_project_context()

    def _load_project_context(self) -> str:
        """Loads the project documentation to provide context to the AI."""
        doc_path = Path(__file__).parent.parent / "project_documentation.md"
        if doc_path.exists():
            with open(doc_path, "r", encoding="utf-8") as f:
                return f.read()
        return "SABPF is a Scalable Automated Bias Prioritization Framework."

    def get_response(self, user_query: str, bias_results: pd.DataFrame = None, chat_history: list = []) -> str:
        """
        Generates a response based on the user query, project context, and current bias results.
        """
        if not self.client:
            return f"❌ AI Client for {self.provider} not initialized. Check your API key in .env."

        # Build the system prompt with context
        context_prompt = f"""
        You are the SABPF AI Assistant. You help users understand the Bias Prioritization Framework.
        --- PROJECT CONTEXT ---
        {self.project_context}
        --- END PROJECT CONTEXT ---
        """

        if bias_results is not None and not bias_results.empty:
            top_findings = bias_results.head(10).to_markdown(index=False)
            context_prompt += f"\n--- BIAS FINDINGS ---\n{top_findings}\n--- END ---"
        
        full_query = f"{context_prompt}\n\nUSER QUESTION: {user_query}"

        try:
            if self.provider == "gemini":
                return self._get_gemini_response(full_query, chat_history)
            elif self.provider == "groq":
                return self._get_groq_response(full_query, chat_history)
        except Exception as e:
            return f"❌ Error from {self.provider.upper()}: {str(e)}"

    def _get_gemini_response(self, query: str, history: list) -> str:
        contents = []
        for msg in history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": query}]})
        
        # Fallback loop inside gemini logic
        models = [self.model_id, "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
        models = list(dict.fromkeys([m for m in models if m]))
        
        for m in models:
            try:
                response = self.client.models.generate_content(model=m, contents=contents)
                return response.text
            except Exception as e:
                if "404" in str(e): continue
                raise e
        return "Gemini failed after all fallbacks."

    def _get_groq_response(self, query: str, history: list) -> str:
        messages = [{"role": "system", "content": "You are a helpful bias detection expert assistant."}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})
        
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
