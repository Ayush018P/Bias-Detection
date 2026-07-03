"""
bias/explainer.py
Translates the entire Bias Analysis page into a plain-English executive
summary that a non-technical stakeholder can fully understand.
"""

import os
import pandas as pd
import logging
from .ai_assistant import SABPFAssistant

logger = logging.getLogger(__name__)


def generate_eli5_summary(
    df_b: pd.DataFrame,
    model_name: str = "Unknown Model",
    dataset_name: str = "the dataset",
    model_accuracy: float = None,
    model_auc: float = None,
) -> str:
    """
    Sends the FULL bias analysis context to an AI Assistant and returns a
    plain-English executive summary suitable for a non-technical audience.

    Parameters
    ----------
    df_b           : Full bias results DataFrame (all subgroups, sorted by BSS)
    model_name     : Name of the audited model (e.g. "XGBoost")
    dataset_name   : Name of the dataset (e.g. "Bank Marketing")
    model_accuracy : Accuracy score of the audited model (0-1)
    model_auc      : AUC-ROC score of the audited model (0-1)
    """
    if df_b.empty:
        return "✅ No bias subgroups were discovered in this audit. The model appears to treat all subgroups fairly based on the available data."

    # ── 1. Build aggregate statistics ──────────────────────────────────────
    total          = len(df_b)
    critical_count = len(df_b[df_b["Priority"] == "Critical"])
    high_count     = len(df_b[df_b["Priority"] == "High"])
    medium_count   = len(df_b[df_b["Priority"] == "Medium"])
    low_count      = len(df_b[df_b["Priority"] == "Low"])

    worst_dpd_row  = df_b.loc[df_b["DPD"].abs().idxmax()]
    worst_eod_row  = df_b.loc[df_b["EOD"].idxmax()]
    worst_bss_row  = df_b.iloc[0]   # already sorted by BSS desc

    intersectional = df_b[df_b["Type"] == "Intersectional"]
    single         = df_b[df_b["Type"] == "Single"]

    max_dpd_pct    = worst_dpd_row["DPD"] * 100
    max_eod        = worst_eod_row["EOD"]

    # ── 2. Format top-10 findings as a readable table ──────────────────────
    cols_to_show  = ["Rank", "Subgroup Name", "Type", "DPD", "EOD", "BSS", "Priority"]
    available_cols = [c for c in cols_to_show if c in df_b.columns]
    top10_md       = df_b[available_cols].head(10).to_markdown(index=False)

    # ── 3. Format critical findings separately ─────────────────────────────
    critical_md = ""
    if critical_count > 0:
        critical_df  = df_b[df_b["Priority"] == "Critical"][available_cols]
        critical_md  = f"\n**Critical Findings Only:**\n{critical_df.to_markdown(index=False)}\n"

    # ── 4. Model performance context ──────────────────────────────────────
    perf_str = ""
    if model_accuracy is not None:
        perf_str += f"- Model Accuracy: {model_accuracy*100:.1f}%\n"
    if model_auc is not None:
        perf_str += f"- AUC-ROC: {model_auc:.3f}\n"

    # ── 5. Build the prompt ────────────────────────────────────────────────
    prompt = f"""
You are an expert writing a professional fairness audit report for a non-technical
business audience such as HR managers, compliance officers, and executives.

Rules:
- Write in plain, professional English. No jargon.
- Do NOT use emojis anywhere in the output.
- Do NOT use phrases like "As an AI" or "I was trained".
- Use clean section headers with no symbols.
- Write like a human analyst, not a chatbot.

---

AUDIT CONTEXT
Dataset: {dataset_name}
Model: {model_name}
{perf_str}
---

METRIC DEFINITIONS (weave these explanations naturally into your report)
- DPD (Demographic Parity Difference): How much more or less often the model gives a
  positive outcome to a group compared to the overall population average.
  A DPD of -0.20 means that group gets 20% fewer positive decisions.
- EOD (Equalized Odds Difference): Whether the model makes more errors for some groups
  than others — even when accuracy looks fine overall.
- BSS (Bias Severity Score): A 0-to-1 risk score combining the size of the affected
  group, statistical confidence, and magnitude of bias. Higher means worse.
- Priority Tiers: Critical, High, Medium, Low.

---

AUDIT NUMBERS
- Total subgroups analysed: {total}
- Critical findings: {critical_count}
- High findings: {high_count}
- Medium findings: {medium_count}
- Low findings: {low_count}
- Single-attribute biases: {len(single)}
- Intersectional biases: {len(intersectional)}
- Most biased group (by DPD): {worst_dpd_row['Subgroup Name']} — disadvantaged by {abs(max_dpd_pct):.1f}%
- Largest accuracy gap (EOD): {worst_eod_row['Subgroup Name']} — gap of {max_eod:.3f}
- Highest risk score (BSS): {worst_bss_row['Subgroup Name']} — score of {worst_bss_row['BSS']:.3f}

{critical_md}

TOP 10 FINDINGS
{top10_md}

---

YOUR TASK
Write a structured plain-English audit report with exactly these four sections:

1. Overview
What was audited, which model and dataset, and a one-line summary of the result.

2. Key Findings
For each important finding, explain in plain terms who is affected and by how much.
Use real-world language (e.g. "customers in the 20-30 age group receive loan offers
22% less often than the average customer"). Cover the top 3 to 5 findings.

3. Severity Assessment
Explain how serious the situation is overall. Describe what Critical and High
priority means in practical terms without using the word BSS directly.

4. Recommended Actions
List 3 concrete steps the organisation should take to address the bias found.
Keep recommendations practical and understandable to a non-technical manager.

Write in clear paragraphs. Do not use bullet points for everything — mix prose
and brief lists naturally. Do not use emojis. Do not use markdown headers with ###.
Use plain bold headers like: Overview, Key Findings, Severity Assessment, Recommended Actions.
"""

    # ── 6. Call AI provider ───────────────────────────────────────────────
    provider  = "groq" if os.getenv("GROQ_API_KEY") else "gemini"
    assistant = SABPFAssistant(provider=provider)
    return assistant.get_response(user_query=prompt)
