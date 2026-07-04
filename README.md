# SABPF: Scalable Automated Bias Prioritization Framework

SABPF is an end-to-end framework for auditing and mitigating algorithmic bias in machine learning models. 

When building predictive models, it is incredibly easy to accidentally learn and amplify historical biases present in the training data. This tool provides a complete pipeline to detect demographic disparities, validate them statistically, rank them by severity, and mathematically mitigate them from the dataset or model.

This project was built to make algorithmic fairness accessible, providing clear, mathematically sound evidence of bias without requiring a background in fairness research.

## Core Features

*   **Interactive What-If Simulator:** A tool designed to prove algorithmic bias to non-technical stakeholders. You can load a real person's profile from your dataset, flip a protected attribute (e.g., changing gender from Male to Female), and instantly see how the model's prediction changes.
*   **Comprehensive Bias Discovery:** Calculates standard fairness metrics including Demographic Parity Difference (DPD), Disparate Impact Ratio (DIR), and Equal Opportunity Difference (EOD) across dozens of single and intersectional subgroups.
*   **Bias Severity Score (BSS):** A proprietary ranking algorithm that scores subgroups based on raw disparity, statistical effect size, population impact, and False Discovery Rate (FDR) corrected p-values. This prevents "alert fatigue" by surfacing only the most critical biases.
*   **Active Mitigation Engines:** Includes a Correlation Remover to mathematically scrub linear bias from your training data, and a Threshold Optimizer to dynamically adjust your model's decision boundaries post-training.
*   **LLM Context Integration:** An optional built-in assistant (supporting Groq and Gemini) that reads your specific dataset findings and provides context-aware guidance on interpreting the results.

## Usage & Workflows

The framework is built as an interactive Streamlit application. A typical workflow involves:

1.  **Data Ingestion:** Load a built-in fairness benchmark (Adult Income, COMPAS, German Credit) or upload your own custom CSV. The system dynamically handles target labels and protected attribute mapping.
2.  **Model Training:** Train models directly in the browser (XGBoost, Random Forest, Logistic Regression, SVM, Decision Tree) or load pre-trained artifacts from disk.
3.  **Auditing:** Run the fairness scanner to generate a ranked heatmap and data table of all discovered disparities.
4.  **Reporting:** Export your findings into compliance-ready PDF and JSON governance reports.

## Local Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd sabpf
2.Set up a virtual environment:
bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
Install the required dependencies:
bash
pip install -r requirements.txt
(Optional) Configure API keys for the Assistant: Create a .env file in the root directory (using .env.example as a template) and add your keys:
env
GROQ_API_KEY=your_key_here

Run the application:
bash
streamlit run ui/app.py
Cloud Deployment
This project is fully compatible with Streamlit Community Cloud. If you deploy it remotely, ensure that you add your GROQ_API_KEY or GEMINI_API_KEY to the Streamlit Secrets Manager, as the .env file is intentionally ignored by git for security.

Technology Stack
Core: Python, Pandas, NumPy, Scikit-learn, XGBoost
Fairness & Statistics: Fairlearn, AIF360, SciPy, Statsmodels
Visualization & UI: Streamlit, Plotly
Reporting: ReportLab
LLM Integration: Groq, Google GenAI
