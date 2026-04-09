# SABPF: Scalable Automated Bias Prioritization Framework

SABPF is an end-to-end machine learning fairness auditing project built with Python and Streamlit. It trains multiple classification models on real-world tabular datasets, detects demographic and intersectional bias across subgroups, validates disparities with statistical testing, ranks findings by severity, and exports governance-ready reports.

This project is designed for academic demonstration, fairness analysis workflows, and applied responsible AI experimentation.

## Core Features

- Train multiple machine learning models on benchmark or custom tabular datasets.
- Audit model predictions for subgroup and intersectional bias.
- Compute fairness metrics such as Demographic Parity Difference, Disparate Impact Ratio, and Equal Opportunity Difference.
- Apply statistical validation using p-values, effect sizes, confidence intervals, and false discovery rate correction.
- Rank findings with a Bias Severity Score (BSS) so the most important issues are reviewed first.
- Explore results in an interactive Streamlit dashboard.
- Export PDF and JSON governance artifacts.
- Use an optional AI assistant backed by environment-based API keys.

## Supported Workflows

SABPF supports two main ways of working:

1. Built-in fairness benchmark datasets.
2. Custom CSV upload where the user selects the target column and protected attributes.

After loading data, the app lets you preprocess, train models, evaluate model quality, run a fairness audit, inspect intersectional findings, review statistical evidence, and export reports.

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- SciPy and Statsmodels
- Fairlearn and AIF360
- ReportLab
- Python Dotenv

## Project Structure

```text
sabpf/
|-- bias/
|   |-- ai_assistant.py
|   |-- detector.py
|   |-- explainer.py
|   |-- metrics.py
|   |-- ranker.py
|   `-- statistical.py
|-- data/
|   |-- loaders.py
|   `-- preprocessor.py
|-- models/
|   |-- evaluator.py
|   |-- trainer.py
|   `-- saved/
|-- reports/
|   `-- generator.py
|-- ui/
|   `-- app.py
|-- .env.example
|-- .gitignore
|-- project_documentation.md
|-- README.md
`-- requirements.txt
```

## Built-In Datasets

The project includes loaders for commonly used fairness-related datasets:

- Adult Income
- COMPAS Recidivism
- German Credit
- Bank Marketing
- Law School Admissions

The exact protected attributes depend on the loader configuration inside the project.

## Models Used

The training pipeline supports multiple classification models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine with RBF kernel

Pretrained artifacts are stored in `models/saved/` when available.

## Bias Metrics and Analysis

The fairness engine focuses on subgroup discovery and prioritization. The project evaluates:

- Demographic Parity Difference (DPD)
- Disparate Impact Ratio (DIR)
- Equal Opportunity Difference (EOD)
- Statistical significance
- Effect size
- Confidence intervals
- Bias Severity Score (BSS)

The system also supports intersectional analysis so combinations such as race plus gender can be audited together.

## Dashboard Pages

The Streamlit app includes these major sections:

- `Dataset Explorer`: dataset size, target balance, and protected attribute distributions.
- `Model Training`: train or load supported models and compare ROC performance.
- `Bias Analysis`: run subgroup audits and rank disparities by severity.
- `Intersectional Explorer`: inspect top intersectional bias findings.
- `Statistical Report`: review p-values, corrected p-values, and effect sizes.
- `Gov Report Export`: generate PDF and JSON governance reports.
- `SABPF AI Assistant`: ask project- and results-related questions using configured AI keys.

## Installation

### 1. Clone or download the project

```bash
git clone <your-repository-url>
cd sabpf
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root using `.env.example` as a template.

Example:

```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

The `.env` file is ignored by Git and should never be pushed to GitHub.

## Running the App

```bash
streamlit run ui/app.py
```

Then open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

## Typical Usage Flow

1. Start the Streamlit app.
2. Choose a built-in dataset or upload a custom CSV.
3. Select the target label and protected attributes if using custom data.
4. Load and preprocess the data.
5. Train selected models or load pretrained ones.
6. Run the comprehensive bias audit.
7. Review subgroup rankings, charts, and statistical evidence.
8. Export PDF or JSON governance reports if needed.

## Git and GitHub Notes

This repository now includes a `.gitignore` file that excludes:

- `.env`
- virtual environments
- Python cache files
- generated report output

That means your secret API keys will stay local and will not be uploaded when you push the project.

## Recommended Files to Keep in Git

- source code under `bias/`, `data/`, `models/`, `reports/`, and `ui/`
- `requirements.txt`
- `README.md`
- `.env.example`
- project documentation files

## License

Add your preferred license here before publishing, such as MIT, Apache-2.0, or a university/project-specific license.

## Author Notes

This project is well suited for:

- final-year academic projects
- responsible AI demonstrations
- fairness auditing prototypes
- model governance showcases

If you publish this on GitHub, consider also adding screenshots of the Streamlit dashboard and a short demo video link to make the repository stronger.
