# Scalable Automated Bias Prioritization Framework (SABPF)
## Comprehensive Project Documentation

This document outlines the architecture, components, and methodologies implemented in the Scalable Automated Bias Prioritization Framework (SABPF). It serves as a comprehensive guide for presenting the project to faculty or stakeholders.

---

## 1. Project Overview & Motivation

Machine learning models inherently risk amplifying historical biases present in their training data. While many tools compute basic fairness metrics, they often fail to:
1. **Statistically validate** if the bias is significant (versus random noise).
2. **Handle intersectional bias** (e.g., bias affecting a specific race *and* gender combination).
3. **Prioritize** which biases are the most critical to address first.

**SABPF solves this problem.** It is an end-to-end Machine Learning pipeline and interactive dashboard that trains predictive models, discovers demographic biases across distinct subgroups, validates them with strict statistical tests, and ranks findings by severity to prevent "alert fatigue."

---

## 2. System Architecture

The project is structured in a highly modular, scalable way. 

### Directory Structure & Responsibilities:
- **`data/`**: Handles dataset loading and standard preprocessing.
- **`models/`**: Manages the training and evaluation of multiple distinct ML algorithms.
- **`bias/`**: The core analytical engine. Computes fairness metrics, runs statistical tests, and ranks the severity of findings.
- **`reports/`**: Generates compliance-ready governance exports (PDF, JSON).
- **`ui/`**: A Streamlit interactive web application (`app.py`) that acts as the user-facing entry point.

---

## 3. Detailed Component Breakdown

### 3.1 Data Management & Customization
- **Pre-configured Real-World Datasets:** Directly imports renowned fairness benchmarks like the Adult Income dataset, COMPAS Recidivism, German Credit, Bank Marketing, and Law School Admissions datasets.
- **Custom CSV Upload:** Implemented a robust data ingestion pipeline where users can upload any custom dataset. The system dynamically permits users to map their own 'Target' variable and arbitrary 'Protected Attributes', cleanly handling formatting errors, and auto-binarizing categorical targets for the models.

### 3.2 Model Training Pipeline
The framework does not lock users into a single model architecture. It concurrently trains and evaluates:
- Logistic Regression
- Decision Trees
- Random Forests
- XGBoost
- Support Vector Machines (SVM with RBF Kernel)

For each model, the system computes overarching performance metrics including **Accuracy**, **AUC-ROC**, and plots dynamic **ROC Curves** so users can weigh the tradeoff between raw predictive power and fairness.

### 3.3 The Bias Detection Engine
The crown jewel of the project involves exhaustive subgroup discovery:
- **Single-Attribute Scanning:** Automates the separation of testing data into distinct subsets based on specific protected variables (e.g., Race=Black, Sex=Female) and evaluates their individual fairness scores against the overall population. Minimum size constraints (`n >= 30`) ensure validity.
- **Intersectional Scanning:** Evaluates multi-dimensional bias by looking at the cross-sections of metrics (e.g., Sex=Female AND Age=Senior). 
- **Core Fairness Metrics:**
  - **DPD (Demographic Parity Difference):** Measures difference in selection/success rates.
  - **DIR (Disparate Impact Ratio):** The ratio of success rates.
  - **EOD (Equal Opportunity Difference):** Measures difference in true positive rates.

### 3.4 Statistical Validation Module
Instead of superficially flagging any slight disparity, SABPF runs rigorous math on the findings:
- **Z-Tests for Proportions:** Tests if the disparity in selection rates is statistically significant.
- **Cohen's h:** Calculates the absolute *Effect Size* indicating the practical magnitude of the bias.
- **Bootstrap Confidence Intervals:** Provides a 95% CI around the bias margin through iterative resampling.
- **FDR Correction (False Discovery Rate):** Adjusts p-values to prevent Type I errors (false positives) when checking dozens of subgroups simultaneously.

### 3.5 The Ranking Engine (BSS)
Data scientists can face hundreds of bias alerts. We developed the **Bias Severity Score (BSS)** to rank them.
- Calculates an aggregate severity based on: Raw Disparity (DPD), Effect Size, Statistical Validity (Corrected P-Value), the relative size of the impacted population, and DIR.
- **Prioritization Tiers:** Categorizes findings logically into **Critical**, **High**, **Medium**, or **Low** alerts for triage.

### 3.6 Automated Governance Reporting
- For compliance and auditing, the `reports/generator.py` script utilizes `reportlab` to compile a **PDF Governance Report** and a **JSON Artifact**.
- It summarizes top-tier "Critical" and "High" severity vulnerabilities, alongside model accuracy, creating a deployable artifact for legal or compliance review boards.

---

## 4. The Interactive User Experience (Streamlit UI)

The web dashboard (`ui/app.py`) provides 6 primary workflows out of the box:
1. **Dataset Explorer:** Visualizes class balance and demographic distributions.
2. **Model Training:** Allows multi-selection of models to train and provides tabular and visual ROC benchmarks.
3. **Bias Analysis:** The core workspace. Users select a trained model, define tolerance thresholds, and view all bias findings cleanly sorted by the Bias Severity Score across an interactive heatmap and ranked dataframes.
4. **Intersectional Explorer:** Zooms tightly into cross-attribute bias disparities.
5. **Statistical Report:** Transparently outlines the raw math (P-Values, Cohen's h distributions) that back up the severity claims.
6. **Gov Report Export:** A one-click generation interface to download the PDFs and JSON packages.

---

## 5. Technical Stack
- **Languages:** Python
- **Core Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost
- **Visualization:** Streamlit, Plotly
- **Reporting:** ReportLab (for PDF compilation), JSON
- **Concepts Applied:** Algorithmic Fairness, Statistical Hypothesis Testing, Machine Learning, Interactive UI Design.
