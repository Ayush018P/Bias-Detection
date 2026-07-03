# Scalable Automated Bias Prioritization Framework (SABPF)
### Mini Project Report — Academic Year 2025–26

**Authors:**
Gaurav Patil — 1272250165
Ayush Patel — 1272250207

**Guide:** P. S. Metkewar Sir
**Co-Guide:** Navnath Shete Sir

**Department of Computer Engineering**
Symbiosis Institute of Technology, Pune

---

## Title

**Scalable Automated Bias Detection and Prioritization in Machine Learning Systems Using Statistical Severity Scoring**

---

## 1. Problem Statement

Machine learning models trained on historical data are increasingly deployed in high-stakes domains such as financial lending, criminal justice, healthcare, and hiring. These models inherit and often amplify the systemic biases present in the training data, producing outcomes that are unfair to specific demographic subgroups based on attributes such as age, gender, race, and marital status.

While several tools exist that detect bias at a surface level, the existing landscape suffers from a critical operational gap: detection without prioritization. In real-world deployments, a single model audit can surface hundreds of potential disparities across different subgroups, metrics, and attribute combinations. Without a systematic method to rank these findings by urgency, practitioners face alert fatigue — an inability to determine which bias requires immediate intervention and which is statistically negligible.

Furthermore, existing tools are typically designed for specific data modalities, rely heavily on manual expert interpretation, and lack formal statistical guarantees that distinguish genuine bias from random sampling noise. This makes them unsuitable for the repeatable, high-confidence auditing workflows demanded by regulatory frameworks such as the EU AI Act and EEOC compliance guidelines.

There is therefore a clear and pressing need for a scalable, automated, statistically grounded framework that transforms bias auditing from a passive diagnostic exercise into an active prioritization and decision-support process.

---

## 2. Research Objectives

The aim of this project is to design, implement, and validate a complete end-to-end bias auditing system — the Scalable Automated Bias Prioritization Framework (SABPF) — that automates the full audit lifecycle from data ingestion to governance export.

The specific objectives are:

1. To develop an automated subgroup discovery engine that identifies bias across both single attributes and intersectional combinations of protected attributes without manual configuration.

2. To implement a multi-metric fairness evaluation layer computing Demographic Parity Difference, Disparate Impact Ratio, Equalized Odds Difference, Predictive Parity, and False Negative Rate Disparity per subgroup.

3. To design a statistical validation layer using two-proportion Z-tests, Cohen's h effect sizes, Bootstrap Confidence Intervals, and Benjamini-Hochberg FDR correction to eliminate false positives across mass parallel testing.

4. To engineer a composite Bias Severity Score (BSS) formula that aggregates metric magnitude, effect size, statistical confidence, and subgroup exposure into a single ranked priority tier system.

5. To build a multi-page interactive Streamlit dashboard that makes all audit findings available to both technical and non-technical stakeholders, including a plain-English AI-generated summary.

6. To produce compliance-ready governance exports in PDF and JSON formats suitable for regulatory audit trails.

---

## 3. Research Papers

### Research Paper 1
**Understanding Bias in Large-Scale Visual Datasets (2024)**

**Overview:**
This work argues that the real manifestations of bias in massive datasets remain insufficiently understood. It decomposes bias into semantic, structural, frequency, and representation dimensions and calls for systematic auditing mechanisms.

**Gaps identified by the authors:**
- No automated way to prioritize harmful disparities.
- Heavy reliance on human interpretation.
- Limited scalability to very large corpora.

**Relation to our work:**
Our framework directly addresses the automation and prioritization gap. Rather than leaving bias ranking to human judgment, SABPF computes a quantitative Bias Severity Score that objectively determines which disparities should be addressed first.

---

### Research Paper 2
**REVISE: A Tool for Measuring Bias in Visual Datasets (2022)**

**Overview:**
REVISE uncovers dataset irregularities related to objects, persons, and geographic attributes. However, responsibility for judging discrimination remains with users.

**Gaps identified:**
- Surfaces anomalies but cannot confirm harm.
- No automatic severity ranking.
- Primarily focused on images and visual corpora.

**Relation to our work:**
SABPF introduces statistical validation and numeric severity scoring at each detected disparity, and is designed explicitly for structured tabular datasets — the modality used in lending, hiring, and criminal justice — making it applicable where REVISE cannot function.

---

### Research Paper 3
**On Detecting Biased Predictions with Post-hoc Explanations (2023)**

**Overview:**
The authors show that explanation techniques can capture global patterns but often miss bias affecting individual or minority subgroups.

**Gaps identified:**
- Weak fine-grained detection at intersectional subgroup level.
- Limited suitability for operational auditing workflows.

**Relation to our work:**
Our framework replaces explanation-only strategies with exhaustive automated subset search — covering all single-attribute and all 2-way intersectional combinations of protected attributes — combined with formal hypothesis testing that produces statistically defensible findings rather than interpretability approximations.

---

## 4. Consolidated Research Gaps

Based on the reviewed literature, the following unresolved challenges persist in the algorithmic fairness field:

**Gap 1 — Absence of Automated Prioritization**
Existing detection tools flag disparities but offer no systematic mechanism to determine which bias requires immediate intervention. Practitioners must manually sort and interpret findings, which is unscalable.

**Gap 2 — Strong Dependence on Human Judgment**
Manual interpretation introduces subjectivity and inconsistency. Two auditors reviewing the same findings may reach different prioritization conclusions.

**Gap 3 — Limited Fine-Grained Intersectional Discovery**
Most tools examine one protected attribute at a time. Intersectional subpopulations — e.g., young + blue-collar, or female + single — remain systematically under-examined and are often where bias is most severe.

**Gap 4 — Modality Restriction**
The majority of mature bias detection tools are optimized for image and text datasets. Structured tabular data, which drives financial and judicial decision-making, lacks purpose-built auditing infrastructure.

**Gap 5 — Weak Operational Readiness**
Few bias detection methods offer repeatable, automated pipelines with statistical guarantees, governance-ready exports, or multi-model comparison capabilities needed for enterprise deployment.

---

## 5. Proposed Solution

We propose and implement the Scalable Automated Bias Prioritization Framework (SABPF), a complete Python-based system that converts bias detection from a manual diagnostic task into an automated, statistically rigorous prioritization workflow.

The framework is organized into four primary components:

1. **Automated Subset Discovery Engine** — identifies all demographic subgroups exhibiting outcome disparities, including intersectional combinations.
2. **Statistical Validation Layer** — verifies that each finding is statistically significant and not a product of random sampling variation.
3. **Bias Severity Score (BSS) Ranking Module** — computes a composite severity score and assigns each finding to a Critical, High, Medium, or Low priority tier.
4. **Interactive Reporting Interface** — presents findings through a professional Streamlit dashboard and produces compliance-grade governance exports.

Instead of merely highlighting anomalies, SABPF guides stakeholders directly toward where intervention is most critical — converting audit output into actionable decision support.

---

## 6. Implementation

### 6.1 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Model Training | Scikit-learn, XGBoost |
| Statistical Analysis | SciPy, Statsmodels |
| Visualization | Plotly (all interactive charts) |
| Dashboard | Streamlit |
| Report Export | ReportLab (PDF), JSON |
| AI-Powered Summary | Groq / Google Gemini API |

---

### 6.2 System Architecture

```
sabpf/
├── data/
│   ├── loaders.py        — Dataset downloaders (Adult Income, COMPAS, Bank Marketing,
│   │                        German Credit, Law School, Custom CSV)
│   └── preprocessor.py   — Cleaning, encoding, stratified train/test splitting
├── models/
│   ├── trainer.py        — Multi-model training pipeline + joblib persistence
│   └── evaluator.py      — Accuracy, AUC-ROC, F1, Precision, Recall, Brier Score
├── bias/
│   ├── detector.py       — Subgroup discovery (single + intersectional)
│   ├── metrics.py        — DPD, DIR, EOD, PPV, FNR, Calibration Error
│   ├── statistical.py    — Z-test, Cohen's h, Bootstrap CI, FDR Correction
│   ├── ranker.py         — BSS formula + Priority Tier assignment
│   ├── explainer.py      — AI-powered plain-English audit summary
│   └── ai_assistant.py   — Multi-provider AI connector (Groq / Gemini)
├── reports/
│   └── generator.py      — PDF and JSON governance report export
├── ui/
│   └── app.py            — Streamlit multi-page dashboard (main entry point)
├── requirements.txt
└── README.md
```

---

### 6.3 Datasets Used (Real Data Only — No Synthetic Data)

| Dataset | Domain | Protected Attributes | Rows |
|---|---|---|---|
| Adult Income (UCI) | Financial / Hiring | Gender, Race, Age | 48,842 |
| COMPAS Recidivism | Criminal Justice | Race, Gender, Age | 7,214 |
| German Credit | Lending | Age, Sex | 1,000 |
| Bank Marketing (UCI) | Financial | Age, Marital Status, Job | 41,188 |
| Law School Admissions | Education | Race, Gender | 20,649 |
| Custom CSV Upload | User-defined | User-defined | Variable |

---

### 6.4 Machine Learning Pipeline

Five distinct model architectures are trained per dataset using a stratified 80/20 train-test split:

| Model | Configuration | Purpose |
|---|---|---|
| Logistic Regression | C=1.0, max_iter=2000, with StandardScaler | Linear interpretable baseline |
| Decision Tree | max_depth=12, min_samples_split=10 | Fully transparent rule-based model |
| Random Forest | n_estimators=300, max_depth=20 | Ensemble variance reduction |
| XGBoost | n_estimators=400, learning_rate=0.08, max_depth=7 | Highest predictive accuracy |
| SVM (RBF Kernel) | C=1.5, probability=True, with StandardScaler | Non-linear margin-based classifier |

Each trained model is saved to disk using `joblib` for re-use without retraining. Evaluation metrics computed per model:
- Accuracy, F1 Score, Precision, Recall
- AUC-ROC with ROC Curve overlay plot
- Brier Score (probability calibration quality)

---

### 6.5 Bias Detection Engine

**Single-Attribute Subgroup Scanning:**
For every protected column, the engine separates the test set into subsets per unique value (e.g., Gender=Male, Gender=Female) and computes all fairness metrics for each subset compared to the overall population baseline.

**Intersectional Subgroup Scanning:**
All 2-way combinations of protected attributes are enumerated (e.g., Gender × Age, Race × Marital Status). Subgroups with fewer than 30 samples are silently skipped to avoid statistically unstable estimates.

**Fairness Metrics Computed per Subgroup:**

| Metric | Formula | Threshold |
|---|---|---|
| DPD — Demographic Parity Difference | P(ŷ=1\|group) − P(ŷ=1\|overall) | \|DPD\| > 0.10 |
| DIR — Disparate Impact Ratio | P(ŷ=1\|group) / P(ŷ=1\|overall) | DIR < 0.8 or > 1.2 |
| EOD — Equalized Odds Difference | max(\|TPR gap\|, \|FPR gap\|) | EOD > 0.10 |
| PPV Gap — Predictive Parity | PPV(subgroup) − PPV(overall) | — |
| FNR Disparity | FNR(subgroup) − FNR(overall) | — |

---

### 6.6 Statistical Validation Layer

Every detected disparity is validated before being ranked. This prevents spurious findings that arise from sampling noise when many subgroups are tested simultaneously.

**Two-Proportion Z-Test:** Tests whether the selection rate difference between the subgroup and the overall population is statistically significant. Raw p-values are collected for all findings simultaneously.

**Cohen's h Effect Size:** An arcsine-transformed measure of the practical magnitude of the proportion difference, independent of sample size. Classified as:
- Negligible: h < 0.1
- Small: 0.1 ≤ h < 0.3
- Medium: 0.3 ≤ h < 0.5
- Large: h ≥ 0.5

**Bootstrap Confidence Intervals:** 200 resampling iterations produce a 95% CI around the DPD estimate for each subgroup.

**Benjamini-Hochberg FDR Correction:** Applied collectively across all raw p-values to control the False Discovery Rate. Only findings with corrected p-value < 0.05 contribute to statistical confidence in the BSS score.

---

### 6.7 Bias Severity Score (BSS) — Core Innovation

The BSS is a composite 0-to-1 risk score that aggregates four dimensions of bias severity into a single, rankable number:

```
BSS = 0.35 × |Metric Deviation|
    + 0.25 × Effect Size (Cohen's h, normalised)
    + 0.25 × Statistical Confidence (1 − p_corrected, only if p < 0.05)
    + 0.15 × Subgroup Exposure (subgroup_n / total_n)
```

**Component Rationale:**

| Component | Weight | What It Captures |
|---|---|---|
| Metric Deviation | 35% | Raw size of the DPD bias — how far from zero |
| Effect Size | 25% | Practical significance — is the gap large or trivial |
| Statistical Confidence | 25% | Is the finding real or noise (FDR-corrected) |
| Exposure Weight | 15% | Larger populations affected = higher urgency |

**Priority Tier Assignment:**

| BSS Score | Priority Tier | Meaning |
|---|---|---|
| ≥ 0.40 | Critical | Immediate intervention required |
| ≥ 0.25 | High | Should be addressed in next model update |
| ≥ 0.10 | Medium | Monitor — may worsen with scale |
| < 0.10 | Low | Negligible, statistically weak |

---

### 6.8 AI-Powered Plain-English Summary

A dedicated "Summarize Bias Analysis" feature passes the full audit context — all findings, tier breakdowns, worst offenders, model accuracy, and dataset name — to a large language model (Groq or Gemini) via a structured fairness-expert prompt.

The AI generates a professional, plain-English four-section report covering:
1. Overview of what was audited
2. Key findings in everyday language (e.g., "customers aged 20–30 receive offers 24% less often")
3. Severity assessment without technical jargon
4. Three concrete recommended actions

This feature makes the audit accessible to non-technical stakeholders such as compliance officers and management without losing any factual accuracy.

---

### 6.9 Interactive Streamlit Dashboard

The dashboard provides seven navigation pages:

| Page | Functionality |
|---|---|
| Dataset Explorer | Class balance, protected attribute distributions, feature statistics |
| Model Training | Train/load any of 5 models, view results table with AUC/F1 heatmap, ROC curves |
| Bias Analysis | Run audit, ranked findings table, severity heatmap treemap, AI plain-English summary |
| Intersectional Explorer | Top-10 intersectional BSS rankings, bar chart visualization |
| Statistical Report | P-value histograms (raw vs FDR-corrected), Cohen's h distribution, full statistics table |
| Gov Report Export | One-click PDF + JSON governance artifact generation and download |
| SABPF AI Assistant | Context-aware conversational AI for on-demand project and findings explanation |

---

### 6.10 Governance Report Export

Two export formats are generated for compliance use:

**PDF Report** (generated via ReportLab):
- Executive summary
- Dataset and model metadata
- Full ranked findings table with Priority tiers
- Remediation recommendations per Critical/High finding

**JSON Export:**
- Machine-readable structured output suitable for downstream compliance systems
- Includes all raw metric values, corrected p-values, BSS scores, and CI bounds

---

## 7. Results

### Model Performance (Bank Marketing Dataset — Representative)

| Model | Accuracy | AUC-ROC | F1 Score |
|---|---|---|---|
| Logistic Regression | 89.8% | 0.880 | 0.406 |
| Decision Tree | 89.7% | 0.833 | 0.495 |
| Random Forest | 90.5% | 0.920 | 0.492 |
| XGBoost | **90.6%** | **0.923** | **0.536** |
| SVM (RBF) | 89.7% | 0.833 | 0.381 |

### Bias Detection (XGBoost on Bank Marketing)

- Total subgroups analysed: 14
- Critical findings: 3
- High findings: 6
- Most affected subgroup: Age group 20–30 — DPD of −0.24 (24% fewer positive outcomes)
- Strongest intersectional bias: Young + Blue-Collar — BSS 0.62

---

## 8. Novelty

SABPF advances the existing fairness auditing landscape in the following ways:

1. **Shifts the paradigm from identification to prioritization.** Rather than surfacing all anomalies equally, SABPF ranks findings so the most harmful biases are addressed first.

2. **Introduces a quantitative, reproducible risk score.** The BSS formula replaces subjective manual assessment with a transparent, weighted formula with clearly defined component thresholds.

3. **Enables discovery of hidden intersectional bias.** The framework automatically enumerates all 2-way attribute combinations, revealing compounded disadvantages invisible to single-attribute tools.

4. **Provides full statistical grounding.** Every finding passes through Z-testing, effect size classification, bootstrap CI estimation, and FDR correction before being reported — preventing false alarms at scale.

5. **Supplies AI-powered non-technical translation.** A built-in LLM plain-English summary makes findings immediately actionable for compliance officers and management, not just data scientists.

6. **Delivers governance-ready outputs.** PDF and JSON exports formatted to meet audit trail requirements of the EU AI Act and EEOC compliance frameworks.

---

## 9. Expected Outcome and Contribution

The SABPF system delivers a working prototype capable of:
- Automatically discovering both single-attribute and intersectional subgroup disparities across five real-world benchmark datasets.
- Validating all findings with formal statistical tests and FDR correction before reporting.
- Ranking every finding by a composite severity score, eliminating alert fatigue for auditors.
- Producing governance-ready compliance exports and AI-written plain-English summaries.

This significantly reduces the expert burden in bias auditing while improving reliability, reproducibility, and fairness governance across the full machine learning deployment lifecycle.

---

## 10. Technical Stack Summary

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.11+ |
| ML Models | Scikit-learn, XGBoost |
| Statistical Tests | SciPy, Statsmodels |
| Visualization | Plotly, Streamlit |
| AI Summary | Groq (llama3-8b-8192), Google Gemini |
| Report Generation | ReportLab, JSON |
| Data Persistence | Joblib (model files), Pandas (dataframes) |

---

**Signatures:**

Student: Gaurav Patil (1272250165) ____________________

Student: Ayush Patel (1272250207) ____________________

Guide: P. S. Metkewar Sir ____________________

Co-Guide: Navnath Shete Sir ____________________
