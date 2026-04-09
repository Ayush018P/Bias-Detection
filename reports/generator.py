"""
reports/generator.py
PDF and JSON export capabilities for the Governance Report.
"""

import os
import json
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def export_json(results_df: pd.DataFrame, model_name: str, dataset_name: str, filepath: str):
    """Exports findings to a machine-readable JSON."""
    if not results_df.empty and "Priority" in results_df.columns:
        crit_high = results_df[results_df["Priority"].isin(["Critical", "High"])]
    else:
        crit_high = pd.DataFrame()
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "dataset": dataset_name,
            "total_findings": len(results_df),
            "critical_findings": len(results_df[results_df["Priority"] == "Critical"])
        },
        "top_findings": crit_high.to_dict(orient="records")
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
        
def export_pdf(results_df: pd.DataFrame, metrics: dict, model_name: str, dataset_name: str, filepath: str):
    """Generates a governance PDF report using ReportLab."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    story.append(Paragraph("SABPF Governance Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Meta
    story.append(Paragraph(f"<b>Dataset:</b> {dataset_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Model:</b> {model_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Model Perf table
    story.append(Paragraph("<b>Model Performance</b>", styles['Heading2']))
    perf_data = [["Metric", "Value"]]
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and dict != type(v):
            perf_data.append([k, f"{v:.4f}"])
            
    t1 = Table(perf_data, colWidths=[150, 100])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4f46e5")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0"))
    ]))
    story.append(t1)
    story.append(Spacer(1, 20))
    
    # Bias Table
    story.append(Paragraph("<b>Critical & High Bias Findings</b>", styles['Heading2']))
    
    crit_high = results_df[results_df["Priority"].isin(["Critical", "High"])]
    if crit_high.empty:
        story.append(Paragraph("No critical or high severity biases detected.", styles['Normal']))
    else:
        headers = ["Rank", "Subgroup", "Priority", "BSS", "DPD"]
        table_data = [headers]
        
        for _, row in crit_high.head(15).iterrows():
            table_data.append([
                str(row["Rank"]),
                str(row["Subgroup Name"])[:25] + ".." if len(str(row["Subgroup Name"])) > 25 else str(row["Subgroup Name"]),
                str(row["Priority"]),
                f"{row['BSS']:.2f}",
                f"{row['DPD']:.3f}"
            ])
            
        t2 = Table(table_data, colWidths=[40, 150, 60, 50, 50])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0"))
        ]))
        story.append(t2)
        
    doc.build(story)
