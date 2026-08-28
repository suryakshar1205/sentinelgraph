import json
import pandas as pd
import numpy as np

# 1. Check cases
with open("cases/artifacts/cases.json") as f:
    cases = json.load(f)
print(f"Actual regenerated case count: {len(cases)}")

# 2. Check ablation metrics & confusion matrix
with open("evaluation/artifacts/ablation_results.json") as f:
    ablation = json.load(f)

c_metrics = ablation["stages"]["stage_c_sentinelgraph"]["classification"]
print("\nStage C Classification Metrics & Confusion Matrix:")
print(f"  TP: {c_metrics['tp']}")
print(f"  FP: {c_metrics['fp']}")
print(f"  TN: {c_metrics['tn']}")
print(f"  FN: {c_metrics['fn']}")
print(f"  Total: {c_metrics['tp'] + c_metrics['fp'] + c_metrics['tn'] + c_metrics['fn']}")
print(f"  Actual Frauds (TP + FN): {c_metrics['tp'] + c_metrics['fn']}")
print(f"  Actual Legitimate (TN + FP): {c_metrics['tn'] + c_metrics['fp']}")

# Mathematical verification
prec = c_metrics['tp'] / (c_metrics['tp'] + c_metrics['fp'])
rec = c_metrics['tp'] / (c_metrics['tp'] + c_metrics['fn'])
f1 = 2 * prec * rec / (prec + rec)
fpr = c_metrics['fp'] / (c_metrics['tn'] + c_metrics['fp'])

print(f"\nCalculated from Confusion Matrix:")
print(f"  Precision: {prec:.4f} (reported: {c_metrics['precision']:.4f})")
print(f"  Recall:    {rec:.4f} (reported: {c_metrics['recall']:.4f})")
print(f"  F1 Score:  {f1:.4f} (reported: {c_metrics['f1']:.4f})")
print(f"  FPR:       {fpr*100:.2f}% (reported: {c_metrics['fpr']*100:.2f}%)")
