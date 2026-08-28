"""
Master Runner & CLI Orchestrator for SentinelGraph.
Razorpay Buildathon 2026 — Track 02: AI Risk Manager.
"""

import sys
import os
import argparse
import subprocess
import pandas as pd

from data.generator import SyntheticDataGenerator
from data.splitter import split_dataset
from models.baseline import TransactionBaselineModel
from graph.builder import EntityGraphBuilder
from graph.communities import GraphCommunityDetector
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from features.pipeline import FeaturePipeline
from fusion.risk import RiskFusionEngine
from cases.builder import RiskCaseBuilder
from cases.store import RiskCaseStore
from evaluation.ablation import AblationRunner


def cmd_generate():
    """Generate 100,000+ synthetic transactions and create chronological splits."""
    print("\n[1/2] Generating synthetic transaction ecosystem (100k+ events)...")
    gen = SyntheticDataGenerator()
    df = gen.generate(output_dir="data/generated")
    
    print("\n[2/2] Splitting into 70% Train / 15% Validation / 15% Frozen Held-Out Test...")
    split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, output_dir="data/generated")
    print("[SUCCESS] Data generation and chronological splitting complete!\n")


def cmd_train():
    """Train transaction baseline model and calibrate probabilities."""
    train_path = "data/generated/train.parquet"
    val_path = "data/generated/val.parquet"
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Data files not found. Running data generation first...")
        cmd_generate()
        
    print("\nTraining Transaction-Level Baseline Model (Behavioral features only)...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    bm = TransactionBaselineModel()
    metrics = bm.fit(train_df, val_df)
    bm.save("models/artifacts/baseline_model.joblib")
    print("[SUCCESS] Baseline model training and calibration complete!\n")


def cmd_build_cases():
    """Build structured risk cases from detected clusters across dataset."""
    print("\nBuilding structured Risk Cases from candidate graph clusters...")
    test_path = "data/generated/test_frozen.parquet"
    if not os.path.exists(test_path):
        print("Test dataset not found. Running data generation first...")
        cmd_generate()
        
    test_df = pd.read_parquet(test_path)
    bm_path = "models/artifacts/baseline_model.joblib"
    if not os.path.exists(bm_path):
        cmd_train()
    baseline_model = TransactionBaselineModel.load(bm_path)

    builder = EntityGraphBuilder()
    graph = builder.build_from_dataframe(test_df)
    
    detector = GraphCommunityDetector(min_cluster_accounts=3)
    clusters = detector.extract_candidate_clusters(graph)
    
    pipeline = FeaturePipeline()
    g_scorer = GraphSignalScorer()
    t_scorer = TemporalEscalationScorer()
    fusion = RiskFusionEngine()
    case_builder = RiskCaseBuilder()
    case_store = RiskCaseStore()

    # Clear previous
    case_store.cases.clear()

    print(f"Discovered {len(clusters)} candidate clusters in held-out test data. Synthesizing risk cases...")
    for idx, cl in enumerate(clusters):
        case_id = f"10{idx+42:02d}"
        
        # Get member transactions
        c_accs = [n.replace("CUST:", "") for n in cl["accounts"]]
        cl_txs = test_df[test_df["customer_id"].isin(c_accs)].copy()
        
        if cl_txs.empty:
            continue
            
        beh_df, ent_df, tem_df = pipeline.extract_full_features(cl_txs)
        beh_probs = baseline_model.predict_proba(cl_txs)
        g_scores = g_scorer.compute_graph_scores_for_dataframe(ent_df)
        t_scores = t_scorer.compute_escalation_scores_for_dataframe(tem_df)
        
        fused_df = fusion.fuse_dataframe(beh_probs, g_scores, t_scores)
        
        # Aggregate cluster metrics
        cluster_fused_summary = {
            "behavioral_score": float(fused_df["behavioral_score"].mean()),
            "graph_score": float(fused_df["graph_score"].mean()),
            "temporal_score": float(fused_df["temporal_score"].mean()),
            "final_risk_score": float(fused_df["final_risk_score"].max()),
            "confidence": float(fused_df["confidence"].mean()),
            "severity": fused_df.loc[fused_df["final_risk_score"].idxmax(), "severity"],
            "recommended_action": fused_df.loc[fused_df["final_risk_score"].idxmax(), "recommended_action"]
        }
        
        # Infer ring_id if majority labeled
        r_ids = cl_txs["ring_id"].dropna().values
        cl["ring_id"] = str(r_ids[0]) if len(r_ids) > 0 else None

        risk_case = case_builder.build_case_from_cluster(
            case_id=case_id,
            cluster_data=cl,
            cluster_txs_df=cl_txs,
            fused_row=cluster_fused_summary
        )
        case_store.add_case(risk_case)

    case_store.save()
    print(f"[SUCCESS] Generated and persisted {len(case_store.list_cases())} risk cases to cases/artifacts/cases.json!\n")


def cmd_evaluate():
    """Run formal 3-stage ablation on frozen test set."""
    test_path = "data/generated/test_frozen.parquet"
    bm_path = "models/artifacts/baseline_model.joblib"
    
    if not os.path.exists(test_path):
        cmd_generate()
    if not os.path.exists(bm_path):
        cmd_train()

    test_df = pd.read_parquet(test_path)
    baseline_model = TransactionBaselineModel.load(bm_path)
    
    runner = AblationRunner()
    results = runner.run_ablation(test_df, baseline_model)
    return results


def cmd_all():
    """Run full end-to-end pipeline."""
    cmd_generate()
    cmd_train()
    cmd_build_cases()
    cmd_evaluate()
    print("\n[DONE] SentinelGraph End-to-End Pipeline Complete!")
    print("Launch dashboard with: python run.py app\n")


def main():
    parser = argparse.ArgumentParser(description="SentinelGraph CLI")
    parser.add_argument("command", choices=["generate", "train", "build-cases", "evaluate", "app", "api", "all"], help="Command to run")
    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate()
    elif args.command == "train":
        cmd_train()
    elif args.command == "build-cases":
        cmd_build_cases()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "all":
        cmd_all()
    elif args.command == "app":
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])
    elif args.command == "api":
        subprocess.run(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        cmd_all()
    else:
        main()
