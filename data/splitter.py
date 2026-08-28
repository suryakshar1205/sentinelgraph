"""
Strict chronological train/validation/test splitter for SentinelGraph.
Ensures frozen held-out test set with zero lookahead leakage.
"""

import os
import json
import pandas as pd
from typing import Tuple


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = "data/generated"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically to prevent temporal lookahead leakage."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Ensure ordered by timestamp
    df = df.sort_values(by="timestamp_unix").reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    # Save splits
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test_frozen.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    split_meta = {
        "train": {
            "count": len(train_df),
            "start": train_df["timestamp"].min(),
            "end": train_df["timestamp"].max(),
            "fraud_rate": float(train_df["is_fraud"].mean()),
            "rings": int(train_df["ring_id"].nunique())
        },
        "validation": {
            "count": len(val_df),
            "start": val_df["timestamp"].min(),
            "end": val_df["timestamp"].max(),
            "fraud_rate": float(val_df["is_fraud"].mean()),
            "rings": int(val_df["ring_id"].nunique())
        },
        "test_frozen": {
            "count": len(test_df),
            "start": test_df["timestamp"].min(),
            "end": test_df["timestamp"].max(),
            "fraud_rate": float(test_df["is_fraud"].mean()),
            "rings": int(test_df["ring_id"].nunique())
        }
    }
    
    with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)
        
    print(f"Dataset split saved:")
    print(f"  Train: {len(train_df):,} rows [{train_df['timestamp'].min()} -> {train_df['timestamp'].max()}]")
    print(f"  Val:   {len(val_df):,} rows [{val_df['timestamp'].min()} -> {val_df['timestamp'].max()}]")
    print(f"  Test:  {len(test_df):,} rows [{test_df['timestamp'].min()} -> {test_df['timestamp'].max()}] (FROZEN)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = pd.read_parquet("data/generated/transactions.parquet")
    split_dataset(df)
