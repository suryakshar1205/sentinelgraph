"""
Deterministic synthetic dataset generator for SentinelGraph.
Generates 100,000+ realistic heterogeneous transactions with ground truth.
"""

import os
import json
import yaml
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from data.patterns import PatternGenerator


class SyntheticDataGenerator:
    """Generates 100,000+ deterministic transactions with ground truth."""

    def __init__(self, config_path: str = "config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "project": {"seed": 20260827},
                "data": {
                    "total_transactions": 100000,
                    "start_date": "2026-08-01T00:00:00",
                    "duration_days": 14,
                    "num_merchants": 50,
                    "num_customers": 15000,
                }
            }
            
        self.seed = int(self.config["project"]["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.start_time = datetime.datetime.fromisoformat(self.config["data"]["start_date"])
        self.duration_days = int(self.config["data"]["duration_days"])
        self.target_tx_count = int(self.config["data"]["total_transactions"])
        self.num_merchants = int(self.config["data"]["num_merchants"])
        self.num_customers = int(self.config["data"]["num_customers"])
        
        self.pattern_gen = PatternGenerator(self.rng, self.start_time)
        self.merchants = [f"merch_{i:04d}" for i in range(self.num_merchants)]

    def generate(self, output_dir: str = "data/generated") -> pd.DataFrame:
        """Generate complete dataset and write metadata."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating synthetic dataset with seed={self.seed}, target={self.target_tx_count} transactions...")

        # 1. Generate Legitimate Customer Profiles
        customer_profiles = {}
        for i in range(self.num_customers):
            c_id = f"cust_{i:06d}"
            customer_profiles[c_id] = self.pattern_gen.generate_legitimate_user_profile(c_id)

        # 2. Generate Base Legitimate Stream
        legit_records = []
        cust_keys = list(customer_profiles.keys())
        
        # Calculate daily volume to reach target
        # Coordinated rings will add ~6,000 txs, isolated anomalies ~1,500 txs
        target_legit = max(90000, self.target_tx_count - 8000)
        
        print("Generating legitimate customer transactions...")
        for c_id in cust_keys:
            prof = customer_profiles[c_id]
            # Number of transactions for this user over duration_days
            num_tx = max(1, int(self.rng.poisson(prof["tx_frequency_per_day"] * self.duration_days)))
            
            for _ in range(num_tx):
                # Sample random day and time
                day_offset = float(self.rng.uniform(0, self.duration_days))
                pref_hour = int(self.rng.choice(prof["pref_hours"]))
                minute = int(self.rng.integers(0, 60))
                second = int(self.rng.integers(0, 60))
                
                tx_time = self.start_time + datetime.timedelta(
                    days=day_offset, hours=pref_hour - self.start_time.hour, minutes=minute, seconds=second
                )
                
                amt = float(max(10.0, self.rng.normal(prof["mean_amt"], prof["std_amt"])))
                
                # 90% primary device/IP, 10% mobile travel/secondary
                if self.rng.random() < 0.90:
                    dev = prof["primary_device"]
                    ip = prof["primary_ip"]
                    card = prof["primary_card"]
                    loc = prof["home_city"]
                else:
                    dev = f"dev_{self.rng.integers(100000, 999999)}"
                    ip = f"ip_{self.rng.integers(10000, 99999)}"
                    card = prof["primary_card"]
                    loc = str(self.rng.choice(self.pattern_gen.cities))

                legit_records.append({
                    "timestamp": tx_time.isoformat(),
                    "timestamp_unix": tx_time.timestamp(),
                    "customer_id": c_id,
                    "account_created_at": prof["account_created_at"],
                    "account_created_unix": prof["account_created_unix"],
                    "device_id": dev,
                    "ip_id": ip,
                    "payment_instrument_id": card,
                    "merchant_id": str(self.rng.choice(self.merchants)),
                    "amount": round(amt, 2),
                    "location": loc,
                    "transaction_type": "payment",
                    "label": "legitimate",
                    "abuse_type": "none",
                    "ring_id": None,
                    "is_fraud": 0
                })

        # 3. Inject Isolated Anomalies (~1.5% of legitimate stream)
        print("Injecting isolated transaction anomalies...")
        num_isolated = int(len(legit_records) * 0.015)
        isolated_indices = self.rng.choice(len(legit_records), size=num_isolated, replace=False)
        for idx in isolated_indices:
            legit_records[idx] = self.pattern_gen.generate_isolated_anomaly(legit_records[idx])

        # 4. Inject Coordinated Abuse Rings (Patterns A through H)
        print("Injecting coordinated abuse rings (Patterns A - H)...")
        ring_records = []
        patterns = [
            "pattern_a_shared_device",
            "pattern_b_shared_ip",
            "pattern_c_shared_payment_instrument",
            "pattern_d_rapid_account_creation",
            "pattern_e_synchronized_activity",
            "pattern_f_gradual_growth",
            "pattern_g_sudden_activation",
            "pattern_h_mixed_cluster"
        ]
        
        # Inject multiple instances of each pattern
        ring_counter = 1001
        for day in range(1, self.duration_days - 1, 2):
            base_time = self.start_time + datetime.timedelta(days=day)
            for pat in patterns:
                ring_id = f"RING_{ring_counter:04d}"
                ring_counter += 1
                r_records = self.pattern_gen.create_coordinated_ring(
                    ring_id=ring_id,
                    pattern_type=pat,
                    base_time=base_time,
                    merchants=self.merchants
                )
                ring_records.extend(r_records)

        # 5. Combine and Sort Chronologically
        all_records = legit_records + ring_records
        df = pd.DataFrame(all_records)
        df.sort_values(by="timestamp_unix", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Assign unique transaction IDs
        df["transaction_id"] = [f"tx_{i:08d}" for i in range(1, len(df) + 1)]

        # Reorder columns
        cols = [
            "transaction_id", "timestamp", "timestamp_unix", "amount", "merchant_id",
            "customer_id", "account_created_at", "account_created_unix",
            "device_id", "ip_id", "payment_instrument_id", "location", "transaction_type",
            "label", "abuse_type", "ring_id", "is_fraud"
        ]
        df = df[cols]

        print(f"Total transactions generated: {len(df):,}")
        print(f"Legitimate count: {(df['label'] == 'legitimate').sum():,}")
        print(f"Isolated anomaly count: {(df['label'] == 'isolated_anomaly').sum():,}")
        print(f"Coordinated ring count: {(df['label'] == 'coordinated_ring').sum():,}")
        print(f"Total Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")

        # Save to disk
        parquet_path = os.path.join(output_dir, "transactions.parquet")
        csv_path = os.path.join(output_dir, "transactions_sample.csv")
        meta_path = os.path.join(output_dir, "dataset_metadata.json")

        df.to_parquet(parquet_path, index=False)
        df.head(5000).to_csv(csv_path, index=False)

        metadata = {
            "dataset_version": "1.0.0",
            "generation_seed": self.seed,
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "total_transactions": len(df),
            "fraud_rate": float(df["is_fraud"].mean()),
            "num_coordinated_rings": int(df["ring_id"].nunique()),
            "schema_version": "2026.08.27",
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max()
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset successfully saved to {parquet_path} and {meta_path}")
        return df


if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    gen.generate()
