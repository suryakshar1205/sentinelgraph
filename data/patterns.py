"""
Abuse and legitimate behavior pattern generators for SentinelGraph.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import datetime


class PatternGenerator:
    """Generates realistic heterogeneous legitimate transactions and coordinated abuse rings."""

    def __init__(self, rng: np.random.Generator, start_time: datetime.datetime):
        self.rng = rng
        self.start_time = start_time
        
        self.cities = [
            "Mumbai", "Bengaluru", "Delhi NCR", "Hyderabad", "Chennai", 
            "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"
        ]
        
        self.payment_types = ["upi", "credit_card", "debit_card", "netbanking", "wallet"]

    def generate_legitimate_user_profile(self, customer_id: str) -> Dict:
        """Create diverse legitimate behavioral profile."""
        segment_type = self.rng.choice(
            ["daily_upi", "business_high_val", "weekend_shopper", "night_owl", "casual_low_freq"],
            p=[0.35, 0.15, 0.20, 0.10, 0.20]
        )
        
        # Account creation 10 to 365 days before start_time
        account_age_days = float(self.rng.uniform(10, 365))
        created_at = self.start_time - datetime.timedelta(days=account_age_days)
        
        # Profile specific parameters
        if segment_type == "daily_upi":
            mean_amt = float(self.rng.uniform(150, 600))
            std_amt = mean_amt * 0.4
            pref_hours = [8, 9, 12, 13, 19, 20, 21]
            tx_frequency_per_day = float(self.rng.uniform(1.5, 4.0))
            pref_payment = "upi"
        elif segment_type == "business_high_val":
            mean_amt = float(self.rng.uniform(8000, 25000))
            std_amt = mean_amt * 0.5
            pref_hours = [10, 11, 14, 15, 16, 17]
            tx_frequency_per_day = float(self.rng.uniform(0.5, 1.5))
            pref_payment = "credit_card"
        elif segment_type == "weekend_shopper":
            mean_amt = float(self.rng.uniform(1200, 4500))
            std_amt = mean_amt * 0.45
            pref_hours = [11, 12, 15, 16, 18, 19, 20]
            tx_frequency_per_day = float(self.rng.uniform(0.3, 1.0))
            pref_payment = "credit_card"
        elif segment_type == "night_owl":
            mean_amt = float(self.rng.uniform(400, 1800))
            std_amt = mean_amt * 0.35
            pref_hours = [22, 23, 0, 1, 2]
            tx_frequency_per_day = float(self.rng.uniform(0.4, 1.2))
            pref_payment = "upi"
        else:  # casual_low_freq
            mean_amt = float(self.rng.uniform(300, 2000))
            std_amt = mean_amt * 0.5
            pref_hours = list(range(9, 22))
            tx_frequency_per_day = float(self.rng.uniform(0.1, 0.5))
            pref_payment = "debit_card"
            
        home_city = str(self.rng.choice(self.cities))
        primary_device = f"dev_{self.rng.integers(100000, 999999)}"
        primary_ip = f"ip_{self.rng.integers(10000, 99999)}"
        primary_card = f"card_{self.rng.integers(100000, 999999)}"

        return {
            "customer_id": customer_id,
            "segment_type": segment_type,
            "account_created_at": created_at.isoformat(),
            "account_created_unix": created_at.timestamp(),
            "mean_amt": max(50.0, mean_amt),
            "std_amt": max(20.0, std_amt),
            "pref_hours": pref_hours,
            "tx_frequency_per_day": tx_frequency_per_day,
            "home_city": home_city,
            "primary_device": primary_device,
            "primary_ip": primary_ip,
            "primary_card": primary_card,
            "pref_payment": pref_payment
        }

    def generate_isolated_anomaly(self, tx_dict: Dict) -> Dict:
        """Create an isolated anomaly (e.g. single large sudden purchase, unfamiliar device)."""
        tx = dict(tx_dict)
        tx["amount"] = float(tx["amount"] * self.rng.uniform(8.0, 18.0))
        tx["label"] = "isolated_anomaly"
        tx["abuse_type"] = "none"
        tx["ring_id"] = None
        tx["is_fraud"] = 1
        # Use random location / device
        tx["device_id"] = f"dev_unfamiliar_{self.rng.integers(10000, 99999)}"
        return tx

    def create_coordinated_ring(
        self,
        ring_id: str,
        pattern_type: str,
        base_time: datetime.datetime,
        merchants: List[str]
    ) -> List[Dict]:
        """Generate a fully interconnected coordinated abuse ring based on the target pattern."""
        records = []
        
        if pattern_type == "pattern_a_shared_device":
            # 15-25 accounts sharing 2-3 devices
            num_accounts = int(self.rng.integers(15, 25))
            devices = [f"ring_dev_{ring_id}_{i}" for i in range(2)]
            ip_pool = [f"ring_ip_{ring_id}_{i}" for i in range(4)]
            cards = [f"ring_card_{ring_id}_{i}" for i in range(num_accounts)]
            
            ring_start = base_time + datetime.timedelta(hours=float(self.rng.uniform(10, 40)))
            for acc_idx in range(num_accounts):
                cust_id = f"cust_{ring_id}_{acc_idx:03d}"
                # Accounts created just hours before ring activity
                acc_created = ring_start - datetime.timedelta(hours=float(self.rng.uniform(2, 24)))
                dev = str(self.rng.choice(devices))
                ip = str(self.rng.choice(ip_pool))
                card = cards[acc_idx]
                
                # Each account makes 2-5 transactions in a burst
                num_tx = int(self.rng.integers(2, 6))
                for t_idx in range(num_tx):
                    tx_time = ring_start + datetime.timedelta(
                        minutes=float(self.rng.uniform(0, 180)) + (t_idx * float(self.rng.uniform(5, 25)))
                    )
                    records.append({
                        "timestamp": tx_time.isoformat(),
                        "timestamp_unix": tx_time.timestamp(),
                        "customer_id": cust_id,
                        "account_created_at": acc_created.isoformat(),
                        "account_created_unix": acc_created.timestamp(),
                        "device_id": dev,
                        "ip_id": ip,
                        "payment_instrument_id": card,
                        "merchant_id": str(self.rng.choice(merchants)),
                        "amount": float(self.rng.uniform(1200, 3800)),
                        "location": "Mumbai",
                        "transaction_type": "payment",
                        "label": "coordinated_ring",
                        "abuse_type": pattern_type,
                        "ring_id": ring_id,
                        "is_fraud": 1
                    })

        elif pattern_type == "pattern_b_shared_ip":
            # 25-40 accounts sharing 1-2 subnet IPs
            num_accounts = int(self.rng.integers(25, 40))
            shared_ips = [f"ring_ip_{ring_id}_core"]
            ring_start = base_time + datetime.timedelta(hours=float(self.rng.uniform(20, 60)))
            for acc_idx in range(num_accounts):
                cust_id = f"cust_{ring_id}_{acc_idx:03d}"
                acc_created = ring_start - datetime.timedelta(hours=float(self.rng.uniform(1, 12)))
                dev = f"dev_{ring_id}_{acc_idx:03d}"
                ip = shared_ips[0]
                card = f"card_{ring_id}_{acc_idx:03d}"
                
                num_tx = int(self.rng.integers(2, 5))
                for t_idx in range(num_tx):
                    tx_time = ring_start + datetime.timedelta(
                        minutes=float(self.rng.uniform(0, 240)) + (t_idx * 15)
                    )
                    records.append({
                        "timestamp": tx_time.isoformat(),
                        "timestamp_unix": tx_time.timestamp(),
                        "customer_id": cust_id,
                        "account_created_at": acc_created.isoformat(),
                        "account_created_unix": acc_created.timestamp(),
                        "device_id": dev,
                        "ip_id": ip,
                        "payment_instrument_id": card,
                        "merchant_id": str(self.rng.choice(merchants)),
                        "amount": float(self.rng.uniform(800, 2900)),
                        "location": "Delhi NCR",
                        "transaction_type": "payment",
                        "label": "coordinated_ring",
                        "abuse_type": pattern_type,
                        "ring_id": ring_id,
                        "is_fraud": 1
                    })

        elif pattern_type == "pattern_c_shared_payment_instrument":
            # 12-20 accounts cycling 2 cards
            num_accounts = int(self.rng.integers(12, 20))
            shared_cards = [f"ring_card_{ring_id}_A", f"ring_card_{ring_id}_B"]
            ring_start = base_time + datetime.timedelta(hours=float(self.rng.uniform(15, 50)))
            for acc_idx in range(num_accounts):
                cust_id = f"cust_{ring_id}_{acc_idx:03d}"
                acc_created = ring_start - datetime.timedelta(days=float(self.rng.uniform(1, 15)))
                dev = f"dev_{ring_id}_{acc_idx:03d}"
                ip = f"ip_{ring_id}_{acc_idx % 4}"
                card = str(self.rng.choice(shared_cards))
                
                num_tx = int(self.rng.integers(3, 7))
                for t_idx in range(num_tx):
                    tx_time = ring_start + datetime.timedelta(
                        minutes=float(self.rng.uniform(0, 300)) + (t_idx * 20)
                    )
                    records.append({
                        "timestamp": tx_time.isoformat(),
                        "timestamp_unix": tx_time.timestamp(),
                        "customer_id": cust_id,
                        "account_created_at": acc_created.isoformat(),
                        "account_created_unix": acc_created.timestamp(),
                        "device_id": dev,
                        "ip_id": ip,
                        "payment_instrument_id": card,
                        "merchant_id": str(self.rng.choice(merchants)),
                        "amount": float(self.rng.uniform(1500, 4800)),
                        "location": "Bengaluru",
                        "transaction_type": "payment",
                        "label": "coordinated_ring",
                        "abuse_type": pattern_type,
                        "ring_id": ring_id,
                        "is_fraud": 1
                    })

        elif pattern_type == "pattern_d_rapid_account_creation":
            # 30 accounts registered in 45 minutes, immediately transacting
            num_accounts = int(self.rng.integers(25, 40))
            burst_start = base_time + datetime.timedelta(hours=float(self.rng.uniform(30, 80)))
            shared_dev = f"ring_dev_{ring_id}_burst"
            shared_ip = f"ring_ip_{ring_id}_burst"
            
            for acc_idx in range(num_accounts):
                # Created within 45 min window
                acc_created = burst_start + datetime.timedelta(minutes=float(self.rng.uniform(0, 45)))
                cust_id = f"cust_{ring_id}_{acc_idx:03d}"
                card = f"card_{ring_id}_{acc_idx:03d}"
                
                # Transact within 10-30 mins of creation
                tx_time = acc_created + datetime.timedelta(minutes=float(self.rng.uniform(5, 25)))
                records.append({
                    "timestamp": tx_time.isoformat(),
                    "timestamp_unix": tx_time.timestamp(),
                    "customer_id": cust_id,
                    "account_created_at": acc_created.isoformat(),
                    "account_created_unix": acc_created.timestamp(),
                    "device_id": shared_dev if acc_idx % 2 == 0 else f"dev_{ring_id}_{acc_idx}",
                    "ip_id": shared_ip,
                    "payment_instrument_id": card,
                    "merchant_id": str(self.rng.choice(merchants)),
                    "amount": float(self.rng.uniform(900, 3100)),
                    "location": "Hyderabad",
                    "transaction_type": "payment",
                    "label": "coordinated_ring",
                    "abuse_type": pattern_type,
                    "ring_id": ring_id,
                    "is_fraud": 1
                })

        elif pattern_type == "pattern_e_synchronized_activity":
            # 15 accounts transacting simultaneously in a 2-minute pulse
            num_accounts = int(self.rng.integers(15, 25))
            pulse_time = base_time + datetime.timedelta(hours=float(self.rng.uniform(25, 70)))
            shared_card = f"ring_card_{ring_id}_sync"
            
            for acc_idx in range(num_accounts):
                cust_id = f"cust_{ring_id}_{acc_idx:03d}"
                acc_created = pulse_time - datetime.timedelta(days=float(self.rng.uniform(3, 30)))
                # Transact within 90 seconds of pulse_time
                tx_time = pulse_time + datetime.timedelta(seconds=float(self.rng.uniform(0, 90)))
                records.append({
                    "timestamp": tx_time.isoformat(),
                    "timestamp_unix": tx_time.timestamp(),
                    "customer_id": cust_id,
                    "account_created_at": acc_created.isoformat(),
                    "account_created_unix": acc_created.timestamp(),
                    "device_id": f"dev_{ring_id}_{acc_idx % 3}",
                    "ip_id": f"ip_{ring_id}_{acc_idx % 2}",
                    "payment_instrument_id": shared_card,
                    "merchant_id": str(self.rng.choice(merchants)),
                    "amount": float(self.rng.uniform(2200, 5000)),
                    "location": "Pune",
                    "transaction_type": "payment",
                    "label": "coordinated_ring",
                    "abuse_type": pattern_type,
                    "ring_id": ring_id,
                    "is_fraud": 1
                })

        elif pattern_type == "pattern_f_gradual_growth":
            # Ring starts with 2 accounts, grows exponentially across 7 days
            ring_start = base_time + datetime.timedelta(hours=float(self.rng.uniform(10, 40)))
            devices = [f"ring_dev_{ring_id}_{i}" for i in range(4)]
            cards = [f"ring_card_{ring_id}_{i}" for i in range(3)]
            ip = f"ring_ip_{ring_id}_grad"
            
            # Days 0 to 6 with escalating counts: [2, 3, 5, 8, 14, 22]
            accounts_per_stage = [2, 3, 5, 8, 12, 18]
            acc_counter = 0
            for stage_day, count in enumerate(accounts_per_stage):
                stage_time = ring_start + datetime.timedelta(days=stage_day)
                for _ in range(count):
                    cust_id = f"cust_{ring_id}_{acc_counter:03d}"
                    acc_counter += 1
                    acc_created = stage_time - datetime.timedelta(hours=float(self.rng.uniform(1, 8)))
                    tx_time = stage_time + datetime.timedelta(minutes=float(self.rng.uniform(0, 180)))
                    records.append({
                        "timestamp": tx_time.isoformat(),
                        "timestamp_unix": tx_time.timestamp(),
                        "customer_id": cust_id,
                        "account_created_at": acc_created.isoformat(),
                        "account_created_unix": acc_created.timestamp(),
                        "device_id": str(self.rng.choice(devices)),
                        "ip_id": ip,
                        "payment_instrument_id": str(self.rng.choice(cards)),
                        "merchant_id": str(self.rng.choice(merchants)),
                        "amount": float(self.rng.uniform(1100, 3900)),
                        "location": "Ahmedabad",
                        "transaction_type": "payment",
                        "label": "coordinated_ring",
                        "abuse_type": pattern_type,
                        "ring_id": ring_id,
                        "is_fraud": 1
                    })

        elif pattern_type == "pattern_g_sudden_activation":
            # 20 dormant accounts created 10 days ago suddenly wake up in a 6-hour burst
            ring_creation = base_time - datetime.timedelta(days=12)
            activation_time = base_time + datetime.timedelta(hours=float(self.rng.uniform(50, 120)))
            shared_device = f"ring_dev_{ring_id}_burst_device"
            shared_card = f"ring_card_{ring_id}_burst_card"
            
            for acc_idx in range(20):
                cust_id = f"cust_{ring_id}_{acc_idx:03d}"
                acc_created = ring_creation + datetime.timedelta(hours=float(self.rng.uniform(0, 48)))
                
                # Transact in high velocity during activation window
                for t_idx in range(int(self.rng.integers(3, 6))):
                    tx_time = activation_time + datetime.timedelta(minutes=float(self.rng.uniform(0, 240)))
                    records.append({
                        "timestamp": tx_time.isoformat(),
                        "timestamp_unix": tx_time.timestamp(),
                        "customer_id": cust_id,
                        "account_created_at": acc_created.isoformat(),
                        "account_created_unix": acc_created.timestamp(),
                        "device_id": shared_device,
                        "ip_id": f"ring_ip_{ring_id}_{acc_idx % 3}",
                        "payment_instrument_id": shared_card,
                        "merchant_id": str(self.rng.choice(merchants)),
                        "amount": float(self.rng.uniform(1800, 5200)),
                        "location": "Jaipur",
                        "transaction_type": "payment",
                        "label": "coordinated_ring",
                        "abuse_type": pattern_type,
                        "ring_id": ring_id,
                        "is_fraud": 1
                    })

        elif pattern_type == "pattern_h_mixed_cluster":
            # Shared public IP (e.g. airport/university WiFi) with 15 fraud accounts and 15 legitimate accounts
            shared_public_ip = f"public_ip_hub_{ring_id}"
            hub_time = base_time + datetime.timedelta(hours=float(self.rng.uniform(20, 80)))
            
            # Fraud ring portion
            for acc_idx in range(15):
                cust_id = f"cust_{ring_id}_bad_{acc_idx:03d}"
                acc_created = hub_time - datetime.timedelta(hours=float(self.rng.uniform(2, 10)))
                tx_time = hub_time + datetime.timedelta(minutes=float(self.rng.uniform(0, 120)))
                records.append({
                    "timestamp": tx_time.isoformat(),
                    "timestamp_unix": tx_time.timestamp(),
                    "customer_id": cust_id,
                    "account_created_at": acc_created.isoformat(),
                    "account_created_unix": acc_created.timestamp(),
                    "device_id": f"ring_dev_{ring_id}_{acc_idx % 2}",
                    "ip_id": shared_public_ip,
                    "payment_instrument_id": f"ring_card_{ring_id}_{acc_idx % 2}",
                    "merchant_id": str(self.rng.choice(merchants)),
                    "amount": float(self.rng.uniform(2000, 4500)),
                    "location": "Kochi",
                    "transaction_type": "payment",
                    "label": "coordinated_ring",
                    "abuse_type": pattern_type,
                    "ring_id": ring_id,
                    "is_fraud": 1
                })
            
            # Legitimate portion sharing the SAME IP
            for acc_idx in range(15):
                cust_id = f"cust_{ring_id}_good_{acc_idx:03d}"
                acc_created = base_time - datetime.timedelta(days=float(self.rng.uniform(30, 200)))
                tx_time = hub_time + datetime.timedelta(minutes=float(self.rng.uniform(0, 300)))
                records.append({
                    "timestamp": tx_time.isoformat(),
                    "timestamp_unix": tx_time.timestamp(),
                    "customer_id": cust_id,
                    "account_created_at": acc_created.isoformat(),
                    "account_created_unix": acc_created.timestamp(),
                    "device_id": f"dev_legit_hub_{cust_id}",
                    "ip_id": shared_public_ip,
                    "payment_instrument_id": f"card_legit_hub_{cust_id}",
                    "merchant_id": str(self.rng.choice(merchants)),
                    "amount": float(self.rng.uniform(200, 1200)),
                    "location": "Kochi",
                    "transaction_type": "payment",
                    "label": "legitimate",
                    "abuse_type": "none",
                    "ring_id": None,
                    "is_fraud": 0
                })

        return records
