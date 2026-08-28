import sys
sys.path.insert(0, r"c:\Users\surya\Desktop\razorpay_buildathon")

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# 1. /health
r = client.get("/health")
assert r.status_code == 200
print("[PASS] GET /health:", r.json())

# 2. /cases
r = client.get("/cases")
assert r.status_code == 200
cases = r.json()
print(f"[PASS] GET /cases: {len(cases)} cases retrieved")
sample_id = cases[0]["case_id"]

# 3. /cases/{id}
r = client.get(f"/cases/{sample_id}")
assert r.status_code == 200
print(f"[PASS] GET /cases/{sample_id}: severity = {r.json()['severity']}")

# 4. /clusters/{id}
r = client.get(f"/clusters/{sample_id}")
assert r.status_code == 200
print(f"[PASS] GET /clusters/{sample_id}: cluster retrieved")

# 5. /metrics
r = client.get("/metrics")
assert r.status_code == 200
print(f"[PASS] GET /metrics: Stage C F1 = {r.json()['stages']['stage_c_sentinelgraph']['classification']['f1']}")

# 6. /explain/{id}
r = client.post(f"/explain/{sample_id}")
assert r.status_code == 200
print(f"[PASS] POST /explain/{sample_id}: rationale length = {len(r.json()['action_rationale'])}")

# 7. /score
r = client.post("/score", json={"transactions": [
    {
        "transaction_id": "tx_test_01",
        "timestamp": "2026-08-15T12:00:00",
        "timestamp_unix": 1786800000.0,
        "customer_id": "cust_1",
        "device_id": "dev_1",
        "ip_id": "ip_1",
        "payment_instrument_id": "card_1",
        "merchant_id": "merch_1",
        "amount": 500.0,
        "account_created_unix": 1786700000.0
    }
]})
assert r.status_code == 200
print(f"[PASS] POST /score: score = {r.json()['scores'][0]['final_risk_score']:.1f}")

print("\nALL 7 API ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")
