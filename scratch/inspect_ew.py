import json

with open("evaluation/artifacts/ablation_results.json") as f:
    d = json.load(f)

for r in d["early_warning_rings_detail"]:
    r_id = r["ring_id"]
    ptype = r["pattern_type"]
    print(f"Ring {r_id} ({ptype}):")
    print(f"  Stage A: first_alert={r['stage_a']['first_alert_time_unix']}, post_coord={r['stage_a']['post_coord_alert_time_unix']}, exp%={r['stage_a']['exposure_at_first_alert_pct']:.1f}%")
    print(f"  Stage B: first_alert={r['stage_b']['first_alert_time_unix']}, post_coord={r['stage_b']['post_coord_alert_time_unix']}, exp%={r['stage_b']['exposure_at_first_alert_pct']:.1f}%")
    print(f"  Stage C: first_alert={r['stage_c']['first_alert_time_unix']}, post_coord={r['stage_c']['post_coord_alert_time_unix']}, exp%={r['stage_c']['exposure_at_first_alert_pct']:.1f}%")
