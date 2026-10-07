"""Compare saved blinded observer estimates with Bi3 ratings for one pair."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pilot_run" / "laas3"
mapping = json.loads((OUT / "sealed_mapping.json").read_text(encoding="utf-8"))
predictions = json.loads((OUT / "blind_predictions.json").read_text(encoding="utf-8"))["predictions"]
with (ROOT / "Bi3" / "Bi3" / "user_impressions" / "laas" / "laas3.csv").open(
    newline="", encoding="utf-8"
) as handle:
    rows = list(csv.DictReader(handle))

results = {}
for label, algorithm in mapping.items():
    relevant = [row for row in rows if row["algorithm"].lower() == algorithm]
    if len(relevant) != 2:
        raise ValueError(f"Expected two survey responses for {algorithm}")
    human_scores = [float(row["discomfort"]) for row in relevant]
    model_scores = [sum(scores) / len(scores) for scores in predictions[label].values()]
    results[algorithm] = {
        "blinded_label": label,
        "human_individual": human_scores,
        "human_pair_mean": sum(human_scores) / 2,
        "model_dark_shirt": model_scores[0],
        "model_light_shirt": model_scores[1],
        "model_pair_mean": sum(model_scores) / 2,
    }

contrasts = {}
for left, right in [("blind", "cv"), ("cohan", "cv"), ("blind", "cohan")]:
    contrasts[f"{left}_minus_{right}"] = {
        "human": results[left]["human_pair_mean"] - results[right]["human_pair_mean"],
        "model": results[left]["model_pair_mean"] - results[right]["model_pair_mean"],
    }

summary = {
    "session": "laas3",
    "unit": "pair mean because the human_index-to-clothing mapping was not verified",
    "results": results,
    "contrasts": contrasts,
    "interpretation": "Descriptive feasibility result only; one pair and sparse video sampling do not estimate general model-human agreement.",
}
(OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
