"""Compare sealed predictions with LAAS participant pair means."""

import csv
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / "Bi3" / "Bi3" / "user_impressions" / "laas"
ATTRS = ["aggressive", "awful", "scary", "awkward", "dangerous", "strange"]
predictions = json.loads((OUT / "blind_predictions.json").read_text())["predictions"]
mapping = json.loads((OUT / "sealed_mapping.json").read_text())
trials = {}

for anonymous_id, scores in predictions.items():
    session = mapping[anonymous_id]["session"]
    algorithm = mapping[anonymous_id]["algorithm"]
    with (DATA / f"{session}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matched = [r for r in rows if r["algorithm"].lower() == algorithm]
    assert len(matched) == 2, (session, algorithm, len(matched))
    human_items = [[float(row[attr]) for attr in ATTRS] for row in matched]
    human_individual = [sum(item) / 6 for item in human_items]
    human_mean = sum(human_individual) / 2
    model_mean = sum(scores) / 6
    trials[anonymous_id] = {
        "session": session,
        "algorithm": algorithm,
        "human_individual": human_individual,
        "human_pair_mean": human_mean,
        "model_scores": scores,
        "model_mean": model_mean,
        "model_minus_human": model_mean - human_mean,
        "absolute_error": abs(model_mean - human_mean),
    }

contrasts = {}
for session in ["laas4", "laas5", "laas6", "laas7"]:
    by_algorithm = {v["algorithm"]: v for v in trials.values() if v["session"] == session}
    contrasts[session] = {}
    for first, second in [("blind", "cv"), ("cohan", "cv"), ("blind", "cohan")]:
        a, b = by_algorithm[first], by_algorithm[second]
        human = a["human_pair_mean"] - b["human_pair_mean"]
        model = a["model_mean"] - b["model_mean"]
        contrasts[session][f"{first}_minus_{second}"] = {
            "human": human,
            "model": model,
            "same_direction": (human > 0 and model > 0) or (human < 0 and model < 0),
        }

overall = {}
for key in contrasts["laas4"]:
    human = [contrasts[s][key]["human"] for s in contrasts]
    model = [contrasts[s][key]["model"] for s in contrasts]
    overall[key] = {
        "human_mean": sum(human) / len(human),
        "model_mean": sum(model) / len(model),
        "direction_matches": sum(contrasts[s][key]["same_direction"] for s in contrasts),
        "pairs": len(contrasts),
    }

result = {
    "trials": trials,
    "contrasts": contrasts,
    "overall": overall,
    "mean_absolute_error": sum(t["absolute_error"] for t in trials.values()) / len(trials),
}
(OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"mean_absolute_error": result["mean_absolute_error"], "overall": overall, "trials": {k: {"algorithm": v["algorithm"], "human": round(v["human_pair_mean"], 2), "model": round(v["model_mean"], 2)} for k,v in trials.items()}}, indent=2))
