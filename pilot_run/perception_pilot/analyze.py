"""Score the frozen incremental predictions against one LAAS session."""

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / "Bi3" / "Bi3" / "user_impressions" / "laas" / "laas10.csv"
ATTRS = ["aggressive", "awful", "scary", "awkward", "dangerous", "strange"]
mapping = json.loads((OUT / "sealed_mapping.json").read_text())
stages = {name: json.loads((OUT / f"{name}_stage_predictions.json").read_text())["trials"] for name in ("frame", "map", "facts")}
with DATA.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

trials = {}
for label, condition in mapping.items():
    matched = [row for row in rows if row["algorithm"].lower() == condition]
    assert len(matched) == 2
    individual = [sum(float(row[a]) for a in ATTRS)/6 for row in matched]
    human = sum(individual)/2
    trials[label] = {
        "condition": condition,
        "human_individual": individual,
        "human_pair_mean": human,
        "predictions": {stage: {"attributes": data[label]["ratings"], "mean": sum(data[label]["ratings"])/6} for stage,data in stages.items()},
    }

summary = {}
for stage in stages:
    errors = [abs(t["predictions"][stage]["mean"]-t["human_pair_mean"]) for t in trials.values()]
    by_condition = {t["condition"]:t for t in trials.values()}
    contrasts = {}
    for a,b in (("blind","cv"),("cohan","cv"),("blind","cohan")):
        contrasts[f"{a}_minus_{b}"] = {
            "human":by_condition[a]["human_pair_mean"]-by_condition[b]["human_pair_mean"],
            "model":by_condition[a]["predictions"][stage]["mean"]-by_condition[b]["predictions"][stage]["mean"],
        }
    summary[stage] = {"mean_absolute_error":sum(errors)/len(errors), "contrasts":contrasts}

truth = json.loads((OUT / "sealed_truth.json").read_text())
perception = {}
for label, data in stages["frame"].items():
    d = truth[label]["minimum_center_distance_m"]
    observed = data["closest_center_distance_bin"]
    actual = "under 0.5 m" if d<0.5 else "0.5 to 1.0 m" if d<1 else "over 1.0 m"
    perception[label]={"frame_bin":observed,"mocap_bin":actual,"bin_correct":observed==actual,"minimum_center_distance_m":d}

result={"trials":trials,"stages":summary,"perception_probe":perception}
(OUT / "results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
