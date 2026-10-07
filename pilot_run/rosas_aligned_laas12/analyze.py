"""Score sealed LAAS12 RoSAS predictions against the two participant records."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DISCOMFORT = ["aggressive", "awful", "scary", "awkward", "dangerous", "strange"]
COMPETENCE = ["knowledgeable", "interactive", "responsive", "capable", "reliable", "competent"]

pred = json.loads((HERE / "blind_predictions.json").read_text())["predictions"]
mapping = json.loads((HERE / "sealed_mapping.json").read_text())
with (ROOT / "Bi3/Bi3/user_impressions/laas/laas12.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

result = {"session": "laas12", "scale": "1–9", "trials": {}, "summary": {}}
errors = {"discomfort": [], "competence": []}
person_errors = {"discomfort": [], "competence": []}
for letter in "ABC":
    controller = mapping[f"laas12_{letter}"]
    people = sorted((r for r in rows if r["algorithm"].lower() == controller), key=lambda r: int(r["human_index"]))
    assert len(people) == 2
    output = {"blind_id": letter, "controller": controller, "model_items": {k: pred[letter][k] for k in DISCOMFORT + COMPETENCE}, "individuals": []}
    for r in people:
        for scale, items in [("discomfort", DISCOMFORT), ("competence", COMPETENCE)]:
            assert abs(sum(float(r[k]) for k in items) / 6 - float(r[scale])) < 1e-9
        output["individuals"].append({"human_index": int(r["human_index"]), "discomfort": float(r["discomfort"]), "competence": float(r["competence"])})
    for scale, items in [("discomfort", DISCOMFORT), ("competence", COMPETENCE)]:
        model = sum(pred[letter][k] for k in items) / 6
        individual = [float(r[scale]) for r in people]
        human_mean = sum(individual) / 2
        output[scale] = {"model": model, "human_mean": human_mean, "signed_error": model - human_mean, "absolute_error": abs(model - human_mean), "human_range": max(individual) - min(individual)}
        errors[scale].append(abs(model - human_mean))
        person_errors[scale].extend(abs(model - x) for x in individual)
    result["trials"][controller] = output

for scale in errors:
    result["summary"][scale] = {"mae_vs_pair_mean": sum(errors[scale]) / 3, "mae_vs_individuals": sum(person_errors[scale]) / 6}

(HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
