"""Create anonymous numeric mocap facts for the final information step."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "Bi3" / "Bi3" / "jsons" / "laas" / "laas10"
mapping = json.loads((OUT / "sealed_mapping.json").read_text())
result = {}

for label, algorithm in mapping.items():
    log = json.loads((DATA / f"{algorithm}.json").read_text())
    sampled = {}
    for e in log:
        sampled.setdefault(int(e["time"]), e)
    entries = [sampled[k] for k in sorted(sampled)]
    distances = [[math.dist(e["robot_state"][:2], e["agent_states"][p][:2]) for e in entries] for p in (0,1)]
    nearest = [min(a,b) for a,b in zip(*distances)]
    speed = [math.dist(entries[i]["robot_state"][:2], entries[i-1]["robot_state"][:2]) for i in range(1,len(entries))]
    episodes = []
    active = None
    for i,d in enumerate(nearest + [float("inf")]):
        if d < 1.0 and active is None:
            active=i
        elif d >= 1.0 and active is not None:
            if i-active >= 2:
                episodes.append({"start_s":active,"end_s":i-1,"min_center_m":round(min(nearest[active:i]),2)})
            active=None
    result[label] = {
        "duration_s": len(entries),
        "nearest_min_center_m": round(min(nearest),2),
        "time_nearest_center_below_1m_s": sum(d<1.0 for d in nearest),
        "time_nearest_center_below_0_75m_s": sum(d<0.75 for d in nearest),
        "per_person_min_center_m": [round(min(ds),2) for ds in distances],
        "per_person_time_center_below_1m_s": [sum(d<1.0 for d in ds) for ds in distances],
        "robot_path_length_m": round(sum(speed),1),
        "robot_mean_speed_m_s": round(sum(speed)/max(1,len(speed)),2),
        "robot_time_displacement_below_0_05m_s": sum(v<0.05 for v in speed),
        "sustained_below_1m_episodes": episodes,
    }
(OUT / "blinded_mocap_facts.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
