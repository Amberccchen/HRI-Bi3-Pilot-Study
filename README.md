# HRI Bi³ pilot study

Exploratory pilots testing whether a vision-language model can judge robot navigation as people in the [Bi³ study](https://arxiv.org/html/2605.06863) did. This repository contains the pilot scripts and questionnaire-aligned prompt. Downloaded Bi³ data, participant-level results, mocap-derived outputs, and generated media are kept outside this public repository.

## Pilots

| Run | Input and purpose | Preparation script |
| --- | --- | --- |
| LAAS3 | First blinded smoke test with 5-second frames | [Script](pilot_run/prepare.py) |
| LAAS4–7 | Four-pair frame and motion-summary pilot | [Script](pilot_run/experiment2/prepare.py) |
| LAAS10 | 15-second frames, close-event frames, map, and mocap facts | [Script](pilot_run/perception_pilot/prepare.py) |
| LAAS11 | 5-second version of the perception ladder | [Script](pilot_run/perception_pilot_5s/prepare.py) |
| LAAS12 | Own-impression ratings of all 12 RoSAS words with the revised prompt | [Script](pilot_run/rosas_aligned_laas12/prepare.py) |

The [questionnaire-aligned prompt](pilot_run/rosas_aligned_protocol.md) was used for LAAS12. Earlier pilots used a different question. These are small pilots, not population estimates or controlled comparisons between sessions.

## Preprocessing: goal labels and video sync

[`preprocessing/annotate_goals.py`](preprocessing/annotate_goals.py) labels, for every mocap sample, which workstation each participant is at or heading to, and the station after that. Bi³ records only the robot's goals. A participant visit is detected as a turnaround within 1 m of a station (people stand 0.3–0.6 m beyond the robot's goal points). When someone goes between two visits at the same or neighbouring stations (at most 1 m apart) via a point more than 1 m from both without a detected turnaround there (a wide U-turn), that far point is filled in as a visit, marked `inferred`. Two tables hold two stations each (LAAS 0/1, UM 4/5), where position alone is ambiguous. Each person's sequence is split into their two tower phases, and a phase's visits to such a table take the phase's margin-weighted majority station when it holds at least 70% of the vote, marked `corrected` (`phase` and `phase_share` columns). This resolves every UM phase but about 75% of LAAS phases; the rest mix both spots, so their goal is the table itself. Goal labels are therefore text: a station (`"0"`–`"5"`) or a shared-table pair (`"0/1"` at LAAS, `"4/5"` at UM); `stations.json` lists every allowed label per site with its coordinates (a pair's are its midpoint). About 7% of LAAS participant visits are `"0/1"`; no UM visit needed a pair. A backward pass then gives every in-transit sample its upcoming goal. `margin` is how much closer the nearest station is than the next nearest, measured before any phase correction.

The same detector run on the robot finds 96% of its recorded goals, and 90% of its detections match one. Participants get a median of 40 visits per trial at both sites (two towers of 10 blocks, pile and tower each). Outputs are mocap-derived and are written outside the repository, to `/persistent/data/Bi3_derived/goals` by default (`--out` to change).

[`preprocessing/sync_video.py`](preprocessing/sync_video.py) estimates each overhead video's time offset from its mocap by cross-correlating frame-to-frame pixel change with the agents' summed mocap speed (`video_time = mocap_time + offset_s`). Across all 139 videos the offsets range from -1.7 to +0.2 s (median about -1 s at both sites). Results go to `/persistent/data/Bi3_derived/video_sync.csv`.

Use [`preprocessing/bi3_derived.py`](preprocessing/bi3_derived.py) to read these outputs rather than rerunning the scripts: `load_video_sync()`, `video_time(site, trial, controller, mocap_time)`, `load_stations(site)`, `load_visits(...)` and `load_frames(..., agent="human0")`. Set `BI3_DERIVED` if the files live elsewhere. Each output folder has a manifest (`goals/manifest.json`, `video_sync.manifest.json`) recording the parameters, the script's SHA-256 and the git commit that produced it, so teammates can confirm they share the same labels.

## Reproduce locally

1. Download Bi³ separately and place the extracted dataset so that `Bi3/Bi3/videos/laas/`, `Bi3/Bi3/jsons/laas/`, and `Bi3/Bi3/user_impressions/laas/` exist under this repository root. The data are not redistributed here.
2. Install Python 3 and `pip install -r requirements.txt`. The scripts call the FFmpeg binary supplied by `imageio-ffmpeg`. Image preparation scripts currently use the Windows Arial font path used for the original run.
3. Preparation can generate a random A/B/C condition key and contact sheets. For a **new blinded evaluation**, do not read the mapping, mocap truth, or survey CSV until the new observer ratings are saved.
4. Run a run's `analyze.py` after preparation and rating to recompute its `results.json`. LAAS3 uses `pilot_run/analyse.py`.

Generated contact sheets, maps, frame caches, and saved outputs are omitted from Git. The model was operated through Codex during these exploratory pilots; there is no automated VLM API runner or claim of deterministic regeneration of its ratings.

## Study limitations

The Bi³ papers identify the 12 RoSAS items and nine-point scale but do not publish the exact online survey form. The revised prompt adapts the original RoSAS anchors; it is not verified verbatim Bi³ wording. Frame sampling misses continuous motion, sudden changes, and participant embodiment. Participants can disagree strongly even within one trial. See each report for the run's specific design and errors.
