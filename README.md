# HRI Bi³ pilot study

Exploratory pilots testing whether a vision-language model can judge robot navigation as people in the [Bi³ study](https://arxiv.org/html/2605.06863) did. This repository contains the pilot scripts and questionnaire-aligned prompt, the preprocessing of the Bi³ motion-capture data (goal labels, video sync and robot trips), and a trajectory-generation pilot that tests whether the model can plan the robot's path. Downloaded Bi³ data, participant-level results, mocap-derived outputs, and generated media are kept outside this public repository.

## Pilots

| Run | Input and purpose | Preparation script |
| --- | --- | --- |
| LAAS3 | First blinded smoke test with 5-second frames | [Script](pilot_run/prepare.py) |
| LAAS4–7 | Four-pair frame and motion-summary pilot | [Script](pilot_run/experiment2/prepare.py) |
| LAAS10 | 15-second frames, close-event frames, map, and mocap facts | [Script](pilot_run/perception_pilot/prepare.py) |
| LAAS11 | 5-second version of the perception ladder | [Script](pilot_run/perception_pilot_5s/prepare.py) |
| LAAS12 | Own-impression ratings of all 12 RoSAS words with the revised prompt | [Script](pilot_run/rosas_aligned_laas12/prepare.py) |

The [questionnaire-aligned prompt](pilot_run/rosas_aligned_protocol.md) was used for LAAS12. Earlier pilots used a different question. These are small pilots, not population estimates or controlled comparisons between sessions.

## Preprocessing: goal labels, video sync and trips

[`preprocessing/annotate_goals.py`](preprocessing/annotate_goals.py) labels, for every mocap sample, which workstation each participant is at or heading to, and the station after that. Bi³ records only the robot's goals. A participant visit is detected as a turnaround within 1 m of a station (people stand 0.3–0.6 m beyond the robot's goal points). When someone goes between two visits at the same or neighbouring stations (at most 1 m apart) via a point more than 1 m from both without a detected turnaround there (a wide U-turn), that far point is filled in as a visit, marked `inferred`. Two tables hold two stations each (LAAS 0/1, UM 4/5), where position alone is ambiguous. Each person's sequence is split into their two tower phases, and a phase's visits to such a table take the phase's margin-weighted majority station when it holds at least 70% of the vote, marked `corrected` (`phase` and `phase_share` columns). This resolves every UM phase but about 75% of LAAS phases. The rest mix both spots, so their goal is the table itself. Goal labels are therefore text: a station (`"0"`–`"5"`) or a shared-table pair (`"0/1"` at LAAS, `"4/5"` at UM). `stations.json` lists every allowed label per site with its coordinates (a pair's are its midpoint). About 7% of LAAS participant visits are `"0/1"`, and no UM visit needed a pair. A backward pass then gives every in-transit sample its upcoming goal. `margin` is how much closer the nearest station is than the next nearest, measured before any phase correction.

The same detector run on the robot finds 96% of its recorded goals, and 90% of its detections match one. Participants get a median of 40 visits per trial at both sites (two towers of 10 blocks, pile and tower each). Outputs are mocap-derived and are written outside the repository, to `/persistent/data/Bi3_derived/goals` by default (`--out` to change).

[`preprocessing/sync_video.py`](preprocessing/sync_video.py) estimates each overhead video's time offset from its mocap by cross-correlating frame-to-frame pixel change with the agents' summed mocap speed (`video_time = mocap_time + offset_s`). Across all 139 videos the offsets range from -1.7 to +0.2 s (median about -1 s at both sites). Results go to `/persistent/data/Bi3_derived/video_sync.csv`.

[`preprocessing/extract_trips.py`](preprocessing/extract_trips.py) cuts robot station-to-station trips into compact samples for trajectory generation. Each trip names its planned agent (`agent`, here the robot) with its goal (`goal`, `goal_xy`, `next_goal`). It holds every agent's cleaned track resampled to 2.5 Hz on a grid anchored at the trip's start (`time` is 0 there), with up to 3 s of history before the start and up to 15 s of replay after the end (`tracks`, keyed `robot`, `human0`, `human1`). `start_index` and `end_index` mark the first and last sample of the real trip. The trip also holds every agent's per-sample labels (`labels`) and the Bi³ metrics ([`metrics/bi3_metrics.py`](metrics/bi3_metrics.py)) of the planned agent's real path against the other agents (`real_metrics`). Coordinates are room-centred, with LAAS rotated 180° so both sites have the shared table on the +x wall. `site_frames.json` records each site's frame, the 3.5 m room, the robot footprint (PR2 0.668 x 0.668 m, Stretch 0.330 x 0.340 m) and the stations. By default it uses the Bi³ test trial `laas18` and the trial `um13`, because the UM test trials (`um16`, `um17`) have no overhead video and the camera-input experiments need one. Results go to `/persistent/data/Bi3_derived/trips_v2` by default (`--out` to change).

Use [`preprocessing/bi3_derived.py`](preprocessing/bi3_derived.py) to read these outputs rather than rerunning the scripts:
- `load_video_sync()` and `video_time(site, trial, controller, mocap_time)` for the video offsets.
- `load_stations(site)` for the allowed goal labels.
- `load_visits(site, trial, controller)` for the detected visits.
- `load_frames(site, trial, controller)` for the per-sample positions and labels of every agent of a recording, keyed by agent id (`"robot"`, `"human0"`, ...).
- `load_trips(trips_dir, **match)` for the trips, optionally filtered by field, for example `load_trips(trips_dir, agent="robot", site="um")`.
- `load_site_frames(trips_dir)` for the room frame, robot footprint and room-frame stations per site.

Set `BI3_DERIVED` if the derived files live elsewhere, and `BI3_RAW` if the raw dataset is not under the repository root. Each script writes a manifest with its output (`goals/manifest.json`, `video_sync.manifest.json`, `trips_v2/manifest.json`) recording the git commit of the code, whether `preprocessing/`, `metrics/` or `traj_gen/` had uncommitted changes (`git_dirty`), and the parameters, so teammates can confirm they share the same labels. The `goals/` and `video_sync` outputs now under `/persistent/data/Bi3_derived` were written by an earlier version, whose manifests record the script's SHA-256, the branch and the commit instead.

## Trajectory generation

[`traj_gen/`](traj_gen) tests whether a vision-language model (local Qwen through vLLM) can plan the path of the planned agent in a Bi³ trip. Given the room, the other agents (positions, recent motion, goals) and the planned agent's goal station, as text and/or a top-down map, the model returns timed waypoints, map pixels or velocities. The executed path is scored against the five real controllers and a straight line on the same trips. The other agents are replayed from the logs and do not react to the planned agent. The experiments are defined in [`traj_gen/experiments.py`](traj_gen/experiments.py), the prompt templates in `traj_gen/prompts/` and the metrics in [`metrics/traj_gen_metrics.py`](metrics/traj_gen_metrics.py).

The planned agent of a trip is `trip["agent"]`, an agent id (`robot` or `human<k>`), and the other agents are all other entries of `trip["tracks"]`. [`traj_gen/agents.py`](traj_gen/agents.py) holds the parameters of each kind of agent: the body radius for scoring, the radius in the prompts and maps, the personal-space radius, the speed cap and the arrival radius. Only the robot is planned so far. The speed cap (0.3 m/s) and the arrival radius (0.5 m) are defined only for the robot, and the prompt templates describe a planned robot.

```
python -m traj_gen.run --run <run> select --trips-dir <trips folder>  # choose 16 test trips, write <run>/test_trips.json and <run>/run.json
python -m traj_gen.run --run <run> run --thinking nothink think       # resumable, needs the vLLM server on :8000
python -m traj_gen.run --run <run> analyze                            # tables and figures of whatever has finished
python -m traj_gen.run --run <run> animate <episode folder> ...       # animation.gif and timeline.png (colour = time) in each folder
python -m traj_gen.report_figures --run <run>                         # write-up figures and personal-space sensitivity table into <run>/report, after analyze
```

The `*_pred` experiments (`ss_wp_bev_pred`, `rh_wp_bev_pred`, `rh_vel1_bev_pred`) are their base experiment plus each person's constant-velocity position at t + 1 s and t + 2 s, in the text and drawn on the map.

Clearance and personal-space overlap ([`metrics/traj_gen_metrics.py`](metrics/traj_gen_metrics.py)) model every agent as a disk. The robot's scoring disk is the footprint's inscribed disk (`body_radius`, half the shorter side: PR2 0.334 m, Stretch 0.165 m), since a planned path has no heading. The person body radius is 0.15 m. The prompts and maps shown to the model use the robot's circumscribed disk (`prompt_radius`: PR2 0.472 m, Stretch 0.237 m) and a 0.25 m person. The circumscribed disk is conservative on purpose, so planned paths keep a margin from people, and it makes the prompts stricter than the metrics.

`clearance_min` is the closest centre distance minus the robot body radius minus the person body radius, so below 0 the bodies touch. `contact_time` is the share of the episode's time steps in contact and `contact_events` the number of separate contacts (a run of consecutive steps counts once). With the 0.15 m person, two real-controller contacts are measured on the 16 test trips of the v1 run, in trips `laas_laas18_static_robot_02` (clearance -0.008 m) and `um_um13_static_robot_05` (-0.018 m), and three in the 133 trips with a full prompt history. They are unlikely to be real collisions, because both bodies are modelled as circles. `personal_space_overlap` is the share of the episode's time steps with the robot within its body radius plus 0.25 m of a person, a modelling choice with 0.20 m as the sensitivity value. `report_figures` writes that sensitivity table.

An episode ends when the executed path first comes within the arrival radius of the goal, or when the replay ends. `goal_reached` uses a 0.15 m slack on top of the arrival radius (0.65 m for the robot), because the cleaned real paths end 0.48–0.60 m from the goal at Bi³'s goal switch. With this slack every real controller reaches its goal.

Every metric belongs to one class (`METRIC_CLASS` in [`metrics/traj_gen_metrics.py`](metrics/traj_gen_metrics.py)): `invalid` (the plan breaks a hard rule of the task), `contact` (closeness to the other agents) or `efficiency` (how the agent moves). `analyze` reports an invalid rate and the rate of each of its parts, each a share of all episodes of the condition. An episode is invalid when its reply is unusable (status `invalid_reply`), the server failed (status `error`), or its plan exceeds the speed cap in any step, does not reach the goal, leaves the room or crosses a station. An episode without a path counts only in its status part. The speed-cap part does not apply to the baselines, which have no model reply, and is NaN for them. Path metrics such as clearance and time to goal use status-`ok` episodes only, and `goal_reached` counts an episode without a usable path as not reached.

**Note:** the official scoring of generated trajectories has not been built yet. The metrics in [`metrics/traj_gen_metrics.py`](metrics/traj_gen_metrics.py) are exploratory definitions written for this pilot, including the body and personal-space radii, not a validated scoring against the Bi³ ratings.

`--run <folder>` (before the subcommand) is required and picks the run folder. `select` writes `<run>/run.json` with the trips folder and the git commit of the code. Every later command reads the trips from that folder. `run` refuses to add episodes when the code is at another commit or has uncommitted changes in `preprocessing/`, `metrics/` or `traj_gen/`, so the episodes of a run folder come from one commit. `analyze`, `animate` and `report_figures` read a run made at any commit.

Each model episode is saved under `<run>/<experiment>/<framing>/<thinking>/<trip>/s<k>/`, and each baseline episode (`straight`, `replay`) under `<run>/<experiment>/<trip>/`. An episode folder holds `inputs/` (the images sent), `trajectory.csv`, `plans.jsonl`, `queries.jsonl`, `result.png` and `manifest.json`. `trajectory.csv` has the columns `t, robot_x, robot_y, human0_x, human0_y, ...`, one per agent in the trip's track order, from the trip start (`t` = 0) to the end of the episode. The manifest records the episode's status, metrics, query counts and the code's git commit and dirty flag. The status is `ok`, `invalid_reply` (an unusable reply: single shot at once, rolling after one retry) or `error` (a server or transport error, with the traceback in the manifest). A resumed `run` runs episodes that ended in `error` again.

`analyze` writes `<run>/analysis/`, with every table and figure split by the planned agent's kind (`robot` or `human`):
- `episodes.csv`: one row per episode.
- `summary.csv`: per kind, condition and site, the status counts and the episode-level rates and means, grouped by metric class.
- `paired_baselines.csv`: per kind, condition and site, each condition minus replay and minus straight, paired on the same trips, with bootstrap intervals over trips.
- `comparisons.csv`: condition comparisons (social - neutral prompt, think - nothink, bev - text input, ...) with bootstrap intervals over trips.
- `figures/`: condition dot plots, speed-cap violations and one overlay per test trip.

All of these are mocap-derived, so run folders stay outside the repository and must not be committed.

## Tests

Run the tests from the repository root with `python -m pytest tests`.

## Reproduce locally

1. Download Bi³ separately and place the extracted dataset so that `Bi3/Bi3/videos/laas/`, `Bi3/Bi3/jsons/laas/`, and `Bi3/Bi3/user_impressions/laas/` exist under this repository root. The data are not redistributed here.
2. Install Python 3.10 or newer and `pip install -r requirements.txt`. The preprocessing and trajectory-generation code (`preprocessing/sync_video.py`, `traj_gen/camera.py`) calls the `ffmpeg` binary, which must be on PATH. The pilot preparation scripts in `pilot_run/` use the FFmpeg binary supplied by `imageio-ffmpeg`. Image preparation scripts currently use the Windows Arial font path used for the original run.
3. Preparation can generate a random A/B/C condition key and contact sheets. For a **new blinded evaluation**, do not read the mapping, mocap truth, or survey CSV until the new observer ratings are saved.
4. Run a run's `analyze.py` after preparation and rating to recompute its `results.json`. LAAS3 uses `pilot_run/analyse.py`.

Generated contact sheets, maps, frame caches, and saved outputs are omitted from Git. The model was operated through Codex during these exploratory rating pilots. For them there is no automated VLM API runner or claim of deterministic regeneration of the ratings.

## Study limitations

The Bi³ papers identify the 12 RoSAS items and nine-point scale but do not publish the exact online survey form. The revised prompt adapts the original RoSAS anchors. It is not verified verbatim Bi³ wording. Frame sampling misses continuous motion, sudden changes, and participant embodiment. Participants can disagree strongly even within one trial. See each report for the run's specific design and errors.
