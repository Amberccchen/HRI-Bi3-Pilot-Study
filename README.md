# HRI Bi³ pilot study

Exploratory pilots testing whether a vision-language model can judge robot navigation as people in the [Bi³ study](https://arxiv.org/html/2605.06863) did. This repository contains the pilot scripts and questionnaire-aligned prompt, the preprocessing of the Bi³ motion-capture data (goal labels, video sync, and robot and person trips), and a trajectory-generation pilot that tests whether the model can plan the path of the robot or of a participant. How the data, metrics and generation are defined is in [DESIGN.md](DESIGN.md). Downloaded Bi³ data, participant-level results, mocap-derived outputs, and generated media are kept outside this public repository.

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

[`preprocessing/annotate_goals.py`](preprocessing/annotate_goals.py) detects each participant's station visits and labels, for every mocap sample, which station each agent is at or heading to and the station after that. Goal labels are text: a station (`"0"`-`"5"`) or a shared-table pair (`"0/1"` at LAAS, `"4/5"` at UM). `stations.json` lists every allowed label per site with its coordinates. Outputs are mocap-derived and are written outside the repository, to `/persistent/data/Bi3_derived/goals` by default (`--out` to change).

[`preprocessing/sync_video.py`](preprocessing/sync_video.py) estimates each overhead video's time offset from its mocap (`video_time = mocap_time + offset_s`). Results go to `/persistent/data/Bi3_derived/video_sync.csv`.

[`preprocessing/extract_trips.py`](preprocessing/extract_trips.py) cuts station-to-station trips of a planned agent into compact, room-centred samples: robot trips by default, person trips with `--agents human`. Each trip names its planned agent (`agent`), the station it leaves (`start_goal`) and its goal (`goal`, `goal_xy`, `next_goal`). Every agent of a recording gets a 10 Hz series once (`recordings.npz`): each sample the mean of the raw samples in a centred 0.1 s bin, with gaps filled by PCHIP. `trips.json` holds each trip's metadata, its index range on its recording's grid and the Bi³ metrics of the real path (`real_metrics`). `preprocessing/bi3_derived.py:load_trips` assembles each trip: every agent's track at 2.5 Hz on the grid anchored at the trip's start (`time`, `tracks` keyed `robot`, `human0`, `human1`, every 4th series sample), with history before the start and replay after the end, `start_index` and `end_index` for the real trip, per-sample labels (`labels`), and the 10 Hz series over the same span (`input_tracks_10hz`). `site_frames.json` records each site's room frame, room size, robot footprint and stations (`xy` for the robot, `human_xy` for people). By default it uses the trials `laas18` and `um13`. Results go to `/persistent/data/Bi3_derived/trips_v3` by default (`--out` to change). The person trips of all 37 trials (`--agents human --all-trials`) are in `/persistent/data/Bi3_derived/trips_v3_human`, and `analyze` uses all of them as the real moves the generated ones are compared with. The folders `trips_v2` and `trips_v2_human` hold the earlier format (tracks smoothed with a 0.5 s moving average, stored per trip), which only the code at `traj_gen` ee7150f reads.

How visits, goal labels, the room frame, stations, trips and moves are defined, and the numbers behind each choice, is in [DESIGN.md](DESIGN.md), Section 1.

Use [`preprocessing/bi3_derived.py`](preprocessing/bi3_derived.py) to read these outputs rather than rerunning the scripts:
- `load_video_sync()` and `video_time(site, trial, controller, mocap_time)` for the video offsets.
- `load_stations(site)` for the allowed goal labels.
- `load_visits(site, trial, controller)` for the detected visits.
- `load_frames(site, trial, controller)` for the per-sample positions and labels of every agent of a recording, keyed by agent id (`"robot"`, `"human0"`, ...).
- `load_trips(trips_dir, **match)` for the trips, optionally filtered by field, for example `load_trips(trips_dir, agent="robot", site="um")`.
- `load_site_frames(trips_dir)` for the room frame, robot footprint and room-frame stations per site.

Set `BI3_DERIVED` if the derived files live elsewhere, and `BI3_RAW` if the raw dataset is not under the repository root. Each script writes a manifest with its output (`goals/manifest.json`, `video_sync.manifest.json`, `trips_v3/manifest.json`) recording the git commit of the code, whether `preprocessing/`, `metrics/` or `traj_gen/` had uncommitted changes (`git_dirty`), and the parameters, so teammates can confirm they share the same labels.

## Trajectory generation

[`traj_gen/`](traj_gen) tests whether a vision-language model (local Qwen through vLLM, or Claude through the Anthropic API with `--backend claude`) can plan the path of the planned agent in a Bi³ trip, the robot or one of the people. Given the room, the other agents and the planned agent's goal station, as text and/or a top-down map, the model returns positions in the world frame, steps from its current heading or velocities, in pixels of the map or metres (the units setting). The executed path is scored against the real path and a straight line on the same trips, with the other agents replayed from the logs. A run is set up with harness presets, defined in [`traj_gen/harness/presets.py`](traj_gen/harness/presets.py): a harness is everything that sets the model's prompts and loop (episode type, output mode, units, images, framing, prediction, plan check and replanning interval), and the experiment id of its run folder is derived from those settings. The agents' parameters in [`traj_gen/agents.py`](traj_gen/agents.py), the prompt templates in `traj_gen/prompts/` and the metrics in [`metrics/traj_gen_metrics.py`](metrics/traj_gen_metrics.py) and [`metrics/human_similarity.py`](metrics/human_similarity.py). What each generator is told, the radii, speed caps, arrival rule, metric classes, similarity measures and map drawing are defined in [DESIGN.md](DESIGN.md), Sections 2 and 3.

```
python -m traj_gen.run --run <run> select --trips-dir <trips folder>  # choose 16 test trips, write <run>/test_trips.json and <run>/run.json
python -m traj_gen.run --run <run> select-human --trips-dir <person trips folder> --close 2 --random 2  # person test trips per site
python -m traj_gen.run --run <run> run --thinking nothink think       # preset pilot_v2 for a planned person on the person trip list, plus both baselines; resumable, needs the vLLM server on :8000
python -m traj_gen.run --run <run> run --robot --thinking think       # the same for the robot as the planned agent, on the robot test trips
python -m traj_gen.run --run <run> run --harness harness_v2 --overrides prediction=true_future check=2 --baselines none  # another preset, with settings changed
python -m traj_gen.run --run <run> run --backend claude [--model claude-opus-5-5] --concurrency 8  # the same through the Anthropic API (ANTHROPIC_API_KEY), no server; one backend and model per run folder
python -m traj_gen.run --run <run> analyze                            # tables and figures of whatever has finished
python -m traj_gen.run --run <run> animate <episode folder> ...       # animation.gif and timeline.png (colour = time) in each folder
python -m traj_gen.run --run <run> animate <condition folder> ...     # one animation.gif of all samples of <experiment>/<framing>/<thinking>/<trip>
```

`run` takes:
- `--harness NAME ...`: presets, `pilot_v2` (default) or `harness_v2`. A preset gives a framing per planned agent: `pilot_v2` is map positions in pixels in one conversation with the `participant` (person) or `neutral` (robot) task, and `harness_v2` the steer harness in pixels with the `scored_distances` or `robot_scored_distances` task. The earlier presets `pilot` and `harness_v1` reproduce their runs only at `traj_gen` ee7150f.
- `--robot`: the robot is the planned agent, so each preset's robot framing and the robot test trips. Without it, a person is planned. A trip list of the other kind is refused.
- `--overrides KEY=VALUE ...`: settings applied to every harness of the command: `episode` (`single_shot`, `rolling`, `rolling_conv`), `output` (`positions`, `steer`, `steer_vel`, `vel1`, `velseq`), `units` (`px` or `m`: every number the model reads or writes, `px` only with a map), `images` (`text`, `bev`, `bev_cam`), `framing`, `prediction` (`none`, `true_future`, `constvel_future`), `check` (plan-check rounds, needs `prediction=true_future`) and `execute_s` (replanning interval of rolling episodes).
- `--baselines straight replay` (default) or `--baselines none`.
- `--trips FILE`: the trip list. The default is `<run>/test_trips.json` if it exists, else [`traj_gen/trip_lists/person_pilot.json`](traj_gen/trip_lists/person_pilot.json) (100 person trips) or, with `--robot`, [`traj_gen/trip_lists/robot_test.json`](traj_gen/trip_lists/robot_test.json) (the 16 robot test trips).

The experiment id is `ss` or `rh`, the output, the units, the images, then `conv` for a conversation, `true_future` or `constvel_future`, `check` (2 rounds) or `check<N>`, and `replan<s>s` when the replanning interval is not the output's default. For example `--harness harness_v2 --overrides prediction=true_future check=2` runs in `rh_steer_px_bev_conv_true_future_check/scored_distances/`. Run folders made before the units setting name the positions output `wp` (metres) or `px` (pixels) and leave the units out of the steer ids (`rh_px_bev_conv`, `rh_steer_bev_conv`); they are analysed with the code at `traj_gen` ee7150f.

Scripts for the standard runs are in [`configs/runs/`](configs/runs): `person_pilot_v2_qwen.sh`, `person_pilot_v2_claude.sh` and `person_harness_v2_claude.sh`. Each writes to a new run folder `trips_v3/<YYYYMMDD>_<script>_<commit>` and passes extra options through to `run`. Notable runs are logged in [`traj_gen/RUNS.md`](traj_gen/RUNS.md).

The `*_constvel_future` experiments (`--overrides prediction=constvel_future`, deprecated) are their base experiment plus each person's constant-velocity position at t + 1 s and t + 2 s, in the text and drawn on the map.

**Note:** the official scoring of generated trajectories has not been built yet. The metrics are exploratory definitions written for this pilot, not a validated scoring against the Bi³ ratings.

`--run <folder>` (before the subcommand) is required and picks the run folder. `select` writes `<run>/run.json` with the trips folder and the git commit of the code. `run` writes it too for a new run folder (trips folder `trips_v3_human` for a person, `trips_v3` for the robot), with a copy of the trip list as `test_trips.json`, and adds the harness names, overrides, baselines and command line of each call. Every later command reads the trips from that folder. Each harness writes its settings to `<run>/<experiment>/<framing>/harness.json` when it first runs, and `run` refuses a harness whose settings differ from the stored ones. `analyze` reads each experiment's settings from its `harness.json`, and the baselines from their folders. `run` refuses to add episodes when the code is at another commit or has uncommitted changes in `preprocessing/`, `metrics/` or `traj_gen/`, so the episodes of a run folder come from one commit. `analyze` and `animate` read a run made at any commit on the same trips format: run folders made with the `trips_v2` folders (those directly under `/persistent/experiments/HRI/bi3_traj_gen/`) are read with the code at `traj_gen` ee7150f, and new runs go under `/persistent/experiments/HRI/bi3_traj_gen/trips_v3/`.

Each model episode is saved under `<run>/<experiment>/<framing>/<thinking>/<trip>/s<k>/`, and each baseline episode (`straight`, `replay`) under `<run>/<experiment>/<trip>/`. An episode folder holds `inputs/` (the images sent), `trajectory.csv`, `plans.jsonl`, `queries.jsonl`, `result.png` and `manifest.json`. `trajectory.csv` has the columns `t, robot_x, robot_y, human0_x, human0_y, ...`, one per agent in the trip's track order, from the trip start (`t` = 0) to the end of the episode. The manifest records the episode's status, metrics, query counts and the code's git commit and dirty flag. The status is `ok`, `invalid_reply` (an unusable reply: single shot at once, rolling after one retry) or `error` (a server or transport error, with the traceback in the manifest). A resumed `run` runs episodes that ended in `error` again.

`analyze` writes `<run>/analysis/`, with every table and figure split by the planned agent's kind (`robot` or `human`):
- `episodes.csv`: one row per episode.
- `summary.csv`: per kind, condition and site, the status counts and the episode-level rates and means, grouped by metric class.
- `paired_baselines.csv`: per kind, condition and site, each condition minus replay and minus straight, paired on the same trips, with bootstrap intervals over trips.
- `comparisons.csv`: condition comparisons (think - nothink, bev - text input, ...) with bootstrap intervals over trips.
- `similarity.csv` (person runs): per condition, site and feature, the Wasserstein-1 distance of the generated moves to real moves ([`metrics/human_similarity.py`](metrics/human_similarity.py)), against the site's moves from other trials and against the same test trips' real moves. The reference is the same distance for 20 matched random subsets of each other trial's moves (another participant pair), pooled, with the numbers of draws and trials it rests on.
- `similarity_pool.csv` (person runs): per site, the person trips in the real pool and those left out.
- `figures/`: condition dot plots (a row per experiment and framing, a column per metric, trip means with 95% bootstrap intervals, hollow for thinking off and filled for on, the straight-line and replay means as reference lines; `analyze.condition_figure` draws the same form for any chosen rows, pooled or not, in panels), speed-cap violations and one overlay per test trip (model conditions only, each panel drawing the straight line and the real path), in the figure scheme of [DESIGN.md](DESIGN.md), Section 3.

All of these are mocap-derived, so run folders stay outside the repository and must not be committed.

## Tests

Run the tests from the repository root with `python -m pytest tests`. `tests/test_snapshots.py` compares each preset's rendered prompts, the positions of a fixed reply, its settings and thinking budget, for a planned person and robot, with `tests/snapshots/<preset>_<kind>.json`. After an intended change to a preset or its prompts, regenerate them with `python -m tests.update_snapshots` and review the diff.

## Reproduce locally

1. Download Bi³ separately and place the extracted dataset so that `Bi3/Bi3/videos/laas/`, `Bi3/Bi3/jsons/laas/`, and `Bi3/Bi3/user_impressions/laas/` exist under this repository root. The data are not redistributed here.
2. Install Python 3.10 or newer and `pip install -r requirements.txt`. The preprocessing and trajectory-generation code (`preprocessing/sync_video.py`, `traj_gen/observation/camera.py`) calls the `ffmpeg` binary, which must be on PATH. The pilot preparation scripts in `pilot_run/` use the FFmpeg binary supplied by `imageio-ffmpeg`. Image preparation scripts currently use the Windows Arial font path used for the original run.
3. Preparation can generate a random A/B/C condition key and contact sheets. For a **new blinded evaluation**, do not read the mapping, mocap truth, or survey CSV until the new observer ratings are saved.
4. Run a run's `analyze.py` after preparation and rating to recompute its `results.json`. LAAS3 uses `pilot_run/analyse.py`.

Generated contact sheets, maps, frame caches, and saved outputs are omitted from Git. The model was operated through Codex during these exploratory rating pilots. For them there is no automated VLM API runner or claim of deterministic regeneration of the ratings.

## Study limitations

The Bi³ papers identify the 12 RoSAS items and nine-point scale but do not publish the exact online survey form. The revised prompt adapts the original RoSAS anchors. It is not verified verbatim Bi³ wording. Frame sampling misses continuous motion, sudden changes, and participant embodiment. Participants can disagree strongly even within one trial. See each report for the run's specific design and errors.
