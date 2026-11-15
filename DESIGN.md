# Design: how the Bi³ data, metrics and trajectory generation are defined

This document states how each key quantity in this repository is defined, where the definition comes from, and which constant or function holds it. Sources are of three kinds, tagged on each entry:
- **Bi³ paper**: Stratton et al., ICRA 2026 ([arXiv 2605.06863](https://arxiv.org/html/2605.06863)), with the section.
- **Measured**: our measurement on the Bi³ data, with the number.
- **Design choice**: a choice made for this study, with its reason.

Code references are `path:symbol`. Where a measurement and the code differ, the code is what runs. Usage (commands, output files) is in the [README](README.md).

## 1. Dataset understanding and handling
These definitions are shared by every study in the repository.

### Bi³ basics
- **Two sites, one robot each.** LAAS-CNRS (Toulouse, France) with a PR2, footprint 0.668 x 0.668 m, and the University of Michigan (UM, Ann Arbor, USA) with a Hello Robot Stretch, footprint 0.330 x 0.340 m. *Source:* Bi³ paper, Sec. I and III-D. *Code:* `preprocessing/extract_trips.py:SITE_FRAMES` (`robot`, `footprint_m`).
- **Five controllers.** All plug into the same MPPI-based model-predictive controller and differ in how they treat people: no prediction (the robot does not perceive people), static obstacles, constant-velocity obstacles, Human Scene Transformer prediction, and CoHAN joint planning. *Source:* Bi³ paper, Sec. III-B to III-D. *Code:* `preprocessing/extract_trips.py:CONTROLLERS` uses the dataset's file names `blind`, `static`, `cv`, `hst`, `cohan`. The paper calls `blind` "NP" and `static` "ST".
- **A trial is one participant pair that ran all five controllers.** Two participants build towers by carrying blocks one at a time from piles to assembly stations along the room's perimeter, while the robot drives a random sequence of workstations that crosses the room between goals. Each pair runs all five controllers in a balanced Latin-square order, so a trial holds five recordings. There are 20 trials at LAAS and 17 at UM, 185 recordings in total. *Source:* Bi³ paper, Sec. IV-A, IV-B and V. A "recording" in this repository is one trial and controller.
- **Train/validation/test split.** Bi³'s test trials are `laas18`-`laas20` and `um16`, `um17`. *Source:* the Bi³ authors' code (fluentrobotics/Bi3, `create_data_npys.py`), not the paper. *Code:* `preprocessing/extract_trips.py:TEST_TRIALS`. The VLM planners are not trained on Bi³, so the split matters only for comparisons with learned models. The default trials for robot trips are `laas18` and `um13` (`DEFAULT_TRIALS`), because the UM test trials have no overhead video and the camera-input experiments need one.

### Room-centred frame and the LAAS rotation
- **Definition.** Every track and station is expressed in a room frame whose origin is the centre of the area people and the robot actually occupy, with LAAS rotated 180° so that both sites have their shared table on the +x wall. Origins in the mocap frame: LAAS (4.88, 15.62) m, UM (-0.23, 0.01) m. *Source:* design choice, so that one prompt layout and one map layout describe both sites. *Code:* `preprocessing/extract_trips.py:SITE_FRAMES`, `to_room`, `to_room_theta`.
- **Limitation.** The origin is set by hand from the occupied area, not from a surveyed room corner. The paper does not state the mocap frame.

### Traversable workspace, 3.5 x 3.5 m
- **Definition.** The room is the square of side 3.5 m around the room-frame origin. It is drawn on every map and is the base of the out-of-room check (Section 2). *Source:* Bi³ paper, Sec. IV-A ("traversable workspace"). *Code:* `preprocessing/extract_trips.py:SITE_FRAMES` (`room_m`).
- **Limitation.** The workstations stand along the perimeter, so people working at them stand at or just beyond its edge (Section 2, room bounds).

### Stations: robot goal points and people's points
- **Robot goal points.** Each site has six stations, numbered 0-5. A station's robot point is one of the six values the recorded robot goal takes at that site. *Source:* measured on the raw `robot_goal` field (Bi³'s own goal coordinates). *Code:* `preprocessing/annotate_goals.py:STATIONS`, written into `stations.json` and, in the room frame, as `xy` in `site_frames.json` (`preprocessing/extract_trips.py:site_frames`).
- **People's points.** A station's people's point is the median position at the turnaround (`t_turn`) over all non-inferred person visits with that label, over every recording of the site. Inferred visits are left out because their `t_turn` is the far point of a missed loop, not a turnaround (including them moves no median by more than 0.019 m). A shared table's point is the midpoint of its two stations' points. *Source:* measured. People's points lie 0.14-0.52 m from the robot points, toward the wall, and are stable across trials (the per-trial median lies a median 0.06-0.13 m from the site point, at most 0.32 m). *Code:* `preprocessing/extract_trips.py:human_goal_points`, stored as `human_xy` in `site_frames.json`. Test trials are included in the medians, since these are fixed station positions, not a fitted parameter.
- **Which point an agent uses.** A planned robot is given and scored against the robot point, a planned person against the people's point. *Code:* `traj_gen/agents.py:STATION_POSITION`. A person trip's `goal_xy` is its goal's `human_xy`.

### Shared tables
- **Definition.** One table holds two stations at each site: LAAS 0 and 1, UM 4 and 5. *Source:* measured: the station positions, and the camera frames show a real shared table at both sites. The paper does not describe the tables. *Code:* `preprocessing/annotate_goals.py:SHARED_TABLES`.
- **Consequence for labels.** Where people stand at a shared table (about 0.35 m apart at LAAS), position alone cannot tell the two stations apart. Visit detection resolves this by tower phase (below), and unresolved visits get the pair label `"0/1"` or `"4/5"`.

### Tables are not modelled as obstacles
- **Definition.** No table or furniture geometry enters the metrics. Stations are points, and the boxes on the maps mark where people go, not furniture. *Source:* design choice. The data give a station only as the robot's stopping point and the people's standing positions, so a table outline would be invented, and a free-space map built from the tracks adds machinery the study does not need. This is also why station crossing is not scored (Section 2).
- **Limitation.** The robot's system prompt still says that nobody passes through a station, so the model is told a constraint that is not scored.

### Visit detection and goal labels
- **Visit.** A person's visit to a station is a turnaround of at least 100° (direction change between the 0.4 m of path before and after a sample) within 1.0 m of a station, located at the closest approach of the turning run. Turnarounds at one station less than 2 s apart merge. The visit spans the contiguous run of samples within the visit's own closest approach plus 0.15 m (`t_arrive` to `t_leave`). *Source:* design choice, because participants shuttle blocks between stations and Bi³ records only the robot's goals. *Code:* `preprocessing/annotate_goals.py:detect_visits`, `TURN_MIN_DEG`, `TURN_LOOK_M`, `VISIT_RADIUS_M`, `MERGE_S`, `AT_STATION_SLACK_M`.
- **Inferred visits.** A wide U-turn can stay under 100°. When someone goes between two visits at the same or neighbouring stations (at most 1 m apart) via a point more than 1 m from both, that far point is filled in as a visit marked `inferred`. *Code:* `detect_visits`, `LOOP_M`.
- **Shared-table correction.** Each person's sequence is split into their two tower phases. A phase's visits to a shared table take the phase's margin-weighted majority station when it holds at least 70% of the vote (marked `corrected`), and keep the pair label otherwise. This resolves every UM phase but about 75% of LAAS phases. About 7% of LAAS person visits are `"0/1"`, and no UM visit needed a pair. *Code:* `correct_by_phase`, `PHASE_MIN_SHARE`.
- **Goal labels.** A backward pass gives every sample the station the person is at or heading to (`goal`) and the one after (`next_goal`). Labels are text: a station `"0"`-`"5"` or a pair. *Code:* `label_from_visits`. The robot's labels come from its recorded goals (`robot_labels`).
- **Check against ground truth.** The same detector run on the robot finds 96% of its recorded goals, and 90% of its detections match one. *Source:* measured (`robot_check`). Participants have a median of 40 visits per recording at both sites.
- **Limitation.** The 1.0 m radius only gates where a turnaround can be detected. The at-station zone is per visit (closest approach plus 0.15 m), and people are a median 0.61 m (LAAS) and 0.43 m (UM) from the robot point at `t_arrive`.

### Video sync
- **Definition.** Each overhead video's time offset from its mocap is the lag that maximises the cross-correlation of frame-to-frame pixel change with the agents' summed mocap speed, both at 10 Hz, searched within ±20 s: `video_time = mocap_time + offset_s`. *Source:* design choice. *Measured:* over all 139 videos the offsets range from -1.7 to +0.2 s, median about -1 s at both sites. *Code:* `preprocessing/sync_video.py`, read with `preprocessing/bi3_derived.py:video_time`.

### Trip cutting
- **Robot trip.** From one goal switch of the robot to the next. The first trip of a recording starts at the recording start, and the last goal is never reached, so it gives no trip. *Code:* `preprocessing/extract_trips.py:robot_trips`.
- **Person trip.** From the person leaving visit k (`t_leave`, where their goal label switches to the next station) to the turnaround of visit k + 1 (`t_turn`). Ending at the turnaround rather than at `t_arrive` gives the replay baseline a real path that reaches where people stand: at `t_arrive` the real path is still a median 0.61 m (LAAS) and 0.46 m (UM) from the people's point (measured). *Code:* `human_trips`.
- **Anchored 0.4 s grid.** Each trip is sampled at 2.5 Hz on a grid anchored at its start (`time` is 0 there), so the start state, the camera frame at the start and the real-path metrics share one time. The grid runs from up to 3 s before the start (history) to up to 15 s after the end (replay padding, so a slower generated agent still has the others to be scored against). `start_index` and `end_index` mark the first and last sample of the real trip. *Source:* design choice (the user's decision to anchor at the goal switch). *Code:* `trip_grid`, `preprocessing/bi3_derived.py:TRIP_HZ`, `TRIP_HISTORY_S`, `preprocessing/extract_trips.py:POST_S`.
- **Smoothing.** Once per recording, repeated mocap samples are dropped, the track is interpolated onto a 10 Hz grid and smoothed with a 0.5 s moving average, then resampled onto the trip grid. *Code:* `clean_track`, `SMOOTH_S`, `BASE_HZ`.
- **Counts.** The person trips of all 37 trials number 14,292 (LAAS 7710, UM 6582). The default robot trips (`laas18`, `um13`) number 144.

### The move-cutting rule, dropped hops and duplicates
- **Move.** The part of a person's path that the human-similarity features describe runs from the trip start to where the path enters its goal station, by the visit annotation's own rule: of the samples within the closest approach to the goal's robot points plus 0.15 m, the first of the contiguous run that ends at the closest approach. The closest approach is searched only up to the first sample within the arrival radius, where an episode stops, so a generated path and the real path of the same trip are cut alike (a replay episode's move equals its trip's real move). A path that never gets within the arrival radius plus `GOAL_SLACK` has no move. *Source:* design choice, so both sets measure the same segment. *Code:* `metrics/traj_gen_metrics.py:move_end_index`.
- **Visit pairs that give no trip.** A pair is dropped when the next visit's `t_arrive` is not after this visit's `t_leave`: 17 LAAS hops between neighbouring stations whose at-station zones overlap (0.2% of LAAS pairs), and 83 LAAS and 26 UM duplicate same-station records. Kept pairs at the same station (LAAS 43, UM 8, loops away from a station and back) are also dropped, because goal-to-goal generation cannot represent them. *Source:* measured. *Code:* `preprocessing/extract_trips.py:human_trips`.
- **Limitation.** The real pool has no examples of the 17 neighbouring-station hops.

## 2. Metrics
All metrics score one executed path of the planned agent against the other agents, replayed from the log. They are exploratory definitions for this pilot, not a validated scoring against the Bi³ ratings.

### Agents as disks
- **Definition.** Every agent is a disk around its tracked centre. A planned path has no heading, so the robot's corners cannot be modelled. *Source:* design choice. *Code:* `metrics/traj_gen_metrics.py` module docstring, `traj_gen/agents.py`.

### Body (scoring) radii
- **Robot.** The footprint's inscribed disk, half the shorter side: PR2 0.334 m, Stretch 0.165 m. *Source:* design choice, the realistic body. *Code:* `traj_gen/agents.py:body_radius`.
- **Person.** 0.15 m, because mocap gives only the body centre. *Source:* design choice (the user's, "the more generous radius"). *Code:* `traj_gen/agents.py:PERSON_BODY_RADIUS_M`.
- **Measured basis.** At these radii, real controllers touch a person (clearance below 0) in 2 of the 16 robot test trips of the first robot-generation run, drawn from `laas18` and `um13` (clearances -0.008 and -0.018 m) and in 3 of the 133 robot trips with a full prompt history. Real people touch the robot in 0.44% (LAAS) and 0.33% (UM) of their moves. *Inferred:* these are unlikely to be real collisions, because both bodies are modelled as circles. *Code:* the counts are in the `metrics/traj_gen_metrics.py` docstring.

### Personal space
- **Definition.** A person has a personal space of radius 0.25 m around their centre, a 0.10 m buffer beyond the body radius. The robot has none. `personal_space_overlap` is the share of the episode's time steps with the planned agent's body inside some person's personal space. *Source:* design choice, not a measurement, with 0.20 m as the sensitivity value (`traj_gen.report_figures` writes that table). Giving the robot a personal space would put a social notion on the robot that the study does not ask for. *Measured:* real people enter the other person's personal space in 0.1% (LAAS) and 0.2% (UM) of moves. *Code:* `traj_gen/agents.py:PERSONAL_SPACE_RADIUS_M`, `metrics/traj_gen_metrics.py:clearance_metrics`.

### Arrival radii and the scoring slack
- **Robot, 0.5 m.** The robot is told it has arrived within 0.5 m of its goal's robot point, and its episode ends there. *Source:* measured: Bi³'s recorded goal switches when the robot is within about 0.5 m of the goal. The paper does not state a switch rule. *Code:* `traj_gen/agents.py:ARRIVAL_RADIUS_M`.
- **Person, 0.4 m.** Around the goal's people's point. *Source:* measured. Real turnarounds lie a median 0.17 m (LAAS) and 0.16 m (UM) from the point, 90th percentile 0.36 and 0.30 m, so 0.4 m is about where people's own standing scatter ends. Real person trips come within 0.4 m of the goal in 91.1% (LAAS) and 89.5% (UM) of trips, and within the scored 0.55 m (0.4 m plus the 0.15 m slack) in 97.0% and 97.9% (7,709 and 6,582 trips of trips_v2_human that start outside 0.4 m). *Code:* `traj_gen/agents.py:ARRIVAL_RADIUS_M`.
- **Slack, 0.15 m.** `goal_reached` counts a path that gets within the arrival radius plus 0.15 m (robot 0.65 m, person 0.55 m). *Source:* measured: the cleaned room-frame real robot paths end 0.48-0.60 m from the goal at their last sample before Bi³'s goal switch, at most 0.602 m over all 144 robot trips, and 0.15 m is a round value with margin above that. With it, every real controller reaches its goal. *Code:* `metrics/traj_gen_metrics.py:GOAL_SLACK`, `traj_gen_metrics`.

### One centre per station (multimodality)
- **Definition.** Each station has one centre per agent kind, and the arrival check is a disk around it. *Source:* design choice (the user's). *Measured:* per station, the radius holding 95% of real person arrivals is 0.37-0.52 m at LAAS and 0.32-0.42 m at UM. Arrivals spread more along the wall (95% half-width 0.23-0.50 m) than across it (0.16-0.36 m), with no sign of two separate clusters on the one-number check made (mean offset across the wall within 0.013 m of zero everywhere).
- **Limitation.** If arrivals cluster on separate sides of a station, a single centre overstates their spread and the arrival check becomes lenient. The check above is a summary number, not a histogram.

### Metric classes and the invalid rate
- **Classes.** Every metric belongs to one class. `invalid`: the plan breaks a hard rule of the task (an unusable reply is invalid too). `contact`: closeness to the other agents, a failure of the planner's reasoning rather than of a hard constraint. `efficiency`: how the agent moves. `similarity`: how person-like a generated person's moves are (person runs only, reported in `similarity.csv`). *Source:* design choice (the user's). *Code:* `metrics/traj_gen_metrics.py:METRIC_CLASS`.
- **Parts of the invalid rate.** An episode is invalid when its reply is unusable (status `invalid_reply`), the server failed (status `error`), its plan exceeds the speed cap in any step, it does not reach the goal, or it leaves the room. Each part and the total are shares of all episodes of the condition. An episode without a path counts only in its status part, and counts as goal not reached. The speed-cap part is NaN for the baselines, which have no model reply. *Code:* `traj_gen/analyze.py:scored`, `INVALID_PARTS`.

### Room bounds
- **Definition.** An agent is out of the room when its centre leaves the square of half side 1.75 m (robot) or 2.0 m (person) around the room centre: the 3.5 m workspace plus a margin of 0 m or 0.25 m. *Source:* measured for people: people stand closer to the walls than the robot's stopping points, and real people leave the 1.75 m square in 11.9% (LAAS) and 35.7% (UM) of moves but the 2.0 m square in only 0.03% and 0.41%. For the robot, the station positions and walls bound it. *Code:* `traj_gen/agents.py:ROOM_MARGIN_M`, applied in `traj_gen/episodes.py` and `metrics/traj_gen_metrics.py:traj_gen_metrics` (`out_of_room`).

### Collisions on the executed path
- **Definition.** `clearance_min` is the smallest centre distance minus both body radii, over the other agents and time, so below 0 the bodies touch. `contact_time` is the share of time steps in contact, and `contact_events` counts separate contacts (a run of consecutive steps counts once). All are computed on the executed path only, not on a longer plan. *Source:* design choice (the user's): not all output modes plan a long horizon and then re-plan, so only executed motion is comparable. *Code:* `metrics/traj_gen_metrics.py:clearance_gap`, `clearance_metrics`.
- **Limitation.** A touch between two 0.4 s samples is missed.

### No station-crossing metric
- **Definition.** No metric checks whether an agent passes through a station. *Source:* design choice (the user's): the tables are not well defined in the data (Section 1), and real people cross the station boxes in 11.9% of LAAS moves. *Code:* `traj_gen/analyze.py:SKIP` skips the two former station-crossing metric names found in older run folders.

### Efficiency metrics
- **Definition.** `time_to_goal` (first sample within the arrival radius plus slack), `path_efficiency` (straight start-to-end distance over distance travelled), path length, and speed and acceleration statistics. Speed, acceleration and path efficiency follow the Bi³ paper's robot metrics. The paper defines only path efficiency precisely, so speed is reported as mean, median, max and standard deviation, and acceleration as the mean magnitude of the 2-D acceleration (with the tangential mean and the max). Bi³'s minimum distance (centre to centre) is reported too, in the contact class. *Source:* Bi³ paper, Sec. VI and Fig. 7. *Code:* `metrics/bi3_metrics.py`.

### Human-similarity measures
- **Features.** Per move: speed per 0.4 s step, the absolute turning angle between consecutive 0.8 s displacements, the minimum centre distance to the robot, and the minimum centre distance to the other person (one value per move). *Source:* THÖR (Rudenko et al., RA-L 2020) supplies the kinds of feature, not their definitions (the user's decision). *Code:* `metrics/human_similarity.py:trajectory_features`, `FEATURE_NAMES`.
- **0.8 s turning window.** THÖR measures curvature over a 4 s segment, longer than a median Bi³ move (3.0 s), so the window is two 0.4 s steps. A turning angle is kept only when both displacements are at least 0.15 m: below that, the median absolute turn is close to the 90° of a random direction (measured: below 0.10 m at LAAS and 0.15 m at UM). The threshold removes about 6% (LAAS) and under 5% (UM) of 0.8 s displacements. A move needs 1.6 s to give any turning sample. *Code:* `metrics/human_similarity.py:TURN_WINDOW_STEPS`, `traj_gen/analyze.py:TURN_MIN_DISPLACEMENT_M`.
- **Wasserstein-1 distance (W1) per feature.** Two sets of moves are compared per feature by the W1 distance of their pooled samples, in the feature's units, with a bootstrap interval over moves resampled within (start station, goal station) strata. *Code:* `metrics/human_similarity.py:feature_distances`.
- **Reference drawn by trial.** Each W1 comes with a real-against-real reference: for every trial of the site that none of the generated moves comes from, its moves matched to the generated set's station mix (as far as the trial allows) are compared with the real set without that trial. A generated set comes from a few trials, and a reference drawn independently across trials would leave out how much participant pairs differ from one another. The reference interval is the 2.5 and 97.5 percentiles over trials, or the min and max below 40 trials. *Source:* design choice. *Measured:* at LAAS, a replay of one recording's real moves fell outside a reference of moves drawn independently of their trial (speed W1 0.086 m/s against a reference median of 0.015 m/s). *Code:* `feature_distances`, `MIN_PERCENTILE_DRAWS`.
- **Two comparisons.** Against the site pool (the site's real moves from trials other than the generated moves' trials) and against the same test trips' real moves. *Code:* `traj_gen/analyze.py:similarity`.
- **Pool exclusions.** The real pool is every person trip of the site in the run's trips folder, minus trips that start within the arrival radius and trips whose real path has no move. Generated episodes that are not `ok` or have no move are left out and counted. *Code:* `traj_gen/analyze.py:real_pool`, written to `similarity_pool.csv`.
- **Limitation.** The bootstrap interval sits above W1 when the generated set is close to the real set, so in that range the reference interval is the comparison that tells a real-like set from an unlike one.

## 3. Generation-specific handling

### The planned agent
- **Definition.** Each trip names the agent being generated (`trip["agent"]`: `robot` or `human<k>`). All other entries of `trip["tracks"]` are replayed from the log. A person is generated as one of the real participants, to that person's actual next station, open loop. *Source:* design choice (the user's). *Code:* `traj_gen/agents.py:agent_kind`, `traj_gen/episodes.py:Episode`.

### What each generator is told
- **Robot.** Its own goal at the robot point, the station robot points, and each person's position, recent motion, goal and next goal. *Code:* `traj_gen/scene.py:Scene.text`, `traj_gen/prompts/scene.txt`, `person.txt`.
- **Person.** Their own goal and next goal at the people's points, and the others' positions and recent motion, plus the station the other person is standing at, but no one's goals. *Source:* design choice: the generator stands in for a participant, and Bi³ participants did not know each other's next station. *Code:* `traj_gen/scene.py:Scene.other_text`, `traj_gen/prompts/scene_human.txt`, `person_human.txt`, the `participant` framing in `traj_gen/prompting.py:FRAMINGS`.
- **Limitation.** The robot is given the people's goals, which a real robot would not know.

### Prompt radius versus scoring radius
- **Definition.** Prompts and model maps use larger disks than scoring: the robot's circumscribed disk (PR2 0.472 m, Stretch 0.237 m) and a 0.25 m person (a typical shoulder half-width). Scoring uses the body radii of Section 2. *Source:* design choice (the user's): planned paths kept colliding, so the prompts are deliberately stricter than the metrics. *Code:* `traj_gen/agents.py:prompt_radius`, `PERSON_PROMPT_RADIUS_M`.

### Speed caps
- **Robot, 0.3 m/s.** *Source:* measured. The real controllers' top speed in a trip has a median of 0.285 m/s (LAAS) and 0.291 m/s (UM) over the 144 robot trips, so 0.3 m/s is the speed the real robots drive at. *Code:* `traj_gen/agents.py:SPEED_CAP_M_S`.
- **Person, 1.5 m/s, one cap for every person.** *Source:* measured: the pooled 99th percentile of real per-step speed (0.4 s steps, raw mocap) over goal-to-goal moves of both sites is 1.501 m/s (LAAS 1.358, UM 1.603). 0.17% (LAAS) and 2.0% (UM) of real steps exceed it. Higher percentiles are not used because UM's tail above 2 m/s is inferred to be mocap jumps: it reaches 4.5 m/s (0.26% of steps), which no one walks in a 3.5 m room.
- **How a cap acts.** A plan is walked at the cap along its own shape, so a too-fast plan falls behind its timing. The plan's raw speeds before limiting give `speed_violation_rate` and `raw_speed_max`, and any step above the cap makes the episode invalid. *Code:* `traj_gen/planners.py:retime`.

### Output modes and episode types
- **Output modes.** Timed waypoints in metres, points on the map in pixels, a velocity sequence, and a single velocity step. A plan has at most 75 steps (30 s). *Code:* `traj_gen/planners.py:WaypointPlanner`, `PixelPlanner`, `VelocityPlanner`, `StepVelocityPlanner`, `MAX_PLAN_STEPS`.
- **Episode types.** Single shot (one query at the start, the whole plan executed), rolling (re-query every 2 s of executed motion in a fresh chat, every 0.4 s for the single velocity step), and conversation (the same in one growing chat). An episode ends at the first executed step within the arrival radius, or when the replay ends. *Code:* `traj_gen/episodes.py`, `traj_gen/experiments.py:EXPERIMENTS`.

### Baselines
- **Straight line.** Straight to the goal at 0.3 m/s for the robot (its cap, since the model-driven robot mostly drives at the cap) and at 0.94 m/s for a person (the median real per-step speed over the same goal-to-goal moves, 0.942 m/s). *Source:* design choice (the user's) and measured. *Code:* `traj_gen/agents.py:STRAIGHT_LINE_SPEED_M_S`, `traj_gen/planners.py:StraightPlanner`.
- **Replay.** The planned agent's real path. *Code:* `traj_gen/planners.py:ReplayPlanner`.

### Map drawing
- **Definition.** A fixed 896 px image of a 4.0 m extent (the room plus 0.25 m each side), 1 m = 200 px, so a drawn mark maps back to metres exactly. The station boxes are drawn at the people's points moved perpendicular onto the nearest side of the 3.5 m square, with a shared table as one box, so every map of a site has one layout, for robot and person runs and for model inputs and analysis figures alike. Only the goal star and arrival circle are drawn at the planned agent's own point. *Source:* design choice (the user's): the paper places workstations along the perimeter, and one drawing is simpler to share. The moved boxes are used only in the drawing: goal points, prompt text and metrics use the station points of Section 1. *Code:* `traj_gen/bev.py:draw_room`, `SIZE_PX`, `EXTENT_M`, `STATION_BOX_M`.
- **Limitation.** For a planned robot, the goal star sits at the robot point, inside the room, while the station boxes sit on the border.

### Figure scheme
- **Definition.** Colour says who controls an agent (purple the VLM, blue the logged robot, orange a logged person), and marker shape says what it is (square robot, triangle person). Final-position disks in analysis figures are drawn at the scoring radius. Maps given to the model keep the prompt radii. *Source:* design choice (the user's). *Code:* `traj_gen/bev.py:VLM_COLOUR`, `KIND_COLOUR`, `KIND_MARKER`, `render_result`, `render_overlay`, `trajectory_legend`.

### Run folders and provenance
- **Definition.** `select` writes `run.json` with the trips folder and the code's git commit. `run` refuses to add episodes when the code is at another commit or has uncommitted changes, so a run folder's episodes come from one commit. Each episode manifest records its status, metrics and the commit. Each preprocessing output has a manifest with the commit, a dirty flag and its parameters. *Source:* design choice (the user's: "the commit is likely enough"). *Code:* `preprocessing/bi3_derived.py:provenance`, `write_manifest`, `traj_gen/run.py:select`, `run`.

## 4. Discriminator
This section will hold the definitions used by the rating-prediction work.

## 5. Known limitations
- **Replayed agents cannot react.** The other agents follow their logs, so they do not respond to the planned agent, and a real controller's replay is flattered by the people having reacted to it. The straight line is the fair baseline. This is a stated limitation, not something a planner is run to fix (the user's decision).
- **Arrival multimodality.** One centre per station (Section 2) makes the arrival check lenient if arrivals cluster on separate sides of a station.
- **Robot and site are confounded.** Robot morphology and culture differ together across the two sites, so site differences cannot be attributed to either. *Source:* Bi³ paper, Sec. VII.
- **Disk bodies.** Contacts are measured between circles, so a contact is not a measured collision, and the robot's corners are not modelled.
- **Exploratory scoring.** The metrics, radii and thresholds here are exploratory, not validated against the Bi³ ratings.
- **Tables are not scored.** Stations are points, and passing through a table is neither modelled nor scored, although the prompts forbid it.
