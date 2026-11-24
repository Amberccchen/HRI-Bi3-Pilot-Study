# Notable runs

The short list of runs whose results matter. Folders are on the author's machine under `/persistent/experiments/HRI/bi3_traj_gen/`, and the index of every run folder there is `README.md`.

These runs used the `trips_v2` folders (the old extraction: tracks smoothed with a 0.5 s moving average) and are read with the code at `traj_gen` ee7150f. Runs on the `trips_v3` folders go under `/persistent/experiments/HRI/bi3_traj_gen/trips_v3/` and get their own table. Commits 6698001, 3177648, f2b3123 and b2aaeb7 are on `origin/archive/robot_wording_test`, not on `traj_gen`.

The findings below are scored on the `trips_v3` folders: each episode's saved path against the v3 tracks of the other agents, with the v3 real moves for the replay baseline and the real-data reference band. Those tables are in each run's `analysis_trips_v3/` (its `manifest.json` names the tool and code), and each run's `analysis/` keeps the original `trips_v2` scores.

`traj_gen` ee7150f is the last commit where the presets `pilot` and `harness_v1` reproduce these runs, and their successors `pilot_v2` and `harness_v2` keep the same setup on the new inputs, units setting and 448 px map.

Harness names: "simple harness" is the preset `pilot` (map positions, one conversation, re-plan every 2 s). "Harness" is `harness_v1` (turn-and-distance steps with the scored distances in the prompt). "+ future trajs" gives the model the other agents' actual future trajectories, and "iter check" adds the plan check against them. Real-data reference band: how far one recorded participant pair is from the others.

## Person
| Folder | Ran | Model | Finding |
| --- | --- | --- | --- |
| `20261115_pilot_52d7b27` | simple harness, 100 trips x 3 | Qwen3.8-27B | Speed, distances and contact close to real; turning angle above the reference band, except against the same trips at LAAS. [Results](https://claude.ai/artifact/1qhLpDxXLitS2Spoq71jqP) |
| `20261123_human_pilot_setup_claude_sonnet_01af90a` | simple harness, same trips | Claude Sonnet 5.5, medium | Slower than real (median 0.90 vs 1.07 m/s); contact in 14% of episodes (Qwen 6%). [Results](https://claude.ai/artifact/F11CM1dCUsaurmbYuJGUmv) |
| `20261117_human_m4a_f2b3123` | harness with the largest step stated, without and with + future trajs | Qwen3.8-27B | Walks at the 1.5 m/s cap (real 1.07); contact 8% of episodes without futures, 1% with. |
| `20261123_human_harness_rate_claude_sonnet_b2aaeb7` | harness (= `harness_v1`) | Claude Sonnet 5.5, medium | Still near the cap (median 1.50 m/s); turning above the band; contact in 15% of episodes. |

## Person, `trips_v3` folders
Scored with the code of each run's own commit. Skills: the model picks walk, pass (social force around one named agent, on a chosen side), wait or continue every 0.4 s, and code moves the person. Report: `/persistent/projects/HRI/scratchpad/bi3_skill_gen/full_study/report/REPORT.md`.

| Folder | Ran | Model | Finding |
| --- | --- | --- | --- |
| `trips_v3/20261124_pilot_v2_full_8ccccaa` | `pilot_v2` (map positions, re-plan every 0.8 s), 100 trips x 3 | Qwen3.8-27B | Speed and distances within or below the reference band; turning above it (UM, and the LAAS point estimate); contact in 2% of episodes (recorded moves 3%). |
| `trips_v3/20261124_skills_future_0dae446` | skills (`harness_v2`, `output=skills`) + future trajs, same trips | Qwen3.8-27B | W1 like the positions arm, turning inside the band at LAAS; faster than real (median 1.25 vs 1.07 m/s); contact in 19% of episodes, mostly while passing the LAAS robot. |
| `trips_v3/20261124_skills_nofuture_0dae446` | skills without future trajs, same trips | Qwen3.8-27B | Same as with future trajs (contact 19%). |

## Robot
| Folder | Ran | Model | Finding |
| --- | --- | --- | --- |
| `20261116_robot_px_f379f0c` | simple harness, 16 robot trips x 3 | Qwen3.8-27B | Reference. On six other trips (3 close encounters) only 4 of 18 episodes are contact-free. |
| `20261117_future_ablation_t17_6698001` | harness, hardest trip x 5 | Qwen3.8-27B | Contact share (time steps in contact) 0.19, no contact-free episode. |
| `20261116_robot_space_check_3177648` | harness + future trajs, and iter check, hardest trip x 5 | Qwen3.8-27B | Contact share 0.006 with futures (personal-space share 0.017); no contact in 5 of 5 with the check, 3 s slower than the real robot. |
| `20261116_robot_check_trips_6698001` | iter check harness + future trajs, six other trips x 3 | Qwen3.8-27B | No contact in 18 of 18 (simple harness: 4 of 18). The model sees the same futures it is scored on. |
