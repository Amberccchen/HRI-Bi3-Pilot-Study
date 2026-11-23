# Notable runs

The short list of runs whose results matter. Folders are on the author's machine under `/persistent/experiments/HRI/bi3_traj_gen/`, and the index of every run folder there is `README.md`.

These runs used the `trips_v2` folders (the old extraction: tracks smoothed with a 0.5 s moving average) and are read with the code at `traj_gen` ee7150f. Runs on the `trips_v3` folders go under `/persistent/experiments/HRI/bi3_traj_gen/trips_v3/` and get their own table. Commits 6698001, 3177648, f2b3123 and b2aaeb7 are on `origin/archive/robot_wording_test`, not on `traj_gen`.

Harness names: "simple harness" is the preset `pilot` (map positions, one conversation, re-plan every 2 s). "Harness" is `harness_v1` (turn-and-distance steps with the scored distances in the prompt). "+ future trajs" gives the model the other agents' actual future trajectories, and "iter check" adds the plan check against them. Real-data reference band: how far one recorded participant pair is from the others.

## Person
| Folder | Ran | Model | Finding |
| --- | --- | --- | --- |
| `20261115_pilot_52d7b27` | simple harness, 100 trips x 3 | Qwen3.8-27B | Speed, distances and contact close to real; turning angle above the reference band. [Results](https://claude.ai/artifact/1qhLpDxXLitS2Spoq71jqP) |
| `20261123_human_pilot_setup_claude_sonnet_01af90a` | simple harness, same trips | Claude Sonnet 5.5, medium | Slower than real (median 0.90 vs 1.05 m/s); contact in 13% of episodes (Qwen 5%). [Results](https://claude.ai/artifact/F11CM1dCUsaurmbYuJGUmv) |
| `20261117_human_m4a_f2b3123` | harness with the largest step stated, without and with + future trajs | Qwen3.8-27B | Walks at the 1.5 m/s cap (real 1.05); contact 8% of episodes without futures, 1% with. |
| `20261123_human_harness_rate_claude_sonnet_b2aaeb7` | harness (= `harness_v1`) | Claude Sonnet 5.5, medium | Still near the cap (median 1.50 m/s); turning above the band; contact in 15% of episodes. |

## Robot
| Folder | Ran | Model | Finding |
| --- | --- | --- | --- |
| `20261116_robot_px_f379f0c` | simple harness, 16 robot trips x 3 | Qwen3.8-27B | Reference. On six other trips (3 close encounters) only 4 of 18 episodes are contact-free. |
| `20261117_future_ablation_t17_6698001` | harness, hardest trip x 5 | Qwen3.8-27B | Contact share 0.19, no contact-free episode. |
| `20261116_robot_space_check_3177648` | harness + future trajs, and iter check, hardest trip x 5 | Qwen3.8-27B | Contact share 0.017 with futures; 0 in 5 of 5 with the check, 3 s slower than the real robot. |
| `20261116_robot_check_trips_6698001` | iter check harness + future trajs, six other trips x 3 | Qwen3.8-27B | No contact in 18 of 18 (simple harness: 4 of 18). The model sees the same futures it is scored on. |
