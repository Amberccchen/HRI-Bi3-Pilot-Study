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

## Reproduce locally

1. Download Bi³ separately and place the extracted dataset so that `Bi3/Bi3/videos/laas/`, `Bi3/Bi3/jsons/laas/`, and `Bi3/Bi3/user_impressions/laas/` exist under this repository root. The data are not redistributed here.
2. Install Python 3 and `pip install -r requirements.txt`. The scripts call the FFmpeg binary supplied by `imageio-ffmpeg`. Image preparation scripts currently use the Windows Arial font path used for the original run.
3. Preparation can generate a random A/B/C condition key and contact sheets. For a **new blinded evaluation**, do not read the mapping, mocap truth, or survey CSV until the new observer ratings are saved.
4. Run a run's `analyze.py` after preparation and rating to recompute its `results.json`. LAAS3 uses `pilot_run/analyse.py`.

Generated contact sheets, maps, frame caches, and saved outputs are omitted from Git. The model was operated through Codex during these exploratory pilots; there is no automated VLM API runner or claim of deterministic regeneration of its ratings.

## Study limitations

The Bi³ papers identify the 12 RoSAS items and nine-point scale but do not publish the exact online survey form. The revised prompt adapts the original RoSAS anchors; it is not verified verbatim Bi³ wording. Frame sampling misses continuous motion, sudden changes, and participant embodiment. Participants can disagree strongly even within one trial. See each report for the run's specific design and errors.
