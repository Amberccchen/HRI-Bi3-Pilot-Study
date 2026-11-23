import json
from dataclasses import replace

import pytest

from traj_gen.harness.experiments import EXPERIMENTS, LEGACY_IDS
from traj_gen.harness.presets import HARNESS_FILE, PRESETS, Harness, parse_overrides, run_experiments, settings_of, write_harness_file


# Registered id (the folder name of the runs before the units setting) -> the id its settings derive.
DERIVED_IDS = {
    "ss_wp_text": "ss_positions_m_text", "ss_wp_bev": "ss_positions_m_bev", "ss_wp_bev_cam": "ss_positions_m_bev_cam", "ss_px_bev": "ss_positions_px_bev",
    "rh_wp_bev": "rh_positions_m_bev", "rh_wp_bev_cam": "rh_positions_m_bev_cam", "rh_px_bev": "rh_positions_px_bev", "rh_vel1_bev": "rh_vel1_m_bev",
    "rh_velseq_bev": "rh_velseq_m_bev", "rh_wp_bev_cam_conv": "rh_positions_m_bev_cam_conv", "rh_px_bev_conv": "rh_positions_px_bev_conv",
    "rh_velseq_bev_conv": "rh_velseq_m_bev_conv", "rh_steer_bev_conv": "rh_steer_px_bev_conv",
    "rh_steer_bev_conv_true_future": "rh_steer_px_bev_conv_true_future", "rh_steer_bev_conv_true_future_check": "rh_steer_px_bev_conv_true_future_check",
    "ss_wp_bev_constvel_future": "ss_positions_m_bev_constvel_future", "rh_wp_bev_constvel_future": "rh_positions_m_bev_constvel_future",
    "rh_vel1_bev_constvel_future": "rh_vel1_m_bev_constvel_future"}


def framing_for(exp) -> str:
    return "scored_distances" if exp.planner.scored_distance_framings else "neutral"


def test_registered_experiments_come_out_of_their_settings_unchanged_but_for_the_id():
    """Each registered model experiment, rebuilt from its settings, must give the same planner, units, images, episode, replanning interval,
    prediction and plan check, under the id the settings derive."""
    for exp_id, exp in EXPERIMENTS.items():
        if not exp.uses_model:
            continue
        rebuilt = Harness(**settings_of(exp, framing_for(exp))).experiment()
        assert rebuilt.id == DERIVED_IDS[exp_id]
        assert replace(rebuilt, id=exp.id, base=exp.base) == exp, exp_id
        assert rebuilt.base == (DERIVED_IDS[exp.base] if exp.base else None)
    assert LEGACY_IDS == {"rh_wp_bev_conv": "rh_wp_bev_cam_conv"} and "rh_wp_bev_conv" not in EXPERIMENTS


def test_preset_ids_and_framings_per_kind():
    expected = {("pilot_v2", "human"): ("rh_positions_px_bev_conv", "participant"), ("pilot_v2", "robot"): ("rh_positions_px_bev_conv", "neutral"),
                ("harness_v2", "human"): ("rh_steer_px_bev_conv", "scored_distances"), ("harness_v2", "robot"): ("rh_steer_px_bev_conv", "robot_scored_distances")}
    assert {(name, kind): (p.harness(kind).id, p.harness(kind).framing) for name, p in PRESETS.items() for kind in ("human", "robot")} == expected


def test_overrides_give_the_harness_levels_of_the_registry():
    """harness_v2 plus prediction=true_future, and plus check=2, must be the registered true_future and plan-check experiments (under their
    derived ids)."""
    def same(exp, exp_id):
        return exp == replace(EXPERIMENTS[exp_id], id=DERIVED_IDS[exp_id])
    for kind in ("human", "robot"):
        assert same(PRESETS["harness_v2"].harness(kind, {"prediction": "true_future"}).experiment(), "rh_steer_bev_conv_true_future")
        assert same(PRESETS["harness_v2"].harness(kind, {"prediction": "true_future", "check": 2}).experiment(), "rh_steer_bev_conv_true_future_check")
    assert same(PRESETS["pilot_v2"].harness("robot", {"episode": "single_shot"}).experiment(), "ss_px_bev")
    assert same(PRESETS["pilot_v2"].harness("robot", {"output": "vel1", "episode": "rolling", "units": "m"}).experiment(), "rh_vel1_bev")  # vel1's own default interval
    assert PRESETS["pilot_v2"].harness("human", {"framing": "scored_distances", "output": "steer"}).framing == "scored_distances"


def test_id_suffixes_for_check_count_and_replanning_interval():
    def h(**overrides):
        return PRESETS["harness_v2"].harness("human", {"prediction": "true_future", **overrides})
    assert h(check=3).id == "rh_steer_px_bev_conv_true_future_check3"
    assert h(execute_s=0.8).id == "rh_steer_px_bev_conv_true_future_replan0.8s"
    assert h(execute_s=2.0).id == "rh_steer_px_bev_conv_true_future"  # the default adds nothing
    assert h(units="m").id == "rh_steer_m_bev_conv_true_future"
    assert Harness("rolling", "vel1", "m", "bev", "neutral").execute_s == 0.4 and Harness("single_shot", "positions", "px", "bev", "neutral").execute_s is None


@pytest.mark.parametrize("settings, message", [
    ({"check": 2}, "needs prediction=true_future"),
    ({"output": "positions"}, "scored-distance framings"),
    ({"units": "cm"}, "units='cm' is not one of"),
    ({"images": "text"}, "units px are pixels of the map"),
    ({"framing": "participant"}, "scored-distance framings"),
    ({"output": "steer_vel", "framing": "participant"}, "scored-distance framings"),
    ({"episode": "single_shot", "execute_s": 2.0}, "only for rolling episodes"),
    ({"execute_s": 0.5}, "whole number of 0.4 s steps"),
    ({"prediction": "future"}, "prediction='future' is not one of"),
    ({"check": -1}, "check must be a whole number"),
])
def test_invalid_harness_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        PRESETS["harness_v2"].harness("human", settings)


def test_velocity_outputs_are_refused_for_a_planned_person():
    """There are no person templates for the velocity outputs, so a person run would fail at its first query."""
    with pytest.raises(ValueError, match="no person templates"):
        Harness("rolling", "velseq", "m", "bev", "participant")


def test_overrides_are_parsed_and_checked():
    assert parse_overrides(["prediction=true_future", "check=2", "execute_s=0.8"]) == {"prediction": "true_future", "check": 2, "execute_s": 0.8}
    for items, message in ((["checks=2"], "not a harness setting"), (["check"], "not KEY=VALUE"), (["check=two"], "whole number"),
                           (["execute_s=soon"], "number of seconds"), (["check=1", "check=2"], "given twice")):
        with pytest.raises(ValueError, match=message):
            parse_overrides(items)


def test_harness_file_is_written_once_and_a_changed_harness_is_refused(tmp_path):
    h = PRESETS["harness_v2"].harness("human")
    write_harness_file(tmp_path, "harness_v2", h, {})
    path = tmp_path / "rh_steer_px_bev_conv" / "scored_distances" / HARNESS_FILE
    assert json.loads(path.read_text())["settings"] == h.settings()
    write_harness_file(tmp_path, "harness_v2", h, {})  # the same settings again: accepted
    stored = json.loads(path.read_text())
    stored["settings"]["execute_s"] = 0.8  # as if the default interval had changed in the code since the first run
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="holds other settings"):
        write_harness_file(tmp_path, "harness_v2", h, {})


def test_analysis_reads_harness_files_and_falls_back_to_the_registry(tmp_path):
    """A folder with harness.json is read from it (also an id the registry does not know), a folder without from the registry,
    and the old folder name rh_wp_bev_conv as rh_wp_bev_cam_conv. Other folders are ignored."""
    new = PRESETS["harness_v2"].harness("human", {"execute_s": 0.8})
    write_harness_file(tmp_path, "harness_v2", new, {"execute_s": 0.8})
    for folder in ("rh_px_bev_conv", "rh_wp_bev_conv", "replay", "smoke"):
        (tmp_path / folder).mkdir()
    found = run_experiments(tmp_path)
    assert set(found) == {new.id, "rh_px_bev_conv", "rh_wp_bev_conv", "replay"}
    assert found[new.id] == new.experiment() and found["rh_wp_bev_conv"].id == "rh_wp_bev_cam_conv" and found["replay"] == EXPERIMENTS["replay"]
