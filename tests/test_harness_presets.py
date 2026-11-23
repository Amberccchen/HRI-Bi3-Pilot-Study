import json

import pytest

from traj_gen.harness.experiments import EXPERIMENTS, LEGACY_IDS
from traj_gen.harness.presets import HARNESS_FILE, PRESETS, Harness, parse_overrides, run_experiments, settings_of, write_harness_file


def test_registered_experiments_come_out_of_their_settings_unchanged():
    """Each registered model experiment, rebuilt from its settings, must give the same id, planner, images, episode, replanning interval,
    prediction and plan check. The one renamed id keeps its old folder name as a legacy alias."""
    for exp_id, exp in EXPERIMENTS.items():
        if not exp.uses_model:
            continue
        framing = "harness" if exp.planner.harness else "neutral"
        assert Harness(**settings_of(exp, framing)).experiment() == exp, exp_id
    assert LEGACY_IDS == {"rh_wp_bev_conv": "rh_wp_bev_cam_conv"} and "rh_wp_bev_conv" not in EXPERIMENTS


def test_preset_ids_and_framings_per_kind():
    expected = {("pilot", "human"): ("rh_px_bev_conv", "participant"), ("pilot", "robot"): ("rh_px_bev_conv", "neutral"),
                ("harness_v1", "human"): ("rh_steer_bev_conv", "harness"), ("harness_v1", "robot"): ("rh_steer_bev_conv", "robot_harness")}
    assert {(name, kind): (p.harness(kind).id, p.harness(kind).framing) for name, p in PRESETS.items() for kind in ("human", "robot")} == expected


def test_overrides_give_the_harness_levels_of_the_registry():
    """harness_v1 plus prediction=true_future, and plus check=2, must be the registered true_future and plan-check experiments."""
    for kind in ("human", "robot"):
        assert PRESETS["harness_v1"].harness(kind, {"prediction": "true_future"}).experiment() == EXPERIMENTS["rh_steer_bev_conv_true_future"]
        assert PRESETS["harness_v1"].harness(kind, {"prediction": "true_future", "check": 2}).experiment() == EXPERIMENTS["rh_steer_bev_conv_true_future_check"]
    assert PRESETS["pilot"].harness("robot", {"episode": "single_shot"}).experiment() == EXPERIMENTS["ss_px_bev"]
    assert PRESETS["pilot"].harness("robot", {"output": "vel1", "episode": "rolling"}).experiment() == EXPERIMENTS["rh_vel1_bev"]  # vel1's own default interval
    assert PRESETS["pilot"].harness("human", {"framing": "harness", "output": "steer"}).framing == "harness"


def test_id_suffixes_for_check_count_and_replanning_interval():
    def h(**overrides):
        return PRESETS["harness_v1"].harness("human", {"prediction": "true_future", **overrides})
    assert h(check=3).id == "rh_steer_bev_conv_true_future_check3"
    assert h(execute_s=0.8).id == "rh_steer_bev_conv_true_future_replan0.8s"
    assert h(execute_s=2.0).id == "rh_steer_bev_conv_true_future"  # the default adds nothing
    assert Harness("rolling", "vel1", "bev", "neutral").execute_s == 0.4 and Harness("single_shot", "px", "bev", "neutral").execute_s is None


@pytest.mark.parametrize("settings, message", [
    ({"check": 2}, "needs prediction=true_future"),
    ({"output": "px"}, "harness framings"),
    ({"framing": "participant"}, "harness framings"),
    ({"episode": "single_shot", "execute_s": 2.0}, "only for rolling episodes"),
    ({"execute_s": 0.5}, "whole number of 0.4 s steps"),
    ({"prediction": "future"}, "prediction='future' is not one of"),
    ({"check": -1}, "check must be a whole number"),
])
def test_invalid_harness_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        PRESETS["harness_v1"].harness("human", settings)


def test_velocity_outputs_are_refused_for_a_planned_person():
    """There are no person templates for the velocity outputs, so a person run would fail at its first query."""
    with pytest.raises(ValueError, match="no person templates"):
        Harness("rolling", "velseq", "bev", "participant")


def test_overrides_are_parsed_and_checked():
    assert parse_overrides(["prediction=true_future", "check=2", "execute_s=0.8"]) == {"prediction": "true_future", "check": 2, "execute_s": 0.8}
    for items, message in ((["checks=2"], "not a harness setting"), (["check"], "not KEY=VALUE"), (["check=two"], "whole number"),
                           (["execute_s=soon"], "number of seconds"), (["check=1", "check=2"], "given twice")):
        with pytest.raises(ValueError, match=message):
            parse_overrides(items)


def test_harness_file_is_written_once_and_a_changed_harness_is_refused(tmp_path):
    h = PRESETS["harness_v1"].harness("human")
    write_harness_file(tmp_path, "harness_v1", h, {})
    path = tmp_path / "rh_steer_bev_conv" / "harness" / HARNESS_FILE
    assert json.loads(path.read_text())["settings"] == h.settings()
    write_harness_file(tmp_path, "harness_v1", h, {})  # the same settings again: accepted
    stored = json.loads(path.read_text())
    stored["settings"]["execute_s"] = 0.8  # as if the default interval had changed in the code since the first run
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="holds other settings"):
        write_harness_file(tmp_path, "harness_v1", h, {})


def test_analysis_reads_harness_files_and_falls_back_to_the_registry(tmp_path):
    """A folder with harness.json is read from it (also an id the registry does not know), a folder without from the registry,
    and the old folder name rh_wp_bev_conv as rh_wp_bev_cam_conv. Other folders are ignored."""
    new = PRESETS["harness_v1"].harness("human", {"execute_s": 0.8})
    write_harness_file(tmp_path, "harness_v1", new, {"execute_s": 0.8})
    for folder in ("rh_px_bev_conv", "rh_wp_bev_conv", "replay", "smoke"):
        (tmp_path / folder).mkdir()
    found = run_experiments(tmp_path)
    assert set(found) == {new.id, "rh_px_bev_conv", "rh_wp_bev_conv", "replay"}
    assert found[new.id] == new.experiment() and found["rh_wp_bev_conv"].id == "rh_wp_bev_cam_conv" and found["replay"] == EXPERIMENTS["replay"]
