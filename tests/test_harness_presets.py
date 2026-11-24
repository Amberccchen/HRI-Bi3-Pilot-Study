import json

import pytest

from traj_gen.harness.experiments import BASELINES
from traj_gen.harness.presets import HARNESS_FILE, PRESETS, Harness, parse_overrides, run_experiments, write_harness_file


def test_preset_ids_and_framings_per_kind():
    expected = {("pilot_v2", "human"): ("rh_positions_px_bev_conv", "participant"), ("pilot_v2", "robot"): ("rh_positions_px_bev_conv", "neutral"),
                ("harness_v2", "human"): ("rh_steer_px_bev_conv", "scored_distances"), ("harness_v2", "robot"): ("rh_steer_px_bev_conv", "robot_scored_distances")}
    assert {(name, kind): (p.harness(kind).id, p.harness(kind).framing) for name, p in PRESETS.items() for kind in ("human", "robot")} == expected


def test_overrides_give_the_harness_levels():
    """harness_v2 plus prediction=true_future, and plus check=2, must give the recorded futures and the plan check, vel1 its own default
    interval, and a framing override must replace the kind's framing."""
    for kind in ("human", "robot"):
        future = PRESETS["harness_v2"].harness(kind, {"prediction": "true_future"}).experiment()
        check = PRESETS["harness_v2"].harness(kind, {"prediction": "true_future", "check": 2}).experiment()
        assert (future.id, future.true_future, future.check_rounds) == ("rh_steer_px_bev_conv_true_future", True, 0)
        assert (check.id, check.true_future, check.check_rounds) == ("rh_steer_px_bev_conv_true_future_check", True, 2)
    single = PRESETS["pilot_v2"].harness("robot", {"episode": "single_shot"}).experiment()
    assert (single.id, single.execute_s) == ("ss_positions_px_bev", None)
    vel1 = PRESETS["pilot_v2"].harness("robot", {"output": "vel1", "episode": "rolling", "units": "m"}).experiment()
    assert (vel1.id, vel1.execute_s) == ("rh_vel1_m_bev", 0.4)
    assert PRESETS["pilot_v2"].harness("human", {"framing": "scored_distances", "output": "steer"}).framing == "scored_distances"


def test_id_suffixes_for_check_count_and_replanning_interval():
    def h(**overrides):
        return PRESETS["harness_v2"].harness("human", {"prediction": "true_future", **overrides})
    assert h(check=3).id == "rh_steer_px_bev_conv_true_future_check3"
    assert h(execute_s=1.2).id == "rh_steer_px_bev_conv_true_future_replan1.2s"
    assert h(execute_s=0.8).id == "rh_steer_px_bev_conv_true_future"  # the default adds nothing
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
    with pytest.raises(ValueError, match="templates for a planned robot only"):
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
    stored["settings"]["execute_s"] = 1.2  # as if the default interval had changed in the code since the first run
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="holds other settings"):
        write_harness_file(tmp_path, "harness_v2", h, {})


def test_analysis_reads_harness_files_and_the_baseline_folders(tmp_path):
    """A folder with harness.json is read from it, a baseline folder as its baseline. Other folders, without harness.json, are ignored."""
    new = PRESETS["harness_v2"].harness("human", {"execute_s": 1.2})
    write_harness_file(tmp_path, "harness_v2", new, {"execute_s": 1.2})
    for folder in ("rh_px_bev_conv", "replay", "smoke"):
        (tmp_path / folder).mkdir()
    found = run_experiments(tmp_path)
    assert set(found) == {new.id, "replay"}
    assert found[new.id] == new.experiment() and found["replay"] == BASELINES["replay"]
