import json
import sys
from pathlib import Path

import pytest

import traj_gen.run as run_module
from tests.test_harness import robot_trip
from tests.test_human_prompt import SITE, person_trip
from traj_gen.episodes import Episode
from traj_gen.harnesses import HARNESS_FILE

COMMIT = "0" * 40


@pytest.fixture
def no_episodes(monkeypatch):
    """run() up to the episode list, with fake trips and code provenance, and every episode counted as done, so no model is called."""
    trips = {t["id"]: t for t in (person_trip(), robot_trip())}

    def load_run(run: Path, trip_list: Path, config: dict | None = None):
        chosen = json.loads(trip_list.read_text())
        return config or json.loads((run / "run.json").read_text()), chosen, trips, {"um": SITE}

    monkeypatch.setattr(run_module, "load_run", load_run)
    monkeypatch.setattr(run_module, "provenance", lambda: {"git_commit": COMMIT, "git_dirty": False})
    monkeypatch.setattr(Episode, "done", lambda self: True)
    monkeypatch.setattr(sys, "argv", list(sys.argv))  # call() sets it


def call(tmp_path: Path, *argv: str) -> None:
    sys.argv = ["run.py", "--run", str(tmp_path / "run"), "run", *argv]
    run_module.main()


def trip_list(tmp_path: Path, trip: dict) -> str:
    path = tmp_path / "trips.json"
    path.write_text(json.dumps([{"id": trip["id"], "stratum": "close"}]))
    return str(path)


def test_new_run_folder_gets_run_json_trip_list_and_harness_files(tmp_path, no_episodes, capsys):
    trips = trip_list(tmp_path, person_trip())
    call(tmp_path, "--harness", "harness_v1", "--overrides", "prediction=true_future", "--trips", trips, "--samples", "2")
    run = tmp_path / "run"
    config = json.loads((run / "run.json").read_text())
    assert config["trips_dir"] == "/persistent/data/Bi3_derived/trips_v3_human" and config["git_commit"] == COMMIT and config["backend"] == "qwen"
    assert config["commands"][0]["harnesses"] == ["harness_v1"]
    assert config["commands"][0]["overrides"] == {"prediction": "true_future"} and "--overrides prediction=true_future" in config["commands"][0]["command"]
    assert json.loads((run / "test_trips.json").read_text()) == json.loads(Path(trips).read_text())
    assert [p.relative_to(run).as_posix() for p in run.glob(f"*/*/{HARNESS_FILE}")] == [f"rh_steer_bev_conv_true_future/harness/{HARNESS_FILE}"]
    assert "0 episodes to run, 4 already done" in capsys.readouterr().out  # 2 samples of one harness, plus the two baselines


def test_robot_flag_must_match_the_framings_and_the_trips(tmp_path, no_episodes):
    with pytest.raises(ValueError, match="pass --robot for a robot framing"):
        call(tmp_path, "--overrides", "framing=neutral")
    with pytest.raises(ValueError, match="drop --robot for a person framing"):
        call(tmp_path, "--robot", "--overrides", "framing=participant")
    with pytest.raises(ValueError, match="has trips that plan a robot, but this run plans a human"):
        call(tmp_path, "--trips", trip_list(tmp_path, robot_trip()))
    assert not (tmp_path / "run").exists()  # nothing is written before the checks pass


def test_robot_takes_the_default_presets_robot_framing_and_no_baselines(tmp_path, no_episodes, capsys):
    call(tmp_path, "--robot", "--baselines", "none", "--trips", trip_list(tmp_path, robot_trip()))
    assert (tmp_path / "run" / "rh_px_bev_conv" / "neutral" / HARNESS_FILE).exists()
    assert "0 episodes to run, 3 already done" in capsys.readouterr().out
    assert json.loads((tmp_path / "run" / "run.json").read_text())["trips_dir"] == "/persistent/data/Bi3_derived/trips_v3"


def test_a_changed_harness_in_the_same_folder_is_refused(tmp_path, no_episodes):
    trips = trip_list(tmp_path, person_trip())
    call(tmp_path, "--harness", "harness_v1", "--trips", trips)
    path = tmp_path / "run" / "rh_steer_bev_conv" / "harness" / HARNESS_FILE
    stored = json.loads(path.read_text())
    stored["settings"]["images"] = "bev_cam"
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="holds other settings than harness harness_v1"):
        call(tmp_path, "--harness", "harness_v1", "--trips", trips)


def test_bad_overrides_are_refused_with_the_harness_name(tmp_path, no_episodes):
    with pytest.raises(ValueError, match="harness pilot with overrides {'check': 2}: check=2 needs prediction=true_future"):
        call(tmp_path, "--overrides", "check=2")
