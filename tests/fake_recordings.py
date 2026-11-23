"""A fake recording with the robot and two people, and the person trips cut from it, shared by the trip tests."""

import numpy as np

from preprocessing import extract_trips
from preprocessing.bi3_derived import trip_arrays


def visit(k, station, label, t_arrive, t_turn, t_leave):
    return {"agent": "human0", "visit": k, "station": station, "label": label, "t_arrive": t_arrive, "t_turn": t_turn, "t_leave": t_leave}


def fake_recording(monkeypatch, visits) -> dict:
    """A 30 s recording at 10 Hz with the robot and two people walking in a line, and the given visits of human0: its frames as load_frames gives them."""
    time = np.arange(0, 30.0001, 0.1)
    walk = np.c_[np.linspace(-1, 1, len(time)), np.zeros(len(time))]
    frames = {agent: {"time": time, "x": walk[:, 0] + offset, "y": walk[:, 1] + offset, "theta": np.zeros(len(time)),
                      "state": np.full(len(time), "transit", dtype=object), "goal": np.full(len(time), "", dtype=object),
                      "next_goal": np.full(len(time), "", dtype=object)}
              for agent, offset in (("robot", 0.0), ("human0", 0.5), ("human1", -0.5))}
    monkeypatch.setattr(extract_trips, "load_visits", lambda *a: visits)
    return frames


def person_trips(monkeypatch, visits, human_xy, site="um") -> tuple[list[dict], dict]:
    """The person trips of a fake recording with the given visits, assembled as load_trips gives them, and the recording's series."""
    frames = fake_recording(monkeypatch, visits)
    grid, series = extract_trips.recording_series(site, frames)
    trips = extract_trips.human_trips(site, f"{site}99", "cv", frames, grid, series, human_xy, history_s=3.0)
    return [{**t, **trip_arrays(t, series)} for t in trips], series


OVERLAPPING_AND_SAME_STATION_VISITS = [
    visit(0, 0, "0", 1.0, 2.0, 3.0),
    visit(1, 2, "2", 6.0, 7.0, 8.0),
    visit(2, 3, "3", 7.5, 7.8, 9.0),    # arrives before visit 1 is left: no move from 1 to 2
    visit(3, 3, "3", 12.0, 12.5, 13.0),  # same station as visit 2: no move from 2 to 3
    visit(4, 5, "5", 16.0, 17.0, 18.0)]
