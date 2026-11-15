import numpy as np

from preprocessing import extract_trips


def visit(k, station, label, t_arrive, t_turn, t_leave):
    return {"agent": "human0", "visit": k, "station": station, "label": label, "t_arrive": t_arrive, "t_turn": t_turn, "t_leave": t_leave}


def fake_recording(monkeypatch, visits):
    """A 30 s recording at 10 Hz with the robot and two people, and the given visits of human0."""
    time = np.arange(0, 30.0001, 0.1)
    walk = np.c_[np.linspace(-1, 1, len(time)), np.zeros(len(time))]
    frames = {agent: {"time": time, "x": walk[:, 0] + offset, "y": walk[:, 1] + offset, "theta": np.zeros(len(time)),
                      "state": np.full(len(time), "transit", dtype=object), "goal": np.full(len(time), "", dtype=object),
                      "next_goal": np.full(len(time), "", dtype=object)}
              for agent, offset in (("robot", 0.0), ("human0", 0.5), ("human1", -0.5))}

    monkeypatch.setattr(extract_trips, "load_frames", lambda *a: frames)
    monkeypatch.setattr(extract_trips, "load_visits", lambda *a: visits)


OVERLAPPING_AND_SAME_STATION_VISITS = [
    visit(0, 0, "0", 1.0, 2.0, 3.0),
    visit(1, 2, "2", 6.0, 7.0, 8.0),
    visit(2, 3, "3", 7.5, 7.8, 9.0),    # arrives before visit 1 is left: no move from 1 to 2
    visit(3, 3, "3", 12.0, 12.5, 13.0),  # same station as visit 2: no move from 2 to 3
    visit(4, 5, "5", 16.0, 17.0, 18.0)]


def test_person_trips_run_from_leaving_one_visit_to_turning_at_the_next(monkeypatch):
    """Off-by-one between visit k's leave and visit k + 1's turn, a kept overlapping or same-station pair, or the wrong
    next goal would all pass silently into the trips."""
    fake_recording(monkeypatch, OVERLAPPING_AND_SAME_STATION_VISITS)
    human_xy = {label: [float(label), 1.0] for label in ("0", "2", "3", "5")}
    trips = extract_trips.human_trips("um", "um99", "cv", human_xy, hz=2.5, history_s=3.0)
    got = [(t["id"], t["start_goal"], t["goal"], t["next_goal"], t["t_start"], t["t_end"], t["goal_xy"]) for t in trips]
    assert got == [("um_um99_cv_human0_00", "0", "2", "3", 3.0, 7.0, [2.0, 1.0]),
                   ("um_um99_cv_human0_03", "3", "5", "", 13.0, 17.0, [5.0, 1.0])]
    for t in trips:
        assert t["agent"] == "human0" and t["time"][t["start_index"]] == 0
        assert abs(t["time"][t["end_index"]] - (t["t_end"] - t["t_start"])) < 0.4


def test_next_goal_skips_later_visits_at_the_goal_station(monkeypatch):
    """A repeated record of the goal station, or a loop back to it, would make the next goal equal the goal."""
    visits = [visit(0, 0, "0", 1.0, 2.0, 3.0), visit(1, 2, "2", 6.0, 7.0, 8.0), visit(2, 2, "2", 9.0, 9.5, 10.0),
              visit(3, 3, "3", 12.0, 12.5, 13.0), visit(4, 2, "2", 16.0, 17.0, 18.0)]
    fake_recording(monkeypatch, visits)
    human_xy = {label: [float(label), 1.0] for label in ("0", "2", "3")}
    trips = extract_trips.human_trips("um", "um99", "cv", human_xy, hz=2.5, history_s=3.0)
    assert [(t["goal"], t["next_goal"]) for t in trips] == [("2", "3"), ("3", "2"), ("2", "")]
