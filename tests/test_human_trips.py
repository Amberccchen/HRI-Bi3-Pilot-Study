from tests.fake_recordings import OVERLAPPING_AND_SAME_STATION_VISITS, person_trips, visit


def test_person_trips_run_from_leaving_one_visit_to_turning_at_the_next(monkeypatch):
    """Off-by-one between visit k's leave and visit k + 1's turn, a kept overlapping or same-station pair, or the wrong
    next goal would all pass silently into the trips."""
    human_xy = {label: [float(label), 1.0] for label in ("0", "2", "3", "5")}
    trips, _ = person_trips(monkeypatch, OVERLAPPING_AND_SAME_STATION_VISITS, human_xy)
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
    human_xy = {label: [float(label), 1.0] for label in ("0", "2", "3")}
    trips, _ = person_trips(monkeypatch, visits, human_xy)
    assert [(t["goal"], t["next_goal"]) for t in trips] == [("2", "3"), ("3", "2"), ("2", "")]
