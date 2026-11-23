"""The condition figure of the analysis: rows that pool experiments per trip, thinking off and on, the baselines as reference lines."""

import numpy as np
import pandas as pd
import pytest

from traj_gen.analysis.analyze import PAIRED, condition_figure, condition_figures, pooled_trip_means
from traj_gen.harness.experiments import BASELINES
from traj_gen.harness.presets import PRESETS


def table(values: dict[str, float]) -> pd.DataFrame:
    """A per-trip table as per_trip_means gives it: every PAIRED metric equal to the trip's value."""
    return pd.DataFrame({m: pd.Series(values) for m in PAIRED}).assign(site="um")


def by_cond_and_experiments():
    pilot, steer = (PRESETS[p].harness("human").experiment() for p in ("pilot_v2", "harness_v2"))
    by_cond = {(pilot.id, "participant", "nothink"): table({"a": 1.0, "b": 3.0}), (pilot.id, "participant", "think"): table({"a": 2.0, "b": 2.0}),
               (steer.id, "participant", "nothink"): table({"a": 3.0, "b": 7.0}),
               ("straight", "-", "-"): table({"a": 0.5, "b": 0.5}), ("replay", "-", "-"): table({"a": 1.5, "b": 1.5})}
    return by_cond, {pilot.id: pilot, steer.id: steer, **BASELINES}, pilot.id, steer.id


def test_pooled_rows_average_the_experiments_per_trip_and_skip_missing_conditions():
    """A pooled row is the mean over experiments of each trip's mean, so each trip weighs the same whatever its sample counts; a row missing
    one of its experiments for a thinking mode gets no dot rather than the mean of the others."""
    by_cond, _, pilot, steer = by_cond_and_experiments()
    pooled = pooled_trip_means(by_cond, [pilot, steer], "participant", "nothink", "path_efficiency")
    assert pooled.to_dict() == pytest.approx({"a": 2.0, "b": 5.0})
    assert pooled_trip_means(by_cond, [pilot, steer], "participant", "think", "path_efficiency") is None


def test_condition_figures_are_written_for_any_rows(tmp_path):
    by_cond, experiments, pilot, steer = by_cond_and_experiments()
    condition_figure(tmp_path / "panels.png", [("(a) one", [("pilot", [pilot], "participant")]), ("(b) pooled", [("both", [pilot, steer], "participant")])],
                     {"path_efficiency": "path efficiency", "goal_reached": "goal reached"}, by_cond, experiments, "test")
    condition_figures(by_cond, "human", tmp_path, experiments)
    assert {p.name for p in tmp_path.iterdir()} >= {"panels.png", "conditions_human.png"}
    assert np.all([p.stat().st_size > 0 for p in tmp_path.iterdir()])
