import numpy as np

from traj_gen.episodes import load_trajectory, save_trajectory


def test_round_trip_keeps_each_agent_under_its_own_id(tmp_path):
    """Three agents with distinct values: a column offset would hand one agent's positions to another."""
    rng = np.random.default_rng(0)
    times = np.round(np.arange(6) * 0.4, 4)
    positions = {agent: np.round(offset + rng.uniform(-1, 1, (6, 2)), 4) for agent, offset in [("robot", 0.0), ("human0", 10.0), ("human1", 20.0)]}
    save_trajectory(tmp_path, times, positions)
    loaded_times, loaded = load_trajectory(tmp_path)
    np.testing.assert_allclose(loaded_times, times, atol=1e-9)
    assert list(loaded) == ["robot", "human0", "human1"]
    for agent, xy in positions.items():
        np.testing.assert_allclose(loaded[agent], xy, atol=1e-9)
