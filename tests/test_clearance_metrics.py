import numpy as np

from metrics.traj_gen_metrics import clearance_metrics


def reference(agent, others, agent_radius, other_radii, personal_space_radii):
    """The same quantities by a plain loop over time steps and other agents."""
    gaps, touching, intruding = [], [], []
    for t in range(len(agent)):
        step_gaps, inside = [], False
        for k in range(others.shape[1]):
            centre = float(np.hypot(*(others[t, k] - agent[t])))
            step_gaps.append(centre - agent_radius - other_radii[k])
            inside = inside or centre - agent_radius < personal_space_radii[k]
        gaps.append(min(step_gaps))
        touching.append(min(step_gaps) < 0)
        intruding.append(inside)
    events = sum(1 for t in range(len(touching)) if touching[t] and (t == 0 or not touching[t - 1]))
    return {"clearance_min": min(gaps), "contact_time": sum(touching) / len(touching), "contact_events": events,
            "personal_space_overlap": sum(intruding) / len(intruding)}


def test_clearance_metrics_match_a_loop_with_unequal_radii():
    """Unequal radii per other agent: subtracting one shared radius, or the radius of the wrong agent, changes the result."""
    agent = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0], [1.2, 0.0], [1.5, 0.0], [1.8, 0.0]])
    big = np.tile([0.9, 0.55], (len(agent), 1))   # passes near the agent's middle steps
    small = np.array([[1.6, 0.3], [1.6, 0.3], [1.6, 0.3], [1.6, 0.3], [1.6, 0.25], [1.6, 0.25], [1.6, 0.25]])
    others = np.stack([big, small], axis=1)
    agent_radius, other_radii, personal_space_radii = 0.2, [0.4, 0.1], [0.5, 0.15]
    got = clearance_metrics(agent, others, agent_radius, other_radii, personal_space_radii)
    want = reference(agent, others, agent_radius, other_radii, personal_space_radii)
    assert got["contact_events"] == want["contact_events"]
    for key in ("clearance_min", "contact_time", "personal_space_overlap"):
        np.testing.assert_allclose(got[key], want[key], atol=1e-12)
    assert 0 < want["contact_time"] < 1 and want["contact_events"] == 2  # the case exercises contact that starts, stops and starts again
