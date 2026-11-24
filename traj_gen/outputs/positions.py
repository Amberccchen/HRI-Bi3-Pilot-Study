"""The positions output mode: points in the world frame, in the run's units, retimed to the speed limit."""

from traj_gen.outputs.planner import QueryPlanner, retime


class PositionPlanner(QueryPlanner):
    """Positions in the world frame (metres of the room frame or pixels of the map, by the units setting), one per step. The speed limit
    retimes the path."""

    name = "positions"
    template = "positions"
    key = "points"  # key of the JSON object the model replies with: {key: [[a, b], ...]}
    # Example numbers in metres (converted to the units in use), in an arbitrary place so they do not hint at where to go:
    # for the robot a short slow path, ~0.1 m per step (0.25 m/s), for a person ~0.4 m per step (1.0 m/s).
    example = {"robot": [(-0.50, 0.40), (-0.40, 0.45), (-0.30, 0.50)], "human": [(-0.50, 0.40), (-0.12, 0.52), (0.26, 0.64)]}

    def positions(self, value, scene):
        return retime(scene.track[-1], self.units.pos_to_metres(value), self.step_m)
