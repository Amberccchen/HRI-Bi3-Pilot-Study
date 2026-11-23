"""The position output modes: waypoints in metres and points on the BEV in pixels, both retimed to the speed limit."""

from traj_gen.outputs.planner import PIXELS, QueryPlanner, retime


class WaypointPlanner(QueryPlanner):
    """Positions in metres. The speed limit retimes the path."""

    name = "waypoints"
    template = "output_waypoints"
    key = "waypoints"  # key of the JSON object the model replies with: {key: [[a, b], ...]}
    # Example numbers in metres (converted to the units in use), in an arbitrary place so they do not hint at where to go:
    # for the robot a short slow path, ~0.1 m per step (0.25 m/s), for a person ~0.4 m per step (1.0 m/s).
    example = {"robot": [(-0.50, 0.40), (-0.40, 0.45), (-0.30, 0.50)], "human": [(-0.50, 0.40), (-0.12, 0.52), (0.26, 0.64)]}

    def positions(self, value, scene):
        return retime(scene.track[-1], value, self.step_m)


class PixelPlanner(WaypointPlanner):
    """Points on the BEV in pixels: mapped to metres, then as waypoints."""

    name = "pixels"
    template = "output_pixels"
    key = "points"
    units = PIXELS
    pixel_ticks = True

    def positions(self, value, scene):
        return super().positions(self.units.transform.to_metre(value), scene)
