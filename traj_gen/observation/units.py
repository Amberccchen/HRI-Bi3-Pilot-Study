"""The coordinate systems the prompts are written in: metres in the room frame, or pixels of the BEV image.

Every number the model reads or writes goes through one of these: positions through the frame (pos, pos_to_metres), vectors such as
velocities through its axes (vec, vec_to_metres), and lengths and speeds through its scale (length, length_to_metres). Internal values
are always metres in the room frame.
"""

import numpy as np


class Metres:
    """Metres in the room frame: x right, y up."""

    name = "m"
    decimals = 2  # centimetres
    # Words for the prompts: the axes and where they point on the map, the coordinates as the position output names them, and the sense
    # of a positive turn, which goes from the first axis toward the second.
    axes = ("x", "y")
    directions = ("right", "up")
    coordinates = "in metres (x, y)"
    positive_turn, negative_turn = "anticlockwise", "clockwise"

    def pos(self, xy) -> np.ndarray:
        return np.asarray(xy, float)

    def vec(self, v) -> np.ndarray:
        return np.asarray(v, float)

    def length(self, m: float) -> float:
        return float(m)

    def pos_to_metres(self, p) -> np.ndarray:
        return np.asarray(p, float)

    def vec_to_metres(self, v) -> np.ndarray:
        return np.asarray(v, float)

    def length_to_metres(self, d):
        return d

    def num(self, v: float) -> str:
        return f"{v:.{self.decimals}f}"

    def pair(self, a: np.ndarray) -> str:
        return f"({self.num(a[0])}, {self.num(a[1])})"

    def pos_text(self, xy) -> str:
        return self.pair(self.pos(xy))

    def vec_text(self, v) -> str:
        return self.pair(self.vec(v))

    def len_text(self, m: float) -> str:
        return self.num(self.length(m))

    def heading_deg(self, heading_rad: float) -> float:
        """A heading in radians in the room frame, as degrees in these units' frame (0 along the first axis, 90 along the second), in (-180, 180]."""
        v = self.vec([np.cos(heading_rad), np.sin(heading_rad)])
        return float(np.degrees(np.arctan2(v[1], v[0])))

    def heading_to_metres(self, heading_deg: float) -> float:
        """A heading in degrees in these units' frame, as radians in the room frame."""
        v = self.vec_to_metres([np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg))])
        return float(np.arctan2(v[1], v[0]))

    def bounds(self, room_m: float) -> dict:
        """The room's extent along the two coordinates, for the frame description."""
        corners = self.pos([[-room_m / 2, -room_m / 2], [room_m / 2, room_m / 2]])
        return {"c0_min": self.num(corners[:, 0].min()), "c0_max": self.num(corners[:, 0].max()),
                "c1_min": self.num(corners[:, 1].min()), "c1_max": self.num(corners[:, 1].max())}


class Pixels(Metres):
    """Pixels of the BEV image (u right, v down, origin top-left). transform is an observation.bev.BevTransform."""

    name = "px"
    decimals = 0
    axes = ("u", "v")
    directions = ("right", "down")
    coordinates = "on the attached map, in pixels (u, v)"
    positive_turn, negative_turn = "clockwise", "anticlockwise"

    def __init__(self, transform):
        self.transform = transform

    def pos(self, xy) -> np.ndarray:
        return self.transform.to_pixel(xy)

    def vec(self, v) -> np.ndarray:
        return self.transform.to_pixel(v) - self.transform.to_pixel([0.0, 0.0])

    def length(self, m: float) -> float:
        return float(m) * self.transform.scale

    def pos_to_metres(self, p) -> np.ndarray:
        return self.transform.to_metre(p)

    def vec_to_metres(self, v) -> np.ndarray:
        return self.transform.to_metre(v) - self.transform.to_metre([0.0, 0.0])

    def length_to_metres(self, d):
        return d / self.transform.scale
