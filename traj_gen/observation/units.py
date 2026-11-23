"""The coordinate systems the prompts are written in: metres in the room frame, or pixels of the BEV image."""

import numpy as np


class Metres:
    """Metres in the room frame: x right, y up."""

    name = "m"
    decimals = 2  # centimetres

    def pos(self, xy) -> np.ndarray:
        return np.asarray(xy, float)

    def vec(self, v) -> np.ndarray:
        return np.asarray(v, float)

    def length(self, m: float) -> float:
        return float(m)

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

    def bounds(self, room_m: float) -> dict:
        """The room's extent along the two coordinates, for the frame description."""
        corners = self.pos([[-room_m / 2, -room_m / 2], [room_m / 2, room_m / 2]])
        return {"c0_min": self.num(corners[:, 0].min()), "c0_max": self.num(corners[:, 0].max()),
                "c1_min": self.num(corners[:, 1].min()), "c1_max": self.num(corners[:, 1].max())}


class Pixels(Metres):
    """Pixels of the BEV image (u right, v down, origin top-left). transform is an observation.bev.BevTransform."""

    name = "px"
    decimals = 0

    def __init__(self, transform):
        self.transform = transform

    def pos(self, xy) -> np.ndarray:
        return self.transform.to_pixel(xy)

    def vec(self, v) -> np.ndarray:
        return self.transform.to_pixel(v) - self.transform.to_pixel([0.0, 0.0])

    def length(self, m: float) -> float:
        return float(m) * self.transform.scale
