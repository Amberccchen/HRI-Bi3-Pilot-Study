"""The model's map: its size, the crop to the room plus the margin, and a pixel frame that is the image's own."""

import io

import numpy as np
from PIL import Image

from tests.fake_trips import SITE, person_trip
from traj_gen.observation.bev import EXTENT_M, SIZE_PX, bev_transform, render_bev
from traj_gen.observation.scene import Scene, track_motion


def rendered(pixel_ticks: bool) -> tuple[np.ndarray, Scene]:
    trip = person_trip()
    i = trip["start_index"]
    track = np.asarray(trip["tracks"]["human0"], float)[: i + 1, :2]
    scene = Scene(trip, SITE, i, track, track_motion(track))
    png, _ = render_bev(scene, pixel_ticks, False)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), int), scene


def test_map_is_448_px_and_its_frame_spans_the_extent():
    """The transform must map the drawn extent onto the whole image: the room's centre to the image centre, the extent's corners to its corners."""
    image, _ = rendered(True)
    assert image.shape == (SIZE_PX, SIZE_PX, 3) == (448, 448, 3)
    t = bev_transform()
    np.testing.assert_allclose(t.to_pixel([[0.0, 0.0], [-EXTENT_M / 2, EXTENT_M / 2], [EXTENT_M / 2, -EXTENT_M / 2]]), [[224, 224], [0, 0], [448, 448]])


def test_walls_and_the_planned_agent_are_drawn_where_the_transform_says():
    """A layout whose axes did not fill the image (margins for tick labels, an offset or another scale) would move the drawing away from the
    pixels the prompts give: the room's walls must be dark exactly at the transform's wall pixels, and the planned agent's cross at its position."""
    for pixel_ticks in (True, False):
        image, scene = rendered(pixel_ticks)
        t = bev_transform()
        lo, hi = t.to_pixel([-SITE["room_m"] / 2, SITE["room_m"] / 2])  # left wall u, top wall v
        left_u, top_v = int(round(lo)), int(round(hi))
        assert (left_u, top_v) == (28, 28)  # 0.25 m of margin at 112 px/m
        rows = slice(150, 300)
        assert image[rows, left_u].max(axis=1).mean() < 80  # the dark wall line
        assert image[rows, left_u + 6].min(axis=1).mean() > 150  # just inside the room: light
        assert image[top_v, rows].max(axis=1).mean() < 80
        u, v = np.round(t.to_pixel(scene.track[-1])).astype(int)
        assert image[v, u].max() < 80  # the black cross at the planned agent's centre
