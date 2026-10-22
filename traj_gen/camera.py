"""Real overhead camera frame at the start of a trip, oriented to match the room frame (and the BEV).

The frame is cut from the Bi3 overhead video with ffmpeg at video_time = mocap_time + offset
(video_offset_s, from preprocessing/sync_video.py), turned by ORIENTATION so that room x is to the
right and y is up like the BEV, and downscaled to MAX_SIDE_PX. Not every trial has its video here:
the UM test trials (um16, um17) have none (so the experiments use um13), and asking for their frame fails.
"""

import io
import subprocess
from pathlib import Path

from PIL import Image


VIDEOS = Path("/persistent/data/Bi3/videos")
MAX_SIDE_PX = 960   # keeps a 1920x1080 frame at about 1000 vision tokens, and the LAAS frames are already 960x720
# Per site: (counter-clockwise quarter turns, then mirror left-right?) that make the raw frame match the room frame.
# Verified by eye against the BEV at the same times (phase2 smoke script, orientation figures): at UM the raw
# frame already has the shared table on the right and S3 on top; at LAAS the raw camera looks along the room's
# x axis (raw right = room +y, raw down = room +x), so a quarter turn counter-clockwise puts x to the right and y up.
# Checked on robot and both people over a trip (their raw image positions move as the room-frame tracks say), no mirror needed.
ORIENTATION = {"laas": (1, False), "um": (0, False)}


def camera_frame(trip: dict) -> bytes:
    """PNG bytes of the overhead frame at the trip start, in the room orientation."""
    path = VIDEOS / trip["site"] / trip["trial"] / f"{trip['controller']}.mp4"
    t = trip["t_start"] + trip["video_offset_s"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
                         capture_output=True, check=True).stdout
    quarter_turns, mirror = ORIENTATION[trip["site"]]
    image = Image.open(io.BytesIO(raw)).convert("RGB").rotate(90 * quarter_turns, expand=True)
    if mirror:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
