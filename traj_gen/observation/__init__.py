"""What the model is told: the scene and its text, the coordinate units, the maps, the camera frame and the prompt assembly.

The prompt text files are under traj_gen/prompts/ (str.format templates), read with load_prompt."""

from functools import cache
from pathlib import Path


PROMPTS = Path(__file__).parent.parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text().strip()
