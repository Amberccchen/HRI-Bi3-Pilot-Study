"""What the model is told: the scene and its text, the coordinate units, the maps, the camera frame and the prompt assembly.

The prompt text files are str.format templates under traj_gen/prompts/, laid out as <part or harness setting>/[<kind>/]<option>.txt: a
human/ or robot/ folder (the planned agent's kind) only where the text differs by kind, a part with one text per kind and no options as
<part>/<kind>.txt, and a part with a single text as <part>.txt. load_prompt maps a part, option and kind to its file.
"""

from functools import cache
from pathlib import Path


PROMPTS = Path(__file__).parent.parent / "prompts"
# The words for the planned agent in the templates that differ by kind only in who is speaking (followup/, output/px.txt, output/wp.txt,
# state/other_person.txt): one template each, filled with the planned agent's kind's words.
PLANNED_AGENT_WORDS = {
    "robot": {"planned": "the robot", "planned_start": "The robot", "planned_pronoun": "it", "planned_possessive": "the robot's",
              "planned_position": "robot position", "planned_goal": "the goal", "other_person_note": ""},
    "human": {"planned": "you", "planned_start": "You", "planned_pronoun": "you", "planned_possessive": "your",
              "planned_position": "position of yours", "planned_goal": "your goal", "other_person_note": " (the other participant)"},
}


@cache
def load_prompt(part: str, option: str | None = None, kind: str | None = None) -> str:
    """The template of a part (a prompt part or harness setting) for an option and the planned agent's kind: <part>/<kind>/<option>.txt
    where the text differs by kind, else <part>/<option>.txt (without an option: <part>/<kind>.txt, else <part>.txt). Raises
    FileNotFoundError if neither file exists, and ValueError if both do, since the layout keeps one of them."""
    if option is None:
        by_kind, shared = PROMPTS / part / f"{kind}.txt", PROMPTS / f"{part}.txt"
    else:
        by_kind, shared = PROMPTS / part / str(kind) / f"{option}.txt", PROMPTS / part / f"{option}.txt"
    paths = [shared] if kind is None else [by_kind, shared]
    found = [p for p in paths if p.is_file()]
    if len(found) > 1:
        raise ValueError(f"both {found[0]} and {found[1]} exist: a text sits either in a kind folder or directly under its part")
    if not found:
        raise FileNotFoundError(f"no prompt file at {' or '.join(map(str, paths))}")
    return found[0].read_text().strip()
