"""What the episodes and planners run: an Experiment (an output mode with its units, the images, an episode type and the levels), and the
baselines without a model.

A model experiment is built from harness settings (harness.presets.Harness.experiment), whose id it carries. The baselines (BASELINES:
straight and replay) are fixed experiments, chosen by name on the command line, each in its own run folder level <run>/<id>/.
"""

from dataclasses import dataclass

from traj_gen.harness.episodes import BaselineEpisode
from traj_gen.outputs.baselines import ReplayPlanner, StraightPlanner
from traj_gen.outputs.planner import QueryPlanner


@dataclass(frozen=True)
class Experiment:
    id: str
    planner: type
    units: str | None  # the units setting (outputs.planner.UNITS), None for the baselines
    images: tuple[str, ...]
    episode: type
    execute_s: float | None  # executed motion between queries (rolling only), checked by the harness settings
    base: str | None = None  # id of the experiment without the prediction (_constvel_future rows only)
    true_future: bool = False  # give the other agents' recorded future positions in the text each turn
    check_rounds: int = 0  # check each plan against the recorded futures and report the first conflict back, up to this many revisions

    @property
    def predict(self) -> bool:
        """Also give each person's constant-velocity position (text and map): the constvel_future prediction, deprecated."""
        return self.base is not None

    @property
    def uses_model(self) -> bool:
        return issubclass(self.planner, QueryPlanner)


BASELINES = {e.id: e for e in (Experiment("straight", StraightPlanner, None, (), BaselineEpisode, None),
                               Experiment("replay", ReplayPlanner, None, (), BaselineEpisode, None))}
