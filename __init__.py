"""The sport-agnostic half of a DFS model.

Nothing in this package may ask which sport it is running on. A sport
declares itself in a SportSpec and hands over a canonical frame; everything
here works the same way for all of them.

That rule is the point. A bug fixed here is fixed for every sport, and a
sport that needs the engine changed has found either a real gap in the spec
or a reason its data does not belong in the canonical frame yet.

    spec.py       what a sport declares about itself
    frame.py      the canonical frame every data layer must emit
    features.py   rolling features, shifted in exactly one place
    model.py      availability x conditional quantiles
    grade.py      walk-forward grading against baselines
    factors.py    a correlation matrix, PSD by construction
    simulate.py   correlated draws through a Gaussian copula
    ownership.py  what the field will roster
    optimise.py   lineups, duplication-adjusted
"""

from .spec import SportSpec
from . import frame, features

__all__ = ["SportSpec", "frame", "features"]
