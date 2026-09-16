"""Baseball, declared. Two specs, because it is two sports.

Hitters and pitchers do not share a scoring system, a roster slot, a feature
set, or even the sign of their correlation with each other. Giving them one
spec with a position flag would ask a single fitted model to learn that a
strikeout is worth +2 to the man throwing it and roughly -0.3 to the man
taking it, from the same column. So: two specs, two fitted models, and the
engine runs each without knowing there is another.

What baseball needs that football did not
-----------------------------------------
**A longer memory.** Football's four-game half-life is tuned to a
sixteen-game season where a role change is the whole story. Over 162 games a
hitter's true talent moves slowly and the noise is enormous, so the memory is
much longer - about five weeks of games. A short half-life here would chase
variance and call it form.

**Opposite correlation signs.** A pitcher is strongly NEGATIVELY correlated
with every opposing hitter: his good day IS their bad one. Football has
nothing like that, which is why the loadings live with the sport.

**Batting order, which is not here yet.** It is the single biggest driver of
a hitter's day and is published about two hours before first pitch. That is a
live-projection problem rather than a historical one, and it is the most
valuable thing still missing from this model - stated here rather than
discovered as a weak grade later.
"""

from __future__ import annotations

import logging

import pandas as pd

from engine import SportSpec
from engine import features as EF

log = logging.getLogger(__name__)

# 162 games. A hitter's line moves slowly and its noise is enormous, so the
# past keeps mattering far longer than it does over a football season.
HALFLIFE = 30.0

HITTERS = SportSpec(
    name="MLB-hitters",
    usage=["plate_appearances", "at_bats", "single", "double", "triple",
           "home_run", "rbi", "run", "walk", "stolen_base"],
    shares=["plate_appearances"],
    touch_columns=["plate_appearances"],
    positions=["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"],
    min_prior_games=10,
    halflife=HALFLIFE,
    min_train_rows=2000,
    min_played_rows=1500,
    loadings={
        # A batting order is a queue: team-mates score in the same innings
        # off the same pitcher, so the team term is strong and the
        # competition term is near zero. Nine hitters do not take each
        # other's plate appearances the way two running backs take carries.
        "game":    {p: 0.18 for p in
                    ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]},
        "team":    {p: 0.42 for p in
                    ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]},
        "compete": {p: 0.05 for p in
                    ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]},
    },
)

PITCHERS = SportSpec(
    name="MLB-pitchers",
    usage=["innings", "strikeout", "batters_faced", "earned_run",
           "hit_allowed", "walk_allowed"],
    shares=[],
    touch_columns=["batters_faced"],
    positions=["SP", "RP", "P"],
    min_prior_games=5,          # a starter makes ~32 appearances a season
    halflife=8.0,               # far fewer appearances, so fewer to remember
    min_train_rows=1000,
    min_played_rows=800,
    loadings={
        # A pitcher shares little with his own side's hitters and a great
        # deal with the game itself. Two pitchers on one team in one game are
        # genuinely competing - innings one throws are innings the other does
        # not - which is the one place baseball looks like football.
        "game":    {"SP": 0.30, "RP": 0.22, "P": 0.26},
        "team":    {"SP": 0.12, "RP": 0.12, "P": 0.12},
        "compete": {"SP": 0.55, "RP": 0.40, "P": 0.45},
    },
)


def to_canonical(hist: pd.DataFrame) -> pd.DataFrame:
    """Box-score rows, renamed to the engine's vocabulary.

    `period` is a day index rather than a week, because baseball has no
    weeks. Anything monotonic within a season works, and calling a date a
    week would have every sport's code quietly lying about what it orders by.
    """
    out = hist.copy()
    missing = [c for c in ("player_id", "team", "points") if c not in out]
    if missing:
        raise ValueError(f"MLB history is missing {missing}")
    out["player_id"] = out["player_id"].astype(str)

    dates = pd.to_datetime(out["date"], errors="coerce")
    out["season"] = dates.dt.year.fillna(0).astype(int)
    # Day of year: monotonic within a season, gapless enough to order by, and
    # it never pretends a date is a week.
    out["period"] = dates.dt.dayofyear.fillna(0).astype(int)
    return out


def split(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two populations, separated the way the scoring already separates.

    `is_pitcher` is set by the data layer from whether a pitching line
    exists, which is the same test the scorer used. Deriving it twice from
    different rules is how the two halves drift apart.
    """
    df = to_canonical(hist)
    is_pit = pd.to_numeric(df.get("is_pitcher", 0),
                           errors="coerce").fillna(0) > 0
    pitchers = df[is_pit].copy()
    hitters = df[~is_pit].copy()
    log.info("split: %d hitter rows, %d pitcher rows",
             len(hitters), len(pitchers))
    return hitters, pitchers


def build(hist: pd.DataFrame, which: str) -> pd.DataFrame:
    """Features for one half of the sport."""
    hitters, pitchers = split(hist)
    if which == "hitters":
        return EF.build(hitters, HITTERS)
    if which == "pitchers":
        return EF.build(pitchers, PITCHERS)
    raise ValueError(f"which must be 'hitters' or 'pitchers', got {which!r}")


SPECS = {"hitters": HITTERS, "pitchers": PITCHERS}
