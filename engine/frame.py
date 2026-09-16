"""The one thing every sport's data layer must produce.

This is the seam. Above it, every sport is different and always will be:
CFBD wants a bearer token and counts weeks, the MLB StatsAPI is open and
counts dates, nflverse publishes release assets. Below it, nothing in the
engine knows or is allowed to ask which sport it is running on.

The canonical frame is one row per player per game, carrying:

    player_id   a STABLE identifier, as text
    season      integer
    period      integer - a week in football, a day index in baseball
    team        the player's own side
    opponent    the other side
    is_home     1, 0, or missing - never guessed
    position    text
    points      the fantasy score under that sport's own rules
    <usage>     whatever columns the sport's spec declares

`period` rather than `week` is deliberate. Baseball has no weeks, and calling
a date a week would have every sport's code quietly lying about what it is
ordering by. Anything monotonic within a season works: a week number, a day
of year, an index of game dates.

`player_id` as TEXT is not a style preference. Athlete ids are digits that
are not numbers - read back as an integer, "0041" becomes 41 and matches
nothing. That has already cost this project once.

validate() is deliberately noisy. A missing column is easy; the expensive
failures are the ones that look fine - a points column that is entirely zero,
an is_home that is all False because a string test silently failed, a frame
that has quietly been duplicated by a bad merge. Those are checked here
because by the time a model has been fitted on them, nothing raises.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED = ["player_id", "season", "period", "team", "opponent",
            "position", "points"]
OPTIONAL = ["is_home", "name"]


class FrameError(ValueError):
    """The canonical contract was not met."""


def validate(df: pd.DataFrame, spec, strict: bool = True) -> list[str]:
    """Check the frame and return the complaints. Raise on any, if strict.

    Returns rather than only raising so a caller can report everything at
    once instead of fixing one problem per run.
    """
    problems = []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
        if strict:
            raise FrameError(problems[-1])
        return problems

    missing_usage = [c for c in spec.usage if c not in df.columns]
    if missing_usage:
        problems.append(
            f"{spec.name} declares usage columns that are not in the frame: "
            f"{missing_usage}")

    if df.empty:
        problems.append("the frame is empty")

    # A player id that is not text will not match a board. This one has
    # already happened.
    if not df.empty:
        sample = df["player_id"].iloc[0]
        if not isinstance(sample, str):
            problems.append(
                f"player_id is {type(sample).__name__}, not text. Numeric "
                f"ids lose leading zeros and stop matching anything.")

    # Duplicates mean a merge fanned out. The fit still runs; the training
    # set is simply reweighted by whatever the duplication depended on.
    dupes = int(df.duplicated(["player_id", "season", "period"]).sum())
    if dupes:
        problems.append(
            f"{dupes} duplicate player-period rows. A merge fanned out on "
            f"duplicate keys - nothing will raise, the model will just be "
            f"silently reweighted.")

    # The failures that look like data.
    pts = pd.to_numeric(df["points"], errors="coerce")
    if pts.notna().sum() == 0:
        problems.append("points is entirely missing")
    elif pts.nunique() <= 1:
        problems.append(
            f"points holds a single value ({pts.iloc[0]}). A constant column "
            f"looks exactly like real data and the model will learn from it.")

    if "is_home" in df.columns:
        home = pd.to_numeric(df["is_home"], errors="coerce")
        if home.notna().any() and home.nunique(dropna=True) <= 1:
            problems.append(
                "is_home never varies. In the NFL build this was a string "
                "comparison failing silently, so every row read as 'away' - "
                "not missing, FALSE, which the model duly learned.")

    known = set(spec.positions)
    have = set(df["position"].dropna().astype(str).unique())
    if known and not (have & known):
        problems.append(
            f"no row carries a position {spec.name} projects. Declared "
            f"{sorted(known)}, frame has {sorted(have)[:8]}")

    if not df.empty:
        per = pd.to_numeric(df["period"], errors="coerce")
        if per.isna().all():
            problems.append("period is not numeric; it must be orderable")

    if problems and strict:
        raise FrameError(f"{spec.name}: canonical frame is not valid:\n  - "
                         + "\n  - ".join(problems))
    for p in problems:
        log.warning("%s: %s", spec.name, p)
    return problems


def describe(df: pd.DataFrame, spec) -> str:
    """A short, honest summary. Printed by every sport's build."""
    pts = pd.to_numeric(df["points"], errors="coerce")
    played = (pts > 0).mean() if len(df) else float("nan")
    lines = [
        f"{spec.name}: {len(df):,} player-periods, "
        f"{df['player_id'].nunique():,} players",
        f"  seasons  : {sorted(int(s) for s in df['season'].unique())}",
        f"  points   : {pts.min():.1f} to {pts.max():.1f}, "
        f"mean {pts.mean():.2f}",
        f"  recorded : {played:.1%} of rows have a non-zero score",
    ]
    by_pos = (df[df["position"].isin(spec.positions)]
              .groupby("position")["points"].agg(["size", "mean"]))
    if not by_pos.empty:
        lines.append("  by position:")
        for pos, r in by_pos.iterrows():
            lines.append(f"    {pos:<5}{int(r['size']):>8,}  "
                         f"mean {r['mean']:.2f}")
    return "\n".join(lines)
