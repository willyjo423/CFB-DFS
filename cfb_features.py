"""What a college player's next Saturday is built from.

Modelled on the NFL feature set and deliberately not a copy of it, because
the two sports do not publish the same things and pretending otherwise would
produce columns of zeros that look exactly like data.

What is missing here and matters
--------------------------------
**Targets.** CFBD publishes receptions, not targets. In the NFL model target
share is the single most informative feature in the set - it separates a
receiver who is the offence from one who happens to play in a good one, and
it does so BEFORE the catches arrive. Receptions are the same signal after
the fact, contaminated by catch rate and by the quarterback. There is no
substitute available, so this set is weaker at receiver than the NFL one and
that should be expected rather than explained away later.

**Air yards and WOPR.** Both derive from targets. Gone for the same reason.

**Snap counts.** Not published. Availability therefore has to be inferred
from whether a man recorded a stat, which is what `played` does.

What is here and matters more than it does in the NFL
-----------------------------------------------------
**Blowout exposure.** College games are lopsided in a way professional ones
are not, and starters sit in the fourth quarter of a forty-point win. That is
a real, frequent, learnable pattern and it is most of what replaces the
injury report that college football does not have. `team_ewm_margin` and
`opp_ewm_margin` carry it.

**Opponent quality.** The gap between the best and worst team on a slate is
enormous compared with the NFL. What a defence has allowed is a far stronger
signal here, and it is built from the same player rows rather than assumed.

Every feature is shifted before use. A feature that includes the row it
predicts is the leak that makes a model look brilliant in backtest and lose
money on Saturday, and the shift is applied in one place - `_ewm` - so there
is exactly one thing to get right.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Raw volume, as CFBD actually publishes it.
USAGE = ["carries", "rec", "rush_yards", "rec_yards", "pass_yards",
         "completions"]

# Shares of the team's own work. These are what separate a player from his
# offence, and with targets unavailable they carry more weight than usual.
SHARES = ["share_carries", "share_rec", "share_rec_yards"]

FEATURES = (
    [f"ewm_{c}" for c in USAGE]
    + [f"ewm_{c}" for c in SHARES]
    + ["ewm_points", "sd_points", "games_played", "ewm_touches",
       "team_ewm_pass_yards", "team_ewm_rush_yards", "team_ewm_points",
       "team_ewm_margin",
       "opp_ewm_points_allowed", "opp_ewm_pass_yards_allowed",
       "opp_ewm_rush_yards_allowed", "opp_ewm_margin",
       "share_of_team_touches",
       "is_home",
       # Availability history. Leak-free, and in the absence of any college
       # injury report it is most of what the play/no-play model has.
       "played_last", "ewm_played", "played_rate"]
)

HALFLIFE = 4.0
MIN_PRIOR_GAMES = 3
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def _ewm(s: pd.Series, halflife: float = HALFLIFE) -> pd.Series:
    """Exponentially weighted mean of everything STRICTLY BEFORE this row.

    The shift is the whole point and it lives here, once, so that adding a
    feature cannot reintroduce the leak. A player's own result must never be
    among the inputs used to predict it.
    """
    return s.shift(1).ewm(halflife=halflife, min_periods=1).mean()


def to_model_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """CFBD's column names, renamed to the ones the engine expects.

    Kept as an explicit, reversible step rather than renaming inside the data
    layer, so that anything reading cfb_data still sees CFBD's own vocabulary
    and only the modelling side sees the engine's.
    """
    out = hist.rename(columns={"athlete_id": "player_id", "school": "team"})
    missing = [c for c in ("player_id", "season", "week", "team", "points")
               if c not in out.columns]
    if missing:
        raise ValueError(f"history is missing {missing}; cannot build "
                         f"features from it")
    return out


def _margins(df: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's scoring margin, from the player rows themselves.

    Fantasy points are not the scoreboard, but a team's fantasy production
    against its opponent's is a serviceable proxy for how lopsided the game
    was, and it needs no extra endpoint. This exists to give the model a
    handle on the thing that ends college starters' afternoons early.
    """
    team_pts = (df.groupby(["team", "season", "week"], as_index=False)
                ["points"].sum().rename(columns={"points": "_team_points"}))
    opp_pts = team_pts.rename(columns={"team": "opponent",
                                       "_team_points": "_opp_points"})
    m = team_pts.merge(opp_pts, on=["season", "week"], how="inner")
    m = m[m["team"] != m["opponent"]]
    m["margin"] = m["_team_points"] - m["_opp_points"]
    return m[["team", "opponent", "season", "week", "margin"]]


def _team_context(df: pd.DataFrame) -> pd.DataFrame:
    """The team's recent offence, what its opponents have managed, and margin.

    Built by summing the players, because there is no team-level row to
    trust, and shifted exactly the way the player features are - a defence's
    record must not include the game being predicted.
    """
    team = (df.groupby(["team", "season", "week"], as_index=False)
            .agg(pass_yards=("pass_yards", "sum"),
                 rush_yards=("rush_yards", "sum"),
                 points=("points", "sum"),
                 touches=("touches", "sum"))
            .sort_values(["team", "season", "week"]))

    margins = _margins(df)
    team = team.merge(margins[["team", "season", "week", "margin"]],
                      on=["team", "season", "week"], how="left")

    g = team.groupby("team", sort=False)
    for src, dest in (("pass_yards", "team_ewm_pass_yards"),
                      ("rush_yards", "team_ewm_rush_yards"),
                      ("points", "team_ewm_points"),
                      ("margin", "team_ewm_margin"),
                      ("touches", "team_ewm_touches")):
        team[dest] = g[src].transform(_ewm)
    return team


def _opponent_context(df: pd.DataFrame) -> pd.DataFrame:
    """What each team has been giving up, indexed so it can join as opponent."""
    allowed = (df.groupby(["opponent", "season", "week"], as_index=False)
               .agg(points_allowed=("points", "sum"),
                    pass_allowed=("pass_yards", "sum"),
                    rush_allowed=("rush_yards", "sum"))
               .rename(columns={"opponent": "team"})
               .sort_values(["team", "season", "week"]))
    g = allowed.groupby("team", sort=False)
    for src, dest in (("points_allowed", "opp_ewm_points_allowed"),
                      ("pass_allowed", "opp_ewm_pass_yards_allowed"),
                      ("rush_allowed", "opp_ewm_rush_yards_allowed")):
        allowed[dest] = g[src].transform(_ewm)
    return allowed


def build(hist: pd.DataFrame) -> pd.DataFrame:
    """One row per player-week, with the target and every feature."""
    df = to_model_frame(hist).copy()
    for c in USAGE:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["touches"] = df["carries"] + df["rec"]

    # Shares within the team-game, computed BEFORE any shifting, because a
    # share describes the game it happened in. The shifting happens when the
    # share is turned into a feature, below.
    tot = (df.groupby(["team", "season", "week"])
           [["carries", "rec", "rec_yards"]].transform("sum"))
    for c in ("carries", "rec", "rec_yards"):
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"share_{c}"] = np.where(tot[c] > 0, df[c] / tot[c], 0.0)

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby("player_id", sort=False)
    for c in USAGE + SHARES:
        df[f"ewm_{c}"] = g[c].transform(_ewm)
    df["ewm_points"] = g["points"].transform(_ewm)
    df["ewm_touches"] = g["touches"].transform(_ewm)
    # Dispersion, because a boom-or-bust receiver and a metronome with the
    # same average are not the same asset in a tournament.
    df["sd_points"] = g["points"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std())
    df["games_played"] = g.cumcount()

    # Availability history. A man who did not record a stat did not play, or
    # played and did nothing - the model cannot tell those apart and neither
    # can DraftKings, which is why this is a probability and not a flag.
    df["_active"] = (df["points"].fillna(0) > 0).astype(float)
    a = df.groupby("player_id", sort=False)["_active"]
    df["played_last"] = a.transform(lambda s: s.shift(1))
    df["ewm_played"] = a.transform(_ewm)
    df["played_rate"] = a.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.drop(columns=["_active"])

    team = _team_context(df)
    df = df.merge(
        team[["team", "season", "week", "team_ewm_pass_yards",
              "team_ewm_rush_yards", "team_ewm_points", "team_ewm_margin",
              "team_ewm_touches"]],
        on=["team", "season", "week"], how="left")

    if "opponent" in df.columns:
        allowed = _opponent_context(df)
        df = df.merge(
            allowed[["team", "season", "week", "opp_ewm_points_allowed",
                     "opp_ewm_pass_yards_allowed",
                     "opp_ewm_rush_yards_allowed"]]
            .rename(columns={"team": "opponent"}),
            on=["opponent", "season", "week"], how="left")
        # The opponent's own margin history: is this a team that gets blown
        # out, which is when the other side's starters leave early.
        opp_margin = (team[["team", "season", "week", "team_ewm_margin"]]
                      .rename(columns={"team": "opponent",
                                       "team_ewm_margin": "opp_ewm_margin"}))
        df = df.merge(opp_margin, on=["opponent", "season", "week"],
                      how="left")
    for c in ("opp_ewm_points_allowed", "opp_ewm_pass_yards_allowed",
              "opp_ewm_rush_yards_allowed", "opp_ewm_margin"):
        if c not in df.columns:
            df[c] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        df["share_of_team_touches"] = np.where(
            df["team_ewm_touches"] > 0,
            df["ewm_touches"] / df["team_ewm_touches"], np.nan)

    df["is_home"] = _home_flag(df)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"build() did not produce {missing}")
    log.info("features: %d rows, %d players, %d columns, seasons %s",
             len(df), df["player_id"].nunique(), len(FEATURES),
             sorted(df["season"].unique()))
    return df


def _home_flag(df: pd.DataFrame) -> pd.Series:
    """Home or away, read as a number if it is one and as text if it is not.

    The NFL version of this read the column as a STRING and tested membership
    in ("1", "true", "home"). When the join began supplying a numeric flag,
    float 1.0 stringified to "1.0", which is not "1", so every row failed and
    came out ZERO - not missing, zero. The feature became the constant claim
    that no team is ever at home, and the model learned from it, because a
    column of zeros looks exactly like real data and nothing raises.

    "We do not know" and "no" are different answers and only one of them is
    safe to guess at.
    """
    for c in ("is_home", "home_away", "location"):
        if c not in df.columns:
            continue
        col = df[c]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().any():
            return num.where(num.isna(), (num > 0).astype(float)).astype(float)
        s = col.astype(str).str.strip().str.lower()
        home = s.isin(("true", "home", "h"))
        away = s.isin(("false", "away", "a", "neutral"))
        if (home | away).any():
            return pd.Series(np.where(home, 1.0, np.where(away, 0.0, np.nan)),
                             index=df.index, dtype=float)
    return pd.Series(np.nan, index=df.index)


def trainable(df: pd.DataFrame, positions: list[str] | None = None
              ) -> pd.DataFrame:
    """Rows with enough history behind them to be worth fitting on.

    A player's first games carry no usable usage signal, and including them
    teaches the model that the features are noise. They are excluded from
    training and still PROJECTED at prediction time, from position and team
    context alone, which is the honest answer for a true freshman's debut -
    and college football has a great many of those.
    """
    positions = positions or SKILL_POSITIONS
    out = df[df["position"].isin(positions)]
    out = out[out["games_played"] >= MIN_PRIOR_GAMES]
    return out.dropna(subset=["points"]).reset_index(drop=True)
