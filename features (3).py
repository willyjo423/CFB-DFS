"""Rolling features, for any sport, from the canonical frame.

Everything here is derived the same way in every sport: a player's own recent
volume, his share of his team's, what his team has been doing, what the
opposition has been allowing, and whether he has been playing at all. The
sport supplies the column names; this supplies the shifting.

The shift lives in ONE function. That is the entire safety argument: a leak
is not a crash, it is a model that grades beautifully and loses money, and
the only defence that scales is having exactly one place where a feature can
fail to be backward-looking.

Sport-specific features - a blowout margin, a park factor, a rest-days count
- are added by the sport's own builder afterwards and declared in the spec's
`extra_features`. This file never knows what they are.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import frame as FR

log = logging.getLogger(__name__)


def ewm(s: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted mean of everything STRICTLY BEFORE this row.

    The `shift(1)` is the whole point and it exists once, here, so that
    adding a feature cannot reintroduce the leak by forgetting it.
    """
    return s.shift(1).ewm(halflife=halflife, min_periods=1).mean()


def build(df: pd.DataFrame, spec, validate: bool = True) -> pd.DataFrame:
    """Canonical frame in, feature frame out. Same rows, more columns."""
    if validate:
        FR.validate(df, spec)
    out = df.copy()
    n_in = len(out)

    for c in spec.usage:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0)
    out["points"] = pd.to_numeric(out["points"], errors="coerce")

    touch_cols = spec.touch_columns or spec.usage
    out["touches"] = sum(out[c] for c in touch_cols if c in out.columns)

    # Shares within the team-period, computed BEFORE shifting, because a
    # share describes the game it happened in. The shift happens when the
    # share becomes a feature, below.
    # Shares are DERIVED into new columns, never written back over the raw
    # ones. Overwriting made `ewm_carries` the rolling mean of a ratio for
    # any column listed in both usage and shares - no error, just a feature
    # quietly measuring something else.
    share_cols = []
    base = [c for c in spec.shares if c in out.columns]
    if base:
        tot = out.groupby(["team", "season", "period"])[base].transform("sum")
        for c in base:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"share_{c}"] = np.where(tot[c] > 0, out[c] / tot[c], 0.0)
            share_cols.append(f"share_{c}")

    out = out.sort_values(["player_id", "season", "period"]).reset_index(
        drop=True)
    g = out.groupby("player_id", sort=False)
    hl = spec.halflife

    for c in list(spec.usage) + share_cols:
        if c in out.columns:
            out[f"ewm_{c}"] = g[c].transform(lambda s: ewm(s, hl))
    out["ewm_points"] = g["points"].transform(lambda s: ewm(s, hl))
    out["ewm_touches"] = g["touches"].transform(lambda s: ewm(s, hl))
    # Dispersion, because a boom-or-bust player and a metronome with the same
    # average are not the same asset in a tournament.
    out["sd_points"] = g["points"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std())
    out["games_played"] = g.cumcount()

    # Availability history. Scoring nothing and not taking part are
    # indistinguishable in most feeds, which is why this ends up as a
    # probability rather than a flag.
    out["_active"] = (out["points"].fillna(0) > 0).astype(float)
    a = out.groupby("player_id", sort=False)["_active"]
    out["played_last"] = a.transform(lambda s: s.shift(1))
    out["ewm_played"] = a.transform(lambda s: ewm(s, hl))
    out["played_rate"] = a.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    out = out.drop(columns=["_active"])

    out = _team_and_opponent(out, spec)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["share_of_team_touches"] = np.where(
            out["team_ewm_touches"] > 0,
            out["ewm_touches"] / out["team_ewm_touches"], np.nan)

    out["is_home"] = home_flag(out)

    if len(out) != n_in:
        raise ValueError(
            f"build changed the row count: {n_in} in, {len(out)} out. A "
            f"merge fanned out on duplicate keys; the features are not "
            f"trustworthy and the fit would be silently reweighted.")
    log.info("%s features: %d rows, %d players", spec.name, len(out),
             out["player_id"].nunique())
    return out


def _team_and_opponent(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Team form and opponent concession, both shifted, both from the rows.

    Aggregated from the players rather than from a team-level feed, because
    there is not always a team-level feed to trust, and because a defence's
    record must not include the game being predicted.

    Idempotent, like add_baselines and for the same reason. These three
    columns arrive by MERGE rather than by assignment, so calling build on a
    frame that already carries them produced `team_ewm_touches_x` / `_y` and
    a KeyError two lines later. That is exactly what stopped the leak proof
    from being able to re-derive features from a tampered frame.
    """
    df = df.drop(columns=[c for c in ("team_ewm_points", "team_ewm_touches",
                                      "opp_ewm_points_allowed")
                          if c in df.columns])
    team = (df.groupby(["team", "season", "period"], as_index=False)
            .agg(points=("points", "sum"), touches=("touches", "sum"))
            .sort_values(["team", "season", "period"]))
    g = team.groupby("team", sort=False)
    team["team_ewm_points"] = g["points"].transform(
        lambda s: ewm(s, spec.halflife))
    team["team_ewm_touches"] = g["touches"].transform(
        lambda s: ewm(s, spec.halflife))
    out = df.merge(team[["team", "season", "period", "team_ewm_points",
                         "team_ewm_touches"]],
                   on=["team", "season", "period"], how="left")

    allowed = (df.groupby(["opponent", "season", "period"], as_index=False)
               .agg(points_allowed=("points", "sum"))
               .rename(columns={"opponent": "team"})
               .sort_values(["team", "season", "period"]))
    allowed["opp_ewm_points_allowed"] = (
        allowed.groupby("team", sort=False)["points_allowed"]
        .transform(lambda s: ewm(s, spec.halflife)))
    out = out.merge(
        allowed[["team", "season", "period", "opp_ewm_points_allowed"]]
        .rename(columns={"team": "opponent"}),
        on=["opponent", "season", "period"], how="left")
    return out


def home_flag(df: pd.DataFrame) -> pd.Series:
    """Home or away: numbers read as numbers, text as text, else missing.

    The NFL version read this column as a STRING and tested membership in
    ("1", "true", "home"). When the join began supplying a numeric flag,
    float 1.0 stringified to "1.0", which is not "1", so every row failed and
    came out ZERO - not missing, zero. The feature became the constant claim
    that no team is ever at home, and the model learned from it, because a
    column of zeros looks exactly like real data.

    "We do not know" and "no" are different answers, and only one of them is
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


def trainable(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Rows with enough history behind them to be worth fitting on."""
    out = df[df["position"].isin(spec.positions)]
    out = out[out["games_played"] >= spec.min_prior_games]
    return out.dropna(subset=["points"]).reset_index(drop=True)
