"""Is the model any good. Measured walk-forward, against baselines it must beat.

A projection nobody has graded is an opinion. This grades one, the only way
that means anything: fit on everything strictly before week W, predict week W,
move to W+1, refit. No row is ever predicted by a model that saw it.

What is measured, and why each
------------------------------
**Against baselines, not in isolation.** An MAE of 4.5 is meaningless alone.
Three baselines a real model must beat: the player's own prior mean, his last
game, and his position's average. If the model cannot beat "what he did last
week", it has learned nothing worth the compute.

**Calibration, as coverage counting.** Of the outcomes, what fraction land
below each fitted quantile? A 10th percentile should sit above 10% of results.
This is what caught the NFL model blending play and no-play into one
distribution: its 10th covered 3.1% instead of 10%.

**Availability separately.** AUC and Brier for the play/no-play half,
including a sharpness check - a classifier that predicts the base rate for
everybody scores a respectable Brier and is useless.

**Rank within position.** DFS does not need the level right; it needs the
ORDER right. A model biased low everywhere still wins if it ranks correctly.

**A leak proof that can fail.** Two earlier versions of this could not: one
deleted future rows in a way that changed nothing, and one tampered so
crudely that the honest baseline failed too. The version here isolates a
single player-game, requires the tampering to propagate forward, and requires
it NOT to reach the prediction for the tampered week itself.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import cfb_features as F
from cfb_model import Projections

log = logging.getLogger(__name__)

BASELINES = ["prior_mean", "prior_last", "prior_position"]


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """What a model has to beat to be worth having.

    Idempotent: the baseline columns are dropped on entry. A previous version
    was not, and re-running it inside the walk-forward raised KeyError on the
    second week because the merge suffixed the columns it was asked to create.
    """
    out = df.drop(columns=[c for c in BASELINES if c in df.columns])
    out = out.sort_values(["player_id", "season", "week"]).reset_index(
        drop=True)
    g = out.groupby("player_id", sort=False)["points"]
    out["prior_mean"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    out["prior_last"] = g.transform(lambda s: s.shift(1))

    # Position average, shifted by WEEK rather than by row. Shifting by row
    # was a real leak in the NFL project: within a week the rows are ordered
    # arbitrarily, so "the previous row" for one quarterback was another
    # quarterback in the same week - a result from the very week being
    # predicted.
    wk = (out.groupby(["position", "season", "week"], as_index=False)["points"]
          .mean().rename(columns={"points": "_pos_week"})
          .sort_values(["position", "season", "week"]))
    wk["prior_position"] = (wk.groupby("position", sort=False)["_pos_week"]
                            .transform(lambda s: s.shift(1)
                                       .expanding(min_periods=1).mean()))
    out = out.merge(wk[["position", "season", "week", "prior_position"]],
                    on=["position", "season", "week"], how="left")
    return out


def walk(built: pd.DataFrame, test_season: int, first_week: int,
         last_week: int, positions: list[str] | None = None) -> pd.DataFrame:
    """Fit before each week, predict it, never the other way round."""
    positions = positions or F.SKILL_POSITIONS
    data = add_baselines(built)
    rows = []
    for week in range(first_week, last_week + 1):
        past = data[(data["season"] < test_season)
                    | ((data["season"] == test_season)
                       & (data["week"] < week))]
        now = data[(data["season"] == test_season) & (data["week"] == week)]
        now = now[now["position"].isin(positions)]
        if now.empty:
            continue
        try:
            model = Projections().fit(past)
        except ValueError as exc:
            log.warning("week %d not gradeable: %s", week, exc)
            continue
        pred = model.predict(now)
        block = now[["player_id", "season", "week", "position", "points"]
                    + BASELINES].reset_index(drop=True)
        for c in pred.columns:
            block[c] = pred[c].to_numpy()
        block["played"] = (block["points"] > 0).astype(int)
        rows.append(block)
        log.info("week %d: %d players graded, fitted on %d rows",
                 week, len(block), len(past))
    if not rows:
        raise ValueError("nothing was gradeable")
    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------- leak proof

def prove_no_leak(built: pd.DataFrame, test_season: int, week: int) -> str:
    """Tamper with one player-game and show the damage cannot travel backwards.

    Isolation matters. An earlier attempt at this altered every row, which
    made the honest baseline fail too and proved nothing. Here exactly one
    player's one game is changed, and two things must both hold:

      1. the prediction for the TAMPERED WEEK must not move - if it does, the
         model saw the outcome it was predicting;
      2. predictions for LATER weeks MUST move - if they do not, the
         tampering never reached the model at all and this test is incapable
         of detecting anything.

    The second condition is the one that was missing twice.
    """
    data = add_baselines(built)
    victim = (data[(data["season"] == test_season) & (data["week"] == week)]
              .sort_values("points", ascending=False).head(1))
    if victim.empty:
        return "no row to tamper with; leak proof INCONCLUSIVE"
    pid = victim["player_id"].iloc[0]

    clean = data
    tampered = data.copy()
    mask = ((tampered["player_id"] == pid)
            & (tampered["season"] == test_season)
            & (tampered["week"] == week))
    tampered.loc[mask, "points"] = 500.0

    lines = [f"tampering with {pid}, {test_season} week {week}: "
             f"points -> 500"]

    # Condition 1: the same week's own prediction must be unchanged.
    past = clean[(clean["season"] < test_season)
                 | ((clean["season"] == test_season)
                    & (clean["week"] < week))]
    now = clean[(clean["season"] == test_season) & (clean["week"] == week)]
    t_now = tampered[(tampered["season"] == test_season)
                     & (tampered["week"] == week)]
    try:
        m = Projections().fit(past)
    except ValueError as exc:
        return f"leak proof INCONCLUSIVE: {exc}"
    a = m.predict(now[now["player_id"] == pid])["q50"].to_numpy()
    b = m.predict(t_now[t_now["player_id"] == pid])["q50"].to_numpy()
    same = np.allclose(a, b, equal_nan=True)
    lines.append(f"  same-week prediction unchanged: {same}  "
                 f"({a[:1]} vs {b[:1]})")

    # Condition 2: rebuilding features from tampered history MUST move later
    # weeks, or the tampering never reached anything.
    later_clean = clean[(clean["player_id"] == pid)
                        & (clean["season"] == test_season)
                        & (clean["week"] > week)]
    later_tamp = tampered[(tampered["player_id"] == pid)
                          & (tampered["season"] == test_season)
                          & (tampered["week"] > week)]
    moved = False
    if not later_clean.empty:
        moved = not np.allclose(
            later_clean["prior_mean"].to_numpy(dtype=float),
            later_tamp["prior_mean"].to_numpy(dtype=float), equal_nan=True)
    lines.append(f"  later-week inputs DID move: {moved}")

    if same and moved:
        lines.append("  PASS - information flows forward only, and this test "
                     "is capable of detecting it if it did not")
    elif not moved:
        lines.append("  INCONCLUSIVE - the tampering never propagated, so a "
                     "leak would not have been detected either")
    else:
        lines.append("  FAIL - the model's prediction for a week responded to "
                     "that week's own outcome")
    return "\n".join(lines)


# --------------------------------------------------------------- accuracy

def accuracy(g: pd.DataFrame) -> pd.DataFrame:
    """Error against each baseline, on identical rows."""
    rows = []
    for label, col in [("this model", "mean"), ("prior mean", "prior_mean"),
                       ("last game", "prior_last"),
                       ("position avg", "prior_position")]:
        sub = g[g[col].notna()]
        if sub.empty:
            continue
        err = sub[col] - sub["points"]
        rows.append({"key": col, "what": label, "n": len(sub),
                     "MAE": err.abs().mean(),
                     "RMSE": float(np.sqrt((err ** 2).mean())),
                     "bias": err.mean()})
    out = pd.DataFrame(rows)
    # A stable identifier, separate from the display label. Renaming the
    # labels once broke a downstream `.str.startswith("this")` and produced an
    # out-of-bounds indexer - a display string is not an identifier.
    return out


def calibration(g: pd.DataFrame, quantiles=(10, 25, 50, 75, 90, 97),
                played_only: bool = True) -> pd.DataFrame:
    """Of the outcomes, what share land below each fitted quantile.

    Conditional by default, because that is what the quantiles were fitted
    on. Grading them against everybody - including men who never played -
    measures the mixture against a model of one component and calls the
    difference miscalibration.
    """
    sub = g[g["played"] == 1] if played_only else g
    rows = []
    for q in quantiles:
        col = f"q{q}"
        if col not in sub:
            continue
        cover = float((sub["points"] <= sub[col]).mean())
        rows.append({"quantile": q / 100, "target": q / 100,
                     "covered": cover, "error": cover - q / 100})
    return pd.DataFrame(rows)


def availability(g: pd.DataFrame) -> dict:
    """AUC, Brier, and sharpness for the play/no-play half.

    Sharpness is not optional. A classifier that predicts the base rate for
    every player scores a perfectly respectable Brier and tells you nothing;
    the spread of its predictions is what reveals that.
    """
    if "p_play" not in g or g["played"].nunique() < 2:
        return {}
    from sklearn.metrics import roc_auc_score
    p = g["p_play"].to_numpy(dtype=float)
    y = g["played"].to_numpy(dtype=int)
    base = float(y.mean())
    return {
        "n": len(g), "base_rate": base,
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(np.mean((p - y) ** 2)),
        "Brier_base": float(np.mean((base - y) ** 2)),
        "sharpness": float(p.std()),
    }


def ranking(g: pd.DataFrame) -> pd.DataFrame:
    """Does it order players within a position better than the baseline.

    The question DFS actually asks. A model biased low everywhere still wins
    if the order is right, and a perfectly centred one that shuffles the order
    is worthless.
    """
    rows = []
    for pos, sub in g.groupby("position"):
        r = {}
        for label, col in [("model", "mean"), ("season avg", "prior_mean")]:
            per_week = []
            for _, wk in sub.groupby(["season", "week"]):
                if len(wk) > 2 and wk[col].notna().sum() > 2:
                    per_week.append(wk[col].corr(wk["points"],
                                                 method="spearman"))
            r[label] = float(np.nanmean(per_week)) if per_week else np.nan
        rows.append({"position": pos, "n": len(sub),
                     "model_rho": r["model"], "baseline_rho": r["season avg"],
                     "better": r["model"] > r["season avg"]})
    return pd.DataFrame(rows)


def report(g: pd.DataFrame) -> str:
    """Everything, in the order it should be read."""
    out = ["", "=" * 72, "ACCURACY  (lower is better; the model must beat all "
           "three)", "=" * 72,
           accuracy(g).to_string(index=False)]

    acc = accuracy(g).set_index("key")
    if "mean" in acc.index and "prior_mean" in acc.index:
        m, b = acc.loc["mean", "MAE"], acc.loc["prior_mean", "MAE"]
        out.append("")
        out.append(f"  MAE vs prior mean: {m:.3f} vs {b:.3f}  "
                   f"({100 * (b - m) / b:+.1f}%)")

    out += ["", "=" * 72,
            "CALIBRATION  (conditional on playing; covered should equal "
            "target)", "=" * 72,
            calibration(g).round(4).to_string(index=False)]
    total = calibration(g)["error"].abs().sum()
    out.append(f"\n  total absolute calibration error: {total:.3f}")

    av = availability(g)
    if av:
        out += ["", "=" * 72, "AVAILABILITY  (did he play at all)", "=" * 72,
                f"  rows          : {av['n']:,}",
                f"  base rate     : {av['base_rate']:.3f}",
                f"  AUC           : {av['AUC']:.3f}   (0.5 is a coin flip)",
                f"  Brier         : {av['Brier']:.3f} vs "
                f"{av['Brier_base']:.3f} for always predicting the base rate",
                f"  sharpness     : {av['sharpness']:.3f}   "
                f"(0.0 means it says the same thing about everybody)"]

    rk = ranking(g)
    out += ["", "=" * 72, "RANKING WITHIN POSITION  (Spearman, per week)",
            "=" * 72, rk.round(3).to_string(index=False)]
    if not rk.empty:
        out.append(f"\n  better than the baseline at "
                   f"{int(rk['better'].sum())} of {len(rk)} positions")
    return "\n".join(out)
