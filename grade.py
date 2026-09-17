"""Is the model any good. Walk-forward, against baselines it must beat.

A projection nobody has graded is an opinion. This grades one the only way
that means anything: fit on everything strictly before period P, predict P,
move on, refit. No row is ever predicted by a model that saw it.

What is measured, and why each
------------------------------
**Against baselines, not in isolation.** An MAE of 4.5 is meaningless alone.
Three baselines a real model must beat: the player's own prior mean, his last
outing, and his position's average. A model that cannot beat "what he did
last time" has learned nothing worth the compute.

**Calibration, as coverage counting.** Of the outcomes, what share land below
each fitted quantile? A 10th percentile should sit above 10% of results. This
is what caught the NFL model blending play and no-play into one distribution:
its 10th covered 3.1%.

**Availability separately**, with a sharpness check - a classifier that
predicts the base rate for everybody scores a respectable Brier and is
useless, and only the spread of its predictions reveals that.

**Rank within position.** DFS does not need the level right; it needs the
ORDER right. A model biased low everywhere still wins if it ranks correctly.

**A leak proof that can fail.** Two earlier versions could not: one deleted
future rows in a way that changed nothing, and one tampered so crudely the
honest baseline failed too. This one isolates a single player-period,
requires the tampering to propagate forward, and requires it NOT to reach the
prediction for the tampered period itself.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import features as EF
from .model import Projections

log = logging.getLogger(__name__)

BASELINES = ["prior_mean", "prior_last", "prior_position"]


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """What a model has to beat to be worth having.

    Idempotent: the baseline columns are dropped on entry. A previous version
    was not, and re-running it inside the walk-forward raised KeyError on the
    second period because the merge suffixed the columns it was creating.
    """
    out = df.drop(columns=[c for c in BASELINES if c in df.columns])
    out = out.sort_values(["player_id", "season", "period"]).reset_index(
        drop=True)
    g = out.groupby("player_id", sort=False)["points"]
    out["prior_mean"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    out["prior_last"] = g.transform(lambda s: s.shift(1))

    # Position average, shifted by PERIOD rather than by row. Shifting by row
    # was a real leak in the NFL project: within a period the rows are
    # ordered arbitrarily, so "the previous row" for one quarterback was
    # another quarterback in the same week - a result from the very week
    # being predicted.
    per = (out.groupby(["position", "season", "period"], as_index=False)
           ["points"].mean().rename(columns={"points": "_pos_period"})
           .sort_values(["position", "season", "period"]))
    per["prior_position"] = (per.groupby("position", sort=False)
                             ["_pos_period"]
                             .transform(lambda s: s.shift(1)
                                        .expanding(min_periods=1).mean()))
    return out.merge(per[["position", "season", "period", "prior_position"]],
                     on=["position", "season", "period"], how="left")


def walk(built: pd.DataFrame, spec, test_season: int, first: int, last: int
         ) -> pd.DataFrame:
    """Fit before each period, predict it, never the other way round."""
    data = add_baselines(built)
    rows = []
    for period in range(first, last + 1):
        past = data[(data["season"] < test_season)
                    | ((data["season"] == test_season)
                       & (data["period"] < period))]
        now = data[(data["season"] == test_season)
                   & (data["period"] == period)]
        now = now[now["position"].isin(spec.positions)]
        if now.empty:
            continue
        try:
            model = Projections(spec).fit(past)
        except ValueError as exc:
            log.warning("period %d not gradeable: %s", period, exc)
            continue
        pred = model.predict(now)
        block = now[["player_id", "season", "period", "position", "points"]
                    + BASELINES].reset_index(drop=True)
        for c in pred.columns:
            block[c] = pred[c].to_numpy()
        block["played"] = (block["points"] > 0).astype(int)
        rows.append(block)
        log.info("period %d: %d players graded, fitted on %d rows",
                 period, len(block), len(past))
    if not rows:
        raise ValueError("nothing was gradeable")
    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------- leak proof

def prove_no_leak(built: pd.DataFrame, spec, test_season: int,
                  period: int) -> str:
    """Tamper with one player-period; show the damage cannot travel backwards.

    Isolation matters. An earlier attempt altered every row, which made the
    honest baseline fail too and proved nothing. Here exactly one player's
    one outing is changed, and BOTH must hold:

      1. the prediction for the TAMPERED period must not move - if it does,
         the model saw the outcome it was predicting;
      2. inputs for LATER periods MUST move - if they do not, the tampering
         never reached anything and this test could not detect a leak either.

    The second condition is the one that was missing twice.

    Why the tampering goes BOTH ways
    --------------------------------
    This used to pick the day's highest scorer and then raise his points to
    500. Those two choices cancel each other out. A gradient-boosted tree only
    changes its answer when an input crosses a split threshold, and every
    threshold it learned lies inside the observed range - so moving a value
    that is ALREADY at the top of that range even higher crosses nothing.

    That was not theoretical. Declaring `points` itself as a feature - about
    as blatant a leak as can be written - still produced an identical
    prediction to fifteen decimal places, and this check called it a PASS.

    So the outcome is now driven to both extremes. A leak has to survive
    being told the player scored nothing AND that he scored 500.
    """
    data = add_baselines(built)
    here = data[(data["season"] == test_season) & (data["period"] == period)]
    if here.empty:
        return ("no row to tamper with; leak proof INCONCLUSIVE - treat as a "
                "failure to verify, not as a pass")
    pid = here.sort_values("points", ascending=False)["player_id"].iloc[0]

    # Tamper with the RAW frame and re-derive EVERYTHING from it.
    #
    # This used to copy `data` - a frame whose features and baselines had
    # already been materialised - and change `points` on the copy. Nothing
    # downstream recomputed, so the later-period inputs could not possibly
    # move, condition 2 was False on every run this check has ever made, and
    # it printed INCONCLUSIVE every single time, at the top of a wall of
    # numbers that looked fine.
    #
    # A check that cannot pass proves exactly as little as one that cannot
    # fail. This project has now produced both.
    mask = ((built["player_id"] == pid)
            & (built["season"] == test_season)
            & (built["period"] == period))
    lines = [f"tampering with {pid}, {test_season} period {period}"]

    past = data[(data["season"] < test_season)
                | ((data["season"] == test_season)
                   & (data["period"] < period))]
    try:
        m = Projections(spec).fit(past)
    except ValueError as exc:
        return f"leak proof INCONCLUSIVE: {exc}"

    a = m.predict(here[here["player_id"] == pid])[spec.qcol(0.5)].to_numpy()
    sel = lambda d: d[(d["player_id"] == pid)                    # noqa: E731
                      & (d["season"] == test_season)
                      & (d["period"] > period)]["prior_mean"].to_numpy(float)
    later_clean = sel(data)

    same, moved = True, False
    for value in (0.0, 500.0):
        raw = built.copy()
        raw.loc[mask, "points"] = value
        tampered = add_baselines(EF.build(raw, spec, validate=False))

        t_here = tampered[(tampered["season"] == test_season)
                          & (tampered["period"] == period)]
        b = m.predict(t_here[t_here["player_id"] == pid])[
            spec.qcol(0.5)].to_numpy()
        unchanged = bool(np.allclose(a, b, equal_nan=True))
        same = same and unchanged
        lines.append(f"  points -> {value:>5.0f}: same-period prediction "
                     f"unchanged: {unchanged}")

        moved = moved or bool(
            len(later_clean)
            and not np.allclose(later_clean, sel(tampered), equal_nan=True))

    lines.append(f"  later-period inputs DID move: {moved}")

    if same and moved:
        lines.append("  PASS - information flows forward only, and this test "
                     "is capable of detecting it if it did not")
    elif not moved:
        lines.append("  INCONCLUSIVE - the tampering never propagated, so a "
                     "leak would not have been detected either")
    else:
        lines.append("  FAIL - a prediction responded to its own outcome")
    return "\n".join(lines)


# ---------------------------------------------------------------- measures

def accuracy(g: pd.DataFrame) -> pd.DataFrame:
    """Error against each baseline, on identical rows."""
    rows = []
    for label, col in [("this model", "mean"), ("prior mean", "prior_mean"),
                       ("last outing", "prior_last"),
                       ("position avg", "prior_position")]:
        sub = g[g[col].notna()]
        if sub.empty:
            continue
        err = sub[col] - sub["points"]
        # `key` is a stable identifier, separate from the display label.
        # Renaming labels once broke a downstream startswith() and produced
        # an out-of-bounds indexer: a display string is not an identifier.
        rows.append({"key": col, "what": label, "n": len(sub),
                     "MAE": err.abs().mean(),
                     "RMSE": float(np.sqrt((err ** 2).mean())),
                     "bias": err.mean()})
    return pd.DataFrame(rows)


def calibration(g: pd.DataFrame, spec, played_only: bool = True
                ) -> pd.DataFrame:
    """Of the outcomes, what share land below each fitted quantile.

    Conditional by default, because that is what the quantiles were fitted
    on. Grading them against everybody - including players who never took
    part - measures the mixture against a model of one component and calls
    the difference miscalibration.
    """
    sub = g[g["played"] == 1] if played_only else g
    rows = []
    for q in spec.quantiles:
        col = spec.qcol(q)
        if col not in sub:
            continue
        cover = float((sub["points"] <= sub[col]).mean())
        rows.append({"quantile": q, "target": q, "covered": cover,
                     "error": cover - q})
    return pd.DataFrame(rows)


def availability(g: pd.DataFrame) -> dict:
    """AUC, Brier and sharpness for the play/no-play half."""
    if "p_play" not in g or g["played"].nunique() < 2:
        return {}
    from sklearn.metrics import roc_auc_score
    p = g["p_play"].to_numpy(dtype=float)
    y = g["played"].to_numpy(dtype=int)
    base = float(y.mean())
    return {"n": len(g), "base_rate": base,
            "AUC": float(roc_auc_score(y, p)),
            "Brier": float(np.mean((p - y) ** 2)),
            "Brier_base": float(np.mean((base - y) ** 2)),
            "sharpness": float(p.std())}


def ranking(g: pd.DataFrame) -> pd.DataFrame:
    """Does it order players within a position better than the baseline.

    The question DFS actually asks. A model biased low everywhere still wins
    if the order is right; a perfectly centred one that shuffles is useless.
    """
    rows = []
    for pos, sub in g.groupby("position"):
        r = {}
        for label, col in [("model", "mean"), ("season avg", "prior_mean")]:
            per = [wk[col].corr(wk["points"], method="spearman")
                   for _, wk in sub.groupby(["season", "period"])
                   if len(wk) > 2 and wk[col].notna().sum() > 2]
            r[label] = float(np.nanmean(per)) if per else np.nan
        rows.append({"position": pos, "n": len(sub),
                     "model_rho": r["model"], "baseline_rho": r["season avg"],
                     "better": r["model"] > r["season avg"]})
    return pd.DataFrame(rows)


def report(g: pd.DataFrame, spec) -> str:
    """Everything, in the order it should be read."""
    acc = accuracy(g)
    out = ["", "=" * 72,
           "ACCURACY  (lower is better; the model must beat all three)",
           "=" * 72, acc.to_string(index=False)]
    idx = acc.set_index("key")
    if "mean" in idx.index and "prior_mean" in idx.index:
        m, b = idx.loc["mean", "MAE"], idx.loc["prior_mean", "MAE"]
        out += ["", f"  MAE vs prior mean: {m:.3f} vs {b:.3f}  "
                    f"({100 * (b - m) / b:+.1f}%)"]

    cal = calibration(g, spec)
    out += ["", "=" * 72,
            "CALIBRATION  (conditional on playing; covered should equal "
            "target)", "=" * 72, cal.round(4).to_string(index=False),
            f"\n  total absolute calibration error: "
            f"{cal['error'].abs().sum():.3f}"]

    av = availability(g)
    if av:
        out += ["", "=" * 72, "AVAILABILITY  (did he take part at all)",
                "=" * 72,
                f"  rows      : {av['n']:,}",
                f"  base rate : {av['base_rate']:.3f}",
                f"  AUC       : {av['AUC']:.3f}   (0.5 is a coin flip)",
                f"  Brier     : {av['Brier']:.3f} vs {av['Brier_base']:.3f} "
                f"for always predicting the base rate",
                f"  sharpness : {av['sharpness']:.3f}   "
                f"(0.0 means it says the same thing about everybody)"]

    rk = ranking(g)
    out += ["", "=" * 72, "RANKING WITHIN POSITION  (Spearman, per period)",
            "=" * 72, rk.round(3).to_string(index=False)]
    if not rk.empty:
        out.append(f"\n  better than the baseline at "
                   f"{int(rk['better'].sum())} of {len(rk)} positions")
    return "\n".join(out)
