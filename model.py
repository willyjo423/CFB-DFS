"""Each player as a distribution, not a number. Any sport.

Why quantiles rather than a mean
--------------------------------
DraftKings pays three points at exactly 100 receiving yards. That is a step
function, so two players with identical averages are worth different amounts.
And a tournament pays almost nothing for the middle of a distribution; it
pays for the top. A mean projection cannot express either fact.

So the model fits percentiles directly with pinball loss, including one out
in the tail, which is the part that wins a tournament and the part a
mean-squared-error fit smooths away.

Two questions, not one
----------------------
The NFL version of this originally fitted its quantiles on every row,
including the ones where a player scored zero because he never took the
field. That makes one model answer two unrelated questions at once - WILL he
play, and HOW WELL - and report the blend as a single distribution. Grading
it proved the damage: among players who actually played, the tenth percentile
covered 3.1% of outcomes instead of 10%, the median 42% instead of 50%. The
whole distribution had been dragged down by games nobody played in. Splitting
it cut total calibration error by 81%.

Every sport has this problem and some have it worse. College football has no
injury report and rosters past a hundred; baseball has a starting pitcher who
appears every fifth day. So the split is in the engine, not in any one sport:
a classifier for whether he takes part, quantiles fitted ONLY on the
occasions he did, and the two recombined at sampling time rather than inside
the fit - which is what keeps each half interpretable on its own.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from . import features as EF

log = logging.getLogger(__name__)

RANDOM_SEED = 1729


class Projections:
    """Fit once, predict a distribution per player."""

    def __init__(self, spec):
        self.spec = spec
        self.quantiles = list(spec.quantiles)
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.availability: HistGradientBoostingClassifier | None = None
        self.base_rate = 1.0
        self.columns: list[str] = []
        self.trained_rows = 0
        self.played_rows = 0

    # ------------------------------------------------------------------ fit
    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[[c for c in self.spec.features if c in df.columns]].copy()
        for pos in self.spec.positions:
            X[f"is_{pos}"] = (df["position"] == pos).astype(float)
        return X

    @staticmethod
    def _usable(X: pd.DataFrame) -> list[str]:
        """Columns with something in them to learn from.

        A feature that is entirely missing, or holds a single repeated value,
        carries no information - and the histogram binner cannot build a
        threshold from one distinct value, so it raises rather than ignoring
        it. That killed the NFL project's first live fit: the weekly player
        file had no home/away flag, `is_home` arrived as a column of NaN, and
        the run died twenty minutes before kickoff.

        Dropped here rather than pruned from the spec, so the build stays
        tolerant of a source that adds or removes a column - which these
        sources demonstrably do.
        """
        keep = []
        for c in X.columns:
            col = X[c]
            if col.notna().sum() < 2 or col.nunique(dropna=True) < 2:
                continue
            keep.append(c)
        return keep

    def fit(self, df: pd.DataFrame) -> "Projections":
        spec = self.spec
        train = EF.trainable(df, spec)
        if len(train) < spec.min_train_rows:
            raise ValueError(
                f"{spec.name}: only {len(train)} usable rows; a quantile fit "
                f"on that little evidence is noise wearing a model's clothes")
        X = self._design(train)
        usable = self._usable(X)
        dropped = [c for c in X.columns if c not in usable]
        if dropped:
            log.warning("%s: dropping %d empty or constant features: %s",
                        spec.name, len(dropped), ", ".join(dropped))
        X = X[usable]
        if X.empty or not len(X.columns):
            raise ValueError(f"{spec.name}: no usable features survived")
        self.columns = list(X.columns)
        self.trained_rows = len(train)

        # --- part one: did he take part at all -----------------------------
        # Fitted on EVERY trainable row, because the question is precisely
        # about the rows where nothing happened. Scoring nothing and not
        # playing are indistinguishable in most feeds, which is why the
        # output is a probability and not a flag.
        played = (train["points"].to_numpy(dtype=float) > 0).astype(int)
        self.base_rate = float(played.mean())
        if played.min() == played.max():
            log.warning("%s: every training row has the same play/no-play "
                        "outcome (%d%%); no classifier can be fitted",
                        spec.name, int(self.base_rate * 100))
            self.availability = None
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=RANDOM_SEED)
            clf.fit(X, played)
            self.availability = clf

        # --- part two: how well, GIVEN that he did -------------------------
        active = played.astype(bool)
        self.played_rows = int(active.sum())
        if self.played_rows < spec.min_played_rows:
            raise ValueError(
                f"{spec.name}: only {self.played_rows} rows involve a player "
                f"who actually took part; the conditional quantiles have "
                f"nothing to fit")
        Xa, ya = X[active], train["points"].to_numpy(dtype=float)[active]

        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=400, learning_rate=0.06,
                max_depth=6, min_samples_leaf=40, l2_regularization=1.0,
                random_state=RANDOM_SEED)
            m.fit(Xa, ya)
            self.models[q] = m

        log.info("%s: fitted %d quantiles on %d played rows (of %d; %.1f%% "
                 "played), %d features", spec.name, len(self.models),
                 self.played_rows, self.trained_rows, self.base_rate * 100,
                 len(self.columns))
        return self

    # -------------------------------------------------------------- predict
    def play_probability(self, X: pd.DataFrame) -> np.ndarray:
        if self.availability is None:
            return np.full(len(X), self.base_rate, dtype=float)
        return self.availability.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("predict() before fit()")
        spec = self.spec
        X = self._design(df)
        for c in self.columns:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.columns]

        out = pd.DataFrame(index=df.index)
        for q in self.quantiles:
            out[spec.qcol(q)] = self.models[q].predict(X)

        # Sort each row, so a 75th can never sit below a 25th. Independently
        # fitted models cross on the odd row.
        cols = spec.qcols
        out[cols] = np.sort(out[cols].to_numpy(), axis=1)
        out[cols] = out[cols].clip(lower=0.0)

        # Everything above is CONDITIONAL on taking part, because that is
        # what the quantiles were fitted on.
        out["p_play"] = np.clip(self.play_probability(X), 0.0, 1.0)
        mid = min(self.quantiles, key=lambda x: abs(x - 0.5))
        out["median"] = out[spec.qcol(mid)]

        # The conditional mean, as the integral of the quantile function.
        # E[X] is the area under Q(u) on [0, 1], which these quantiles sample
        # directly. The NFL version hardcoded weights against q10..q90 that
        # put ZERO weight above the 90th - understating a skewed player by
        # about 14%, which matters because this is what cash lineups and
        # ownership are priced from. Ends extended flat, which understates
        # the extreme tail slightly; the tournament side reads the top fitted
        # quantile directly and is not smoothed at all.
        u = np.concatenate([[0.0], np.asarray(self.quantiles, float), [1.0]])
        grid = out[cols].to_numpy(dtype=float)
        vals = np.concatenate([grid[:, :1], grid, grid[:, -1:]], axis=1)
        out["cond_mean"] = np.trapezoid(vals, u, axis=1)

        # And the UNCONDITIONAL expectation - what he is worth before you
        # know whether he suits up. Keeping the two apart under different
        # names is the whole point of the split: a doubtful starter has a
        # large `cond_mean` and a modest `mean`, and collapsing them back
        # into one column rebuilds the exact bug the split removes.
        out["mean"] = out["cond_mean"] * out["p_play"]
        out["ceiling"] = out[spec.qcol(self.quantiles[-1])]
        out["spread"] = (out[spec.qcol(self.quantiles[-1])]
                         - out[spec.qcol(self.quantiles[0])])
        return out

    # ------------------------------------------------------------- sampling
    def sample(self, projected: pd.DataFrame, n: int,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw outcomes from each player's own fitted distribution.

        Inverse-CDF sampling through the quantiles: pick a uniform, find
        where it falls between two fitted percentiles, interpolate. This lets
        the simulator work with the SHAPE the model produced rather than
        flattening every player to a mean and a standard deviation, which
        would discard the reason for fitting quantiles at all.
        """
        rng = rng or np.random.default_rng(RANDOM_SEED)
        qs = np.array(self.quantiles, dtype=float)
        grid = projected[self.spec.qcols].to_numpy(dtype=float)

        u = rng.random((len(projected), n))
        out = np.empty_like(u)
        for i in range(len(projected)):
            out[i] = np.interp(u[i], qs, grid[i])

        # The availability gate. The curve says what he scores when he plays;
        # this says whether he played. Multiplying recombines two honest
        # halves into the mixture a lineup actually faces, and doing it HERE
        # rather than inside the fitted quantiles is what keeps each half
        # interpretable on its own.
        if "p_play" in projected.columns:
            p = pd.to_numeric(projected["p_play"], errors="coerce").to_numpy(
                dtype=float)
            p = np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)[:, None]
            out = out * (rng.random(out.shape) < p)
        return out


def latest_rows(built: pd.DataFrame) -> pd.DataFrame:
    """The most recent feature row per player - what a projection is made from.

    Deliberately the last row available rather than a row for the upcoming
    game: the features describe what a player has already done, and the
    upcoming game has not happened yet.
    """
    return (built.sort_values(["player_id", "season", "period"])
            .groupby("player_id", as_index=False).tail(1)
            .reset_index(drop=True))
