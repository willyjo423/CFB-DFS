"""Each player as a distribution, not a number.

Why quantiles rather than a mean
--------------------------------
DraftKings pays three points at exactly 100 receiving yards. That is a step
function, so two players with identical averages are worth different amounts -
the wider one crosses the line more often. And a tournament pays almost
nothing for the middle of a distribution; it pays for the top of it. A mean
projection cannot express either fact.

So the model fits the 10th, 25th, 50th, 75th, 90th and 97th percentiles
directly with pinball loss. The 97th is in there deliberately: it is the part
of a player that wins a tournament, and the part a mean-squared-error fit
smooths away.

One model per quantile, not per position
----------------------------------------
Position enters as a feature instead. Splitting by position sounds tidier but
quarters the data behind each fit, and the thing being learned - that volume
predicts points, that a role change matters more than a season average - is
shared across positions.

Two questions, not one
----------------------
The NFL version of this model originally fitted its quantiles on every row,
including the ones where a player scored zero because he never took the
field. That makes one model answer two unrelated questions at once - WILL he
play, and HOW WELL - and report the blend as a single distribution. Grading
it proved the damage: among players who actually played, every fitted
quantile undershot. The tenth percentile covered 3.1% of outcomes instead of
10%; the median covered 42% instead of 50%. The whole distribution had been
dragged down by games nobody played in. Splitting it cut total calibration
error by 81%.

College football needs that split MORE, not less. There is no injury report
here, rosters run past a hundred, and starters leave lopsided games in the
third quarter. The proportion of priced players who record nothing on a given
Saturday is larger than in the NFL, and it is exactly the population a single
blended fit would quietly average into everybody else.

So: a classifier for whether he plays, quantiles fitted ONLY on the games
that were played, and the two recombined in the simulator rather than inside
the fit - which is what keeps each half interpretable on its own.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import cfb_features as F

log = logging.getLogger(__name__)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
RANDOM_SEED = 1729
POSITIONS = F.SKILL_POSITIONS

# Below these, a fit is noise wearing a model's clothes. They are deliberately
# separate numbers: the first is how much evidence exists at all, the second
# is how much of it involves somebody actually playing, and only the second
# constrains the quantile fit.
MIN_TRAIN_ROWS = 500
MIN_PLAYED_ROWS = 400


class Projections:
    """Fit once, predict a distribution per player."""

    def __init__(self, quantiles: list[float] | None = None):
        self.quantiles = quantiles or QUANTILES
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.availability: HistGradientBoostingClassifier | None = None
        self.base_rate = 1.0
        self.columns: list[str] = []
        self.trained_rows = 0
        self.played_rows = 0

    # ------------------------------------------------------------------ fit
    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[[c for c in F.FEATURES if c in df.columns]].copy()
        for pos in POSITIONS:
            X[f"is_{pos}"] = (df["position"] == pos).astype(float)
        return X

    @staticmethod
    def _usable(X: pd.DataFrame) -> list[str]:
        """Columns with something in them to learn from.

        A feature that is entirely missing, or holds a single repeated value,
        carries no information - and the histogram binner cannot build a
        threshold from one distinct value, so it raises rather than ignoring
        it. That is what killed the NFL project's first live fit: the weekly
        player file had no home/away flag, so `is_home` arrived as a column of
        NaN and took the run down twenty minutes before kickoff.

        Dropping them here rather than pruning the feature list keeps the
        build tolerant of a source that adds or removes a column, which these
        sources demonstrably do.
        """
        keep = []
        for c in X.columns:
            col = X[c]
            if col.notna().sum() < 2:
                continue
            if col.nunique(dropna=True) < 2:
                continue
            keep.append(c)
        return keep

    def fit(self, df: pd.DataFrame) -> "Projections":
        train = F.trainable(df)
        if len(train) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"only {len(train)} usable rows; a quantile fit on that "
                f"little evidence is noise wearing a model's clothes")
        X = self._design(train)
        usable = self._usable(X)
        dropped = [c for c in X.columns if c not in usable]
        if dropped:
            log.warning("dropping %d empty or constant features: %s",
                        len(dropped), ", ".join(dropped))
        X = X[usable]
        if X.empty or not len(X.columns):
            raise ValueError("no usable features survived")
        self.columns = list(X.columns)
        self.trained_rows = len(train)

        # --- part one: did he play at all ---------------------------------
        # Fitted on EVERY trainable row, because the question is precisely
        # about the rows where nothing happened. Scoring nothing and not
        # taking the field are indistinguishable in this data, which is why
        # the output is a probability rather than a flag.
        played = (train["points"].to_numpy(dtype=float) > 0).astype(int)
        self.base_rate = float(played.mean())
        if played.min() == played.max():
            log.warning("every training row has the same play/no-play "
                        "outcome (%d%%); no classifier can be fitted",
                        int(self.base_rate * 100))
            self.availability = None
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=RANDOM_SEED)
            clf.fit(X, played)
            self.availability = clf

        # --- part two: how well, GIVEN that he played ---------------------
        active = played.astype(bool)
        self.played_rows = int(active.sum())
        if self.played_rows < MIN_PLAYED_ROWS:
            raise ValueError(
                f"only {self.played_rows} games were actually played in the "
                f"training set; the conditional quantiles have nothing to fit")
        Xa = X[active]
        ya = train["points"].to_numpy(dtype=float)[active]

        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=400,
                learning_rate=0.06, max_depth=6, min_samples_leaf=40,
                l2_regularization=1.0, random_state=RANDOM_SEED)
            m.fit(Xa, ya)
            self.models[q] = m

        log.info("fitted %d quantiles on %d played games (of %d rows; "
                 "%.1f%% played), %d features",
                 len(self.models), self.played_rows, self.trained_rows,
                 self.base_rate * 100, len(self.columns))
        return self

    # -------------------------------------------------------------- predict
    def _nearest(self, q: float) -> float:
        """The fitted quantile closest to q. Nothing here assumes 0.5 exists."""
        return min(self.quantiles, key=lambda x: abs(x - q))

    def play_probability(self, X: pd.DataFrame) -> np.ndarray:
        if self.availability is None:
            return np.full(len(X), self.base_rate, dtype=float)
        return self.availability.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("predict() before fit()")
        X = self._design(df)
        for c in self.columns:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.columns]

        out = pd.DataFrame(index=df.index)
        for q in self.quantiles:
            out[f"q{int(q * 100)}"] = self.models[q].predict(X)

        # Sort each row, so a 75th can never sit below a 25th. Six
        # independently fitted models can cross on an odd row.
        cols = [f"q{int(q * 100)}" for q in self.quantiles]
        out[cols] = np.sort(out[cols].to_numpy(), axis=1)
        out[cols] = out[cols].clip(lower=0.0)

        # Everything above is CONDITIONAL on taking the field, because that
        # is what the quantiles were fitted on.
        out["p_play"] = np.clip(self.play_probability(X), 0.0, 1.0)
        out["median"] = out[f"q{int(self._nearest(0.5) * 100)}"]

        # The conditional mean, as the integral of the quantile function.
        #
        # E[X] = the area under Q(u) for u in [0, 1], which the fitted
        # quantiles sample directly. The NFL version hardcoded weights of
        # 0.1/0.2/0.4/0.2/0.1 against q10..q90 - fine for one fixed set of
        # quantiles and an immediate KeyError for any other, which is how a
        # smoke test with three quantiles instead of six found this.
        #
        # The ends are extended flat, which slightly understates the extreme
        # tail. That is deliberate and safe here: `cond_mean` drives cash
        # lineups and ownership, where the tail is not the question, and the
        # tournament side reads `ceiling` - the top fitted quantile - which
        # is not smoothed at all.
        u = np.concatenate([[0.0], np.asarray(self.quantiles, dtype=float),
                            [1.0]])
        grid = out[cols].to_numpy(dtype=float)
        vals = np.concatenate([grid[:, :1], grid, grid[:, -1:]], axis=1)
        out["cond_mean"] = np.trapezoid(vals, u, axis=1)
        # And this is the UNCONDITIONAL expectation - what he is worth before
        # you know whether he suits up. Keeping the two apart under different
        # names is the whole point of the split: a doubtful starter has a
        # large `cond_mean` and a modest `mean`, and collapsing them back into
        # one column rebuilds the exact bug the split exists to remove.
        out["mean"] = out["cond_mean"] * out["p_play"]
        out["ceiling"] = out[f"q{int(self.quantiles[-1] * 100)}"]
        out["spread"] = (out[f"q{int(self.quantiles[-1] * 100)}"]
                         - out[f"q{int(self.quantiles[0] * 100)}"])
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
        cols = [f"q{int(q * 100)}" for q in self.quantiles]
        grid = projected[cols].to_numpy(dtype=float)

        u = rng.random((len(projected), n))
        out = np.empty_like(u)
        for i in range(len(projected)):
            out[i] = np.interp(u[i], qs, grid[i])

        # The availability gate. The curve above says what he scores when he
        # plays; this says whether he played. Multiplying recombines two
        # honest halves into the mixture a lineup actually faces, and doing it
        # HERE rather than inside the fitted quantiles is what keeps each half
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
    week: the features describe what a player has already done, and the
    upcoming week has not happened yet.
    """
    return (built.sort_values(["player_id", "season", "week"])
            .groupby("player_id", as_index=False).tail(1)
            .reset_index(drop=True))
