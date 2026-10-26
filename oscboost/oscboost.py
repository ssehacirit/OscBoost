"""OscBoost: gradient boosting with residual sign history features.

Training
--------
For every training sample i, OscBoost follows the sign of its pseudo
residual across boosting iterations and keeps two counts:

* the oscillation count S_i, which grows by 1 at every sign change, and
* the magnitude count V_i, which grows at every sign change by the swing
  eta * (E_i - E_i_last), where E_i is the path excess of the sample.

The path excess compares how far the raw tree outputs u of a sample have
travelled in total with how far they moved from the start,
E = sum |u| - |sum u|. Tree T_m receives the two counts from before
iteration m, divided by m, as two extra input features:

    z_i = [x_i, S_i / m, V_i / m].

A sign change is counted only when two consecutive pseudo residuals have
strictly opposite signs (a zero residual has sign 0).

Inference (carry mean)
----------------------
Both features depend on the target, so they are unknown for a new sample.
After training, every tree stores two lookup tables over its leaves:
the median of the final score S/M and the mean of the final magnitude V/M
of the training rows in each leaf. A test input is passed through the trees
in order. Each tree reads the stored values of the leaf the input reaches,
and the features given to the next tree are the running means of all values
read so far (starting from 0).
"""

import numpy as np

from .baseline import GradientBoostedTrees, squared_error_gradients, _gain_importance, _HESS_EPS


class OscBoost(GradientBoostedTrees):
    """Gradient boosting with the oscillation score and oscillation magnitude.

    Takes the same parameters as :class:`GradientBoostedTrees`. OscBoost adds
    no hyperparameter of its own. The two features are appended after the
    original inputs, so ``max_features`` is a fraction of p + 2 columns.
    """

    FEATURE_NAMES = ('oscillation_score', 'oscillation_magnitude')

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        lr = self.learning_rate
        M_trees = self.n_estimators

        self.init_pred_ = float(np.mean(y))
        self.rng_ = np.random.RandomState(self.random_state)
        self.trees_ = []
        F = np.full(n, self.init_pred_)

        S = np.zeros(n)            # oscillation count
        V = np.zeros(n)            # magnitude count
        E = np.zeros(n)            # path excess of the raw tree outputs
        D = np.zeros(n)            # cumulative raw tree output
        E_last = np.zeros(n)       # path excess at the last sign change
        prev_sign = None
        leaf_ids = np.empty((M_trees, n), dtype=np.int64)

        for m in range(1, M_trees + 1):
            # Features for tree m use the counts from before iteration m.
            Z = np.column_stack([X, S / m, V / m])

            grad, hess = squared_error_gradients(y, F)
            residuals = -grad / (hess + _HESS_EPS)

            sign = np.sign(residuals)
            if prev_sign is not None:
                flip = (sign * prev_sign < 0).astype(float)
                swing = lr * (E - E_last)
                S += flip
                V += flip * swing
                changed = flip.astype(bool)
                E_last[changed] = E[changed]
            prev_sign = sign.copy()

            idx = self._subsample_index(n)
            tree = self._make_tree()
            tree.fit(Z[idx], residuals[idx])

            # Leaf of every training row for the lookup tables. The study
            # uses the counts after the update of iteration m here.
            leaf_ids[m - 1] = tree.apply(np.column_stack([X, S / m, V / m]))

            u = tree.predict(Z)
            F += lr * u
            E += np.abs(u) - (np.abs(D + u) - np.abs(D))
            D += u
            self.trees_.append(tree)

        self.score_tables_, self.magnitude_tables_ = _build_tables(
            leaf_ids, S / M_trees, V / M_trees)
        self.feature_importances_ = _gain_importance(self.trees_, X.shape[1] + 2)
        self.train_score_ = S / M_trees
        self.train_magnitude_ = V / M_trees
        return self

    def predict(self, X, return_features=False):
        """Predict with the carry mean reconstruction of both features.

        If ``return_features`` is True, also return the carry mean estimates
        of the oscillation score and the magnitude after the last tree.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        F = np.full(n, self.init_pred_)
        o_hat = np.zeros(n)
        v_hat = np.zeros(n)
        sum_o = np.zeros(n)
        sum_v = np.zeros(n)

        for t, tree in enumerate(self.trees_, start=1):
            Z = np.column_stack([X, o_hat, v_hat])
            F += self.learning_rate * tree.predict(Z)
            leaves = tree.apply(Z)
            sum_o += _lookup(self.score_tables_[t - 1], leaves)
            sum_v += _lookup(self.magnitude_tables_[t - 1], leaves)
            o_hat = sum_o / t
            v_hat = sum_v / t

        F = self._clip(F)
        return (F, o_hat, v_hat) if return_features else F


def _build_tables(leaf_ids, final_score, final_magnitude):
    """Per tree: leaf -> median final score, leaf -> mean final magnitude."""
    score_tables, magnitude_tables = [], []
    for ids in leaf_ids:
        st, mt = {}, {}
        for leaf in np.unique(ids):
            mask = ids == leaf
            st[leaf] = float(np.median(final_score[mask]))
            mt[leaf] = float(np.mean(final_magnitude[mask]))
        score_tables.append(st)
        magnitude_tables.append(mt)
    return score_tables, magnitude_tables


def _lookup(table, leaves):
    """Stored leaf values; a leaf without training rows returns 0."""
    return np.array([table.get(leaf, 0.0) for leaf in leaves])
