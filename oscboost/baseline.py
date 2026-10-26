"""Standard gradient boosting with regression trees (the baseline of the thesis)."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

# Added to the Hessian before the Newton step. Under squared error loss the
# Hessian is 1, so this only reproduces the arithmetic of the study code.
_HESS_EPS = 1e-8


def squared_error_gradients(y_true, y_pred):
    """Gradient and Hessian of the squared error loss 0.5 * (y - F)^2."""
    return y_pred - y_true, np.ones_like(y_true)


class GradientBoostedTrees(BaseEstimator, RegressorMixin):
    """Gradient boosting with squared error loss and regression trees.

    Each iteration fits one tree to the pseudo residuals of a row subsample
    drawn without replacement, and adds it to the ensemble with the
    learning rate.

    Parameters
    ----------
    n_estimators : int
        Number of boosting iterations M.
    learning_rate : float
        Learning rate eta.
    subsample : float
        Fraction rho of training rows drawn for each tree.
    max_depth, min_samples_leaf, max_features
        Passed to ``sklearn.tree.DecisionTreeRegressor``.
    random_state : int
        Seed of the row subsampling and of every tree.
    output_range : tuple of float or None
        If given, predictions are clipped to this range. The thesis uses
        (0, 1) because the target is a capacity factor.
    """

    def __init__(self, n_estimators=500, learning_rate=0.05, subsample=0.8,
                 max_depth=6, min_samples_leaf=20, max_features=0.8,
                 random_state=42, output_range=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.output_range = output_range

    def _make_tree(self):
        return DecisionTreeRegressor(max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     max_features=self.max_features,
                                     random_state=self.random_state)

    def _subsample_index(self, n):
        if self.subsample < 1.0:
            return self.rng_.choice(n, size=int(n * self.subsample), replace=False)
        return np.arange(n)

    def _clip(self, F):
        if self.output_range is None:
            return F
        return np.clip(F, *self.output_range)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self.init_pred_ = float(np.mean(y))
        self.rng_ = np.random.RandomState(self.random_state)
        self.trees_ = []
        F = np.full(n, self.init_pred_)

        for _ in range(self.n_estimators):
            grad, hess = squared_error_gradients(y, F)
            residuals = -grad / (hess + _HESS_EPS)
            idx = self._subsample_index(n)
            tree = self._make_tree()
            tree.fit(X[idx], residuals[idx])
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

        self.feature_importances_ = _gain_importance(self.trees_, X.shape[1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_pred_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return self._clip(F)


def _gain_importance(trees, n_features):
    """Gain importance: per-tree normalized gains, summed and renormalized."""
    total = np.zeros(n_features)
    for tree in trees:
        total[:len(tree.feature_importances_)] += tree.feature_importances_
    s = total.sum()
    return total / s if s > 0 else total
