import numpy as np
import pytest

from oscboost import OscBoost, GradientBoostedTrees, wmape, mae, rmse


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(600, 5))
    y = np.clip(X[:, 0] ** 3 + 0.3 * X[:, 1] + rng.normal(0, 0.1, 600), 0, 1)
    return X[:400], y[:400], X[400:], y[400:]


PARAMS = dict(n_estimators=60, learning_rate=0.1, subsample=0.8,
              max_depth=3, min_samples_leaf=10, max_features=1.0, random_state=1)


def test_baseline_learns(data):
    Xtr, ytr, Xte, yte = data
    model = GradientBoostedTrees(**PARAMS).fit(Xtr, ytr)
    assert wmape(yte, model.predict(Xte)) < wmape(yte, np.full(len(yte), ytr.mean()))


def test_oscboost_learns_and_is_deterministic(data):
    Xtr, ytr, Xte, yte = data
    p1 = OscBoost(**PARAMS).fit(Xtr, ytr).predict(Xte)
    p2 = OscBoost(**PARAMS).fit(Xtr, ytr).predict(Xte)
    np.testing.assert_array_equal(p1, p2)
    assert wmape(yte, p1) < wmape(yte, np.full(len(yte), ytr.mean()))


def test_training_features_are_bounded(data):
    Xtr, ytr, _, _ = data
    model = OscBoost(**PARAMS).fit(Xtr, ytr)
    assert np.all((model.train_score_ >= 0) & (model.train_score_ < 1))
    # The path excess never decreases, so swings are >= 0 up to rounding.
    assert np.all(model.train_magnitude_ >= -1e-12)
    assert len(model.score_tables_) == len(model.magnitude_tables_) == PARAMS['n_estimators']


def test_feature_importances_include_both_features(data):
    Xtr, ytr, _, _ = data
    model = OscBoost(**PARAMS).fit(Xtr, ytr)
    assert model.feature_importances_.shape == (Xtr.shape[1] + 2,)
    assert np.isclose(model.feature_importances_.sum(), 1.0)


def test_carry_mean_features_and_clipping(data):
    Xtr, ytr, Xte, _ = data
    model = OscBoost(**PARAMS, output_range=(0, 1)).fit(Xtr, ytr)
    pred, o_hat, v_hat = model.predict(Xte, return_features=True)
    assert np.all((pred >= 0) & (pred <= 1))
    assert o_hat.shape == v_hat.shape == (len(Xte),)
    assert np.all(o_hat >= 0) and np.all(v_hat >= -1e-12)


def test_metrics():
    y, p = np.array([1.0, 0.0, 2.0]), np.array([0.5, 0.5, 2.0])
    assert np.isclose(wmape(y, p), 1.0 / 3.0)
    assert np.isclose(mae(y, p), 1.0 / 3.0)
    assert np.isclose(rmse(y, p), np.sqrt(0.5 / 3.0))
