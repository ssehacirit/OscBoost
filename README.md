# OscBoost

OscBoost is a modification of gradient boosted regression trees. It records how the pseudo residual of each training sample changes sign from one boosting iteration to the next. It gives this history to the later trees as two extra input features.

This repository contains the source code of the method from the MSc thesis *Sequential Feature Embeddings in Gradient Boosting Ensembles for Wind Power Forecasting* (Selahattin Seha Cirit, Boğaziçi University, 2026). It contains the method and the baseline only. It does not contain the data, the experiment scripts or the plotting code. 
Thesis supervisor: Assoc. Prof. Mustafa Gökçe Baydoğan"

## Method

Standard gradient boosting discards the residual history of a sample after each iteration. OscBoost keeps two counts for every training sample i:

- **Oscillation count S_i.** It grows by 1 when the pseudo residual changes sign. A sign change is counted only when two consecutive residuals have strictly opposite signs.
- **Magnitude count V_i.** At each sign change it grows by the swing η(E_i − E_i^last). E_i is the path excess of the raw tree outputs u of the sample, E = Σ|u| − |Σu|. E_i^last is its value at the last sign change.

Tree T_m is fitted on the augmented input z_i = [x_i, S_i/m, V_i/m], where the counts are taken from before iteration m.

Both features depend on the target, so they are not available for a new input. After training, each tree stores two lookup tables over its leaves:

- the median of the final score S/M of the training rows in each leaf;
- the mean of their final magnitude V/M.

At inference a new input passes through the trees in order. Each tree reads the stored values of the leaf that the input reaches. The features given to the next tree are the running means of all values read so far. The thesis calls this the **carry mean**.

OscBoost adds no hyperparameter to gradient boosting.

## Installation

```bash
pip install -r requirements.txt
pip install .
```

Python 3.10 or newer is required. The thesis used Python 3.12, NumPy 2.5 and scikit-learn 1.9.

## Usage

```python
from oscboost import OscBoost, GradientBoostedTrees, wmape

model = OscBoost(n_estimators=1000, learning_rate=0.01, subsample=0.7,
                 max_depth=5, min_samples_leaf=150, max_features=0.75,
                 random_state=42, output_range=(0, 1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)                 # carry mean inference
print(wmape(y_test, y_pred))

# Estimated features after the last tree
y_pred, score_hat, magnitude_hat = model.predict(X_test, return_features=True)

baseline = GradientBoostedTrees(n_estimators=1000, learning_rate=0.01,
                                random_state=42, output_range=(0, 1))
baseline.fit(X_train, y_train)
```

Both models use squared error loss and the regression tree of scikit-learn. The parameters are:

| Parameter | Meaning |
|---|---|
| `n_estimators` | number of boosting iterations M |
| `learning_rate` | learning rate η |
| `subsample` | fraction of rows drawn without replacement for each tree |
| `max_depth`, `min_samples_leaf`, `max_features` | tree parameters (for OscBoost `max_features` is a fraction of p + 2 columns) |
| `random_state` | seed of the row subsampling and of every tree |
| `output_range` | optional clipping range of the predictions; the thesis uses (0, 1) |

After fitting, `feature_importances_` gives the gain importance. For OscBoost the last two entries belong to the oscillation score and the oscillation magnitude.

## Tests

```bash
pip install pytest
pytest tests
```

## Citation

If you use this code, please cite the thesis:

```
S.S. Cirit, Sequential Feature Embeddings in Gradient Boosting Ensembles for Wind Power Forecasting,
MSc Thesis, Boğaziçi University, Istanbul, 2026.
```

## License

MIT License. See [LICENSE](LICENSE).
