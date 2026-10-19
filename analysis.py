"""Baseline MLP vs CV-tuned LightGBM (+ sin/cos features) on OpenML kin8nm.
kin8nm is a public stand-in for sensor-based force regression (the paper's data is not public)."""
import json
import os
import sys
import time
from io import BytesIO

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import requests
from scipy.io import arff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # model-randomness seeds, used for both models
DATA_ID = 507                 # OpenML kin8nm
SOURCE_URL = "https://www.openml.org/d/507"
MAX_ROWS = 200_000            # subsample guard (kin8nm is far smaller)


def get_with_retry(url, retries=5):
    """HTTP GET with timeout=60; exponential backoff on HTTP 429/5xx and network errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException as e:
            err = repr(e)
        else:
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()  # other 4xx: fail fast, no retry
                return r
        print(f"  attempt {attempt + 1} failed ({err}); backing off")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {err}")


def load_data():
    """Metadata via the openml package; the ARFF file itself via requests (timeout + retry)."""
    ds = openml.datasets.get_dataset(DATA_ID, download_data=False, download_qualities=False)
    data, _ = arff.loadarff(BytesIO(get_with_retry(ds.url).content))
    df = pd.DataFrame(data).astype(float)
    target = ds.default_target_attribute or df.columns[-1]
    if len(df) > MAX_ROWS:  # subsample very large data (not triggered here)
        df = df.sample(MAX_ROWS, random_state=SEED)
    y = df.pop(target).to_numpy()
    return df.to_numpy(), y, target


def main():
    print("Loading kin8nm from OpenML ...")
    X, y, target = load_data()
    print(f"  rows={len(X)} features={X.shape[1]} target={target}")

    # One fixed train/test split; all scalers are fit on training data only
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    Xtr_s, Xte_s = sx.transform(X_tr), sx.transform(X_te)   # baseline inputs
    ytr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()       # scaled target for both models

    # Improved inputs: raw joint angles plus sine/cosine of each (trees are scale-invariant)
    def add_trig(A):
        return np.hstack([A, np.sin(A), np.cos(A)])
    Xtr_f, Xte_f = add_trig(X_tr), add_trig(X_te)

    # Small hyperparameter grid, selected by 5-fold CV on the training set only
    print("Tuning LightGBM with 5-fold CV on training set ...")
    grid = {"num_leaves": [15, 31], "learning_rate": [0.03, 0.1],
            "n_estimators": [300, 800], "min_child_samples": [10, 30]}
    search = GridSearchCV(
        lgb.LGBMRegressor(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=SEED, verbose=-1),
        grid, cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_mean_absolute_error", n_jobs=1)
    search.fit(Xtr_f, ytr_s)
    best = search.best_params_
    print(f"  best params={best}, CV MAE (scaled target)={-search.best_score_:.4f}")

    def evaluate(pred_s):
        """Same evaluation for both models; metrics in original target units."""
        pred = sy.inverse_transform(np.asarray(pred_s).reshape(-1, 1)).ravel()
        err = np.abs(y_te - pred)
        return {"mae": float(mean_absolute_error(y_te, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                "r2": float(r2_score(y_te, pred)),
                "err_std": float(err.std())}  # spread of absolute error: uniformity proxy

    base, impr = [], []
    for s in SEEDS:
        print(f"Seed {s}: baseline MLP ...")
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000,
                           early_stopping=True, random_state=s)
        mlp.fit(Xtr_s, ytr_s)
        base.append(evaluate(mlp.predict(Xte_s)))

        print(f"Seed {s}: improved LightGBM ...")
        gbm = lgb.LGBMRegressor(**best, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, random_state=s, verbose=-1)
        gbm.fit(Xtr_f, ytr_s)
        impr.append(evaluate(gbm.predict(Xte_f)))

    def agg(runs, key):
        """Mean and sample std across seeds."""
        vals = np.array([r[key] for r in runs])
        return float(vals.mean()), float(vals.std(ddof=1))

    b_mae, b_sd = agg(base, "mae")
    i_mae, i_sd = agg(impr, "mae")
    metrics = {
        "baseline_score": b_mae, "improved_score": i_mae,
        "improvement": b_mae - i_mae,  # MAE is lower-is-better: baseline - improved
        "baseline_score_std": b_sd, "improved_score_std": i_sd,
        "baseline_rmse": agg(base, "rmse")[0], "improved_rmse": agg(impr, "rmse")[0],
        "baseline_r2": agg(base, "r2")[0], "improved_r2": agg(impr, "r2")[0],
        "baseline_abs_err_std": agg(base, "err_std")[0],
        "improved_abs_err_std": agg(impr, "err_std")[0],
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("Non-finite metrics; cannot produce meaningful results.")
        sys.exit(1)

    # Bar chart of test MAE, baseline vs improved
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline MLP", "Improved LightGBM"], [b_mae, i_mae],
           yerr=[b_sd, i_sd], capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel("Test MAE (original units)")
    ax.set_title("kin8nm test MAE (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig("figures/mae_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": ("Can a cross-validated, gradient-boosted regressor give lower and more "
                     "uniform error than a standard MLP baseline when estimating a continuous "
                     "physical quantity from robot-arm sensor readings on a public stand-in dataset?"),
        "method": ("OpenML kin8nm, one 80/20 split (fixed seed), inputs standardised on training "
                   "data. Baseline: sklearn MLPRegressor. Improved: LightGBM with hyperparameters "
                   "chosen by 5-fold CV on the training set plus sine/cosine features. Test-set "
                   "MAE is the main metric; mean over three seeds."),
        "dataset": {
            "name": "OpenML kin8nm (robot-arm forward kinematics)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "MAE",
        "higher_is_better": False,
        "baseline": ("scikit-learn MLPRegressor, two hidden layers of 64 units, early stopping, "
                     "standardised inputs and target; stands in for the paper's CNN-RNN."),
        "improved": ("LightGBM regressor with grid-searched hyperparameters (5-fold CV on the "
                     "training set only) and sine/cosine features of the eight inputs."),
        "metrics": metrics,
        "notes": ("kin8nm is a simulated robot-arm dataset, not contact-force data; it is a "
                  "stand-in only. The paper's neuromorphic-camera CRNN was not reproduced; the "
                  "MLP is a generic baseline. Single train/test split; seeds vary model "
                  "randomness only. Hyperparameters chosen by CV on training data; test set used "
                  "once. Secondary metrics are RMSE, R^2 and the std of absolute error (uniformity "
                  "proxy); classification F1 does not apply to regression."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json and figures/mae_comparison.png")


if __name__ == "__main__":
    main()
