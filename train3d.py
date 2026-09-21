"""
train3d.py — train and compare surrogate models on dataset3d.csv.

Predicts fillet stress and tip displacement of the L-bracket from its five
geometry parameters. Reports accuracy on a held-out test set, runs a
permutation sensitivity study, and writes diagnostic plots.

Run:  python train3d.py
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["tb", "tw", "rf", "Hw", "rh"]
TARGETS = ["max_vm_fillet", "max_disp"]
TEST_FRACTION = 0.2
SEED = 42


def load_data(path="dataset3d.csv"):
    df = pd.read_csv(path)
    ok = df[df.status == "ok"].copy()
    ok = ok.dropna(subset=FEATURES + TARGETS)
    print(f"loaded {len(df)} rows, {len(ok)} usable\n")
    if len(ok) < 40:
        raise SystemExit("too few successful runs to train on")
    return ok


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rng = float(y_true.max() - y_true.min())
    ape = np.abs((y_pred - y_true) / y_true) * 100.0
    return {
        "rmse": rmse,
        "nrmse_pct": 100.0 * rmse / rng,
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(ape)),
        "median_ape_pct": float(np.median(ape)),
        "worst_ape_pct": float(np.max(ape)),
    }


def build_models():
    kernel = (ConstantKernel(1.0)
              * Matern(length_scale=[1.0] * len(FEATURES), nu=2.5)
              + WhiteKernel(noise_level=1e-3))
    return {
        "linear": LinearRegression(),
        "gaussian_process": GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
            random_state=SEED),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            random_state=SEED),
    }


def train_target(df, target):
    X = df[FEATURES].to_numpy(float)
    y_raw = df[target].to_numpy(float)
    y = np.log(y_raw)          # both targets are positive and right-skewed

    X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        X, y, y_raw, test_size=TEST_FRACTION, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"=== {target} ===")
    print(f"train {len(X_tr)}  test {len(X_te)}   "
          f"range {raw_tr.min():.4g} to {raw_tr.max():.4g}\n")

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(X_tr_s, y_tr)
        pred = np.exp(model.predict(X_te_s))
        m = metrics(raw_te, pred)
        results[name] = m
        fitted[name] = (model, scaler, pred)
        print(f"{name:>19}   R2={m['r2']:.4f}   "
              f"nRMSE={m['nrmse_pct']:.2f}%   "
              f"median err={m['median_ape_pct']:.2f}%   "
              f"worst={m['worst_ape_pct']:.1f}%")

    best = max(results, key=lambda k: results[k]["r2"])
    print(f"\nbest: {best}")

    # Which parameters actually matter? Permutation importance measures the
    # accuracy lost when one feature is shuffled. In 2D every parameter was
    # load-bearing; here the bolt holes sit far from the fillet, so this is
    # worth checking rather than assuming.
    model, _, _ = fitted[best]
    imp = permutation_importance(model, X_te_s, y_te, n_repeats=20,
                                 random_state=SEED, scoring="r2")
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nsensitivity (permutation importance, R2 lost when shuffled):")
    for i in order:
        print(f"  {FEATURES[i]:>4}  {imp.importances_mean[i]:>7.4f} "
              f"± {imp.importances_std[i]:.4f}")
    print()

    sens = {FEATURES[i]: float(imp.importances_mean[i]) for i in order}
    return results, fitted, raw_te, best, sens


def plot_target(target, fitted, raw_te, best, sens):
    _, _, pred = fitted[best]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    lo, hi = raw_te.min(), raw_te.max()
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    ax[0].scatter(raw_te, pred, s=20, alpha=0.75)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(f"FEA {target}")
    ax[0].set_ylabel(f"predicted {target}")
    ax[0].set_title(f"{best}: predicted vs FEA")
    ax[0].legend()

    err = (pred - raw_te) / raw_te * 100.0
    ax[1].hist(err, bins=25)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("relative error (%)")
    ax[1].set_ylabel("count")
    ax[1].set_title("error distribution")

    names = list(sens.keys())
    vals = [sens[n] for n in names]
    ax[2].barh(names[::-1], vals[::-1])
    ax[2].set_xlabel("R² lost when shuffled")
    ax[2].set_title("parameter sensitivity")

    fig.tight_layout()
    out = f"results3d_{target}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    df = load_data()
    all_results = {}

    for target in TARGETS:
        results, fitted, raw_te, best, sens = train_target(df, target)
        all_results[target] = {"metrics": results, "sensitivity": sens}
        plot_target(target, fitted, raw_te, best, sens)

        model, scaler, _ = fitted[best]
        joblib.dump({"model": model, "scaler": scaler,
                     "features": FEATURES, "target": target,
                     "log_target": True, "model_name": best},
                    f"surrogate3d_{target}.joblib")
        print(f"saved surrogate3d_{target}.joblib\n")

    with open("metrics3d.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print("wrote metrics3d.json")


if __name__ == "__main__":
    main()
