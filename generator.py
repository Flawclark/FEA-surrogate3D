"""
generator.py — build a surrogate for a design space you define at runtime.

The fixed pipeline (sweep.py / train.py) samples one hard-coded parameter box.
This module takes the ranges as an argument instead, runs the sweep, trains a
Gaussian Process per target, and stores the result so the app can load it
later.

A surrogate is only valid inside the box it was trained on, so every generated
model is saved together with the ranges that produced it. The app reads those
ranges back and refuses to predict outside them.
"""

import json
import os
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "models"
MANIFEST = os.path.join(MODEL_DIR, "manifest.json")
SEED = 42
MIN_SAMPLES = 40


# ------------------------------------------------------------- the families

def _solve_2d(p):
    from simulate import simulate
    r = simulate(p["W"], p["H"], p["R"], p["t"])
    return {"max_von_mises": r["max_von_mises"], "max_disp": r["max_disp"]}


def _solve_3d(p):
    from simulate3d import simulate3d
    r = simulate3d(p["tb"], p["tw"], p["rf"], p["Hw"], p["rh"])
    return {"max_vm_fillet": r["max_vm_fillet"], "max_disp": r["max_disp"]}


FAMILIES = {
    "2d_plate": {
        "label": "2D plate with a central hole",
        "params": ["W", "H", "R", "t"],
        "labels": {
            "W": "Plate width W (mm)",
            "H": "Plate height H (mm)",
            "R": "Hole radius R (mm)",
            "t": "Thickness t (mm)",
        },
        # widest values the solver is known to handle
        "limits": {"W": (30.0, 220.0), "H": (30.0, 220.0),
                   "R": (2.0, 60.0), "t": (1.0, 25.0)},
        "defaults": {"W": (50.0, 140.0), "H": (40.0, 140.0),
                     "R": (5.0, 30.0), "t": (2.0, 10.0)},
        "targets": {"max_von_mises": "Peak von Mises (MPa)",
                    "max_disp": "Max displacement (mm)"},
        "solve": _solve_2d,
        "seconds_per_run": 0.2,
    },
    "3d_bracket": {
        "label": "3D L-bracket with a fillet",
        "params": ["tb", "tw", "rf", "Hw", "rh"],
        "labels": {
            "tb": "Base thickness tb (mm)",
            "tw": "Web thickness tw (mm)",
            "rf": "Fillet radius rf (mm)",
            "Hw": "Web height Hw (mm)",
            "rh": "Bolt hole radius rh (mm)",
        },
        "limits": {"tb": (3.0, 20.0), "tw": (3.0, 20.0), "rf": (1.0, 20.0),
                   "Hw": (25.0, 90.0), "rh": (2.0, 11.0)},
        "defaults": {"tb": (5.0, 15.0), "tw": (5.0, 15.0), "rf": (2.0, 16.0),
                     "Hw": (30.0, 70.0), "rh": (3.0, 8.0)},
        "targets": {"max_vm_fillet": "Fillet von Mises (MPa)",
                    "max_disp": "Max displacement (mm)"},
        "solve": _solve_3d,
        "seconds_per_run": 8.0,
    },
}


def estimate_minutes(family_key, n_samples):
    return FAMILIES[family_key]["seconds_per_run"] * n_samples / 60.0


def suggested_samples(family_key):
    """Rule of thumb: ~100 points per input dimension for a smooth response."""
    return 100 * len(FAMILIES[family_key]["params"])


# ------------------------------------------------------------------ manifest

def _load_manifest():
    if not os.path.exists(MANIFEST):
        return []
    try:
        with open(MANIFEST) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(entries):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(entries, fh, indent=2)


def list_surrogates():
    """Generated surrogates, newest first, skipping any with missing files."""
    out = []
    for e in _load_manifest():
        if all(os.path.exists(p) for p in e["files"].values()):
            out.append(e)
    return sorted(out, key=lambda e: e["created"], reverse=True)


def delete_surrogate(sid):
    entries = _load_manifest()
    keep = []
    for e in entries:
        if e["id"] == sid:
            for p in list(e["files"].values()) + [e.get("dataset", "")]:
                if p and os.path.exists(p):
                    os.remove(p)
        else:
            keep.append(e)
    _save_manifest(keep)


# ------------------------------------------------------------------ sampling

def run_sweep(family_key, ranges, n_samples, progress=None, seed=SEED):
    """Latin Hypercube sweep over user-supplied ranges.

    progress(done, total, n_ok, n_failed) is called after every run so a UI
    can show where it is. Returns (DataFrame of successes, failure count,
    list of example failure reasons).
    """
    fam = FAMILIES[family_key]
    names = fam["params"]

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    lo = [ranges[k][0] for k in names]
    hi = [ranges[k][1] for k in names]
    points = qmc.scale(sampler.random(n_samples), lo, hi)

    rows, reasons = [], []
    n_ok = n_failed = 0

    for i, row in enumerate(points):
        p = dict(zip(names, (float(v) for v in row)))
        try:
            rows.append({**p, **fam["solve"](p)})
            n_ok += 1
        except Exception as exc:
            n_failed += 1
            if len(reasons) < 5:
                reasons.append(f"{type(exc).__name__}: {exc}"[:120])
        if progress is not None:
            progress(i + 1, n_samples, n_ok, n_failed)

    return pd.DataFrame(rows), n_failed, reasons


# ------------------------------------------------------------------ training

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rng = float(y_true.max() - y_true.min())
    ape = np.abs((y_pred - y_true) / y_true) * 100.0
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "nrmse_pct": 100.0 * rmse / rng if rng > 0 else float("nan"),
        "median_ape_pct": float(np.median(ape)),
        "worst_ape_pct": float(np.max(ape)),
    }


def _fit_one(df, features, target):
    X = df[features].to_numpy(float)
    y_raw = df[target].to_numpy(float)

    # Targets are positive and right-skewed, so fit in log space and
    # transform back. Errors are always reported in physical units.
    if (y_raw <= 0).any():
        raise ValueError(f"target {target} has non-positive values")
    y = np.log(y_raw)

    X_tr, X_te, y_tr, _, raw_tr, raw_te = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    kernel = (ConstantKernel(1.0)
              * Matern(length_scale=[1.0] * len(features), nu=2.5)
              + WhiteKernel(noise_level=1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=SEED)
    gp.fit(X_tr_s, y_tr)
    gp_metrics = _metrics(raw_te, np.exp(gp.predict(X_te_s)))

    # Linear baseline, kept so the GP result can be judged against something.
    lin = LinearRegression().fit(X_tr_s, y_tr)
    lin_metrics = _metrics(raw_te, np.exp(lin.predict(X_te_s)))

    bundle = {"model": gp, "scaler": scaler, "features": features,
              "target": target, "log_target": True,
              "model_name": "gaussian_process"}
    return bundle, {"gaussian_process": gp_metrics, "linear": lin_metrics}


def generate(family_key, ranges, n_samples, name,
             progress=None, keep_dataset=True, seed=SEED):
    """Sweep, train, and store a surrogate. Returns the manifest entry."""
    fam = FAMILIES[family_key]

    for k, (lo, hi) in ranges.items():
        if hi <= lo:
            raise ValueError(f"{k}: upper bound must exceed lower bound")

    t0 = time.time()
    df, n_failed, reasons = run_sweep(family_key, ranges, n_samples,
                                      progress=progress, seed=seed)

    if len(df) < MIN_SAMPLES:
        raise RuntimeError(
            f"only {len(df)} of {n_samples} runs succeeded — too few to train "
            f"on (need {MIN_SAMPLES}). Most failures were: "
            + ("; ".join(reasons) if reasons else "unknown")
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    sid = uuid.uuid4().hex[:8]

    files, metrics = {}, {}
    for target in fam["targets"]:
        bundle, m = _fit_one(df, fam["params"], target)
        path = os.path.join(MODEL_DIR, f"{sid}_{target}.joblib")
        joblib.dump(bundle, path)
        files[target] = path
        metrics[target] = m

    dataset_path = ""
    if keep_dataset:
        dataset_path = os.path.join(MODEL_DIR, f"{sid}_dataset.csv")
        df.to_csv(dataset_path, index=False)

    entry = {
        "id": sid,
        "name": name.strip() or f"{fam['label']} ({sid})",
        "family": family_key,
        "family_label": fam["label"],
        "params": fam["params"],
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "targets": fam["targets"],
        "n_requested": int(n_samples),
        "n_success": int(len(df)),
        "n_failed": int(n_failed),
        "failure_examples": reasons,
        "metrics": metrics,
        "files": files,
        "dataset": dataset_path,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "build_minutes": round((time.time() - t0) / 60.0, 2),
    }

    entries = _load_manifest()
    entries.append(entry)
    _save_manifest(entries)
    return entry


# ------------------------------------------------- user-defined 2D parts

CUSTOM_TARGETS = {"max_von_mises": "Peak von Mises (MPa)",
                  "max_disp": "Max displacement (mm)"}


def generate_custom(part, ranges, n_samples, name,
                    progress=None, mesh_size=None, seed=SEED):
    """Sweep and train a surrogate for a part the user defined.

    `ranges` maps a subset of the part's parameters to (low, high). Any
    parameter left out is held at its value in the definition — so the
    surrogate's input vector is exactly the swept parameters, in that order.
    """
    import custom2d

    swept = list(ranges)
    if not swept:
        raise ValueError("choose at least one parameter to vary")
    for k, (lo, hi) in ranges.items():
        if hi <= lo:
            raise ValueError(f"{k}: upper bound must exceed lower bound")

    t0 = time.time()
    sampler = qmc.LatinHypercube(d=len(swept), seed=seed)
    lo = [ranges[k][0] for k in swept]
    hi = [ranges[k][1] for k in swept]
    points = qmc.scale(sampler.random(n_samples), lo, hi)

    rows, reasons = [], []
    n_ok = n_failed = 0

    for i, row in enumerate(points):
        p = dict(zip(swept, (float(v) for v in row)))
        try:
            r = custom2d.solve_part(part, params=p, mesh_size=mesh_size)
            rows.append({**p,
                         "max_von_mises": r["max_von_mises"],
                         "max_disp": r["max_disp"]})
            n_ok += 1
        except Exception as exc:
            n_failed += 1
            if len(reasons) < 5:
                reasons.append(f"{type(exc).__name__}: {exc}"[:120])
        if progress is not None:
            progress(i + 1, n_samples, n_ok, n_failed)

    df = pd.DataFrame(rows)
    if len(df) < MIN_SAMPLES:
        raise RuntimeError(
            f"only {len(df)} of {n_samples} runs succeeded — too few to train "
            f"on (need {MIN_SAMPLES}). Most failures were: "
            + ("; ".join(reasons) if reasons else "unknown")
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    sid = uuid.uuid4().hex[:8]

    files, metrics = {}, {}
    for target in CUSTOM_TARGETS:
        bundle, m = _fit_one(df, swept, target)
        path = os.path.join(MODEL_DIR, f"{sid}_{target}.joblib")
        joblib.dump(bundle, path)
        files[target] = path
        metrics[target] = m

    dataset_path = os.path.join(MODEL_DIR, f"{sid}_dataset.csv")
    df.to_csv(dataset_path, index=False)

    entry = {
        "id": sid,
        "name": name.strip() or part.get("name", f"custom part ({sid})"),
        "family": "custom_2d",
        "family_label": "User-defined 2D part",
        "params": swept,
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "targets": CUSTOM_TARGETS,
        "part": part,                      # so the app can redraw and re-solve
        "mesh_size": mesh_size,
        "n_requested": int(n_samples),
        "n_success": int(len(df)),
        "n_failed": int(n_failed),
        "failure_examples": reasons,
        "metrics": metrics,
        "files": files,
        "dataset": dataset_path,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "build_minutes": round((time.time() - t0) / 60.0, 2),
    }

    entries = _load_manifest()
    entries.append(entry)
    _save_manifest(entries)
    return entry


def save_part(part, path):
    with open(path, "w") as fh:
        json.dump(part, fh, indent=2)


def load_part(path_or_text):
    if os.path.exists(str(path_or_text)):
        with open(path_or_text) as fh:
            return json.load(fh)
    return json.loads(path_or_text)


# --------------------------------------------- user-defined 3D (extruded)

CUSTOM3D_TARGETS = {"max_von_mises": "Peak von Mises (MPa)",
                    "max_disp": "Max displacement (mm)"}


def generate_custom3d(part, ranges, n_samples, name,
                      progress=None, mesh_size=None, seed=SEED):
    """Sweep and train a surrogate for a user-defined extruded 3D part.

    Same contract as generate_custom, but each solve costs seconds rather
    than a fraction of one, so sample counts should be chosen accordingly.
    """
    import custom3d

    swept = list(ranges)
    if not swept:
        raise ValueError("choose at least one parameter to vary")
    for k, (lo, hi) in ranges.items():
        if hi <= lo:
            raise ValueError(f"{k}: upper bound must exceed lower bound")

    t0 = time.time()
    sampler = qmc.LatinHypercube(d=len(swept), seed=seed)
    lo = [ranges[k][0] for k in swept]
    hi = [ranges[k][1] for k in swept]
    points = qmc.scale(sampler.random(n_samples), lo, hi)

    rows, reasons = [], []
    n_ok = n_failed = 0

    for i, row in enumerate(points):
        p = dict(zip(swept, (float(v) for v in row)))
        try:
            r = custom3d.solve_part(part, params=p, mesh_size=mesh_size)
            rows.append({**p,
                         "max_von_mises": r["max_von_mises"],
                         "max_disp": r["max_disp"]})
            n_ok += 1
        except Exception as exc:
            n_failed += 1
            if len(reasons) < 5:
                reasons.append(f"{type(exc).__name__}: {exc}"[:120])
        if progress is not None:
            progress(i + 1, n_samples, n_ok, n_failed)

    df = pd.DataFrame(rows)
    if len(df) < MIN_SAMPLES:
        raise RuntimeError(
            f"only {len(df)} of {n_samples} runs succeeded — too few to train "
            f"on (need {MIN_SAMPLES}). Most failures were: "
            + ("; ".join(reasons) if reasons else "unknown")
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    sid = uuid.uuid4().hex[:8]

    files, metrics = {}, {}
    for target in CUSTOM3D_TARGETS:
        bundle, m = _fit_one(df, swept, target)
        path = os.path.join(MODEL_DIR, f"{sid}_{target}.joblib")
        joblib.dump(bundle, path)
        files[target] = path
        metrics[target] = m

    dataset_path = os.path.join(MODEL_DIR, f"{sid}_dataset.csv")
    df.to_csv(dataset_path, index=False)

    entry = {
        "id": sid,
        "name": name.strip() or part.get("name", f"3D part ({sid})"),
        "family": "custom_3d",
        "family_label": "User-defined 3D part (extruded)",
        "params": swept,
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "targets": CUSTOM3D_TARGETS,
        "part": part,
        "mesh_size": mesh_size,
        "n_requested": int(n_samples),
        "n_success": int(len(df)),
        "n_failed": int(n_failed),
        "failure_examples": reasons,
        "metrics": metrics,
        "files": files,
        "dataset": dataset_path,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "build_minutes": round((time.time() - t0) / 60.0, 2),
    }

    entries = _load_manifest()
    entries.append(entry)
    _save_manifest(entries)
    return entry
