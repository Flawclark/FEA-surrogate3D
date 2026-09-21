"""
sweep3d.py — build the 3D surrogate training dataset.

Same structure as sweep.py, but each solve takes seconds rather than
milliseconds, so this one writes results incrementally and can resume
if it is interrupted.

Run:  python sweep3d.py
      python sweep3d.py --n 300
"""

import argparse
import csv
import os
import time
import traceback

import numpy as np
from scipy.stats import qmc

from simulate3d import simulate3d

OUT_FILE = "dataset3d.csv"

# Parameter ranges in mm: (low, high)
RANGES = {
    "tb": (5.0, 15.0),    # base plate thickness
    "tw": (5.0, 15.0),    # web thickness
    "rf": (2.0, 16.0),    # fillet radius
    "Hw": (30.0, 70.0),   # web height
    "rh": (3.0, 8.0),     # bolt hole radius
}

FIELDS = ["run_id", "tb", "tw", "rf", "Hw", "rh", "status",
          "max_vm_fillet", "max_vm_global", "max_disp",
          "nominal_bending", "Kt_fillet",
          "n_elements", "n_dofs", "solve_time_s", "error"]


def self_test():
    """One known-good case before the sweep. Stop early if it fails."""
    print("Self-test: simulate3d(8, 8, 6, 50, 4) ...", flush=True)
    try:
        res = simulate3d(8.0, 8.0, 6.0, 50.0, 4.0)
    except Exception:
        print("\n*** SELF-TEST FAILED — stopping before the sweep.\n")
        traceback.print_exc()
        return False
    print(f"  Kt_fillet = {res['Kt_fillet']:.3f}  (expected ~1.08)")
    print(f"  solve time = {res['solve_time_s']} s")
    print("Self-test passed.\n", flush=True)
    return True


def make_samples(n, seed=42):
    names = list(RANGES)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    lower = [RANGES[k][0] for k in names]
    upper = [RANGES[k][1] for k in names]
    scaled = qmc.scale(sampler.random(n), lower, upper)
    return [dict(zip(names, row)) for row in scaled]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=str, default=OUT_FILE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mesh-fraction", type=float, default=3.0)
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_self_test and not self_test():
        return

    samples = make_samples(args.n, args.seed)

    # Resume: 3D sweeps run for hours, so an interrupted run should not
    # start over. Completed run_ids are read back from the CSV.
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(int(row["run_id"]))
        print(f"resuming — {len(done)} runs already recorded\n", flush=True)

    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    n_ok = n_invalid = n_unexpected = 0
    t_start = time.time()

    for i, s in enumerate(samples):
        if i in done:
            continue

        row = {"run_id": i, **{k: round(v, 3) for k, v in s.items()},
               "status": "", "error": ""}
        t0 = time.time()

        try:
            res = simulate3d(s["tb"], s["tw"], s["rf"], s["Hw"], s["rh"],
                             mesh_fraction=args.mesh_fraction)
            row.update({
                "status": "ok",
                "max_vm_fillet": round(res["max_vm_fillet"], 3),
                "max_vm_global": round(res["max_vm_global"], 3),
                "max_disp": round(res["max_disp"], 6),
                "nominal_bending": round(res["nominal_bending"], 3),
                "Kt_fillet": round(res["Kt_fillet"], 4),
                "n_elements": res["n_elements"],
                "n_dofs": res["n_dofs"],
            })
            n_ok += 1

        except ValueError as exc:
            # Expected: geometrically invalid combination.
            row["status"] = "invalid_geometry"
            row["error"] = str(exc)[:150]
            n_invalid += 1

        except Exception as exc:
            # Meshing or solver failure — real in 3D, unlike 2D.
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"[:150]
            n_unexpected += 1
            if n_unexpected <= 3:
                print(f"\n--- unexpected error on run {i}: {s} ---", flush=True)
                traceback.print_exc()

        row["solve_time_s"] = round(time.time() - t0, 2)
        writer.writerow(row)
        fh.flush()

        n_done = i + 1 - len(done)
        if n_done > 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            remaining = (len(samples) - i - 1) / max(rate, 1e-9)
            print(f"{i+1:>4}/{len(samples)}   ok={n_ok}  invalid={n_invalid}  "
                  f"errors={n_unexpected}   {elapsed/60:.1f} min elapsed, "
                  f"~{remaining/60:.0f} min left", flush=True)

    fh.close()

    total = n_ok + n_invalid + n_unexpected
    print("\n===== 3D sweep finished =====")
    print(f"successful           : {n_ok}")
    print(f"invalid geometry     : {n_invalid}")
    print(f"unexpected errors    : {n_unexpected}")
    if total:
        print(f"invalid rate         : {100 * n_invalid / total:.1f}%")
        print(f"error rate           : {100 * n_unexpected / total:.1f}%")
    print(f"wall time            : {(time.time() - t_start) / 60:.1f} min")
    print(f"written to           : {args.out}")


if __name__ == "__main__":
    main()
