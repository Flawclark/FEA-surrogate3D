"""
app.py — FEA surrogate demonstrator, generator and part editor.

Three modes:
  Use a surrogate     sliders, instant prediction, FEA verification, heat map
  Define a part       build your own 2D geometry, loads and supports
  Create a surrogate  choose a design space, sweep it, train a model on it

The built-in surrogates cover two fixed parts. The part editor lets you define
your own 2D part, and the generator will train a surrogate on whichever of its
parameters you choose to vary.

Run:  streamlit run app.py
"""

import copy
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import custom2d
import custom3d
import generator
import plots

# --------------------------------------------------------- built-in models

BUILTIN = {
    "2D plate with a hole": {
        "key": "2d",
        "features": ["W", "H", "R", "t"],
        "ranges": {"W": (50.0, 140.0), "H": (40.0, 140.0),
                   "R": (5.0, 30.0), "t": (2.0, 10.0)},
        "steps": {"W": 1.0, "H": 1.0, "R": 0.5, "t": 0.1},
        "models": {"stress": "surrogate_max_von_mises.joblib",
                   "disp": "surrogate_max_disp.joblib"},
        "stress_label": "Peak von Mises",
        "load_note": "Load: 10 kN tension along x.  Steel, E = 210 GPa.",
        "fea_note": "The 2D solve takes about 0.15 s.",
    },
    "3D L-bracket with a fillet": {
        "key": "3d",
        "features": ["tb", "tw", "rf", "Hw", "rh"],
        "ranges": {"tb": (5.0, 15.0), "tw": (5.0, 15.0), "rf": (2.0, 16.0),
                   "Hw": (30.0, 70.0), "rh": (3.0, 8.0)},
        "steps": {"tb": 0.5, "tw": 0.5, "rf": 0.5, "Hw": 1.0, "rh": 0.25},
        "models": {"stress": "surrogate3d_max_vm_fillet.joblib",
                   "disp": "surrogate3d_max_disp.joblib"},
        "stress_label": "Fillet von Mises",
        "load_note": "Load: 1 kN at the top of the web.  Base fully clamped. "
                     "Steel, E = 210 GPa.",
        "fea_note": "The 3D solve takes about 8 s — the whole point of the "
                    "surrogate.",
    },
}

FAMILY_KEY = {"2d_plate": "2d", "3d_bracket": "3d",
              "custom_2d": "custom", "custom_3d": "custom3d"}


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


def predict(bundle, params):
    x = np.array([[params[f] for f in bundle["features"]]], dtype=float)
    xs = bundle["scaler"].transform(x)
    model = bundle["model"]
    if bundle["model_name"] == "gaussian_process":
        # fitted on log(target): the interval is asymmetric in physical units
        mu, sd = model.predict(xs, return_std=True)
        mu, sd = float(mu[0]), float(sd[0])
        return np.exp(mu), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd)
    mu = float(model.predict(xs)[0])
    return (np.exp(mu) if bundle["log_target"] else mu), None, None


def validity(case_key, p, part=None):
    if case_key == "custom3d":
        probs = custom3d.validate(part, p)
        return "; ".join(probs) if probs else None
    if case_key == "custom":
        probs = custom2d.validate(part, p)
        return "; ".join(probs) if probs else None
    if case_key == "2d":
        if p["R"] >= 0.45 * min(p["W"], p["H"]):
            return (f"A hole of radius {p['R']:.1f} mm does not fit in a "
                    f"{p['W']:.0f} x {p['H']:.0f} mm plate.")
    else:
        if p["rf"] > p["Hw"] - p["tb"] - 4.0:
            return "Fillet radius is too large for this web height."
        if p["rf"] > 80.0 - p["tw"] - 2 * p["rh"] - 20.0:
            return "Fillet radius is too large for the base length."
        if p["rh"] > 60.0 / 4.0 - 3.0:
            return "Bolt hole radius is too large for the bracket width."
    return None


def run_fea(case_key, p, part=None, mesh_size=None):
    if case_key == "custom3d":
        r = custom3d.solve_part(part, params=p, mesh_size=mesh_size,
                                return_field=True)
        return r["max_von_mises"], r["max_disp"], r["field"]
    if case_key == "custom":
        r = custom2d.solve_part(part, params=p, mesh_size=mesh_size,
                                return_field=True)
        return r["max_von_mises"], r["max_disp"], r["field"]
    if case_key == "2d":
        from simulate import simulate
        r = simulate(p["W"], p["H"], p["R"], p["t"], return_field=True)
        return r["max_von_mises"], r["max_disp"], r["field"]
    from simulate3d import simulate3d
    r = simulate3d(p["tb"], p["tw"], p["rf"], p["Hw"], p["rh"],
                   return_field=True)
    return r["max_vm_fillet"], r["max_disp"], r["field"]


def draw_geometry(case_key, p, part=None):
    if case_key == "custom3d":
        return plots.geometry_custom3d(part, p)
    if case_key == "custom":
        return plots.geometry_custom(part, p)
    if case_key == "2d":
        return plots.geometry_2d(p["W"], p["H"], p["R"])
    return plots.geometry_3d(p["tb"], p["tw"], p["rf"], p["Hw"], p["rh"])


# ------------------------------------------------- part <-> editable tables

def part_to_tables(part):
    prm = pd.DataFrame([{"parameter": k, "value": float(v)}
                        for k, v in part["parameters"].items()])
    out = pd.DataFrame([{"point": i, "x": x, "y": y}
                        for i, (x, y) in enumerate(part["outline"])])
    holes = pd.DataFrame(
        [{"x": h["x"], "y": h["y"], "r": h["r"]}
         for h in part.get("holes", [])]
        or [{"x": "", "y": "", "r": ""}][:0])
    segs = {int(k): v for k, v in part.get("segments", {}).items()}
    seg_rows = []
    for i in range(len(part["outline"])):
        s = segs.get(i, {"kind": "free"})
        seg_rows.append({
            "segment": i,
            "condition": s.get("kind", "free"),
            "fix": s.get("dof", "xy"),
            "Fx (N)": float(s.get("fx", 0.0)),
            "Fy (N)": float(s.get("fy", 0.0)),
        })
    return prm, out, holes, pd.DataFrame(seg_rows)


def tables_to_part(name, prm, out, holes, segs, thickness):
    part = {
        "name": name,
        "parameters": {str(r["parameter"]).strip(): float(r["value"])
                       for _, r in prm.iterrows()
                       if str(r["parameter"]).strip()},
        "outline": [[str(r["x"]).strip(), str(r["y"]).strip()]
                    for _, r in out.iterrows()
                    if str(r["x"]).strip() and str(r["y"]).strip()],
        "holes": [{"x": str(r["x"]).strip(), "y": str(r["y"]).strip(),
                   "r": str(r["r"]).strip()}
                  for _, r in holes.iterrows()
                  if str(r["x"]).strip() and str(r["r"]).strip()],
        "thickness": thickness,
        "segments": {},
    }
    for _, r in segs.iterrows():
        i = int(r["segment"])
        kind = str(r["condition"])
        if kind == "fixed":
            part["segments"][str(i)] = {"kind": "fixed", "dof": str(r["fix"])}
        elif kind == "load":
            part["segments"][str(i)] = {"kind": "load",
                                        "fx": float(r["Fx (N)"]),
                                        "fy": float(r["Fy (N)"])}
        else:
            part["segments"][str(i)] = {"kind": "free"}
    return part


def faces_to_table(part):
    rows = []
    faces = part.get("faces", {})
    for name in custom3d.face_names(part):
        f = faces.get(name, {"kind": "free"})
        rows.append({
            "face": name,
            "condition": f.get("kind", "free"),
            "fix": f.get("dof", "xyz"),
            "Fx (N)": float(f.get("fx", 0.0)),
            "Fy (N)": float(f.get("fy", 0.0)),
            "Fz (N)": float(f.get("fz", 0.0)),
        })
    return pd.DataFrame(rows)


def tables_to_part3d(name, prm, out, holes, faces_df, depth):
    part = {
        "name": name,
        "kind": "extruded",
        "parameters": {str(r["parameter"]).strip(): float(r["value"])
                       for _, r in prm.iterrows()
                       if str(r["parameter"]).strip()},
        "outline": [[str(r["x"]).strip(), str(r["y"]).strip()]
                    for _, r in out.iterrows()
                    if str(r["x"]).strip() and str(r["y"]).strip()],
        "holes": [{"x": str(r["x"]).strip(), "y": str(r["y"]).strip(),
                   "r": str(r["r"]).strip()}
                  for _, r in holes.iterrows()
                  if str(r["x"]).strip() and str(r["r"]).strip()],
        "depth": depth,
        "faces": {},
    }
    for _, r in faces_df.iterrows():
        nm = str(r["face"])
        kind = str(r["condition"])
        if kind == "fixed":
            part["faces"][nm] = {"kind": "fixed", "dof": str(r["fix"])}
        elif kind == "load":
            part["faces"][nm] = {"kind": "load", "fx": float(r["Fx (N)"]),
                                 "fy": float(r["Fy (N)"]),
                                 "fz": float(r["Fz (N)"])}
    return part


# ================================================================== page

st.set_page_config(page_title="FEA Surrogate", layout="wide")
st.title("FEA surrogate")

mode = st.sidebar.radio(
    "Mode", ["Use a surrogate", "Define a part", "Create a surrogate"])
st.sidebar.divider()


# ------------------------------------------------------- MODE 1: use

if mode == "Use a surrogate":
    generated = generator.list_surrogates()
    available = [n for n, c in BUILTIN.items()
                 if all(os.path.exists(p) for p in c["models"].values())]
    missing_builtins = [n for n in BUILTIN if n not in available]
    choices = available + [f"★ {g['name']}" for g in generated]

    if not choices:
        st.warning("No surrogates available yet. Run `python train.py` / "
                   "`python train3d.py`, or build one in **Create a "
                   "surrogate**.")
        st.stop()

    choice = st.sidebar.selectbox("Surrogate", choices)
    if missing_builtins:
        st.sidebar.caption("Not listed (never trained here): "
                           + ", ".join(missing_builtins))

    part = None
    mesh_size = None
    if choice.startswith("★ "):
        entry = generated[choices.index(choice) - len(available)]
        case_key = FAMILY_KEY[entry["family"]]
        features = entry["params"]
        ranges = {k: tuple(v) for k, v in entry["ranges"].items()}
        stress_target = [t for t in entry["files"] if t != "max_disp"][0]
        stress_label = entry["targets"][stress_target]
        paths = {"stress": entry["files"][stress_target],
                 "disp": entry["files"]["max_disp"]}
        steps = {k: (v[1] - v[0]) / 100.0 for k, v in ranges.items()}

        if case_key in ("custom", "custom3d"):
            part = entry["part"]
            mesh_size = entry.get("mesh_size")
            labels = {k: f"{k} (mm)" for k in features}
            load_note = "Loads and supports are stored in the part definition."
            fea_note = ("The 3D solve takes several seconds."
                        if case_key == "custom3d"
                        else "The 2D solve takes about half a second.")
        else:
            fam = "2d_plate" if case_key == "2d" else "3d_bracket"
            labels = generator.FAMILIES[fam]["labels"]
            b = BUILTIN["2D plate with a hole" if case_key == "2d"
                        else "3D L-bracket with a fillet"]
            load_note, fea_note = b["load_note"], b["fea_note"]
            steps = b["steps"]

        st.sidebar.caption(f"Generated {entry['created']} from "
                           f"{entry['n_success']} simulations "
                           f"({entry['n_failed']} failed).")
    else:
        case = BUILTIN[choice]
        case_key = case["key"]
        features = case["features"]
        ranges = case["ranges"]
        steps = case["steps"]
        stress_label = case["stress_label"]
        load_note, fea_note = case["load_note"], case["fea_note"]
        paths = case["models"]
        fam = "2d_plate" if case_key == "2d" else "3d_bracket"
        labels = generator.FAMILIES[fam]["labels"]

    st.sidebar.caption(load_note)

    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        st.error("Model files not found: " + ", ".join(missing))
        st.stop()

    bundles = {k: load_bundle(v) for k, v in paths.items()}
    st.caption(f"{choice.lstrip('★ ')} — a Gaussian Process trained on finite "
               "element simulations. Predictions take well under a "
               "millisecond.")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Geometry")
        params = {}
        for f in features:
            lo, hi = ranges[f]
            params[f] = st.slider(labels.get(f, f), float(lo), float(hi),
                                  float((lo + hi) / 2),
                                  float(steps.get(f, (hi - lo) / 100)))
        if part is not None:
            full = dict(part["parameters"]); full.update(params)
            params_draw = full
        else:
            params_draw = params

        st.subheader("Geometry preview")
        try:
            st.pyplot(draw_geometry(case_key, params_draw, part),
                      clear_figure=True)
        except Exception as exc:
            st.warning(f"Cannot draw this geometry: {exc}")

    problem = validity(case_key, params_draw if part else params, part)

    with right:
        if problem:
            st.error(problem)
        else:
            t0 = time.perf_counter()
            stress, s_lo, s_hi = predict(bundles["stress"], params)
            disp, d_lo, d_hi = predict(bundles["disp"], params)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            st.subheader("Prediction")
            c1, c2 = st.columns(2)
            c1.metric(stress_label, f"{stress:.1f} MPa")
            c2.metric("Max displacement", f"{disp:.4f} mm")
            if s_lo is not None:
                st.write(f"95% confidence: **{s_lo:.1f} – {s_hi:.1f} MPa** for "
                         f"stress, **{d_lo:.4f} – {d_hi:.4f} mm** for "
                         "displacement.")
            st.caption(f"Predicted in {dt_ms:.2f} ms.")

            factor = 250.0 / stress
            if factor < 1.0:
                st.error(f"Predicted stress exceeds yield (250 MPa). "
                         f"Utilisation {1/factor:.2f}x.")
            elif factor < 1.5:
                st.warning(f"Safety factor {factor:.2f} — marginal.")
            else:
                st.success(f"Safety factor {factor:.2f} against yield.")

            st.divider()
            st.subheader("Verify against FEA")
            st.caption(fea_note)

            if st.button("Run FEA now"):
                try:
                    with st.spinner("Meshing and solving..."):
                        t0 = time.perf_counter()
                        fea_s, fea_d, field = run_fea(
                            case_key, params_draw if part else params,
                            part, mesh_size)
                        fea_ms = (time.perf_counter() - t0) * 1000.0

                    a, b = st.columns(2)
                    a.metric(f"FEA {stress_label.lower()}", f"{fea_s:.1f} MPa",
                             f"{100*(stress-fea_s)/fea_s:+.2f}% surrogate error")
                    b.metric("FEA max displacement", f"{fea_d:.4f} mm",
                             f"{100*(disp-fea_d)/fea_d:+.2f}% surrogate error")
                    st.caption(f"FEA took {fea_ms:.0f} ms against "
                               f"{dt_ms:.2f} ms — a {fea_ms/dt_ms:,.0f}x "
                               "speedup.")

                    st.subheader("Stress field")
                    st.caption("The surrogate predicts a single number. The "
                               "full field comes only from the solve.")
                    if case_key in ("3d", "custom3d"):
                        st.pyplot(plots.heatmap_3d(field), clear_figure=True)
                    else:
                        st.pyplot(plots.heatmap_2d(field), clear_figure=True)
                except Exception as exc:
                    st.exception(exc)

    st.divider()
    rng = ",  ".join(f"{k} {v[0]:g}–{v[1]:g}" for k, v in ranges.items())
    st.markdown(f"**Limitations.** Valid only for this part and for {rng} mm. "
                "Linear elastic, single load case, small displacements. The "
                "surrogate approximates **the simulation**, not physical "
                "reality. For design exploration, not certification.")


# -------------------------------------------------- MODE 2: define a part

elif mode == "Define a part":
    st.subheader("Define a part")
    st.write(
        "Build a part from an outline, holes, loads and supports. Coordinates "
        "are **expressions in the parameters**, so one definition describes a "
        "whole family of shapes — which is what makes it possible to train a "
        "surrogate on it afterwards."
    )

    dim = st.radio("Part type", ["2D (plane stress)", "3D (extruded profile)"],
                   horizontal=True, key="dim_choice")
    is3d = dim.startswith("3D")

    if "part" not in st.session_state:
        st.session_state.part = copy.deepcopy(
            custom2d.PRESETS["Plate with a central hole"])
    # Streamlit remembers data_editor contents by widget key. Bump this when a
    # preset or file is loaded so the tables are rebuilt instead of the stale
    # edits being reapplied on top of the new part.
    if "part_version" not in st.session_state:
        st.session_state.part_version = 0

    preset_source = custom3d.PRESETS if is3d else custom2d.PRESETS
    store = "part3d" if is3d else "part"
    if store not in st.session_state:
        st.session_state[store] = copy.deepcopy(
            list(preset_source.values())[0])

    c1, c2 = st.columns([2, 1])
    preset = c1.selectbox("Start from", list(preset_source))
    if c2.button("Load preset"):
        st.session_state[store] = copy.deepcopy(preset_source[preset])
        st.session_state.part_version += 1
        st.rerun()

    uploaded = st.file_uploader("…or load a saved part (.json)", type="json")
    if uploaded is not None:
        # Streamlit keeps the uploaded file across reruns, so this block runs
        # on every interaction. Without a marker it would reload the file over
        # the user's edits every time and look like the editor resetting.
        stamp = (uploaded.name, uploaded.size, store)
        if st.session_state.get("last_upload") != stamp:
            st.session_state.last_upload = stamp
            try:
                uploaded.seek(0)
                st.session_state[store] = json.load(uploaded)
                st.session_state.part_version += 1
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")

    part = st.session_state[store]
    prm_df, out_df, hole_df, seg_df = part_to_tables(part)
    face_df = faces_to_table(part) if is3d else None

    edit, view = st.columns([1, 1])

    with edit:
        name = st.text_input("Part name", part.get("name", "my part"),
                             key=f"pname_{st.session_state.part_version}")

        st.markdown("**Parameters** — named values you can reference below")
        prm_e = st.data_editor(prm_df, num_rows="dynamic", hide_index=True,
                               key=f"prm_{st.session_state.part_version}", use_container_width=True)

        st.markdown(
            "**Outline** — corner points, counter-clockwise. Each cell is an "
            "expression: a number (`40`), a parameter name (`W`), or "
            "arithmetic on them (`H/2`, `W-a`). Do not write `x=` or `W=100` "
            "— parameter values go in the table above.")
        out_e = st.data_editor(out_df, num_rows="dynamic", hide_index=True,
                               key=f"out_{st.session_state.part_version}", use_container_width=True,
                               column_config={"point": st.column_config
                                              .NumberColumn(disabled=True)})

        st.markdown("**Holes** — optional circular cut-outs")
        out_holes = st.data_editor(hole_df, num_rows="dynamic",
                                   hide_index=True, key=f"holes_{st.session_state.part_version}",
                                   use_container_width=True)

        thickness = st.text_input(
            "Extrusion depth" if is3d else "Thickness",
            str(part.get("depth" if is3d else "thickness", "D" if is3d else "t")),
            key=f"thick_{st.session_state.part_version}",
            help="A parameter name such as D, or a number such as 10. "
                 "Not an assignment — write D, not D=10.")

        # Plain widgets rather than an editable table: the in-cell dropdown
        # of st.data_editor needs a second click to open and is easy to miss.
        v = st.session_state.part_version
        KINDS = ["free", "fixed", "load"]

        if is3d:
            st.markdown("**Faces** — outline segment *i* raises `side_i` when "
                        "extruded; `bottom` and `top` are the caps and "
                        "`hole_j` the bore walls. Names are drawn on the "
                        "preview.")
            rows = []
            for _, r in face_df.iterrows():
                nm = str(r["face"])
                st.markdown(f"`{nm}`")
                c = st.columns([1.1, 1.0, 1.0, 1.0, 1.0])
                kind = c[0].selectbox(
                    "condition", KINDS,
                    index=KINDS.index(str(r["condition"])),
                    key=f"fk_{v}_{nm}")
                dof, fx, fy, fz = str(r["fix"]), 0.0, 0.0, 0.0
                if kind == "fixed":
                    opts = ["xyz", "x", "y", "z", "xy", "xz", "yz"]
                    dof = c[1].selectbox(
                        "hold", opts,
                        index=opts.index(dof) if dof in opts else 0,
                        key=f"fd_{v}_{nm}")
                elif kind == "load":
                    fx = c[1].number_input("Fx (N)", value=float(r["Fx (N)"]),
                                           step=100.0, key=f"fx_{v}_{nm}")
                    fy = c[2].number_input("Fy (N)", value=float(r["Fy (N)"]),
                                           step=100.0, key=f"fy_{v}_{nm}")
                    fz = c[3].number_input("Fz (N)", value=float(r["Fz (N)"]),
                                           step=100.0, key=f"fz_{v}_{nm}")
                rows.append({"face": nm, "condition": kind, "fix": dof,
                             "Fx (N)": fx, "Fy (N)": fy, "Fz (N)": fz})
            face_e = pd.DataFrame(rows)
            seg_e = seg_df
        else:
            st.markdown("**Edges** — segment *i* runs from point *i* to point "
                        "*i+1*. The numbers are drawn on the preview.")
            rows = []
            for _, r in seg_df.iterrows():
                i = int(r["segment"])
                st.markdown(f"**Edge {i}**")
                c = st.columns([1.1, 1.0, 1.0, 1.0])
                kind = c[0].selectbox(
                    "condition", KINDS,
                    index=KINDS.index(str(r["condition"])),
                    key=f"sk_{v}_{i}")
                dof, fx, fy = str(r["fix"]), 0.0, 0.0
                if kind == "fixed":
                    opts = ["xy", "x", "y"]
                    dof = c[1].selectbox(
                        "hold", opts,
                        index=opts.index(dof) if dof in opts else 0,
                        key=f"sd_{v}_{i}")
                elif kind == "load":
                    fx = c[1].number_input("Fx (N)", value=float(r["Fx (N)"]),
                                           step=500.0, key=f"sx_{v}_{i}")
                    fy = c[2].number_input("Fy (N)", value=float(r["Fy (N)"]),
                                           step=500.0, key=f"sy_{v}_{i}")
                rows.append({"segment": i, "condition": kind, "fix": dof,
                             "Fx (N)": fx, "Fy (N)": fy})
            face_e = None
            seg_e = pd.DataFrame(rows)

    if is3d:
        new_part = tables_to_part3d(name, prm_e, out_e, out_holes,
                                    face_e, thickness)
    else:
        new_part = tables_to_part(name, prm_e, out_e, out_holes, seg_e,
                                  thickness)
    st.session_state[store] = new_part

    with view:
        st.markdown("**Preview**")
        mod = custom3d if is3d else custom2d
        problems = mod.validate(new_part)
        try:
            if is3d:
                st.pyplot(plots.geometry_custom3d(new_part), clear_figure=True)
            else:
                st.pyplot(plots.geometry_custom(new_part), clear_figure=True)
        except Exception as exc:
            st.warning(f"Cannot draw this part yet: {exc}")

        if problems:
            for p in problems:
                st.warning(p)
        else:
            st.success("Part is valid and ready to solve.")

        st.download_button("Download part (.json)",
                           json.dumps(new_part, indent=2),
                           file_name=f"{name.replace(' ', '_')}.json",
                           mime="application/json")

        if is3d:
            st.caption("A 3D solve takes several seconds.")
        if st.button("Solve once", type="primary", disabled=bool(problems)):
            try:
                with st.spinner("Meshing and solving..."):
                    r = mod.solve_part(new_part, return_field=True)
                a, b = st.columns(2)
                a.metric("Peak von Mises", f"{r['max_von_mises']:.1f} MPa")
                b.metric("Max displacement", f"{r['max_disp']:.4f} mm")
                st.caption(f"{r['n_elements']} elements, "
                           f"{r['solve_time_s']:.2f} s")
                if is3d:
                    st.pyplot(plots.heatmap_3d(r["field"]), clear_figure=True)
                else:
                    st.pyplot(plots.heatmap_2d(r["field"]), clear_figure=True)
            except Exception as exc:
                st.error(str(exc))

    st.info("Happy with the part? Go to **Create a surrogate** and choose "
            + ("**User-defined 3D part**" if is3d else "**User-defined part**")
            + " to train a model on it.")


# ---------------------------------------------------- MODE 3: create

else:
    st.subheader("Create a surrogate")
    st.write("A surrogate is only valid inside the design space it was "
             "trained on. Define that space, and the app samples it, solves "
             "at every point, and fits a Gaussian Process.")
    st.caption("Parts you build in **Define a part** appear here once you "
               "have opened that tab — 2D and 3D are listed separately.")

    options = [f["label"] for f in generator.FAMILIES.values()]
    if "part" in st.session_state:
        options.append("User-defined part")
    if "part3d" in st.session_state:
        options.append("User-defined 3D part")

    fam_label = st.selectbox("Part", options)
    is_custom3d = fam_label == "User-defined 3D part"

    # ---------- user-defined part, 2D or extruded 3D
    if fam_label in ("User-defined part", "User-defined 3D part"):
        part = st.session_state["part3d" if is_custom3d else "part"]
        mod = custom3d if is_custom3d else custom2d
        st.caption(f"Using **{part.get('name','part')}** from the part editor.")

        problems = mod.validate(part)
        if problems:
            st.error("This part cannot be solved yet:\n\n"
                     + "\n".join(f"- {p}" for p in problems))
            st.caption("Go to **Define a part** and set the `condition` "
                       "column: one "
                       + ("face" if is_custom3d else "edge") + " `fixed`, "
                       "another `load` with a non-zero force.")
            st.stop()

        c1, c2 = st.columns([1, 1])
        with c1:
            if is_custom3d:
                st.pyplot(plots.geometry_custom3d(part), clear_figure=True)
            else:
                st.pyplot(plots.geometry_custom(part), clear_figure=True)
        with c2:
            st.markdown("**Which parameters vary?**")
            st.caption("Anything unticked is held at its value in the "
                       "definition. The ticked ones, in order, become the "
                       "surrogate's inputs.")
            ranges = {}
            for k, v in part["parameters"].items():
                cols = st.columns([1, 1, 1])
                use = cols[0].checkbox(k, value=True, key=f"use_{k}")
                if use:
                    lo = cols[1].number_input(
                        "min", value=float(v) * 0.6, step=0.5, key=f"clo_{k}",
                        label_visibility="collapsed")
                    hi = cols[2].number_input(
                        "max", value=float(v) * 1.4, step=0.5, key=f"chi_{k}",
                        label_visibility="collapsed")
                    ranges[k] = (lo, hi)

        bad = [k for k, (lo, hi) in ranges.items() if hi <= lo]
        if bad:
            st.error("Upper bound must exceed lower bound for: "
                     + ", ".join(bad))
        if not ranges:
            st.warning("Tick at least one parameter to vary.")

        suggested = max(60, 100 * len(ranges)) if not is_custom3d \
            else max(45, 40 * len(ranges))
        n = st.number_input("Number of simulations", 40, 2000,
                            suggested, 10)
        mesh = st.slider("Mesh size (mm) — smaller is more accurate and "
                         "slower",
                         2.0 if is_custom3d else 1.0,
                         12.0 if is_custom3d else 8.0,
                         5.0 if is_custom3d else 3.0, 0.5)
        # Solve cost scales with element count: 1/h^3 in 3D, 1/h^2 in 2D.
        # Calibrated against measured runs (5 mm 3D ≈ 1.7 s, 3 mm 2D ≈ 0.45 s).
        per_run = (1.7 * (5.0 / mesh) ** 3 if is_custom3d
                   else 0.45 * (3.0 / mesh) ** 2)
        st.caption(f"{len(ranges)} parameters · about {per_run:.2f} s per "
                   f"solve at this mesh size · roughly "
                   f"**{n * per_run / 60:.1f} min** total")
        if is_custom3d:
            st.caption("3D needs fewer samples per dimension to stay "
                       "practical; the trade-off is a coarser model.")

        name = st.text_input("Name this surrogate",
                             value=f"{part.get('name','part')} — surrogate")

        if st.button("Build surrogate", type="primary",
                     disabled=bool(bad) or not ranges):
            bar, status = st.progress(0.0), st.empty()
            t_start = time.time()

            def on_progress(done, total, n_ok, n_failed):
                bar.progress(done / total)
                el = time.time() - t_start
                left = (total - done) / max(done / max(el, 1e-9), 1e-9)
                status.write(f"{done}/{total} · {n_ok} ok · {n_failed} failed "
                             f"· ~{left/60:.1f} min left")

            try:
                build = (generator.generate_custom3d if is_custom3d
                         else generator.generate_custom)
                entry = build(part, ranges, int(n), name,
                              progress=on_progress, mesh_size=mesh)
            except Exception as exc:
                bar.empty(); status.empty(); st.error(str(exc))
            else:
                bar.empty(); status.empty()
                st.success(f"Built **{entry['name']}** from "
                           f"{entry['n_success']} simulations in "
                           f"{entry['build_minutes']:.1f} min.")
                st.table([{
                    "target": entry["targets"][t],
                    "GP R²": f"{m['gaussian_process']['r2']:.4f}",
                    "GP median error":
                        f"{m['gaussian_process']['median_ape_pct']:.2f}%",
                    "linear R² (baseline)": f"{m['linear']['r2']:.4f}",
                } for t, m in entry["metrics"].items()])
                st.info("Switch to **Use a surrogate** — it is in the list, "
                        "marked with a star.")

    # ---------- built-in families
    else:
        fam_key = next(k for k, f in generator.FAMILIES.items()
                       if f["label"] == fam_label)
        fam = generator.FAMILIES[fam_key]

        st.markdown("**Design space**")
        ranges = {}
        for p in fam["params"]:
            lo_lim, hi_lim = fam["limits"][p]
            d_lo, d_hi = fam["defaults"][p]
            c1, c2 = st.columns(2)
            lo = c1.number_input(f"{fam['labels'][p]} — min", float(lo_lim),
                                 float(hi_lim), float(d_lo), 0.5, key=f"lo_{p}")
            hi = c2.number_input(f"{fam['labels'][p]} — max", float(lo_lim),
                                 float(hi_lim), float(d_hi), 0.5, key=f"hi_{p}")
            ranges[p] = (lo, hi)

        bad = [p for p, (lo, hi) in ranges.items() if hi <= lo]
        if bad:
            st.error("Upper bound must exceed lower bound for: "
                     + ", ".join(bad))

        suggested = generator.suggested_samples(fam_key)
        n = st.number_input("Number of simulations", 50, 2000, suggested, 10)
        est = generator.estimate_minutes(fam_key, n)
        st.caption(f"{len(fam['params'])} parameters · suggested {suggested} "
                   f"· estimated build time **{est:.0f} min**")
        if est > 10:
            st.warning(f"About {est:.0f} minutes. Leave the browser tab open "
                       "— closing it cancels the build.")

        name = st.text_input("Name this surrogate",
                             value=f"{fam_label} — custom")

        if st.button("Build surrogate", type="primary", disabled=bool(bad)):
            bar, status = st.progress(0.0), st.empty()
            t_start = time.time()

            def on_progress(done, total, n_ok, n_failed):
                bar.progress(done / total)
                el = time.time() - t_start
                left = (total - done) / max(done / max(el, 1e-9), 1e-9)
                status.write(f"{done}/{total} simulations · {n_ok} succeeded "
                             f"· {n_failed} invalid · ~{left/60:.1f} min left")

            try:
                entry = generator.generate(fam_key, ranges, int(n), name,
                                           progress=on_progress)
            except Exception as exc:
                bar.empty(); status.empty(); st.error(str(exc))
            else:
                bar.empty(); status.empty()
                st.success(f"Built **{entry['name']}** from "
                           f"{entry['n_success']} simulations in "
                           f"{entry['build_minutes']:.1f} min "
                           f"({entry['n_failed']} invalid skipped).")
                st.table([{
                    "target": entry["targets"][t],
                    "GP R²": f"{m['gaussian_process']['r2']:.4f}",
                    "GP median error":
                        f"{m['gaussian_process']['median_ape_pct']:.2f}%",
                    "linear R² (baseline)": f"{m['linear']['r2']:.4f}",
                } for t, m in entry["metrics"].items()])
                st.info("Switch to **Use a surrogate** — it is in the list, "
                        "marked with a star.")

    existing = generator.list_surrogates()
    if existing:
        st.divider()
        st.markdown("**Surrogates you have built**")
        for e in existing:
            c1, c2 = st.columns([5, 1])
            rng = ",  ".join(f"{k} {v[0]:g}–{v[1]:g}"
                             for k, v in e["ranges"].items())
            best = max(m["gaussian_process"]["r2"]
                       for m in e["metrics"].values())
            c1.write(f"**{e['name']}** · {e['family_label']} · "
                     f"{e['n_success']} runs · best R² {best:.4f}")
            c1.caption(f"{rng} mm · built {e['created']}")
            if c2.button("Delete", key=f"del_{e['id']}"):
                generator.delete_surrogate(e["id"])
                st.rerun()
