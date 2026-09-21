"""
custom3d.py — a general 3D solver for extruded parts.

A 2D outline (with optional holes) extruded along z to a depth. That covers
plate, laser-cut and prismatic machined parts, which is most of what a 2D
profile editor is useful for anyway.

Every face can carry its own load or support:

    side_<i>   the wall raised by outline segment i
    bottom     the z = 0 cap
    top        the z = depth cap
    hole_<j>   the cylindrical wall of hole j

A definition looks like:

    {
      "name": "bracket plate",
      "kind": "extruded",
      "parameters": {"W": 120.0, "H": 60.0, "D": 10.0, "R": 8.0},
      "outline": [["0","0"], ["W","0"], ["W","H"], ["0","H"]],
      "holes":   [{"x": "W-20", "y": "H/2", "r": "R"}],
      "depth": "D",
      "faces": {
          "side_3":  {"kind": "fixed", "dof": "xyz"},
          "side_1":  {"kind": "load",  "fx": 0.0, "fy": 0.0, "fz": -4000.0}
      }
    }

Faces not listed are free.
"""

import os
import tempfile
import time

import gmsh
import meshio
import numpy as np
from scipy.spatial import cKDTree

from skfem import (
    MeshTet, Basis, ElementVector, ElementTetP2, FacetBasis,
    BilinearForm, LinearForm, asm, condense, solve
)
from skfem.helpers import ddot, sym_grad, trace

from custom2d import evaluate, _point_in_polygon

E_MODULUS = 210_000.0
NU = 0.3


def lame_3d(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def resolve(part, params=None):
    p = dict(part["parameters"])
    if params:
        p.update(params)
    pts = np.array([[evaluate(x, p), evaluate(y, p)]
                    for x, y in part["outline"]], dtype=float)
    holes = [(evaluate(h["x"], p), evaluate(h["y"], p), evaluate(h["r"], p))
             for h in part.get("holes", [])]
    depth = evaluate(part.get("depth", 10.0), p)
    return pts, holes, depth


def face_names(part):
    names = [f"side_{i}" for i in range(len(part["outline"]))]
    names += ["bottom", "top"]
    names += [f"hole_{j}" for j in range(len(part.get("holes", [])))]
    return names


def validate(part, params=None):
    problems = []
    try:
        pts, holes, depth = resolve(part, params)
    except ValueError as exc:
        return [str(exc)]

    if len(pts) < 3:
        problems.append("outline needs at least 3 points")
    if depth <= 0:
        problems.append("depth must be positive")

    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 1e-6:
        problems.append("outline has zero area")

    for j, (hx, hy, hr) in enumerate(holes):
        if hr <= 0:
            problems.append(f"hole {j+1}: radius must be positive")
        elif not _point_in_polygon(hx, hy, pts):
            problems.append(f"hole {j+1}: centre is outside the outline")

    faces = part.get("faces", {})
    known = set(face_names(part))
    for name in faces:
        if name not in known:
            problems.append(f"unknown face {name!r}")
    if not any(f.get("kind") == "fixed" for f in faces.values()):
        problems.append("at least one face must be fixed, or the part floats")
    if not any(f.get("kind") == "load" for f in faces.values()):
        problems.append("at least one face must carry a load")

    return problems


def build_mesh(pts, holes, depth, mesh_size=None):
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("extruded")

    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    if mesh_size is None:
        smallest = min([h[2] for h in holes], default=diag / 6.0)
        mesh_size = max(min(diag / 14.0, smallest / 2.0, depth / 2.0),
                        diag / 60.0)

    try:
        occ = gmsh.model.occ
        gp = [occ.addPoint(float(a), float(b), 0.0) for a, b in pts]
        lines = [occ.addLine(gp[i], gp[(i + 1) % len(gp)])
                 for i in range(len(gp))]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])

        out = [(2, face)]
        if holes:
            discs = [(2, occ.addDisk(hx, hy, 0.0, hr, hr))
                     for hx, hy, hr in holes]
            out, _ = occ.cut(out, discs)
        occ.synchronize()

        surfaces = [s for s in out if s[0] == 2]
        if not surfaces:
            raise RuntimeError("profile collapsed — holes may cover the part")

        ext = occ.extrude(surfaces, 0.0, 0.0, float(depth))
        occ.synchronize()
        vols = [e[1] for e in ext if e[0] == 3]
        if not vols:
            raise RuntimeError("extrusion produced no solid")
        gmsh.model.addPhysicalGroup(3, vols, tag=1)

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.4)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 14)
        gmsh.model.mesh.generate(3)

        fd, path = tempfile.mkstemp(suffix=".msh")
        os.close(fd)
        gmsh.write(path)
    finally:
        gmsh.finalize()

    m = meshio.read(path)
    os.remove(path)

    tets = None
    for block in m.cells:
        if block.type == "tetra":
            tets = block.data.T
    if tets is None:
        raise RuntimeError("meshing produced no tetrahedra")
    return MeshTet(m.points.T, tets)


def classify_faces(mids, pts, holes, depth, tol_frac=0.03):
    """Label each boundary facet midpoint with the face it belongs to."""
    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    tol = diag * tol_frac
    ztol = depth * 1e-3

    labels = np.full(mids.shape[1], "", dtype=object)

    labels[np.abs(mids[2]) < ztol] = "bottom"
    labels[np.abs(mids[2] - depth) < ztol] = "top"

    xy = mids[:2].T
    free = labels == ""

    # hole walls first: they sit at a fixed radius from a known centre
    for j, (hx, hy, hr) in enumerate(holes):
        d = np.abs(np.linalg.norm(xy - np.array([hx, hy]), axis=1) - hr)
        take = free & (d < max(hr * 0.12, tol * 0.4))
        labels[take] = f"hole_{j}"
        free = labels == ""

    # remaining side walls match an outline segment in plan
    n = len(pts)
    best_d = np.full(mids.shape[1], np.inf)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        s = np.clip(((xy - a) @ ab) / L2, 0.0, 1.0)
        proj = a + s[:, None] * ab
        d = np.linalg.norm(xy - proj, axis=1)
        take = free & (d < np.minimum(best_d, tol))
        labels[take] = f"side_{i}"
        best_d[take] = d[take]

    return labels


def solve_part(part, params=None, mesh_size=None, return_field=False):
    problems = validate(part, params)
    if problems:
        raise ValueError("; ".join(problems))

    t_start = time.time()
    pts, holes, depth = resolve(part, params)
    mesh = build_mesh(pts, holes, depth, mesh_size)

    element = ElementVector(ElementTetP2())
    basis = Basis(mesh, element, intorder=2)
    lam, mu = lame_3d(E_MODULUS, NU)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis)

    bfacets = mesh.boundary_facets()
    mids = mesh.p[:, mesh.facets[:, bfacets]].mean(axis=1)
    labels = classify_faces(mids, pts, holes, depth)

    faces = part.get("faces", {})

    # ---- loads: total force on a face, spread over its measured area
    f = np.zeros(basis.N)
    for name, spec in faces.items():
        if spec.get("kind") != "load":
            continue
        facets = bfacets[labels == name]
        if facets.size == 0:
            continue
        fb = FacetBasis(mesh, element, facets=facets, intorder=2)
        # measured area of the face: the quadrature weights integrate 1 over it
        area = max(float(fb.dx.sum()), 1e-9)
        tx = float(spec.get("fx", 0.0)) / area
        ty = float(spec.get("fy", 0.0)) / area
        tz = float(spec.get("fz", 0.0)) / area

        @LinearForm
        def load(v, w, tx=tx, ty=ty, tz=tz):
            return tx * v[0] + ty * v[1] + tz * v[2]

        f += asm(load, fb)

    # ---- supports
    fixed = []
    for name, spec in faces.items():
        if spec.get("kind") != "fixed":
            continue
        facets = bfacets[labels == name]
        if facets.size == 0:
            continue
        dofs = basis.get_dofs(facets=facets)
        dof = spec.get("dof", "xyz")
        if dof == "xyz":
            fixed.append(dofs.all())
        else:
            for axis, comp in (("x", 1), ("y", 2), ("z", 3)):
                if axis in dof:
                    fixed.append(dofs.all(f"u^{comp}"))

    if not fixed:
        raise ValueError("no mesh facets matched the fixed faces — check the "
                         "face names against the preview")
    fixed = np.unique(np.concatenate(fixed))

    u = solve(*condense(K, f, D=fixed))

    disp = u[basis.nodal_dofs]
    max_disp = float(np.max(np.linalg.norm(disp, axis=0)))

    eps = sym_grad(basis.interpolate(u))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = [[lam * tr * (1.0 if i == j else 0.0) + 2.0 * mu * eps[i, j]
          for j in range(3)] for i in range(3)]
    vm = np.sqrt(0.5 * ((s[0][0] - s[1][1]) ** 2
                        + (s[1][1] - s[2][2]) ** 2
                        + (s[2][2] - s[0][0]) ** 2
                        + 6.0 * (s[0][1] ** 2 + s[1][2] ** 2 + s[2][0] ** 2)))
    vm_elem = vm.max(axis=1)

    # A fully clamped face is singular at its edge, so the global maximum
    # there is mesh dependent. Report both and say which is which.
    clamped = [n for n, sp in faces.items() if sp.get("kind") == "fixed"]
    near_support = np.zeros(mesh.t.shape[1], dtype=bool)
    if clamped:
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        diag = float(np.linalg.norm(mesh.p.max(axis=1) - mesh.p.min(axis=1)))
        for name in clamped:
            facets = bfacets[labels == name]
            if facets.size == 0:
                continue
            fmids = mesh.p[:, mesh.facets[:, facets]].mean(axis=1)
            tree = cKDTree(fmids.T)
            d, _ = tree.query(centroids.T)
            near_support |= d < diag * 0.04

    away = ~near_support
    out = {
        "max_von_mises": float(vm_elem.max()),
        "max_von_mises_away_from_supports":
            float(vm_elem[away].max()) if away.any() else float("nan"),
        "max_disp": max_disp,
        "n_elements": int(mesh.t.shape[1]),
        "n_dofs": int(basis.N),
        "solve_time_s": round(time.time() - t_start, 2),
    }
    if return_field:
        facet_nodes = mesh.facets[:, bfacets]
        owner = mesh.f2t[0, bfacets]
        out["field"] = {
            "points": mesh.p.copy(),
            "surf_tris": facet_nodes.copy(),
            "surf_vm": vm_elem[owner].copy(),
        }
    return out


# --------------------------------------------------------------- presets

PRESETS = {
    "Mounting plate (cantilevered, one hole)": {
        "name": "Mounting plate",
        "kind": "extruded",
        "parameters": {"W": 120.0, "H": 60.0, "D": 10.0, "R": 8.0},
        "outline": [["0", "0"], ["W", "0"], ["W", "H"], ["0", "H"]],
        "holes": [{"x": "W-25", "y": "H/2", "r": "R"}],
        "depth": "D",
        "faces": {
            "side_3": {"kind": "fixed", "dof": "xyz"},
            "side_1": {"kind": "load", "fx": 0.0, "fy": 0.0, "fz": -4000.0},
        },
    },
    "Tension strip with a hole": {
        "name": "Tension strip",
        "kind": "extruded",
        "parameters": {"W": 100.0, "H": 50.0, "D": 6.0, "R": 10.0},
        "outline": [["0", "0"], ["W", "0"], ["W", "H"], ["0", "H"]],
        "holes": [{"x": "W/2", "y": "H/2", "r": "R"}],
        "depth": "D",
        "faces": {
            "side_3": {"kind": "fixed", "dof": "xyz"},
            "side_1": {"kind": "load", "fx": 15000.0, "fy": 0.0, "fz": 0.0},
        },
    },
    "L-profile bracket (extruded)": {
        "name": "L-profile bracket",
        "kind": "extruded",
        "parameters": {"L": 80.0, "Hw": 50.0, "tb": 8.0, "tw": 8.0,
                       "D": 60.0},
        "outline": [["0", "0"], ["L", "0"], ["L", "tb"],
                    ["tw", "tb"], ["tw", "Hw"], ["0", "Hw"]],
        "holes": [],
        "depth": "D",
        "faces": {
            "side_0": {"kind": "fixed", "dof": "xyz"},
            "side_4": {"kind": "load", "fx": 1000.0, "fy": 0.0, "fz": 0.0},
        },
    },
}
