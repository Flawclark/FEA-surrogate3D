"""
custom2d.py — a general 2D plane-stress solver driven by a part definition.

The fixed pipelines (simulate.py, simulate3d.py) each hard-code one shape.
This module takes the shape, the loads and the supports as data, so a user can
define their own part and still get a solve.

A part definition looks like:

    {
      "name": "plate with a hole",
      "parameters": {"W": 100.0, "H": 100.0, "R": 10.0, "t": 5.0},
      "outline":  [["0","0"], ["W","0"], ["W","H"], ["0","H"]],
      "holes":    [{"x": "W/2", "y": "H/2", "r": "R"}],
      "thickness": "t",
      "segments": {
          "0": {"kind": "free"},
          "1": {"kind": "load", "fx": 10000.0, "fy": 0.0},
          "2": {"kind": "free"},
          "3": {"kind": "fixed", "dof": "xy"}
      }
    }

Outline coordinates are expressions in the parameters, so one definition
describes a family of shapes. Segment i runs from outline point i to point
i+1, wrapping at the end; that indexing is what the UI labels.
"""

import ast
import math
import operator
import os
import tempfile
import time

import gmsh
import meshio
import numpy as np

from skfem import (
    MeshTri, Basis, ElementVector, ElementTriP2, FacetBasis,
    BilinearForm, LinearForm, asm, condense, solve
)
from skfem.helpers import ddot, sym_grad, trace

E_MODULUS = 210_000.0     # MPa
NU = 0.3
TOL = 1e-6


# ----------------------------------------------------- safe expressions

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "min": min, "max": max, "radians": math.radians,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def evaluate(expr, params):
    """Evaluate an arithmetic expression over the named parameters.

    Deliberately not eval(): only arithmetic, a short list of functions and
    the parameter names are reachable, so a part definition cannot run code.
    """
    if isinstance(expr, (int, float)):
        return float(expr)

    try:
        tree = ast.parse(str(expr), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expr!r}: {exc}") from None

    def walk(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"not a number: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in params:
                return float(params[node.id])
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCS:
                raise ValueError(f"function {node.func.id!r} not allowed")
            return _FUNCS[node.func.id](*[walk(a) for a in node.args])
        raise ValueError(f"expression not allowed: {ast.dump(node)[:60]}")

    return float(walk(tree))


def resolve(part, params=None):
    """Turn a part definition into concrete numbers for one parameter set."""
    p = dict(part["parameters"])
    if params:
        p.update(params)

    pts = np.array([[evaluate(x, p), evaluate(y, p)]
                    for x, y in part["outline"]], dtype=float)
    holes = [(evaluate(h["x"], p), evaluate(h["y"], p), evaluate(h["r"], p))
             for h in part.get("holes", [])]
    thickness = evaluate(part.get("thickness", 1.0), p)
    return pts, holes, thickness


# ------------------------------------------------------------ validation

def validate(part, params=None):
    """Return a list of problems; empty means the part can be solved."""
    problems = []
    try:
        pts, holes, t = resolve(part, params)
    except ValueError as exc:
        return [str(exc)]

    if len(pts) < 3:
        problems.append("outline needs at least 3 points")
    if t <= 0:
        problems.append("thickness must be positive")

    # polygon area via the shoelace formula; zero means degenerate
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 1e-6:
        problems.append("outline has zero area — check the coordinates")

    for i, (hx, hy, hr) in enumerate(holes):
        if hr <= 0:
            problems.append(f"hole {i+1}: radius must be positive")
            continue
        if not _point_in_polygon(hx, hy, pts):
            problems.append(f"hole {i+1}: centre is outside the outline")
        if math.pi * hr ** 2 > 0.8 * area:
            problems.append(f"hole {i+1}: too large for the part")

    segs = part.get("segments", {})
    has_fix = any(s.get("kind") == "fixed" for s in segs.values())
    has_load = any(s.get("kind") == "load" for s in segs.values())
    if not has_fix:
        problems.append("at least one edge must be fixed, or the part floats")
    if not has_load:
        problems.append("at least one edge must carry a load")

    return problems


def _point_in_polygon(px, py, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < xint:
                inside = not inside
    return inside


# ---------------------------------------------------------------- meshing

def build_mesh(pts, holes, mesh_size=None):
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("part")

    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    if mesh_size is None:
        smallest = min([h[2] for h in holes], default=diag / 6.0)
        mesh_size = max(min(diag / 28.0, smallest / 3.0), diag / 220.0)

    try:
        occ = gmsh.model.occ
        gpts = [occ.addPoint(float(x), float(y), 0.0) for x, y in pts]
        lines = [occ.addLine(gpts[i], gpts[(i + 1) % len(gpts)])
                 for i in range(len(gpts))]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])

        out = [(2, surf)]
        if holes:
            discs = [(2, occ.addDisk(hx, hy, 0.0, hr, hr))
                     for hx, hy, hr in holes]
            out, _ = occ.cut(out, discs)
        occ.synchronize()

        faces = [s[1] for s in out if s[0] == 2]
        if not faces:
            raise RuntimeError("geometry collapsed — holes may cover the part")
        gmsh.model.addPhysicalGroup(2, faces, tag=1)

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.35)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 22)
        gmsh.model.mesh.generate(2)

        fd, path = tempfile.mkstemp(suffix=".msh")
        os.close(fd)
        gmsh.write(path)
    finally:
        gmsh.finalize()

    m = meshio.read(path)
    os.remove(path)

    tris = None
    for block in m.cells:
        if block.type == "triangle":
            tris = block.data.T
    if tris is None:
        raise RuntimeError("meshing produced no triangles")
    return MeshTri(m.points[:, :2].T, tris)


# ------------------------------------------------- segment identification

def _segment_of(midpoints, pts, tol_frac=0.02):
    """Which outline segment each boundary facet midpoint lies on.

    Returns an array of segment indices, -1 where the facet belongs to a hole
    or otherwise sits on no outline segment.
    """
    n = len(pts)
    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    tol = diag * tol_frac

    best = np.full(midpoints.shape[1], -1, dtype=int)
    best_d = np.full(midpoints.shape[1], np.inf)

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        ap = midpoints.T - a
        s = np.clip((ap @ ab) / L2, 0.0, 1.0)
        proj = a + s[:, None] * ab
        d = np.linalg.norm(midpoints.T - proj, axis=1)
        take = d < np.minimum(best_d, tol)
        best[take] = i
        best_d[take] = d[take]

    return best


# ----------------------------------------------------------------- solve

def lame_plane_stress(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return 2 * lam * mu / (lam + 2 * mu), mu


def solve_part(part, params=None, mesh_size=None, return_field=False):
    """Solve one instance of a part definition.

    Loads are given as total force in newtons on a segment and converted to
    a traction using that segment's length and the part thickness.
    """
    problems = validate(part, params)
    if problems:
        raise ValueError("; ".join(problems))

    t_start = time.time()
    pts, holes, thickness = resolve(part, params)
    mesh = build_mesh(pts, holes, mesh_size)

    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element, intorder=4)
    lam, mu = lame_plane_stress(E_MODULUS, NU)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis)

    bfacets = mesh.boundary_facets()
    mids = mesh.p[:, mesh.facets[:, bfacets]].mean(axis=1)
    seg_of = _segment_of(mids, pts)

    segments = {int(k): v for k, v in part.get("segments", {}).items()}

    # ---- loads
    f = np.zeros(basis.N)
    for idx, spec in segments.items():
        if spec.get("kind") != "load":
            continue
        facets = bfacets[seg_of == idx]
        if facets.size == 0:
            continue

        a, b = pts[idx], pts[(idx + 1) % len(pts)]
        length = float(np.linalg.norm(b - a))
        area = max(length * thickness, 1e-12)
        tx = float(spec.get("fx", 0.0)) / area
        ty = float(spec.get("fy", 0.0)) / area

        fb = FacetBasis(mesh, element, facets=facets, intorder=4)

        @LinearForm
        def load(v, w, tx=tx, ty=ty):
            return tx * v[0] + ty * v[1]

        f += asm(load, fb)

    # ---- supports
    fixed = []
    for idx, spec in segments.items():
        if spec.get("kind") != "fixed":
            continue
        facets = bfacets[seg_of == idx]
        if facets.size == 0:
            continue
        nodes = np.unique(mesh.facets[:, facets])
        dof = spec.get("dof", "xy")
        if "x" in dof:
            fixed.append(basis.nodal_dofs[0, nodes])
        if "y" in dof:
            fixed.append(basis.nodal_dofs[1, nodes])
        # P2 elements also carry mid-facet dofs; constrain those too so the
        # edge is genuinely clamped rather than pinned at its corners only
        edge_dofs = basis.get_dofs(facets=facets)
        if dof == "xy":
            fixed.append(edge_dofs.all())
        else:
            fixed.append(edge_dofs.all(f"u^{1 if 'x' in dof else 2}"))

    if not fixed:
        raise ValueError("no mesh facets matched the fixed edges")
    fixed = np.unique(np.concatenate(fixed))

    u = solve(*condense(K, f, D=fixed))

    disp = u[basis.nodal_dofs]
    max_disp = float(np.max(np.linalg.norm(disp, axis=0)))

    eps = sym_grad(basis.interpolate(u))
    tr = eps[0, 0] + eps[1, 1]
    sxx = lam * tr + 2 * mu * eps[0, 0]
    syy = lam * tr + 2 * mu * eps[1, 1]
    sxy = 2 * mu * eps[0, 1]
    vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    vm_elem = vm.max(axis=1)

    out = {
        "max_von_mises": float(vm_elem.max()),
        "max_disp": max_disp,
        "n_elements": int(mesh.t.shape[1]),
        "n_dofs": int(basis.N),
        "solve_time_s": round(time.time() - t_start, 3),
    }
    if return_field:
        out["field"] = {
            "points": mesh.p.copy(),
            "tris": mesh.t.copy(),
            "vm_elem": vm_elem,
            "disp": disp.copy(),
        }
    return out


# --------------------------------------------------------------- presets

PRESETS = {
    "Plate with a central hole": {
        "name": "Plate with a central hole",
        "parameters": {"W": 100.0, "H": 100.0, "R": 10.0, "t": 5.0},
        "outline": [["0", "0"], ["W", "0"], ["W", "H"], ["0", "H"]],
        "holes": [{"x": "W/2", "y": "H/2", "r": "R"}],
        "thickness": "t",
        "segments": {
            "0": {"kind": "free"},
            "1": {"kind": "load", "fx": 10000.0, "fy": 0.0},
            "2": {"kind": "free"},
            "3": {"kind": "fixed", "dof": "xy"},
        },
    },
    "Cantilever beam (tip load)": {
        "name": "Cantilever beam",
        "parameters": {"L": 120.0, "H": 20.0, "t": 8.0},
        "outline": [["0", "0"], ["L", "0"], ["L", "H"], ["0", "H"]],
        "holes": [],
        "thickness": "t",
        "segments": {
            "0": {"kind": "free"},
            "1": {"kind": "load", "fx": 0.0, "fy": -2000.0},
            "2": {"kind": "free"},
            "3": {"kind": "fixed", "dof": "xy"},
        },
    },
    "L-shaped bracket (re-entrant corner)": {
        "name": "L-bracket",
        "parameters": {"L": 90.0, "H": 90.0, "a": 30.0, "b": 30.0, "t": 6.0},
        "outline": [["0", "0"], ["L", "0"], ["L", "b"],
                    ["a", "b"], ["a", "H"], ["0", "H"]],
        "holes": [],
        "thickness": "t",
        "segments": {
            "0": {"kind": "fixed", "dof": "xy"},
            "1": {"kind": "free"},
            "2": {"kind": "load", "fx": 0.0, "fy": -1500.0},
            "3": {"kind": "free"},
            "4": {"kind": "free"},
            "5": {"kind": "free"},
        },
    },
    "Perforated strip (two holes)": {
        "name": "Perforated strip",
        "parameters": {"W": 140.0, "H": 60.0, "R": 10.0, "t": 5.0},
        "outline": [["0", "0"], ["W", "0"], ["W", "H"], ["0", "H"]],
        "holes": [{"x": "W/3", "y": "H/2", "r": "R"},
                  {"x": "2*W/3", "y": "H/2", "r": "R"}],
        "thickness": "t",
        "segments": {
            "0": {"kind": "free"},
            "1": {"kind": "load", "fx": 8000.0, "fy": 0.0},
            "2": {"kind": "free"},
            "3": {"kind": "fixed", "dof": "xy"},
        },
    },
}
