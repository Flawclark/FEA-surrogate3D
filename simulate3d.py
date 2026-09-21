"""
simulate3d.py — one parametric 3D FEA run.

L-bracket: a base plate bolted down, a vertical web, a fillet at the inner
corner, and a load applied at the top of the web. The stress concentration
lives in the fillet, which has no 2D equivalent.

Units: mm, N, MPa.

simulate3d(tb, tw, rf, Hw, rh) -> dict
"""

import os
import tempfile
import time

import gmsh
import meshio
import numpy as np

from skfem import (
    MeshTet, Basis, ElementVector, ElementTetP2, FacetBasis,
    BilinearForm, LinearForm, asm, condense, solve
)
from skfem.helpers import ddot, sym_grad, trace

E_MODULUS = 210_000.0    # MPa
NU = 0.3
FORCE_TOTAL = 1_000.0    # N, applied at the top of the web in +x

# Fixed overall dimensions (mm)
L_BASE = 80.0            # base length, x
W_BASE = 60.0            # bracket width, y
X_HOLE = 60.0            # bolt hole centres, x


def lame_3d(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def check_valid(tb, tw, rf, Hw, rh):
    """Geometric feasibility. Raises ValueError for impossible combinations."""
    if rf > Hw - tb - 4.0:
        raise ValueError("fillet too large for web height")
    if rf > L_BASE - tw - 2 * rh - 20.0:
        raise ValueError("fillet too large for base length")
    if X_HOLE + rh > L_BASE - 3.0:
        raise ValueError("hole too close to base edge")
    if rh > W_BASE / 4.0 - 3.0:
        raise ValueError("hole too large for bracket width")


def build_mesh(tb, tw, rf, Hw, rh, mesh_fraction=3.0):
    """L-profile in the x-z plane, extruded in y, with two bolt holes."""
    check_valid(tb, tw, rf, Hw, rh)

    # A previous call may have died before finalize() ran, which leaves gmsh
    # initialized inside a long-lived process. Clear it first.
    if gmsh.isInitialized():
        gmsh.finalize()

    # interruptible=False skips gmsh's SIGINT handler, which Python only
    # permits from the main thread (needed when called from a worker thread).
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bracket")

    h_fine = max(rf / mesh_fraction, 0.6)
    h_coarse = L_BASE / 8.0

    try:
        occ = gmsh.model.occ

        # L-shaped cross-section, drawn counter-clockwise in the x-z plane.
        p = [
            occ.addPoint(0.0, 0.0, 0.0),              # 0  outer heel
            occ.addPoint(L_BASE, 0.0, 0.0),           # 1  base tip, bottom
            occ.addPoint(L_BASE, 0.0, tb),            # 2  base tip, top
            occ.addPoint(tw + rf, 0.0, tb),           # 3  fillet start
            occ.addPoint(tw, 0.0, tb + rf),           # 4  fillet end
            occ.addPoint(tw, 0.0, Hw),                # 5  web tip, inner
            occ.addPoint(0.0, 0.0, Hw),               # 6  web tip, outer
        ]
        centre = occ.addPoint(tw + rf, 0.0, tb + rf)  # fillet centre

        curves = [
            occ.addLine(p[0], p[1]),
            occ.addLine(p[1], p[2]),
            occ.addLine(p[2], p[3]),
            occ.addCircleArc(p[3], centre, p[4]),
            occ.addLine(p[4], p[5]),
            occ.addLine(p[5], p[6]),
            occ.addLine(p[6], p[0]),
        ]
        loop = occ.addCurveLoop(curves)
        face = occ.addPlaneSurface([loop])

        # Extrude the profile across the bracket width.
        ext = occ.extrude([(2, face)], 0.0, W_BASE, 0.0)
        solid = [e for e in ext if e[0] == 3]

        # Two bolt holes through the base plate.
        holes = []
        for y in (W_BASE / 4.0, 3.0 * W_BASE / 4.0):
            holes.append((3, occ.addCylinder(X_HOLE, y, -1.0,
                                             0.0, 0.0, tb + 2.0, rh)))

        out, _ = occ.cut(solid, holes)
        occ.synchronize()

        vols = [v[1] for v in out]
        gmsh.model.addPhysicalGroup(3, vols, tag=1)

        # Refine near the fillet: that curved surface carries the concentration.
        fillet_surfs = []
        for dim, tag in gmsh.model.getBoundary(out, oriented=False):
            if dim != 2:
                continue
            com = occ.getCenterOfMass(dim, tag)
            near_corner = (abs(com[0] - (tw + rf * 0.3)) < rf
                           and abs(com[2] - (tb + rf * 0.3)) < rf)
            if near_corner:
                fillet_surfs.append(tag)

        if fillet_surfs:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "SurfacesList", fillet_surfs)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "InField", 1)
            gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
            gmsh.model.mesh.field.setNumber(2, "SizeMax", h_coarse)
            gmsh.model.mesh.field.setNumber(2, "DistMin", rf * 0.5)
            gmsh.model.mesh.field.setNumber(2, "DistMax", rf * 4.0)
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", h_coarse)
            gmsh.option.setNumber("Mesh.MeshSizeMin", h_fine)

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
        raise RuntimeError("no tetrahedra in mesh")

    return MeshTet(m.points.T, tets)


def simulate3d(tb, tw, rf, Hw, rh, mesh_fraction=3.0, return_field=False):
    """Solve one bracket geometry.

    return_field=True additionally returns the boundary surface triangles and
    their von Mises values, for plotting. The sweep never asks for it.
    """
    t_start = time.time()
    mesh = build_mesh(tb, tw, rf, Hw, rh, mesh_fraction)

    element = ElementVector(ElementTetP2())
    basis = Basis(mesh, element, intorder=2)
    lam, mu = lame_3d(E_MODULUS, NU)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis)

    # Load: traction in +x on the top face of the web (z = Hw).
    top = mesh.facets_satisfying(lambda x: np.abs(x[2] - Hw) < 1e-6)
    fbasis = FacetBasis(mesh, element, facets=top, intorder=2)
    traction = FORCE_TOTAL / (tw * W_BASE)

    @LinearForm
    def load(v, w):
        return traction * v[0]

    f = asm(load, fbasis)

    # Fully clamp the underside of the base plate (bolted to a rigid surface).
    fixed = basis.get_dofs(lambda x: np.abs(x[2]) < 1e-6).all()

    u = solve(*condense(K, f, D=fixed))

    disp = u[basis.nodal_dofs]
    max_disp = float(np.max(np.linalg.norm(disp, axis=0)))

    # Element-wise von Mises at quadrature points
    eps = sym_grad(basis.interpolate(u))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = [[lam * tr * (1.0 if i == j else 0.0) + 2.0 * mu * eps[i, j]
          for j in range(3)] for i in range(3)]
    vm = np.sqrt(0.5 * ((s[0][0] - s[1][1]) ** 2
                        + (s[1][1] - s[2][2]) ** 2
                        + (s[2][2] - s[0][0]) ** 2
                        + 6.0 * (s[0][1] ** 2 + s[1][2] ** 2 + s[2][0] ** 2)))
    vm_elem = vm.max(axis=1)

    # The global maximum sits at the clamped edge, where the boundary
    # condition is singular and the value is mesh dependent. The fillet
    # stress is the physically meaningful label.
    centroids = mesh.p[:, mesh.t].mean(axis=1)
    in_fillet = ((centroids[0] < tw + 3.0 * rf)
                 & (centroids[2] > tb * 0.6)
                 & (centroids[2] < tb + 3.0 * rf))

    max_vm_global = float(vm_elem.max())
    max_vm_fillet = (float(vm_elem[in_fillet].max())
                     if in_fillet.any() else float("nan"))

    # Nominal bending stress in the web at the fillet root, for reference.
    moment = FORCE_TOTAL * (Hw - tb - rf)
    section_modulus = W_BASE * tw ** 2 / 6.0
    nominal_bending = moment / section_modulus

    out = {
        "tb": tb, "tw": tw, "rf": rf, "Hw": Hw, "rh": rh,
        "max_vm_fillet": max_vm_fillet,
        "max_vm_global": max_vm_global,
        "max_disp": max_disp,
        "nominal_bending": nominal_bending,
        "Kt_fillet": max_vm_fillet / nominal_bending,
        "n_dofs": basis.N,
        "n_elements": mesh.t.shape[1],
        "solve_time_s": round(time.time() - t_start, 2),
    }

    if return_field:
        # Only the outer surface is drawn: each boundary facet takes the
        # stress of the single element it belongs to.
        bfacets = mesh.boundary_facets()
        facet_nodes = mesh.facets[:, bfacets]          # (3, n_bfacets)
        owner = mesh.f2t[0, bfacets]                   # element behind facet
        out["field"] = {
            "points": mesh.p.copy(),                   # (3, n_nodes)
            "surf_tris": facet_nodes.copy(),           # (3, n_bfacets)
            "surf_vm": vm_elem[owner].copy(),          # one value per facet
        }

    return out


if __name__ == "__main__":
    res = simulate3d(tb=8.0, tw=8.0, rf=6.0, Hw=50.0, rh=4.0)
    for k, v in res.items():
        print(f"{k:>18}: {v}")
