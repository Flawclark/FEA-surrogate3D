"""
simulate.py — one parametric 2D FEA run.

Plate with a central hole, loaded in tension along x.
Units: mm, N, MPa.

simulate(W, H, R, t) -> dict with max von Mises stress, max displacement, etc.
"""

import numpy as np
import gmsh
import meshio
import tempfile
import os

from skfem import (
    MeshTri, Basis, ElementVector, ElementTriP2, FacetBasis,
    BilinearForm, LinearForm, asm, condense, solve
)
from skfem.helpers import ddot, sym_grad, eye, trace

E_MODULUS = 210_000.0   # MPa (steel)
NU = 0.3
FORCE_TOTAL = 10_000.0  # N applied on the right edge


def build_mesh(W, H, R, mesh_fraction=8.0):
    """Parametric geometry + mesh in gmsh. Returns a skfem MeshTri."""
    # A previous call may have died before finalize() ran (this happens when
    # the first initialize() raised), which leaves gmsh initialized inside a
    # long-lived process such as a Streamlit server. Clear it first.
    if gmsh.isInitialized():
        gmsh.finalize()

    # interruptible=False skips gmsh's SIGINT handler, which Python only
    # permits from the main thread. Required when called from a worker thread.
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")

    # Mesh sizing rule: tie element size to the smallest feature (the hole).
    h_fine = R / mesh_fraction
    h_coarse = min(W, H) / 12.0

    try:
        # Outer rectangle, corners at (0,0) .. (W,H)
        rect = gmsh.model.occ.addRectangle(0, 0, 0, W, H)
        hole = gmsh.model.occ.addDisk(W / 2, H / 2, 0, R, R)
        out, _ = gmsh.model.occ.cut([(2, rect)], [(2, hole)])
        gmsh.model.occ.synchronize()

        surf = [s[1] for s in out]
        gmsh.model.addPhysicalGroup(2, surf, tag=1)

        # Size field: fine near the hole, coarse away from it
        gmsh.model.mesh.field.add("Distance", 1)
        hole_curves = []
        for dim, tag in gmsh.model.getBoundary(out, oriented=False):
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            # the hole boundary is the curve whose centre of mass is the hole centre
            if abs(com[0] - W / 2) < 1e-6 and abs(com[1] - H / 2) < 1e-6:
                hole_curves.append(tag)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curves)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(2, "DistMin", R * 0.5)
        gmsh.model.mesh.field.setNumber(2, "DistMax", R * 3.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)

        fd, path = tempfile.mkstemp(suffix=".msh")
        os.close(fd)
        gmsh.write(path)
    finally:
        gmsh.finalize()

    m = meshio.read(path)
    os.remove(path)

    pts = m.points[:, :2].T
    tris = None
    for cell_block in m.cells:
        if cell_block.type == "triangle":
            tris = cell_block.data.T
    if tris is None:
        raise RuntimeError("no triangles in mesh")

    return MeshTri(pts, tris)


def lame_plane_stress(E, nu):
    """Lame parameters modified so a plane-strain form gives plane stress."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    lam_bar = 2 * lam * mu / (lam + 2 * mu)
    return lam_bar, mu


def simulate(W, H, R, t, mesh_fraction=8.0, return_field=False):
    """Solve one geometry.

    return_field=True additionally returns the mesh and the element-wise
    von Mises field, for plotting. The sweep never asks for it, so the
    dataset generation path is unchanged.
    """
    if R >= 0.45 * min(W, H):
        raise ValueError("hole too large for plate")

    mesh = build_mesh(W, H, R, mesh_fraction)
    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element, intorder=4)

    lam, mu = lame_plane_stress(E_MODULUS, NU)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis)

    # Traction on the right edge (x = W)
    right = basis.mesh.facets_satisfying(lambda x: np.abs(x[0] - W) < 1e-8)
    fbasis = FacetBasis(mesh, element, facets=right, intorder=4)
    traction = FORCE_TOTAL / (H * t)  # MPa

    @LinearForm
    def load(v, w):
        return traction * v[0]

    f = asm(load, fbasis)

    # BCs: left edge fixed in x, plus ONE node fixed in y so the system is
    # not singular. get_dofs() works on facets, so a point predicate matches
    # nothing -- the y constraint must be picked from the nodes directly.
    dofs_left_x = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-8).all('u^1')

    left_nodes = np.flatnonzero(np.abs(mesh.p[0]) < 1e-8)
    anchor = left_nodes[np.argmin(mesh.p[1][left_nodes])]   # lowest-left node
    dof_anchor_y = basis.nodal_dofs[1, anchor]

    fixed = np.unique(np.append(dofs_left_x, dof_anchor_y))

    u = solve(*condense(K, f, D=fixed))

    # --- post-processing ---
    disp = u[basis.nodal_dofs]                       # (2, n_nodes)
    max_disp = float(np.max(np.linalg.norm(disp, axis=0)))

    # element-wise stress at quadrature points
    sbasis = basis.with_element(element)
    eps = sym_grad(sbasis.interpolate(u))
    tr = eps[0, 0] + eps[1, 1]
    sxx = lam * tr + 2 * mu * eps[0, 0]
    syy = lam * tr + 2 * mu * eps[1, 1]
    sxy = 2 * mu * eps[0, 1]
    vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    max_vm = float(np.max(vm))

    net_area = (H - 2 * R) * t
    nominal = FORCE_TOTAL / net_area

    out = {
        "W": W, "H": H, "R": R, "t": t,
        "max_von_mises": max_vm,
        "max_disp": max_disp,
        "nominal_stress_net": nominal,
        "Kt_net": max_vm / nominal,
        "n_dofs": basis.N,
        "n_elements": mesh.t.shape[1],
    }

    if return_field:
        out["field"] = {
            "points": mesh.p.copy(),              # (2, n_nodes)
            "tris": mesh.t.copy(),                # (3, n_elements)
            # max over quadrature points, matching how max_von_mises above
            # is reduced -- so the plotted peak equals the reported scalar
            "vm_elem": vm.max(axis=1),
            "disp": disp.copy(),                  # (2, n_nodes)
        }

    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = simulate(W=100.0, H=100.0, R=10.0, t=5.0)
    res["solve_time_s"] = round(time.time() - t0, 2)
    for k, v in res.items():
        print(f"{k:>22}: {v}")
