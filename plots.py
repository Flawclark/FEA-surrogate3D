"""
plots.py — geometry previews and stress heat maps for the demonstrator.

Geometry drawing is instant and needs no solve, so it updates live with the
sliders. Heat maps need a real FEA result and are drawn after one is run.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CMAP = "turbo"
EDGE = "#1B3B8F"
FILL = "#eef2fa"


# --------------------------------------------------------------- geometry

def geometry_2d(W, H, R):
    """Scale drawing of the plate, with load and support marked."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    ax.add_patch(Rectangle((0, 0), W, H, facecolor=FILL,
                           edgecolor=EDGE, linewidth=1.8))
    ax.add_patch(Circle((W / 2, H / 2), R, facecolor="white",
                        edgecolor=EDGE, linewidth=1.8))

    # fixed left edge
    ax.plot([0, 0], [0, H], color="#333", linewidth=3)
    for y in np.linspace(0, H, 12):
        ax.plot([-0.05 * W, 0], [y - 0.03 * H, y], color="#888", linewidth=1)

    # load on the right edge
    for y in np.linspace(0.12 * H, 0.88 * H, 5):
        ax.annotate("", xy=(W + 0.14 * W, y), xytext=(W, y),
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.6))
    ax.text(W + 0.16 * W, H / 2, "10 kN", color="#C0392B",
            va="center", fontsize=10)

    ax.set_xlim(-0.16 * W, W + 0.34 * W)
    ax.set_ylim(-0.12 * H, 1.12 * H)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"W={W:.0f}  H={H:.0f}  R={R:.1f} mm", fontsize=10)
    fig.tight_layout()
    return fig


def _bracket_profile(tb, tw, rf, Hw, n_arc=24):
    """Outline of the L cross-section, fillet included."""
    L = 80.0
    cx, cy = tw + rf, tb + rf
    # fillet arc: from the base top surface round to the web inner face
    ang = np.linspace(np.pi * 1.5, np.pi, n_arc)
    arc = np.column_stack([cx + rf * np.cos(ang), cy + rf * np.sin(ang)])
    pts = [(0, 0), (L, 0), (L, tb), (tw + rf, tb)]
    pts += [tuple(p) for p in arc]
    pts += [(tw, Hw), (0, Hw)]
    return np.array(pts)


def geometry_3d(tb, tw, rf, Hw, rh):
    """Side and plan views of the bracket, to scale."""
    L, Wb, xh = 80.0, 60.0, 60.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # --- side view
    poly = _bracket_profile(tb, tw, rf, Hw)
    ax1.add_patch(Polygon(poly, closed=True, facecolor=FILL,
                          edgecolor=EDGE, linewidth=1.8))
    ax1.plot([0, L], [0, 0], color="#333", linewidth=3)
    for x in np.linspace(0, L, 14):
        ax1.plot([x - 3, x], [-4, 0], color="#888", linewidth=1)
    ax1.annotate("", xy=(tw + 26, Hw), xytext=(tw, Hw),
                 arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2))
    ax1.text(tw + 29, Hw, "1 kN", color="#C0392B", va="center", fontsize=10)
    ax1.set_xlim(-8, L + 30)
    ax1.set_ylim(-10, max(Hw, 40) + 12)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("z (mm)")
    ax1.set_title("side view", fontsize=10)

    # --- plan view
    ax2.add_patch(Rectangle((0, 0), L, Wb, facecolor="#f6f8fd",
                            edgecolor=EDGE, linewidth=1.8))
    ax2.add_patch(Rectangle((0, 0), tw, Wb, facecolor="#d8e1f5",
                            edgecolor=EDGE, linewidth=1.0))
    for y in (Wb / 4, 3 * Wb / 4):
        ax2.add_patch(Circle((xh, y), rh, facecolor="white",
                             edgecolor=EDGE, linewidth=1.5))
    ax2.set_xlim(-6, L + 6)
    ax2.set_ylim(-6, Wb + 6)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)")
    ax2.set_title("plan view", fontsize=10)

    fig.suptitle(f"tb={tb:.1f}  tw={tw:.1f}  rf={rf:.1f}  "
                 f"Hw={Hw:.0f}  rh={rh:.2f} mm", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


# --------------------------------------------------------------- heat maps

def heatmap_2d(field, scale=0.0):
    """Element-wise von Mises on the deformed or undeformed mesh."""
    pts = field["points"].copy()
    if scale:
        pts = pts + scale * field["disp"]

    tris = field["tris"]
    verts = [pts[:, tris[:, e]].T for e in range(tris.shape[1])]
    vm = field["vm_elem"]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    coll = PolyCollection(verts, array=vm, cmap=CMAP,
                          edgecolors="none", linewidths=0)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    # The peak goes in the title rather than a floating callout, which
    # collided with the hole or the colourbar depending on geometry.
    peak = int(np.argmax(vm))
    cx, cy = pts[:, tris[:, peak]].mean(axis=1)
    ax.plot(cx, cy, "o", mfc="none", mec="white", mew=2.0, ms=14)
    ax.set_title(f"von Mises stress — peak {vm.max():.1f} MPa (circled)",
                 fontsize=10)

    fig.colorbar(coll, ax=ax, shrink=0.85, label="MPa")
    fig.tight_layout()
    return fig


def heatmap_3d(field, elev=25, azim=-45, clip_percentile=99.0):
    """Surface von Mises on the bracket.

    The colour scale is clipped at a high percentile because the clamped
    edge is a boundary-condition singularity: left unclipped it dominates
    the scale and the fillet, which is the region of interest, washes out.
    """
    pts = field["points"]
    tris = field["surf_tris"]
    vm = field["surf_vm"]

    vmax = float(np.percentile(vm, clip_percentile))
    vmin = float(vm.min())

    verts = [pts[:, tris[:, i]].T for i in range(tris.shape[1])]

    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[CMAP]
    coll = Poly3DCollection(verts, facecolors=cmap(norm(vm)),
                            edgecolors="none", linewidths=0)
    ax.add_collection3d(coll)

    lo = pts.min(axis=1)
    hi = pts.max(axis=1)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # true proportions, without padding the view out to a cube
    ax.set_box_aspect(tuple(hi - lo))

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (mm)", labelpad=2)
    ax.set_ylabel("y (mm)", labelpad=2)
    ax.set_zlabel("z (mm)", labelpad=2)
    ax.tick_params(labelsize=8)
    ax.set_title("surface von Mises stress", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.12, label="MPa")
    cb.ax.tick_params(labelsize=8)
    fig.text(0.02, 0.96,
             f"colour scale clipped at the {clip_percentile:.0f}th percentile "
             "— the clamped edge is a singularity, not a real stress",
             fontsize=7.5, color="#555", va="top")
    return fig


# ------------------------------------------------- user-defined parts

def geometry_custom(part, params=None, show_labels=True):
    """Draw a user-defined part with its segments numbered.

    The numbering is what the editor uses to attach loads and supports, so
    it has to be visible on the drawing.
    """
    import custom2d

    pts, holes, _ = custom2d.resolve(part, params)
    n = len(pts)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.add_patch(Polygon(pts, closed=True, facecolor=FILL,
                         edgecolor=EDGE, linewidth=1.8))
    for hx, hy, hr in holes:
        ax.add_patch(Circle((hx, hy), hr, facecolor="white",
                            edgecolor=EDGE, linewidth=1.6))

    span = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    segs = {int(k): v for k, v in part.get("segments", {}).items()}

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        mid = (a + b) / 2.0
        d = b - a
        L = float(np.hypot(*d)) or 1.0
        # outward normal, assuming a counter-clockwise outline
        nrm = np.array([d[1], -d[0]]) / L

        spec = segs.get(i, {"kind": "free"})
        kind = spec.get("kind", "free")

        if kind == "fixed":
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#333", linewidth=3.5)
            for s in np.linspace(0.08, 0.92, 7):
                base = a + s * d
                tip = base + nrm * span * 0.035
                ax.plot([base[0], tip[0]], [base[1], tip[1]],
                        color="#888", linewidth=1.1)
        elif kind == "load":
            fx, fy = float(spec.get("fx", 0.0)), float(spec.get("fy", 0.0))
            mag = math.hypot(fx, fy)
            if mag > 0:
                ux, uy = fx / mag, fy / mag
                for s in np.linspace(0.15, 0.85, 4):
                    base = a + s * d
                    ax.annotate("", xy=(base[0] + ux * span * 0.1,
                                        base[1] + uy * span * 0.1),
                                xytext=(base[0], base[1]),
                                arrowprops=dict(arrowstyle="->",
                                                color="#C0392B", lw=1.6))
                ax.text(mid[0] + ux * span * 0.16,
                        mid[1] + uy * span * 0.16,
                        f"{mag:.0f} N", color="#C0392B", fontsize=9,
                        ha="center", va="center")

        if show_labels:
            lab = mid + nrm * span * 0.055
            colour = {"fixed": "#333", "load": "#C0392B"}.get(kind, "#1B3B8F")
            ax.text(lab[0], lab[1], str(i), fontsize=9, color=colour,
                    ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                              ec=colour, lw=1.0))

    pad = span * 0.22
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(part.get("name", "part"), fontsize=10)
    fig.tight_layout()
    return fig


def geometry_custom3d(part, params=None, elev=24, azim=-55):
    """Profile view with numbered segments, next to the extruded solid.

    The profile carries the segment numbers because that is how face names
    are formed (segment i raises side_i); the 3D view shows what the numbers
    correspond to once extruded.
    """
    import custom3d

    pts, holes, depth = custom3d.resolve(part, params)
    n = len(pts)
    faces = part.get("faces", {})
    span = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))

    fig = plt.figure(figsize=(10.4, 4.4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection="3d")

    def kind_of(name):
        return faces.get(name, {}).get("kind", "free")

    colour = {"fixed": "#333333", "load": "#C0392B", "free": EDGE}

    # ---------- left: profile with segment numbers
    ax1.add_patch(Polygon(pts, closed=True, facecolor=FILL,
                          edgecolor=EDGE, linewidth=1.6))
    for hx, hy, hr in holes:
        ax1.add_patch(Circle((hx, hy), hr, facecolor="white",
                             edgecolor=EDGE, linewidth=1.4))

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        mid = (a + b) / 2.0
        d = b - a
        L = float(np.hypot(*d)) or 1.0
        nrm = np.array([d[1], -d[0]]) / L
        k = kind_of(f"side_{i}")
        if k != "free":
            ax1.plot([a[0], b[0]], [a[1], b[1]], color=colour[k], linewidth=3.2)
        lab = mid + nrm * span * 0.06
        ax1.text(lab[0], lab[1], str(i), fontsize=9, ha="center", va="center",
                 color=colour[k], weight="bold",
                 bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                           ec=colour[k], lw=1.0))

    pad = span * 0.2
    ax1.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax1.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)")
    ax1.set_title("profile — segment i raises side_i", fontsize=10)

    # ---------- right: the extruded solid
    polys, cols = [], []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        polys.append([(a[0], a[1], 0.0), (b[0], b[1], 0.0),
                      (b[0], b[1], depth), (a[0], a[1], depth)])
        k = kind_of(f"side_{i}")
        cols.append({"fixed": "#9a9a9a", "load": "#e8a99b",
                     "free": "#cfdaf0"}[k])

    for z, nm in ((0.0, "bottom"), (depth, "top")):
        polys.append([(x, y, z) for x, y in pts])
        k = kind_of(nm)
        cols.append({"fixed": "#9a9a9a", "load": "#e8a99b",
                     "free": "#e8eefb"}[k])

    coll = Poly3DCollection(polys, facecolors=cols, edgecolors=EDGE,
                            linewidths=0.6, alpha=0.92)
    ax2.add_collection3d(coll)

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        mid = (a + b) / 2.0
        k = kind_of(f"side_{i}")
        if k != "free":
            ax2.text(mid[0], mid[1], depth / 2.0, f"side_{i}",
                     fontsize=7.5, color=colour[k], weight="bold")

    lo = np.array([pts[:, 0].min(), pts[:, 1].min(), 0.0])
    hi = np.array([pts[:, 0].max(), pts[:, 1].max(), depth])
    ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(lo[1], hi[1])
    ax2.set_zlim(lo[2], hi[2])
    ax2.set_box_aspect(tuple(np.maximum(hi - lo, 1e-6)))
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xlabel("x", labelpad=1); ax2.set_ylabel("y", labelpad=1)
    ax2.set_zlabel("z", labelpad=1)
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"extruded, depth {depth:.1f} mm", fontsize=10)

    fig.suptitle(part.get("name", "part"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
