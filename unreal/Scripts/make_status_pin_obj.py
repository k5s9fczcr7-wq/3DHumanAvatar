"""Status-pin mesh for the kiosk (a realistic frame around the lens, so it looks like an actual object).
A lathe-turned pin, centimetres, +Z = front, pivot at the middle of its thickness so the
FabricPinComponent can seat it by half its thickness. Material groups: Lens (slot 0, recoloured by the indicator
Blueprint) then Bezel (slot 1, metal: ring + back plate)."""
import math
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "SM_StatusPin.obj"
SEG = 96                              # around the axis
R_OUT, R_IN = 1.90, 1.42              # bezel outer radius, lens radius (cm)
Z_BACK, Z_CREST = -0.35, 0.35         # back face, top of the bezel crest
Z_LENS_EDGE, Z_LENS_TOP = 0.16, 0.31  # lens meets the bezel slightly below the crest; dome apex just under it

verts, norms, uvs, faces = [], [], [], {"Lens": [], "Bezel": []}


def add(p, n, uv):
    verts.append(p); norms.append(n); uvs.append(uv)
    return len(verts)                 # OBJ is 1-based


def lathe(profile, group, v0=0.0, v1=1.0):
    """Revolve a 2D profile [(r, z, nr, nz), ...] (normals in the r-z plane) around Z into quads."""
    rings = []
    for i, (r, z, nr, nz) in enumerate(profile):
        ring = []
        for s in range(SEG + 1):
            a = 2 * math.pi * s / SEG
            c, si = math.cos(a), math.sin(a)
            ring.append(add((r * c, r * si, z), (nr * c, nr * si, nz), (s / SEG, v0 + (v1 - v0) * i / (len(profile) - 1))))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for s in range(SEG):
            a, b, c, d = rings[i][s], rings[i][s + 1], rings[i + 1][s + 1], rings[i + 1][s]
            faces[group].append((a, b, c))
            faces[group].append((a, c, d))


def profile_from_points(pts):
    """Smooth normals for a polyline profile in (r, z); travelling along it, the surface faces to the left."""
    pts = np.array(pts, dtype=float)
    out = []
    for i in range(len(pts)):
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        out.append((pts[i][0], pts[i][1], -t[1], t[0]))    # tangent rotated +90 deg: up on top, inward on the inner wall
    return out


def arc(cr, cz, rad, a0, a1, n):
    return [(cr + rad * math.cos(a0 + (a1 - a0) * i / n), cz + rad * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


# Lens: a shallow dome from the centre out to the bezel (normals from the dome's slope).
lens = []
K = 24
for i in range(1, K + 1):            # rings from the first step out; the centre is one vertex (a ring of radius ~0
    r = R_IN * i / K                  # would make needle-thin triangles with zero normals)
    z = Z_LENS_EDGE + (Z_LENS_TOP - Z_LENS_EDGE) * (1 - (r / R_IN) ** 2)
    dzdr = -2 * (Z_LENS_TOP - Z_LENS_EDGE) * r / R_IN ** 2
    n = np.array([-dzdr, 1.0]); n /= np.linalg.norm(n)
    lens.append((r, z, n[0], n[1]))
first_ring = len(verts) + 1
lathe(lens, "Lens")
apex = add((0.0, 0.0, Z_LENS_TOP), (0.0, 0.0, 1.0), (0.0, 0.0))
for s in range(SEG):
    faces["Lens"].append((apex, first_ring + s, first_ring + s + 1))

# Bezel cross-section, travelling from under the lens rim, up the inner wall, over a rounded crest, down the outer wall.
RI, RO = 0.10, 0.13                                    # rounding radii of the inner and outer top edges
pts = [(R_IN, Z_LENS_EDGE - 0.03), (R_IN, Z_CREST - RI - 0.02)]
pts += arc(R_IN + RI, Z_CREST - RI, RI, math.pi, math.pi / 2, 8)[1:]     # inner top edge
pts += arc(R_OUT - RO, Z_CREST - RO, RO, math.pi / 2, 0.0, 8)           # flat crest, then the outer top edge
pts += [(R_OUT, Z_BACK + 0.05), (R_OUT - 0.05, Z_BACK)]
lathe(profile_from_points(pts), "Bezel")

# Back plate (flat, facing -Z).
centre = add((0.0, 0.0, Z_BACK), (0.0, 0.0, -1.0), (0.5, 0.5))
rim = [add(((R_OUT - 0.06) * math.cos(2 * math.pi * s / SEG), (R_OUT - 0.06) * math.sin(2 * math.pi * s / SEG), Z_BACK),
           (0.0, 0.0, -1.0), (0.5 + 0.5 * math.cos(2 * math.pi * s / SEG), 0.5 + 0.5 * math.sin(2 * math.pi * s / SEG)))
       for s in range(SEG + 1)]
for s in range(SEG):
    faces["Bezel"].append((centre, rim[s + 1], rim[s]))

V3, N3 = np.array(verts), np.array(norms)
for g in faces:
    fixed = []
    for a, b, c in faces[g]:
        geo = np.cross(V3[b - 1] - V3[a - 1], V3[c - 1] - V3[a - 1])
        fixed.append((a, b, c) if geo @ (N3[a - 1] + N3[b - 1] + N3[c - 1]) >= 0 else (a, c, b))
    faces[g] = fixed

with open(OUT, "w", encoding="ascii") as f:
    f.write("# status pin, cm, +Z front\n")
    for v in verts:
        f.write("v %.5f %.5f %.5f\n" % v)
    for t in uvs:
        f.write("vt %.5f %.5f\n" % t)
    for n in norms:
        f.write("vn %.5f %.5f %.5f\n" % n)
    for g in ("Lens", "Bezel"):
        # One object with two material sections (a "g" line would split it). "s 1" = one smoothing group: Unreal
        # recomputes normals instead of importing the OBJ's, and without a group it makes every edge hard.
        f.write(f"usemtl {g}\ns 1\n")
        for a, b, c in faces[g]:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
V = np.array(verts)
print(f"{OUT}: {len(verts)} verts, lens {len(faces['Lens'])} + bezel {len(faces['Bezel'])} tris; "
      f"extent x {V[:,0].min():.2f}..{V[:,0].max():.2f} z {V[:,2].min():.2f}..{V[:,2].max():.2f}")
