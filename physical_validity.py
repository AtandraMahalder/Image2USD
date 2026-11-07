"""Deterministic, IMAGES-ONLY physical-validity check for a PREDICTED scene.

Uses ONLY the prediction's own prims + the rules in asset_directory.py -- no GT USD -- so it is a
valid reward term under the images-only constraint. TRAINING-time constraints (the reward judges the
model's training-time output).

Checks:
  (1) OUT-OF-BOUNDS: object.location outside the background's scene `bounds` (off-screen / through
      walls/floor -- the base342 place-value 'flung object' failure the reward never penalized).
  (2) BG-INCOMPATIBLE: object type not in `environment_catgories[bg]` (training-time; warehouse=all).
  (3) INTERSECTION: object OBBs collide (OBB + SAT, the exact method generate_train_data.py uses to
      REJECT colliding placements -> GT is collision-free by construction). Valid container<->contents
      pairs (shelf_info: small objects / crates on racks/tables/shelves) are EXCLUDED, matching the
      generator which collision-checks representative unit bounds, not shelved objects individually.

CLI (same style as compute_* scripts):
  python physical_validity.py --eval_dataset test_data --suffix image_2_usd
  python physical_validity.py --eval_dataset test_data --suffix code    # GT sanity (should be ~0)
"""
import os
import math
import argparse
import numpy as np
from pxr import Usd, UsdGeom
from natsort import natsorted
from tqdm import tqdm
from utils.utils import extract_prims_from_usda
from utils.asset_directory import bounds, environment_catgories, shelf_info, address, scale

BG = {"room", "warehouse_1", "warehouse_2"}
ASSET_PATH = os.path.realpath("assets")

# valid container -> set of contents it may legitimately hold (from shelf_info; each shelf entry is
# [height, left, right, acceptable_objects]). A container<->contents overlap is EXPECTED, not a bug.
CONTAINERS = set(shelf_info.keys())
CONTAINER_CONTENTS = {c: set().union(*[set(shelf[3]) for shelf in info[2:]]) if len(info) > 2 else set()
                      for c, info in shelf_info.items()}


# ---------------- OBB + SAT (copied from generate_train_data.py; identical method) ----------------
def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _rot_from_euler(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _transform_bbox(corners, R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    pts = np.column_stack([np.array(corners), np.ones(len(corners))])
    return ((T @ pts.T).T)[:, :3]


def _obb_axes(corners):
    corners = np.array(corners)
    c0 = corners[0]
    d = [np.linalg.norm(c0 - c) for c in corners[1:]]
    adj = np.argsort(d)[:3] + 1
    return [_normalize(corners[i] - c0) for i in adj]


def _obb_intersect(a, b):
    """SAT test between two oriented bounding boxes (8-corner arrays)."""
    axes = _obb_axes(a) + _obb_axes(b)
    for ax_a in _obb_axes(a):
        for ax_b in _obb_axes(b):
            cr = np.cross(ax_a, ax_b)
            if np.linalg.norm(cr) > 1e-6:
                axes.append(_normalize(cr))
    for ax in axes:
        pa = np.array(a) @ ax; pb = np.array(b) @ ax
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False   # separating axis found
    return True


# ---------------- per-asset identity bbox (lazy, cached) ----------------
_BBOX = {}


def _bbox(prim_type):
    if prim_type in _BBOX:
        return _BBOX[prim_type]
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.centimeters)
    p = stage.DefinePrim("/World/prim")
    p.GetReferences().AddReference(os.path.join(ASSET_PATH, address[prim_type]))
    UsdGeom.XformCommonAPI(p).SetScale(scale[prim_type])
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]) \
        .ComputeWorldBound(p).ComputeAlignedBox()
    mn, mx = r.GetMin(), r.GetMax()
    corners = [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    _BBOX[prim_type] = corners
    return corners


def _world_obb(prim):
    R = _rot_from_euler(*[float(a) * np.pi / 180 for a in prim.rotation])
    return _transform_bbox(_bbox(prim.type), R, [float(v) for v in prim.location])


# tolerances (cm). An object is 'supported' if on the floor OR at a rack's shelf level within its x-y
# footprint. Objects on the same rack form one placed UNIT; the generator only collision-checks unit
# representative bounds (so shelf-mates may overlap) -> same-unit pairs are excluded from intersection.
Z_TOL = 8.0
LAT_MARGIN = 30.0
INTERSECT_MARGIN = 0.12   # shrink OBBs toward center before SAT so touching / loose-bbox grazing isn't flagged


def _shrink(corners, m=INTERSECT_MARGIN):
    c = np.array(corners); ctr = c.mean(axis=0)
    return ctr + (c - ctr) * (1 - m)


def _rack_info(fg):
    """(index, x_min, x_max, y_min, y_max, world_shelf_levels) for each rack/shelf/table present."""
    info = []
    for i, p in enumerate(fg):
        if p.type in CONTAINERS:
            o = np.array(_world_obb(p)); rz = float(p.location[2])
            levels = [rz + shelf[0] for shelf in shelf_info[p.type][2:]]
            info.append((i, o[:, 0].min(), o[:, 0].max(), o[:, 1].min(), o[:, 1].max(), levels))
    return info


def _supporting_rack(p, racks):
    """Index of the rack this object validly rests on (shelf level + footprint), else None."""
    x, y, z = (float(v) for v in p.location)
    for (ri, xmn, xmx, ymn, ymx, levels) in racks:
        if any(abs(z - lvl) <= Z_TOL for lvl in levels) \
           and (xmn - LAT_MARGIN <= x <= xmx + LAT_MARGIN) and (ymn - LAT_MARGIN <= y <= ymx + LAT_MARGIN):
            return ri
    return None


def count_intersections(fg):
    """# of colliding pairs (shrunk OBB + SAT). Objects in the same placed UNIT (a rack and the
    objects on its shelves) are NOT checked -- the generator only collision-checks unit bounds, so
    shelf-mates may overlap by design. Only cross-unit penetration is a real collision."""
    obbs = [_shrink(_world_obb(p)) for p in fg]
    racks = _rack_info(fg)
    unit = [None if p.type in CONTAINERS else _supporting_rack(p, racks) for p in fg]  # rack index or None
    n = 0
    for i in range(len(fg)):
        for j in range(i + 1, len(fg)):
            same_unit = (unit[i] is not None and unit[i] == unit[j]) or unit[i] == j or unit[j] == i
            if same_unit:
                continue
            if _obb_intersect(obbs[i], obbs[j]):
                n += 1
    return n


def count_floating(fg, bg):
    """# of objects hovering with no support: elevated but not on the floor and not on any rack's
    shelf level within its footprint (rack placement per generate_train_data.py)."""
    floor_z = bounds[bg]["z"][0]
    racks = _rack_info(fg)
    n_float = 0
    for p in fg:
        if p.type in CONTAINERS:
            continue
        if float(p.location[2]) <= floor_z + Z_TOL:
            continue                                  # resting on the floor -> fine
        if _supporting_rack(p, racks) is None:        # elevated with no supporting rack -> floating
            n_float += 1
    return n_float


def physical_validity(prims, check_intersect=True):
    bg = next((p.type for p in prims if p.type in BG), None)
    fg = [p for p in prims if p.type not in BG]
    n = len(fg)
    if n == 0:
        return {"validity": 0.0, "n_obj": 0, "n_oob": 0, "n_incompat": 0, "n_float": 0, "n_intersect": 0, "bg": bg, "checkable": False}
    if bg is None or bg not in bounds:
        return {"validity": 1.0, "n_obj": n, "n_oob": 0, "n_incompat": 0, "n_float": 0, "n_intersect": 0, "bg": bg, "checkable": False}

    bb = bounds[bg]
    allowed = set(environment_catgories.get(bg, []))
    n_oob = n_incompat = 0
    for p in fg:
        x, y, z = (float(v) for v in p.location)
        if not (bb["x"][0] <= x <= bb["x"][1] and bb["y"][0] <= y <= bb["y"][1] and bb["z"][0] <= z <= bb["z"][1]):
            n_oob += 1
        if allowed and p.type not in allowed:
            n_incompat += 1
    n_intersect = count_intersections(fg) if check_intersect else 0
    n_float = count_floating(fg, bg)

    # object-level violation fraction: OOB + incompat + floating over objects, plus intersecting-pairs
    # (each implicates ~1 offending object) -> validity in [0,1]
    violations = n_oob + n_incompat + n_float + n_intersect
    validity = max(0.0, 1.0 - violations / n)
    return {"validity": validity, "n_obj": n, "n_oob": n_oob, "n_incompat": n_incompat,
            "n_float": n_float, "n_intersect": n_intersect, "bg": bg, "checkable": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dataset", default="test_data")
    ap.add_argument("--suffix", default="image_2_usd", help="pred_<suffix>.usda; use 'code' for GT")
    ap.add_argument("--percent_eval", type=float, default=1.0)
    ap.add_argument("--no_intersect", action="store_true", help="skip the (slower) intersection check")
    args = ap.parse_args()
    fname = "code.usda" if args.suffix == "code" else f"pred_{args.suffix}.usda"

    tot = {"validity": 0.0, "oob": 0, "incompat": 0, "float": 0, "intersect": 0, "obj": 0}
    n_samples = 0
    for num_obj in tqdm(natsorted(os.listdir(args.eval_dataset)), desc="Object Count"):
        samples = natsorted(os.listdir(os.path.join(args.eval_dataset, num_obj)))
        samples = samples[:math.floor(len(samples) * args.percent_eval)]
        m = {"validity": 0.0, "oob": 0, "incompat": 0, "float": 0, "intersect": 0, "obj": 0}; nb = 0
        for s in samples:
            path = os.path.join(args.eval_dataset, num_obj, s, fname)
            if not os.path.exists(path):
                continue
            try:
                prims = extract_prims_from_usda(open(path).read(), ASSET_PATH)[0]
                r = physical_validity(prims, check_intersect=not args.no_intersect)
                m["validity"] += r["validity"]; m["oob"] += r["n_oob"]; m["incompat"] += r["n_incompat"]
                m["float"] += r["n_float"]; m["intersect"] += r["n_intersect"]; m["obj"] += r["n_obj"]; nb += 1
            except Exception as e:
                tqdm.write(f"{num_obj}/{s}: {e}")
        if nb:
            print(f"  {num_obj:>3} obj (n={nb}): validity={m['validity']/nb:.3f}  OOB={m['oob']}  "
                  f"floating={m['float']}  intersect_pairs={m['intersect']}  incompat={m['incompat']}  (objs={m['obj']})", flush=True)
        for k in tot:
            tot[k] += m[k]
        n_samples += nb

    N = max(1, n_samples)
    print("=" * 24)
    print(f"OVERALL ({args.suffix}) on {args.eval_dataset}  [n={n_samples}]")
    print(f"  mean validity={tot['validity']/N:.3f}   OOB={tot['oob']}/{tot['obj']} "
          f"({100*tot['oob']/max(1,tot['obj']):.1f}%)   floating={tot['float']}   "
          f"intersecting_pairs={tot['intersect']}   incompat={tot['incompat']}")


if __name__ == "__main__":
    main()
