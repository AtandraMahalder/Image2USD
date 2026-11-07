"""Per-object evaluation for image->USD predictions, matching pred_<suffix>.usda against the
GT code.usda per sample. Same convention as compute_{geometric,visual}_metrics.py: --eval_dataset
+ --suffix, per-object-count bin loop, writes obj_metrics_<suffix>.json per sample, prints per-bin
+ overall.

Matching: Hungarian assignment on 3D translation distance; a pair is a true positive only if
within --match_threshold (cm). Unmatched GT = miss, unmatched pred = hallucination.

Metrics (rotation error is intentionally left for later):
  count error   |n_pred - n_gt|            (per sample, no matching)
  recall/prec/F1                            (from matching)
  asset-recognition  fraction of matched pairs with correct type
  translation error  mean ||dt|| over matched pairs (cm)
  scale error        mean relative ||ds||/||s_gt|| over matched pairs
  camera-pose error  |dloc| (cm) + view-direction angle (deg)   (per sample)
"""
import os, sys, json, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from utils.utils import extract_prims_from_usda
from physical_validity import BG, _bbox, _transform_bbox, _rot_from_euler

ASSET = os.path.realpath("assets")

_MASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'symmask.json')
SYM = json.load(open(_MASK))   # class -> rotational order (0 = continuous)
def _fold(dy, order):
    """Yaw error modulo the object's own rotational symmetry.
    order 1 -> unchanged (max 180). order 2 -> max 90. order 4 -> max 45. order 0 -> 0 (yaw carries
    no information, so any prediction is correct)."""
    if order == 0: return 0.0
    period = 360.0 / order
    d = dy % period
    return min(d, period - d)

def scene_objects(usda):
    """-> (types list, translations Nx3, scales Nx3, yaw N), foreground only."""
    prims, _ = extract_prims_from_usda(open(usda).read(), ASSET)
    fg = [p for p in prims if p.type not in BG]
    types = [p.type for p in fg]
    t = np.array([[float(v) for v in p.location] for p in fg], dtype=np.float64).reshape(-1, 3)
    s = np.array([[float(v) for v in p.scale] for p in fg], dtype=np.float64).reshape(-1, 3)
    yaw = np.array([float(p.rotation[2]) for p in fg], dtype=np.float64).reshape(-1)   # deg; GT+preds are upright
    return types, t, s, yaw

def cam_error(gt_cam, pred_cam):
    gl = np.array(gt_cam["location"], float); gt = np.array(gt_cam["look_at"], float)
    pl = np.array(pred_cam["location"], float); pt = np.array(pred_cam["look_at"], float)
    loc_err = float(np.linalg.norm(pl - gl))
    gv = gt - gl; pv = pt - pl
    gv /= (np.linalg.norm(gv) + 1e-9); pv /= (np.linalg.norm(pv) + 1e-9)
    view_err = float(math.degrees(math.acos(np.clip(gv @ pv, -1, 1))))
    return loc_err, view_err

def _aabb(types, t, yaw):
    """World axis-aligned boxes (N,6 = xmin,ymin,zmin,xmax,ymax,zmax): the per-asset bbox rotated by
    yaw (objects are upright) and centered at t, then take the axis-aligned envelope. Uses
    physical_validity's cached _bbox (opens each asset once, then cached)."""
    out = np.empty((len(types), 6), np.float64)
    for i in range(len(types)):
        R = _rot_from_euler(0.0, 0.0, math.radians(float(yaw[i])))          # upright: yaw about Z only
        w = _transform_bbox(_bbox(types[i]), R, [float(v) for v in t[i]])   # (8,3) world OBB corners
        out[i, :3] = w.min(0); out[i, 3:] = w.max(0)
    return out

def _aabb_iou(a, b):
    """Pairwise 3D axis-aligned-box IoU. a:(M,6) b:(N,6) -> (M,N)."""
    lo = np.maximum(a[:, None, :3], b[None, :, :3])
    hi = np.minimum(a[:, None, 3:], b[None, :, 3:])
    inter = np.prod(np.clip(hi - lo, 0.0, None), axis=2)                    # (M,N) intersection volume
    va = np.prod(a[:, 3:] - a[:, :3], axis=1); vb = np.prod(b[:, 3:] - b[:, :3], axis=1)
    return inter / np.maximum(va[:, None] + vb[None, :] - inter, 1e-9)

def eval_sample(d, suffix, thr, match="dist", iou_thr=0.25):
    gt_usda = os.path.join(d, "code.usda"); pr_usda = os.path.join(d, f"pred_{suffix}.usda")
    if not (os.path.exists(gt_usda) and os.path.exists(pr_usda)):
        return None
    gt_types, gt_t, gt_s, gt_yaw = scene_objects(gt_usda)
    pr_types, pr_t, pr_s, pr_yaw = scene_objects(pr_usda)
    ng, npd = len(gt_types), len(pr_types)
    r = {"n_gt": ng, "n_pred": npd, "count_err": abs(npd - ng), "count_err_signed": npd - ng,
         "tp": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0,
         "recognition": None, "trans_err": None, "scale_err": None, "rot_err": None}
    if ng and npd:
        D = np.linalg.norm(gt_t[:, None, :] - pr_t[None, :, :], axis=2)      # ng x npd center distance (cm) -- always, for trans_err
        if match == "iou":                                                   # match on 3D AABB IoU (size-aware); TP if IoU >= iou_thr
            iou = _aabb_iou(_aabb(gt_types, gt_t, gt_yaw), _aabb(pr_types, pr_t, pr_yaw))
            gi, pj = linear_sum_assignment(-iou)
            keep = iou[gi, pj] >= iou_thr
        else:                                                                # match on center distance; TP if <= thr (cm)
            gi, pj = linear_sum_assignment(D)
            keep = D[gi, pj] <= thr
        gi, pj = gi[keep], pj[keep]
        tp = len(gi)
        r["tp"] = tp
        r["recall"] = tp / ng; r["precision"] = tp / npd
        r["f1"] = 2 * r["recall"] * r["precision"] / (r["recall"] + r["precision"]) if tp else 0.0
        if tp:
            r["recognition"] = float(np.mean([gt_types[i] == pr_types[j] for i, j in zip(gi, pj)]))
            r["trans_err"] = float(np.mean(D[gi, pj]))                        # center distance over matched (consistent across match modes)
            sden = np.linalg.norm(gt_s[gi], axis=1) + 1e-9
            r["scale_err"] = float(np.mean(np.linalg.norm(pr_s[pj] - gt_s[gi], axis=1) / sden))
            dyaw = np.abs(pr_yaw[pj] - gt_yaw[gi]) % 360.0          # NO-SYMMETRY: shortest yaw diff -> [0,180]
            r["rot_err"] = float(np.mean(np.minimum(dyaw, 360.0 - dyaw)))
            r["rot_err_sym"] = float(np.mean([_fold(float(dy), SYM.get(gt_types[i], 1))
                                              for dy, i in zip(dyaw, gi)]))
    # camera
    gc = os.path.join(d, "camera.json"); pc = os.path.join(d, f"pred_camera_{suffix}.json")
    if os.path.exists(gc) and os.path.exists(pc):
        r["cam_loc_err"], r["cam_view_err"] = cam_error(json.load(open(gc)), json.load(open(pc)))
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dataset", default="test_data")
    ap.add_argument("--suffix", default="image_2_usd")
    ap.add_argument("--percent_eval", type=float, default=1.0)
    ap.add_argument("--match_threshold", type=float, default=100.0)   # cm; TP if matched within this (dist mode)
    ap.add_argument("--match", choices=["dist", "iou"], default="dist", help="matching criterion: center distance (<= match_threshold cm) or 3D axis-aligned box IoU (>= iou_thr)")
    ap.add_argument("--iou_thr", type=float, default=0.25, help="IoU threshold for a TP when --match iou (3D AABB; 0.5 is strict in 3D, 0.25 a reasonable detection default)")
    a = ap.parse_args()

    # object-level accumulators (micro-avg over matched pairs) + sample-level (macro)
    def newacc(): return {"n_samp": 0, "tp": 0, "count_err": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0,
                          "recog": 0.0, "trans": 0.0, "scale": 0.0, "rot": 0.0, "rot_sym": 0.0, "cam_loc": 0.0, "cam_view": 0.0, "n_cam": 0}
    overall = newacc()
    crit = f"IoU>={a.iou_thr}" if a.match == "iou" else f"dist<={a.match_threshold}cm"
    print(f"per-object metrics  suffix={a.suffix}  match={a.match} ({crit})\n")
    print(f"{'#obj':>5} {'n':>4} {'cnt_err':>7} {'recall':>6} {'prec':>5} {'F1':>5} {'recog':>6} {'trans(cm)':>9} {'scale':>6} {'rot(deg)':>8} {'rotSYM':>7}")
    for obj in tqdm(natsorted(os.listdir(a.eval_dataset))):
        bd = os.path.join(a.eval_dataset, obj)
        if not os.path.isdir(bd): continue
        samples = natsorted(os.listdir(bd))
        samples = samples[:max(1, math.floor(len(samples) * a.percent_eval))]
        acc = newacc()
        for s in tqdm(samples):
            d = os.path.join(bd, s)
            try:
                r = eval_sample(d, a.suffix, a.match_threshold, a.match, a.iou_thr)
            except Exception as e:
                tqdm.write(f"{d}: {e}"); r = None
            if r is None: continue
            for A in (acc, overall):
                A["n_samp"] += 1; A["count_err"] += r["count_err"]
                A["recall"] += r["recall"]; A["precision"] += r["precision"]; A["f1"] += r["f1"]; A["tp"] += r["tp"]
                if r["recognition"] is not None:
                    A["recog"] += r["recognition"] * r["tp"]; A["trans"] += r["trans_err"] * r["tp"]
                    A["scale"] += r["scale_err"] * r["tp"]; A["rot"] += r["rot_err"] * r["tp"]; A["rot_sym"] += r["rot_err_sym"] * r["tp"]
                if "cam_loc_err" in r:
                    A["cam_loc"] += r["cam_loc_err"]; A["cam_view"] += r["cam_view_err"]; A["n_cam"] += 1
        n = max(1, acc["n_samp"]); tp = max(1, acc["tp"])
        print(f"{obj:>5} {acc['n_samp']:>4} {acc['count_err']/n:>7.2f} {acc['recall']/n:>6.2f} {acc['precision']/n:>5.2f} "
              f"{acc['f1']/n:>5.2f} {acc['recog']/tp:>6.2f} {acc['trans']/tp:>9.1f} {acc['scale']/tp:>6.3f} {acc['rot']/tp:>8.1f} {acc['rot_sym']/tp:>7.1f}")

    n = max(1, overall["n_samp"]); tp = max(1, overall["tp"]); nc = max(1, overall["n_cam"])
    print("\n" + "=" * 60)
    print(f"OVERALL ({a.suffix})  samples={overall['n_samp']}  matched objects={overall['tp']}")
    print(f"  count error (|n_pred-n_gt|): {overall['count_err']/n:.2f}")
    print(f"  recall / precision / F1:     {overall['recall']/n:.3f} / {overall['precision']/n:.3f} / {overall['f1']/n:.3f}")
    print(f"  asset-recognition (matched): {overall['recog']/tp:.3f}")
    print(f"  translation error (cm):      {overall['trans']/tp:.1f}")
    print(f"  scale error (relative):      {overall['scale']/tp:.3f}")
    print(f"  rotation error (deg, NO-sym):{overall['rot']/tp:.1f}")
    print(f"  rotation error (deg, SYM):   {overall['rot_sym']/tp:.1f}")
    print(f"  camera loc error (cm):       {overall['cam_loc']/nc:.1f}")
    print(f"  camera view error (deg):     {overall['cam_view']/nc:.1f}")

if __name__ == "__main__":
    main()
