# -*- coding: utf-8 -*-
"""DSM 变体终筛：在"最吃 ECD"的大场景上比较密度/连通/多尺度。

线上标定已完成（v21 实测）：
    预测 FINAL = 0.6*(3.4018 + 0.161*dCD_local) + 0.4*(5.9081 + 0.42*dECD_local)
    对 v21(g3) 预测 4.2271 vs 实测 4.22585 -> 模型误差 0.001，可用。
所以现在只需在大场景上找 dECD_local 更大的变体。

变体：
    g3    基准：3 m 格网、格内取 z 最小、丢弃 >9 m 的边（当前 v21）
    g2    2 m 格网（更密）
    g1.5  1.5 m 格网
    g3op  同 g3 但不丢长边（连通更多区域）
    g3x2  双尺度 g3 ∪ g15（细部 + 大跨度）
    g3mm  min ∪ max（地面 + 顶面）

用法:
    python src/_t_dsm_var.py --rec data/work/bw600/wC --gt data/work/bw600_gt \
        --ply data/work/bw600/ply --ids logs/big_ids40.txt --jobs 3 \
        --ns 15000 --ne 30000 --out logs/dsm_var.json
"""
import argparse
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.spatial import cKDTree, Delaunay

if 'faiss' not in sys.modules:
    _s = types.ModuleType('faiss')
    _s.IndexFlatL2 = object
    _s.StandardGpuResources = object
    _s.index_cpu_to_gpu = lambda *a, **k: None
    sys.modules['faiss'] = _s
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"D:/LiDAR2026"
OPS = ["g3", "g2", "g15", "g3op", "g3x2", "g3mm", "g3opmm", "g2mm"]


def L(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def resolve(rec, sid):
    for suf in ("", "_ReconstructedModel"):
        p = os.path.join(rec, sid + suf + ".obj")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def load_ply_pts(p, cap=400000, seed=0):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    v = np.asarray(m.vertices, float)
    if len(v) > cap:
        v = v[np.random.default_rng(seed).choice(len(v), cap, replace=False)]
    return v


def dsm(pts, g, zmode="min", max_edge=None, cell_cap=100000, seed=0):
    if len(pts) < 8:
        return None
    key = np.floor(pts[:, :2] / g).astype(np.int64)
    rev = (zmode == "max")
    order = np.lexsort((-pts[:, 2] if rev else pts[:, 2], key[:, 1], key[:, 0]))
    k = key[order]
    new = np.any(k[1:] != k[:-1], axis=1)
    p = pts[order[np.concatenate([[True], new])]]
    if len(p) < 8:
        return None
    if len(p) > cell_cap:
        p = p[np.random.default_rng(seed + 7).choice(len(p), cell_cap, replace=False)]
    try:
        d = Delaunay(p[:, :2])
    except Exception:
        return None
    t = d.simplices
    if max_edge is not None:
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        e = np.maximum(np.maximum(np.linalg.norm(a - b, axis=1),
                                  np.linalg.norm(b - c, axis=1)),
                       np.linalg.norm(c - a, axis=1))
        t = t[(e <= max_edge) & (e > 1e-6)]
    if len(t) < 4:
        return None
    return trimesh.Trimesh(vertices=p, faces=t, process=False)


def union(*ms):
    ms = [m for m in ms if m is not None and len(np.asarray(m.faces)) > 0]
    if not ms:
        return None
    if len(ms) == 1:
        return ms[0]
    return trimesh.util.concatenate(ms)


def prep_gt(gt, ns, ne, seed):
    import official_evaluate as OE
    np.random.seed(seed)
    gs = np.asarray(trimesh.sample.sample_surface(gt, ns)[0], float)
    geg = OE.extract_sharp_edges(gt, 30.0)
    gep = (np.zeros((0, 3)) if len(geg) == 0 else
           np.asarray(OE.sample_points_on_edges_global(
               np.asarray(gt.vertices, float), geg, ne), float))
    return gs, gep, (cKDTree(gep) if len(gep) else None)


def score(m, gt, gtpre, ns, ne, seed):
    import official_evaluate as OE
    gs, gep, tgep = gtpre
    if m is None or len(np.asarray(m.faces)) == 0:
        return None
    try:
        np.random.seed(seed)
        ps = np.asarray(trimesh.sample.sample_surface(m, ns)[0], float)
        cp = np.asarray(m.nearest.on_surface(gs)[0], float)
        cg = np.asarray(gt.nearest.on_surface(ps)[0], float)
        d1 = float(np.linalg.norm(ps - cg, axis=1).mean())
        d2 = float(np.linalg.norm(gs - cp, axis=1).mean())
        ep = OE.extract_sharp_edges(m, 30.0)
        if len(ep) == 0 or tgep is None:
            return None
        np.random.seed(seed)
        pep = np.asarray(OE.sample_points_on_edges_global(
            np.asarray(m.vertices, float), ep, ne), float)
        e1 = float(tgep.query(pep)[0].mean())
        e2 = float(cKDTree(pep).query(gep)[0].mean())
        return {"CD": d1 + d2, "CD1": d1, "ECD": e1 + e2, "E1": e1, "E2": e2,
                "FINAL": 0.6 * (d1 + d2) + 0.4 * (e1 + e2),
                "F": int(len(np.asarray(m.faces)))}
    except Exception:
        return None


def one(job):
    sid, rec, gtd, plyd, ns, ne, seed = job
    try:
        gtp = os.path.join(gtd, sid + "_gt.obj")
        rp = resolve(rec, sid)
        pp = os.path.join(plyd, sid + ".ply")
        if not os.path.exists(gtp) or rp is None or not os.path.exists(pp):
            return {"id": sid, "err": "missing"}
        gt = L(gtp)
        pre = prep_gt(gt, ns, ne, seed)
        base = L(rp)
        pts = load_ply_pts(pp)
        sd = abs(hash(sid)) % (2 ** 31)
        res = {"raw": score(base, gt, pre, ns, ne, seed)}
        res["g3"] = score(union(base, dsm(pts, 3.0, "min", 9.0, seed=sd)), gt, pre, ns, ne, seed)
        res["g2"] = score(union(base, dsm(pts, 2.0, "min", 6.0, seed=sd)), gt, pre, ns, ne, seed)
        res["g15"] = score(union(base, dsm(pts, 1.5, "min", 4.5, seed=sd)), gt, pre, ns, ne, seed)
        res["g3op"] = score(union(base, dsm(pts, 3.0, "min", None, seed=sd)), gt, pre, ns, ne, seed)
        res["g3x2"] = score(union(base, dsm(pts, 3.0, "min", 9.0, seed=sd),
                                  dsm(pts, 15.0, "min", 45.0, seed=sd + 1)), gt, pre, ns, ne, seed)
        res["g3mm"] = score(union(base, dsm(pts, 3.0, "min", 9.0, seed=sd),
                                  dsm(pts, 3.0, "max", 9.0, seed=sd + 1)), gt, pre, ns, ne, seed)
        res["g2mm"] = score(union(base, dsm(pts, 2.0, "min", 6.0, seed=sd),
                                  dsm(pts, 2.0, "max", 6.0, seed=sd + 1)), gt, pre, ns, ne, seed)
        res["g3opmm"] = score(union(base, dsm(pts, 3.0, "min", None, seed=sd),
                                     dsm(pts, 3.0, "max", None, seed=sd + 1)),
                               gt, pre, ns, ne, seed)
        return {"id": sid, "gtF": int(len(np.asarray(gt.faces))), "npt": int(len(pts)),
                "diag": float(np.linalg.norm(pts.max(0) - pts.min(0))),
                "F0": int(len(np.asarray(base.faces))), "cfg": res}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--ns", type=int, default=15000)
    ap.add_argument("--ne", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", default="")
    a = ap.parse_args()
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()]
    print("DSM 变体终筛 n=%d" % len(ids), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.rec, a.gt, a.ply, a.ns, a.ne, a.seed)
                                           for i in ids], chunksize=1)):
            rows.append(r)
            print("  %d/%d %s" % (k + 1, len(ids), r.get("err") or "ok"), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)
    clean = [r for r in rows if "err" not in r and r["cfg"].get("raw")
             and r["cfg"]["raw"]["CD1"] < 50]
    print("\n可用 %d / %d" % (len(clean), len(rows)))
    if not clean:
        return 1
    print("base: CD %.4f  ECD %.4f  FINAL %.4f  F %.0f  gtF %.0f" %
          (np.mean([r["cfg"]["raw"]["CD"] for r in clean]),
           np.mean([r["cfg"]["raw"]["ECD"] for r in clean]),
           np.mean([r["cfg"]["raw"]["FINAL"] for r in clean]),
           np.mean([r["F0"] for r in clean]), np.mean([r["gtF"] for r in clean])))
    print("\n%-7s %9s %9s %9s %9s %9s %8s %8s" %
          ("op", "dFINAL", "dFNLmed", "dCD", "dECD", "dE2", "meanF", "好/坏"))
    use = [x for x in (a.ops.split(",") if a.ops else OPS) if x]
    for op in use:
        v = [(r["cfg"][op]["FINAL"] - r["cfg"]["raw"]["FINAL"],
              r["cfg"][op]["CD"] - r["cfg"]["raw"]["CD"],
              r["cfg"][op]["ECD"] - r["cfg"]["raw"]["ECD"],
              r["cfg"][op]["E2"] - r["cfg"]["raw"]["E2"],
              r["cfg"][op]["F"]) for r in clean if r["cfg"].get(op)]
        if len(v) < 3:
            print("%-7s (样本不足 %d)" % (op, len(v)))
            continue
        A = np.array(v)
        print("%-7s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f %8.0f  %d/%d" %
              (op, A[:, 0].mean(), np.median(A[:, 0]), A[:, 1].mean(),
               A[:, 2].mean(), A[:, 3].mean(), A[:, 4].mean(),
               int((A[:, 0] < 0).sum()), int((A[:, 0] > 0).sum())))
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
