# -*- coding: utf-8 -*-
"""门控内快速验证：只算 raw / g3 / g3mm 三个配置，用于给 g3mm 一个可信的本地读数。

与 _t_dsm_var.py 的差别：
  * 只构建 3 次 Delaunay（raw 不建），而不是 12 次 -> 约 6 倍加速；
  * 增量统计只报"门控内（ply_diag>=gate）"的样本，这正是线上受影响的那部分；
  * 直接套用线上标定式给出 FINAL 预测（锚点 = v20 裸基座）。

标定式（锚点 v20 = fd202a97）：
    线上CD  = 3.40181 + 0.161 * dCD_离线
    线上ECD = 5.90812 + t     * dECD_离线
t = 0.424（乐观，v21 实测族） / 0.300（保守）

用法:
    python src/_t_mm_fast.py --rec data/work/bw600/wC --gt data/work/bw600_gt \
        --ply data/work/bw600/ply --ids logs/op_ids_gate.txt --jobs 4 --out logs/mm_fast.json
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
V20 = dict(CD=3.40181, ECD=5.90812)
V21 = dict(CD=3.32290, ECD=5.58026, FINAL=4.22585)


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
    return ms[0] if len(ms) == 1 else trimesh.util.concatenate(ms)


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


def build(pts, op, sd):
    """按算子名构造与 base 取并集后的网格。所有算子都保留 max_edge=3g 的硬约束。"""
    if op == "g3":
        return dsm(pts, 3.0, "min", 9.0, seed=sd)
    if op == "g3mm":
        return union(dsm(pts, 3.0, "min", 9.0, seed=sd),
                     dsm(pts, 3.0, "max", 9.0, seed=sd + 1))
    if op == "g2":
        return dsm(pts, 2.0, "min", 6.0, seed=sd)
    if op == "g2mm":
        return union(dsm(pts, 2.0, "min", 6.0, seed=sd),
                     dsm(pts, 2.0, "max", 6.0, seed=sd + 1))
    if op == "g15":
        return dsm(pts, 1.5, "min", 4.5, seed=sd)
    if op == "g15mm":
        return union(dsm(pts, 1.5, "min", 4.5, seed=sd),
                     dsm(pts, 1.5, "max", 4.5, seed=sd + 1))
    if op == "g3x2":          # 多尺度：3 m 细层 ∪ 15 m 粗层
        return union(dsm(pts, 3.0, "min", 9.0, seed=sd),
                     dsm(pts, 15.0, "min", 45.0, seed=sd + 1))
    if op == "g3mmx2":        # min ∪ max ∪ 15 m 粗层
        return union(dsm(pts, 3.0, "min", 9.0, seed=sd),
                     dsm(pts, 3.0, "max", 9.0, seed=sd + 1),
                     dsm(pts, 15.0, "min", 45.0, seed=sd + 2))
    if op == "g2mmx2":
        return union(dsm(pts, 2.0, "min", 6.0, seed=sd),
                     dsm(pts, 2.0, "max", 6.0, seed=sd + 1),
                     dsm(pts, 10.0, "min", 30.0, seed=sd + 2))
    raise KeyError(op)


def one(job):
    sid, rec, gtd, plyd, ns, ne, seed, ops = job
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
        cfg = {"raw": score(base, gt, pre, ns, ne, seed)}
        for op in ops:
            cfg[op] = score(union(base, build(pts, op, sd)), gt, pre, ns, ne, seed)
        return {"id": sid, "gtF": int(len(np.asarray(gt.faces))),
                "npt": int(len(pts)),
                "diag": float(np.linalg.norm(pts.max(0) - pts.min(0))),
                "F0": int(len(np.asarray(base.faces))),
                "cfg": cfg}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ns", type=int, default=15000)
    ap.add_argument("--ne", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--gate", type=float, default=30.0)
    ap.add_argument("--ops", default="g3,g3mm",
                    help="逗号分隔；可选 g3 g3mm g2 g2mm g15 g15mm g3x2 g3mmx2 g2mmx2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ops = [x for x in a.ops.split(",") if x]
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()]
    print("门控内快速验证 n=%d  ops=%s" % (len(ids), ",".join(ops)), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.rec, a.gt, a.ply, a.ns, a.ne, a.seed, ops)
                                           for i in ids], chunksize=1)):
            rows.append(r)
            if (k + 1) % 25 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)
    clean = [r for r in rows if "err" not in r and r["cfg"].get("raw")
             and r["cfg"]["raw"]["CD1"] < 50]
    g = [r for r in clean if r["diag"] >= a.gate]
    print("\n可用 %d / %d   门控内 %d" % (len(clean), len(rows), len(g)))
    if not g:
        return 1
    b = {k: np.mean([r["cfg"]["raw"][k] for r in g]) for k in ("CD", "ECD", "FINAL")}
    print("门控内 base: CD %.4f  ECD %.4f  FINAL %.4f  F %.0f  gtF %.0f\n" %
          (b["CD"], b["ECD"], b["FINAL"], np.mean([r["F0"] for r in g]),
           np.mean([r["gtF"] for r in g])))
    print("%-7s %6s %9s %9s %9s | %9s %9s %9s %8s" %
          ("op", "n", "dCD_L", "dECD_L", "dFNL_L", "pred_opt", "pred_cons",
           "dFNLmed", "好/坏"))
    for op in ops:
        v = [(r["cfg"][op]["CD"] - r["cfg"]["raw"]["CD"],
              r["cfg"][op]["ECD"] - r["cfg"]["raw"]["ECD"],
              r["cfg"][op]["FINAL"] - r["cfg"]["raw"]["FINAL"],
              r["cfg"][op]["F"]) for r in g if r["cfg"].get(op)]
        if len(v) < 3:
            print("%-7s (样本不足 %d)" % (op, len(v)))
            continue
        A = np.array(v)
        po = 0.6 * (V20["CD"] + 0.161 * A[:, 0].mean()) + \
             0.4 * (V20["ECD"] + 0.424 * A[:, 1].mean())
        pc = 0.6 * (V20["CD"] + 0.161 * A[:, 0].mean()) + \
             0.4 * (V20["ECD"] + 0.300 * A[:, 1].mean())
        print("%-7s %6d %+9.3f %+9.3f %+9.3f | %9.4f %9.4f %+8.3f %d/%d" %
              (op, len(v), A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(),
               po, pc, np.median(A[:, 2]),
               int((A[:, 2] < 0).sum()), int((A[:, 2] > 0).sum())))
    print("\n锚点: v21 实测 FINAL %.5f | plqueffe 4.1955 | Chuniby0 4.3059"
          % V21["FINAL"])
    print("写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
