# -*- coding: utf-8 -*-
"""无偏算子对比：DSM 叠加 vs 封边(cap) vs 微焊接(snap)，在随机样本上。

为什么要重做：
    之前 dsm_ids 是为了研究 DSM 特意挑的"大场景/高真值面数"样本，
    base CD 5.92 / ECD 10.93，比线上(3.40/5.91)差得多 -> 增益被高估。
    本脚本用 42 个**随机**样本重新估计"线上期望增益"。

新线索（本轮 ECD 分类分解）：
    我们锐边的构成：边界边 44%(长度占比) 折边 56%，非流形 0；
    E1 = 0.442*6.36(bnd) + 0.558*3.88(ang)  -> **边界边贡献了 50% 的 E1**，
    且 bnd 的 E1_self 高达 6.36 m（p90 14.8 m）=> 我们的开放边界离真值锐边很远。
    我们 40~49% 的边是边界边（mesh 不水密）——T 型接点/未缝合。
    => 假说：**把边界补上(cap)可让 E1 直接掉一半**，且 cap 产生的新折边
       落在墙脚/真实轮廓处，可能同时改善 E2。

算子：
    raw   原始 wC
    g3/g5/g8   DSM 格网叠加（g 米一格）
    cap   边界环用"环心扇形"封口（保留 3D 坐标）
    capf  边界环先投影到最佳拟合平面再扇形封口（封口尽量平，少造杂折边）
    s3/s1 顶点按 3mm / 1cm 量化后合并（消 T 型接点，几乎不动几何）
    g3cap = g3 再 cap
    sub2  二次细分（对照：证明"加面"本身无用）

用法:
    python src/_t_op_bench2.py --rec data/work/bw600/wC --gt data/work/bw600_gt \
        --ply data/work/bw600/ply --ids logs/op_ids40.txt --jobs 3 --out logs/op_bench40.json
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
OPS = ["raw", "g3", "g5", "g8", "cap", "capf", "s3", "s1", "g3cap", "sub2"]


def L(p):
    m = trimesh.load(p)
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
    m = trimesh.load(p)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    v = np.asarray(m.vertices, float)
    if len(v) > cap:
        v = v[np.random.default_rng(seed).choice(len(v), cap, replace=False)]
    return v


def union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return trimesh.util.concatenate([a, b])


# ---------------- DSM ----------------
def dsm_grid(pts, g, max_edge=None):
    if len(pts) < 8:
        return None
    key = np.floor(pts[:, :2] / g).astype(np.int64)
    order = np.lexsort((pts[:, 2], key[:, 1], key[:, 0]))
    k = key[order]
    new = np.any(k[1:] != k[:-1], axis=1)
    first = np.concatenate([[True], new])
    p = pts[order[first]]
    if len(p) < 8:
        return None
    try:
        d = Delaunay(p[:, :2])
    except Exception:
        return None
    t = d.simplices
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    e = np.maximum(np.maximum(np.linalg.norm(a - b, axis=1),
                              np.linalg.norm(b - c, axis=1)),
                   np.linalg.norm(c - a, axis=1))
    lim = max_edge if max_edge else 3.0 * g
    keep = (e <= lim) & (e > 1e-6)
    if keep.sum() < 4:
        return None
    return trimesh.Trimesh(vertices=p, faces=t[keep], process=False)


# ---------------- 封边 ----------------
def boundary_loops(m):
    F = np.asarray(m.faces, int)
    em = {}
    for fi, f in enumerate(F):
        for e in (tuple(sorted((f[0], f[1]))), tuple(sorted((f[1], f[2]))),
                  tuple(sorted((f[2], f[0])))):
            em.setdefault(e, []).append(fi)
    be = [e for e, fl in em.items() if len(fl) == 1]
    if not be:
        return []
    nbr = {}
    for a, b in be:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    used = set()
    loops = []
    for a, b in be:
        if (a, b) in used:
            continue
        loop = [a]
        used.add((a, b))
        cur, prev = b, a
        ok = True
        while cur != a:
            if cur in loop:
                ok = False
                break
            loop.append(cur)
            cand = [x for x in nbr.get(cur, []) if x != prev]
            if not cand:
                ok = False
                break
            nxt = cand[0]
            if (cur, nxt) in used and nxt != a:
                ok = False
                break
            used.add((cur, nxt))
            used.add((nxt, cur))
            prev, cur = cur, nxt
            if len(loop) > 4000:
                ok = False
                break
        if ok and len(loop) >= 3:
            loops.append(loop)
    return loops


def cap_loops(m, planar=False):
    """只返回**封口面**（自带顶点），不含原网格，供 union 使用。"""
    V = np.asarray(m.vertices, float)
    loops = boundary_loops(m)
    allv = []
    allf = []
    for lp in loops:
        pts = V[np.asarray(lp)]
        if len(lp) < 3 or len(lp) > 400:
            continue
        if planar:
            mu = pts.mean(0)
            ctr = pts - mu
            _, _, vt = np.linalg.svd(ctr, full_matrices=False)
            flat = ctr @ vt.T
            flat[:, 2] = 0.0
            pts = flat @ vt + mu
        b = len(allv)
        n = len(pts)
        allv.extend(pts)
        allv.append(pts.mean(0))
        ci = b + n
        for i in range(n):
            allf.append([ci, b + i, b + (i + 1) % n])
    if not allf:
        return None
    return trimesh.Trimesh(vertices=np.asarray(allv, float),
                           faces=np.asarray(allf, int), process=False)


def snap(m, tol):
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces, int)
    key = np.round(V / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n = int(inv.max()) + 1
    newV = np.zeros((n, 3))
    cnt = np.zeros(n)
    np.add.at(newV, inv, V)
    np.add.at(cnt, inv, 1.0)
    newV /= cnt[:, None]
    newF = inv[F]
    ok = (newF[:, 0] != newF[:, 1]) & (newF[:, 1] != newF[:, 2]) & (newF[:, 0] != newF[:, 2])
    if ok.sum() < 4:
        return None
    return trimesh.Trimesh(vertices=newV, faces=newF[ok], process=False)


def sub(m, k):
    for _ in range(k):
        V = np.asarray(m.vertices, float)
        F = np.asarray(m.faces, int)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        nv = np.vstack([V, ab, bc, ca])
        n = len(V)
        f1 = np.stack([F[:, 0], np.arange(n, n + len(F)), np.arange(2 * n, 2 * n + len(F))], 1)
        f2 = np.stack([np.arange(n, n + len(F)), F[:, 1], np.arange(3 * n, 3 * n + len(F))], 1)
        f3 = np.stack([np.arange(2 * n, 2 * n + len(F)), np.arange(3 * n, 3 * n + len(F)), F[:, 2]], 1)
        f4 = np.stack([np.arange(n, n + len(F)), np.arange(3 * n, 3 * n + len(F)),
                       np.arange(2 * n, 2 * n + len(F))], 1)
        m = trimesh.Trimesh(vertices=nv, faces=np.vstack([f1, f2, f3, f4]), process=False)
    return m


def build_ops(base, pts):
    o = {"raw": base}
    o["g3"] = union(base, dsm_grid(pts, 3.0))
    o["g5"] = union(base, dsm_grid(pts, 5.0))
    o["g8"] = union(base, dsm_grid(pts, 8.0))
    o["cap"] = union(base, cap_loops(base, planar=False))
    o["capf"] = union(base, cap_loops(base, planar=True))
    o["s3"] = snap(base, 0.003)
    o["s1"] = snap(base, 0.01)
    o["g3cap"] = union(o["g3"], cap_loops(base, planar=True))
    o["sub2"] = sub(base, 2)
    return o


# ---------------- 评分（与官方口径一致，raw vs raw） ----------------
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
        return {"CD1": d1, "CD2": d2, "CD": d1 + d2, "ECD": e1 + e2,
                "E1": e1, "E2": e2, "FINAL": 0.6 * (d1 + d2) + 0.4 * (e1 + e2),
                "F": int(len(np.asarray(m.faces))), "nE": int(len(ep))}
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
        gtpre = prep_gt(gt, ns, ne, seed)
        base = L(rp)
        pts = load_ply_pts(pp)
        ops = build_ops(base, pts)
        res = {}
        for k in OPS:
            res[k] = score(ops.get(k), gt, gtpre, ns, ne, seed)
        return {"id": sid, "gtF": int(len(np.asarray(gt.faces))),
                "gtV": int(len(np.asarray(gt.vertices))),
                "npt": int(len(pts)),
                "F0": int(len(np.asarray(base.faces))),
                "V0": int(len(np.asarray(base.vertices))),
                "cfg": res}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:100]}


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
    a = ap.parse_args()
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()]
    print("无偏算子对比 n=%d" % len(ids), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.rec, a.gt, a.ply, a.ns, a.ne, a.seed)
                                           for i in ids], chunksize=1)):
            rows.append(r)
            print("  %d/%d %s" % (k + 1, len(ids), r.get("err") or "ok"), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)

    ok = [r for r in rows if "err" not in r and r["cfg"].get("raw")]
    clean = [r for r in ok if r["cfg"]["raw"]["CD1"] < 50]
    print("\n可用 %d / %d （剔帧污染后 %d）" % (len(ok), len(rows), len(clean)))
    for r in rows:
        if "err" in r:
            print("  err %s: %s" % (r["id"], r["err"]))
    if not clean:
        return 1
    b = np.array([r["cfg"]["raw"]["FINAL"] for r in clean])
    print("base: CD %.4f  ECD %.4f (E1 %.4f+E2 %.4f)  FINAL %.4f  F %.0f  gtF %.0f  npt %.0f" %
          (np.mean([r["cfg"]["raw"]["CD"] for r in clean]),
           np.mean([r["cfg"]["raw"]["ECD"] for r in clean]),
           np.mean([r["cfg"]["raw"]["E1"] for r in clean]),
           np.mean([r["cfg"]["raw"]["E2"] for r in clean]), b.mean(),
           np.mean([r["F0"] for r in clean]), np.mean([r["gtF"] for r in clean]),
           np.mean([r["npt"] for r in clean])))
    print("\n%-7s %4s %9s %9s %9s %9s %8s %8s %8s %9s" %
          ("op", "n", "dFINAL", "dFNLmed", "dCD", "dECD", "dE1", "dE2",
           "meanF", "rel%"))
    for op in OPS:
        if op == "raw":
            continue
        v = [(r["cfg"][op]["FINAL"] - r["cfg"]["raw"]["FINAL"],
              r["cfg"][op]["CD"] - r["cfg"]["raw"]["CD"],
              r["cfg"][op]["ECD"] - r["cfg"]["raw"]["ECD"],
              r["cfg"][op]["E1"] - r["cfg"]["raw"]["E1"],
              r["cfg"][op]["E2"] - r["cfg"]["raw"]["E2"],
              r["cfg"][op]["F"]) for r in clean if r["cfg"].get(op)]
        if len(v) < 3:
            print("%-7s %4d  (样本不足)" % (op, len(v)))
            continue
        A = np.array(v)
        print("%-7s %4d %+9.4f %+9.4f %+9.4f %+9.4f %+8.4f %+8.4f %8.0f %+8.2f%%   %d/%d" %
              (op, len(v), A[:, 0].mean(), np.median(A[:, 0]), A[:, 1].mean(),
               A[:, 2].mean(), A[:, 3].mean(), A[:, 4].mean(), A[:, 5].mean(),
               100 * A[:, 0].mean() / b.mean(),
               int((A[:, 0] < 0).sum()), int((A[:, 0] > 0).sum())))
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
