# -*- coding: utf-8 -*-
"""v10：walls 立面补全算子（唯一在真值上净增益的算子）。

已验证配置（realbench 32 栋真值, logs/walls_tune.json）：
    只加足迹棱柱的**侧壁**，不加顶、不加底
    z0 = 足迹内点云 z 的 p2 分位，z1 = p30 分位
    相对 ΔFINAL -3.65%（-0.0560），改善 25 / 变差 6，ΔCD -0.0944，ΔECD +0.0016
机制：CD1 与 CD2 **同时**改善（0.744→0.703 / 0.491→0.463），说明补的墙几乎不引入误差
    却覆盖了原来缺失的立面——这是"只补墙不补顶"独有的性质，含平顶的完整棱柱一律净负。

场景保护（本脚本新增）：足迹 XY 对角线 > 150 m 视为场景外轮廓，跳过（126 栋, 3.1%），
    避免给场景 tile 造出巨墙；realbench 的足迹全部远小于该阈值，故已验证增益不受影响。
另外：足迹顶点按位置焊接后再取边界边，避免多边形汤造成"逐三角形造墙"。

用法：
    python _t_build_walls.py --mode bench        # 在 realbench 上复验
    python _t_build_walls.py --mode build        # 生成 4000 栋到 data/work/final_v10
"""
import argparse
import json
import os
import shutil
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.spatial import cKDTree

if 'faiss' not in sys.modules:
    _s = types.ModuleType('faiss')
    _s.IndexFlatL2 = object
    _s.StandardGpuResources = object
    _s.index_cpu_to_gpu = lambda *a, **k: None
    sys.modules['faiss'] = _s
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"D:\LiDAR2026"
FP_DIR = os.path.join(ROOT, "data", "work", "recon_v5")
PLY_DIR = os.path.join(ROOT, "data", "work", "ply")
V9R = os.path.join(ROOT, "data", "work", "final_v9r")

# 已验证配置，不要随意改动
Z0Q, Z1Q = 2.0, 30.0
H_CAP = 150.0
FP_DIAG_MAX = 150.0
RB = os.path.join(ROOT, "data", "realbench")
GT_DIR = os.path.join(RB, "gt")
RB_REC = os.path.join(RB, "_rsweep", "base", "recon")
RB_PLY = os.path.join(RB, "_rsweep", "base", "ply")
RB_SUF = "_ReconstructedModel.obj"
RB_FPSUF = "_GeneratedFootprint.obj"
N, NE = 15000, 30000


def load(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def pcv(pc):
    return np.asarray(pc.vertices if hasattr(pc, "vertices") else pc.points, float)


def weld_footprint(V, F, tol=1e-4):
    """按位置焊接顶点，避免多边形汤导致逐三角形造墙。"""
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    V2 = np.zeros((len(first), 3), float)
    V2[inv] = V
    F2 = np.asarray(F, int).reshape(-1, 3)
    F2 = inv[F2]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V2, F2[keep]


def build_walls(V, F, z0, z1):
    """只保留侧壁：把足迹的边界边各拉一组四边形。"""
    if len(F) == 0 or z1 <= z0:
        return None
    W = V.copy()
    W[:, 2] = z0
    nv = len(W)
    W2 = np.vstack([W, W + np.array([0, 0, z1 - z0])])
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    if len(bnd) == 0:
        return None
    a, b = bnd[:, 0], bnd[:, 1]
    wf = np.vstack([np.column_stack([a, b, b + nv]),
                    np.column_stack([a, b + nv, a + nv])])
    return trimesh.Trimesh(vertices=W2, faces=wf, process=False)


def make_walls(fp_path, ply_path, z0q=None, z1q=None):
    """返回 (walls_mesh, info) 或 (None, info)。"""
    z0q = Z0Q if z0q is None else z0q
    z1q = Z1Q if z1q is None else z1q
    if not os.path.exists(fp_path) or not os.path.exists(ply_path):
        return None, {"skip": "missing"}
    fpm = load(fp_path)
    FV = np.asarray(fpm.vertices, float)
    FF = np.asarray(fpm.faces, int).reshape(-1, 3)
    if len(FF) == 0:
        return None, {"skip": "empty-fp"}
    fdiag = float(np.linalg.norm(FV.max(0)[:2] - FV.min(0)[:2]))
    if fdiag > FP_DIAG_MAX:
        return None, {"skip": "scene-fp", "fp_diag": fdiag}
    pv = pcv(load(ply_path))
    if len(pv) == 0:
        return None, {"skip": "empty-ply"}
    lo, hi = FV.min(0)[:2], FV.max(0)[:2]
    inb = ((pv[:, 0] >= lo[0]) & (pv[:, 0] <= hi[0])
           & (pv[:, 1] >= lo[1]) & (pv[:, 1] <= hi[1]))
    zin = pv[inb][:, 2] if inb.sum() > 50 else pv[:, 2]
    z0 = float(np.percentile(zin, z0q))
    z1 = min(float(np.percentile(zin, z1q)), z0 + H_CAP)
    if z1 - z0 < 0.5:
        return None, {"skip": "flat", "z0": z0, "z1": z1}
    Vw, Fw = weld_footprint(FV, FF)
    w = build_walls(Vw, Fw, z0, z1)
    if w is None or len(w.faces) == 0:
        return None, {"skip": "no-walls"}
    return w, {"fp_diag": fdiag, "z0": z0, "z1": z1, "h": z1 - z0,
               "n_wall_f": int(len(w.faces)), "n_bnd": int(len(w.faces)) // 2}


def union(a, b):
    if b is None or len(np.asarray(b.faces)) == 0:
        return a
    va, fa = np.asarray(a.vertices, float), np.asarray(a.faces, int).reshape(-1, 3)
    vb, fb = np.asarray(b.vertices, float), np.asarray(b.faces, int).reshape(-1, 3)
    return trimesh.Trimesh(vertices=np.vstack([va, vb]),
                           faces=np.vstack([fa, fb + len(va)]), process=False)


# ------------------------------------------------------------------ bench
def bench_one(job):
    i, z0q, z1q = job
    import official_evaluate as OE
    try:
        gt = load(os.path.join(GT_DIR, i + "_gt.obj"))
        vg = np.asarray(gt.vertices, float)
        gt = trimesh.Trimesh(vertices=vg - 0.5 * (vg.min(0) + vg.max(0)),
                             faces=np.asarray(gt.faces), process=False)
        base = load(os.path.join(RB_REC, i + RB_SUF))
        gs = trimesh.sample.sample_surface(gt, N)[0]
        geg = OE.extract_sharp_edges(gt, 30.0)
        gep = (np.zeros((0, 3)) if len(geg) == 0 else
               OE.sample_points_on_edges_global(np.asarray(gt.vertices, float), geg, NE))

        def score(m):
            if m is None or len(np.asarray(m.faces)) == 0:
                return None
            ps = trimesh.sample.sample_surface(m, N)[0]
            cp = np.asarray(m.nearest.on_surface(gs)[0], float)
            cg = np.asarray(gt.nearest.on_surface(ps)[0], float)
            d1 = float(np.linalg.norm(ps - cg, axis=1).mean())
            d2 = float(np.linalg.norm(gs - cp, axis=1).mean())
            e = float("nan")
            ep = OE.extract_sharp_edges(m, 30.0)
            if len(ep) and len(gep):
                pep = OE.sample_points_on_edges_global(np.asarray(m.vertices, float), ep, NE)
                e = float(cKDTree(gep).query(pep)[0].mean()
                          + cKDTree(pep).query(gep)[0].mean())
            if not np.isfinite(e):
                return None
            return {"CD": d1 + d2, "CD1": d1, "CD2": d2, "ECD": e,
                    "FINAL": 0.6 * (d1 + d2) + 0.4 * e, "F": int(len(m.faces))}

        w, info = make_walls(os.path.join(RB_REC, i + RB_FPSUF),
                             os.path.join(RB_PLY, i + ".ply"), z0q, z1q)
        return {"id": i, "base": score(base), "v10": score(union(base, w)),
                "info": info}
    except Exception as ex:                                  # noqa: BLE001
        return {"id": i, "err": str(ex)[:90]}


def run_bench():
    ids = sorted(f[:-len("_gt.obj")] for f in os.listdir(GT_DIR) if f.endswith("_gt.obj"))
    ids = [i for i in ids if os.path.exists(os.path.join(RB_REC, i + RB_SUF))
           and os.path.exists(os.path.join(RB_PLY, i + ".ply"))]
    print("realbench %d 栋  (z0q=%g z1q=%g)" % (len(ids), Z0Q, Z1Q), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, r in enumerate(ex.map(bench_one, [(i, Z0Q, Z1Q) for i in ids], chunksize=1)):
            rows.append(r)
            print("  %d/%d %s %s" % (k + 1, len(ids), r.get("id"), r.get("err", "")),
                  flush=True)
    tag = "z%g_%g" % (Z0Q, Z1Q)
    json.dump(rows, open(os.path.join(ROOT, "logs", "walls_bench_%s.json" % tag), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = [r for r in rows if "err" not in r and r.get("base") and r.get("v10")]
    skips = [r["id"] for r in rows if "err" not in r and r.get("base") and not r.get("v10")]
    print("\n可用 %d / %d   (未加墙 %d: %s)" % (len(ok), len(rows), len(skips), skips[:8]))
    if not ok:
        return 1
    f = lambda k, R: float(np.mean([r[R][k] for r in ok]))     # noqa: E731
    print("\n%-6s %8s %8s %8s %8s %9s %7s" % ("variant", "CD", "CD1", "CD2", "ECD",
                                              "FINAL", "F"))
    for R in ("base", "v10"):
        print("%-6s %8.4f %8.4f %8.4f %8.4f %9.4f %7.0f"
              % (R, f("CD", R), f("CD1", R), f("CD2", R), f("ECD", R), f("FINAL", R),
                 f("F", R)))
    d = np.array([r["v10"]["FINAL"] - r["base"]["FINAL"] for r in ok])
    b0 = f("FINAL", "base")
    print("\nΔFINAL %+.4f (相对 %+.2f%%)  改善 %d / 变差 %d"
          % (d.mean(), 100 * d.mean() / b0, int((d < -1e-9).sum()), int((d > 1e-9).sum())))
    dcd = np.array([r["v10"]["CD"] - r["base"]["CD"] for r in ok])
    dec = np.array([r["v10"]["ECD"] - r["base"]["ECD"] for r in ok])
    print("ΔCD %+.4f (相对 %+.2f%%)   ΔECD %+.4f (相对 %+.2f%%)"
          % (dcd.mean(), 100 * dcd.mean() / f("CD", "base"),
             dec.mean(), 100 * dec.mean() / f("ECD", "base")))
    if skips:
        for r in rows:
            if r.get("id") in skips and "base" in r:
                print("  跳过 %s : %s" % (r["id"], r.get("info")))
    print("\n写出 logs/walls_bench_%s.json" % tag)
    return 0


# ------------------------------------------------------------------ build
BASE_DIR = V9R      # 由 --base 覆盖


def build_one(job):
    sid, out_dir, z0q, z1q = job
    d = {"id": sid}
    try:
        src = os.path.join(BASE_DIR, sid + ".obj")
        dst = os.path.join(out_dir, sid + ".obj")
        if not os.path.exists(src) or os.path.getsize(src) == 0:
            return {**d, "err": "no-base"}
        w, info = make_walls(os.path.join(FP_DIR, sid + "_GeneratedFootprint.obj"),
                             os.path.join(PLY_DIR, sid + ".ply"), z0q, z1q)
        d.update(info)
        if w is None:
            shutil.copyfile(src, dst)
            d["out"] = "copy"
            return d
        base = load(src)
        if len(np.asarray(base.faces)) == 0:
            shutil.copyfile(src, dst)
            d["out"] = "copy-empty-base"
            return d
        m = union(base, w)
        m.export(dst)
        d["out"] = "walls"
        d["F_before"] = int(len(np.asarray(base.faces)))
        d["F_after"] = int(len(m.faces))
        return d
    except Exception as ex:                                  # noqa: BLE001
        try:
            shutil.copyfile(os.path.join(BASE_DIR, sid + ".obj"),
                            os.path.join(out_dir, sid + ".obj"))
            d["out"] = "copy-exc"
            d["err"] = str(ex)[:70]
        except Exception:                                    # noqa: BLE001
            d["err"] = str(ex)[:70]
        return d


def run_build(out_dir, jobs, z0q=None, z1q=None):
    z0q = Z0Q if z0q is None else z0q
    z1q = Z1Q if z1q is None else z1q
    ids = sorted((f[:-4] for f in os.listdir(BASE_DIR) if f.endswith(".obj")),
                 key=lambda s: int(s))
    print("构建 %d 栋 -> %s  (基座 %s, z0q=%g z1q=%g)"
          % (len(ids), out_dir, BASE_DIR, z0q, z1q), flush=True)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, r in enumerate(ex.map(build_one, [(i, out_dir, z0q, z1q) for i in ids],
                                     chunksize=20)):
            rows.append(r)
            if (k + 1) % 500 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    json.dump(rows, open(os.path.join(ROOT, "logs", "walls_build_%s.json"
                                      % os.path.basename(out_dir)), "w",
                         encoding="utf-8"), ensure_ascii=False)
    nw = sum(1 for r in rows if r.get("out") == "walls")
    ncp = sum(1 for r in rows if str(r.get("out", "")).startswith("copy"))
    print("\n加墙 %d 栋 / 原样复制 %d 栋 / 出错 %d"
          % (nw, ncp, sum(1 for r in rows if "err" in r)))
    from collections import Counter
    print("复制原因: %s" % Counter(r.get("skip", r.get("out", "?")) for r in rows
                                  if str(r.get("out", "")).startswith("copy")))
    zs = [r["z1"] - r["z0"] for r in rows if r.get("out") == "walls"]
    if zs:
        print("墙高: 中位 %.1f  p90 %.1f  最大 %.1f" % (np.median(zs), np.percentile(zs, 90),
                                                      np.max(zs)))
    n = len([f for f in os.listdir(out_dir) if f.endswith(".obj")])
    z0n = sum(1 for f in os.listdir(out_dir) if f.endswith(".obj")
              and os.path.getsize(os.path.join(out_dir, f)) == 0)
    print("输出目录 %d 个 obj，其中 0 字节 %d 个" % (n, z0n))
    print("写出 logs/walls_build_%s.json" % os.path.basename(out_dir))
    return 0


def main():
    global BASE_DIR, Z0Q, Z1Q
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench", choices=["bench", "build"])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "work", "final_v10"))
    ap.add_argument("--base", default=V9R, help="基座目录（build 模式）")
    ap.add_argument("--z0q", type=float, default=Z0Q, help="墙底 z 分位（默认 p2）")
    ap.add_argument("--z1q", type=float, default=Z1Q, help="墙顶 z 分位（默认 p30）")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    BASE_DIR = a.base
    Z0Q, Z1Q = a.z0q, a.z1q
    return run_bench() if a.mode == "bench" else run_build(a.out, a.jobs, a.z0q, a.z1q)


if __name__ == "__main__":
    sys.exit(main())
