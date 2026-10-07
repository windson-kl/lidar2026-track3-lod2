# -*- coding: utf-8 -*-
"""覆盖度体检（无需真值）：我们的 mesh 相对输入点云是否"漏建"。

动机：
    随机抽到的 bw600 样本 7192：输入点云跨度 13.6x14.7x9.1 m，
    但 GT 跨度 67.6x72.9x45.8 m（5 倍差）——G T 与 ply 不在同一范围。
    这有两种可能：(a) 该行 oracle 数据本身坏了；(b) 我们的重建严重漏建。
    两种都必须先查清楚，否则所有聚合结论都不可信。

本脚本对 (ply 目录, mesh 目录) 逐样本计算：
    npt      点云点数
    ply_diag 点云包围盒对角线
    mesh_diag/ply_diag  覆盖率 = 网格对角线 / 点云对角线
    F, V, area
并给出比值分布 + 最差的样本清单。附带 --gt 可选，用于同时体检真值帧。

用法:
    # 体检测试集提交（无需真值）
    python src/_t_coverage.py --ply data/work/ply --mesh data/work/final_v20 \
        --jobs 3 --limit 0 --out logs/cov_test.json
    # 体检 oracle 帧
    python src/_t_coverage.py --ply data/work/bw600/ply --mesh data/work/bw600/wC \
        --gt data/work/bw600_gt --jobs 3 --out logs/cov_oracle.json
"""
import argparse
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh

if 'faiss' not in sys.modules:
    _s = types.ModuleType('faiss')
    _s.IndexFlatL2 = object
    _s.StandardGpuResources = object
    _s.index_cpu_to_gpu = lambda *a, **k: None
    sys.modules['faiss'] = _s
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"D:/LiDAR2026"


def L(p):
    m = trimesh.load(p)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def diag(v):
    if v is None or len(v) == 0:
        return np.nan
    return float(np.linalg.norm(v.max(0) - v.min(0)))


def one(job):
    sid, plyd, meshd, gtd, cap = job
    try:
        pp = os.path.join(plyd, sid + ".ply")
        mp = None
        for suf in ("", "_ReconstructedModel"):
            q = os.path.join(meshd, sid + suf + ".obj")
            if os.path.exists(q) and os.path.getsize(q) > 0:
                mp = q
                break
        if mp is None:
            return {"id": sid, "err": "no-mesh"}
        if not os.path.exists(pp):
            return {"id": sid, "err": "no-ply"}
        pm = L(pp)
        pv = np.asarray(pm.vertices, float)
        npt = len(pv)
        pd = diag(pv)
        m = L(mp)
        mv = np.asarray(m.vertices, float)
        md = diag(mv)
        r = {"id": sid, "npt": int(npt),
             "ply_diag": float(pd), "mesh_diag": float(md),
             "ratio": float(md / pd) if pd > 0 else np.nan,
             "F": int(len(np.asarray(m.faces))), "V": int(len(mv)),
             "area": float(m.area)}
        if gtd:
            gp = os.path.join(gtd, sid + "_gt.obj")
            if not os.path.exists(gp):
                r["err"] = "no-gt"
                return r
            g = L(gp)
            gv = np.asarray(g.vertices, float)
            r["gt_diag"] = float(diag(gv))
            r["gt_F"] = int(len(np.asarray(g.faces)))
            r["gt_ratio"] = float(diag(gv) / pd) if pd > 0 else np.nan
        return r
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.ids:
        ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()]
    else:
        ids = sorted(os.path.basename(p)[:-4] for p in
                     os.listdir(os.path.join(ROOT, a.mesh)) if p.endswith(".obj"))
    if a.limit:
        ids = ids[::max(1, len(ids) // a.limit)][:a.limit]
    print("覆盖度体检 n=%d" % len(ids), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.ply, a.mesh, a.gt, 0) for i in ids],
                                     chunksize=4)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)
    ok = [r for r in rows if "err" not in r and np.isfinite(r.get("ratio", np.nan))]
    print("\n可用 %d / %d" % (len(ok), len(rows)))
    from collections import Counter
    print("错误分布:", Counter(r["err"] for r in rows if "err" in r))
    if not ok:
        return 1
    ra = np.array([r["ratio"] for r in ok])
    print("mesh_diag/ply_diag: 中位 %.3f  均值 %.3f  p05 %.3f  p25 %.3f  p75 %.3f  p95 %.3f" %
          (np.median(ra), ra.mean(), np.percentile(ra, 5), np.percentile(ra, 25),
           np.percentile(ra, 75), np.percentile(ra, 95)))
    for th in (0.9, 0.7, 0.5, 0.3):
        print("  比值 < %.1f 的样本 %d (%.1f%%)" % (th, (ra < th).sum(), 100 * (ra < th).mean()))
    print("  比值 > 1.3 的样本 %d (%.1f%%)" % ((ra > 1.3).sum(), 100 * (ra > 1.3).mean()))
    if "gt_ratio" in ok[0]:
        gr = np.array([r["gt_ratio"] for r in ok if np.isfinite(r.get("gt_ratio", np.nan))])
        print("gt_diag/ply_diag: 中位 %.3f  均值 %.3f  p05 %.3f  p95 %.3f" %
              (np.median(gr), gr.mean(), np.percentile(gr, 5), np.percentile(gr, 95)))
        bad = [r for r in ok if np.isfinite(r.get("gt_ratio", np.nan))
               and (r["gt_ratio"] < 0.7 or r["gt_ratio"] > 1.4)]
        print("  真值帧与点云不匹配(比值<0.7 或 >1.4)的行: %d" % len(bad))
        for r in bad[:15]:
            print("    %-8s ply %.1f  gt %.1f  gt/ply %.2f  npt %d" %
                  (r["id"], r["ply_diag"], r["gt_diag"], r["gt_ratio"], r["npt"]))
    print("\n最漏建的 20 个（比值最小，npt>=200）:")
    cand = [r for r in ok if r["npt"] >= 200]
    for r in sorted(cand, key=lambda x: x["ratio"])[:20]:
        print("   %-8s npt %7d  ply %8.1f  mesh %8.1f  比值 %.3f  F %5d" %
              (r["id"], r["npt"], r["ply_diag"], r["mesh_diag"], r["ratio"], r["F"]))
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
