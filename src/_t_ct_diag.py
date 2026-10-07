# -*- coding: utf-8 -*-
"""真实提交上的「面质心离点云距离」画像：
    ct1 到底删了什么？有没有误伤"顶点都贴近点云、只是质心偏远"的合法大面片？

对每个场景统计 final_v28 的面：
    d_c   = 面质心 -> 最近点云点 距离
    d_vmax= 三顶点到点云距离的最大值
    a     = 面面积
    e     = 面最大边长
分层输出：按 d_c 分箱的面积占比；并按 "是否所有顶点都 <1 m" 区分"疑似合法大面片"。
"""
import argparse
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = r"D:/LiDAR2026"
CUTS = [0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1e9]


def L(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def load_ply_pts(p, cap=400000, seed=0):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    v = np.asarray(m.vertices, float)
    if len(v) > cap:
        v = v[np.random.default_rng(seed).choice(len(v), cap, replace=False)]
    return v


def one(job):
    sid, meshd, plyd = job
    try:
        m = L(os.path.join(meshd, sid + ".obj"))
        pts = load_ply_pts(os.path.join(plyd, sid + ".ply"))
        if len(pts) < 8:
            return {"id": sid, "err": "tiny"}
        tree = cKDTree(pts)
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        if len(F) == 0:
            return {"id": sid, "err": "no_face"}
        A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                          V[F[:, 2]] - V[F[:, 0]]), axis=1)
        dc = tree.query(V[F].mean(1))[0]
        dv = np.stack([tree.query(V[F[:, k]])[0] for k in range(3)], 1)
        dvmax = dv.max(1)
        e = np.stack([np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1),
                      np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1),
                      np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1)], 1).max(1)
        At = float(A.sum())
        bins = []
        for i in range(len(CUTS) - 1):
            s = (dc >= CUTS[i]) & (dc < CUTS[i + 1])
            bins.append({"lo": CUTS[i], "hi": CUTS[i + 1],
                         "a_share": float(A[s].sum() / max(At, 1e-9)),
                         "n_share": float(s.mean())})
        # 疑似合法：质心远但三顶点都近
        far = dc > 1.0
        legit_like = far & (dvmax <= 1.0)
        return {"id": sid, "F": int(len(F)), "area": At,
                "dc_q": [float(np.percentile(dc, p)) for p in (50, 90, 99, 100)],
                "dvmax_q": [float(np.percentile(dvmax, p)) for p in (50, 90, 99, 100)],
                "far_a_share": float(A[far].sum() / max(At, 1e-9)),
                "far_n_share": float(far.mean()),
                "legitlike_a_share": float(A[legit_like].sum() / max(At, 1e-9)),
                "legitlike_n": int(legit_like.sum()),
                "far_maxE_med": float(np.median(e[far])) if far.any() else 0.0,
                "bins": bins}
    except Exception as ex:                                            # noqa: BLE001
        return {"id": sid, "err": "%s: %s" % (type(ex).__name__, str(ex)[:80])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshd", default="data/work/final_v28")
    ap.add_argument("--plyd", default="data/work/ply")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="logs/ct_diag.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.ids:
        ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()]
    else:
        ids = sorted(set(os.path.splitext(f)[0] for f in os.listdir(a.meshd)
                         if f.endswith(".obj")))
        if a.n and a.n < len(ids):
            ids = list(np.random.default_rng(a.seed).choice(ids, a.n, replace=False))
    ids = [i for i in ids if os.path.exists(os.path.join(a.meshd, i + ".obj"))
           and os.path.exists(os.path.join(a.plyd, i + ".ply"))]
    print("场景", len(ids), flush=True)
    rows = []
    jl = os.path.join(ROOT, a.out.replace(".json", ".jsonl"))
    open(jl, "w", encoding="utf-8").close()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.meshd, a.plyd) for i in ids],
                                     chunksize=1)):
            rows.append(r)
            with open(jl, "a", encoding="utf-8") as fj:
                fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (k + 1) % 50 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)
    ok = [r for r in rows if "err" not in r]
    print("\n有效 %d" % len(ok))
    if not ok:
        return 1
    print("\n=== 面质心离点云距离 d_c 的分箱（按面积占比，跨场景平均）===")
    for i in range(len(CUTS) - 1):
        s = float(np.mean([r["bins"][i]["a_share"] for r in ok]))
        n = float(np.mean([r["bins"][i]["n_share"] for r in ok]))
        if s > 1e-6 or n > 1e-6:
            print("   d_c ∈ [%7.2f,%9.2f)   面积占比 %7.4f   面数占比 %7.4f"
                  % (CUTS[i], CUTS[i + 1], s, n))
    print("\n=== 关键量（跨场景平均）===")
    print("   d_c > 1 m 的面:  面积占比 %.4f   面数占比 %.4f"
          % (float(np.mean([r["far_a_share"] for r in ok])),
             float(np.mean([r["far_n_share"] for r in ok]))))
    print("   其中「三顶点都 <=1 m」的疑似合法大面片: 面积占比 %.4f  平均 %.1f 个/场景"
          % (float(np.mean([r["legitlike_a_share"] for r in ok])),
             float(np.mean([r["legitlike_n"] for r in ok]))))
    print("   d_c > 1 m 的面 的边长中位: %.2f m"
          % float(np.mean([r["far_maxE_med"] for r in ok])))
    q = lambda k, p: float(np.percentile([r[k] for r in ok], p))
    print("\n   d_c   分位 p50=%.3f p90=%.3f p99=%.3f max=%.2f"
          % tuple([float(np.median([r["dc_q"][i] for r in ok])) for i in range(4)]))
    print("   d_vmax 分位 p50=%.3f p90=%.3f p99=%.3f max=%.2f"
          % tuple([float(np.median([r["dvmax_q"][i] for r in ok])) for i in range(4)]))
    print("\n写入", a.out)


if __name__ == "__main__":
    main()
