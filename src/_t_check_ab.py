# -*- coding: utf-8 -*-
"""通用 GT-free 几何代理 A/B 对比（提交前安全检查）。

对比两个目录下同 id 的网格在**无真值代理**上的表现：
    p2m      点云 -> 网格 的平均距离（约等于 CD2，越接近 0 越好）
    p2m_p90  同上 90 分位（与真实 ECD/FINAL 相关性 +0.91 / +0.89）
    m2p      网格 -> 点云 的平均距离（CD1 的代理）
    area/F   面积与面片数（确认没有爆炸）

用法:
    python src/_t_check_ab.py --a data/work/final_v9r --b data/work/final_v11 --n 400
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = r"D:/LiDAR2026"
PLY = os.path.join(ROOT, "data", "work", "ply")


def L(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def one(job):
    sid, dirs = job
    try:
        pv = np.asarray(L(os.path.join(PLY, sid + ".ply")).vertices, float)
        rng = np.random.default_rng(0)
        q = pv if len(pv) <= 5000 else pv[rng.choice(len(pv), 5000, replace=False)]
        tree = cKDTree(pv if len(pv) <= 20000
                       else pv[rng.choice(len(pv), 20000, replace=False)])
        o = {"id": sid}
        for tag, d in dirs:
            p = os.path.join(d, sid + ".obj")
            if not os.path.exists(p):
                o[tag] = None
                continue
            m = L(p)
            F = np.asarray(m.faces, int).reshape(-1, 3)
            if len(F) == 0:
                o[tag] = None
                continue
            dd = np.asarray(m.nearest.on_surface(q)[1], float)
            np.random.seed(0)
            ps = trimesh.sample.sample_surface(m, 5000)[0]
            dm = np.asarray(tree.query(ps, k=1)[0], float)
            o[tag] = {"p2m": float(dd.mean()),
                      "p2m_p90": float(np.percentile(dd, 90)),
                      "m2p": float(dm.mean()), "F": int(len(F)),
                      "area": float(m.area)}
        return o
    except Exception as e:
        return {"id": sid, "err": str(e)[:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(ROOT, "data", "work", "final_v9r"))
    ap.add_argument("--b", default=os.path.join(ROOT, "data", "work", "final_v11"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tailf = os.path.join(ROOT, "logs", "final_hat_ids.txt")
    tail = ([l.strip() for l in open(tailf, encoding="utf-8") if l.strip()][:300]
            if os.path.exists(tailf) else [])
    allids = sorted(set(tail) | {f[:-4] for f in os.listdir(a.b) if f.endswith(".obj")},
                    key=int)[-a.n:]
    dirs = [("A", a.a), ("B", a.b)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(one, [(i, dirs) for i in allids], chunksize=10):
            rows.append(r)
    ok = [r for r in rows if "err" not in r and r.get("A") and r.get("B")]
    print("A = %s" % a.a)
    print("B = %s" % a.b)
    print("样本 %d / 请求 %d（跳过 %d）" % (len(ok), len(allids), len(allids) - len(ok)))
    for k in ("p2m", "p2m_p90", "m2p", "area", "F"):
        x = np.array([r["A"][k] for r in ok])
        y = np.array([r["B"][k] for r in ok])
        d = y - x
        print("%-9s A %10.3f -> B %10.3f  d%+9.3f  变差 %3d / 改善 %3d"
              % (k, x.mean(), y.mean(), d.mean(), int((d > 1e-9).sum()),
                 int((d < -1e-9).sum())))
    dp = np.array([r["B"]["p2m"] - r["A"]["p2m"] for r in ok])
    dq = np.array([r["B"]["p2m_p90"] - r["A"]["p2m_p90"] for r in ok])
    bad = (dp > 1e-9) & (dq > 1e-9)
    ids = [r["id"] for r, m in zip(ok, bad) if m]
    print("\np2m 与 p2m_p90 同时变差: %d 栋" % len(ids))
    if ids:
        print("  %s" % ", ".join(ids[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
