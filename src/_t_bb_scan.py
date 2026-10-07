# -*- coding: utf-8 -*-
"""轻量扫描：真实提交上，bbox 规则各 pad 会删掉多少面/面积（不做采样，秒级完成）。"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh

ROOT = r"D:/LiDAR2026"
PADS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]


def L(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def one(job):
    sid, meshd, plyd = job
    try:
        m = L(os.path.join(meshd, sid + ".obj"))
        pts = np.asarray(trimesh.load(os.path.join(plyd, sid + ".ply"),
                                      process=False).vertices, float)
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        if len(F) == 0 or len(pts) < 8:
            return {"id": sid, "err": "empty"}
        A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                          V[F[:, 2]] - V[F[:, 0]]), axis=1)
        At = float(A.sum())
        lo, hi = pts.min(0), pts.max(0)
        d = float(np.linalg.norm(hi - lo))
        row = {"id": sid, "F": int(len(F)), "area": At, "pdiag": d,
               "m_diag": float(np.linalg.norm(V.max(0) - V.min(0))),
               "over_axis": float(np.max(np.maximum((V.min(0) - lo), (hi - V.max(0)))
                                         / max(d, 1e-9))), "pads": {}}
        for p in PADS:
            pad = p * d
            ins = np.all((V >= lo - pad) & (V <= hi + pad), axis=1)
            fi = ins[F].all(axis=1)
            row["pads"]["%.2f" % p] = {
                "rm_f": float((~fi).mean()),
                "rm_a": float(A[~fi].sum() / max(At, 1e-9))}
        return row
    except Exception as ex:                                                 # noqa: BLE001
        return {"id": sid, "err": "%s: %s" % (type(ex).__name__, str(ex)[:70])}


def main():
    meshd = sys.argv[1] if len(sys.argv) > 1 else "data/work/final_v28"
    plyd = sys.argv[2] if len(sys.argv) > 2 else "data/work/ply"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    os.chdir(ROOT)
    ids = sorted(set(os.path.splitext(f)[0] for f in os.listdir(meshd)
                     if f.endswith(".obj")))
    ids = [i for i in ids if os.path.exists(os.path.join(plyd, i + ".ply"))]
    if n and n < len(ids):
        ids = list(np.random.default_rng(5).choice(ids, n, replace=False))
    print("%s  %d 场景" % (meshd, len(ids)), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(one, [(i, meshd, plyd) for i in ids], chunksize=8):
            rows.append(r)
    ok = [r for r in rows if "err" not in r]
    print("有效 %d" % len(ok))
    if not ok:
        return
    for p in PADS:
        k = "%.2f" % p
        print("  pad=%4.0f%%  删面 均值%6.3f%% 中位%6.3f%% p90%6.3f%% | "
              "删面积 均值%6.3f%% 中位%6.3f%% p90%6.3f%% | 有删除的场景占比 %5.1f%%"
              % (100 * p,
                 100 * np.mean([r["pads"][k]["rm_f"] for r in ok]),
                 100 * np.median([r["pads"][k]["rm_f"] for r in ok]),
                 100 * np.percentile([r["pads"][k]["rm_f"] for r in ok], 90),
                 100 * np.mean([r["pads"][k]["rm_a"] for r in ok]),
                 100 * np.median([r["pads"][k]["rm_a"] for r in ok]),
                 100 * np.percentile([r["pads"][k]["rm_a"] for r in ok], 90),
                 100 * np.mean([r["pads"][k]["rm_f"] > 0 for r in ok])))
    ov = [r["over_axis"] for r in ok]
    print("\n  单轴越界量 / 点云diag: p50=%.4f p90=%.4f p99=%.4f max=%.4f"
          % tuple(float(np.percentile(ov, q)) for q in (50, 90, 99, 100)))


if __name__ == "__main__":
    main()
