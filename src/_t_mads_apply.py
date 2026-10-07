# -*- coding: utf-8 -*-
"""生产算子 v30：**多轴高程面 MADS**——在 v28(z 轴) 基础上再加 x / y 两轴的高程面。

依据（logs/mads_rand16.json，全集随机 16 场景，gate=0，官方双向口径）：
    base   CD1 4.2277 CD2 1.5524 E1 7.7164 E2 2.3022 FINAL 7.4755
    mz3    CD1 3.8570 CD2 1.4579 E1 7.4746 E2 1.9927 FINAL 6.9758  ΔFINAL −0.4996
    mxy3   CD1 3.6322 CD2 1.4341 E1 7.2823 E2 1.8940 FINAL 6.7103  ΔFINAL −0.7652
    mxy4   CD1 3.7014 CD2 1.4334 E1 7.3601 E2 1.8896 FINAL 6.7808  ΔFINAL −0.6947
⇒ 在 v28 之上再加 x/y 两轴各一对 min/max 高程面，净赚 −0.27 本地 FINAL（CD1/E1 为主）。

机制：真值面积 89% 是近竖直面，而"沿 z 投影的高程面"在墙中段完全没有几何。
     沿 x / y 投影的 min/max 面恰好补上 ±x / ±y 朝向的墙，且按构造贴近点云
     （而真值与点云的偏差只有 0.12 m）⇒ 同时压低 CD1 与 E1。

只使用：自己的网格 + 该样本的输入点云。不读任何真值。输出为"纯文本追加"，
保证是 v28/v21 的字节前缀超集。

用法:
    python src/_t_mads_apply.py --meshd data/work/final_v28 --plyd data/work/ply \
        --outd data/work/final_v30 --g 3.0 --diag-min 30 --axes z,x,y --jobs 3 \
        --face-cap 40000 --report logs/apply_v30.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _t_dsm_apply import dsm_grid, load_pts, count_v          # noqa: E402

ROOT = r"D:/LiDAR2026"
PERM = {0: (1, 2, 0), 1: (0, 2, 1), 2: (0, 1, 2)}


def dsm_grid_axis(pts, g, max_edge, zmode, face_cap, cell_cap, seed, axis):
    """沿 axis 投影做高程面。axis=2 与 _t_dsm_apply.dsm_grid 完全一致。"""
    if axis == 2:
        return dsm_grid(pts, g, max_edge, zmode, face_cap, seed, cell_cap)
    perm = list(PERM[axis])
    q = np.ascontiguousarray(pts[:, perm])
    r = dsm_grid(q, g, max_edge, zmode, face_cap, seed, cell_cap)
    if r is None:
        return None
    p, t = r
    out = np.empty_like(p)
    out[:, perm[0]] = p[:, 0]
    out[:, perm[1]] = p[:, 1]
    out[:, perm[2]] = p[:, 2]
    return out, t


def one(job):
    sid, meshd, plyd, outd, g, dmin, axes, face_cap, cell_cap, a_max_edge = job
    try:
        mp = os.path.join(meshd, sid + ".obj")
        pp = os.path.join(plyd, sid + ".ply")
        if not os.path.exists(mp):
            return {"id": sid, "err": "no-mesh"}
        with open(mp, "r", encoding="utf-8", errors="ignore") as f:
            base = f.read()
        if not os.path.exists(pp):
            open(os.path.join(outd, sid + ".obj"), "w").write(base)
            return {"id": sid, "gate": 0, "nadd": 0, "why": "no-ply"}
        pts = load_pts(pp)
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        if diag < dmin:
            open(os.path.join(outd, sid + ".obj"), "w").write(base)
            return {"id": sid, "gate": 0, "nadd": 0, "diag": diag,
                    "npt": int(len(pts)), "why": "small"}
        me = (3.0 * g) if a_max_edge is None else (None if a_max_edge == 0 else a_max_edge)
        blocks = []
        ii = 0
        for ax in axes:
            for md in ("min", "max"):
                r = dsm_grid_axis(pts, g, me, md, face_cap, cell_cap,
                                  abs(hash(sid)) % (2 ** 31) + ii * 7 + ax, ax)
                ii += 1
                if r is not None:
                    blocks.append(r)
        if not blocks:
            open(os.path.join(outd, sid + ".obj"), "w").write(base)
            return {"id": sid, "gate": 1, "nadd": 0, "diag": diag,
                    "npt": int(len(pts)), "why": "dsm-fail"}
        if not base.endswith("\n"):
            base += "\n"
        buf = [base]
        off = count_v(base)
        nadd = 0
        for p, t in blocks:
            buf.append("".join("v %.6f %.6f %.6f\n" % (x, y, z) for x, y, z in p))
            buf.append("".join("f %d %d %d\n" % (off + a + 1, off + b + 1, off + c + 1)
                               for a, b, c in t))
            off += len(p)
            nadd += len(t)
        open(os.path.join(outd, sid + ".obj"), "w").write("".join(buf))
        return {"id": sid, "gate": 1, "nadd": int(nadd), "diag": diag,
                "npt": int(len(pts)), "nblk": len(blocks)}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshd", required=True)
    ap.add_argument("--plyd", required=True)
    ap.add_argument("--outd", required=True)
    ap.add_argument("--g", type=float, default=3.0)
    ap.add_argument("--diag-min", type=float, default=30.0)
    ap.add_argument("--axes", default="z,x,y")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--face-cap", type=int, default=40000)
    ap.add_argument("--cell-cap", type=int, default=100000)
    ap.add_argument("--max-edge", type=float, default=None)
    ap.add_argument("--report", default="logs/apply_v30.json")
    a = ap.parse_args()
    axes = [{"z": 2, "x": 0, "y": 1}[t] for t in a.axes.split(",") if t]
    os.makedirs(a.outd, exist_ok=True)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(a.meshd) if f.endswith(".obj"))
    print("[mads_apply] %d 个网格, axes=%s g=%.2f diag_min=%.0f jobs=%d" %
          (len(ids), a.axes, a.g, a.diag_min, a.jobs), flush=True)
    t0 = time.time()
    rec = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(
                one, [(i, a.meshd, a.plyd, a.outd, a.g, a.diag_min, axes,
                       a.face_cap, a.cell_cap, a.max_edge) for i in ids], chunksize=8)):
            rec.append(r)
            if (k + 1) % 200 == 0 or k + 1 == len(ids):
                print("  %5d/%d  %.1f min" % (k + 1, len(ids), (time.time() - t0) / 60), flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.report)), exist_ok=True)
    json.dump(rec, open(os.path.join(ROOT, a.report), "w", encoding="utf-8"), ensure_ascii=False)
    ok = [r for r in rec if "err" not in r]
    g1 = [r for r in ok if r.get("gate") == 1]
    print("\n[完成] 有效 %d  门控命中 %d (%.1f%%)  错误 %d" %
          (len(ok), len(g1), 100.0 * len(g1) / max(len(ok), 1), len(rec) - len(ok)))
    if g1:
        n = np.array([r["nadd"] for r in g1])
        print("  新增面数 中位 %d / 均值 %.0f / 合计 %d" %
              (int(np.median(n)), n.mean(), int(n.sum())))
    tot = sum(os.path.getsize(os.path.join(a.outd, f))
              for f in os.listdir(a.outd) if f.endswith(".obj"))
    print("  输出目录合计 %.1f MB  (%.1f min)" % (tot / 1e6, (time.time() - t0) / 60))
    for r in rec:
        if "err" in r:
            print("   ERR", r["id"], r["err"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
