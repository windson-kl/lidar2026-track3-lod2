# -*- coding: utf-8 -*-
"""参数化的 walls 造墙器 + bw600 真值裁决（2026-10-05 新增）。

动机：v10 的墙参数（z0=p2, z1=p30）与 v11 的墙参数（z0=p0, z1=p25）给出**互相矛盾**
的两条证据：
  * realbench 32 栋真值（§38，配对 seed 固定）：z0=p0 更好（ΔCD −0.1138 vs −0.0925）
  * GT-free 门控（4000 栋）：z0=p0 更差（m2p 1.966→1.997, +1.58%）
32 栋太少、代理又不可信 ⇒ 用 **bw600（466 栋干净子集）** 做独立第三方裁决。

与 `_t_build_walls.py` 的区别：所有目录/参数都从命令行给，不改全局（Windows spawn 安全）。

用法：
  # 1) 造两套墙
  python src/_t_walls_cfg.py --mode build --base data/work/bw600/final \
      --fp data/work/bw600/recon --ply data/work/bw600/ply \
      --out data/work/bw600/wA --z0q 2 --z1q 30 --jobs 6
  python src/_t_walls_cfg.py --mode build --base data/work/bw600/final \
      --fp data/work/bw600/recon --ply data/work/bw600/ply \
      --out data/work/bw600/wB --z0q 0 --z1q 25 --jobs 6
  # 2) 打分（用现成的官方打分器）
  python src/_t_bw_score.py --rec data/work/bw600/wA --gt data/work/bw600_gt \
      --ply data/work/bw600/ply --out logs/bw600_wA.json --jobs 6
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import trimesh

ROOT = r"D:/LiDAR2026"
H_CAP = 150.0
FP_DIAG_MAX = 150.0


def load(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def pcv(pc):
    return np.asarray(pc.vertices if hasattr(pc, "vertices") else pc.points, float)


def weld_footprint(V, F, tol=1e-4):
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


def union(a, b):
    if b is None or len(np.asarray(b.faces)) == 0:
        return a
    va, fa = np.asarray(a.vertices, float), np.asarray(a.faces, int).reshape(-1, 3)
    vb, fb = np.asarray(b.vertices, float), np.asarray(b.faces, int).reshape(-1, 3)
    return trimesh.Trimesh(vertices=np.vstack([va, vb]),
                           faces=np.vstack([fa, fb + len(va)]), process=False)


def make_walls(fp_path, ply_path, z0q, z1q, fpmax=FP_DIAG_MAX, hcap=H_CAP):
    if not os.path.exists(fp_path) or not os.path.exists(ply_path):
        return None, {"skip": "missing"}
    try:
        fpm = load(fp_path)
        FV = np.asarray(fpm.vertices, float)
        FF = np.asarray(fpm.faces, int).reshape(-1, 3)
    except Exception as ex:                                  # noqa: BLE001
        return None, {"skip": "bad-fp:%s" % str(ex)[:40]}
    if len(FF) == 0:
        return None, {"skip": "empty-fp"}
    fdiag = float(np.linalg.norm(FV.max(0)[:2] - FV.min(0)[:2]))
    if fdiag > fpmax:
        return None, {"skip": "scene-fp", "fp_diag": fdiag}
    pv = pcv(load(ply_path))
    if len(pv) == 0:
        return None, {"skip": "empty-ply"}
    lo, hi = FV.min(0)[:2], FV.max(0)[:2]
    inb = ((pv[:, 0] >= lo[0]) & (pv[:, 0] <= hi[0])
           & (pv[:, 1] >= lo[1]) & (pv[:, 1] <= hi[1]))
    zin = pv[inb][:, 2] if inb.sum() > 50 else pv[:, 2]
    z0 = float(np.percentile(zin, z0q))
    z1 = min(float(np.percentile(zin, z1q)), z0 + hcap)
    if z1 - z0 < 0.5:
        return None, {"skip": "flat", "z0": z0, "z1": z1}
    Vw, Fw = weld_footprint(FV, FF)
    w = build_walls(Vw, Fw, z0, z1)
    if w is None or len(w.faces) == 0:
        return None, {"skip": "no-walls"}
    return w, {"fp_diag": fdiag, "z0": z0, "z1": z1, "h": z1 - z0,
               "n_wall_f": int(len(w.faces))}


def one(job):
    sid, base_d, fp_d, ply_d, out_d, z0q, z1q, fpmax, hcap = job
    try:
        bp = None
        for suf in ("", "_ReconstructedModel"):
            p = os.path.join(base_d, sid + suf + ".obj")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                bp = p
                break
        if bp is None:
            return {"id": sid, "err": "no-base"}
        base = load(bp)
        w, info = make_walls(os.path.join(fp_d, sid + "_GeneratedFootprint.obj"),
                             os.path.join(ply_d, sid + ".ply"), z0q, z1q, fpmax, hcap)
        out = union(base, w) if w is not None else base
        op = os.path.join(out_d, sid + ".obj")
        out.export(op)
        info["id"] = sid
        info["ok"] = True
        info["F"] = int(len(np.asarray(out.faces)))
        return info
    except Exception as ex:                                  # noqa: BLE001
        return {"id": sid, "err": str(ex)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build"], default="build")
    ap.add_argument("--base", required=True)
    ap.add_argument("--fp", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z0q", type=float, required=True)
    ap.add_argument("--z1q", type=float, required=True)
    ap.add_argument("--fpmax", type=float, default=FP_DIAG_MAX)
    ap.add_argument("--hcap", type=float, default=H_CAP)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.ids:
        ids = a.ids
    else:
        ids = sorted(f[:-4] for f in os.listdir(a.base) if f.endswith(".obj"))
    print("base=%s  待处理 %d 栋  z0q=%g z1q=%g -> %s"
          % (a.base, len(ids), a.z0q, a.z1q, a.out), flush=True)

    from concurrent.futures import ProcessPoolExecutor
    jobs = [(i, a.base, a.fp, a.ply, a.out, a.z0q, a.z1q, a.fpmax, a.hcap) for i in ids]
    rows, n_wall, n_skip, n_err, n_copy = [], 0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if r.get("err"):
                n_err += 1
            elif r.get("skip"):
                n_skip += 1
                n_copy += 1
            else:
                n_wall += 1
            if (k + 1) % 100 == 0:
                print("  %d/%d 加墙 %d 跳过 %d 出错 %d"
                      % (k + 1, len(ids), n_wall, n_skip, n_err), flush=True)

    hs = [r["h"] for r in rows if r.get("ok") and not r.get("skip")]
    tag = os.path.basename(os.path.normpath(a.out))
    lg = os.path.join(ROOT, "logs", "walls_cfg_%s.json" % tag)
    json.dump(rows, open(lg, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n加墙 %d / 跳过(原样复制) %d / 出错 %d  总 %d" % (n_wall, n_copy, n_err, len(ids)))
    if hs:
        hs = np.array(hs)
        print("墙高 中位 %.1f  p90 %.1f  max %.1f" % (np.median(hs), np.percentile(hs, 90), hs.max()))
    from collections import Counter
    print("跳过原因:", Counter(r.get("skip", "err") for r in rows if r.get("skip") or r.get("err")))
    print("写出 %s ; 日志 %s" % (a.out, lg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
