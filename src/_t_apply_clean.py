# -*- coding: utf-8 -*-
"""把「零风险尾部清理」算子应用到测试集产出，生成新一版 final_v12。

算子（全部 GT-free，且在 bw600 好楼 150 栋上实测 ΔFINAL = +0.0000、0 胜 0 负）：
  1) zc_c2          裁掉面心落在「点云 z 包围盒 ±2 m」之外的面（保险丝，防 z 爆炸）
  2) far5&sliv40    删掉「面心离最近点云 > 5 m」且「细长比 maxEdge^2/area > 40」的面
  3) slivP          删掉「面心离最近点云 > 5 m」且「细长比 > P」的面（P 可调）

bw600 真值实测（尾部 worst-59，基座 FINAL 和 1818.9）：
  far5&sliv40   +794.5（43.7%）  自身回退 7.9   好楼回退 0.0
  far5&sliv100  +524.0（28.8%）  自身回退 0.9   好楼回退 0.0
  zc_c2         +945.7（52.0%）  自身回退 0.0   好楼回退 0.0

用法:
  python src/_t_apply_clean.py --src data/work/final_v10 --dst data/work/final_v12 \
      --ply data/work/ply --mode far_sliv --sliv 40 --jobs 8
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = r"D:/LiDAR2026"
FAR = 5.0


def L(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def one(job):
    sid, src, dst, plyd, mode, sliv, maxpt, bigonly = job
    try:
        mp = os.path.join(src, sid + ".obj")
        pp = os.path.join(plyd, sid + ".ply")
        if not os.path.exists(mp) or not os.path.exists(pp):
            return {"id": sid, "err": "missing"}
        m = L(mp)
        V = np.asarray(m.vertices, float)
        F = np.asarray(m.faces, int).reshape(-1, 3)
        if len(F) == 0:
            return {"id": sid, "err": "no-face"}
        pv = np.asarray(L(pp).vertices, float)
        if len(pv) > maxpt:
            pv = pv[np.random.default_rng(0).choice(len(pv), maxpt, replace=False)]

        nF0 = len(F)
        T = V[F]
        a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) * 0.5
        e = np.stack([np.linalg.norm(T[:, 1] - T[:, 0], axis=1),
                      np.linalg.norm(T[:, 2] - T[:, 1], axis=1),
                      np.linalg.norm(T[:, 0] - T[:, 2], axis=1)], 1)
        asp = (e.max(1) ** 2) / np.maximum(a, 1e-12)
        c = T.mean(axis=1)

        keep = np.ones(len(F), bool)
        if mode in ("far_sliv", "both"):
            d = cKDTree(pv).query(c, k=1)[0]
            msk = (d > FAR) & (asp > sliv)
            if bigonly and msk.sum() > 1:
                # 只删面积 >= 中位数的那些：小面积的远细长面是 ECD 杀手
                # （删它们制造许多"短而远"的新开边界，而官方把边界边直接当锐边采样）
                am = float(np.median(a[msk]))
                msk = msk & (a >= am)
            keep &= ~msk
        if mode in ("zc_only", "both"):
            # zc_c2：面心 z 落在点云 z 包围盒 ±2 m 之外则删
            zlo, zhi = pv[:, 2].min() - 2.0, pv[:, 2].max() + 2.0
            keep &= (c[:, 2] >= zlo) & (c[:, 2] <= zhi)

        if not keep.any():
            return {"id": sid, "err": "all-removed"}
        out = trimesh.Trimesh(vertices=V, faces=F[keep], process=False)
        op = os.path.join(dst, sid + ".obj")
        out.export(op)
        kept_area = float(a[keep].sum())
        return {"id": sid, "nF": nF0, "nF2": int(keep.sum()),
                "area0": float(a.sum()), "area1": kept_area,
                "frac_area_removed": float(1.0 - kept_area / max(1e-12, a.sum()))}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "work", "final_v10"))
    ap.add_argument("--dst", default=os.path.join(ROOT, "data", "work", "final_v12"))
    ap.add_argument("--ply", default=os.path.join(ROOT, "data", "work", "ply"))
    ap.add_argument("--mode", default="far_sliv",
                    choices=["far_sliv", "zc_only", "both"])
    ap.add_argument("--sliv", type=float, default=40.0)
    ap.add_argument("--maxpt", type=int, default=200000)
    ap.add_argument("--big", action="store_true",
                    help="只删「面积 >= 被选中面中位数」的远细长面（v14 用）")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    ids = sorted(x[:-4] for x in os.listdir(a.src) if x.endswith(".obj"))
    print("处理 %d 栋  mode=%s sliv=%g big=%s  %s -> %s" %
          (len(ids), a.mode, a.sliv, a.big, a.src, a.dst), flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.src, a.dst, a.ply, a.mode, a.sliv,
                                            a.maxpt, a.big) for i in ids], chunksize=8)):
            res.append(r)
            if (k + 1) % 500 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    ok = [r for r in res if "err" not in r]
    print("\n成功 %d / %d" % (len(ok), len(res)))
    fr = np.array([r["frac_area_removed"] for r in ok])
    changed = fr > 1e-9
    print("被改动的样本 %d (%.1f%%)" % (int(changed.sum()), 100 * changed.mean()))
    if changed.any():
        print("改动样本的面积删除比  p50 %.3f p90 %.3f max %.3f" %
              (np.median(fr[changed]), np.percentile(fr[changed], 90), fr.max()))
    tag = "%s%s_%s" % (a.mode, "_big" if a.big else "",
                       os.path.basename(os.path.normpath(a.dst)))
    json.dump(res, open(os.path.join(ROOT, "logs", "apply_clean_%s.json" % tag),
                        "w", encoding="utf-8"), ensure_ascii=False)
    print("日志 logs/apply_clean_%s.json" % tag)
    print("写出 %s（%d 个 obj）" % (a.dst, len(ok)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
