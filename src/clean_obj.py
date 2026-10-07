# -*- coding: utf-8 -*-
"""文本级 OBJ 清理器：删除"含超长边"的三角面 —— 修复我们自己重建管线产生的垃圾薄片。

依据（已实测）：
    · 官方 ECD 的 E1 = 在我方锐边上按**长度加权**抽样后求到真值锐边的平均距离；
      "1 个面/非流形/二面角>30°"都算锐边。
    · 已提交的 final_v28(4000)：>50 m 的边 35206 条(8.8/场景)、>100 m 9093 条、最长 733 m。
      base(wC,598)：>50 m 13927 条、最长 764,092 m。
    · logs/e1_root.json：>50 m 的边只占锐边长度的 8.4%，却贡献 **E1 的 85.4%**
      （它们到真值锐边平均 41.4 m，而正常短边只有 0.9–1.8 m）。
⇒ 删掉这些面：任何 >thr 的边必然被它相邻的（≤2 个）面全部覆盖，两个面都会被删掉，
  所以清理后不会残留 >thr 的边。

只操作自己的网格文件，不读真值、不使用点云（除非显式开启 --bbox）。

用法:
    python src/clean_obj.py --indir data/work/final_v30 --outdir data/work/final_v31 \
        --max-edge 30 --jobs 6 --report logs/clean_v31.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = r"D:/LiDAR2026"


def clean_text(txt, thr, bbox=None, pad=0.0):
    V = []
    F = []
    other = []
    for line in txt.splitlines():
        if not line:
            continue
        k = line[:2]
        if k == "v ":
            t = line.split()
            V.append((float(t[1]), float(t[2]), float(t[3])))
            other.append(None)
        elif k == "f ":
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            if len(idx) == 3:
                F.append(idx)
            elif len(idx) == 4:          # 四边形 → 拆两个三角
                F.append([idx[0], idx[1], idx[2]])
                F.append([idx[0], idx[2], idx[3]])
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(V) == 0 or len(F) == 0:
        return txt, dict(nv0=len(V), nf0=len(F), nv=len(V), nf=len(F), nlong=0, nclip=0)
    e = np.maximum(
        np.maximum(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1),
                   np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1)),
        np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1))
    keep = e <= thr
    nlong = int((~keep).sum())
    nclip = 0
    if bbox is not None:
        lo = np.asarray(bbox[0], float) - pad
        hi = np.asarray(bbox[1], float) + pad
        inside = np.all((V >= lo) & (V <= hi), axis=1)
        fi = inside[F].all(axis=1)
        nclip = int((keep & ~fi).sum())
        keep = keep & fi
    if keep.all():
        return txt, dict(nv0=len(V), nf0=len(F), nv=len(V), nf=len(F),
                         nlong=nlong, nclip=nclip)
    F2 = F[keep]
    used = np.unique(F2)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    out = ["# cleaned by clean_obj.py (removed %d long-edge faces, %d out-of-bbox faces)\n"
           % (nlong, nclip)]
    out.extend("v %.6f %.6f %.6f\n" % (x, y, z) for x, y, z in V[used])
    out.extend("f %d %d %d\n" % (remap[a] + 1, remap[b] + 1, remap[c] + 1) for a, b, c in F2)
    return "".join(out), dict(nv0=len(V), nf0=len(F), nv=int(len(used)),
                             nf=int(len(F2)), nlong=nlong, nclip=nclip)


def _run(job):
    fn, indir, outdir, thr, bb = job
    try:
        src = os.path.join(indir, fn)
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        pad = 0.0
        bbxy = None
        if bb is not None:
            bbxy = (bb[0], bb[1])
            pad = bb[2]
        new, st = clean_text(txt, thr, bbxy, pad=pad)
        with open(os.path.join(outdir, fn), "w", encoding="utf-8") as f:
            f.write(new)
        st["file"] = fn
        st["s0"] = len(txt)
        st["s1"] = len(new)
        return st
    except Exception as ex:
        return {"file": fn, "err": str(ex)[:100]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-edge", type=float, default=30.0,
                    help="边长的绝对上限（米）。最终阈值 = max(max_edge, rel_frac*diag)")
    ap.add_argument("--rel-frac", type=float, default=0.0,
                    help="边长相对点云 diag 的上限比例（需配合 --bbox-from 得到 diag）")
    ap.add_argument("--bbox-pad-frac", type=float, default=0.10,
                    help="点云 bbox 的外扩比例，顶点越界的面一律删除")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--bbox-from", default=None,
                    help="可选：点云目录（.ply），用于取每个场景的 bbox / diag")
    ap.add_argument("--report", default="logs/clean_obj.json")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.indir) if f.endswith(".obj"))
    print("[clean_obj] %d 个文件  max_edge=%.1f rel_frac=%.2f pad_frac=%.2f bbox=%s" %
          (len(files), a.max_edge, a.rel_frac, a.bbox_pad_frac, a.bbox_from), flush=True)
    jobs = []
    nbb = 0
    for fn in files:
        bb = None
        thr = a.max_edge
        if a.bbox_from:
            pp = os.path.join(a.bbox_from, fn[:-4] + ".ply")
            if os.path.exists(pp):
                try:
                    import trimesh
                    v = np.asarray(trimesh.load(pp, process=False).vertices, float)
                    d = float(np.linalg.norm(v.max(0) - v.min(0)))
                    bb = (v.min(0).tolist(), v.max(0).tolist(), a.bbox_pad_frac * d)
                    thr = max(thr, a.rel_frac * d)
                    nbb += 1
                except Exception:
                    bb = None
        jobs.append((fn, a.indir, a.outdir, thr, bb))
    print("[clean_obj] 取得点云 bbox 的场景 %d / %d" % (nbb, len(files)), flush=True)
    t0 = time.time()
    rec = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(_run, jobs, chunksize=16)):
            rec.append(r)
            if (k + 1) % 500 == 0 or k + 1 == len(jobs):
                print("  %5d/%d  %.1f min" % (k + 1, len(jobs), (time.time() - t0) / 60), flush=True)
    json.dump(rec, open(os.path.join(ROOT, a.report), "w", encoding="utf-8"), ensure_ascii=False)
    ok = [r for r in rec if "err" not in r]
    f0 = sum(r["nf0"] for r in ok)
    f1 = sum(r["nf"] for r in ok)
    s0 = sum(r["s0"] for r in ok)
    s1 = sum(r["s1"] for r in ok)
    print("\n[完成] %d 文件  面数 %d → %d (删 %.2f%%)  字节 %.1f MB → %.1f MB (%.2f%%)  %.1f min" %
          (len(ok), f0, f1, 100 * (f0 - f1) / max(f0, 1), s0 / 1e6, s1 / 1e6,
           100 * s1 / max(s0, 1), (time.time() - t0) / 60))
    tot = sum(os.path.getsize(os.path.join(a.outdir, f))
              for f in os.listdir(a.outdir) if f.endswith(".obj"))
    print("  输出目录 %.1f MB" % (tot / 1e6))
    nz = [r for r in ok if r["nlong"] > 0]
    print("  有删面的文件 %d (%.1f%%)" % (len(nz), 100 * len(nz) / max(len(ok), 1)))
    for r in rec:
        if "err" in r:
            print("   ERR", r["file"], r["err"])
    return 0


def _run(job):
    fn, indir, outdir, thr, bb = job
    try:
        src = os.path.join(indir, fn)
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        pad = 0.0
        if bb is not None:
            bb = (bb[0], bb[1])
            pad = job[4][2] if isinstance(job[4], tuple) and len(job[4]) > 2 else 0.0
        new, st = clean_text(txt, thr, bb, pad=pad)
        with open(os.path.join(outdir, fn), "w", encoding="utf-8") as f:
            f.write(new)
        st["file"] = fn
        st["s0"] = len(txt)
        st["s1"] = len(new)
        return st
    except Exception as ex:
        return {"file": fn, "err": str(ex)[:100]}


if __name__ == "__main__":
    sys.exit(main())
