# -*- coding: utf-8 -*-
"""文本级 OBJ 清洗器（第二代）：按「面质心到最近点云点的距离」删面。

为什么换规则（2026-10-07 在真实提交上实测，见 logs/audit_test200.txt）：
    在 200 个**真实测试场景**上用测试点云计算双向代理（CD1=我方面采样->最近点云点，
    CD2=点云点->最近**网格曲面**）：
        长度规则 t20  删 5.3% 面 / 28.0% 面积 ⇒ dCD1 -0.770, dCD2 +0.541, dFINAL **+0.075**
        质心规则 ct1  删 18.4% 面 / 34.1% 面积 ⇒ dCD1 -0.957, dCD2 +0.015, dFINAL **-0.565**
        质心规则 ct4  删 4.1% 面 / 11.3% 面积 ⇒ dCD1 -0.542, dCD2 +0.000, dFINAL **-0.325**
    原因：长度规则会误删大场景里**合法的大面片**（大地面/大屋顶），砸掉 CD2 覆盖；
          质心规则直接度量"这块面有没有点云支撑"，删掉的东西本来就没人用它，
          所以 CD2 几乎不动。
    真实提交 final_v28 的 d_c 分箱（120 场景实测，logs/ct_diag.json）：
        [0,0.25) 面积 31.8% | [0.25,0.5) 19.2% | [0.5,1) 12.2% | [1,2) 10.7%
        [2,4) 13.0% | [4,8) 8.1% | [8,16) 3.1% | [16,32) 1.6% | >=32 0.3%
        d_c 中位 0.263 m，p99 3.43 m ⇒ 我方网格整体**紧贴点云**，垃圾只是尾部。

重要：本清洗器应施加在**基座**上（如 final_v21），之后再用 _t_dsm_apply.py /
_t_mads_apply.py 追加 DSM 壳；不要直接清洗已含 DSM 壳的成品，否则可能误删壳体。

用法:
    python src/clean_ct.py --indir data/work/final_v21 --outdir data/work/base_ct1 \
        --pcdir data/work/ply --dc 1.0 --jobs 6 --report logs/clean_ct1.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = r"D:/LiDAR2026"


def parse_obj(txt):
    V, F = [], []
    for line in txt.splitlines():
        if not line:
            continue
        k = line[:2]
        if k == "v ":
            t = line.split()
            V.append((float(t[1]), float(t[2]), float(t[3])))
        elif k == "f ":
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            if len(idx) == 3:
                F.append(idx)
            elif len(idx) == 4:
                F.append([idx[0], idx[1], idx[2]])
                F.append([idx[0], idx[2], idx[3]])
    return np.asarray(V, float), np.asarray(F, np.int64)


def emit(V, F, used, comment):
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    out = [comment]
    out.extend("v %.6f %.6f %.6f\n" % (x, y, z) for x, y, z in V[used])
    out.extend("f %d %d %d\n" % (remap[a] + 1, remap[b] + 1, remap[c] + 1)
               for a, b, c in F)
    return "".join(out)


def clean_text(txt, tree, dc_thr, dv_thr=None):
    V, F = parse_obj(txt)
    st = dict(nv0=len(V), nf0=len(F), nv=len(V), nf=len(F), n_dc=0, n_dv=0)
    if len(V) == 0 or len(F) == 0:
        return txt, st
    dc = tree.query(V[F].mean(1))[0]
    keep = dc <= dc_thr
    n_dc = int((~keep).sum())
    n_dv = 0
    if dv_thr is not None:
        dv = np.stack([tree.query(V[F[:, k]])[0] for k in range(3)], 1).max(1)
        bad = dv > dv_thr
        n_dv = int((keep & bad).sum())
        keep = keep & ~bad
    st.update(n_dc=n_dc, n_dv=n_dv, nv=int(len(np.unique(F[keep]))) if keep.any() else 0,
              nf=int(keep.sum()))
    if keep.all():
        return txt, st
    used = np.unique(F[keep])
    return emit(V, F[keep], used,
                "# cleaned by clean_ct.py (dc>%.3g: %d faces, dv>%.3g: %d faces)\n"
                % (dc_thr, n_dc, dv_thr if dv_thr else -1, n_dv)), st


_CACHE = {}


def load_pts(p):
    if p in _CACHE:
        return _CACHE[p]
    import trimesh
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    v = np.asarray(m.vertices, float)
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[p] = v
    return v


def _run(job):
    fn, indir, outdir, pcdir, dc, dv = job
    try:
        from scipy.spatial import cKDTree
        with open(os.path.join(indir, fn), "r", encoding="utf-8",
                  errors="ignore") as f:
            txt = f.read()
        pts = load_pts(os.path.join(pcdir, fn[:-4] + ".ply"))
        if len(pts) < 8:
            new, st = txt, dict(nv0=0, nf0=0, nv=0, nf=0, n_dc=0, n_dv=0)
        else:
            new, st = clean_text(txt, cKDTree(pts), dc, dv)
        with open(os.path.join(outdir, fn), "w", encoding="utf-8") as f:
            f.write(new)
        st.update(file=fn, s0=len(txt), s1=len(new))
        return st
    except Exception as ex:                                          # noqa: BLE001
        return {"file": fn, "err": "%s: %s" % (type(ex).__name__, str(ex)[:90])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pcdir", required=True)
    ap.add_argument("--dc", type=float, default=1.0,
                    help="面质心到最近点云点的距离上限（米）")
    ap.add_argument("--dv", type=float, default=None,
                    help="可选：三顶点到点云距离上限（米），更保守")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="logs/clean_ct.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.indir) if f.endswith(".obj"))
    if a.limit:
        files = files[:a.limit]
    print("[clean_ct] %d 文件  dc=%.3g  dv=%s  pc=%s" %
          (len(files), a.dc, a.dv, a.pcdir), flush=True)
    jobs = [(f, a.indir, a.outdir, a.pcdir, a.dc, a.dv) for f in files]
    t0 = time.time()
    rec = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(_run, jobs, chunksize=8)):
            rec.append(r)
            if (k + 1) % 1000 == 0 or k + 1 == len(jobs):
                print("  %5d/%d  %.1f min" % (k + 1, len(jobs), (time.time() - t0) / 60),
                      flush=True)
    json.dump(rec, open(os.path.join(ROOT, a.report), "w", encoding="utf-8"),
              ensure_ascii=False)
    ok = [r for r in rec if "err" not in r]
    f0 = sum(r["nf0"] for r in ok); f1 = sum(r["nf"] for r in ok)
    s0 = sum(r["s0"] for r in ok); s1 = sum(r["s1"] for r in ok)
    ndc = sum(r["n_dc"] for r in ok); ndv = sum(r.get("n_dv", 0) for r in ok)
    print("\n[完成] %d 文件  %.1f min" % (len(ok), (time.time() - t0) / 60))
    print("  面数 %d → %d (删 %.2f%%)，其中 d_c 超限 %d、d_v 超限 %d"
          % (f0, f1, 100 * (f0 - f1) / max(f0, 1), ndc, ndv))
    print("  字节 %.1f MB → %.1f MB (%.2f%%)" % (s0 / 1e6, s1 / 1e6, 100 * s1 / max(s0, 1)))
    nz = [r for r in ok if r["n_dc"] + r.get("n_dv", 0) > 0]
    print("  有删面的文件 %d (%.1f%%)" % (len(nz), 100 * len(nz) / max(len(ok), 1)))
    for r in rec[:5]:
        if "err" in r:
            print("   ERR", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
