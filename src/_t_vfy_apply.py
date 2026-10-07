# -*- coding: utf-8 -*-
"""端到端复验：生产脚本写出的 OBJ（文本追加 DSM）是否真的复现了内存里测到的增益。

内存实验（`_t_dsm_grid.py`）用 trimesh.util.concatenate 合并网格后打分；
生产脚本（`_t_dsm_apply.py`）改为**文本追加**，且官方加载器默认会 merge 顶点。
两者必须给出一致的 Δ，否则线上行为会与本地预测不符。

用法:
    python src/_t_vfy_apply.py --raw data/work/bw600/wC --new data/work/_vfy_v21 \
        --gt data/work/bw600_gt --ids logs/vfy_ids.txt --jobs 2 \
        --ns 15000 --ne 30000 --out logs/vfy_apply.json
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

if 'faiss' not in sys.modules:
    _s = types.ModuleType('faiss')
    _s.IndexFlatL2 = object
    _s.StandardGpuResources = object
    _s.index_cpu_to_gpu = lambda *a, **k: None
    sys.modules['faiss'] = _s
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"D:/LiDAR2026"


def L(p):
    """与官方 load_mesh 完全一致：默认参数"""
    m = trimesh.load(p)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def prep_gt(gt, ns, ne, seed):
    import official_evaluate as OE
    np.random.seed(seed)
    gs = np.asarray(trimesh.sample.sample_surface(gt, ns)[0], float)
    geg = OE.extract_sharp_edges(gt, 30.0)
    gep = (np.zeros((0, 3)) if len(geg) == 0 else
           np.asarray(OE.sample_points_on_edges_global(
               np.asarray(gt.vertices, float), geg, ne), float))
    return gs, gep, (cKDTree(gep) if len(gep) else None)


def score(m, gt, gtpre, ns, ne, seed):
    import official_evaluate as OE
    gs, gep, tgep = gtpre
    if m is None or len(np.asarray(m.faces)) == 0:
        return None
    try:
        np.random.seed(seed)
        ps = np.asarray(trimesh.sample.sample_surface(m, ns)[0], float)
        cp = np.asarray(m.nearest.on_surface(gs)[0], float)
        cg = np.asarray(gt.nearest.on_surface(ps)[0], float)
        d1 = float(np.linalg.norm(ps - cg, axis=1).mean())
        d2 = float(np.linalg.norm(gs - cp, axis=1).mean())
        ep = OE.extract_sharp_edges(m, 30.0)
        if len(ep) == 0 or tgep is None:
            return None
        np.random.seed(seed)
        pep = np.asarray(OE.sample_points_on_edges_global(
            np.asarray(m.vertices, float), ep, ne), float)
        e1 = float(tgep.query(pep)[0].mean())
        e2 = float(cKDTree(pep).query(gep)[0].mean())
        return {"CD": d1 + d2, "CD1": d1, "CD2": d2, "ECD": e1 + e2,
                "E1": e1, "E2": e2, "FINAL": 0.6 * (d1 + d2) + 0.4 * (e1 + e2),
                "F": int(len(np.asarray(m.faces))),
                "V": int(len(np.asarray(m.vertices))), "nE": int(len(ep))}
    except Exception:
        return None


def one(job):
    sid, rawd, newd, gtd, ns, ne, seed = job
    try:
        gp = os.path.join(gtd, sid + "_gt.obj")
        rp = os.path.join(rawd, sid + ".obj")
        np_ = os.path.join(newd, sid + ".obj")
        for q in (gp, rp, np_):
            if not os.path.exists(q):
                return {"id": sid, "err": "missing %s" % os.path.basename(q)}
        gt = L(gp)
        pre = prep_gt(gt, ns, ne, seed)
        a = score(L(rp), gt, pre, ns, ne, seed)
        b = score(L(np_), gt, pre, ns, ne, seed)
        if a is None or b is None:
            return {"id": sid, "err": "score-fail"}
        return {"id": sid, "raw": a, "new": b,
                "gtF": int(len(np.asarray(gt.faces)))}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--ns", type=int, default=15000)
    ap.add_argument("--ne", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()]
    print("端到端复验 n=%d" % len(ids), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.raw, a.new, a.gt, a.ns, a.ne, a.seed)
                                           for i in ids], chunksize=1)):
            rows.append(r)
            print("  %d/%d %s" % (k + 1, len(ids), r.get("err") or "ok"), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)
    clean = [r for r in rows if "err" not in r and r["raw"]["CD1"] < 50]
    print("\n可用 %d / %d" % (len(clean), len(rows)))
    if not clean:
        return 1
    print("%-8s %8s %8s %8s %8s %8s %8s %8s" %
          ("id", "Fraw", "Fnew", "dCD", "dECD", "rawFNL", "newFNL", "dFNL"))
    d = []
    for r in sorted(clean, key=lambda x: x["raw"]["FINAL"], reverse=True):
        df = r["new"]["FINAL"] - r["raw"]["FINAL"]
        d.append(df)
        print("%-8s %8d %8d %+8.3f %+8.3f %8.3f %8.3f %+8.3f" %
              (r["id"], r["raw"]["F"], r["new"]["F"],
               r["new"]["CD"] - r["raw"]["CD"],
               r["new"]["ECD"] - r["raw"]["ECD"],
               r["raw"]["FINAL"], r["new"]["FINAL"], df))
    d = np.array(d)
    print("\nΔFINAL: 均值 %+.4f  中位 %+.4f  %d 变好 / %d 变差" %
          (d.mean(), np.median(d), int((d < 0).sum()), int((d > 0).sum())))
    print("ΔECD  均值 %+.4f   ΔCD 均值 %+.4f" %
          (np.mean([r["new"]["ECD"] - r["raw"]["ECD"] for r in clean]),
           np.mean([r["new"]["CD"] - r["raw"]["CD"] for r in clean])))
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
