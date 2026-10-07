# -*- coding: utf-8 -*-
"""★ 垃圾长边清理算子 —— 本赛程发现的最大杠杆。

根因（logs/e1_root.json，官方口径，10 个全集随机场景）：
    E1 = 6.8896，其中**边长 >50 m 的 13 条边（只占长度 8.4%）贡献 85.4%**！
    它们到真值锐边的平均距离 41.4 m（正常短边只有 0.9–1.8 m）。
    最恶劣：场景 16777 有 5 条 167 m 长的边，从 z=+161.2 直插 z=−5.7  （离真值锐边 73 m）。
规模（直接扫文本统计）：
    base(wC,598)      >50m 边 13927 条(23.3/场景)，>100m 4603 条，最长 764,092 m
    **已提交 final_v28(4000)**  >50m 边 35206 条(8.8/场景)，>100m 9093 条，最长 733 m
    ⇒ 每场景超长边占锐边长度 0.0384（提交件）/ 0.141（base）
⇒ 这是**我们自己重建管线产生的垃圾三角形**（把远距离顶点粘成巨型薄片），
  与真值无关，修复它完全合规、也不涉及任何禁用手段。

本脚本对照若干"只用点云定标"的清理规则，测量 ECD/CD/FINAL 变化：
    none   不做处理（对照）
    a10/a20/a30/a50   删除最长边 >20/…/50 m 的面
    rel    删除最长边 > max(20 m, 0.25·diag_cloud) 的面
    bbox   删除任一顶点落在点云 bbox 外扩 0.05·diag 之外的面
    rel+bb bbox 之后再套 rel

用法:
    python src/_t_clean.py --rec data/work/bw600/wC --gt data/work/bw600_gt \
        --ply data/work/bw600/ply --ids logs/rand16.txt --n 10 --jobs 1 \
        --ops none,a10,a20,a30,a50,rel,bbox,relbb --out logs/clean_rand16.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _t_surf_only import L, cat, load_ply_pts, resolve, score       # noqa: E402
from _t_mads import shell                                            # noqa: E402
import official_evaluate as OE                                       # noqa: E402

ROOT = r"D:/LiDAR2026"
ALL_OPS = ["none", "t20", "a20", "a30", "a50", "rel", "bbox", "relbb",
           "ct0.5", "ct1", "ct2", "ct4", "ct8", "ct15"]


def edge_max(m):
    F = np.asarray(m.faces, np.int64)
    V = np.asarray(m.vertices, float)
    if len(F) == 0:
        return np.zeros(0), V
    e0 = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    e1 = np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1)
    e2 = np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1)
    return np.maximum(np.maximum(e0, e1), e2), V


def subset(m, keep):
    F = np.asarray(m.faces, np.int64)
    V = np.asarray(m.vertices, float)
    F = F[keep]
    if len(F) < 4:
        return None
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(vertices=V[used], faces=remap[F], process=False)


def clean(base, cpts, ops):
    """返回 {tag: mesh}，按 ops 指定的规则依次作用（可叠加，如 relbb）。

    新增「质心离群距离」规则族 ct<T>：
        删除 面质心 到最近点云点 距离 > T 的面。
    动机：长度规则(t/a)会误删大场景里**合法的大面片**（大地面/大屋顶），
          砸掉 CD2 覆盖；而真正该删的是"离点云很远"的浮空巨型薄片。
          质心规则直接度量"这块面有没有点云支撑"，比边长精准得多。
    """
    diag = float(np.linalg.norm(cpts.max(0) - cpts.min(0)))
    lo, hi = cpts.min(0), cpts.max(0)
    pad = 0.05 * diag
    out = {}
    fm, V = edge_max(base)
    inside = np.ones(len(V), bool)
    if len(V):
        inside = np.all((V >= lo - pad) & (V <= hi + pad), axis=1)
    Fv = np.asarray(base.faces, np.int64)
    face_inside = inside[Fv].all(axis=1) if len(Fv) else np.zeros(0, bool)
    # 面质心 -> 最近点云点 距离
    ctree = cKDTree(cpts)
    cen_d = (ctree.query(V[Fv].mean(1))[0] if len(Fv) else np.zeros(0))

    def mk(ratio=None, abs_thr=None, use_bbox=False):
        if len(fm) == 0:
            return None
        thr = np.inf
        if abs_thr is not None:
            thr = abs_thr
        if ratio is not None:
            thr = min(thr, max(20.0, ratio * diag))
        keep = fm <= thr
        if use_bbox:
            keep &= face_inside
        return subset(base, keep)

    for op in ops:
        if op == "none":
            out["none"] = base
        elif op.startswith("a") and op[1:].isdigit():
            out[op] = mk(abs_thr=float(op[1:]))
        elif op == "t20":
            out["t20"] = mk(abs_thr=20.0)
        elif op.startswith("ct"):
            out[op] = subset(base, cen_d <= float(op[2:]))
        elif op == "rel":
            out["rel"] = mk(ratio=0.25)
        elif op == "bbox":
            out["bbox"] = subset(base, face_inside)
        elif op == "relbb":
            out["relbb"] = mk(ratio=0.25, use_bbox=True)
    return out


def one(job):
    sid, rec, gtd, plyd, ops = job
    try:
        rp = resolve(rec, sid)
        gtp = os.path.join(gtd, sid + "_gt.obj")
        pp = os.path.join(plyd, sid + ".ply")
        if rp is None or not os.path.exists(gtp) or not os.path.exists(pp):
            return {"id": sid, "err": "missing"}
        gt = L(gtp)
        base = L(rp)
        cpts = load_ply_pts(pp)
        sd = abs(hash(sid)) % (2 ** 31)
        np.random.seed(sd % 313)
        gsp = np.asarray(trimesh.sample.sample_surface(gt, 300000)[0], float)
        gs_tree = cKDTree(gsp)
        geg = OE.extract_sharp_edges(gt, 30.0)
        np.random.seed(sd % 317)
        gep = np.asarray(OE.sample_points_on_edges_global(
            np.asarray(gt.vertices, float), geg, 100000), float)
        gep_tree = cKDTree(gep)
        sub = gsp[np.random.default_rng(7).choice(len(gsp), 60000, replace=False)]

        shells = [shell(cpts, 3.0, 9.0, ax, seed=sd + 10 * ax) for ax in (2, 0, 1)]
        res = {}
        for tag, bm in clean(base, cpts, ops).items():
            if bm is None:
                res[tag] = None
                continue
            m = cat(bm, *shells)
            if m is None or len(np.asarray(m.faces)) == 0:
                res[tag] = None
                continue
            r = score(m, gs_tree, gep_tree, gep)
            if r is None:
                res[tag] = None
                continue
            ps = np.asarray(trimesh.sample.sample_surface(m, 60000)[0], float)
            r["CD2"] = float(cKDTree(ps).query(sub)[0].mean())
            r["CDsum"] = r["CD1"] + r["CD2"]
            r["FINAL"] = 0.6 * r["CDsum"] + 0.4 * r["ECD"]
            fm2, _ = edge_max(m)
            r["maxE"] = float(fm2.max()) if len(fm2) else 0.0
            r["nLong"] = int((fm2 > 50).sum()) if len(fm2) else 0
            r["F"] = int(len(np.asarray(m.faces)))
            res[tag] = r
        return {"id": sid, "diag": float(np.linalg.norm(cpts.max(0) - cpts.min(0))),
                "m_diag": float(np.linalg.norm(np.asarray(base.vertices, float).max(0)
                                               - np.asarray(base.vertices, float).min(0))),
                "gtF": int(len(np.asarray(gt.faces))), "res": res}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:110]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--ply", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--ops", default="none,a10,a20,a30,a50,rel,bbox,relbb")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ops = [s for s in a.ops.split(",") if s]
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()][:a.n]
    rows = []
    jl = os.path.join(ROOT, a.out.replace(".json", ".jsonl"))
    open(jl, "w", encoding="utf-8").close()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.rec, a.gt, a.ply, ops) for i in ids], chunksize=1)):
            rows.append(r)
            with open(jl, "a", encoding="utf-8") as fj:
                fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("  %d/%d %s" % (k + 1, len(ids), r.get("err", "ok")), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), ensure_ascii=False)
    cl = [r for r in rows if "err" not in r and r.get("diag", 0) >= a.gate]
    print("\n有效 %d / %d  diag 中位 %.1f" %
          (len(cl), len(rows), float(np.median([r["diag"] for r in cl])) if cl else -1))
    if not cl:
        for r in rows[:6]:
            print("  ", r)
        return 1
    order = [o for o in ALL_OPS if any(r["res"].get(o) for r in cl)]
    ref = None
    print("\n%-7s %8s %8s %8s %8s %8s %8s %9s %9s %7s %8s" %
          ("规则", "CD1", "CD2", "CD", "E1", "E2", "ECD", "FINAL", "dFINAL", "最长边", ">50m边"))
    for t in order:
        gg = [r["res"][t] for r in cl if r["res"].get(t)]
        if not gg:
            continue
        mm = lambda k: float(np.mean([x[k] for x in gg]))
        if t == "none":
            ref = mm("FINAL")
        print("%-7s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %9.4f %+9.4f %7.1f %8.1f" %
              (t, mm("CD1"), mm("CD2"), mm("CDsum"), mm("E1"), mm("E2"), mm("ECD"),
               mm("FINAL"), (mm("FINAL") - ref) if ref is not None else 0.0,
               mm("maxE"), mm("nLong")))
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
