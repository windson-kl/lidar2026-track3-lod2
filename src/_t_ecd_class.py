# -*- coding: utf-8 -*-
"""ECD 质量分解：把我们的锐边按"成因"分类，定位 ECD 误差的质量归属。

背景（本轮结论）：
    FINAL = 0.6*CD + 0.4*ECD，线上我们 CD 3.4018（优于 rank4/rank5），
    但 ECD 5.9081 是前 8 名最差 -> ECD 是唯一瓶颈（ECD 降 6.5% 即达 4.25）。
    本地测得：我们锐边数 209 vs GT 3172，锐边总长 4249m vs GT 8776m，
    且我们的**边界边占比 66.8% vs GT 10.2%**。
    --> 假说：我们的 ECD 误差主要来自"开放边界"，而不是真实屋脊折边。

extract_sharp_edges 的三条规则：
    (1) 只被 1 个面共享 -> 边界边 -> 锐
    (2) 被 >2 个面共享 -> 非流形 -> 锐
    (3) 恰好 2 个面且 |cos(n1·n2)| < cos(30°) -> 折边 -> 锐
本脚本按这三类分离，量化：
    * 每类锐边的**长度占比** = 该类在 E1 采样中的权重
    * E1_self = 该类采样点到 GT 锐边点的平均距离 => 可加分解 E1 = Σ w_c * E1_c
    * E2_only = 反事实："如果我们只有该类锐边"，GT 锐边点到它最近距离的均值
    * 每类到 GT 锐边的距离分位数（中位/90%）——区分"错得离谱"还是"略有偏差"

用法:
    python src/_t_ecd_class.py --rec data/work/bw600/wC --gt data/work/bw600_gt \
        --ids logs/dsm_ids.txt --jobs 3 --ns 15000 --ne 100000 --out logs/ecd_class.json
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
CLASSES = ("bnd", "nm", "ang")


def L(p):
    """与官方 load_mesh 一致：默认参数（会 merge 顶点）"""
    m = trimesh.load(p)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def resolve(rec, sid):
    for suf in ("", "_ReconstructedModel"):
        p = os.path.join(rec, sid + suf + ".obj")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def classify_sharp(mesh, angle_threshold_deg=30.0):
    """复刻 extract_sharp_edges 并返回 (edges, cls_idx)；cls_idx: 0=bnd 1=nm 2=ang"""
    faces = np.asarray(mesh.faces)
    fn = np.asarray(mesh.face_normals, float)
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    mapping = {}
    for fi, f in enumerate(faces):
        for e in (tuple(sorted((f[0], f[1]))), tuple(sorted((f[1], f[2]))),
                  tuple(sorted((f[2], f[0])))):
            mapping.setdefault(e, []).append(fi)
    edges, cls = [], []
    cos_thresh = np.cos(np.deg2rad(angle_threshold_deg))
    for e, fl in mapping.items():
        if len(fl) == 1:
            edges.append(e); cls.append(0); continue
        if len(fl) > 2:
            edges.append(e); cls.append(1); continue
        n1, n2 = fn[fl[0]], fn[fl[1]]
        if abs(float(np.clip(np.dot(n1, n2), -1.0, 1.0))) < cos_thresh:
            edges.append(e); cls.append(2)
    if not edges:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    return np.asarray(edges, np.int64), np.asarray(cls, np.int64)


def class_stats(vertices, edges, cls):
    p1, p2 = vertices[edges[:, 0]], vertices[edges[:, 1]]
    ln = np.linalg.norm(p2 - p1, axis=1)
    out = {}
    tot = ln.sum() + 1e-12
    for ci, name in enumerate(CLASSES):
        m = cls == ci
        out[name] = {"nE": int(m.sum()),
                     "len": float(ln[m].sum()),
                     "w": float(ln[m].sum() / tot)}
    out["tot_len"] = float(ln.sum())
    out["nE_tot"] = int(len(edges))
    return out, ln


def sample_class(vertices, edges, cls, ln, ci, n, rng):
    """只从第 ci 类锐边采样 n 个点（按长度加权）"""
    m = cls == ci
    if m.sum() == 0:
        return np.zeros((0, 3))
    e = edges[m]
    l = ln[m]
    probs = l / (l.sum() + 1e-12)
    idx = rng.choice(len(e), size=n, p=probs)
    p1, p2 = vertices[e[idx, 0]], vertices[e[idx, 1]]
    t = rng.random((n, 1))
    return (1 - t) * p1 + t * p2


def one(job):
    sid, rec, gtd, ns, ne, seed = job
    try:
        gtp = os.path.join(gtd, sid + "_gt.obj")
        rp = resolve(rec, sid)
        if not os.path.exists(gtp) or rp is None:
            return {"id": sid, "err": "missing"}
        import official_evaluate as OE
        gt = L(gtp)
        pred = L(rp)
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        # --- CD ---
        ps = np.asarray(trimesh.sample.sample_surface(pred, ns)[0], float)
        gs = np.asarray(trimesh.sample.sample_surface(gt, ns)[0], float)
        cg = np.asarray(gt.nearest.on_surface(ps)[0], float)
        cp = np.asarray(pred.nearest.on_surface(gs)[0], float)
        d1 = float(np.linalg.norm(ps - cg, axis=1).mean())
        d2 = float(np.linalg.norm(gs - cp, axis=1).mean())

        ge = OE.extract_sharp_edges(gt, 30.0)
        gv = np.asarray(gt.vertices, float)
        if len(ge) == 0:
            return {"id": sid, "err": "no-gt-edge"}
        np.random.seed(seed)
        gep = np.asarray(OE.sample_points_on_edges_global(gv, ge, ne), float)
        tge = cKDTree(gep)

        pe = OE.extract_sharp_edges(pred, 30.0)
        if len(pe) == 0:
            return {"id": sid, "err": "no-pred-edge"}
        pe2, pc = classify_sharp(pred, 30.0)
        if len(pe2) != len(pe):
            return {"id": sid, "err": "cls-mismatch %d/%d" % (len(pe2), len(pe))}
        pv = np.asarray(pred.vertices, float)
        st, ln = class_stats(pv, pe, pc)

        np.random.seed(seed)
        pep = np.asarray(OE.sample_points_on_edges_global(pv, pe, ne), float)
        e1 = float(tge.query(pep)[0].mean())
        e2 = float(cKDTree(pep).query(gep)[0].mean())

        per = {}
        for ci, name in enumerate(CLASSES):
            q = sample_class(pv, pe, pc, ln, ci, ne, rng)
            if len(q) == 0:
                per[name] = None
                continue
            dd = tge.query(q)[0]
            per[name] = {"E1_self": float(dd.mean()),
                         "E1_med": float(np.median(dd)),
                         "E1_p90": float(np.percentile(dd, 90)),
                         "E2_only": float(cKDTree(q).query(gep)[0].mean()),
                         "contr_E1": float(st[name]["w"] * dd.mean())}
        gt2, gtc = classify_sharp(gt, 30.0)
        gst, _ = class_stats(gv, gt2, gtc)

        return {"id": sid, "gtF": int(len(np.asarray(gt.faces))),
                "F": int(len(np.asarray(pred.faces))),
                "V": int(len(pv)),
                "CD": d1 + d2, "CD1": d1, "CD2": d2, "ECD": e1 + e2,
                "E1": e1, "E2": e2,
                "FINAL": 0.6 * (d1 + d2) + 0.4 * (e1 + e2),
                "pred": st, "per": per, "gtcls": gst}
    except Exception as ex:
        return {"id": sid, "err": str(ex)[:100]}


def clsmean(clean, name, field):
    v = [r["per"][name][field] for r in clean
         if r["per"].get(name) and r["per"][name].get(field) is not None]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--ns", type=int, default=15000)
    ap.add_argument("--ne", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8") if l.strip()]
    print("ECD 分类分解 n=%d  ne=%d" % (len(ids), a.ne), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.rec, a.gt, a.ns, a.ne, a.seed)
                                           for i in ids], chunksize=1)):
            rows.append(r)
            print("  %d/%d %s" % (k + 1, len(ids), r.get("err") or "ok"), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)

    clean = [r for r in rows if "err" not in r and r["CD1"] < 50]
    print("\n可用 %d / %d" % (len(clean), len(rows)))
    for r in rows:
        if "err" in r:
            print("  err %s: %s" % (r["id"], r["err"]))
    if not clean:
        return 1
    print("base: CD %.4f (%.4f+%.4f)  ECD %.4f (E1 %.4f + E2 %.4f)  FINAL %.4f" %
          (np.mean([r["CD"] for r in clean]), np.mean([r["CD1"] for r in clean]),
           np.mean([r["CD2"] for r in clean]), np.mean([r["ECD"] for r in clean]),
           np.mean([r["E1"] for r in clean]), np.mean([r["E2"] for r in clean]),
           np.mean([r["FINAL"] for r in clean])))
    print("F %.1f  V %.1f  gtF %.1f" % (np.mean([r["F"] for r in clean]),
                                        np.mean([r["V"] for r in clean]),
                                        np.mean([r["gtF"] for r in clean])))

    print("\n=== 我们锐边的类别构成 ===")
    print("%-5s %9s %12s %8s %10s %10s" %
          ("cls", "nE均值", "长度均值m", "长度占比", "E1_self", "E1_p90"))
    for name in CLASSES:
        print("%-5s %9.0f %12.0f %7.1f%% %10.3f %10.3f" %
              (name, np.mean([r["pred"][name]["nE"] for r in clean]),
               np.mean([r["pred"][name]["len"] for r in clean]),
               100 * np.mean([r["pred"][name]["w"] for r in clean]),
               clsmean(clean, name, "E1_self"), clsmean(clean, name, "E1_p90")))
    print("锐边总数 %.0f   总长 %.0f m" %
          (np.mean([r["pred"]["nE_tot"] for r in clean]),
           np.mean([r["pred"]["tot_len"] for r in clean])))

    print("\n=== GT 锐边的类别构成（参照） ===")
    print("%-5s %9s %12s %8s" % ("cls", "nE均值", "长度均值m", "长度占比"))
    for name in CLASSES:
        print("%-5s %9.0f %12.0f %7.1f%%" %
              (name, np.mean([r["gtcls"][name]["nE"] for r in clean]),
               np.mean([r["gtcls"][name]["len"] for r in clean]),
               100 * np.mean([r["gtcls"][name]["w"] for r in clean])))
    print("GT 锐边总数 %.0f   总长 %.0f m" %
          (np.mean([r["gtcls"]["nE_tot"] for r in clean]),
           np.mean([r["gtcls"]["tot_len"] for r in clean])))

    e1b = np.mean([r["E1"] for r in clean])
    print("\n=== E1 的加性分解（= Σ 长度占比 × E1_self） ===")
    print("官方 E1 基线 %.4f" % e1b)
    tot = 0.0
    for name in CLASSES:
        w = np.mean([r["pred"][name]["w"] for r in clean])
        c = clsmean(clean, name, "contr_E1")
        if not np.isnan(c):
            print("  %-4s 权重 %5.1f%%  E1_self %7.3f  贡献 %7.4f  占 E1 %5.1f%%" %
                  (name, 100 * w, clsmean(clean, name, "E1_self"), c, 100 * c / e1b))
            tot += c
    print("  合计 %.4f" % tot)

    e2b = np.mean([r["E2"] for r in clean])
    print("\n=== 反事实 E2：如果锐边只剩某一类 ===")
    print("官方 E2 基线 %.4f" % e2b)
    for name in CLASSES:
        v = clsmean(clean, name, "E2_only")
        if not np.isnan(v):
            print("  只有 %-4s -> E2 %7.3f  (%+7.3f)" % (name, v, v - e2b))

    print("\n=== 单样本明细（ECD 降序） ===")
    print("%-7s %7s %6s %8s %7s %7s %8s %8s %7s %7s" %
          ("id", "gtF", "F", "ECD", "E1", "E2", "E1bnd", "E1ang", "wbnd", "wang"))
    for r in sorted(clean, key=lambda x: -x["ECD"]):
        pb, pa = r["per"].get("bnd"), r["per"].get("ang")
        print("%-7s %7d %6d %8.3f %7.3f %7.3f %8.3f %8.3f %6.1f%% %6.1f%%" %
              (r["id"], r["gtF"], r["F"], r["ECD"], r["E1"], r["E2"],
               pb["E1_self"] if pb else np.nan, pa["E1_self"] if pa else np.nan,
               100 * r["pred"]["bnd"]["w"], 100 * r["pred"]["ang"]["w"]))
    print("\n写出 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
