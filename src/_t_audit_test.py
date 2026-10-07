# -*- coding: utf-8 -*-
"""★★ 在**真实测试集**上审计裁剪规则（无真值，双向代理）。

纠错背景：
    本地 bw600 参考的裁剪 A/B 给出 ΔFINAL ≈ -4.0，但那几乎全来自场景 16777
    （本地参考重建 bbox 炸到 764 km）。真实提交 data/work/final_v28 的
    网格/点云 diag 比上限只有 ~1.14 ⇒ 那个巨大收益是本地假象。

方法（无真值、作用在提交本体上）：
    官方 CD = 0.6 权重，且 = mean(模型面采样点 -> 真值采样点) + mean(真值采样点 -> 模型面采样点)
    已知"真值曲面贴合点云"（pc -> gt 仅 0.1222 m，中位 0.1155）
    ⇒ 用**测试点云**当靶子，双向都算：
        m2p = 网格面积均匀采样点 -> 最近点云点      （对应 CD1）
        p2m = 点云点 -> 最近网格采样点              （对应 CD2）
    两个方向都随裁剪变化，于是"删多了砸覆盖"能立刻暴露。

    裁剪规则与 clean_obj.py 完全一致（内存施加，不需要加载 v31/v32）：
        keep = (面最大边 <= max(thr, rel*pdiag)) 且 (三个顶点都在 点云bbox±pad 内)

用法:
    python src/_t_audit_test.py --meshd data/work/final_v28 --plyd data/work/ply \
        --n 300 --jobs 6 --ns 150000 --out logs/audit_test.json
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
import official_evaluate as OE                                    # noqa: E402

ROOT = r"D:/LiDAR2026"

# 规则族。命名约定：
#   none            不裁剪
#   t<N>            面最大边 <= N                       （长度规则，旧 clean_obj）
#   ct<N>           面质心到点云距离 <= N               （离群距离规则，新）
#   v<N>            面三个顶点到点云距离的最大值 <= N
#   t<N>c<M>        max(边<=N, 质心<=M) 都满足才留（交集，更保守）
#   rel<R>          max(边<=R*pdiag)
#   bb              顶点在点云 bbox±5% 内
CFG = {
    "none":  ("none",),
    "t10":   ("t", 10.0),
    "t15":   ("t", 15.0),
    "t20":   ("t", 20.0),
    "t25":   ("t", 25.0),
    "t30":   ("t", 30.0),
    "t50":   ("t", 50.0),
    "ct0.5": ("ct", 0.5),
    "ct1":   ("ct", 1.0),
    "ct2":   ("ct", 2.0),
    "ct4":   ("ct", 4.0),
    "ct8":   ("ct", 8.0),
    "ct15":  ("ct", 15.0),
    "t20c4": ("tc", 20.0, 4.0),
    "t20c8": ("tc", 20.0, 8.0),
    "t50c8": ("tc", 50.0, 8.0),
    "rel30": ("rel", 0.30),
    "bb5":   ("bb", 0.05),
    "bb10":  ("bb", 0.10),
    "bb20":  ("bb", 0.20),
    "bb40":  ("bb", 0.40),
    "bb10c4": ("bbc", 0.10, 4.0),
    "bb20c8": ("bbc", 0.20, 8.0),
}


def keep_mask(name, V, F, e, plo, phi, pdiag, cen_d):
    kind = CFG[name]
    if kind[0] == "none":
        return np.ones(len(F), bool)
    if kind[0] == "t":
        return e <= kind[1]
    if kind[0] == "ct":
        return cen_d <= kind[1]
    if kind[0] == "tc":            # 交集：两条都要满足才保留
        return (e <= kind[1]) & (cen_d <= kind[2])
    if kind[0] == "rel":
        return e <= kind[1] * pdiag
    if kind[0] == "bb":
        pad = kind[1] * pdiag
        ins = np.all((V >= plo - pad) & (V <= phi + pad), axis=1)
        return ins[F].all(axis=1)
    if kind[0] == "bbc":           # bbox 保护 + 质心离群，交集
        pad = kind[1] * pdiag
        ins = np.all((V >= plo - pad) & (V <= phi + pad), axis=1)
        return ins[F].all(axis=1) & (cen_d <= kind[2])
    raise KeyError(name)


def L(p):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def load_ply_pts(p, cap=400000, seed=0):
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    v = np.asarray(m.vertices, float)
    if len(v) > cap:
        v = v[np.random.default_rng(seed).choice(len(v), cap, replace=False)]
    return v


def face_area(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                         V[F[:, 2]] - V[F[:, 0]]), axis=1)


def face_maxedge(V, F):
    return np.stack([np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1),
                     np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1),
                     np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1)], 1).max(1)


def submesh(V, F, keep):
    Fk = np.asarray(F)[keep]
    if len(Fk) == 0:
        return None
    u, inv = np.unique(Fk, return_inverse=True)
    return trimesh.Trimesh(vertices=np.asarray(V)[u],
                           faces=inv.reshape(-1, 3).astype(np.int64), process=False)


def m2p(m, ptree, ns, seed=7):
    """面积均匀采样 -> 最近点云点距离均值（CD1 代理）。"""
    np.random.seed(seed)
    ps = np.asarray(trimesh.sample.sample_surface(m, ns)[0], float)
    return float(ptree.query(ps)[0].mean())


def p2m(m, pts, ns, seed=7):
    """点云 -> 最近网格采样点距离均值（CD2 代理）。"""
    np.random.seed(seed)
    ms = np.asarray(trimesh.sample.sample_surface(m, ns)[0], float)
    return float(cKDTree(ms).query(pts)[0].mean())


def p2m_surf(m, pts):
    """点云 -> 最近**网格曲面**距离均值（CD2 忠实代理：官方 CD2 = 真值采样点 ->
    我方 1e6 采样点，密度足够时≈到曲面距离）。"""
    try:
        _, d, _ = trimesh.proximity.closest_point(m, pts)
        return float(np.asarray(d, float).mean())
    except Exception:                                              # noqa: BLE001
        return None


def one(job):
    sid, meshd, plyd, cfgs, ns = job
    try:
        mp = os.path.join(meshd, sid + ".obj")
        pp = os.path.join(plyd, sid + ".ply")
        if not os.path.exists(mp) or not os.path.exists(pp):
            return {"id": sid, "err": "missing"}
        pts = load_ply_pts(pp)
        if len(pts) < 8:
            return {"id": sid, "err": "tiny_pc"}
        ptree = cKDTree(pts)
        plo, phi = pts.min(0), pts.max(0)
        pdiag = float(np.linalg.norm(phi - plo))

        m = L(mp)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        V = np.asarray(m.vertices, float)
        F = np.asarray(m.faces, np.int64)
        if len(F) == 0:
            return {"id": sid, "err": "no_face"}
        mlo, mhi = V.min(0), V.max(0)
        mdiag = float(np.linalg.norm(mhi - mlo))
        A = face_area(V, F)
        Atot = float(A.sum())
        e = face_maxedge(V, F)
        cen = V[F].mean(1)
        cen_d = ptree.query(cen)[0]

        res = {}
        for name in cfgs:
            keep = keep_mask(name, V, F, e, plo, phi, pdiag, cen_d)
            if keep.sum() == 0:
                res[name] = None
                continue
            mk = m if keep.all() else submesh(V, F, keep)
            if mk is None:
                res[name] = None
                continue
            np.random.seed(7)
            ps = np.asarray(trimesh.sample.sample_surface(mk, ns)[0], float)
            d1 = float(ptree.query(ps)[0].mean())
            d2 = p2m_surf(mk, pts)
            Ln = np.linalg.norm(V[F[keep]][:, [0, 1, 2, 0]] - V[F[keep]][:, [1, 2, 0, 1]],
                                axis=2)
            res[name] = {
                "F": int(keep.sum()), "F_rm": int((~keep).sum()),
                "area": float(A[keep].sum()),
                "ratio": float(A[keep].sum() / max(Atot, 1e-9)),
                "CD1": d1, "CD2": d2,
                "CD": (d1 + d2) if d2 is not None else None,
                "elen": float(Ln.sum()),
            }
        return {"id": sid, "pc_diag": pdiag, "m_diag": mdiag,
                "diag_ratio": mdiag / max(pdiag, 1e-9),
                "F": int(len(F)), "area": Atot, "maxE": float(e.max()),
                "n_gt50": int((e > 50).sum()), "n_gt100": int((e > 100).sum()),
                "area_gt50": float(A[e > 50].sum() / max(Atot, 1e-9)),
                "res": res}
    except Exception as ex:                                        # noqa: BLE001
        return {"id": sid, "err": "%s: %s" % (type(ex).__name__, ex)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshd", default="data/work/final_v28")
    ap.add_argument("--plyd", default="data/work/ply")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--ns", type=int, default=150000)
    ap.add_argument("--cfgs", default="none,bb5,bb10,bb20,bb40,t20,ct2,ct4,ct8,bb10c4,bb20c8")
    ap.add_argument("--out", default="logs/audit_test.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    cfgs = [c.strip() for c in a.cfgs.split(",") if c.strip()]

    if a.ids:
        ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()]
    else:
        ids = sorted(set(os.path.splitext(f)[0]
                         for f in os.listdir(a.meshd) if f.endswith(".obj")))
        if a.n and a.n < len(ids):
            ids = list(np.random.default_rng(a.seed).choice(ids, a.n, replace=False))
    ids = [i for i in ids if os.path.exists(os.path.join(a.plyd, i + ".ply"))]
    print("场景 %d  ns=%d  配置=%s" % (len(ids), a.ns, cfgs), flush=True)

    jl = os.path.join(ROOT, a.out.replace(".json", ".jsonl"))
    open(jl, "w", encoding="utf-8").close()
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        jobs = [(i, a.meshd, a.plyd, cfgs, a.ns) for i in ids]
        for k, r in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.append(r)
            with open(jl, "a", encoding="utf-8") as fj:
                fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (k + 1) % 25 == 0 or k + 1 == len(ids):
                print("  %d/%d" % (k + 1, len(ids)), flush=True)

    ok = [r for r in rows if "err" not in r]
    json.dump({"n": len(ids), "ok": len(ok), "cfgs": cfgs, "rows": rows},
              open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if not ok:
        print("无有效结果")
        return

    q = lambda k, p: float(np.percentile([r[k] for r in ok], p))
    print("\n=== 提交本体画像（%d 场景）===" % len(ok))
    for k, lab in [("diag_ratio", "网格/点云 diag 比"), ("maxE", "最大边长"),
                   ("n_gt50", ">50m 边数"), ("area_gt50", ">50m 面 面积占比")]:
        print("  %-20s p50=%10.3f p90=%10.3f p99=%11.3f max=%12.2f"
              % (lab, q(k, 50), q(k, 90), q(k, 99), q(k, 100)))

    base = [r for r in ok if r["res"].get("none")]
    gg = lambda rs, name, k: float(np.mean([r["res"][name][k] for r in rs
                                            if r["res"].get(name)
                                            and r["res"][name].get(k) is not None]))
    b = {k: gg(base, "none", k) for k in ("CD1", "CD2", "CD")}
    print("\n  基线(无裁剪) 平均 CD1=%.4f CD2=%.4f CD=%.4f  (ΔFINAL = 0.6*ΔCD)"
          % (b["CD1"], b["CD2"], b["CD"]))
    print("\n  %-8s %7s %8s %8s %8s %8s %8s %8s %9s" %
          ("配置", "删面%", "删面积%", "CD1", "dCD1", "CD2", "dCD2", "dCD", "dFINAL"))
    out = []
    for name in cfgs:
        if name == "none":
            continue
        rs = [r for r in ok if r["res"].get(name) and r["res"][name].get("CD") is not None]
        if not rs:
            continue
        c1, c2, cd = (gg(rs, name, k) for k in ("CD1", "CD2", "CD"))
        fr = float(np.mean([r["res"][name]["F_rm"] / max(r["F"], 1) for r in rs]))
        ar = float(np.mean([1 - r["res"][name]["ratio"] for r in rs]))
        print("  %-8s %7.2f %8.2f %8.4f %+8.4f %8.4f %+8.4f %+8.4f %+9.4f"
              % (name, 100 * fr, 100 * ar, c1, c1 - b["CD1"], c2, c2 - b["CD2"],
                 cd - b["CD"], 0.6 * (cd - b["CD"])))
        d = np.array([r["res"][name]["CD"] - r["res"]["none"]["CD"] for r in rs
                      if r["res"].get("none") and r["res"]["none"].get("CD") is not None])
        out.append({"cfg": name, "dCD": cd - b["CD"], "dFINAL": 0.6 * (cd - b["CD"]),
                    "rm_face_pct": 100 * fr, "rm_area_pct": 100 * ar,
                    "dCD1": c1 - b["CD1"], "dCD2": c2 - b["CD2"],
                    "med_dCD": float(np.median(d)),
                    "improve_pct": float(100 * (d < 0).mean()),
                    "worst": float(d.max()), "best": float(d.min())})
    out.sort(key=lambda x: x["dFINAL"])
    print("\n  按 dFINAL 排序:")
    for o in out:
        print("    %-8s dFINAL=%+8.4f  dCD=%+9.4f  中位 %+8.4f  改善 %5.1f%%  最差 %+9.3f"
              % (o["cfg"], o["dFINAL"], o["dCD"], o["med_dCD"], o["improve_pct"],
                 o["worst"]))
    print("\n  写入", a.out)


if __name__ == "__main__":
    main()
