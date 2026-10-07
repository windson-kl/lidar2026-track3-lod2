# -*- coding: utf-8 -*-
"""★★★ 受控实验：注入与 final_v28 同量级的"浮空薄片"，用真值+官方指标验证 ct 规则。

关键标定（反推真实提交）：
    final_v28 代理 CD1 = 1.37，其中约 34% 面积是离群面。
    设离群面平均离真值 D、其余 0.5 ⇒ 1.37 ≈ 0.34·D + 0.66·0.5 ⇒ **D ≈ 3.1 m**。
    即真实垃圾**并不远**，只是又大又薄又稍微浮起来（1~6 m），
    面积巨大（1.8% 的面吃 57% 的面积）⇒ 按面积采样的 CD1 被它主导。

注入方式（贴合上述标定）：
    junk = K 份「真实面片」的复制品，按面积加权随机挑面，
           绕质心放大 s 倍（普通件 2~5，长边件 8~25），再整体抬高 h（1~6 m）。
    于是垃圾与真值的距离 ≈ h（1~6 m），且长边可到几十~上百米。

同时直接测量规则的**查全率/误删率**（因为注入时知道哪些面是垃圾）：
    recall = 被删掉的垃圾面积 / 垃圾总面积
    FP     = 被误删的合法面积 / 合法总面积

用法:
    python src/_t_junk_ct.py --ids logs/all_ids.txt --n 40 --jobs 4 \
        --ops t20,ct0.5,ct1,ct2,ct4,ct8 --out logs/junk_ct.json
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
from _t_surf_only import L, cat, load_ply_pts, resolve, score        # noqa: E402
from _t_mads import shell                                             # noqa: E402
import official_evaluate as OE                                        # noqa: E402

ROOT = r"D:/LiDAR2026"
JUNK_FRAC = 0.34          # 注入垃圾占总面积的比例（由上面反推标定）


def farea(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                         V[F[:, 2]] - V[F[:, 0]]), axis=1)


def add_junk(m, seed=0, frac=JUNK_FRAC):
    """返回 (junk_mesh, n_base_face, junk_face_lo)。垃圾面索引 >= n_base_face。"""
    rng = np.random.default_rng(seed)
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces, np.int64)
    A = farea(V, F)
    nb = len(F)
    need = frac / (1 - frac) * A.sum()
    tris, acc, guard = [], 0.0, 0
    while acc < need and guard < 4000:
        guard += 1
        i = int(np.searchsorted(np.cumsum(A / A.sum()), rng.random()))
        i = min(i, nb - 1)
        tri = V[F[i]]
        cen = tri.mean(0)
        long_ = rng.random() < 0.2
        s = rng.uniform(8, 25) if long_ else rng.uniform(2, 5)
        h = rng.uniform(1, 4) if long_ else rng.uniform(1, 6)
        tri = (tri - cen) * s + cen + np.array([0.0, 0.0, h])
        tris.append(tri)
        acc += 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    nv0 = len(V)
    nv = np.vstack([V] + tris)
    nf = np.vstack([F, np.arange(nv0, nv0 + 3 * len(tris)).reshape(-1, 3)])
    return (trimesh.Trimesh(vertices=nv, faces=nf.astype(np.int64), process=False),
            nb, float(acc))


def subset(V, F, keep):
    Fk = F[keep]
    if len(Fk) < 4:
        return None
    u, inv = np.unique(Fk, return_inverse=True)
    return trimesh.Trimesh(vertices=V[u], faces=inv.reshape(-1, 3).astype(np.int64),
                           process=False)


def rule_keep(rule, V, F, cpts, diag):
    if rule == "none":
        return np.ones(len(F), bool)
    e = np.stack([np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1),
                  np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1),
                  np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1)], 1).max(1)
    if rule == "t20":
        return e <= 20.0
    if rule.startswith("ct"):
        return cKDTree(cpts).query(V[F].mean(1))[0] <= float(rule[2:])
    if rule.startswith("v"):          # 三顶点最大距离
        d = np.stack([cKDTree(cpts).query(V[F[:, k]])[0] for k in range(3)], 1).max(1)
        return d <= float(rule[1:])
    raise KeyError(rule)


def one(job):
    sid, rec, gtd, plyd, ops = job
    try:
        rp = resolve(rec, sid)
        gtp = os.path.join(gtd, sid + "_gt.obj")
        pp = os.path.join(plyd, sid + ".ply")
        if rp is None or not os.path.exists(gtp) or not os.path.exists(pp):
            return {"id": sid, "err": "missing"}
        gt = L(gtp); base0 = L(rp); cpts = load_ply_pts(pp)
        diag = float(np.linalg.norm(cpts.max(0) - cpts.min(0)))
        V0 = np.asarray(base0.vertices, float); F0 = np.asarray(base0.faces, np.int64)
        if len(F0) == 0:
            return {"id": sid, "err": "no_face"}
        e0 = np.stack([np.linalg.norm(V0[F0[:, 0]] - V0[F0[:, 1]], axis=1),
                       np.linalg.norm(V0[F0[:, 1]] - V0[F0[:, 2]], axis=1),
                       np.linalg.norm(V0[F0[:, 2]] - V0[F0[:, 0]], axis=1)], 1).max(1)
        if e0.max() >= 1.5 * diag:
            return {"id": sid, "err": "skip_insane", "ratio": float(e0.max() / diag)}

        sd = abs(hash(sid)) % (2 ** 31)
        np.random.seed(sd % 313)
        gsp = np.asarray(trimesh.sample.sample_surface(gt, 300000)[0], float)
        gs_tree = cKDTree(gsp)
        gsub = gsp[np.random.default_rng(7).choice(len(gsp), 60000, replace=False)]
        geg = OE.extract_sharp_edges(gt, 30.0)
        np.random.seed(sd % 317)
        gep = np.asarray(OE.sample_points_on_edges_global(
            np.asarray(gt.vertices, float), geg, 100000), float)
        gep_tree = cKDTree(gep)

        shells = [shell(cpts, 3.0, 9.0, ax, seed=sd + 10 * ax) for ax in (2, 0, 1)]
        base = cat(base0, *shells)
        if base is None:
            return {"id": sid, "err": "no_base"}
        junk, nb, junkA = add_junk(base, seed=sd)
        Vj = np.asarray(junk.vertices, float); Fj = np.asarray(junk.faces, np.int64)
        Aj = farea(Vj, Fj)
        isjunk = np.zeros(len(Fj), bool); isjunk[nb:] = True

        def ev(mesh):
            r = score(mesh, gs_tree, gep_tree, gep)
            if r is None:
                return None
            r["CD2"] = float(cKDTree(np.asarray(trimesh.sample.sample_surface(
                mesh, 60000)[0], float)).query(gsub)[0].mean())
            r["CDsum"] = r["CD1"] + r["CD2"]
            r["FINAL"] = 0.6 * r["CDsum"] + 0.4 * r["ECD"]
            return r

        res = {"base": ev(base), "junk": ev(junk)}
        for rule in ops:
            keep = rule_keep(rule, Vj, Fj, cpts, diag)
            mk = subset(Vj, Fj, keep)
            r = ev(mk) if mk is not None else None
            if r is not None:
                rm = ~keep
                r["recall"] = float(Aj[rm & isjunk].sum() / max(Aj[isjunk].sum(), 1e-9))
                r["FP"] = float(Aj[rm & ~isjunk].sum() / max(Aj[~isjunk].sum(), 1e-9))
                r["F_rm"] = float(rm.mean())
            res["junk+" + rule] = r
        return {"id": sid, "diag": diag, "gtF": int(len(np.asarray(gt.faces))),
                "junk_A_frac": float(junkA / Aj.sum()), "res": res}
    except Exception as ex:                                             # noqa: BLE001
        return {"id": sid, "err": "%s: %s" % (type(ex).__name__, str(ex)[:90])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="data/work/bw600/wC")
    ap.add_argument("--gt", default="data/work/bw600_gt")
    ap.add_argument("--ply", default="data/work/bw600/ply")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ops", default="t20,ct0.5,ct1,ct2,ct4,ct8,v2")
    ap.add_argument("--out", default="logs/junk_ct.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    ops = [s.strip() for s in a.ops.split(",") if s.strip()]
    ids = [l.strip().replace("\r", "") for l in open(a.ids, encoding="utf-8")
           if l.strip()][:a.n]
    print("场景 %d  ops=%s" % (len(ids), ops), flush=True)
    jl = os.path.join(ROOT, a.out.replace(".json", ".jsonl"))
    open(jl, "w", encoding="utf-8").close()
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(one, [(i, a.rec, a.gt, a.ply, ops)
                                           for i in ids], chunksize=1)):
            rows.append(r)
            with open(jl, "a", encoding="utf-8") as fj:
                fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("  %d/%d %s" % (k + 1, len(ids), r.get("err", "ok")), flush=True)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False)

    cl = [r for r in rows if "err" not in r and r["res"].get("junk")]
    print("\n有效 %d（剔除本体重建爆炸 %d）"
          % (len(cl), sum(1 for r in rows if r.get("err") == "skip_insane")))
    if not cl:
        return 1
    g = lambda tag, k: float(np.mean([r["res"][tag][k] for r in cl
                                      if r["res"].get(tag) and
                                      r["res"][tag].get(k) is not None]))
    bf = g("base", "FINAL")
    print("注入垃圾面积占比 %.3f（实测 final_v28 长边面面积占比 0.573）"
          % float(np.median([r["junk_A_frac"] for r in cl])))
    print("\n%-12s %8s %8s %8s %8s %8s %8s %10s %9s %7s %7s %7s" %
          ("方案", "CD1", "CD2", "CD", "E1", "E2", "ECD", "FINAL", "vs base",
           "recall", "FP", "删面%"))
    for tag in ["base", "junk"] + ["junk+" + o for o in ops]:
        if not any(r["res"].get(tag) for r in cl):
            continue
        m = lambda k: g(tag, k)
        rc = m("recall") if tag.startswith("junk+") else 0.0
        fp = m("FP") if tag.startswith("junk+") else 0.0
        fr = m("F_rm") if tag.startswith("junk+") else 0.0
        print("%-12s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10.4f %+9.4f %7.3f %7.3f %7.2f" %
              (tag, m("CD1"), m("CD2"), m("CDsum"), m("E1"), m("E2"), m("ECD"),
               m("FINAL"), m("FINAL") - bf, rc, fp, 100 * fr))
    print("\n判据: junk+X 的 FINAL 越接近 base 越好；FP 大 = 误删合法几何。")
    print("写入", a.out)


if __name__ == "__main__":
    main()
