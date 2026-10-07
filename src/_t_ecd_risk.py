# -*- coding: utf-8 -*-
"""测试集侧「算子对 ECD 的损伤」无真值代理。

动机（2026-10-05 v12 事后剖析）：
    v12 = v10 + (zc_c2 & far5&sliv40)。实测 ΔCD = -0.01301（改善，符合预测），
    但 ΔECD = +0.02976（**变差**），导致 FINAL 反而 +0.00410。
    机制假设：**逐面删除会在网格上制造新的开边界**，而官方 ECD 把「边界边」直接
    当作锐边（src/official_evaluate.py 的 extract_sharp_edges 第 1 条），
    这些新边界离真值锐边很远 → ECD 第一项（预测锐边→真值锐边）被抬高。
    bw600 尾部看不到这个机制，因为那边是 z 爆炸壳（整块孤立几何），删掉它
    减少的远边远多于新增的边界边；测试集这边是**连通蛛网**，删中段必然increases 周长。

本脚本做三件事（全部 **不需要真值**）：
  1. 连通分量普查：`trimesh.split()` 后有几个分量、面积加权平均面心距 → 判断
     `comp_far`（整块删除）在测试集上到底有没有机会生效。
  2. 边界-长度账本：对某算子，统计
        L_bnd_before / L_bnd_after      网格**边界边总长**
        L_new / L_gone                  新增 / 消失的边界边长度
     ECD 第一项的分子代理 = Σ_{边界边 e} |e| · dist(e→点云)，
     于是 ΔECD_proxy ≈ (E_after - E_before) / L_sharp_after（L_sharp 用边界+二面角近似）。
  3. 跨算子对比，按 ΔECD_proxy 排序，并与 ΔCD1_proxy 一起看性价比。

用法：
  python src/_t_ecd_risk.py --obj-dir data/work/final_v10 --ply data/work/ply \
      --ids 367 2377 2572 1241 552 --mode far_sliv --sliv 40 --out logs/ecd_risk_farsliv40.json
"""
import os, sys, json, time, argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

FAR = 5.0


def _tri_area(V, F):
    return np.linalg.norm(np.cross(V[F][:, 1] - V[F][:, 0],
                                   V[F][:, 2] - V[F][:, 0]), axis=1) * 0.5


def _load_cloud(ply_dir, i):
    for ext in (".ply", ".xyz", ".npy"):
        p = os.path.join(ply_dir, "%s%s" % (i, ext))
        if os.path.exists(p):
            if ext == ".npy":
                return np.asarray(np.load(p), float)
            if ext == ".xyz":
                return np.asarray(np.loadtxt(p, dtype=float), float)[:, :3]
            m = trimesh.load(p, process=False)
            V = np.asarray(m.vertices, float)
            return V
    return None


def _edge_arrays(F):
    """返回唯一边 (E,2) 与每个面的三条边索引。"""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Es = np.sort(E, axis=1)
    uniq, inv = np.unique(Es, axis=0, return_inverse=True)
    return uniq, inv


def _boundary_mask(uniq, inv, nf):
    """边被几个面共享 == 1 ⇒ 边界边。用 bincount 数每个唯一边的出现次数。
    inv 长度 = 3*nf（三组）。"""
    cnt = np.bincount(inv, minlength=len(uniq))
    return cnt == 1, cnt


def _sharp_stats(V, F, pv_tree):
    """返回 (L_sharp, E_sharp) —— 锐边总长 + Σ|e|·d(e,cloud)。
    锐边 = 边界边 ∪ 二面角>30°。非流形边此处从简（占比小）。"""
    uniq, inv = _edge_arrays(F)
    cnt = np.bincount(inv, minlength=len(uniq))
    is_b = cnt == 1
    # 二面角
    nf = len(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(fn, axis=1)
    ln[ln < 1e-12] = 1e-12
    fn = fn / ln[:, None]
    # 每条唯一边记录它所属的面
    order = np.argsort(inv, kind="stable")
    owner = order // 3  # inv 的第 k 项对应面 k//3
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.searchsorted(inv[order], np.arange(len(uniq)), side="right")
    is_dih = np.zeros(len(uniq), bool)
    k2 = np.where(cnt == 2)[0]
    if len(k2):
        f1 = owner[order[starts[k2]]]
        f2 = owner[order[starts[k2] + 1]]
        c = np.clip(np.einsum("ij,ij->i", fn[f1], fn[f2]), -1.0, 1.0)
        is_dih[k2] = np.degrees(np.arccos(c)) > 30.0
    sharp = is_b | is_dih
    sv = V[uniq[sharp]]
    elen = np.linalg.norm(sv[:, 1] - sv[:, 0], axis=1)
    mid = 0.5 * (sv[:, 0] + sv[:, 1])
    d = pv_tree.query(mid, k=1)[0]
    return float(elen.sum()), float((elen * d).sum()), is_b


def analyse(obj_path, ply_dir, i, ops):
    o = trimesh.load(obj_path, process=False)
    if isinstance(o, trimesh.Scene):
        o = trimesh.util.concatenate(tuple(o.geometry.values()))
    V = np.asarray(o.vertices, float)
    F = np.asarray(o.faces, int).reshape(-1, 3)
    pv = _load_cloud(ply_dir, i)
    if pv is None or len(pv) == 0 or len(F) == 0:
        return None
    pv = np.ascontiguousarray(pv[:, :3], float)
    tree = cKDTree(pv)

    A = _tri_area(V, F)
    C = V[F].mean(axis=1)
    d_c = tree.query(C, k=1)[0]
    asp = np.zeros(len(F))
    e = np.stack([np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
                  np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
                  np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)], axis=1)
    mx = e.max(axis=1)
    asp = np.where(A > 1e-12, mx * mx / np.maximum(A, 1e-12), 1e9)

    # 连通分量结构
    try:
        parts = o.split(only_watertight=False)
        ncomp = len(parts) if parts is not None else 0
        comp_info = []
        if parts is not None and ncomp > 0:
            for p in parts:
                Vp = np.asarray(p.vertices, float)
                Fp = np.asarray(p.faces, int).reshape(-1, 3)
                ap = _tri_area(Vp, Fp)
                if len(Fp) == 0:
                    continue
                cp = Vp[Fp].mean(axis=1)
                dp = tree.query(cp, k=1)[0]
                comp_info.append({
                    "nf": int(len(Fp)),
                    "area": float(ap.sum()),
                    "wm2p": float((dp * ap).sum() / max(1e-12, ap.sum())),
                })
            comp_info.sort(key=lambda z: -z["area"])
    except Exception as ex:
        ncomp = -1
        comp_info = []

    L0, E0, isb0 = _sharp_stats(V, F, tree)

    res = {"id": str(i), "nf": int(len(F)), "area": float(A.sum()),
           "m2p": float((d_c * A).sum() / max(1e-12, A.sum())),
           "L_sharp0": L0, "E_sharp0": E0,
           "ecd_proxy0": (E0 / L0) if L0 > 0 else 0.0,
           "ncomp": ncomp, "ncomp_area_gt1pct": sum(
               1 for c in comp_info if c["area"] > 0.01 * max(1e-12, A.sum())),
           "top_comps": comp_info[:6],
           "ops": {}}

    for name, keep in ops.items():
        if keep is None or keep.sum() == 0 or keep.all():
            res["ops"][name] = {"fired": False}
            continue
        Fk = F[keep]
        Vk = V
        # 只保留被引用的顶点（不重新焊接，保持拓扑）
        used = np.unique(Fk)
        remap = -np.ones(len(V), int)
        remap[used] = np.arange(len(used))
        Vk = V[used]
        Fk = remap[Fk]
        L1, E1, isb1 = _sharp_stats(Vk, Fk, tree)
        res["ops"][name] = {
            "fired": True,
            "n_del": int((~keep).sum()),
            "del_area_frac": float(A[~keep].sum() / max(1e-12, A.sum())),
            "L_sharp1": L1, "E_sharp1": E1,
            "ecd_proxy1": (E1 / L1) if L1 > 0 else 0.0,
            "dECD_proxy": ((E1 / L1) - (E0 / L0)) if (L0 > 0 and L1 > 0) else 0.0,
            "dCD1_proxy": float((d_c[keep] * A[keep]).sum() / max(1e-12, A[keep].sum())
                                - (d_c * A).sum() / max(1e-12, A.sum())),
        }
    return res


def build_ops(V, F, tree, modes):
    A = _tri_area(V, F)
    C = V[F].mean(axis=1)
    d = tree.query(C, k=1)[0]
    e = np.stack([np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
                  np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
                  np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)], axis=1)
    asp = np.where(A > 1e-12, e.max(axis=1) ** 2 / np.maximum(A, 1e-12), 1e9)
    # ⚠ 约定：ops[m] 返回的是【保留】掩码（keep），analyse 里 keep=F[ops[m]]。
    # 因此「删除 X」= keep = ~X。2026-10-05 首次运行把方向搞反了，重跑前已修正。
    ops = {}
    for m in modes:
        if m == "far5_only":            # 删除：面心距点云 > 5 m
            ops[m] = ~(d > FAR)
        elif m == "sliv40_only":        # 删除：细长比 > 40
            ops[m] = ~(asp > 40)
        elif m == "sliv100_only":
            ops[m] = ~(asp > 100)
        elif m == "far5_sliv40":        # 删除：远 且 细长（= 现行照抄版本）
            ops[m] = ~((d > FAR) & (asp > 40))
        elif m == "far5_sliv100":       # 删除：远 且 极细长（更保守，新增边界更少）
            ops[m] = ~((d > FAR) & (asp > 100))
        elif m == "far2_sliv40":
            ops[m] = ~((d > 2.0) & (asp > 40))
        elif m == "far5_sliv40_small":
            # 只删「小面」的远细长面：删掉的面本身短 ⇒ 新边界也短
            ops[m] = ~((d > FAR) & (asp > 40) & (A < np.percentile(A, 50)))
        elif m == "far5_sliv40_big":
            # 只删「大面」的远细长面：面积收益大，但新边界长
            ops[m] = ~((d > FAR) & (asp > 40) & (A >= np.percentile(A, 50)))
        elif m == "far5_notsliv":
            ops[m] = ~((d > FAR) & (asp <= 40))
        else:
            ops[m] = None
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["far5_only", "sliv40_only",
                                                   "far5_sliv40", "far5_sliv100"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for i in a.ids:
        for cand in ("%s.obj" % i, os.path.join(a.obj_dir, "%s.obj" % i),
                     os.path.join(a.obj_dir, str(i), "%s.obj" % i)):
            if os.path.exists(cand):
                p = cand
                break
        else:
            # 目录形如 final_v10/<id>.obj 或 final_v10/<id>/...
            import glob
            g = glob.glob(os.path.join(a.obj_dir, "**", "%s.obj" % i), recursive=True)
            if not g:
                print("[skip] no obj for", i)
                continue
            p = g[0]
        o = trimesh.load(p, process=False)
        if isinstance(o, trimesh.Scene):
            o = trimesh.util.concatenate(tuple(o.geometry.values()))
        V = np.asarray(o.vertices, float)
        F = np.asarray(o.faces, int).reshape(-1, 3)
        pv = _load_cloud(a.ply, i)
        if pv is None:
            print("[skip] no cloud for", i)
            continue
        tree = cKDTree(np.ascontiguousarray(pv[:, :3], float))
        ops = build_ops(V, F, tree, a.modes)
        r = analyse(p, a.ply, i, ops)
        if r:
            rows.append(r)
            print("== id=%s nf=%d area=%.0f m2p=%.3f ncomp=%d L_sharp0=%.0f ecd0=%.3f"
                  % (r["id"], r["nf"], r["area"], r["m2p"], r["ncomp"],
                     r["L_sharp0"], r["ecd_proxy0"]))
            for k, v in r["ops"].items():
                if v.get("fired"):
                    print("   %-16s del%5.1f%%  L_sharp %.0f->%.0f  "
                          "ecd_proxy %.3f->%.3f (d%+.3f)  dCD1 %+.3f"
                          % (k, 100 * v["del_area_frac"], r["L_sharp0"], v["L_sharp1"],
                             r["ecd_proxy0"], v["ecd_proxy1"], v["dECD_proxy"],
                             v["dCD1_proxy"]))

    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n[out]", a.out)
    # 汇总
    if rows:
        print("\n=== 汇总（跨 %d 个样本，按面积加权） ===" % len(rows))
        for m in a.modes:
            fire = [r for r in rows if r["ops"].get(m, {}).get("fired")]
            if not fire:
                print("%-18s 从不生效" % m)
                continue
            wa = sum(r["area"] for r in fire)
            dd = sum(r["area"] * r["ops"][m]["dECD_proxy"] for r in fire) / wa
            dc = sum(r["area"] * r["ops"][m]["dCD1_proxy"] for r in fire) / wa
            dfr = sum(r["area"] * r["ops"][m]["del_area_frac"] for r in fire) / wa
            # 未加权（每栋等权）——面积加权会被 2377 这种巨型瓦片独占，必须同时看
            dd0 = float(np.mean([r["ops"][m]["dECD_proxy"] for r in fire]))
            dc0 = float(np.mean([r["ops"][m]["dCD1_proxy"] for r in fire]))
            win = sum(1 for r in fire if r["ops"][m]["dECD_proxy"] < 0)
            print("%-20s 生效%2d/%d 删面%5.1f%% | 面积加权 ΔCD1 %+8.3f ΔECD %+7.3f "
                  "| 等权 ΔCD1 %+7.3f ΔECD %+7.3f (ECD 改善 %d/%d)"
                  % (m, len(fire), len(rows), 100 * dfr, dc, dd, dc0, dd0, win,
                     len(fire)))


if __name__ == "__main__":
    main()
