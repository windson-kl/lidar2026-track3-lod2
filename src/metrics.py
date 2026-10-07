"""
赛道三评测指标本地实现

对应官方 BuildingWorld_Evaluation 的五项指标：
    CD        Chamfer Distance        预测与真值的整体几何偏差（越小越好）
    ECD       Edge Chamfer Distance   仅边缘线的几何偏差（越小越好）
    NC        Normal Consistency      表面法向一致性（越大越好）
    V_Ratio   顶点数比                 pred_V / gt_V（越接近 1 越好）
    F_Ratio   面片数比                 pred_F / gt_F（越接近 1 越好）

注意：官方权重未公开，本实现按 Building3D 惯例给出「越大越好」的统一得分
     和一个等权综合分，仅用于本地相对比较，不等价于官方榜单分数。
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

O3D = None
try:                                    # 有 open3d 就用它的采样（更均匀）
    import open3d as O3D
except Exception:
    pass


# ---------------------------------------------------------------- 采样工具

def _surface_samples(mesh, n, seed=0):
    """在 mesh 表面做面积加权采样，返回 (点, 面法向)"""
    if len(mesh.faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts, fid = trimesh.sample.sample_surface(mesh, int(n), seed=seed)
    return np.asarray(pts, float), np.asarray(mesh.face_normals)[fid]


def _edge_samples(mesh, per_edge=12):
    """沿所有唯一边均匀采样，用于 ECD"""
    e = np.asarray(mesh.edges_unique, int)
    if len(e) == 0:
        return np.zeros((0, 3))
    v0 = np.asarray(mesh.vertices, float)[e[:, 0]]
    v1 = np.asarray(mesh.vertices, float)[e[:, 1]]
    ts = np.linspace(0.0, 1.0, per_edge)[:, None, None]
    pts = (1.0 - ts) * v0[None] + ts * v1[None]
    return pts.reshape(-1, 3)


def _chamfer(a, b):
    """对称 chamfer 距离（双向平均最近点距离）"""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab = ta.query(b)[0]
    d_ba = tb.query(a)[0]
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


# ---------------------------------------------------------------- 五项指标

def chamfer_distance(pred, gt, n=60000, seed=0):
    p, _ = _surface_samples(pred, n, seed)
    g, _ = _surface_samples(gt, n, seed + 1)
    return _chamfer(p, g)


def edge_chamfer_distance(pred, gt, per_edge=12):
    p = _edge_samples(pred, per_edge)
    g = _edge_samples(gt, per_edge)
    return _chamfer(p, g)


def normal_consistency(pred, gt, n=60000, seed=0):
    if len(pred.faces) == 0 or len(gt.faces) == 0:
        return float("nan")
    pp, pn = _surface_samples(pred, n, seed)
    gp, gn = _surface_samples(gt, n, seed + 1)
    _, idx = cKDTree(gp).query(pp)
    return float(np.abs((pn * gn[idx]).sum(axis=1)).mean())


def vf_ratio(pred, gt):
    v = len(pred.vertices) / max(1, len(gt.vertices))
    f = len(pred.faces) / max(1, len(gt.faces))
    return float(v), float(f)


# ---------------------------------------------------------------- 归一化

def _to_score(value, good, bad):
    """线性映射到 0~1，value==good 得 1，value==bad 得 0"""
    if not np.isfinite(value):
        return 0.0
    if good == bad:
        return 0.0
    s = (value - bad) / (good - bad)
    return float(np.clip(s, 0.0, 1.0))


def _ratio_score(r):
    """V_Ratio / F_Ratio 的得分：r=1 得满分，偏离按比例衰减"""
    if not np.isfinite(r) or r <= 0:
        return 0.0
    return float(np.clip(min(r, 1.0 / r), 0.0, 1.0))


def evaluate(pred, gt, cd_bad=None, ecd_bad=None, n=60000, seed=0):
    """
    计算全部五项指标，并给出「越大越好」的归一化得分。

    cd_bad / ecd_bad 是归一化参考上限；不给则动态取 GT 自身包围盒对角线的 5% / 3%。
    """
    diag = float(np.linalg.norm(np.asarray(gt.bounds)[1] - np.asarray(gt.bounds)[0]))
    cd_bad = cd_bad if cd_bad is not None else 0.05 * diag
    ecd_bad = ecd_bad if ecd_bad is not None else 0.03 * diag

    cd = chamfer_distance(pred, gt, n, seed)
    ecd = edge_chamfer_distance(pred, gt)
    nc = normal_consistency(pred, gt, n, seed)
    vr, fr = vf_ratio(pred, gt)

    raw = {"CD": cd, "ECD": ecd, "NC": nc, "V_Ratio": vr, "F_Ratio": fr}
    score = {
        "CD": _to_score(cd, 0.0, cd_bad),
        "ECD": _to_score(ecd, 0.0, ecd_bad),
        "NC": _to_score(nc, 1.0, 0.0),
        "V_Ratio": _ratio_score(vr),
        "F_Ratio": _ratio_score(fr),
    }
    score["OVERALL"] = float(np.mean(list(score.values())))
    return raw, score, {"cd_bad": cd_bad, "ecd_bad": ecd_bad, "diag": diag}


# ---------------------------------------------------------------- 自检

def _selftest():
    """用合成数据自检：GT 与自身比应近乎满分；加噪声后分数应下降"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from synth import prism_rect, gable_roof

    gt = gable_roof(0, 0, 20, 14, 0, 9, 13)
    print(f"GT: V={len(gt.vertices)} F={len(gt.faces)}")

    print("\n--- 场景1：预测 == 真值（应近乎满分）---")
    raw, sc, ref = evaluate(gt, gt)
    for k in ["CD", "ECD", "NC", "V_Ratio", "F_Ratio"]:
        print(f"  {k:<8} raw={raw[k]:>10.5f}   score={sc[k]:.4f}")
    print(f"  {'OVERALL':<8} {'':>10}   score={sc['OVERALL']:.4f}")

    print("\n--- 场景2：把 GT 简化成 4 个面片的盒子（F_Ratio 应崩）---")
    box = prism_rect(0, 0, 20, 14, 0, 11)
    raw, sc, ref = evaluate(box, gt)
    for k in ["CD", "ECD", "NC", "V_Ratio", "F_Ratio"]:
        print(f"  {k:<8} raw={raw[k]:>10.5f}   score={sc[k]:.4f}")
    print(f"  {'OVERALL':<8} {'':>10}   score={sc['OVERALL']:.4f}")

    print("\n--- 场景3：GT 加入测量噪声 ---")
    noisy = gt.copy()
    rng = np.random.default_rng(0)
    noisy.vertices = noisy.vertices + rng.normal(0, 0.15, noisy.vertices.shape)
    raw, sc, ref = evaluate(noisy, gt)
    print(f"  CD={raw['CD']:.4f}  NC={raw['NC']:.4f}  OVERALL={sc['OVERALL']:.4f}")


if __name__ == "__main__":
    _selftest()
