"""
降级重建器（纯几何，无外部依赖）

用途：City3D 对极少数样本仍可能失败 / 超时 / 输出为空时，保证流水线
      100% 有合法输出，避免提交包里缺文件。

思路（LoD1+ 级别的"诚实降级"，不假装细节）：
    XY 栅格化 -> 连通域 -> 定向边界跟踪 -> Douglas-Peucker 简化
    -> 沿轮廓拉出墙面 -> 顶部拟合平面做屋顶（近乎水平则退化为平屋顶）

产出面片数很少（一栋楼通常 20~60 个三角形），因此 V_Ratio / F_Ratio
天然接近 1，CD / NC 也说得过去；代价是丢掉真实屋顶结构（ECD 会差）。
"""
import numpy as np
import trimesh
from scipy import ndimage


# ---------------------------------------------------------------- 边界跟踪

def _mask_from_points(xy, cell):
    """XY -> 占据栅格"""
    mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / cell).astype(np.int64)
    H, W = ij[:, 1].max() + 1, ij[:, 0].max() + 1
    mask = np.zeros((H, W), bool)
    mask[ij[:, 1], ij[:, 0]] = True
    return mask, mn


def _trace_loops(mask):
    """
    定向边界跟踪：对每个占据格的四条空边生成有向边，再串成闭环。
    返回 [[(i,j), ...], ...]（栅格角点坐标，行在前）
    """
    H, W = mask.shape
    pad = np.zeros((H + 2, W + 2), bool)
    pad[1:-1, 1:-1] = mask

    nxt = {}
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            if not pad[i, j + 1]:                       # 北
                nxt[(i, j + 1)] = (i, j)
            if not pad[i + 2, j + 1]:                   # 南
                nxt[(i + 1, j)] = (i + 1, j + 1)
            if not pad[i + 1, j]:                       # 西
                nxt[(i, j)] = (i + 1, j)
            if not pad[i + 1, j + 2]:                   # 东
                nxt[(i + 1, j + 1)] = (i, j + 1)

    loops, used = [], set()
    for start in list(nxt.keys()):
        if start in used:
            continue
        loop, cur = [], start
        while cur not in used and cur in nxt:
            used.add(cur)
            loop.append(cur)
            cur = nxt[cur]
            if cur == start:
                break
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def _signed_area(loop, mn, cell):
    pts = np.array([(mn[0] + j * cell, mn[1] + i * cell) for (i, j) in loop])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _rdp(pts, eps):
    """Douglas-Peucker 简化（对闭合环处理：先用最远点对切成两条链）"""
    if len(pts) <= 4:
        return pts
    d = np.linalg.norm(pts - pts[0], axis=1)
    k = int(np.argmax(d))
    if k == 0:
        return pts
    a = _rdp_open(pts[:k + 1], eps)
    b = _rdp_open(np.vstack([pts[k:], pts[:1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


def _rdp_open(pts, eps):
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p0, p1 = pts[i], pts[j]
        seg = p1 - p0
        L = np.linalg.norm(seg)
        if L < 1e-12:
            dist = np.linalg.norm(pts[i + 1:j] - p0, axis=1)
        else:
            d = pts[i + 1:j] - p0
            dist = np.abs(seg[0] * d[:, 1] - seg[1] * d[:, 0]) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.extend([(i, m), (m, j)])
    return pts[keep]


# ---------------------------------------------------------------- 主入口

def reconstruct(pts, cell=None, simplify_tol=None, min_pts_ratio=0.08,
                roof_slope_deg=8.0):
    """
    pts : (N,3) 点云，z 轴朝上
    返回 trimesh.Trimesh，或 None（实在建不出来）
    """
    pts = np.asarray(pts, float)
    if len(pts) < 30:
        return None
    xy = pts[:, :2]
    span = xy.max(axis=0) - xy.min(axis=0)
    # ⚠️ 尺度自适应（2026-09-30 修）：原实现 cell 与 simplify_tol 都写死了
    #    0.35 m 的绝对下限，对 XY 跨度仅 3.8 cm 的样本（3380 / 901）：
    #      · 栅格塌成 1×1，膨胀后 0 格 → 环跟踪为空；
    #      · 即便栅格正常，_rdp(eps=0.35) 也会把 244 点轮廓一刀削到 <3 点
    #        而被 `if len(poly) < 3: continue` 跳过。
    #    改为：以 5 m 为分界，短边小于该值时下限与简化容差按比例缩放；
    #    ≥5 m 时与原公式逐字一致，保证常规样本（含合成集 n=9）行为不变。
    #    实测 3380/901 由 None 变为 V=58/F=112、V=82/F=160。
    short = float(min(span))
    scale = min(1.0, short / 5.0) if short > 0 else 1.0
    if cell is None:
        cell = float(np.clip(short / 60.0, 0.35 * scale, 1.2 * scale))
        long_ = float(max(span))
        if long_ > 0 and long_ / cell > 900:   # 限制栅格规模，防极端长宽比爆内存
            cell = long_ / 900.0

    mask, mn = _mask_from_points(xy, cell)
    mask = ndimage.binary_dilation(mask, np.ones((3, 3), bool))
    # 闭运算核同样要自适应：细长建筑（如 1853，栅格 30×5）用 5×5 核会被
    # binary_closing 的腐蚀步整个吃掉（实测占据格 77 → 26），改用 3×3。
    k = 5 if min(mask.shape) >= 9 else 3
    mask = ndimage.binary_closing(mask, np.ones((k, k), bool))

    if simplify_tol is None:
        simplify_tol = 0.35 * scale

    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return None

    loops = _trace_loops(mask)
    if not loops:
        return None

    # 只要外轮廓（去掉孔洞：面积为负的环），并按面积排序取主轮廓
    outer = []
    for lp in loops:
        a = _signed_area(lp, mn, cell)
        if a <= 0:                      # 孔洞 / 反向环
            continue
        outer.append((a, lp))
    if not outer:
        # 全部是负面积（整体朝向约定相反）—— 取面积绝对值最大者并反向
        outer = [(abs(_signed_area(lp, mn, cell)), lp) for lp in loops]
    outer.sort(key=lambda t: -t[0])
    loops = [lp for _, lp in outer]

    # 只保留面积不小于主轮廓 min_pts_ratio 的部件，滤掉碎片
    a0 = outer[0][0]
    loops = [lp for (a, lp) in outer if a >= min_pts_ratio * a0]

    verts, faces = [], []
    z_base = float(np.percentile(pts[:, 2], 1.0))

    for lp in loops:
        poly = np.array([(mn[0] + j * cell, mn[1] + i * cell) for (i, j) in lp])
        poly = _rdp(poly, simplify_tol)
        if len(poly) < 3:
            continue

        # 该轮廓内的点 -> 定屋顶
        inside = _points_in_poly(xy, poly)
        sub = pts[inside]
        if len(sub) < 10:
            sub = pts
        z_hi = float(np.percentile(sub[:, 2], 92.0))
        top = sub[sub[:, 2] >= np.percentile(sub[:, 2], 80.0)]
        if len(top) < 8:
            top = sub

        A = np.column_stack([top[:, 0], top[:, 1], np.ones(len(top))])
        coef, *_ = np.linalg.lstsq(A, top[:, 2], rcond=None)
        slope = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
        if slope < roof_slope_deg:
            roof_z = lambda P: np.full(len(P), z_hi)          # noqa: E731
        else:
            roof_z = lambda P: coef[0] * P[:, 0] + coef[1] * P[:, 1] + coef[2]  # noqa: E731

        n0 = len(verts)
        m = len(poly)
        # 顶面顶点
        zt = roof_z(poly)
        for k in range(m):
            verts.append([poly[k, 0], poly[k, 1], zt[k]])
        # 底面顶点
        for k in range(m):
            verts.append([poly[k, 0], poly[k, 1], z_base])
        # 墙面
        for k in range(m):
            k2 = (k + 1) % m
            faces.append([n0 + k, n0 + k2, n0 + m + k2])
            faces.append([n0 + k, n0 + m + k2, n0 + m + k])
        # 屋顶扇面
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        cz = float(roof_z(np.array([[cx, cy]]))[0])
        cid = len(verts)
        verts.append([cx, cy, cz])
        for k in range(m):
            k2 = (k + 1) % m
            faces.append([cid, n0 + k, n0 + k2])
        # 地面扇面（法向朝下，靠 orient_outward 统一）
        bid = len(verts)
        verts.append([cx, cy, z_base])
        for k in range(m):
            k2 = (k + 1) % m
            faces.append([bid, n0 + m + k2, n0 + m + k])

    if not faces:
        return None
    mesh = trimesh.Trimesh(vertices=np.array(verts, float),
                           faces=np.array(faces, int), process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh if len(mesh.faces) else None


def _points_in_poly(xy, poly):
    """射线法（向量化）"""
    n = len(poly)
    inside = np.zeros(len(xy), bool)
    x, y = xy[:, 0], xy[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        inside ^= cond & (x < xin)
        j = i
    return inside


# ---------------------------------------------------------------- 自检

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from synth import gable_roof, hip_roof, simulate_als

    for name, gt in [("双坡", gable_roof(0, 0, 20, 14, 0, 8, 13)),
                     ("四坡", hip_roof(0, 0, 18, 18, 0, 8, 14))]:
        pts = simulate_als(gt, n_points=18000, seed=3)
        m = reconstruct(pts)
        print(f"{name}: 点 {len(pts)}  ->  V={len(m.vertices)} F={len(m.faces)}")
