"""
指标导向的后处理模块

核心思路：官方五项指标是公开的，因此可以「反向设计」后处理，让每一项都朝好的方向走。
    CD          -> 去掉离群面片（地面板、悬挂片、内部冗余面）
    ECD         -> 主方向对齐 + 直角约束（曼哈顿世界假设）
    NC          -> 统一法向朝外
    V_Ratio     -> 去冗余顶点
    F_Ratio     -> 共面片合并 + 可选进一步简化

所有函数都保持输入坐标系不变（只做几何合并与简化，不做平移/旋转/缩放）。
"""
import numpy as np
import trimesh


# ---------------------------------------------------------------- 顶点清理

def clean_vertices(mesh, tol=1e-6):
    """合并重合顶点，删除未被引用的顶点"""
    m = mesh.copy()
    d = int(-np.log10(tol))
    # trimesh 4.x 的参数名是 digits_vertex（旧版为 digits）
    try:
        m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=d)
    except TypeError:
        m.merge_vertices(merge_tex=True, merge_norm=True, digits=d)
    m.remove_unreferenced_vertices()
    try:
        m.update_faces(m.nondegenerate_faces())
    except Exception:
        m.remove_degenerate_faces()
    return m


# ---------------------------------------------------------------- 共面合并

def merge_coplanar(mesh, angle_tol_deg=3.0, dist_tol=None):
    """
    把近似共面的相邻三角形合并成大面片。

    这是压 F_Ratio 最有效的一步：一个平面屋顶原本被切成几十个三角形，
    合并后可能只剩 2 个，而几何几乎不变（CD/ECD 不受损）。

    ⚠️ `dist_tol` 默认按模型自身尺度自适应（2026-09-30 发现）：
        测试集样本跨度从 0.017 m 到 626 m。写死 0.02 m 时，
        对归一化坐标的样本（跨度 0.017 m）这个容差比整栋建筑还大，
        会把本不该合并的面也并掉。改成最大跨度的 0.2%。
    """
    m = clean_vertices(mesh)
    if len(m.faces) == 0:
        return m

    if dist_tol is None:
        v = np.asarray(m.vertices, float)
        span = float(np.ptp(v, axis=0).max()) if len(v) else 0.0
        dist_tol = max(0.002 * span, 1e-7) if span > 0 else 0.02

    cos_tol = np.cos(np.radians(angle_tol_deg))
    n = np.asarray(m.face_normals)
    c = np.asarray(m.triangles_center)

    # 并查集：把法向接近且平面距离小的相邻面归为一组
    parent = np.arange(len(m.faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f1, f2 in np.asarray(m.face_adjacency):
        if abs(np.dot(n[f1], n[f2])) < cos_tol:
            continue
        # 两面的平面距离（用面心到对方平面的距离衡量）
        d = abs(np.dot(c[f2] - c[f1], n[f1]))
        if d <= dist_tol:
            union(f1, f2)

    groups = {}
    for i in range(len(m.faces)):
        groups.setdefault(find(i), []).append(i)

    # 对每组做「共面面片合并」——用顶点聚类生成简化面
    new_mesh = m
    if len(groups) < len(m.faces):
        new_mesh = _collapse_groups(m, groups)
    return new_mesh


def _collapse_groups(mesh, groups):
    """把每组共面三角形用其顶点凸包替代（保守做法：保留组内顶点，重建凸包面）"""
    V = np.asarray(mesh.vertices, float)
    new_faces = []
    for gid, fidx in groups.items():
        if len(fidx) <= 1:
            new_faces.append(np.asarray(mesh.faces)[fidx[0]])
            continue
        # 取组内所有顶点
        vids = np.unique(np.asarray(mesh.faces)[fidx].ravel())
        pts = V[vids]
        # 投影到该组主平面，做 2D 凸包
        n = np.mean(np.asarray(mesh.face_normals)[fidx], axis=0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            new_faces.extend(np.asarray(mesh.faces)[fidx])
            continue
        n = n / nn
        # 构造平面内正交基
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        P = np.stack([pts @ u, pts @ v], axis=1)
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(P)
            loop = h.vertices
        except Exception:
            new_faces.extend(np.asarray(mesh.faces)[fidx])
            continue
        if len(loop) < 3:
            new_faces.extend(np.asarray(mesh.faces)[fidx])
            continue
        idx = vids[loop]
        for k in range(1, len(idx) - 1):
            new_faces.append([idx[0], idx[k], idx[k + 1]])

    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, float),
                          faces=np.asarray(new_faces, int), process=False)
    out = clean_vertices(out)
    return out


# ---------------------------------------------------------------- 法向

def orient_outward(mesh):
    """统一法向朝外（NC 指标要求朝向一致）"""
    m = mesh.copy()
    m.remove_unreferenced_vertices()
    try:
        m.fix_normals(multibody=True)
    except Exception:
        # 兜底：按体心判断
        c = m.vertices.mean(axis=0)
        d = ((m.triangles_center - c) * m.face_normals).sum(axis=1)
        f = np.asarray(m.faces).copy()
        flip = d < 0
        if flip.any():
            f[flip] = f[flip][:, ::-1]
        m = trimesh.Trimesh(vertices=m.vertices, faces=f, process=False)
    # 整体体积为负说明法向整体朝内
    if m.volume < 0:
        f = np.asarray(m.faces).copy()[:, ::-1]
        m = trimesh.Trimesh(vertices=m.vertices, faces=f, process=False)
    return m


# ---------------------------------------------------------------- 离群面

def remove_outliers(mesh, max_dist_factor=1.8):
    """
    删除远离主体的面片（地面板、悬挂片、飞点构成的碎片）。

    做法：以包围盒对角线为尺度，丢掉质心距离主体中位面过远的三角形。
    """
    m = clean_vertices(mesh)
    if len(m.faces) < 4:
        return m
    diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
    fc = np.asarray(m.triangles_center)
    med = np.median(fc, axis=0)
    d = np.linalg.norm(fc - med, axis=1)
    keep = d <= max_dist_factor * np.median(d + 1e-9) + 0.15 * diag
    if keep.sum() < 4:
        return m
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return m


# ---------------------------------------------------------------- 规则化

def regularize_xy(mesh, angle_tol_deg=6.0):
    """
    曼哈顿世界假设下的水平方向规则化：
    统计轮廓边的主方向，把接近主方向的边吸附过去。
    对 ECD（边缘 Chamfer）提升最明显。
    """
    m = clean_vertices(mesh)
    V = np.asarray(m.vertices, float).copy()
    E = np.asarray(m.edges_unique, int)
    if len(E) == 0:
        return m

    d = V[E[:, 1]] - V[E[:, 0]]
    horiz = np.abs(d[:, 2]) < 0.15 * (np.abs(d[:, :2]).max() + 1e-9)
    if horiz.sum() < 3:
        return m

    ang = np.arctan2(d[horiz, 1], d[horiz, 0]) % np.pi
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi))
    main = []
    order = np.argsort(hist)[::-1]
    for k in order[:4]:
        c = 0.5 * (edges[k] + edges[k + 1])
        if hist[k] < 0.05 * horiz.sum():
            continue
        if any(min(abs(c - p), np.pi - abs(c - p)) < np.radians(12) for p in main):
            continue
        main.append(c)
    if not main:
        return m

    tol = np.radians(angle_tol_deg)
    changed = False
    for (i, j) in E[horiz]:
        p0, p1 = V[i], V[j]
        a = np.arctan2(p1[1] - p0[1], p1[0] - p0[0]) % np.pi
        for tgt in main:
            diff = a - tgt
            if abs(diff) < tol or abs(abs(diff) - np.pi) < tol:
                mid = 0.5 * (p0 + p1)
                L = 0.5 * np.linalg.norm(p1[:2] - p0[:2])
                dirv = np.array([np.cos(tgt), np.sin(tgt)])
                V[i, :2] = mid[:2] - L * dirv
                V[j, :2] = mid[:2] + L * dirv
                changed = True
                break
    if not changed:
        return m
    out = trimesh.Trimesh(vertices=V, faces=np.asarray(m.faces), process=False)
    return clean_vertices(out)


# ---------------------------------------------------------------- 主入口

def process(mesh, do_outlier=True, do_regular=False, do_coplanar=True,
            do_orient=True, coplanar_tol=3.0, coplanar_dist=None):
    """
    按推荐顺序执行后处理。

    ⚠️ do_regular 默认 **关闭**（2026-09-30 合成集消融实验结论）：
        在自造验证集（12 栋，有真值）上对比各环节：

        配置                     CD      NC     V_Ratio  F_Ratio  综合分
        原始输出（无后处理）      0.949   0.976   6.40     2.85    0.5060
        仅去离群 + 统一法向       0.949   0.976   2.69     2.85    0.5587
        去离群 + 共面合并         0.949   0.976   0.89     0.89    0.6701  ← 最优
        去离群 + 规则化           1.376   0.842   2.69     2.85    0.4494
        全部开启（原默认）        1.368   0.824   2.29     1.89    0.4515

      · `merge_coplanar` 是最大功臣：顶点/面片比从 2.85 压到 0.89，逼近真值，
        同时完全不动几何（CD/NC 不变）。
      · `regularize_xy` 是负收益：它按主方向硬吸附顶点，把 CD 从 0.949 推到 1.376、
        NC 从 0.976 压到 0.842 —— 破坏了原本正确的几何。当前实现过于激进，
        在 ECD 上的收益远抵不上 CD/NC 的损失，故默认关闭。
      · `remove_outliers` 在这批数据上无影响（City3D 输出本来就没有离群面片），
        保留默认开启作为保险。
    """
    m = clean_vertices(mesh)
    if do_outlier:
        m = remove_outliers(m)
    if do_orient:
        m = orient_outward(m)
    if do_regular:
        m = regularize_xy(m)
    if do_coplanar:
        m = merge_coplanar(m, angle_tol_deg=coplanar_tol, dist_tol=coplanar_dist)
    if do_orient:
        m = orient_outward(m)
    m = clean_vertices(m)
    return m
