"""
合成 LoD2 建筑数据集：程序化生成 GT mesh + 虚拟机载 LiDAR 扫描点云

复刻 BuildingWorld 的构造方式：先有 LoD2 模型，用虚拟激光扫出点云，
于是「点云 + 精确 GT」成对，可在没有官方数据时完整验证整条流水线。

用法：
    python synth.py --out ../data/synthetic --n 24
"""
import argparse
import json
import os
import numpy as np
import trimesh


# ---------------------------------------------------------------- 基础构件

def mesh_from_quads(verts, faces):
    """由顶点表 + 面表构造 mesh，面可以是三角形、四边形或任意 n 边形"""
    tris = []
    for q in faces:
        q = list(q)
        if len(q) == 3:
            tris.append(q)
        else:                                   # n 边形扇形三角化
            for i in range(1, len(q) - 1):
                tris.append([q[0], q[i], q[i + 1]])
    m = trimesh.Trimesh(vertices=np.asarray(verts, float),
                        faces=np.asarray(tris, int), process=False)
    m.fix_normals()
    return m


def prism_rect(x0, y0, x1, y1, z0=0.0, z1=None, h=None):
    """轴对齐矩形棱柱（平屋顶）"""
    if z1 is None:
        z1 = z0 + h
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    q = [[3, 2, 1, 0], [4, 5, 6, 7],
         [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return mesh_from_quads(v, q)


def gable_roof(x0, y0, x1, y1, z0=0.0, z_eave=8.0, z_ridge=12.0, ridge_along="x"):
    """双坡屋顶：两侧山墙 + 两个坡面"""
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z_eave), (x1, y0, z_eave), (x1, y1, z_eave), (x0, y1, z_eave)]
    if ridge_along == "x":
        v += [(x0, cy, z_ridge), (x1, cy, z_ridge)]
        q = [[3, 2, 1, 0],
             [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
             [4, 5, 9, 8], [6, 7, 8, 9],
             [4, 8, 7], [5, 6, 9]]
    else:
        v += [(cx, y0, z_ridge), (cx, y1, z_ridge)]
        q = [[3, 2, 1, 0],
             [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
             [4, 5, 9, 8], [6, 7, 8, 9],
             [4, 8, 5], [7, 6, 9]]
    return mesh_from_quads(v, q)


def hip_roof(x0, y0, x1, y1, z0=0.0, z_eave=8.0, z_ridge=12.0, inset=None):
    """四坡屋顶：两个梯形坡面 + 两个三角形坡面"""
    cy = (y0 + y1) / 2.0
    if inset is None:
        inset = 0.35 * (x1 - x0)
    inset = min(inset, 0.45 * (x1 - x0))
    xa, xb = x0 + inset, x1 - inset
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z_eave), (x1, y0, z_eave), (x1, y1, z_eave), (x0, y1, z_eave),
         (xa, cy, z_ridge), (xb, cy, z_ridge)]
    q = [[3, 2, 1, 0],
         [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
         [4, 5, 9, 8],       # 前坡（梯形）
         [6, 7, 8, 9],       # 后坡（梯形）
         [4, 7, 8],          # 左坡（三角形）
         [5, 6, 9]]          # 右坡（三角形）
    return mesh_from_quads(v, q)


def setback_building(x0, y0, x1, y1, levels):
    """退台建筑：多层矩形棱柱叠加，形成阶梯屋顶"""
    parts = []
    z = 0.0
    for (fx0, fy0, fx1, fy1, dz) in levels:
        parts.append(prism_rect(fx0, fy0, fx1, fy1, z, z + dz))
        z += dz
    m = trimesh.util.concatenate(parts)
    m.fix_normals()
    return m


# ---------------------------------------------------------------- 随机建筑

def random_building(rng):
    """随机生成一栋 LoD2 建筑，返回 (mesh, meta)"""
    kind = rng.choice(["flat", "gable", "hip", "setback"], p=[0.30, 0.30, 0.25, 0.15])
    w = float(rng.uniform(10, 26))
    d = float(rng.uniform(9, 22))
    x0, y0 = 0.0, 0.0
    x1, y1 = w, d

    if kind == "flat":
        h = float(rng.uniform(7, 22))
        m = prism_rect(x0, y0, x1, y1, 0.0, h)
        meta = {"kind": "flat", "height": round(h, 2)}
    elif kind == "gable":
        ze = float(rng.uniform(6, 14))
        zr = ze + float(rng.uniform(2.5, 7))
        m = gable_roof(x0, y0, x1, y1, 0.0, ze, zr,
                       ridge_along=str(rng.choice(["x", "y"])))
        meta = {"kind": "gable", "z_eave": round(ze, 2), "z_ridge": round(zr, 2)}
    elif kind == "hip":
        ze = float(rng.uniform(6, 14))
        zr = ze + float(rng.uniform(2.5, 6))
        m = hip_roof(x0, y0, x1, y1, 0.0, ze, zr,
                     inset=float(rng.uniform(0.15, 0.42) * w))
        meta = {"kind": "hip", "z_eave": round(ze, 2), "z_ridge": round(zr, 2)}
    else:
        h1 = float(rng.uniform(6, 12))
        h2 = float(rng.uniform(3, 9))
        ix0 = x0 + rng.uniform(0.08, 0.3) * w
        ix1 = x1 - rng.uniform(0.08, 0.3) * w
        iy0 = y0 + rng.uniform(0.08, 0.3) * d
        iy1 = y1 - rng.uniform(0.08, 0.3) * d
        m = setback_building(x0, y0, x1, y1, [
            (x0, y0, x1, y1, h1),
            (ix0, iy0, ix1, iy1, h2)])
        meta = {"kind": "setback", "h_lower": round(h1, 2), "h_upper": round(h2, 2)}

    # 随机水平旋转，检验算法对朝向的鲁棒性
    ang = float(rng.uniform(0, 2 * np.pi))
    T = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])
    m.apply_transform(T)

    # 平移到随机位置，模拟真实场景中的不规则布局
    m.apply_translation([rng.uniform(-80, 80), rng.uniform(-80, 80), 0.0])

    meta["rotation_deg"] = round(float(np.degrees(ang)), 2)
    meta["n_vertices"] = int(len(m.vertices))
    meta["n_faces"] = int(len(m.faces))
    meta["bounds"] = np.asarray(m.bounds).round(3).tolist()
    return m, meta


# ---------------------------------------------------------------- 虚拟扫描

def simulate_als(mesh, n_points=25000, wall_keep=0.15,
                 noise=0.015, seed=0):
    """
    模拟机载 LiDAR 扫描。

    关键仿真特性：
      · 面积加权采样 -> 大面片自然获得更多点
      · 朝上的面（屋顶）几乎全保留，竖直墙面大量丢失
        （机载扫描从上往下打，墙面掠射角大，回波很少）
      · 加高斯测距噪声
    """
    rng = np.random.default_rng(seed)
    pts, fid = trimesh.sample.sample_surface(mesh, int(n_points * 4), seed=seed)
    nz = np.abs(np.asarray(mesh.face_normals)[fid][:, 2])

    # 朝上的面保留概率 1.0，墙面按 wall_keep 抽稀
    p_keep = np.where(nz > 0.30, 1.0, wall_keep)
    pts = pts[rng.random(len(pts)) < p_keep]

    if len(pts) > n_points:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    pts = pts + rng.normal(0.0, noise, pts.shape)

    order = np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))
    return pts[order]


def als_quality_report(pts):
    """点云质量体检：密度、高度分层"""
    xy = pts[:, :2]
    ex = float(np.ptp(xy[:, 0]))
    ey = float(np.ptp(xy[:, 1]))
    area = (ex + 1e-9) * (ey + 1e-9)
    return {
        "n_points": int(len(pts)),
        "density_per_m2": round(float(len(pts) / area), 2),
        "z_min": round(float(pts[:, 2].min()), 3),
        "z_max": round(float(pts[:, 2].max()), 3),
        "extent_xy": [round(ex, 2), round(ey, 2)],
    }


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/synthetic")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--points", type=int, default=25000)
    args = ap.parse_args()

    gt_dir = os.path.abspath(os.path.join(args.out, "gt"))
    pc_dir = os.path.abspath(os.path.join(args.out, "pointcloud"))
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifest = []

    for i in range(args.n):
        sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))  # 每栋独立种子
        mesh, meta = random_building(sub)
        seed_i = int(rng.integers(0, 2**31 - 1))
        pts = simulate_als(mesh, n_points=args.points, seed=seed_i)

        name = f"B{i:03d}"
        mesh.export(os.path.join(gt_dir, f"{name}_gt.obj"))
        trimesh.PointCloud(pts).export(os.path.join(pc_dir, f"{name}.ply"))

        meta.update({"id": name})
        meta.update(als_quality_report(pts))
        manifest.append(meta)
        print(f"[{i+1:>3}/{args.n}] {name}  {meta['kind']:<8} "
              f"V={meta['n_vertices']:>4} F={meta['n_faces']:>4} "
              f"pts={meta['n_points']:>6}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    kinds = {}
    for m in manifest:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(f"\n生成完毕：{len(manifest)} 栋")
    print(f"  类型分布: {kinds}")
    print(f"  GT mesh : {gt_dir}")
    print(f"  点云    : {pc_dir}")


if __name__ == "__main__":
    main()
