"""
输入预处理：把官方 .xyz 点云转成 City3D 可读的 .ply，并做几何体检

官方测试集格式：纯文本三列 "x y z"，空格分隔，坐标以原点为中心、单位为米。
City3D 的 CLI_Example_2 只扫描 .ply 文件，所以需要转换。

同时输出一份体检报告，供参数自适应使用（点数、尺寸、点密度、高度）。

用法：
    python prepare.py --src ../data/raw/test_xyz/LiDAR_xyz \
                      --dst ../data/work/ply \
                      [--shift-z] [--limit 0]
"""
import argparse
import json
import os
import numpy as np


def read_xyz(path):
    """读三列 xyz 文本"""
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return np.ascontiguousarray(a[:, :3], dtype=np.float64)


def write_ply(path, pts):
    """写 ascii PLY 点云"""
    n = len(pts)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, pts, fmt="%.6f")


def profile(pts):
    """几何体检：给参数自适应提供依据"""
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ext = mx - mn
    area = float(ext[0] * ext[1]) + 1e-9
    return {
        "n_points": int(len(pts)),
        "extent_xy": [round(float(ext[0]), 3), round(float(ext[1]), 3)],
        "height": round(float(ext[2]), 3),
        "z_min": round(float(mn[2]), 3),
        "z_max": round(float(mx[2]), 3),
        "density": round(float(len(pts) / area), 2),
        # 推荐参数：点密/建筑大 -> 更细的分辨率与更高的最小点数
        "rec_pixel_size": round(float(np.clip(0.05 * min(ext[0], ext[1]), 0.10, 0.60)), 3),
        "rec_min_points": int(np.clip(0.004 * len(pts), 15, 120)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".xyz 点云目录")
    ap.add_argument("--dst", required=True, help=".ply 输出目录")
    ap.add_argument("--shift-z", action="store_true",
                    help="把最低点平移到 z=0（便于按地平面处理）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".xyz", ".txt", ".asc")))
    if args.limit > 0:
        files = files[:args.limit]

    report, n_ok, n_bad = [], 0, 0
    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        try:
            pts = read_xyz(os.path.join(args.src, fn))
            if len(pts) < 10:
                raise ValueError(f"点数过少 ({len(pts)})")
            if args.shift_z:
                pts = pts - np.array([0.0, 0.0, pts[:, 2].min()])
            write_ply(os.path.join(args.dst, stem + ".ply"), pts)
            r = profile(pts)
            r["id"] = stem
            report.append(r)
            n_ok += 1
        except Exception as ex:
            print(f"  [跳过] {fn}: {ex}")
            n_bad += 1

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i+1}/{len(files)}")

    rp = os.path.join(args.dst, "_profile.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print(f"\n转换完成: 成功 {n_ok}, 失败 {n_bad}")
    print(f"  .ply 输出 : {os.path.abspath(args.dst)}")
    print(f"  体检报告  : {rp}")

    if report:
        ns = np.array([r["n_points"] for r in report])
        dn = np.array([r["density"] for r in report])
        hz = np.array([r["height"] for r in report])
        print(f"\n  点数   : min={ns.min()} 中位={int(np.median(ns))} max={ns.max()}")
        print(f"  密度   : min={dn.min():.1f} 中位={np.median(dn):.1f} max={dn.max():.1f} 点/m2")
        print(f"  高度   : min={hz.min():.2f} 中位={np.median(hz):.2f} max={hz.max():.2f} m")
        print(f"  推荐参数 pixel_size 中位 = "
              f"{np.median([r['rec_pixel_size'] for r in report]):.3f}")
        print(f"  推荐参数 min_points 中位 = "
              f"{int(np.median([r['rec_min_points'] for r in report]))}")


if __name__ == "__main__":
    main()
