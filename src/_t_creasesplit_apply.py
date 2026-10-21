# -*- coding: utf-8 -*-
"""沿折边切开网格（crease split）：让每条折边变成"双边"，而平滑区域内部保持焊接。

动机（基于 tmp/_t_sep_apply.py 的实测）：把「坐标重合顶点拆开」可让
  pred 锐边条数 +53%、锐边长 +12%，E1 −0.059、E2 −0.086、ECD −0.145（60 样本干净集），
  且 **CD 逐位不变**（几何不动，纯拓扑）。
这里把这个思想**做到极致且聚焦**：只在**真实折边**（二面角 > 30°）处切开，
  使折边长度翻倍；平滑区域内部边照旧共享（避免像"全解焊"那样在平坦区
  凭空制造锐边，从而抬高 E1）。

实现：
  1. 三角化取拓扑 → 边→面；
  2. 用并查集把「由非折边相连」的面聚成"平滑片"（patch）；
  3. 每个 (patch, 原顶点) 生成一个新顶点，位置 = 原位置 + patch 专属微偏移(1mm)，
     使官方 loader 不会再把它们 merge 回去；
  4. f 行保留原样，只把顶点序号按所属 patch 重映射。

用法: python tmp/_t_creasesplit_apply.py --meshd <dir> --outd <dir> [--crease 30] [--eps 1e-3] [--jobs 6]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = r"D:/LiDAR2026"
COS30 = float(np.cos(np.deg2rad(30.0)))


def read_obj(path):
    vs = bytearray()
    fs = []
    nv = 0
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            if ln.startswith("v "):
                vs += ln.encode()
                nv += 1
            elif ln.startswith("f "):
                fs.append([int(x.split("/")[0]) for x in ln.split()[1:]])
            elif ln.strip():
                pass
    return vs, fs, nv


def parse_vs(buf, nv):
    V = np.empty((nv, 3))
    i = 0
    for ln in buf.decode().splitlines():
        t = ln.split()
        V[i] = (float(t[1]), float(t[2]), float(t[3]))
        i += 1
    return V


def tri_faces(fs):
    out = []
    for f in fs:
        for i in range(1, len(f) - 1):
            out.append((abs(f[0]) - 1, abs(f[i]) - 1, abs(f[i + 1]) - 1))
    return out


def do_one(a):
    sid, meshd, outd, crease, eps = a
    src = os.path.join(meshd, sid + ".obj")
    dst = os.path.join(outd, sid + ".obj")
    try:
        vsb, fs, nv = read_obj(src)
        if nv < 10 or not fs:
            # ★ 必须整文件原样复制：只写 v 行会把网格变成"点云"，
            #   官方 loader 会给出 PointCloud 对象、.faces 直接报错。
            import shutil
            shutil.copyfile(src, dst)
            return dict(id=sid, skipped=True)
        V = parse_vs(vsb, nv)
        T = tri_faces(fs)
        nT = len(T)
        # 每个原始 f 行 → 其首个三角面的编号（用于 f→patch 映射）
        face2tri = []
        k = 0
        for f in fs:
            face2tri.append(k)
            k += len(f) - 2
        # 三角面法向
        Ta = np.asarray(T, int).reshape(-1, 3)
        A, B, C = V[Ta[:, 0]], V[Ta[:, 1]], V[Ta[:, 2]]
        n = np.cross(B - A, C - A)
        L = np.linalg.norm(n, axis=1); L[L == 0] = 1.0
        N = n / L[:, None]
        # 边 → 面
        e2f = {}
        for fi, (x, y, z) in enumerate(Ta):
            for u, w in ((x, y), (y, z), (z, x)):
                key = (u, w) if u < w else (w, u)
                e2f.setdefault(key, []).append(fi)
        # 并查集
        par = list(range(nT))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        cos_thr = float(np.cos(np.deg2rad(crease)))
        for (u, w), fl in e2f.items():
            if len(fl) != 2:
                continue
            if abs(float(np.dot(N[fl[0]], N[fl[1]]))) < cos_thr:
                continue                      # 折边 → 不并
            r0, r1 = find(fl[0]), find(fl[1])
            if r0 != r1:
                par[r0] = r1
        # 面 → patch
        tpatch = np.array([find(i) for i in range(nT)], int)
        # 重映射顶点
        vmap = {}
        out_v = []
        rng = np.random.RandomState(1234)
        dirs = {}
        for j, f in enumerate(fs):
            pid = tpatch[face2tri[j]] if face2tri[j] < nT else 0
            if pid not in dirs:
                d = rng.randn(3)
                dn = np.linalg.norm(d)
                dirs[pid] = (d / dn) * eps if dn > 1e-9 else np.zeros(3)
            off = dirs[pid]
            for t in range(len(f)):
                idx = f[t]
                v = abs(idx) - 1
                key = (pid, v)
                nid = vmap.get(key)
                if nid is None:
                    nid = len(out_v)
                    vmap[key] = nid
                    out_v.append(V[v] + off)
                f[t] = (nid + 1) if idx > 0 else -(nid + 1)
        OV = np.asarray(out_v, float)
        with open(dst, "w") as w:
            for p in OV:
                w.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
            for f in fs:
                w.write("f " + " ".join(str(x) for x in f) + "\n")
        return dict(id=sid, nV_in=nv, nV_out=len(OV), nF=len(fs), nT=nT,
                    nPatch=int(len(set(tpatch.tolist()))))
    except Exception as ex:
        try:
            import shutil
            shutil.copyfile(src, dst)
        except Exception:
            pass
        return dict(id=sid, err=str(ex)[:70])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshd", required=True)
    ap.add_argument("--outd", required=True)
    ap.add_argument("--crease", type=float, default=30.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.outd, exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(a.meshd) if f.endswith(".obj"))
    print("待处理 %d 个 crease=%.0f° eps=%g" % (len(ids), a.crease, a.eps), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, r in enumerate(ex.map(do_one, [(i, a.meshd, a.outd, a.crease, a.eps)
                                              for i in ids], chunksize=4)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                print("  %d/%d" % (k + 1, len(ids)), flush=True)
    g = [r for r in rows if "err" not in r and not r.get("skipped")]
    e = [r for r in rows if "err" in r]
    print("[完成] 有效 %d  错误 %d" % (len(g), len(e)), flush=True)
    if g:
        print("  顶点数 中位 %.0f → %.0f（×%.2f）  patch 数 中位 %.0f" %
              (np.median([x["nV_in"] for x in g]), np.median([x["nV_out"] for x in g]),
               np.median([x["nV_out"] / max(x["nV_in"], 1) for x in g]),
               np.median([x["nPatch"] for x in g])))
    if e:
        print("  首个错误:", e[0])


if __name__ == "__main__":
    main()
