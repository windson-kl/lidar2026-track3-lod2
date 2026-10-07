# -*- coding: utf-8 -*-
"""bw600 上 walls 参数配置的配对真值裁决（2026-10-05）。

对比 base（bw600/final，后处理输出）vs A（z0=p2, z1=p30，= v10）vs B（z0=p0, z1=p25，= v11）。
只在「干净子集」（logs/bw600_clean_score.json，云/GT 尺度一致）上做配对统计，
避免 1662（FINAL 38 万）把均值彻底带偏。
"""
import json
import os
import sys
import numpy as np

ROOT = r"D:/LiDAR2026"
L = lambda p: os.path.join(ROOT, "logs", p)


def idx(path):
    out = {}
    for r in json.load(open(path, encoding="utf-8")):
        if "err" in r:
            continue
        out[r["id"]] = r
    return out


def main():
    base = idx(L("bw600_final_score.json"))
    A = idx(L("bw600_wA.json"))
    B = idx(L("bw600_wB.json"))
    clean = {r["id"] for r in json.load(open(L("bw600_clean_score.json"), encoding="utf-8"))
             if "err" not in r}
    ids = sorted(i for i in clean if i in base and i in A and i in B)
    print("干净子集交集 %d 栋" % len(ids))
    if not ids:
        return 1

    keys = ["CD", "CD1", "CD2", "ECD", "FINAL"]
    for k in keys:
        b = np.array([base[i][k] for i in ids])
        print("  base %-5s 均值 %10.4f  中位 %9.4f" % (k, b.mean(), np.median(b)))

    print("\n%-28s %10s %10s %10s %10s %8s %8s" %
          ("对比", "ΔCD", "ΔCD1", "ΔCD2", "ΔECD", "ΔFINAL", "改善/变差"))
    for tag, X in (("A(z0=p2,z1=p30)=v10", A), ("B(z0=p0,z1=p25)=v11", B)):
        d = np.array([X[i]["FINAL"] for i in ids]) - np.array([base[i]["FINAL"] for i in ids])
        row = []
        for k in keys[:-1]:
            row.append(np.mean([X[i][k] - base[i][k] for i in ids]))
        print("%-28s %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f  %d/%d"
              % (tag, row[0], row[1], row[2], row[3], d.mean(),
                 int((d < -1e-9).sum()), int((d > 1e-9).sum())))

    # A vs B 直接配对
    dAB = np.array([B[i]["FINAL"] for i in ids]) - np.array([A[i]["FINAL"] for i in ids])
    print("\n=== B - A 直接配对（>0 表示 B 更差）===")
    print("ΔFINAL 均值 %+.4f  中位 %+.4f  改善(B<A) %d / 变差(B>A) %d / 平 %d"
          % (dAB.mean(), np.median(dAB), int((dAB < -1e-9).sum()),
             int((dAB > 1e-9).sum()), int((np.abs(dAB) <= 1e-9).sum())))
    for k in ("CD", "CD1", "CD2", "ECD"):
        dd = np.array([B[i][k] - A[i][k] for i in ids])
        print("  Δ%-5s 均值 %+.4f  中位 %+.4f  B越好 %d / B越差 %d"
              % (k, dd.mean(), np.median(dd), int((dd < -1e-9).sum()), int((dd > 1e-9).sum())))
    # 只看两侧都有效的（>0）
    fin = np.array([A[i]["FINAL"] for i in ids])
    for lo in (0.5, 1.0, 2.0, 5.0):
        m = fin >= lo
        if m.sum() < 3:
            continue
        print("  限定 base(A).FINAL>=%.1f 的 %d 栋: Δ(B-A) 均值 %+.4f 改善 %d / 变差 %d"
              % (lo, m.sum(), dAB[m].mean(), int((dAB[m] < -1e-9).sum()),
                 int((dAB[m] > 1e-9).sum())))
    # 去掉 1662 后（若它在集合里）
    m = np.array([i != "1662" for i in ids])
    print("\n去掉 1662 后 %d 栋: Δ(B-A) 均值 %+.4f  中位 %+.4f  改善 %d / 变差 %d"
          % (m.sum(), dAB[m].mean(), np.median(dAB[m]),
             int((dAB[m] < -1e-9).sum()), int((dAB[m] > 1e-9).sum())))
    print("  base FINAL 均值 %.4f -> A %.4f -> B %.4f (去掉 1662)"
          % (np.mean([base[i]["FINAL"] for i in ids]) if False else np.mean(
              [base[i]["FINAL"] for i in np.array(ids)[m]]),
             np.mean([A[i]["FINAL"] for i in np.array(ids)[m]]),
             np.mean([B[i]["FINAL"] for i in np.array(ids)[m]])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
