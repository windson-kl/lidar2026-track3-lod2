# -*- coding: utf-8 -*-
"""构建 v9r：以 v9z 为底，只把「p2m 确实改善」的样本换成 v9s 的版本。

理由（本轮分析）：
  - 官方 CD 里 CD1(pred->GT) 约占 78%，而我们的 p2m 代理**只测得到 CD2(gt->pred)**，
    对 CD1 完全盲 —— 任何改动的 CD1 代价都看不见。
  - v9s 对 167 栋动了手，其中只有 67 栋 p2m 真的改善（合计 -343.2），
    另 100 栋 p2m 变差但幅度极小（合计 +39.7，中位 +0.10 m）。
  - 因此把这 100 栋回退：p2m 收益几乎全保，却把 CD1 暴露面砍掉 60%。
  - 加 margin 后（要求改善 >= 0.05 m）只需改 ~41 栋即可拿到几乎全部收益，
    同时也规避了估计器噪声（未改动样本的 Δp2m 标准差仅 0.0078）。

用法: python src/_t_build_v9r.py --margin 0.05
"""
import argparse
import glob
import json
import os
import shutil
import sys

ROOT = r"D:\LiDAR2026"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", default=os.path.join(ROOT, "logs", "cmp_v9z_v9s_all.json"))
    ap.add_argument("--z", default=os.path.join(ROOT, "data", "work", "final_v9z"))
    ap.add_argument("--s", default=os.path.join(ROOT, "data", "work", "final_v9s"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "work", "final_v9r"))
    ap.add_argument("--margin", type=float, default=0.05,
                    help="要求 p2m 绝对改善至少这么多米才采纳")
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.cmp, encoding="utf-8"))
            if "err" not in r and r.get("a") and r.get("b")]
    keep = []
    for r in rows:
        if r["a"]["F"] == r["b"]["F"]:
            continue
        if (r["a"]["p2m"] - r["b"]["p2m"]) >= a.margin:
            keep.append(r["id"])
    keep.sort()
    print("采纳 %d 栋 (margin=%.2f m)" % (len(keep), a.margin))

    os.makedirs(a.out, exist_ok=True)
    all_ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.z, "*.obj")))
    n_copied = 0
    for i in all_ids:
        src = os.path.join(a.s if i in set(keep) else a.z, i + ".obj")
        dst = os.path.join(a.out, i + ".obj")
        shutil.copyfile(src, dst)
        n_copied += 1
    got = len(glob.glob(os.path.join(a.out, "*.obj")))
    empty = sum(1 for p in glob.glob(os.path.join(a.out, "*.obj")) if os.path.getsize(p) == 0)
    print("写出 %d 个 obj (源 %d)  零字节 %d" % (got, n_copied, empty))

    # 与 v9z 的差异核对
    diff = []
    for i in all_ids:
        pz = os.path.join(a.z, i + ".obj")
        pr = os.path.join(a.out, i + ".obj")
        if os.path.getsize(pz) != os.path.getsize(pr):
            diff.append(i)
        else:
            with open(pz, "rb") as f1, open(pr, "rb") as f2:
                if f1.read() != f2.read():
                    diff.append(i)
    print("与 v9z 不同的样本: %d  (期望 %d)" % (len(diff), len(keep)))
    unexp = sorted(set(diff) - set(keep))
    print("非预期差异: %d %s" % (len(unexp), unexp[:10]))
    with open(os.path.join(ROOT, "logs", "ids_v9r_keep.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(keep) + "\n")
    print("写出 logs/ids_v9r_keep.txt")
    return 0 if (got == len(all_ids) and empty == 0 and not unexp) else 1


if __name__ == "__main__":
    sys.exit(main())
