# -*- coding: utf-8 -*-
"""读取 _t_dsm_var.py 的输出，只统计"门控内"样本，并用线上标定模型给出 FINAL 预测。

线上标定（两点标定，v21/v22 实测反解传递率）：
    v20->v21 (g3,  有 9m 上限): dCD_local=-0.490 dECD_local=-0.773
        -> 线上 CD 3.40181->3.32290 (-0.07891, 传递率 0.161)
           online ECD 5.90812->5.58026 (-0.32786, 传递率 0.424)
    v21->v22 (g3op, 去 9m 上限): dCD_local=-0.628 dECD_local=-0.776
        -> 线上 CD 3.32290->3.29601 (-0.02689, 传递率 0.043)
           online ECD 5.58026->5.69141 (+0.11115, 传递率 -0.143)  <<< 反号！

所以本报告给出两套预测：
    "v21 传递率"（乐观，适用于与 g3 同族的加性改动）
    "保守"（ECD 传递率取 0.30，留出 v22 那种反号风险的一半余量）
用法:
    python src/_t_mm_rep.py --json logs/dsm_mm_all.json --base v21
"""
import argparse
import json
import os

import numpy as np

ROOT = r"D:/LiDAR2026"

# 线上锚点
# 注意：标定式的锚点必须是 **v20（未叠加 DSM 的裸基座）**，
# 因为 ΔCD_local / ΔECD_local 都是相对 raw 基座定义的。
# 校验：g3 的门控内 Δ=(-0.490,-0.773) -> 预测 CD 3.3229 / ECD 5.5804，
#       与 v21 实测 3.32290 / 5.58026 完全吻合。
V20 = dict(CD=3.40181, ECD=5.90812)
V21 = dict(CD=3.32290, ECD=5.58026, FINAL=4.22585)
V22 = dict(CD=3.29601, ECD=5.69141, FINAL=4.25417)
GATE = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--ops", default="")
    a = ap.parse_args()
    rows = json.load(open(os.path.join(ROOT, a.json), encoding="utf-8"))
    clean = [r for r in rows if "err" not in r and r.get("cfg", {}).get("raw")
             and r["cfg"]["raw"]["CD1"] < 50]
    gated = [r for r in clean if r.get("diag", 0) >= a.gate]
    print("总行 %d   可用 %d   门控内(diag>=%.0f) %d" %
          (len(rows), len(clean), a.gate, len(gated)))
    if not gated:
        return 1
    b = dict(CD=np.mean([r["cfg"]["raw"]["CD"] for r in gated]),
             ECD=np.mean([r["cfg"]["raw"]["ECD"] for r in gated]),
             FINAL=np.mean([r["cfg"]["raw"]["FINAL"] for r in gated]))
    print("门控内 base: CD %.4f  ECD %.4f  FINAL %.4f  F %.0f  gtF %.0f\n" %
          (b["CD"], b["ECD"], b["FINAL"],
           np.mean([r["F0"] for r in gated]), np.mean([r["gtF"] for r in gated])))

    ops = a.ops.split(",") if a.ops else \
        sorted({k for r in gated for k in r["cfg"] if k != "raw"})
    print("%-8s %6s %9s %9s %9s | %9s %9s %9s | %8s" %
          ("op", "n", "dCD_L", "dECD_L", "dFNL_L", "pred_opt", "pred_cons", "dFNL_med", "好/坏"))
    out = []
    for op in ops:
        v = [(r["cfg"][op]["CD"] - r["cfg"]["raw"]["CD"],
              r["cfg"][op]["ECD"] - r["cfg"]["raw"]["ECD"],
              r["cfg"][op]["FINAL"] - r["cfg"]["raw"]["FINAL"],
              r["cfg"][op]["F"]) for r in gated if r["cfg"].get(op)]
        if len(v) < 3:
            print("%-8s (样本不足 %d)" % (op, len(v)))
            continue
        A = np.array(v)
        dcd, decd, dfnl = A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean()
        p_opt = 0.6 * (V20["CD"] + 0.161 * dcd) + 0.4 * (V20["ECD"] + 0.424 * decd)
        p_con = 0.6 * (V20["CD"] + 0.161 * dcd) + 0.4 * (V20["ECD"] + 0.300 * decd)
        print("%-8s %6d %+9.3f %+9.3f %+9.3f | %9.4f %9.4f | %+8.3f  %d/%d" %
              (op, len(v), dcd, decd, dfnl, p_opt, p_con, np.median(A[:, 2]),
               int((A[:, 2] < 0).sum()), int((A[:, 2] > 0).sum())))
        out.append((p_opt, op))
    print("\n(锚点 v21 FINAL %.5f；plqueffe 4.1955 / Chuniby0 4.3059)" % V21["FINAL"])
    if out:
        out.sort()
        print("按乐观预测排序: " + " > ".join("%s %.4f" % (o, p) for p, o in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
