"""
批量评测：把重建结果与真值配对，计算五项指标

用法：
    python evaluate.py --pred ../data/work/final --gt ../data/synthetic/gt \
                       --out ../out/eval [--radar]
"""
import argparse
import glob
import json
import os
import re
import sys
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402


def collect(d, patterns):
    """按 ID 收集文件：返回 {id: path}"""
    out = {}
    for pat in patterns:
        for f in glob.glob(os.path.join(d, pat)):
            b = os.path.basename(f)
            for suf in ("_ReconstructedModel.obj", "_gt.obj", ".obj", ".ply"):
                if b.endswith(suf):
                    b = b[: -len(suf)]
                    break
            out[b] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="../out/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--radar", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    preds = collect(args.pred, ["*.obj", "*.ply"])
    gts = collect(args.gt, ["*.obj", "*.ply"])
    ids = sorted(set(preds) & set(gts))
    if args.limit > 0:
        ids = ids[: args.limit]
    print(f"预测 {len(preds)} 个，真值 {len(gts)} 个，可配对 {len(ids)} 个")
    if not ids:
        print("没有可配对的样本"); return 1

    rows = []
    for i, k in enumerate(ids):
        try:
            p = trimesh.load(preds[k], process=False)
            g = trimesh.load(gts[k], process=False)
            if len(p.faces) == 0 or len(g.faces) == 0:
                continue
            raw, sc, _ = M.evaluate(p, g)
            rows.append(dict(id=k, **{f"raw_{a}": b for a, b in raw.items()},
                             **{f"sc_{a}": b for a, b in sc.items()},
                             pred_V=len(p.vertices), pred_F=len(p.faces),
                             gt_V=len(g.vertices), gt_F=len(g.faces)))
        except Exception as ex:
            print(f"  [跳过] {k}: {ex}")
        if (i + 1) % 5 == 0:
            print(f"  已评测 {i+1}/{len(ids)}")

    if not rows:
        print("全部评测失败"); return 1

    print(f"\n{'ID':<10}{'CD':>8}{'ECD':>8}{'NC':>7}"
          f"{'V_R':>7}{'F_R':>7}{'综合':>8}")
    print("-" * 58)
    for r in rows:
        print(f"{r['id']:<10}{r['raw_CD']:>8.3f}{r['raw_ECD']:>8.3f}"
              f"{r['raw_NC']:>7.3f}{r['raw_V_Ratio']:>7.2f}"
              f"{r['raw_F_Ratio']:>7.2f}{r['sc_OVERALL']:>8.4f}")

    keys = ["CD", "ECD", "NC", "V_Ratio", "F_Ratio"]
    mean_sc = {f"mean_sc_{k}": float(np.mean([r[f"sc_{k}"] for r in rows])) for k in keys}
    mean_raw = {f"mean_raw_{k}": float(np.mean([r[f"raw_{k}"] for r in rows])) for k in keys}
    overall = float(np.mean([r["sc_OVERALL"] for r in rows]))

    print("-" * 58)
    print("均值  raw: " + "  ".join(f"{k}={mean_raw[f'mean_raw_{k}']:.3f}" for k in keys))
    print("均值  score: " + "  ".join(f"{k}={mean_sc[f'mean_sc_{k}']:.3f}" for k in keys))
    print(f"综合分 OVERALL = {overall:.4f}")

    summary = dict(n=len(rows), overall=overall, **mean_raw, **mean_sc,
                   rows=rows)
    with open(os.path.join(args.out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n结果已保存: {os.path.join(args.out, 'eval.json')}")

    if args.radar:
        import viz
        sc = {k: mean_sc[f"mean_sc_{k}"] for k in keys}
        sc["OVERALL"] = overall
        viz.viz_radar(sc, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
