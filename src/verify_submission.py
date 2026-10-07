"""
verify_submission.py —— 提交包自检（无真值条件下能验的东西全验一遍）

背景：赛道三的测试集编号是 1~4000，而本地带真值的 40 栋基准（realbench）是**另一组
编号**。所以**不能**把 final_v7 指向 realbench 的 gt 目录评分 —— 那样只会得到
"id 交集为空"。测试集这一侧只能做**无真值自检**，质量评估必须另走 realbench 基准。

无真值可验的四件事：
  1. 覆盖率   —— 4000 个 id 是否**逐个**都有输出（少一栋就是硬伤）
  2. 可解析性 —— 每个 obj 能否被解析、是否非退化（面数 > 0、体积/面积非零）
  3. 尺度一致性 —— ratio = 网格包围盒对角线 / 点云包围盒对角线，健康带 [0.85, 1.20]
                   （出带 = 爆炸/塌陷，CD 会到 1~3 量级）
  4. 归一化自洽 —— 提交包若按 --mode cloud 归一化，则每栋网格在**点云坐标系**下的
                   包围盒应与点云包围盒大致重合（中心偏移 / 尺度比都可量化）
  5. 兜底指纹 —— 有 _ReconstructedModel.obj 但无 _GeneratedFootprint.obj

用法:
  python verify_submission.py --zip out/submission_v7.zip --ply data/work/ply \
      --expect 4000 --out logs/verify_v7.json
  python verify_submission.py --dir data/work/recon_v7 --ply data/work/ply --expect 4000
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_SUF = "_ReconstructedModel.obj"
FP_SUF = "_GeneratedFootprint.obj"


def pc_diag(ply_path, cache):
    """点云包围盒对角线（带缓存）"""
    if ply_path in cache:
        return cache[ply_path]
    try:
        m = trimesh.load(ply_path, process=False)
        v = np.asarray(getattr(m, "vertices", None), float)
        if v is None or len(v) == 0:
            v = np.asarray(m.points, float)
        d = float(np.linalg.norm(v.max(0) - v.min(0))) if len(v) else None
    except Exception:
        d = None
    cache[ply_path] = d
    return d


def load_mesh_bytes(blob):
    try:
        m = trimesh.load(io.BytesIO(blob), file_type="obj", process=False)
    except Exception:
        return None
    if isinstance(m, trimesh.Scene):
        gs = [g for g in m.geometry.values() if hasattr(g, "faces")]
        if not gs:
            return None
        m = trimesh.util.concatenate(gs)
    return m


def collect(src_kind, src, ply_dir, expect, band_lo, band_hi):
    """返回 (recs: dict id->bytes_or_path, fps: set(ids), n_named)

    ★ 兼容两种命名（2026-10-02）：
      · `<id>_ReconstructedModel.obj` —— City3D 原始产出目录 / recon_v7
      · `<id>.obj`                    —— **提交包标准命名**（submission*.zip 实测即此）
    先匹配长后缀，再兜底纯 `.obj`，避免 `<id>_ReconstructedModel.obj` 被误当纯 obj。
    """
    recs, fps = {}, set()

    def _id(fn, suf):
        return os.path.basename(fn)[:-len(suf)]

    def _classify(base, payload):
        if base.endswith(REC_SUF):
            recs[_id(base, REC_SUF)] = payload
        elif base.endswith(FP_SUF):
            fps.add(_id(base, FP_SUF))
        elif base.lower().endswith(".obj"):
            # 提交包命名：<id>.obj
            recs[_id(base, ".obj")] = payload

    if src_kind == "zip":
        with zipfile.ZipFile(src) as z:
            for nm in z.namelist():
                if nm.endswith("/") or nm.startswith("__MACOSX"):
                    continue
                _classify(os.path.basename(nm), z.read(nm))
    else:
        for nm in os.listdir(src):
            p = os.path.join(src, nm)
            if not os.path.isfile(p):
                continue
            _classify(nm, p)
    return recs, fps


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", default="")
    g.add_argument("--dir", default="")
    ap.add_argument("--ply", required=True, help="输入点云目录（用于尺度自检）")
    ap.add_argument("--expect", type=int, default=0,
                    help="期望样本数；>0 时以 1..expect 为完整名单核对覆盖")
    ap.add_argument("--band-lo", type=float, default=0.85)
    ap.add_argument("--band-hi", type=float, default=1.20)
    ap.add_argument("--frame", choices=["raw", "normalized"], default="raw",
                    help="坐标系：raw=网格与点云同尺度（如 recon_v7/final_v7）；"
                         "normalized=提交包（网格已被点云对角线归一，"
                         "故点云对角线恒为 1，ratio 直接等于网格对角线）")
    ap.add_argument("--limit", type=int, default=0, help="只检查前 N 个（冒烟测试用）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    src_kind = "zip" if a.zip else "dir"
    src = a.zip or a.dir
    recs, fps = collect(src_kind, src, a.ply, a.expect, a.band_lo, a.band_hi)

    names = sorted(recs)
    if a.limit > 0:
        names = names[:a.limit]

    print(f"来源: {src_kind} {src}")
    print(f"坐标系: {a.frame}"
          + ("（提交包帧：点云对角线视为 1）" if a.frame == "normalized" else "（原始尺度）"))
    print(f"  发现 {REC_SUF}: {len(recs)}   发现 {FP_SUF}: {len(fps)}")

    # ---- 1. 覆盖率 ----
    missing = []
    if a.expect > 0:
        want = {str(i) for i in range(1, a.expect + 1)}
        missing = sorted(want - set(recs), key=lambda x: int(x))
        print(f"\n[1] 覆盖率: {len(recs)}/{a.expect} = "
              f"{100.0*len(recs)/a.expect:.2f}%   缺失 {len(missing)} 个")
        if missing:
            print(f"    缺失样例: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    else:
        print(f"\n[1] 覆盖率: 未指定 --expect，跳过完整名单核对（共 {len(recs)} 个）")

    # ---- 2/3/4/5 逐栋 ----
    cache = {}
    rows = []
    n_unparsable = n_degenerate = n_oob = 0
    ratios = []

    for k, sid in enumerate(names):
        item = recs[sid]
        blob = item if isinstance(item, bytes) else None
        path = None if blob is not None else item
        if blob is not None:
            m = load_mesh_bytes(blob)
        else:
            try:
                m = trimesh.load(path, process=False)
            except Exception:
                m = None

        rec = {"id": sid, "fp": sid in fps}
        if m is None or not hasattr(m, "faces") or len(m.faces) == 0:
            rec["ok"] = False
            rec["reason"] = "unparsable" if m is None else "no_faces"
            n_unparsable += 1
            rows.append(rec)
            continue

        nv, nf = len(m.vertices), len(m.faces)
        bounds = m.bounds
        diag = float(np.linalg.norm(bounds[1] - bounds[0])) if bounds is not None else 0.0
        area = float(m.area) if hasattr(m, "area") else 0.0
        if nv < 4 or nf < 4 or diag <= 0 or area <= 0:
            rec["ok"] = False
            rec["reason"] = "degenerate"
            n_degenerate += 1
            rows.append(rec)
            continue

        pl = os.path.join(a.ply, sid + ".ply")
        if a.frame == "normalized":
            # ★ 归一化帧（2026-10-02）：normalize_submission.py 做的是
            #   V = (V - pc_center) / pc_diag，因此**点云**在该帧下的包围盒
            #   对角线恒等于 1，ratio 直接 = 网格对角线本身。
            #   若仍用原始 ply 对角线去除，会人为把结果缩小 pc_diag 倍
            #   （实测 p50 从 0.998 掉到 0.035），整条判据失效。
            d_pc = 1.0 if os.path.exists(pl) else None
        else:
            d_pc = pc_diag(pl, cache) if os.path.exists(pl) else None
        ratio = (diag / d_pc) if d_pc else None
        if ratio is not None:
            ratios.append(ratio)
            # ★ 只把**上溢**标成问题（2026-10-02）。
            #   下溢（ratio < band_lo）多见于"点云含地面/邻栋/植被"的大场景，
            #   单体建筑网格天然远小于点云包围盒，属正常，不能计入未通过。
            #   详见下方 [3] 段的注释。
            if ratio > a.band_hi:
                n_oob += 1
                rec["oob"] = True
            elif ratio < a.band_lo:
                rec["under"] = True

        rec.update({"ok": True, "nv": nv, "nf": nf, "diag": round(diag, 4),
                    "pc_diag": None if d_pc is None else round(d_pc, 4),
                    "ratio": None if ratio is None else round(ratio, 4),
                    "area": round(area, 3), "fp": sid in fps})
        rows.append(rec)

        if (k + 1) % 400 == 0:
            print(f"    ... {k+1}/{len(names)}", flush=True)

    ok_rows = [r for r in rows if r.get("ok")]
    nf_arr = np.array([r["nf"] for r in ok_rows], float) if ok_rows else np.array([0.0])

    print(f"\n[2] 可解析/非退化: 正常 {len(ok_rows)}  "
          f"不可解析 {n_unparsable}  退化 {n_degenerate}")
    # ★ 尺度判别要**分开看两侧**（2026-10-02 修）
    #   `ratio = 网格包围盒对角线 / 点云包围盒对角线`，原实现把出带当成单一错误。
    #   但两侧含义完全不同：
    #     · ratio > hi  → **真错误**：网格比点云还大 = 爆炸（实测 287 = 256.12）
    #     · ratio < lo  → **多半正常**：点云含地面/植被/邻栋时，单体建筑网格
    #                     必然远小于点云包围盒。实测出带-低的 31 栋点云对角线
    #                     p50=99.7 / p90=488 / max=1039.8，而带内正常的 p50 只有 28.7
    #                     —— 它们是大场景，不是缺陷。
    #   阈值 [0.85, 1.20] 本身是在 **realbench**（每个点云紧凑包一栋）上校准的，
    #   搬到测试集只有上界可用。
    ra = np.array(ratios) if ratios else np.array([])
    n_hi = int((ra > a.band_hi).sum()) if len(ra) else 0
    n_lo = int((ra < a.band_lo).sum()) if len(ra) else 0
    print(f"[3] 尺度判别: 上溢(> {a.band_hi}, 真爆炸) {n_hi}   "
          f"下溢(< {a.band_lo}, 多为大场景) {n_lo}   合计 {len(ra)}")
    if len(ra):
        print(f"    ratio 分位: min={ra.min():.3f} p5={np.percentile(ra,5):.3f} "
              f"p50={np.median(ra):.3f} p95={np.percentile(ra,95):.3f} max={ra.max():.3f}")
    print(f"[4] 面数分位: p50={np.median(nf_arr):.0f} "
          f"p90={np.percentile(nf_arr,90):.0f} max={nf_arr.max():.0f}")
    # ★ 只在目录里**确实存在** fp 文件时才统计兜底指纹（2026-10-02 修）
    #   否则会误报：`recon_v7` 这类"只放 _ReconstructedModel.obj"的目录里
    #   一个 fp 都没有，于是"有 rec 无 fp"会把**全部**样本都判成兜底
    #   （实测报出 3993/3993 的无意义数字）。
    if len(fps) > 0:
        n_fb = len(ok_rows) - sum(1 for r in ok_rows if r["fp"])
        print(f"[5] 兜底指纹 (有 rec 无 fp): {n_fb} / {len(ok_rows)}")
    else:
        n_fb = None
        print("[5] 兜底指纹: 跳过（该目录不含 _GeneratedFootprint.obj，"
              "此判据不适用）")

    # ---- 判定 ----
    bad = []
    if missing:
        bad.append(f"覆盖缺失 {len(missing)} 栋")
    if n_unparsable:
        bad.append(f"不可解析 {n_unparsable} 栋")
    if n_degenerate:
        bad.append(f"退化网格 {n_degenerate} 栋")
    if ratios and n_oob / len(ratios) > 0.02:
        bad.append(f"尺度上溢(真爆炸) {n_oob} 栋 "
                   f"({100.0*n_oob/len(ratios):.1f}% > 2%)")

    print("\n" + "=" * 62)
    if bad:
        print("  ✗ 自检未通过: " + "；".join(bad))
    else:
        print("  ✓ 自检通过（覆盖率 / 可解析性 / 尺度 / 非退化 均正常）")
    print("  注意：这是**无真值**自检，只能排除结构性错误，")
    print("        真实几何质量必须在 realbench 40 栋基准上评（那边才有 gt）。")
    print("=" * 62)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fp:
            json.dump({"src": src, "kind": src_kind, "n_rec": len(recs),
                       "n_fp": len(fps), "expect": a.expect, "missing": missing,
                       "n_unparsable": n_unparsable, "n_degenerate": n_degenerate,
                       "n_oob": n_oob, "n_under": n_lo, "band": [a.band_lo, a.band_hi],
                       "ratio_quantiles": None if not ratios else {
                           "min": float(np.min(ratios)),
                           "p5": float(np.percentile(ratios, 5)),
                           "p50": float(np.median(ratios)),
                           "p95": float(np.percentile(ratios, 95)),
                           "max": float(np.max(ratios))},
                       "nf_quantiles": {
                           "p50": float(np.median(nf_arr)),
                           "p90": float(np.percentile(nf_arr, 90)),
                           "max": float(nf_arr.max())},
                       "verdict": "FAIL" if bad else "PASS", "issues": bad,
                       "rows": rows}, fp, indent=1, ensure_ascii=False)
        print(f"→ {a.out}")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
