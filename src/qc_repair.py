# -*- coding: utf-8 -*-
"""
质检 + 定向补救（2026-10-01）

背景
----
`run_batch.py` 的流水线有三个盲点，全部在真实数据上被实测证实：

1. **兜底网格无法被识别**
   `stage_rescue` 用 `fallback.reconstruct()` 给失败的样本补一个纯几何网格，
   写进与 City3D **同一个文件名后缀**（`_ReconstructedModel.obj`）。
   于是一个 4000 栋的目录里，City3D 真品与兜底品混在一起无从区分。
   唯一可靠的指纹：**City3D 每处理一栋必写 `_GeneratedFootprint.obj`，
   兜底不写**。→ `has_rec and not has_fp` 就是兜底。
   实测（旧 `data/work/recon`）：4000 个 `_ReconstructedModel.obj` 里只有
   425 个 `_GeneratedFootprint.obj`，即 **89% 是兜底**。

2. **"爆炸/塌陷网格"没有任何拦截**
   `_t_maxfaces_sweep` 实测 4559 在 mf=15000 时 CD 退化到 3.08（正常应为 0.02 量级）；
   `realbench_eval_mfvars` 实测 15487 在 mf=2000 时 CD=3.4642。
   这些网格照样被 City3D 判为"成功"，照样进提交包。
   廉价判别式：`ratio = 我方网格包围盒对角线 / 输入点云包围盒对角线`。
   实测健康样本 ratio ∈ [0.937, 1.029]（40 栋全体），
   而灾难样本为 17.74 / 1.654 / 0.616 / 0.542 / 0.195，
   两侧都拉得很开，因此用 [0.85, 1.20] 的带就能干净分开。

3. **失败原因不复用**
   同一份点云，`max_faces` 与 `pixel_size` 换个取值常常就能成功
   （City3D 的 `max_allowed_candidate_faces` 是**分支开关、非单调**：
   候选面 > 上限就退到"剔除检测竖直线"的妥协路径）。
   所以对失败样本应当**先换配置重跑**，全部失败才退回纯几何兜底。

本脚本做这三件事：识别 → 阶梯重跑 → 择优落盘。

用法
----
  # 只看不改
  python qc_repair.py scan --rec-dir <recon> --ply-dir <ply> --out logs/qc.json

  # 识别 + 补救，结果写到新目录
  python qc_repair.py repair --rec-dir <recon> --ply-dir <ply> --out-dir <recon_v6> \
      --jobs 8 --timeout 120
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import trimesh

# ★ 强制 stdout/stderr 走 UTF-8（2026-10-02）
#   本脚本的告警文案里有 `⚠`(U+26A0) 等 cp936 编不出的字符；在中文 Windows 下
#   若 stdout 取 locale 编码，一句提示 print 就会抛 UnicodeEncodeError 并中断整个
#   补救阶段（v7 实测被这个坑崩过一次）。固定为 UTF-8 后与启动方式无关。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c3run

SUF = "_ReconstructedModel.obj"
FP_SUF = "_GeneratedFootprint.obj"


# ------------------------------------------------------------------ 基础量测

def bbox_diag(path):
    """网格/点云的包围盒对角线；读取失败返回 None"""
    try:
        m = trimesh.load(path, process=False)
    except Exception:
        return None
    if isinstance(m, trimesh.Scene):
        try:
            m = m.dump(concatenate=True)
        except Exception:
            return None
    v = np.asarray(getattr(m, "vertices", []), float)
    if v.size == 0:
        return None
    lo, hi = v.min(axis=0), v.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def mesh_stat(path):
    """(包围盒对角线, 顶点数, 面数)"""
    try:
        m = trimesh.load(path, process=False)
    except Exception:
        return None
    if isinstance(m, trimesh.Scene):
        try:
            m = m.dump(concatenate=True)
        except Exception:
            return None
    v = np.asarray(getattr(m, "vertices", []), float)
    f = np.asarray(getattr(m, "faces", []), int)
    if v.size == 0:
        return None
    lo, hi = v.min(axis=0), v.max(axis=0)
    return float(np.linalg.norm(hi - lo)), int(len(v)), int(len(f))


def auto_pixel_size(profile, name, lo=1e-4, hi=0.60):
    """与 run_batch._auto_pixel_size 保持一致的公式 ps = span/130"""
    r = (profile or {}).get(name)
    if not r:
        return None
    span = max(r.get("extent_xy", [0.0, 0.0]))
    if span <= 0:
        return None
    return float(min(max(span / 130.0, lo), hi))


# ------------------------------------------------------------------ 识别

def scan(rec_dir, ply_dir, lo=0.85, hi=1.20, limit=0):
    """
    返回 (rows, profile)。每行：
      id / kind / ratio / nv / nf / pred_diag / pc_diag
    kind:
      ok        真品且比值在带内
      explode   真品但比值出带
      fallback  有重建无 footprint（= rescue 兜底）
      missing   没有重建
    """
    prof_p = os.path.join(ply_dir, "_profile.json")
    profile = {}
    if os.path.exists(prof_p):
        with open(prof_p, "r", encoding="utf-8") as fp:
            profile = {r["id"]: r for r in json.load(fp)}

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ply_dir)
                   if f.lower().endswith(".ply"))
    if limit > 0:
        names = names[:limit]

    rows = []
    for n in names:
        rec = os.path.join(rec_dir, n + SUF)
        fp = os.path.join(rec_dir, n + FP_SUF)
        pc = os.path.join(ply_dir, n + ".ply")
        r = {"id": n, "has_rec": os.path.exists(rec),
             "has_fp": os.path.exists(fp)}
        if not r["has_rec"]:
            r.update(kind="missing", ratio=None, nv=None, nf=None,
                     pred_diag=None, pc_diag=None)
            rows.append(r)
            continue
        st = mesh_stat(rec)
        pd_ = bbox_diag(pc)
        r["pred_diag"] = st[0] if st else None
        r["nv"] = st[1] if st else None
        r["nf"] = st[2] if st else None
        r["pc_diag"] = pd_
        if st and pd_ and pd_ > 0:
            r["ratio"] = st[0] / pd_
        else:
            r["ratio"] = None
        if not r["has_fp"]:
            r["kind"] = "fallback"
        elif r["ratio"] is None or not (lo <= r["ratio"] <= hi):
            r["kind"] = "explode"
        else:
            r["kind"] = "ok"
        rows.append(r)

    return rows, profile


def summarize(rows):
    from collections import Counter
    c = Counter(r["kind"] for r in rows)
    n = len(rows)
    print("  %-10s %6s  %6s" % ("类别", "栋数", "占比"))
    for k in ("ok", "explode", "fallback", "missing"):
        print("  %-10s %6d  %5.2f%%" % (k, c[k], 100.0 * c[k] / max(n, 1)))
    print("  合计       %6d" % n)
    return c


# ------------------------------------------------------------------ 补救

def _ds_in_dir(ply, name, workroot, target):
    """
    造一个**只含这一栋**的降采样输入目录（City3D 的 --input 是目录，
    混入别的样本会把它们的产物算进本栋，早期踩过这个坑）。

    为什么需要降采样档（2026-10-01 实测）：
      `non-simple footprint polygon` 这类失败**对 pixel_size 完全不敏感** ——
      实测 10851 在 ps=0.15/0.30/0.60/1.00/2.00 下全部以同一错误失败。
      唯一有效的手段是**减少点数让轮廓变简单**（v4 里该样本正是靠
      `stage_retry` 的体素降采样救回的）。所以阶梯里必须有降采样档。
    """
    d = os.path.join(workroot, "ds%d" % target, name)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, name + ".ply")
    if not os.path.exists(dst):
        m = trimesh.load(ply, process=False)
        pts = np.asarray(getattr(m, "vertices", []), float)
        if len(pts) <= target:
            shutil.copy2(ply, dst)
        else:
            import run_batch as RB
            RB.write_ply(dst, RB._voxel_downsample(pts, target))
    return d


def _try_one(name, ply, configs, workroot, timeout, lo=0.85, hi=1.20):
    """
    对一栋样本按阶梯依次尝试，**取 ratio 最接近 1 的那一档**，而不是"首个成功"。

    为什么不能取首个成功（2026-10-01 realbench 实测）：
      City3D 的 `max_allowed_candidate_faces` 是分支开关，不同取值会产出
      **质量天差地别**的网格。四个"爆炸"样本上实测：
        id      mf=2000(ratio/CD)   mf=5000(ratio/CD)
        15487   17.74 / 3.4642      0.998 / 0.0474
        19984    0.195 / 0.3798     0.990 / 0.0282
        15677    1.654 / 0.2275     0.992 / 0.0686
        6103     0.542 / 0.1607     0.997 / 0.0264
      → "选 ratio 最接近 1 的变体" 4/4 命中真正最优解。
        只取"首个成功"会 4/4 选到灾难解。

    命中带内即提前退出（绝大多数样本第 1~2 档就命中）。
    """
    pc_diag = bbox_diag(ply)
    tries = []
    in_d = os.path.join(workroot, "in", name)
    os.makedirs(in_d, exist_ok=True)
    tgt = os.path.join(in_d, name + ".ply")
    if not os.path.exists(tgt):
        try:
            os.link(ply, tgt)
        except Exception:
            shutil.copy2(ply, tgt)

    best = None
    for mf, ps, ds in configs:
        src_d = _ds_in_dir(ply, name, workroot, ds) if ds else in_d
        out_d = os.path.join(
            workroot,
            "out_%s_%s_%s" % (mf, ("%.6g" % ps).replace(".", "p"), ds or 0), name)
        os.makedirs(out_d, exist_ok=True)
        outp = os.path.join(out_d, name + SUF)
        if os.path.exists(outp):
            st = mesh_stat(outp)
        else:
            try:
                c3run.run_city3d(src_d, out_d, 40, ps, 0, timeout=timeout,
                                 scip_limit=20, max_faces=mf)
            except Exception as ex:
                tries.append({"mf": mf, "ps": ps, "ds": ds, "err": str(ex)[:120]})
                continue
            st = mesh_stat(outp)
        if st and st[2] > 0:
            ratio = (st[0] / pc_diag) if (pc_diag and pc_diag > 0) else None
            dev = abs(math.log(ratio)) if ratio and ratio > 0 else 9.0
            tries.append({"mf": mf, "ps": ps, "ds": ds, "ok": True,
                          "ratio": ratio, "nf": st[2], "dev": dev})
            if best is None or dev < best["dev"]:
                best = {"ok": True, "mf": mf, "ps": ps, "ds": ds, "ratio": ratio,
                        "nf": st[2], "src": outp, "dev": dev}
            if lo <= (ratio or 0) <= hi:      # 已经落带，无需再试
                break
        else:
            tries.append({"mf": mf, "ps": ps, "ds": ds, "ok": False})
    if best is None:
        return {"ok": False, "src": None, "tries": tries}
    best["tries"] = tries
    return best


def build_configs(profile, name, base_ps, ladder):
    """把阶梯说明（相对量）展开成 (max_faces, pixel_size, ds_target) 列表"""
    ps0 = auto_pixel_size(profile, name) or base_ps
    out = []
    for ent in ladder:
        mf, k = ent[0], ent[1]
        ds = ent[2] if len(ent) > 2 else None
        ps = ps0 if k is None else max(min(ps0 * k, 2.0), 1e-5)
        out.append((mf, ps, ds))
    return out


# 阶梯按「样本类别」分两套 —— 不同失败模式的解药不同，混在一起会让
# 前半段全是无效尝试，白白吃掉超时预算。
#   explode  ：网格几何错乱 → 换 max_faces 分支最有效
#   missing / fallback：City3D 根本没成功 → non_simple 靠降采样，too_many 靠抬高上限
LADDER_EXPLODE = [
    (2000, None, None),      # 与现行的 5000 走不同分支
    (800, None, None),
    (15000, None, None),
    (30000, None, None),
    (5000, 0.5, None),       # 更细栅格
    (5000, None, 2000),      # 降采样
    (5000, None, 8000),
]

LADDER_MISSING = [
    (15000, None, None),     # too_many_candidate_faces 是首要失败原因
    (5000, None, 2000),      # non_simple 只能靠降采样
    (5000, None, 6000),
    (30000, None, None),
    (2000, None, 2000),
    (5000, 0.5, None),
]


def repair(rec_dir, ply_dir, out_dir, rows, profile, jobs=8, timeout=90,
           base_ps=0.15, ladder=None, limit=0, kinds=("fallback", "missing", "explode"),
           lo=0.85, hi=1.20):
    os.makedirs(out_dir, exist_ok=True)
    # 工作目录放在 out_dir **外面** —— out_dir 之后要直接喂给 postproc / 打包，
    # 保持它只有 <id>_ReconstructedModel.obj 一种文件最省心。
    workroot = os.path.join(os.path.dirname(os.path.abspath(out_dir)),
                            "_repair_work_" + os.path.basename(os.path.abspath(out_dir)))
    os.makedirs(workroot, exist_ok=True)

    todo = [r for r in rows if r["kind"] in kinds]
    if limit > 0:
        todo = todo[:limit]
    print("\n  需补救 %d 栋（%s）" % (len(todo), ",".join(kinds)))
    if not todo:
        return []

    t0 = time.time()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {}
        for r in todo:
            n = r["id"]
            ply = os.path.join(ply_dir, n + ".ply")
            lad = ladder or (LADDER_EXPLODE if r["kind"] == "explode"
                             else LADDER_MISSING)
            cfg = build_configs(profile, n, base_ps, lad)
            futs[ex.submit(_try_one, n, ply, cfg, workroot, timeout, lo, hi)] = n
        for fu in as_completed(futs):
            n = futs[fu]
            try:
                res = fu.result()
            except Exception as ex2:
                res = {"ok": False, "src": None, "tries": [{"err": str(ex2)[:120]}]}
            res["id"] = n
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(todo):
                okn = sum(1 for x in results if x.get("ok"))
                print("    %d/%d  已救回 %d  用时 %.0fs"
                      % (done, len(todo), okn, time.time() - t0), flush=True)

    okn = sum(1 for x in results if x.get("ok"))
    print("  补救成功 %d/%d（%.1f%%）  用时 %.0fs"
          % (okn, len(todo), 100.0 * okn / max(len(todo), 1), time.time() - t0))
    return results


def materialize(rec_dir, ply_dir, out_dir, rows, results, prefer_ratio=True):
    """
    把最终网格汇总到 out_dir：
      * 非可疑样本 → 硬链接原文件（不复制，省空间）
      * 可疑样本   → 补救结果与原件比 **ratio 偏差**，谁更接近 1 用谁；
                     补救全失败则保留原件（宁可留兜底，也不能缺文件）
    """
    os.makedirs(out_dir, exist_ok=True)
    okmap = {r["id"]: r for r in results if r.get("ok")}
    rowmap = {r["id"]: r for r in rows}

    n_link = n_new = n_keep = n_drop = 0
    for r in rows:
        n = r["id"]
        dst = os.path.join(out_dir, n + SUF)
        if os.path.exists(dst):
            continue

        orig = os.path.join(rec_dir, n + SUF)
        src, tag = None, "none"
        cand = okmap.get(n)
        if cand is not None:
            od = (rowmap[n] or {}).get("ratio")
            nd = cand.get("ratio")
            better = True
            if prefer_ratio and od and od > 0:
                if nd and nd > 0:
                    better = abs(math.log(nd)) < abs(math.log(od))
                else:
                    better = False
            if better:
                src, tag = cand["src"], "repaired"
            else:
                n_drop += 1
        if src is None and os.path.exists(orig):
            src, tag = orig, "original"
        if src is None:
            continue
        try:
            os.link(src, dst)
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception:
                continue
        n_link += 1
        if tag == "repaired":
            n_new += 1
        else:
            n_keep += 1
    print("  落盘 %d 个（补救产物 %d，沿用原样 %d，补救更差被弃 %d）"
          % (n_link, n_new, n_keep, n_drop))
    return n_link


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan")
    sp.add_argument("--rec-dir", required=True)
    sp.add_argument("--ply-dir", required=True)
    sp.add_argument("--band-lo", type=float, default=0.85)
    sp.add_argument("--band-hi", type=float, default=1.20)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--out", default="")

    rp = sub.add_parser("repair")
    rp.add_argument("--rec-dir", required=True)
    rp.add_argument("--ply-dir", required=True)
    rp.add_argument("--out-dir", required=True)
    rp.add_argument("--band-lo", type=float, default=0.85)
    rp.add_argument("--band-hi", type=float, default=1.20)
    rp.add_argument("--jobs", type=int, default=8)
    rp.add_argument("--timeout", type=float, default=120.0)
    rp.add_argument("--base-ps", type=float, default=0.15)
    rp.add_argument("--limit", type=int, default=0)
    rp.add_argument("--kinds", default="fallback,missing,explode")
    rp.add_argument("--out", default="")

    a = ap.parse_args()

    if a.cmd == "scan":
        rows, _ = scan(a.rec_dir, a.ply_dir, a.band_lo, a.band_hi, a.limit)
        print("\n=== 质检 %s ===" % a.rec_dir)
        summarize(rows)
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as fp:
                json.dump(rows, fp, ensure_ascii=False, indent=1)
            print("  已写入 %s" % a.out)
        return 0

    rows, profile = scan(a.rec_dir, a.ply_dir, a.band_lo, a.band_hi, a.limit)
    print("=== 质检 %s ===" % a.rec_dir)
    summarize(rows)
    kinds = tuple(x.strip() for x in a.kinds.split(",") if x.strip())
    res = repair(a.rec_dir, a.ply_dir, a.out_dir, rows, profile,
                 jobs=a.jobs, timeout=a.timeout, base_ps=a.base_ps,
                 limit=a.limit, kinds=kinds, lo=a.band_lo, hi=a.band_hi)
    materialize(a.rec_dir, a.ply_dir, a.out_dir, rows, res)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fp:
            json.dump({"scanned": len(rows), "repaired": res}, fp,
                      ensure_ascii=False, indent=1)
        print("  已写入 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
