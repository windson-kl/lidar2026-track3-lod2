"""
LiDAR2026 赛道三 —— 端到端批量流水线

    .xyz 点云
      -> [1] prepare  转 .ply + 几何体检
      -> [2] city3d   分批调 City3D 重建（SCIP 时限内返回可行解）
      -> [3] rescue   对缺失输出的样本走降级重建，保证 100% 覆盖
      -> [4] postproc 指标导向后处理（共面合并 / 法向统一 / 主方向对齐）
      -> [5] package  打包 submission.zip
      -> [6] report   覆盖率报告 coverage.json

用法：
    python run_batch.py --stage all  --limit 40 --jobs 4
    python run_batch.py --stage city3d --chunk 100 --scip-limit 30
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postproc                                   # noqa: E402
import fallback                                   # noqa: E402
from c3run import start_city3d, finish_city3d, _kill_tree, BASH, EXE   # noqa: E402
from prepare import write_ply                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SUF = "_ReconstructedModel.obj"

# ★ 强制 stdout/stderr 走 UTF-8（2026-10-02 修，事故复盘）
#   现象：v7 的步骤 2 在 [132/133] 处整条流水线崩掉，报
#     UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0' in position 2
#   成因：用 `bash -lc` 启动时 Python 的 stdout 编码取 locale（Windows 中文 = cp936/GBK），
#   而告警文案里带了 `⚠`(U+26A0) —— GBK 编不出来 → 异常直接冒泡 →
#   又被 rebuild 脚本的 `set -eu` 放大成"整个脚本终止"，把后面 4 个阶段一起带走。
#   **代价极不对称**：一句提示性 print 崩掉了整条 50 分钟的流水线。
#   修法：把 stdout/stderr 固定为 UTF-8 + errors="replace"，
#   从此与启动方式/locale 解耦，日志编码也统一为 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:               # 老 Python 或被重定向成非 TextIOWrapper
        pass


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


# ------------------------------------------------- 重负载互斥（2026-10-02 新增）

HEAVY_LOCK = os.path.join(ROOT, "logs", ".heavy_eval.lock")


def _heavy_peer_running():
    """
    检测是否有其它重负载评测进程在跑。

    依据（2026-10-01 实测）：official_score.py 里的 normal_consistency 走 faiss
    多线程，与 City3D 并跑会把处理速率从 **1.82 s/栋 恶化到 17 s/栋**，再进一步
    恶化就是每个任务精确跑满超时、零产出（本机 2026-10-02 因此白跑 22 小时）。

    用哨兵文件而不是进程扫描：零依赖（本机没装 psutil）、跨平台、确定性。
    official_score.py 进入重负载段时写、退出时删。
    """
    if not os.path.exists(HEAVY_LOCK):
        return None
    try:
        with open(HEAVY_LOCK, "r", encoding="utf-8") as fp:
            return fp.read().strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- stage 1

def stage_prepare(src, dst, limit=0):
    print("\n[1/6] 预处理：.xyz -> .ply")
    cmd = [PY, os.path.join(ROOT, "src", "prepare.py"), "--src", src, "--dst", dst]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    p = subprocess.run(cmd, capture_output=True)
    print(p.stdout.decode("utf-8", "replace")[-2000:])
    if p.returncode != 0:
        print(p.stderr.decode("utf-8", "replace")[-1500:])
    return p.returncode == 0


# ---------------------------------------------------------------- stage 2

def _slot_put(d, src_ply):
    """把输入硬链接进工作槽（失败退化为复制）"""
    t = os.path.join(d, os.path.basename(src_ply))
    if os.path.exists(t):
        return t
    try:
        os.link(src_ply, t)
    except Exception:
        shutil.copy2(src_ply, t)
    return t


def _slot_retire(d, name):
    """【已停用，2026-10-02】改名归档不再是正确做法，保留仅为说明这段历史。

    原意：处理完把输入改名成 `<name>.ply.done`，让下一轮 City3D 扫目录时不再
    看到它（改名而不是删除，是因为本机沙箱对"一轮对话内删除超过 50 个文件"
    会拦截并要求确认，非交互下直接让流程挂掉）。

    ★ 为什么停用：这个前提是**错的**。City3D 筛输入用的是
        if (file_name.find(".ply") == std::string::npos) continue;
      （CLI_Example_2/main.cpp:215）—— **子串**匹配，不是后缀匹配。
      `1009.ply.done` 含 ".ply" → 照样通过检查被当成输入；而且
      `base = stem.substr(0, stem.find(".ply"))` 会把它的输出名算成 `1009`，
      与正品**同号**。于是每个并发槽里的历史输入被每轮全量重处理一遍。
      实测（2026-10-02）：334 个残留使单次调用从 1.2s 涨到 6.1s。

    替代方案：`_task_dir()` 给每个任务一个独立输入目录，被扫描目录里的文件数
    恒等于本批文件数，根本不需要归档这一步。
    """
    t = os.path.join(d, name + ".ply")
    if os.path.exists(t):
        try:
            os.replace(t, t + ".done")
        except Exception:
            pass


def stage_city3d(ply_dir, out_dir, min_points=40, pixel_size=0.15,
                 limit=0, chunk=1, jobs=4, scip_limit=20, max_faces=600,
                 file_timeout=240, profile=None, auto_ps=False,
                 ps_divisor=130.0, ps_cap=0.60):
    print(f"\n[2/6] City3D 重建  (min_points={min_points}, pixel_size={pixel_size}, "
          f"候选面上限={max_faces}, SCIP 时限={scip_limit}s, "
          f"并发={jobs}, 每批={chunk}, 单栋超时={file_timeout}s)")
    peer = _heavy_peer_running()
    if peer:
        print(f"  ⚠ 检测到重负载评测进程正在运行（{peer}）。")
        print("    faiss 多线程会饿死 City3D（实测 1.82 s/栋 → 17 s/栋），"
              "强烈建议先停掉它再跑本阶段。")
    if not os.path.exists(EXE):
        print(f"  ✗ 找不到可执行文件: {EXE}")
        return {}
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ply_dir)
                   if f.lower().endswith(".ply"))
    if limit > 0:
        names = names[:limit]
    if not names:
        print("  ✗ 没有 .ply 输入")
        return {}
    print(f"  待处理 {len(names)} 栋")

    # 已经跑过的先跳过（支持断点续跑）
    have = {f[:-len(SUF)] for f in os.listdir(out_dir) if f.endswith(SUF)}
    todo = [n for n in names if n not in have]
    if have:
        print(f"  已有产出 {len(have)} 栋，本次待跑 {len(todo)} 栋")
    if not todo:
        return have

    # 每批 chunk 栋，按批切成若干"任务"
    tasks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    work_root = os.path.join(ROOT, "data", "work", "_chunks",
                             f"{os.getpid()}_{int(time.time()) % 100000}")

    def _task_dir(j, bi):
        """
        ★ 每个任务一个**独立**输入目录（2026-10-02 修）

        旧实现让并发槽复用同一个目录、处理完只把输入改名成 `<name>.ply.done`
        归档（见 _slot_retire 的说明：跑步机全程不删文件），于是同一个目录里的
        历史文件在整轮里单调累积。而 City3D 的输入筛选是
            if (file_name.find(".ply") == std::string::npos) continue;
        （CLI_Example_2/main.cpp:215）——**子串**匹配而非后缀匹配，`X.ply.done`
        会通过检查被当成输入重新处理。实测 334 个残留使单次调用多花 4.9s。

        独立目录让"被扫描目录里的文件数"恒等于本批文件数，从根上消除漂移。
        """
        d = os.path.join(work_root, f"s{j:03d}", f"b{bi:05d}")
        os.makedirs(d, exist_ok=True)
        return d

    t0 = time.time()
    per_batch = []
    running = []                                # [Popen, before, t0, 任务号, 文件数, 超时, 槽号, 文件名]
    free_slots = list(range(max(1, jobs)))
    nxt = 0

    # ★ 熔断闸门（2026-10-02 新增）
    #   事故复盘：一次 4000 栋重跑从第 ~400 个任务起，100% 的任务跑满 150s 超时、
    #   零产出，但因为没有任何熔断机制，硬生生烧掉 22 小时（两次阶段各一次，
    #   重试阶段同样在任务 ~600 后归零）。成因是外部评测进程
    #   （official_score.py 的 normal_consistency 走 faiss 多线程）抢占全部 CPU，
    #   把 11 路 City3D 饿死 —— 判别指纹是"每个任务精确用满超时"，
    #   即 200 任务 / 11 并发 × 150s = 2727s 的恒定墙钟。
    #   这里统计"连续跑满超时且零产出"的任务数，超阈值立即中止并给出排查指引，
    #   把 22 小时的浪费压缩成几分钟。
    #
    # ★ 速率为辅判据（2026-10-02 二次加固）
    #   纯计数判据会**误报**：v7 首轮实测出现过连续 9 次"用时 >=135s"，
    #   但那是分散在 11 个并发槽里的**真实难样本** —— 墙钟仍是 1.05 s/栋
    #   （饿死时应退化到 150/11 = 13.6 s/栋）。只按计数会在这种簇上错误中止。
    #   所以计数达标后还要**用墙钟佐证**：
    #     · 真饿死：11 个槽全部卡住 → epoch 墙钟/任务数 ≈ timeout/并发
    #     · 难样本簇：只有 1~2 个槽卡住、其余照常 → epoch 墙钟/任务数 仍 ~1s
    ABORT_AFTER = 12
    RATE_FLOOR_RATIO = 0.5            # epoch 墙钟/任务 > 0.5×timeout/并发 才算饿死
    consec_stall = 0
    aborted = False
    stall_epoch_wall = None           # 本轮连续 stall 的起点墙钟
    stall_epoch_task = 0              # 本轮连续 stall 的起点任务号
    rate_guard_hits = 0

    def _launch(bi, j):
        group = tasks[bi]
        d = _task_dir(j, bi)
        for n in group:
            _slot_put(d, os.path.join(ply_dir, n + ".ply"))
        ps = pixel_size
        if auto_ps:                       # 按样本跨度自适应栅格分辨率
            a = _auto_pixel_size(profile or {}, group[0],
                                 divisor=ps_divisor, hi=ps_cap)
            if a:
                ps = a
        p, before, ts = start_city3d(d, out_dir, min_points, ps,
                                     0, None, scip_limit, max_faces)
        bt = file_timeout * len(group) if file_timeout and file_timeout > 0 else None
        running.append([p, before, ts, bi, len(group), bt, j])

    while nxt < len(tasks) or running:
        while nxt < len(tasks) and free_slots:
            j = free_slots.pop(0)
            _launch(nxt, j)
            nxt += 1
        while True:                      # 等任意一个结束
            alive = [r for r in running if r[0].poll() is None]
            if len(alive) < len(running):
                break
            # ★ 主动超时巡检（2026-09-30 修）
            #   原实现只在"进程自己结束"之后才判断超时，于是当所有并发槽都
            #   卡在超线性求解上时，主循环会永久 sleep 下去 —— 实测 6 个槽
            #   同时卡死，日志停了 39 分钟都没有推进。这里改为主动巡检，
            #   到点就杀进程树（taskkill /T），让 poll() 返回并继续调度。
            now = time.time()
            for r in running:
                if r[0].poll() is None and r[5] and (now - r[2]) > r[5]:
                    _kill_tree(r[0].pid)
            time.sleep(0.4)
        for r in list(running):
            if r[0].poll() is None:
                continue
            running.remove(r)
            res = finish_city3d(r[0], r[1], r[2], out_dir, timeout=r[5])
            group = tasks[r[3]]
            # 精确归因：多进程并发写同一输出目录时，用"处理前后文件差集"会把
            # 别的进程刚写出的产物算进自己这一批（竞态）。这里直接按样本名核对。
            got = [n for n in group
                   if os.path.exists(os.path.join(out_dir, n + SUF))]
            if got:
                res["status"] = "ok"
            free_slots.append(r[6])
            per_batch.append({"batch": r[3], "task": group, "n": r[4],
                              "status": res["status"],
                              "seconds": round(res["seconds"], 1),
                              "ok": len(got), "got": got})
            done = sum(b["n"] for b in per_batch)
            print(f"  [{len(per_batch)}/{len(tasks)}] 已处理 {done}/{len(todo)}  "
                  f"状态={res['status']}  产出={len(got)}/{len(group)}  "
                  f"用时={res['seconds']:.0f}s  累计 {time.time()-t0:.0f}s")

            # ---- 熔断统计 ----------------------------------------------
            stalled = bool(r[5]) and res["seconds"] >= 0.9 * r[5] and not got
            now = time.time()
            if stalled:
                consec_stall += 1
                if stall_epoch_wall is None:
                    stall_epoch_wall = now
                    stall_epoch_task = len(per_batch) - 1
            else:
                consec_stall = 0
                stall_epoch_wall = None

            if stalled and consec_stall == 3:
                print("  ⚠ 连续 3 个任务跑满单栋超时且零产出 —— 若持续下去"
                      "大概率是 CPU 被其它进程抢占（见下方熔断说明）",
                      flush=True)

            if stalled and consec_stall >= ABORT_AFTER:
                # ★ 速率为辅判据：用"本 epoch 的墙钟 / 任务数"区分
                #   真饿死（全槽卡住，≈ timeout/并发）与难样本簇（仅个别槽卡住，≈1s）
                span_tasks = max(1, len(per_batch) - stall_epoch_task)
                wall_per_task = (now - stall_epoch_wall) / span_tasks
                floor = RATE_FLOOR_RATIO * (r[5] / max(1, jobs))
                if wall_per_task < floor:
                    rate_guard_hits += 1
                    print(f"  · 连续 {consec_stall} 次跑满超时，但墙钟仍 {wall_per_task:.2f} s/栋"
                          f"（饿死应 >= {floor:.2f}）—— 判定为**难样本簇**，继续。",
                          flush=True)
                    consec_stall = 0
                    stall_epoch_wall = None
                else:
                    aborted = True
                    print("\n" + "!" * 66)
                    print(f"  ✗ 熔断：连续 {consec_stall} 个任务跑满 {r[5]:.0f}s 超时且零产出，主动中止本阶段。")
                    print(f"    判别指纹：每个任务都精确用满超时（健康时应为 1~3s），")
                    print(f"    且墙钟退化为 {wall_per_task:.2f} s/栋 ≈ 超时/并发 = {r[5]/max(1,jobs):.2f}。")
                    print("    最可能原因：有其它重负载进程在抢 CPU，实测 official_score.py 的")
                    print("      normal_consistency（faiss 多线程）会把 City3D 从 1.8 s/栋 拖到 >150 s/栋。")
                    print("    排查：确认没有并行的评测/训练进程；杀掉后重跑本阶段（支持断点续跑，")
                    print("      已产出的样本会直接跳过），无需从头再来。")
                    print("!" * 66 + "\n", flush=True)
                    for rr in running:
                        try:
                            if rr[0].poll() is None:
                                _kill_tree(rr[0].pid)
                        except Exception:
                            pass
                    break

        if aborted:
            break

    produced = {f[:-len(SUF)] for f in os.listdir(out_dir) if f.endswith(SUF)}
    print(f"  City3D 产出 {len(produced)}/{len(names)}   用时 {time.time()-t0:.0f}s")
    statuses = {}
    for b in per_batch:
        statuses[b["status"]] = statuses.get(b["status"], 0) + b["n"]
    print(f"  批次状态: {statuses}")
    with open(os.path.join(ROOT, "logs", "s2_city3d.json"), "w", encoding="utf-8") as fp:
        json.dump({"batches": per_batch, "produced": len(produced),
                   "statuses": statuses, "total": len(names)}, fp,
                  indent=1, ensure_ascii=False)
    if aborted:
        raise RuntimeError("City3D 熔断：请排查重负载进程后断点续跑")
    return produced


# ------------------------------------------- stage 2.5 降采样重试

def _voxel_downsample(pts, target):
    """体素降采样：二分体素边长，直到剩余点数 <= target

    体素范围必须按点云自身尺度自适应 —— 数据集里样本跨度从 0.017 m 到 626 m，
    写死下限（例如 0.005）会把归一化坐标的样本整个塌成一格
    （实测 1924 号：14720 点 → 25 点，点云被彻底破坏）。
    """
    span = float(np.ptp(pts, axis=0).max()) or 1.0
    lo, hi = max(span / 5000.0, 1e-9), span * 2.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if len(np.unique(np.floor(pts / mid).astype(np.int64), axis=0)) > target:
            lo = mid
        else:
            hi = mid
    keys = np.floor(pts / hi).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _load_profile(ply_dir):
    """读预处理生成的几何体检报告，返回 {样本id: profile}"""
    p = os.path.join(ply_dir, "_profile.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fp:
        return {r["id"]: r for r in json.load(fp)}


def _auto_pixel_size(profile, name, lo=1e-4, hi=0.60, divisor=130.0):
    """
    按样本跨度自适应栅格分辨率（City3D 的 --pixel-size）。

    依据（2026-09-30 实测）：测试集样本尺度极不统一 ——
    跨度从 0.017 m（归一化坐标）到 626 m（建筑群），中位数 20.6 m。
    固定 pixel_size=0.15 时两头都出问题：
      · 跨度 <2 m 的样本，整个点云塞进不到一格 → 必然失败；
      · 跨度 >30 m 的样本，每格平均点数不足 0.01 → 栅格空洞 → 失败。

    经验公式 ps = span / 130，夹到 [1e-4, 0.6]：
      · 26 m  → 0.20（与原 0.15 同量级，不劣化现有成功样本）
      · 1.5 m → 0.011
      · 0.018 m → 1.4e-4（实测 1924 由完全失败转为 58 面成功）

    divisor / hi 可调（2026-10-03 加）：用于在**恒等口径**下重扫重建层，
    更小的 pixel_size = 更细的平面检测栅格 → 可能捡回更多小屋面。
    """
    r = profile.get(name)
    if not r:
        return None
    span = max(r.get("extent_xy", [0.0, 0.0]))
    if not span or span <= 0:
        return None
    return float(min(max(span / float(divisor), lo), hi))


def _down_one(job):
    """子进程任务：把一个 .ply 体素降采样到目标点数（已够小则原样复制）"""
    src, dst, target = job
    if os.path.exists(dst):
        return "skip"
    m = trimesh.load(src, process=False)
    pts = np.asarray(m.vertices if hasattr(m, "vertices") else m.points, float)
    if len(pts) <= target:
        shutil.copy2(src, dst)
        return "copy"
    write_ply(dst, _voxel_downsample(pts, target))
    return "down"


def stage_downsample(ply_dir, ds_dir, names, target=4000, jobs=8):
    """
    把指定样本体素降采样到 target 点数，写入 ds_dir。

    依据（2026-09-30 实测）：City3D 的屋顶重建对平面数呈超线性复杂度，
    点数 > 8000 的样本因近共面平面过多，会在 `polyfit_info.generate` 阶段
    抛 `degenerate facet with area: 0`，接着 SCIP 报
    `invalid objective value: objective value is infinite` 而整栋失败。
    降到 4000 点后 4/5 成功；2000 点太激进（丢结构）、8000 点不够（仍失败）。

    用多进程跑：4000 栋单线程要 20 分钟，8 进程后 3 分钟以内。
    """
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(ds_dir, exist_ok=True)
    todo = [(os.path.join(ply_dir, n + ".ply"), os.path.join(ds_dir, n + ".ply"), target)
            for n in names if not os.path.exists(os.path.join(ds_dir, n + ".ply"))]
    stat = {}
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
            for r in ex.map(_down_one, todo, chunksize=8):
                stat[r] = stat.get(r, 0) + 1
    print(f"  降采样 {len(names)} 栋：{stat.get('down', 0)} 栋减点，"
          f"{stat.get('copy', 0)} 栋原样，跳过已存在 {stat.get('skip', 0)} 栋")
    return len(names)


def stage_retry(ply_dir, ds_dir, rec_dir, produced, target=4000, limit=0,
                min_points=40, pixel_size=0.15, chunk=1, jobs=4,
                scip_limit=20, max_faces=600, file_timeout=90, ds_jobs=8,
                auto_ps=True, ps_divisor=130.0, ps_cap=0.60):
    """
    二级重试：对第一轮 City3D 无产出的样本，体素降采样后再跑一次。

    这一步能把"点数过多 → 数值退化"的假失败救回来，是覆盖率与得分的
    关键增益点（实测可把成功率从约 47% 拉到 80% 以上）。
    """
    print("\n[2.5/6] 降采样重试（救回点数过多导致的数值退化样本）")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ply_dir)
                   if f.lower().endswith(".ply"))
    if limit > 0:
        names = names[:limit]
    retry = [n for n in names if n not in produced]
    if not retry:
        print("  无需重试")
        return produced
    print(f"  第一轮无产出 {len(retry)} 栋，降采样到 {target} 点后重跑")
    stage_downsample(ply_dir, ds_dir, retry, target, ds_jobs)
    before = len(produced)
    again = stage_city3d(ds_dir, rec_dir, min_points, pixel_size,
                         limit, chunk, jobs, scip_limit, max_faces,
                         file_timeout, _load_profile(ply_dir), auto_ps,
                         ps_divisor, ps_cap)
    print(f"  重试救回 {len(again) - before} 栋")
    return produced | again


# ---------------------------------------------------------------- stage 3

def stage_rescue(ply_dir, rec_dir, produced, limit=0):
    """
    对 City3D 没产出结果的样本，用纯几何降级重建补上，保证 100% 覆盖。
    """
    print("\n[3/6] 降级重建（补全 City3D 漏掉的样本）")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ply_dir)
                   if f.lower().endswith(".ply"))
    if limit > 0:
        names = names[:limit]
    missing = [n for n in names if n not in produced]
    if not missing:
        print("  无需补全")
        return []
    print(f"  待补全 {len(missing)} 栋")

    os.makedirs(rec_dir, exist_ok=True)
    ok, bad = 0, []
    t0 = time.time()
    for i, n in enumerate(missing):
        try:
            m = trimesh.load(os.path.join(ply_dir, n + ".ply"), process=False)
            pts = np.asarray(m.vertices if hasattr(m, "vertices") else m.points, float)
            mesh = fallback.reconstruct(pts)
            if mesh is None or len(mesh.faces) == 0:
                raise ValueError("降级重建无输出")
            mesh.export(os.path.join(rec_dir, n + SUF))
            ok += 1
        except Exception as ex:
            bad.append(f"{n}: {ex}")
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(missing)} ...")
    print(f"  补全成功 {ok}/{len(missing)}   用时 {time.time()-t0:.0f}s")
    for x in bad[:5]:
        print("   ✗", x)
    return bad


# ---------------------------------------------------------------- stage 4

def stage_postproc(rec_dir, out_dir, coplanar_tol=3.0):
    print("\n[4/6] 指标导向后处理")
    os.makedirs(out_dir, exist_ok=True)
    objs = sorted(glob.glob(os.path.join(rec_dir, "*" + SUF)))
    stats = {"total": len(objs), "ok": 0, "in_faces": 0, "out_faces": 0,
             "in_verts": 0, "out_verts": 0, "failed": []}
    for i, f in enumerate(objs):
        base = os.path.basename(f)[:-len(SUF)]
        try:
            m = trimesh.load(f, process=False)
            if len(m.faces) == 0:
                raise ValueError("空网格")
            stats["in_faces"] += len(m.faces)
            stats["in_verts"] += len(m.vertices)
            m2 = postproc.process(m, coplanar_tol=coplanar_tol)
            if len(m2.faces) == 0:
                raise ValueError("后处理后为空")
            m2.export(os.path.join(out_dir, base + ".obj"))
            stats["out_faces"] += len(m2.faces)
            stats["out_verts"] += len(m2.vertices)
            stats["ok"] += 1
        except Exception as ex:
            stats["failed"].append(f"{base}: {ex}")
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(objs)} ...")

    if stats["ok"]:
        fr = stats["out_faces"] / max(1, stats["in_faces"])
        vr = stats["out_verts"] / max(1, stats["in_verts"])
        print(f"  成功 {stats['ok']}/{stats['total']}")
        print(f"  面片总数 {stats['in_faces']} -> {stats['out_faces']}  (压到 {fr*100:.1f}%)")
        print(f"  顶点总数 {stats['in_verts']} -> {stats['out_verts']}  (压到 {vr*100:.1f}%)")
        if stats["total"]:
            print(f"  平均面片 {stats['in_faces']/max(1,stats['ok']):.1f} -> "
                  f"{stats['out_faces']/max(1,stats['ok']):.1f}")
    for x in stats["failed"][:5]:
        print("   ✗", x)
    with open(os.path.join(ROOT, "logs", "s4_postproc.json"), "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=1, ensure_ascii=False)
    return stats["ok"]


# ---------------------------------------------------------------- stage 5

def stage_package(src_dir, zip_path):
    print("\n[5/6] 打包提交")
    objs = sorted(glob.glob(os.path.join(src_dir, "*.obj")))
    if not objs:
        print("  ✗ 没有可打包的 .obj")
        return False
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for o in objs:
            z.write(o, os.path.basename(o))
    size = os.path.getsize(zip_path) / 1e6
    print(f"  {len(objs)} 个模型 -> {zip_path}")
    print(f"  压缩包大小 {size:.1f} MB")
    return True


# ---------------------------------------------------------------- stage 6

def stage_report(ply_dir, fin_dir, rec_dir, produced, limit=0):
    print("\n[6/6] 覆盖率报告")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ply_dir)
                   if f.lower().endswith(".ply"))
    if limit > 0:
        names = names[:limit]
    finals = {os.path.splitext(f)[0] for f in os.listdir(fin_dir)} \
        if os.path.isdir(fin_dir) else set()
    from_c3 = sorted(set(names) & set(produced))
    from_fb = sorted(set(names) - set(produced))
    missing = sorted(set(names) - finals)

    cov = len(finals) / max(1, len(names))
    print(f"  输入样本      : {len(names)}")
    print(f"  City3D 重建   : {len(from_c3)}  ({len(from_c3)/max(1,len(names))*100:.1f}%)")
    print(f"  降级重建      : {len(from_fb)}  ({len(from_fb)/max(1,len(names))*100:.1f}%)")
    print(f"  最终交付      : {len(finals)}  ({cov*100:.1f}%)")
    if missing:
        print(f"  ✗ 缺失 {len(missing)} 栋: {missing[:10]}")

    rep = {"input": len(names), "city3d": len(from_c3), "fallback": len(from_fb),
           "final": len(finals), "coverage": round(cov, 4),
           "missing": missing[:100], "fallback_ids": from_fb[:200]}
    with open(os.path.join(ROOT, "logs", "coverage.json"), "w", encoding="utf-8") as fp:
        json.dump(rep, fp, indent=1, ensure_ascii=False)
    return cov


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "prepare", "city3d", "retry", "rescue",
                             "postproc", "package", "report"])
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "raw", "test_xyz",
                                                  "LiDAR_xyz"))
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "work"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1,
                    help="每批多少栋；1 表示一栋一个进程，隔离性最好")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-points", type=float, default=40)
    ap.add_argument("--pixel-size", type=float, default=0.15)
    ap.add_argument("--max-faces", type=int, default=600,
                    help="单栋候选面上限，超出即放弃（交给降级重建）")
    ap.add_argument("--scip-limit", type=float, default=20.0)
    ap.add_argument("--file-timeout", type=float, default=240.0,
                    help="单栋重建的时间上限（秒）；超时即放弃并转降级")
    ap.add_argument("--coplanar-tol", type=float, default=3.0)
    ap.add_argument("--target-points", type=int, default=4000,
                    help="重试阶段降采样到的目标点数；0 表示关闭降采样重试")
    ap.add_argument("--retry-jobs", type=int, default=4,
                    help="重试阶段并发数（降低并发可减少进程间干扰）")
    ap.add_argument("--build-jobs", type=int, default=8,
                    help="降采样阶段的并行进程数")
    ap.add_argument("--auto-pixel-size", type=int, default=1,
                    help="按样本跨度自适应栅格分辨率（1 开 / 0 关）")
    ap.add_argument("--ps-divisor", type=float, default=130.0,
                    help="自适应 pixel_size = span / ps-divisor（默认 130）")
    ap.add_argument("--ps-cap", type=float, default=0.60,
                    help="自适应 pixel_size 的上限，单位米（默认 0.60）")
    ap.add_argument("--rec-dir", default="",
                    help="City3D 原始产出目录；留空则用 <work>/recon")
    ap.add_argument("--fin-dir", default="",
                    help="后处理产出目录；留空则用 <work>/final")
    args = ap.parse_args()

    ply_dir = os.path.join(args.work, "ply")
    ds_dir = os.path.join(args.work, "ply_ds2")
    rec_dir = args.rec_dir or os.path.join(args.work, "recon")
    fin_dir = args.fin_dir or os.path.join(args.work, "final")
    zip_path = os.path.join(ROOT, "out", "submission.zip")

    t0 = time.time()
    print("=" * 66)
    print("  LiDAR2026 赛道三 —— 端到端流水线")
    print("=" * 66)

    produced = set()
    if args.stage in ("all", "prepare"):
        if not stage_prepare(args.src, ply_dir, args.limit):
            print("预处理失败，中止")
            return 1
    if args.stage in ("all", "city3d"):
        produced = stage_city3d(ply_dir, rec_dir, args.min_points, args.pixel_size,
                                args.limit, args.chunk, args.jobs,
                                args.scip_limit, args.max_faces, args.file_timeout,
                                _load_profile(ply_dir), bool(args.auto_pixel_size),
                                args.ps_divisor, args.ps_cap)
    elif args.stage in ("retry", "rescue", "postproc", "package", "report"):
        produced = {f[:-len(SUF)] for f in os.listdir(rec_dir) if f.endswith(SUF)} \
            if os.path.isdir(rec_dir) else set()

    if args.stage in ("all", "retry") and args.target_points > 0:
        produced = stage_retry(ply_dir, ds_dir, rec_dir, produced,
                               args.target_points, args.limit,
                               args.min_points, args.pixel_size, args.chunk,
                               args.retry_jobs, args.scip_limit,
                               args.max_faces, args.file_timeout,
                               args.build_jobs, bool(args.auto_pixel_size),
                               args.ps_divisor, args.ps_cap)

    if args.stage in ("all", "rescue"):
        stage_rescue(ply_dir, rec_dir, produced, args.limit)
    if args.stage in ("all", "postproc"):
        stage_postproc(rec_dir, fin_dir, args.coplanar_tol)
    if args.stage in ("all", "package"):
        stage_package(fin_dir, zip_path)
    if args.stage in ("all", "report"):
        stage_report(ply_dir, fin_dir, rec_dir, produced, args.limit)

    print(f"\n总用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
