"""
City3D 调用层（统一封装）

要点：
  * City3D 的 CLI 输出中文（Windows 上是 GBK 字节），不能用 text=True 按 UTF-8 解，
    否则 reader 线程抛 UnicodeDecodeError。这里统一按字节收，再宽松解码。
  * 判定"是否成功"不看返回码，而看是否产出了 <stem>_ReconstructedModel.obj
    —— bad_alloc 是 C++ 异常，返回码不一定可靠。
  * 必须用绝对路径指定 bash：Python 的 subprocess 若只写 "bash"，在 Windows 上
    会解析到 C:\\Windows\\System32\\bash.exe（WSL），本机被安全策略拦截，
    表现为"0.1 秒返回、stdout 为空"，极易被误判成 City3D 本身失败。
  * c3.sh 包装脚本是必需的：直接拼命令行会因为 $PATH 里的 "(x86)" 破坏嵌套引号。
  * 超时必须杀进程树：kill 掉 bash 后，CLI_Example_2.exe 会变成孤儿进程继续占着
    输出文件和管道，导致 communicate() 永远等不到 EOF。
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C3_SH = os.path.join(ROOT, "src", "c3.sh")
EXE = os.path.join(ROOT, "city3d", "code", "Release", "bin", "CLI_Example_2.exe")

BASH = None
for _cand in (r"D:\msys64\usr\bin\bash.exe",
              r"C:\Program Files\Git\bin\bash.exe",
              r"C:\Program Files (x86)\Git\bin\bash.exe"):
    if os.path.exists(_cand):
        BASH = _cand
        break

BAD_ALLOC = ("bad_alloc", "MemoryError")
STD_ALLOC = ("std::length_error", "cannot allocate")


def _decode(b):
    if not b:
        return ""
    for enc in ("gbk", "utf-8"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("gbk", "replace")


def _kill_tree(pid):
    """Windows 下按进程树强杀（否则会留下孤儿 exe 占文件、占管道）"""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=30)
    except Exception:
        pass


def _read_objs(out_dir):
    if not os.path.isdir(out_dir):
        return set()
    return {f[: -len("_ReconstructedModel.obj")]
            for f in os.listdir(out_dir)
            if f.endswith("_ReconstructedModel.obj")}


def build_cmd(in_dir, out_dir, min_points=40, pixel_size=0.15,
              limit=0, c3_sh=None, max_faces=None):
    c3_sh = c3_sh or C3_SH
    cmd = [BASH, c3_sh, in_dir, out_dir, str(min_points), str(pixel_size), str(limit)]
    if max_faces is not None:
        cmd.append(str(max_faces))
    return cmd


def make_env(scip_limit=None, max_faces=None):
    env = dict(os.environ)
    if scip_limit is not None:
        env["CITY3D_SCIP_TIME_LIMIT"] = str(scip_limit)
    if max_faces is not None:
        env["CITY3D_MAX_CANDIDATE_FACES"] = str(max_faces)
    return env


def start_city3d(in_dir, out_dir, min_points=40, pixel_size=0.15,
                 limit=0, c3_sh=None, scip_limit=None, max_faces=None):
    """非阻塞启动，返回 (Popen, 启动前的产物集合, 起始时间)"""
    if BASH is None:
        raise RuntimeError("找不到可用的 bash（需要 MSYS2 或 Git Bash）")
    # ★ 必须绝对化（2026-10-01 实测）
    #   c3.sh 内部有 `cd /d/LiDAR2026/city3d/code/Release/bin`，相对路径会在 cd
    #   之后被重新解释到 bin 目录下。此时 City3D 报
    #     could not mkdir<out_dir>  /  无法创建输出目录: <out_dir>
    #   然后**退出码 0 静默失败**，不产出任何文件 —— 在批量日志里表现为
    #   "状态=fail 产出=0/N 用时≈1s"，极易被误判成数据问题或算法失败。
    #   实测：同一份点云，相对路径 0/1 失败，绝对路径 1/1 成功。
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    before = _read_objs(out_dir)
    cmd = build_cmd(in_dir, out_dir, min_points, pixel_size, limit, c3_sh, max_faces)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=make_env(scip_limit, max_faces), cwd=ROOT)
    return p, before, time.time()


def finish_city3d(p, before, t0, out_dir, timeout=None):
    """等待并汇总一次运行结果"""
    timed_out = False
    try:
        out_b, err_b = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(p.pid)
        try:
            out_b, err_b = p.communicate(timeout=60)
        except Exception:
            out_b, err_b = b"", b""
        rc = -1

    dt = time.time() - t0
    out, err = _decode(out_b), _decode(err_b)
    blob = out + "\n" + err
    new_objs = _read_objs(out_dir) - before

    if timed_out:
        status = "timeout"
    elif new_objs:
        status = "ok"
    elif any(t in blob for t in BAD_ALLOC):
        status = "bad_alloc"
    elif any(t in blob for t in STD_ALLOC):
        status = "alloc"
    else:
        status = "fail"

    return {"ok": bool(new_objs), "status": status, "new_objs": sorted(new_objs),
            "stdout": out, "stderr": err, "returncode": rc,
            "seconds": dt, "timed_out": timed_out}


def run_city3d(in_dir, out_dir, min_points=40, pixel_size=0.15,
               limit=0, timeout=None, c3_sh=None, scip_limit=None,
               max_faces=None):
    """阻塞调用一次 City3D"""
    p, before, t0 = start_city3d(in_dir, out_dir, min_points, pixel_size,
                                 limit, c3_sh, scip_limit, max_faces)
    return finish_city3d(p, before, t0, out_dir, timeout)
