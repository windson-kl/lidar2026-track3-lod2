# -*- coding: utf-8 -*-
"""
submit_space.py -- BuildingWorld 赛道三 评测平台(HF Space) 提交助手

平台:  https://buildingworld-10thlidarconference.hf.space
鉴权:  HF OAuth 会话 Cookie (仅 Cookie 有效; 平台对未登录返回 "Invalid token. Please login.")

用法 (Cookie 从命令行或 tmp/.hf_cookie 文件读取, 只取第一个非空行):
    python src/submit_space.py status                 # 校验登录态 + 打印我的提交列表
    python src/submit_space.py submit <zip路径>       # 提交一个 zip
    python src/submit_space.py subs                   # 只看 /my_submissions
    python src/submit_space.py poll [轮数] [间隔秒]    # 轮询 /my_submissions 直到出现分数

约定: 输出仅用 ASCII, 规避 bash -lc 下 GBK 编码炸字符的问题。
"""
import os
import re
import sys
import json
import time
import uuid
import urllib.request
import urllib.error

BASE = "https://buildingworld-10thlidarconference.hf.space"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIE_FILE = os.path.join(ROOT, "tmp", ".hf_cookie")


def normalize_cookie(raw):
    """平台 Cookie 头. 支持三种输入:
    1) 完整 Cookie 头 "session=xxx"; 2) 仅 session 值(带签名, 含'.')
    注意: session 值的 payload 尾部带 '=='(base64 padding), 不能用 "是否含 =" 来判断,
    否则会把裸值误认为已带 Cookie 名, 导致发出去的头部缺少 "session=" 前缀.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.match(r"^[A-Za-z0-9_\-]{1,20}=", raw)   # 形如 "name=" 才算已带 Cookie 名
    if m:
        return raw
    return "session=" + raw


def load_cookie():
    # 优先环境变量, 其次 cookie 文件
    env = os.environ.get("HF_SPACE_COOKIE", "").strip()
    if env:
        return normalize_cookie(env)
    if os.path.exists(COOKIE_FILE):
        with open(COOKIE_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    return normalize_cookie(line)
    return ""


def call(path, data=None, content_type=None, cookie="", method=None, raw=None):
    url = BASE + path
    headers = {"User-Agent": "wb-submit/1.0"}
    if cookie:
        headers["Cookie"] = cookie
    if raw is not None:
        body = raw
    elif data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        body = None
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=180)
        return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return "ERR", str(e)


def build_multipart(fields, files):
    boundary = "----WBBoundary" + uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (boundary, k, v)).encode("utf-8"))
    for k, (fn, blob, ct) in files.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n" % (boundary, k, fn, ct)).encode("utf-8"))
        out.append(blob)
        out.append(b"\r\n")
    out.append(("--%s--\r\n" % boundary).encode("utf-8"))
    return b"".join(out), "multipart/form-data; boundary=%s" % boundary


def check_login(cookie):
    val = cookie.split("session=", 1)[-1] if "session=" in cookie else cookie
    segs = val.split(".")
    print("[cookie] len=%d segments=%d (Starlette 会话需 3 段: payload.timestamp.signature)"
          % (len(val), len(segs)))
    s, t = call("/login_status", cookie=cookie)
    try:
        logged = json.loads(t).get("response") == 2
    except Exception:
        logged = False
    print("[login_status] http=%s body=%s logged_in=%s" % (s, t.strip(), logged))
    return logged


def show_subs(cookie):
    s, t = call("/my_submissions", data={}, cookie=cookie)
    print("[/my_submissions] http=%s" % s)
    try:
        j = json.loads(t)
        resp = j.get("response", {})
        print("--- submission_text ---")
        print(resp.get("submission_text", ""))
        subs = resp.get("submissions", "")
        if subs:
            print("--- submissions ---")
            print(subs if isinstance(subs, str) else json.dumps(subs, ensure_ascii=False, indent=2))
        if resp.get("error"):
            print("--- error ---")
            print(resp["error"])
        return resp
    except Exception as e:
        print(t[:2000])
        print("parse error:", e)
        return None


def do_submit(zip_path, comment, cookie):
    if not os.path.exists(zip_path):
        print("[ERR] zip not found:", zip_path)
        return
    with open(zip_path, "rb") as f:
        blob = f.read()
    print("[submit] file=%s size=%.2f MB" % (os.path.basename(zip_path), len(blob) / 1e6))
    fields = {"hub_model": "None", "submission_comment": comment}
    files = {"submission_file": (os.path.basename(zip_path), blob, "application/zip")}
    body, ct = build_multipart(fields, files)
    s, t = call("/new_submission", content_type=ct, cookie=cookie, raw=body, method="POST")
    print("[new_submission] http=%s" % s)
    print(t[:3000])


def get_subs(cookie):
    """返回 /my_submissions 里的 submissions 列表（已解析）。"""
    s, t = call("/my_submissions", data={}, cookie=cookie)
    j = json.loads(t)
    subs = j.get("response", {}).get("submissions", "")
    if isinstance(subs, str):
        subs = json.loads(subs) if subs.strip() else []
    return subs


def score_of(it):
    """FINAL 分数；无分/失败记为 +inf（排序时排最后）。"""
    sc = (it.get("public_score") or "").strip()
    if not sc or sc == "{}":
        return float("inf")
    try:
        return float(json.loads(sc).get("final_score", float("inf")))
    except Exception:
        return float("inf")


def do_select(ids, cookie, dry=False):
    """把选中列表整体覆盖为 ids（平台侧是覆盖语义，务必传全量）。"""
    payload = {"submission_ids": ",".join(ids)}
    if dry:
        print("[dry-run] would POST submission_ids = %s" % (payload["submission_ids"] or "(empty)"))
        return
    s, t = call("/update_selected_submissions", data=payload, cookie=cookie, method="POST")
    print("[update_selected_submissions] http=%s  n=%d" % (s, len(ids)))
    print(t[:1500])


def cmd_select(args, cookie):
    """select show|best N|set <id前缀...>|add <id前缀...>|clear --yes

    ★ 平台只允许最多 40 个提交进入私有榜；最终排名按私有榜算。
      未 select 任何提交 => 最终成绩为空，务必在 10-09 前选中最好的若干条。
    """
    mode = args[1] if len(args) > 1 else "show"
    subs = get_subs(cookie)
    if not subs:
        print("[ERR] /my_submissions 返回空，无法操作")
        return
    if mode in ("show", "list"):
        ranked = sorted(subs, key=score_of)
        print("%-10s %-20s %10s  %s  %s" % ("id", "datetime", "FINAL", "sel", "status"))
        for it in ranked:
            sc = score_of(it)
            print("%-10s %-20s %10s  %-3s  %s"
                  % (it.get("submission_id", "")[:8], it.get("datetime", ""),
                     ("%.5f" % sc) if sc != float("inf") else "-",
                     "Y" if it.get("selected") else ".", it.get("status", "")))
        cur = [it["submission_id"][:8] for it in subs if it.get("selected")]
        print("\n已选 %d 个: %s" % (len(cur), ", ".join(cur) if cur else "(无)"))
        print("可用: select best N | select set <前缀...> | select add <前缀...>")
        return

    if mode == "clear":
        if "--yes" not in args:
            print("[ERR] clear 会把选中列表清空（不可逆到'无选中'状态）。加 --yes 确认。")
            return
        do_select([], cookie)
        return

    if mode == "best":
        n = int(args[2]) if len(args) > 2 else 1
        ranked = sorted(subs, key=score_of)
        ids = [it["submission_id"] for it in ranked[:n] if score_of(it) != float("inf")]
        if not ids:
            print("[ERR] 没有出分的提交")
            return
        print("[select best %d]" % n)
        for it in ranked[:n]:
            print("  %s  %-20s  %.5f" % (it["submission_id"][:8], it.get("datetime", ""),
                                         score_of(it)))
        do_select(ids, cookie, dry=("--dry" in args))
        return

    if mode in ("set", "add"):
        pfx = [a for a in args[2:] if not a.startswith("--")]
        if not pfx:
            print("[ERR] 需要提交 id 前缀")
            return
        chosen, miss = [], []
        for p in pfx:
            hit = [it["submission_id"] for it in subs if it["submission_id"].startswith(p)]
            if len(hit) == 1:
                chosen.append(hit[0])
            else:
                miss.append(p)
        if miss:
            print("[ERR] 前缀不唯一或未找到:", miss)
            return
        if mode == "add":
            cur = [it["submission_id"] for it in subs if it.get("selected")]
            ids = list(dict.fromkeys(cur + chosen))
        else:
            ids = chosen
        if len(ids) > 40:
            print("[ERR] 平台上限 40 个，当前会提交 %d 个" % len(ids))
            return
        print("[select %s] %d 个: %s" % (mode, len(ids), ", ".join(x[:8] for x in ids)))
        do_select(ids, cookie, dry=("--dry" in args))
        return

    print("[ERR] unknown select mode:", mode)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    cookie = load_cookie()
    if not cookie:
        print("[ERR] no cookie. Put it in %s (one line) or set HF_SPACE_COOKIE." % COOKIE_FILE)
        return
    if cmd == "status":
        if check_login(cookie):
            show_subs(cookie)
    elif cmd == "subs":
        show_subs(cookie)
    elif cmd == "submit":
        if len(args) < 2:
            print("[ERR] usage: submit <zip> [comment]")
            return
        comment = args[2] if len(args) > 2 else "probe v8_raw"
        if not check_login(cookie):
            print("[ERR] not logged in; abort.")
            return
        do_submit(args[1], comment, cookie)
        time.sleep(3)
        show_subs(cookie)
    elif cmd == "select":
        cmd_select(args, cookie)
    elif cmd == "poll":
        rounds = int(args[1]) if len(args) > 1 else 60
        gap = int(args[2]) if len(args) > 2 else 60
        for i in range(rounds):
            print("=== poll %d/%d  %s ===" % (i + 1, rounds, time.strftime("%H:%M:%S")))
            s, t = call("/my_submissions", data={}, cookie=cookie)
            latest, done = None, False
            try:
                subs = json.loads(t)["response"]["submissions"]
                if isinstance(subs, str):
                    subs = json.loads(subs) if subs.strip() else []
                for it in subs:
                    sc = (it.get("public_score") or "").strip()
                    print("  id=%s  %s  %s  %s" % (it.get("submission_id", "")[:8],
                                                  it.get("datetime", ""),
                                                  it.get("status", ""),
                                                  (sc[:140] if sc and sc != "{}" else "-")))
                    if latest is None:
                        latest = it
                # 只以"最新一条"是否出分作为终止条件（旧提交的分数不算）
                if latest is not None:
                    lsc = (latest.get("public_score") or "").strip()
                    if latest.get("status") == "SUCCESS" and lsc and lsc != "{}":
                        print("[poll] latest submission scored -> stop.")
                        done = True
            except Exception as e:
                print("  parse error:", e, t[:200])
            if done:
                print("[poll] score detected -> stop.")
                break
            if i < rounds - 1:
                time.sleep(gap)
    else:
        print("[ERR] unknown cmd:", cmd)
        print(__doc__)


if __name__ == "__main__":
    main()
