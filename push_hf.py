# -*- coding: utf-8 -*-
"""一键把 release_ccby4/ 发布为 Hugging Face 公开仓库（CC BY 4.0）。

用法:
    pip install -U huggingface_hub
    export HF_TOKEN=hf_xxxxxxxx        # 必须是 write 权限的 token
    python release_ccby4/push_hf.py --repo <user>/lidar2026-track3-lod2

会话 cookie 里的 `hf_oauth_...` 只有 `read-repos`，**不能**建仓库；
请到 https://huggingface.co/settings/tokens 新建一个 write token。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<user>/<name>，例如 wind2001911/lidar2026-track3-lod2")
    ap.add_argument("--token", default=None, help="HF write token；也可用环境变量 HF_TOKEN")
    ap.add_argument("--private", action="store_true", help="默认公开（赛事要求公开可访问）")
    ap.add_argument("--src", default=ROOT)
    a = ap.parse_args()

    token = a.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[ERR] 没有 token。设置 HF_TOKEN 或用 --token 传入（须 write 权限）。")
        return 2
    if not a.private and (not token.startswith("hf_") or len(token) < 20):
        print("[WARN] token 看起来不像标准 HF token，仍继续尝试。")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERR] 缺少 huggingface_hub：pip install -U huggingface_hub")
        return 2

    api = HfApi(token=token)
    try:
        who = api.whoami()
        print("[auth] 登录为 %s" % who.get("name"))
    except Exception as ex:                                    # noqa: BLE001
        print("[ERR] token 校验失败：%s" % ex)
        return 2

    url = api.create_repo(repo_id=a.repo, repo_type="model",
                          private=a.private, exist_ok=True)
    print("[repo] %s" % url)

    api.upload_folder(
        repo_id=a.repo, repo_type="model", folder_path=a.src,
        ignore_patterns=["push_hf.py", "**/__pycache__/**", "**/*.pyc", "**/.git/**"],
        commit_message="LoD2 building reconstruction from airborne LiDAR (CC BY 4.0) - team wind",
    )
    print("[ok] 上传完成")

    try:
        api.update_repo_card  # noqa: B018  (仅探测版本，不存在则跳过)
    except Exception:                                          # noqa: BLE001
        pass
    try:
        api.model_info(a.repo)
    except Exception:                                          # noqa: BLE001
        pass

    addr = "https://huggingface.co/%s" % a.repo
    print("\n公开地址: %s" % addr)
    print("下一步：把该地址写进 方法说明文档 与 扩展摘要，并在竞赛平台补交。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
