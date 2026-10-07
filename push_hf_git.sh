#!/usr/bin/env bash
# ============================================================================
# 一键把 release_ccby4/ 发布为 Hugging Face 公开仓库（CC BY 4.0）
#
# 依赖：零。只用 git（本机已装 2.55），不需要 huggingface_hub / requests。
#
# 用法：
#   export HF_USER=wind20011911        # 你的 HF 用户名
#   export HF_TOKEN=hf_xxxxxxxxxxxx    # 必须是 **write** 权限的 token
#   bash release_ccby4/push_hf_git.sh [仓库名]
#
# 前置条件：**必须先开 VPN** —— 本机实测 huggingface.co 直连返回 000（被墙）。
#
# 安全说明：token 仅出现在本次命令的命令行里，**不会**写进 .git/config
#           （这里刻意不 git remote add，直接用一次性 URL 推送）。
# ============================================================================
set -euo pipefail

_here="$(cd "$(dirname "$0")" && pwd)"
_root="$(cd "$_here/.." && pwd)"
_read_first() { [ -f "$1" ] && awk 'NF{print; exit}' "$1" 2>/dev/null || true; }

HF_USER="${HF_USER:-$(_read_first "$_root/tmp/.hf_user")}"
HF_USER="${HF_USER:-$(_read_first "$_here/tmp/.hf_user")}"
HF_USER="${HF_USER:?[ERR] 请 export HF_USER=<你的 HF 用户名>，或写入 D:/LiDAR2026/tmp/.hf_user}"

HF_TOKEN="${HF_TOKEN:-$(_read_first "$_root/tmp/.hf_write_token")}"
HF_TOKEN="${HF_TOKEN:-$(_read_first "$_here/tmp/.hf_write_token")}"
HF_TOKEN="${HF_TOKEN:?[ERR] 请 export HF_TOKEN=<write token>，或写入 D:/LiDAR2026/tmp/.hf_write_token}"

REPO="${1:-lidar2026-track3-lod2}"

cd "$(dirname "$0")"

echo "[1/4] 连通性检查（huggingface.co 需 VPN）..."
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://huggingface.co || true)
if [ "$code" = "000" ] || [ -z "$code" ]; then
  echo "[ERR] huggingface.co 不可达（$code）。请先开启 VPN 再重试。"
  echo "      备选：改用 GitHub（见 PUBLISH_发布步骤.md 方案 B，github.com 本机可达）"
  exit 2
fi
echo "      huggingface.co -> $code"

echo "[2/4] 校验 token 与写权限..."
who=$(curl -s --max-time 20 -H "Authorization: Bearer ${HF_TOKEN}" \
      https://huggingface.co/api/whoami)
echo "      $who" | head -c 400; echo
case "$who" in
  *'"name"'*) : ;;
  *) echo "[ERR] token 校验失败（响应见上）。请确认 token 有效且为 write 权限。"; exit 2 ;;
esac

echo "[3/4] 推送（首次推送会自动创建公开仓库；已存在则为增量覆盖）..."
git push "https://${HF_USER}:${HF_TOKEN}@huggingface.co/${HF_USER}/${REPO}.git" main

echo "[4/4] 完成 ✅"
echo
echo "公开地址: https://huggingface.co/${HF_USER}/${REPO}"
echo "下一步：把该地址写进 docs/赛道三_方法说明文档.md 与 docs/赛道三_扩展摘要_4页.md，"
echo "        替换其中的 <CODE_URL_PLACEHOLDER>，并在竞赛平台/邮件补交。"
