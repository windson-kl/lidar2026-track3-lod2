#!/usr/bin/env bash
# ============================================================================
# 一键把 release_ccby4/ 发布为 Gitee（码云）公开仓库（CC BY 4.0）
#
# 依赖：零。只用 git + curl（本机 git 2.55 已装）。
# 特点：**国内直连可用，不需要 VPN**（实测 gitee.com 200 / 1.5s）。
#
# 用法：
#   export GITEE_USER=wind20011911              # 你的 Gitee 用户名（登录名，非昵称）
#   export GITEE_TOKEN=xxxxxxxxxxxxxxxxxxxx     # Gitee 私人令牌，勾 projects 权限
#   bash release_ccby4/push_gitee.sh [仓库名]
#
# 令牌生成：https://gitee.com/profile/personal_access_tokens → 生成新令牌
#           权限至少勾选「projects」（仓库读写）。
#
# 安全说明：token 仅出现在本次命令内，不会写进 .git/config。
# ============================================================================
set -euo pipefail

_here="$(cd "$(dirname "$0")" && pwd)"
_root="$(cd "$_here/.." && pwd)"
_read_first() { [ -f "$1" ] && awk 'NF{print; exit}' "$1" 2>/dev/null || true; }

GITEE_USER="${GITEE_USER:-$(_read_first "$_root/tmp/.gitee_user")}"
GITEE_USER="${GITEE_USER:-$(_read_first "$_here/tmp/.gitee_user")}"
GITEE_USER="${GITEE_USER:?[ERR] 请 export GITEE_USER=<你的 Gitee 用户名>，或写入 D:/LiDAR2026/tmp/.gitee_user}"

GITEE_TOKEN="${GITEE_TOKEN:-$(_read_first "$_root/tmp/.gitee_token")}"
GITEE_TOKEN="${GITEE_TOKEN:-$(_read_first "$_here/tmp/.gitee_token")}"
GITEE_TOKEN="${GITEE_TOKEN:?[ERR] 请 export GITEE_TOKEN=<私人令牌>，或写入 D:/LiDAR2026/tmp/.gitee_token}"

REPO="${1:-lidar2026-track3-lod2}"

cd "$(dirname "$0")"

echo "[1/5] 连通性检查（Gitee 国内直连，无需 VPN）..."
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://gitee.com || true)
if [ "$code" = "000" ] || [ -z "$code" ]; then
  echo "[ERR] gitee.com 不可达（$code）。检查本机网络。"
  exit 2
fi
echo "      gitee.com -> $code"

echo "[2/5] 校验令牌..."
who=$(curl -s --max-time 20 "https://gitee.com/api/v5/user?access_token=${GITEE_TOKEN}")
case "$who" in
  *'"login"'*) echo "      已认证：$(echo "$who" | tr ',' '\n' | grep -m1 '"login"' | tr -d ' \"')" ;;
  *) echo "[ERR] 令牌校验失败：$who"; exit 2 ;;
esac

echo "[3/5] 确保远端公开仓库存在（不存在则创建，已存在则跳过）..."
create=$(curl -s --max-time 25 -X POST "https://gitee.com/api/v5/user/repos" \
  -d "access_token=${GITEE_TOKEN}" \
  -d "name=${REPO}" \
  -d "private=false" \
  -d "auto_init=false" \
  -d "description=10th LiDAR Conference Track 3 (LoD2 building reconstruction) - team wind" \
  -d "license=CC-BY-4.0" || true)
case "$create" in
  *'"full_name"'*) echo "      仓库就绪：${GITEE_USER}/${REPO}" ;;
  *"already exists"*|*"已存在"*) echo "      仓库已存在，跳过创建" ;;
  *) echo "      (创建返回，若已存在可忽略) $(echo "$create" | head -c 200)" ;;
esac

echo "[4/5] 推送..."
git push "https://${GITEE_USER}:${GITEE_TOKEN}@gitee.com/${GITEE_USER}/${REPO}.git" main

echo "[5/5] 完成 ✅"
echo
echo "公开地址: https://gitee.com/${GITEE_USER}/${REPO}"
echo "下一步：把该地址写进 docs/赛道三_方法说明文档.md 与 docs/赛道三_扩展摘要_4页.md，"
echo "        替换其中的 <CODE_URL_PLACEHOLDER>，并在竞赛平台/邮件补交。"
