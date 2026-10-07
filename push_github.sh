#!/usr/bin/env bash
# ============================================================================
# 一键把 release_ccby4/ 发布为 GitHub 公开仓库（CC BY 4.0）
#
# 依赖：零。只用 git + curl（本机 git 2.55 已装）。
# 特点：**需要 VPN**（本机实测 github.com 直连 000；VPN 下 200 / 8s，
#       git 通道实测 `git ls-remote` 可用）。
#
# 用法：
#   export GITHUB_USER=yourname            # GitHub 用户名
#   export GITHUB_TOKEN=ghp_xxxx           # PAT（凭据）
#   bash release_ccby4/push_github.sh [仓库名]
#
# 令牌建议：
#   - Classic PAT，权限勾 `repo`（可自动建仓 + 推送）—— 最省事
#   - 或 Fine-grained PAT：`Contents: Read and write` + `Administration: Read and write`
#   生成：https://github.com/settings/tokens
#
# 安全说明：token 仅出现在本次命令内，不会写进 .git/config。
# ============================================================================
set -euo pipefail

GITHUB_USER="${GITHUB_USER:?[ERR] 请先 export GITHUB_USER=<你的 GitHub 用户名>}"
GITHUB_TOKEN="${GITHUB_TOKEN:?[ERR] 请先 export GITHUB_TOKEN=<PAT，见 github.com/settings/tokens>}"
REPO="${1:-lidar2026-track3-lod2}"

cd "$(dirname "$0")"

api="https://api.github.com"

echo "[1/5] 连通性检查（GitHub 需要 VPN）..."
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$api" || true)
if [ "$code" = "000" ] || [ -z "$code" ]; then
  echo "[ERR] api.github.com 不可达（$code）。请先开启 VPN 再重试。"
  echo "      无 VPN 备选：bash release_ccby4/push_gitee.sh（Gitee，国内直连）"
  exit 2
fi
echo "      api.github.com -> $code"

echo "[2/5] 校验 token..."
who=$(curl -s --max-time 25 -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" "$api/user")
login=$(printf '%s' "$who" | tr ',' '\n' | grep -m1 '"login"' | sed 's/.*: *"//; s/".*//' || true)
if [ -z "$login" ]; then
  echo "[ERR] token 校验失败：$(printf '%s' "$who" | head -c 300)"
  exit 2
fi
echo "      已认证：${login}"

echo "[3/5] 确保远端公开仓库存在（不存在则创建）..."
resp=$(curl -s --max-time 30 -X POST "$api/user/repos" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"name\":\"${REPO}\",\"description\":\"10th National LiDAR Conference Track 3 (LoD2 building reconstruction from airborne LiDAR) - team wind\",\"private\":false,\"has_issues\":true,\"has_wiki\":false,\"auto_init\":false}")
case "$resp" in
  *'"full_name"'*) echo "      仓库已创建：${GITHUB_USER}/${REPO}" ;;
  *"already exists"*) echo "      仓库已存在，跳过创建" ;;
  *) echo "      (创建返回，若已存在可忽略) $(printf '%s' "$resp" | head -c 200)" ;;
esac

echo "[4/5] 推送（首次推送数据量约几 MB，VPN 慢，请耐心）..."
git push "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO}.git" main

echo "[5/5] 完成 ✅"
echo
echo "公开地址: https://github.com/${GITHUB_USER}/${REPO}"
echo "下一步：把该地址写进 docs/赛道三_方法说明文档.md 与 docs/赛道三_扩展摘要_4页.md，"
echo "        替换其中的 <CODE_URL_PLACEHOLDER>，并在竞赛平台/邮件补交。"
