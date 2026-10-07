# 发布步骤（CC BY 4.0 开源代码）

官方获奖条件原文（`docs/official/赛道三_官方_competition_info.txt`）：

> - 提交一份详细说明解决方案的技术说明文档；
> - 以扩展摘要（4 页）或完整论文的形式提交至组织者；
> - **提供能够生成获奖提交结果的代码；**
> - **代码需采用 CC BY 4.0 许可证发布。**

前两条已完成，并已于 **2026-10-07** 更新至当前最佳方案 v37
（`docs/赛道三_方法说明文档.md`、`docs/赛道三_扩展摘要_4页.md`）。
**第三条（代码发布）已经准备好，只差最后一次推送**：
本目录已初始化为 git 仓库并完成首个提交（分支 `main`、42 个文件、commit `d96d828`），
仅缺一个**有写权限的凭据**把提交推到公网。

> **网络实测（2026-10-07 晚复测）**：境外托管站**全部 000 不可达** ——
> `huggingface.co` 000、`github.com` 000、`raw.githubusercontent.com` 000（早先 github 曾 200，现已不可达）。
> 国内托管站**全部可达** —— `gitee.com` 200(1.5s)、`gitcode.com` 200、`atomgit.com` 200；
> 评测平台 `buildingworld-...hf.space` 200、`pypi.tuna.tsinghua.edu.cn` 200。
>
> **结论**：→ 无 VPN 时走 **方案 C（Gitee）**；有 VPN 时 HF（方案 A）与 GitHub（方案 B）均可。
> 官方只要求"代码以 CC BY 4.0 公开发布"，**未限定平台**，Gitee 公开仓库完全满足。

发布前已完成的一次密钥扫描：包内**无任何 Cookie / token / 密码**，
`submit_space.py` 仅从 `tmp/.hf_cookie` 或环境变量读取凭据，可安全公开。

---

## 包内容

```
release_ccby4/
├── LICENSE                       CC BY 4.0 全文（官方要求）
├── README.md                     双语说明：任务、流程、复现步骤、方法要点、局限
├── PUBLISH_发布步骤.md            本文件
├── src/                          18 个脚本
│   ├── 主流水线  prepare.py c3run.py run_batch.py fallback.py postproc.py
│   │            qc_repair.py package.py _t_build_v9r.py
│   ├── 收官算子  _t_build_walls.py（v10 墙）_t_apply_clean.py（v12 清理）
│   ├── 自检      verify_submission.py submit_space.py
│   └── 诊断工具  _t_check_ab.py _t_ecd_risk.py _t_walls_cfg.py _t_walls_cfg_cmp.py
│                synth.py metrics.py evaluate.py
└── docs/
    ├── 赛道三_方法说明文档.md
    └── 赛道三_扩展摘要_4页.md
```

**特意没有打包的东西**（见 README §3.3）：
官方评测脚本 `official_evaluate.py`、BuildingWorld 数据集、4000 个测试点云与产物网格、
City3D 本体 —— 这些要么是组织方的、要么是第三方的，不应由我们再分发。

---

## 方案 A：发到 Hugging Face（与赛事同平台）

> ⚠️ **前提：必须先开 VPN。** 本机实测 `huggingface.co` 直连返回 **000（被墙）**，
> 而 `hf-mirror.com` 虽可达（200）但**只做下载加速、不支持上传**（`/api/models/*` 会 307 跳回主站）。
> 因此从国内网络发布到 HF，**没有 VPN 就走不通**。没有 VPN 请直接用方案 B（GitHub）。

本地仓库已就绪（`main` / 42 files / commit `d96d828`），**不需要重新建仓**，只差推送。

### 第 1 步：开 VPN，确认能通

```bash
curl -s -o /dev/null -w '%{http_code}\n' --max-time 15 https://huggingface.co
# 期望：200 或 307。仍是 000 说明 VPN 没生效。
```

### 第 2 步：建一个 **write** 权限的 token

打开 https://huggingface.co/settings/tokens → **New token** → Type 选 **Write** → 复制 `hf_...`。

> 注意：我们提交评测用的那个会话 Cookie 里的 `hf_oauth_...` 只有 `read-repos`，**不能建仓库、不能推送**。

### 第 3 步：发布（二选一）

**A1（推荐，零依赖，只用 git）**

```bash
cd /d/LiDAR2026
export HF_USER=wind20011911          # 你的 HF 用户名
export HF_TOKEN=hf_xxxxxxxxxxxx      # 第 2 步的 write token
bash release_ccby4/push_hf_git.sh     # 默认仓库名 lidar2026-track3-lod2
```

`push_hf_git.sh` 会：查连通性 → 校验 token → `git push`（首次推送**自动创建公开仓库**）→ 打印公开地址。
token **不会**写进 `.git/config`（脚本刻意不用 `git remote add`，走一次性 URL）。

**A2（用 huggingface_hub，需先装依赖）**

```bash
# pip 源可达（实测 pypi 200），不需要 VPN 也能装
D:/Python39/python.exe -m pip install -U huggingface_hub \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_TOKEN=hf_xxxxxxxxxxxx
D:/Python39/python.exe -u release_ccby4/push_hf.py \
    --repo wind20011911/lidar2026-track3-lod2
```

### 第 4 步：拿到地址后收尾

公开地址形如 `https://huggingface.co/<用户名>/lidar2026-track3-lod2`。
把它替换进这两处的 `<CODE_URL_PLACEHOLDER>`：

- `docs/赛道三_方法说明文档.md`（§10 许可与数据来源）
- `docs/赛道三_扩展摘要_4页.md`（§2 数据与合规性 或 §7 结论处）

然后在竞赛平台/邮件中补交该地址。

## 方案 C：发到 Gitee 码云（**无 VPN 时推荐**）

国内直连可用（gitee.com 200 / 1.5s），**不需要 VPN**，公开仓库对境外同样可访问。

**第 1 步**：登录 https://gitee.com → 右上角头像 → **设置** → **私人令牌**（https://gitee.com/profile/personal_access_tokens）
→ **生成新令牌** → 权限至少勾 **`projects`**（仓库读写）→ 复制令牌。

**第 2 步**：一键发布

```bash
cd /d/LiDAR2026
export GITEE_USER=wind20011911          # 你的 Gitee 登录名（不是昵称）
export GITEE_TOKEN=你的私人令牌
bash release_ccby4/push_gitee.sh        # 默认仓库名 lidar2026-track3-lod2
```

`push_gitee.sh` 会：查连通性 → 校验令牌 → 通过 API **自动创建公开仓库**（已存在则跳过）
→ `git push` → 打印公开地址。token 不会写进 `.git/config`。

**第 3 步**：公开地址形如 `https://gitee.com/<用户名>/lidar2026-track3-lod2`，
替换进两处 `<CODE_URL_PLACEHOLDER>`（见文末"第 4 步"）。

> 也可用 **手动**方式：在 https://gitee.com/projects/new 建一个公开仓库（不勾初始化），然后
> `git remote add origin https://gitee.com/<用户名>/lidar2026-track3-lod2.git && git push -u origin main`，
> 弹窗时用户名填 Gitee 用户名、密码填私人令牌。

---

## 方案 B：发到 GitHub（需 VPN）

**前置**：GitHub 需 VPN。本机 VPN 下实测：`api.github.com` 200(1.1s)、
`github.com` 200(8s)、`git ls-remote https://github.com/git/git.git HEAD` **成功返回 HEAD** ⇒ git 通道可用。

**第 1 步：建 PAT**

https://github.com/settings/tokens → **Generate new token (classic)** → Scope 勾 **`repo`**
（`repo` 同时覆盖"建公开仓库 + 推送"，最省事，推荐）。

> **实测坑（2026-10-07）**：**Fine-grained PAT 默认权限不足以建仓库**。
> 用 `windson-kl` 的 fine-grained token 测试：`GET /user` 正常（200），但
> `POST /user/repos` 返回 **403 `Resource not accessible by personal access token`**，
> 且 `GET /user/repos` 返回 **`[]`**（没有任何仓库访问权）。
> ⇒ 若坚持用 fine-grained，必须改三处：
> ① Repository access 选 **All repositories**；
> ② Permissions → Repository permissions → **Administration: Read and write**（建仓必需）；
> ③ **Contents: Read and write**（推送必需）。
> 改权限比换 token 麻烦，**建议直接用 Classic `repo`**。
> 另注：Classic token 生成入口是 https://github.com/settings/tokens/new （不带 `?type=beta`）。

**第 2 步：一键发布**（二选一）

*2a — 环境变量*

```bash
cd /d/LiDAR2026
export GITHUB_USER=你的GitHub用户名
export GITHUB_TOKEN=ghp_xxxxxxxx        # 第 1 步的 PAT
bash release_ccby4/push_github.sh       # 默认仓库名 lidar2026-track3-lod2
```

*2b — 凭据文件（token 不进命令行、不留 history）*

```bash
printf '%s\n' '你的GitHub用户名' > D:/LiDAR2026/tmp/.gh_user
printf '%s\n' 'ghp_xxxxxxxx'      > D:/LiDAR2026/tmp/.gh_token
bash release_ccby4/push_github.sh
```

> `D:/LiDAR2026/tmp/` **在仓库之外**且已被 `.gitignore` 忽略，凭据绝不会被推送到公网。

`push_github.sh` 会：查连通性 → 校验 token → 通过 API **自动创建公开仓库**（已存在则跳过）
→ `git push` → 打印公开地址。token 不会写进 `.git/config`。

**第 3 步**：公开地址形如 `https://github.com/<用户名>/lidar2026-track3-lod2`，
替换进两处 `<CODE_URL_PLACEHOLDER>`。

> 也可用**纯手动**方式（不用脚本）：
> ```bash
> cd /d/LiDAR2026/release_ccby4
> git remote add origin https://github.com/<用户名>/lidar2026-track3-lod2.git
> git push -u origin main      # 提示登录时：用户名填 GitHub 用户名，密码填 PAT
> ```
> 需先在 https://github.com/new 建好 Public 仓库（**不要**勾 Add README/.gitignore/license）。

---

## 发布后要做的两件事

1. **把公开地址补进方法说明文档与扩展摘要**（评审要求"提供代码"，通常要能点开）。
   在 `docs/赛道三_方法说明文档.md` 的 §7/附录里加一行 `Code: <URL>`。
2. **把地址填到竞赛平台的提交说明/评论区**（或按组织方邮件要求补交）。

## 待办勾选

- [ ] **无 VPN** → 建 Gitee 私人令牌，执行方案 C
- [ ] **有 VPN** → 建 HF `write` token（方案 A）或 GitHub PAT（方案 B）
- [ ] 执行所选方案，拿到公开 URL
- [ ] 把 URL 写进方法说明文档与扩展摘要（替换 `<CODE_URL_PLACEHOLDER>`）
- [ ] 在竞赛平台/邮件里补交代码地址
