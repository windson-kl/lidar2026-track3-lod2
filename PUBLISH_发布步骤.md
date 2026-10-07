# 发布步骤（CC BY 4.0 开源代码）

官方获奖条件原文（`docs/official/赛道三_官方_competition_info.txt`）：

> - 提交一份详细说明解决方案的技术说明文档；
> - 以扩展摘要（4 页）或完整论文的形式提交至组织者；
> - **提供能够生成获奖提交结果的代码；**
> - **代码需采用 CC BY 4.0 许可证发布。**

前两条已完成（`docs/赛道三_方法说明文档.md`、`docs/赛道三_扩展摘要_4页.md`），
**第三条（代码发布）目前还没做** —— 本地没有 git 仓库、也没有任何公开地址。
本目录 `D:\LiDAR2026\release_ccby4\` 就是为此打包好的发布包。

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

## 方案 A：发到 Hugging Face（推荐，与赛事同平台）

需要一个**有写权限**的 token（`hf_oauth_...` 的会话 token 只有 `read-repos`，不能建仓库）。
在 https://huggingface.co/settings/tokens 建一个 `write` token，然后：

```bash
pip install -U huggingface_hub
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
python release_ccby4/push_hf.py --repo <你的用户名>/lidar2026-track3-lod2 --token $HF_TOKEN
```

`push_hf.py` 会自动：建公开仓库 → 上传 LICENSE/README/src/docs →
打上 `license: cc-by-4.0` 标签 → 打印可引用的仓库地址。

## 方案 B：发到 GitHub

```bash
cd D:/LiDAR2026/release_ccby4
git init -b main
git add -A
git commit -m "LoD2 building reconstruction from airborne LiDAR (CC BY 4.0) - team wind"
gh repo create lidar2026-track3-lod2 --public --source=. --push
```

---

## 发布后要做的两件事

1. **把公开地址补进方法说明文档与扩展摘要**（评审要求"提供代码"，通常要能点开）。
   在 `docs/赛道三_方法说明文档.md` 的 §7/附录里加一行 `Code: <URL>`。
2. **把地址填到竞赛平台的提交说明/评论区**（或按组织方邮件要求补交）。

## 待办勾选

- [ ] 建 HF `write` token 或登录 GitHub
- [ ] 执行方案 A 或 B，拿到公开 URL
- [ ] 把 URL 写进方法说明文档与扩展摘要
- [ ] 在竞赛平台/邮件里补交代码地址
