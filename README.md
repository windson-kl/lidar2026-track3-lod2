# LoD2 building reconstruction from airborne LiDAR — 10th National LiDAR Conference, Track 3

Team **wind** · 第十届全国激光雷达大会 数据处理大赛 · 赛道三（LoD2 建筑物三维重建）

This repository is the complete, self-contained code that generates our competition
submissions. It is released under **CC BY 4.0** (see `LICENSE`), as required by the
competition's award conditions.

本仓库是生成参赛提交结果的**完整可复现代码**，按赛事要求以 **CC BY 4.0** 发布。

---

## 1. What this is / 这是什么

**Task.** Given an airborne LiDAR point cloud for each of 4,000 test entries, produce a
LoD2 building mesh (`<id>.obj`). The leaderboard score is

```
FINAL = 0.6 × CD + 0.4 × ECD
```

where `CD` is Chamfer distance and `ECD` is Edge Chamfer distance, both computed in the
**original coordinate frame, in metres** (the organiser's `evaluate.py` has the
normalisation lines commented out — see `docs/赛道三_方法说明文档.md` §2.1).

**Result.** Best public-leaderboard submission **`v37`**: `FINAL = 3.83416`
(CD 2.96191 / ECD 5.14252) — a **−25.8 %** improvement over our first valid submission
(5.16638). The winning operator is the **multi-axis min∪max height shell** (see §2 step 9),
which lifts `v29` (4.00841) to 3.83416 with **both** CD and ECD decreasing.

---

## 2. Pipeline / 流程

```
test .xyz (4000)
  │
  ├─[1] src/prepare.py            .xyz → .ply + geometry report (points / size / density)
  │
  ├─[2] City3D (modified CLI_Example_2)   footprint → segmentation → roof plane extraction
  │       src/c3run.py            run wrapper (encoding / timeout tree-kill / concurrency)
  │       src/run_batch.py        end-to-end batch driver
  │
  ├─[3] src/fallback.py           pure-geometry fallback (guarantees 100 % coverage)
  │
  ├─[4] src/postproc.py           metric-oriented post-processing
  │                               outlier-face removal (CD) · outward normals (NC)
  │                               principal-axis + right angles (ECD) · coplanar merge (F_Ratio)
  │
  ├─[5] src/qc_repair.py          QC + targeted repair (detect exploded meshes, re-run with
  │                               an alternate config ladder, keep the better one)
  │
  ├─[6] src/_t_build_walls.py     ★ walls operator (v10 core): add footprint side walls only
  │                               — no roof, no floor
  │
  ├─[7] src/_t_apply_clean.py     zero-risk tail cleanup (v12): z-clip fuse + far&sliver removal
  │
  ├─[8] src/_t_dsm_apply.py       ★ gated DSM densification (v21→v28→v29 / v35 / v36)
  │                               z-projected min∪max height blocks over the input cloud
  │
  ├─[9] src/_t_mads_apply.py      ★ multi-axis min∪max height shells (v37, the winning lever)
  │                               x/y-projected min/max height fields → appends 4 shells,
  │                               covering the vertical walls that a z-height field misses
  │
  └─[10] src/package.py           → submission.zip (4000 .obj, flat at archive root)
```

`src/_t_build_v9r.py` rebuilds the `v9r` base from the City3D output before the walls/cleanup
steps; `src/verify_submission.py` is the pre-submission format checker.

---

## 3. Reproduce / 复现

### 3.1 Environment

```
Python 3.9
numpy 2.0.2 / scipy 1.13.1 / trimesh 4.9.0
```

City3D must be built separately from its own public source (see below); we use a lightly
modified CLI so that the parameters below can be passed on the command line.

```bash
pip install numpy scipy trimesh
```

### 3.2 Steps

```bash
# [1] preprocess
python src/prepare.py --src <test_xyz_dir> --out data/work/ply

# [2] main reconstruction (City3D)
python src/run_batch.py --ply data/work/ply --out data/work/recon --mode cloud

# [3][4][5] fallback + postprocess + QC repair
python src/fallback.py   --recon data/work/recon --ply data/work/ply --out data/work/fb
python src/postproc.py   --src data/work/fb --out data/work/pp
python src/qc_repair.py  --src data/work/pp --out data/work/final_v9r

# [6] walls operator (the single operator that is a net gain against ground truth)
python src/_t_build_walls.py --mode build --base data/work/final_v9r \
       --out data/work/final_v10 --z0q 2 --z1q 30 --jobs 8

# [7] tail cleanup (optional, near-neutral: see docs)
python src/_t_apply_clean.py --src data/work/final_v10 --dst data/work/final_v12 \
       --ply data/work/ply --mode both --sliv 40 --jobs 8

# [8] gated DSM densification  (v21 → v28 → v29; keep max_edge = 3g)
python src/_t_dsm_apply.py --meshd data/work/final_v20 --plyd data/work/ply \
       --outd data/work/final_v21 --g 3.0 --diag-min 30 --zmode min --jobs 5
python src/_t_dsm_apply.py --meshd data/work/final_v21 --plyd data/work/ply \
       --outd data/work/final_v28 --g 3.0 --zmode max --diag-min 30 --jobs 6
python src/_t_dsm_apply.py --meshd data/work/final_v28 --plyd data/work/ply \
       --outd data/work/final_v29 --g 2.0 --zmode mm  --diag-min 30 --jobs 6

# [9] multi-axis min∪max wall shells  (v37 = our best submission)
python src/_t_mads_apply.py --meshd data/work/final_v29 --plyd data/work/ply \
       --outd data/work/final_v37 --axes x,y --g 2.0 --diag-min 30 \
       --face-cap 200000 --cell-cap 400000 --jobs 6

# [10] package + verify
python src/package.py --src data/work/final_v37 --out out/submission_v37_xy.zip
python src/verify_submission.py --zip out/submission_v37_xy.zip --ply data/work/ply --expect 4000
```

`--mode cloud` is important: **no coordinate normalisation** — the score is computed in the
original metric frame.

### 3.3 What is *not* included

* The organisers' evaluation script `official_evaluate.py` and the BuildingWorld dataset.
  Both must be obtained from the competition platform. Our own metric code is
  `src/metrics.py`; place the official script alongside the pipeline if you want to
  reproduce the ground-truth validation numbers in `docs/`.
* The 4,000 test point clouds and the produced meshes.
* City3D itself (third-party, its own licence).

---

## 4. Method notes / 方法要点

1. **Calibrate the ruler first.** We re-implemented the organiser's metrics and verified
   that the online score uses the *original* metric coordinate frame. All conclusions
   obtained earlier under normalised coordinates were discarded and re-derived.
2. **Decompose CD.** Splitting CD into its two directions showed that the dominant error
   channel is the *mesh → cloud* direction, which is what the walls operator attacks.
3. **Ground-truth-free proxies, validated for significance.** `p2m`, `p2m_p90` and `m2p`
   are computed without ground truth and used as pre-submission gates.
4. **Two hard rules learned from ground-truth experiments**
   (see `docs/赛道三_方法说明文档.md` §6):
   * deleting geometry always improves one CD direction and destroys the other — it is
     essentially never a net gain;
   * adding a *closed prism* (walls **plus** roof) is strongly negative, while adding
     **side walls only** is a net gain. The roof is where the damage happens.
5. **Multi-axis beats single-axis upsampling.** Refining the z-axis height layers only
   (1.5 m / 1.0 m) bought −0.015 / −0.091 FINAL; adding **x/y min∪max height shells** (v37)
   at the same budget bought **−0.174** — about **1.9×** the best z-axis result. The reason is
   structural: ≈89 % of the ground-truth area is near-vertical wall, which a z-projected height
   field cannot represent at all. *Spend the budget where the information is missing, not where
   it is already dense.*
6. **Known limits.** On scene-scale tiles (a whole city block delivered as one entry) our
   mesh degenerates into a bounding-box-hugging shell; `ECD` reflects this. The principled
   fix is to segment the scene into per-building clouds before reconstruction — a
   structural change we did not have time to complete. This is stated as a limitation in
   the extended abstract.

`src/_t_ecd_risk.py`, `src/_t_check_ab.py`, `src/_t_walls_cfg.py` and
`src/_t_walls_cfg_cmp.py` are the diagnostic tools used to reach these conclusions; they
are included so the analysis can be re-run.

---

## 5. Citation / 引用

If you use this code, please cite the BuildingWorld dataset:

```bibtex
@inproceedings{huang2026buildingworld,
  title={Buildingworld: A structured 3d building dataset for urban foundation models},
  author={Huang, Shangfeng and Wang, Ruisheng and Wang, Xin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40}, number={7}, pages={5085--5094}, year={2026}
}
```

## 6. Licence

CC BY 4.0 — see `LICENSE`. Contact: team **wind**, 第十届全国激光雷达大会 赛道三.
