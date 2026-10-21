# LoD2 building reconstruction from airborne LiDAR — 10th National LiDAR Conference, Track 3

Team **wind** · 第十届全国激光雷达大会 数据处理大赛 · 赛道三（LoD2 建筑物三维重建）

This repository contains the complete pipeline that produces our final competition
submission. It is released under **CC BY 4.0** (see `LICENSE`), as required by the
competition's award conditions.

本仓库是生成参赛提交结果的完整可复现代码，按赛事要求以 CC BY 4.0 发布。

---

## 1. Task and result / 任务与结果

**Task.** Given an airborne LiDAR point cloud for each of 4,000 test entries, produce a
LoD2 building mesh (`<id>.obj`). The leaderboard score is

```
FINAL = 0.6 × CD + 0.4 × ECD
```

where `CD` is Chamfer distance and `ECD` is Edge Chamfer distance, both computed in the
**original coordinate frame, in metres**. The organiser's `evaluate.py` has the
centroid-removal and scale-normalisation lines commented out; any pipeline that
normalises coordinates will inflate CD by roughly the tile's diagonal.

**Result.** `FINAL = 3.56886` (CD 2.59888, ECD 5.02382), against an official baseline of
9.8437. The submission consists of 4,000 flat `.obj` files in a single zip.

The test tiles are scene tiles, not single buildings. The largest contains 933,386 points
and spans 626 × 510 × 655 m. Median tile diagonal over the set is 28.8 m, p90 is 104.4 m.
One pipeline therefore has to handle both compact tiles and whole-block tiles, whose
optimal parameters differ.

---

## 2. Pipeline / 流程

```
test .xyz (4000)
  │
  ├─[1] src/prepare.py              .xyz → .ply + geometry report (count / extent / density)
  │
  ├─[2] City3D (modified CLI_Example_2)   candidate planes → integer-programming face selection
  │       src/c3run.py              run wrapper (encoding, timeout tree-kill, concurrency)
  │       src/run_batch.py          batch driver
  │
  ├─[3] src/fallback.py             pure-geometry fallback (guarantees 100 % coverage)
  │
  ├─[4] src/postproc.py             outlier faces (CD) · outward normals (NC) ·
  │                                 principal-axis orthogonalisation (ECD) · coplanar merge
  │
  ├─[5] src/qc_repair.py            structural QC + targeted repair with an alternate config
  │
  ├─[6] src/_t_build_walls.py       walls: footprint side walls only — no roof, no floor
  │
  ├─[7] src/_t_apply_clean.py       tail cleanup: z-clip fuse + far/sliver removal
  │
  ├─[8] src/_t_dsm_apply.py         gated DSM: grid-sampled z-height field, Delaunay,
  │                                 long-edge filter, appended as text
  │
  ├─[9] src/_t_mads_apply.py        multi-axis height fields: x/y min∪max shells, with
  │                                 grid coarsening on large tiles
  │
  ├─[10] src/_t_creasesplit_apply.py  crease split at dihedral > 30° (topology only)
  │
  └─[11] src/package.py             → submission.zip (4000 .obj, flat at archive root)
```

`src/verify_submission.py` is the pre-submission format checker: it verifies entry count,
that every entry parses, that no mesh is degenerate, and that the bounding-box ratio
`mesh_diag / cloud_diag` falls in [0.85, 1.20].

Steps 6 to 10 only use the sample's own point cloud and its own mesh. They never read
ground truth, and they write by appending text or remapping vertex indices, so the input
geometry is never destroyed — if an operator fails, the result degrades to its input.

---

## 3. Reproduce / 复现

### 3.1 Environment

```
Python 3.9
numpy 2.0.2 / scipy 1.13.1 / trimesh 4.9.0
```

City3D must be built from its own public source; we use a lightly modified CLI so that
the parameters below can be passed on the command line.

```bash
pip install numpy scipy trimesh
```

### 3.2 Steps

```bash
# [1] preprocess
python src/prepare.py --src <test_xyz_dir> --dst data/work/ply

# [2]-[5] main reconstruction, fallback, post-processing, QC
#         run_batch chains these stages; the candidate-face cap goes 600 -> 5000
python src/run_batch.py --stage all --src <test_xyz_dir> --work data/work \
       --jobs 8 --max-faces 5000
#         output lands in data/work/final

# [6] walls: side walls only, z0 = p2, z1 = p30 of the cloud inside each footprint
python src/_t_build_walls.py --mode build --base data/work/final --out data/work/walls \
       --z0q 2 --z1q 30 --jobs 8

# [7] tail cleanup
python src/_t_apply_clean.py --src data/work/walls --dst data/work/clean \
       --ply data/work/ply --mode far_sliv --sliv 40 --jobs 8

# [8] gated DSM (3 m grid, long-edge cap = 3g = 9 m, gate at 40 m diagonal)
python src/_t_dsm_apply.py --meshd data/work/clean --plyd data/work/ply \
       --outd data/work/dsm --g 3.0 --diag-min 40 --jobs 5

# [9] multi-axis height fields, with grid coarsening above 80 m diagonal
python src/_t_mads_apply.py --meshd data/work/dsm --plyd data/work/ply \
       --outd data/work/mads --axes x,y --g 1.5 --diag-min 0 \
       --g-big 3.0 --diag-split 80 --face-cap 40000 --cell-cap 100000 \
       --jobs 6 --report logs/apply_mads.json

# [10] crease split at 30° (same threshold as the organiser's sharp-edge criterion)
python src/_t_creasesplit_apply.py --meshd data/work/mads --outd data/work/cs \
       --crease 30 --jobs 6

# [11] package + verify
python src/package.py --src data/work/cs --out out/submission.zip
python src/verify_submission.py --zip out/submission.zip --ply data/work/ply --expect 4000
```

All stages work in the original coordinate frame. `run_batch.py` calls `fallback.py`,
`postproc.py` and `qc_repair.py` internally; those three are modules, not standalone CLIs.

### 3.3 What is *not* included

* The organiser's evaluation script and the BuildingWorld dataset — both must be obtained
  from the competition platform. Our own metric implementation is `src/metrics.py`.
* The 4,000 test point clouds and the produced meshes.
* City3D itself (third-party, under its own licence).

---

## 4. Method notes / 方法要点

1. **Calibrate the metric before changing the method.** We re-implemented the organiser's
   metrics and verified, sample by sample, that the online score uses the original metric
   frame. Everything derived earlier under normalised coordinates was discarded.

2. **The candidate-face cap is a branch switch, not a threshold.** City3D's default of 600
   is never reached on a single building, but on a scene tile the accumulated candidate
   planes exceed it and the solver silently falls back to a compromise path that drops the
   vertical-direction constraint, producing ground-hugging sheets instead of walls. Raising
   it to 5000 removes the fallback. There is no useful intermediate value.

3. **Add side walls only.** Adding a closed prism (walls plus roof) is negative on every
   test set we tried, while adding side walls alone is a net gain with CD1 and CD2 both
   decreasing. The damage comes from the roof: it introduces a large horizontal surface
   that sits further from the reference than the gap it fills.

4. **Where the reference is dense, that is where to add.** The reference meshes average 871
   faces against roughly 101 for a typical LoD2 output, and about 89 % of their area is
   near-vertical. A z-projected height field cannot represent a wall's mid-section at all,
   which is why x/y-projected fields contribute independently and their gains add.

5. **Gate by scale, not by sample.** DSM gain correlates with cloud diagonal at −0.826
   (Spearman −0.749); it is near zero on compact tiles and large on scene tiles. The gate
   at 40 m is what keeps the operator from adding useless faces on small samples.

6. **Keep edges local.** Offline ΔECD does not transfer to the online score at a constant
   rate — the same offline change has been observed to produce −0.328 and, elsewhere,
   +0.111. Any change that introduces long edges unrelated to the reference structure is
   rejected before submission, even if the offline metric improves.

7. **Known limits.** We do not segment scene tiles into per-building clouds, so buildings
   with small gaps and similar roof heights tend to be merged into one polyhedron. On the
   60-sample benchmark, the worst 10 % of samples account for 36.0 % of the total FINAL
   sum, and every operator we tried on that tail was a net loss against ground truth.

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
