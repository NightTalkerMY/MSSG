# CATT-ASD experiment order

The selected configuration is CE+MSE with CE weight 0.8. Columbia is excluded
from training and checkpoint selection.

1. Finish graph feature extraction stages 2 through 8.
2. Run `bash run_catt_plan.sh audit-inputs`.
   Then run `bash run_catt_plan.sh audit-inputs-deep` before archiving crop folders.
3. Run `bash run_catt_plan.sh prepare-ava-labels`.
4. Run `bash run_catt_plan.sh build-val-cache` as the smaller cache check.
5. Run `bash run_catt_plan.sh build-train-cache`.
6. Run `bash run_catt_plan.sh audit-caches`.
7. Run `bash run_catt_plan.sh smoke` and verify one epoch, validation,
   checkpoint saving and the run manifest.
8. Run `bash run_catt_plan.sh single` for the seed-42 CE+MSE pilot.
9. Review its curve and runtime before running
   `bash run_catt_plan.sh single-ablation`.
10. Evaluate each selected checkpoint on Columbia only after AVA-based selection.
11. Run `bash run_catt_plan.sh multi-selected --notify` for paired seeds 42,
    123 and 2026. The command skips completed runs. Do not use the legacy
    `multi` command, which retains the earlier 0.4 weight only for provenance.
12. After all paired runs finish, run
    `bash run_catt_plan.sh remaining-experiments --notify`.

The resumable `remaining-experiments` command performs, in order:

1. Missing AVA and Columbia evaluation for all selected and CE-only seeds.
2. Three-seed mean, sample standard deviation, and AVA-frozen Columbia
   threshold analysis.
3. A fast forward/backward architecture smoke test.
4. Seed-42 retraining and immediate evaluation without buffer attention.
5. Seed-42 retraining and immediate evaluation without cross-speaker attention.
6. A parameter-controlled seed-42 non-causal comparison that reuses the same
   GRU weights in forward and reverse temporal order.
7. Future-input perturbation tests for all causal seeds and the non-causal
   comparison.
8. Three-seed audio/video lag evaluation at -10, -5, -2, -1, 0, 1, 2, 5 and
   10 frames.
9. A consolidated JSON result manifest.

Every stage skips an existing completed output. The two required retrained
component ablations run before the optional non-causal comparison, so an
interrupted command retains the highest-priority evidence first.

Prepare the held-out Columbia cache independently with:

1. `bash run_catt_plan.sh download-columbia`
2. Visually validate boxes with `columbia/validate_boxes.py`.
3. Run Columbia label correction stage 0, visually validate the corrected
   labels, then run data preparation stages 1, 2 and 4--8
   and graph feature stages 2--8.
4. `bash run_catt_plan.sh audit-columbia-inputs`
5. `bash run_catt_plan.sh build-columbia-cache`
6. `bash run_catt_plan.sh audit-columbia-cache`

Building this cache does not authorize its use for training or checkpoint selection.

The training program selects checkpoints using AVA final-frame validation mAP.
It never loads Columbia during training or checkpoint selection.
