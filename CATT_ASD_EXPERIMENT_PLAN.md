# CATT-ASD experiment order

Do not start multi-seed training until the single-seed results have been reviewed.

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
11. Run `bash run_catt_plan.sh multi` only after explicit approval and if time remains.

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
