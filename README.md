# kin8nm: baseline MLP vs improved LightGBM/MLP blend

Question: can a cross-validated, gradient-boosted regressor give lower and more uniform error than a standard MLP baseline on a public stand-in for sensor-based regression (OpenML kin8nm, robot-arm kinematics)?

Run: `pip install -r requirements.txt && python analysis.py`

What it does:
- Loads OpenML dataset 507 (kin8nm) and makes one fixed 80/20 split (seed 42). Scalers are fit on the training data only.
- Baseline: `MLPRegressor` with two hidden layers of 64 units, early stopping.
- Improved: a blend of LightGBM and the same MLP. LightGBM uses sine/cosine features and a 5-fold CV grid search on the training set, which includes linear leaves. The blend weight is chosen on out-of-fold training predictions.
- Three model seeds (42, 43, 44) for both models. Test MAE is the main metric, and results go to `results.json`.

The test set is used only for the final evaluation. It is not used for any tuning or weight selection.

Caveat: kin8nm is simulated robot-arm data, not contact-force data. It is a stand-in for the original paper's dataset, which is not public.

## Results (latest run)

On the public OpenML kin8nm dataset, a standard scikit-learn MLP regressor reached a test MAE of 0.078, while a CV-tuned LightGBM–MLP blend reached 0.075 on the same held-out split, a small gain.

| Metric | Value |
|---|---|
| baseline_score | 0.07818 |
| improved_score | 0.07474 |
| improvement | 0.003444 |
| baseline_score_std | 0.0006023 |
| improved_score_std | 0.0003206 |
| baseline_rmse | 0.1048 |
| improved_rmse | 0.1016 |
| baseline_r2 | 0.734 |
| improved_r2 | 0.7503 |
| baseline_abs_err_std | 0.06979 |
| improved_abs_err_std | 0.06875 |
| blend_weight_lightgbm | 0.4 |
| cv_oof_mae_blend_scaled | 0.3736 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: kin8nm is a simulated robot-arm dataset, not contact-force data; it is a stand-in only. The paper's neuromorphic-camera CRNN was not reproduced; the MLP is a generic baseline. Single train/test split; seeds vary model randomness only. Hyperparameters and blend weight chosen by CV on training data; test set used once. The first LightGBM-only attempt did not beat the MLP on this smooth target, so tree leaves were made linear and the MLP was blended in. Secondary metrics are RMSE, R^2 and the std of absolute error (uniformity proxy); classification F1 does not apply to regression.

## Reference

This project is motivated by the following paper:

> F Baghaei Naeini, D Makris, D Gan, Y Zweiri (2020). *Dynamic-vision-based force measurements using convolutional recurrent neural networks*. Sensors. https://www.mdpi.com/1424-8220/20/16/4469

Paper author: Prof. Dimitrios Makris, Kingston University London (d.makris@kingston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
