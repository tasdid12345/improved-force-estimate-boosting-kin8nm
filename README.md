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

## Reference

This project is motivated by the following paper:

> F Baghaei Naeini, D Makris, D Gan, Y Zweiri (2020). *Dynamic-vision-based force measurements using convolutional recurrent neural networks*. Sensors. https://www.mdpi.com/1424-8220/20/16/4469

Paper author: Prof. Dimitrios Makris, Kingston University London (d.makris@kingston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
