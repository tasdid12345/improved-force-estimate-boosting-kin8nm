# Gradient-boosted vs MLP regression on kin8nm (force-regression stand-in)

The paper "Dynamic-vision-based force measurements using convolutional recurrent neural networks" (2020) estimates contact forces from dynamic vision data recorded by a neuromorphic camera, using a convolutional recurrent network. Its recordings are not public, so this experiment asks a related, testable question on a public robot-arm dataset: whether a cross-validated gradient-boosted model can beat a standard neural-network baseline on both average and spread of error when regressing a continuous physical quantity from sensor-style readings.

**Question:** Can a cross-validated, gradient-boosted regressor give lower and more uniform error than a standard MLP baseline when estimating a continuous physical quantity from robot-arm sensor readings on a public stand-in dataset?

**Dataset:** OpenML kin8nm (robot-arm kinematics): eight joint-angle inputs and a continuous target, downloaded via the `openml` package. It is a stand-in for the paper's contact-force data and is not the paper's own dataset.

**Baseline:** scikit-learn `MLPRegressor` with standardised inputs and target. It replaces the paper's convolutional recurrent network, which is not specified in enough detail to reproduce.

**Improved model:** LightGBM regressor whose hyperparameters are chosen by k-fold cross-validation on the training split only, with sine and cosine features added for each input.

Both models use the same single train/test split, preprocessing, evaluation function and seeds. The main metric is test MAE; RMSE, R² and the spread of absolute error are secondary.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts saved to `figures/`).

## Reference

This project is motivated by the following paper:

> F Baghaei Naeini, D Makris, D Gan, Y Zweiri (2020). *Dynamic-vision-based force measurements using convolutional recurrent neural networks*. Sensors. https://www.mdpi.com/1424-8220/20/16/4469

Paper author: Prof. Dimitrios Makris, Kingston University London (d.makris@kingston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
