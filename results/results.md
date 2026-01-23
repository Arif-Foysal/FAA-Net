## Dataset Statistics

=== Table D1: Dataset Statistics ===
| Split | Samples | Minority (Attack) | Majority (Normal) | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- |
| Train | 175341 | 119341 | 56000 | 0.469244 |
| Test | 82332 | 45332 | 37000 | 0.816200 |
| Total | 257673 | 164673 | 93000 | 0.564756 |

=== Table D2: Per-Attack Sample Distribution ===
| Attack Category | Train | Test | Total |
| :--- | :--- | :--- | :--- |
| 0 | 2000 | 677 | 2677 |
| 1 | 1746 | 583 | 2329 |
| 2 | 12264 | 4089 | 16353 |
| 3 | 33393 | 11132 | 44525 |
| 4 | 18184 | 6062 | 24246 |
| 5 | 40000 | 18871 | 58871 |
| 6 | 56000 | 37000 | 93000 |
| 7 | 10491 | 3496 | 13987 |
| 8 | 1133 | 378 | 1511 |
| 9 | 130 | 44 | 174 |



## FAIIA-IDS: Training Baselines (XGBoost & LightGBM)

Loading cached preprocessed data from faiia_preprocessed_data.pkl...

--- Training XGBoost ---

XGBoost Results:
------------------------------
  Accuracy       : 0.8961
  Precision      : 0.8856
  Recall         : 0.9317
  F1-Score       : 0.9081
  AUC-ROC        : 0.9765
  Avg Precision  : 0.9830

--- Training LightGBM ---
[LightGBM] [Info] Number of positive: 119341, number of negative: 56000
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.005765 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 4809
[LightGBM] [Info] Number of data points in the train set: 175341, number of used features: 33
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
[LightGBM] [Info] Start training from score 0.000000

LightGBM Results:
------------------------------
  Accuracy       : 0.8969
  Precision      : 0.8858
  Recall         : 0.9330
  F1-Score       : 0.9088
  AUC-ROC        : 0.9776
  Avg Precision  : 0.9839

=== Baseline Models Summary ===
          Accuracy  Precision    Recall  F1-Score   AUC-ROC  Avg Precision
XGBoost   0.896128   0.885632  0.931660  0.908063  0.976454       0.983042
LightGBM  0.896893   0.885831  0.932983  0.908796  0.977633       0.983866


## Figures I have for now 
- Convergence plot (Loss vs Epoch) (FAIIA)
- F1-Score vs Epoch (FAIIA)
- Recall vs Epoch (FAIIA)
- ROC Curve (4 ablation models)
- PR Curve (4 ablation models)
- confusion matrix (4 ablation models)

## Ablation results

=== Consolidated Evaluation Results of All Models ===
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FAIIA (EDAN v3) | 0.877532 | 0.856562 | 0.933976 | 0.893595 | 0.971650 | 0.979503 |
| XGBoost | 0.896128 | 0.885632 | 0.931660 | 0.908063 | 0.976454 | 0.983042 |
| LightGBM | 0.896893 | 0.885831 | 0.932983 | 0.908796 | 0.977633 | 0.983866 |
| Vanilla DNN + BCE | 0.894561 | 0.897421 | 0.912843 | 0.905067 | 0.972070 | 0.979862 |
| Vanilla DNN + Focal | 0.904290 | 0.944018 | 0.878254 | 0.909949 | 0.970721 | 0.978928 |
| FAIIA + BCE | 0.887808 | 0.870502 | 0.935388 | 0.901779 | 0.972523 | 0.976153 |
| FAIIA + Focal | 0.874095 | 0.843545 | 0.946969 | 0.892270 | 0.972186 | 0.979489 |


=== Per-attack metrics (Minority < 5000 vs Majority >= 5000) ===
| Attack ID | Category | Samples | Detection Rate | Type |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Normal(FPR) | 37000 | 0.191622 | Majority |
| 9 | Generic | 18871 | 0.998463 | Majority |
| 4 | Exploits | 11132 | 0.970715 | Majority |
| 6 | Fuzzers | 6062 | 0.609865 | Majority |
| 3 | DoS | 4089 | 0.979702 | Minority |
| 1 | Reconnaissance | 3496 | 0.957952 | Minority |
| 5 | Analysis | 677 | 0.994092 | Minority |
| 2 | Backdoor | 583 | 0.989708 | Minority |
| 8 | Shellcode | 378 | 0.917989 | Minority |
| 7 | Worms | 44 | 0.954545 | Minority |


## model complexity

=== Model Complexity Comparison ===
| Model | Parameters | Inference |
| :--- | :--- | :--- |
| Vanilla DNN | 54657 | Fast |
| FAIIA (EDAN v3) | 142436 | Moderate |

