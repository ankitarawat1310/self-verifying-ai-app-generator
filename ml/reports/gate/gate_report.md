# Learned release gate

135 Demo 1 runs (M0, M1, M2 × 15 test tasks × 3), 34 actually correct (25%). Leave-one-task-out cross-validation; features are decision-time signals only.

| Gate | Shipped | Correct shipped | Broken shipped | False assurance | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Rule (all 6 checks pass) | 59 | 21 | 38 | 64% | 0.36 | 0.62 | 0.45 | n/a |
| logistic regression, same # shipped as rule | 59 | 19 | 40 | 68% | 0.32 | 0.56 | 0.41 | 0.67 |
| logistic regression, threshold 0.5 | 45 | 17 | 28 | 62% | 0.38 | 0.50 | 0.43 | 0.67 |
| logistic regression, rule AND model | 30 | 13 | 17 | 57% | 0.43 | 0.38 | 0.41 | 0.67 |
| random forest, same # shipped as rule | 59 | 22 | 37 | 63% | 0.37 | 0.65 | 0.47 | 0.66 |
| random forest, threshold 0.5 | 33 | 11 | 22 | 67% | 0.33 | 0.32 | 0.33 | 0.66 |
| random forest, rule AND model | 28 | 8 | 20 | 71% | 0.29 | 0.23 | 0.26 | 0.66 |
| gradient boosting, same # shipped as rule | 59 | 22 | 37 | 63% | 0.37 | 0.65 | 0.47 | 0.71 |
| gradient boosting, threshold 0.5 | 20 | 9 | 11 | 55% | 0.45 | 0.27 | 0.33 | 0.71 |
| gradient boosting, rule AND model | 15 | 5 | 10 | 67% | 0.33 | 0.15 | 0.20 | 0.71 |

Best model: gradient_boosting, ROC-AUC 95% interval over tasks [0.591, 0.816].
Top features: generation_seconds (0.1104), own_mutation_score (0.1018), own_tests_pass_ratio (0.0225), tokens (0.0199), v_pytest_sandbox (0.0129), app_lines (0.0125).

Small data (135 runs, one LLM): treat as a first estimate. More runs from the fine-tuned models will be added to the same table.