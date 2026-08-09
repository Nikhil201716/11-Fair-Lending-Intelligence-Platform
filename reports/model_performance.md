# Credit Risk Model Performance

Neither variant uses `demographic_group` as an input.

| Variant | Model | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|
| with_proxy | logistic_regression | 0.7114 | 0.3058 | 0.10731 |
| with_proxy | gradient_boosting | 0.7098 | 0.3033 | 0.10749 |
| without_proxy | logistic_regression | 0.7114 | 0.3058 | 0.10731 |
| without_proxy | gradient_boosting | 0.7094 | 0.3022 | 0.10754 |

AUC cost of removing the geographic proxy features (gradient boosting): **+0.00040**
