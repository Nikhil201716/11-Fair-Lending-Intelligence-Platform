# SHAP Analysis, Validated Against Ground Truth

| Rank | Feature | Mean \|SHAP\| | True coefficient |
|---|---|---|---|
| 1 | num_delinquencies_2y | 0.390467 | 0.62 |
| 2 | debt_to_income | 0.313033 | 3.1 |
| 3 | credit_history_length_years | 0.300020 | -0.075 |
| 4 | log_income | 0.118333 | (derived) |
| 5 | income_vs_nbh | 0.062580 | (derived) |
| 6 | loan_to_income | 0.011324 | (derived) |
| 7 | employment_length_years | 0.007412 | (derived) |
| 8 | loan_amount | 0.006541 | (derived) |
| 9 | payment_to_income | 0.004550 | (derived) |
| 10 | neighborhood_risk_score | 0.004257 | 0.0 |
| 11 | nbh_median_income | 0.003901 | (derived) |
| 12 | nbh_avg_dti | 0.002737 | (derived) |
| 13 | loan_term_months | 0.000466 | (derived) |
| 14 | has_delinquency | 0.000000 | (derived) |
| 15 | thin_file | 0.000000 | (derived) |

## Validation checks

- **PASS** — debt_to_income (true coeff 3.10) ranks in the top 2
- **PASS** — neighborhood_risk_score (TRUE causal effect 0.0) ranks in the bottom half
- **PASS** — num_delinquencies_2y (true coeff 0.62) outranks the zero-effect proxy
