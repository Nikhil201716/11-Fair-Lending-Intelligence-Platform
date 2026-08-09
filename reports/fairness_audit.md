# Fair Lending Audit

Approve when predicted default probability < 0.15.

| Variant | Approval A | Approval B | Disparate Impact | 4/5 rule | Parity diff | EO TPR gap |
|---|---|---|---|---|---|---|
| with_proxy | 0.719 | 0.639 | 0.8895 | PASS | 0.0794 | 0.0744 |
| without_proxy | 0.720 | 0.638 | 0.8870 | PASS | 0.0814 | 0.0761 |

- Removing the geographic proxy moved the disparate impact ratio **0.8895 -> 0.8870** at an AUC cost of **+0.00040**.
