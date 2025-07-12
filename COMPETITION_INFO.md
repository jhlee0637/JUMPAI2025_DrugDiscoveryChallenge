[한국어](./README_kor.md)

---
### [Jump AI 2025: The 3rd AI Drug Development Competition](https://dacon.io/competitions/official/236530/overview/description)
2025.07.07 ~ 2025.08.25
## Overview
### Topic
Development of a prediction model for MAP3K5 IC50 activity value
### Description
Develop an IC50 value prediction model based on experimental compound information collected from PubChem, ChEMBL, CAS, etc.

Use the structural information of 127 types of compounds as input values to predict and submit the IC50 values for ASK1 of those compounds.

### Hosted / Organized / Managed
Hosted/Organized: [Korea Pharmaceutical and Bio-Pharma Manufacturers Association](https://www.kpbma.or.kr/eng)

Sponsored by: [Ministry of Health and Welfare](https://www.mohw.go.kr/eng/), [Yuhan](http://eng.yuhan.co.kr/Main/), CAS

Managed by: Dacon

### Eligibility
Open to everyone

## Rules
**Leaderboard**
 - Evaluation Metric
   - A = Normalized RMSE of Inhibition (%), measures prediction accuracy.
  
      $\text{Normalized RMSE}=\frac{\text{RMSE}}{\text{max}(y)-\text{min}(y)}$
   - B = The square of the linear correlation between the predicted value and the actual value based on the converted pIC50 value

     $R^2=\text{Pearson}(\text{pIC}\_{50}^{\text{true}}, \text{pIC}\_{50}^{\text{pred}})^2$
     - Score = 0.4 x (1 - min(A, 1)) + 0.6 x B
     - Public score: 40% of the total test data sampled in advance
     - Private score: The remaining 60% of the total test data

