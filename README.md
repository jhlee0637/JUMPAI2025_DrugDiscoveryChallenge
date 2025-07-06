[한국어](./README_kor.md)

---
### [Boost up AI 2025: New Drug Development Competition](https://dacon.io/competitions/official/236518/overview/description)
2025.06.23 ~ 2025.07.31
## Overview
**Topic**

Development of a predictive model for the inhibition of the CYP3A4 enzyme, which is involved in drug metabolism in the human body.

**Description**

Develop a predictive model using a training dataset of 1,681 compounds, which includes their chemical structures and CYP3A4 enzyme inhibition rates (% inhibition).

Submit the predicted values for the competition's evaluation data using the developed model.

**Hosted by / Organized by / Sponsored by / Managed by**

Hosted by: [Korea Research Institute of Chemical Technology (KRICT)](https://www.krict.re.kr/eng/), [Korea Research Institute of Bioscience and Biotechnology (KRIBB)](https://www.kribb.re.kr/kor/main/main.jsp)

Organized by: [Korea Chemical Bank](https://chembank.org), [Korean Bioinformation Center (KOBIC)](https://www.kobic.re.kr/kobic/?lang=en)

Sponsored by: [Ministry of Science and ICT (MSIT)](https://www.msit.go.kr/eng/index.do)

Managed by: DACON

**Participants**

Open to any Korean citizen interested in new drug development and artificial intelligence.

## Rules
**Leaderboard**
 - Evaluation Metric
   - A = Normalized RMSE of Inhibition (%), measures prediction accuracy.
  
      $\text{Normalized RMSE}=\frac{\text{RMSE}}{\text{max}(y)-\text{min}(y)}$
   - B = Measures the linear correlation between predicted and actual values, assessing how well the predictions reflect the trend of the actual values.
  
      $\text{Pearson Correlation Coefficient}=\text{clip}(\frac{\text{Cov}(y,\hat{y})}{\delta_y \cdot \delta_{\hat{y}}}),0,1$
     - Score = 0.5 x (1 - min(A, 1)) + 0.5 x B
 - Public score: Evaluated on 80% of a pre-sampled 50% of the total test data, randomly selected upon each submission to the leaderboard.
 - Private score: Based on the remaining 50% of the total test data.
※ A difference (shake-up) between the Public and Private scores may occur. The first evaluation round will be decided based on the Private Score results.
