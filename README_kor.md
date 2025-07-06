[English](/README.md)

---
### [Boost up AI 2025 : 신약 개발 경진대회](https://dacon.io/competitions/official/236518/overview/description)
2025.06.23 ~ 2025.07.31
## 개요
**주제**

인체 내 약물 대사에 관여하는 CYP3A4 효소 저해 예측모델 개발

**설명**

화합물의 구조 및 CYP3A4 효소 저해율(%inhibition)에 대한 학습용 데이터 1,681종을 이용해 예측 모델을 개발

개발한 예측 모델로 경진대회 평가 데이터를 사용하여 예측한 값을 제출

**주최 / 주관 / 후원 / 운영**

주최 : [한국화학연구원](https://www.krict.re.kr/), [한국생명공학연구원](https://www.kribb.re.kr/kor/main/main.jsp)

주관 : [한국화합물은행](https://chembank.org/), [KOBIC](https://www.kobic.re.kr/kobic/)

후원 : [과학기술정보통신부 ](https://www.msit.go.kr)

운영 : 데이콘

**참가 대상**

신약개발과 인공지능에 관심있는 전국민 누구나

## 규칙
**리더보드**

- 평가 산식
  - A = Inhibition(%)의 Normalized RMSE 오차, 예측 정확도 측정
    
    $\text{Normalized RMSE}=\frac{\text{RMSE}}{\text{max}(y)-\text{min}(y)}$
  - B = 예측값과 실제값 간의 선형 상관관계를 측정, 예측값이 실제값의 변화 경향성을 잘 반영하는지 측정

    $\text{Pearson Correlation Coefficient}=\text{clip}(\frac{\text{Cov}(y,\hat{y})}{\delta_y \cdot \delta_{\hat{y}}}),0,1$
  - Score = 0.5 x (1 - min(A, 1)) + 0.5 x B
- Public score : 전체 테스트 데이터 중 사전 샘플링된 50% 중 리더보드 제출 시 무작위로 80% 샘플을 선정하여 평가
- Private score : 전체 테스트 데이터 중 나머지 50%
※ Public score와 Private score 점수 간 점수 차이(shake-up)가 발생할 수 있으며, 1차 평가는 Private Score 결과를 기준으로 결정됨
