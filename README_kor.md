[English](/README.md)

---
### [Boost up AI 2025 : 신약 개발 경진대회](https://dacon.io/competitions/official/236518/overview/description)
2025.06.23 ~ 2025.07.31
## 개요
### 주제
MAP3K5 IC50 활성값 예측 모델 개발

### 설명
PubChem, ChEMBL, CAS 등에서 수집한 실험 기반 화합물 정보를 기반으로 IC50값 예측모델 개발

127종 화합물의 구조 정보를 입력값으로 사용하여 해당 화합물들의 ASK1에 대한 IC50 값을 예측하여 제출

### 주최 / 주관 / 운영
주최/주관 : 한국제약바이오협회
후원 : 보건복지부, 유한양행, CAS
운영 : 데이콘

### 참가 대상
전국민 누구나

## 규칙
**리더보드**

- 평가 산식
  - A = IC50(nM) 단위의 Normalized RMSE 오차, 예측 정확도 측정
 
    $\text{Normalized RMSE}=\frac{\text{RMSE}}{\text{max}(y)-\text{min}(y)}$    
  - B = pIC50 변환값을 기준으로 한 예측값과 실제값 간의 선형 상관관계의 제곱

    $R^2=\text{Pearson}(\text{pIC}\_{50}^{\text{true}}, \text{pIC}\_{50}^{\text{pred}})^2$
    - Score = 0.4 x (1 - min(A, 1)) + 0.6 x B
    - Public score : 전체 테스트 데이터 중 사전 샘플링된 40%
    - Private score : 전체 테스트 데이터 중 나머지 60%
