#!/usr/bin/env python3
"""
ASK1 IC50 Predictor - GNN-Matched Direct Approach
직접적으로 GNN 결과를 참고하여 성능 향상
"""

import pandas as pd
import numpy as np
import warnings
import random
import os
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import joblib

# 시드 고정
SEED = 5
random.seed(SEED)
np.random.seed(SEED)

def IC50_to_pIC50(ic50_nM):
    """IC50 to pIC50 변환"""
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50):
    """pIC50 to IC50 변환"""
    return 10 ** (9 - pIC50)

class OptimizedMolecularFeaturizer:
    """최적화된 분자 특성 추출기"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=200)
        
    def compute_molecular_features(self, smiles):
        """분자 특성 계산"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(1000)
        
        features = []
        
        00# 1. 핵심 현재RDKit Descriptors
        try:
            features.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCsp3(mol),
                Descriptors.TPSA(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
                Descriptors.HallKierAlpha(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.MaxAbsPartialCharge(mol),
                Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol),
                Descriptors.NumAromaticCarbocycles(mol),
                Descriptors.NumAromaticHeterocycles(mol),
                Descriptors.NumSaturatedCarbocycles(mol),
                Descriptors.NumSaturatedHeterocycles(mol),
                Descriptors.NumAliphaticCarbocycles(mol),
                Descriptors.NumAliphaticHeterocycles(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.NumValenceElectrons(mol),
            ])
        except:
            features.extend([0.0] * 31)
        
        # 2. Morgan Fingerprints (다양한 설정)
        for radius in [2, 3]:
            for nBits in [512, 1024]:
                try:
                    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fp_array = np.array(fp)
                    features.extend(fp_array)
                except:
                    features.extend([0] * nBits)
        
        # 3. 원자 타입 카운트
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        
        for symbol in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']:
            features.append(atom_counts.get(symbol, 0))
        
        # 4. 결합 정보
        bond_counts = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0, 'AROMATIC': 0}
        for bond in mol.GetBonds():
            bond_type = str(bond.GetBondType())
            if bond_type in bond_counts:
                bond_counts[bond_type] += 1
        features.extend(bond_counts.values())
        
        # 5. 분자 복잡성 지표
        features.extend([
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            mol.GetNumHeavyAtoms(),
            len(Chem.GetMolFrags(mol)),
            mol.GetRingInfo().NumRings(),
        ])
        
        # 길이 맞춤
        if len(features) > 1000:
            features = features[:1000]
        else:
            features.extend([0.0] * (1000 - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def fit_transform(self, smiles_list, y=None):
        """특성 추출 및 변환"""
        print(f"분자 특성 추출 중... ({len(smiles_list)}개 분자)")
        
        X = np.array([self.compute_molecular_features(smiles) for smiles in smiles_list])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = self.scaler.fit_transform(X)
        
        if y is not None:
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled
        
        print(f"특성 추출 완료: {X_selected.shape[1]}개 특성")
        return X_selected
    
    def transform(self, smiles_list):
        """새로운 데이터 변환"""
        X = np.array([self.compute_molecular_features(smiles) for smiles in smiles_list])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        return X_selected

class PowerfulEnsemble:
    """강력한 앙상블 모델"""
    
    def __init__(self):
        self.models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=SEED,
                n_jobs=-1
            ),
            'et': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=SEED,
                n_jobs=-1
            )
        }
        
        self.weights = None
        
    def fit(self, X, y):
        """앙상블 훈련"""
        print("강력한 앙상블 훈련 중...")
        
        # 교차 검증으로 모델 성능 평가
        cv_scores = {}
        
        for name, model in self.models.items():
            print(f"  {name} 모델 훈련 중...")
            try:
                model.fit(X, y)
                scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                cv_scores[name] = -np.mean(scores)
                print(f"    CV MSE: {cv_scores[name]:.4f}")
            except Exception as e:
                print(f"    오류: {e}")
                cv_scores[name] = float('inf')
        
        # 가중치 계산 (MSE 역수)
        weights = {}
        for name, mse in cv_scores.items():
            if mse != float('inf'):
                weights[name] = 1.0 / (1.0 + mse)
            else:
                weights[name] = 0.0
        
        # 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in weights.items()}
        else:
            self.weights = {k: 1.0/len(self.models) for k in self.models.keys()}
        
        print(f"앙상블 가중치: {self.weights}")
        
    def predict(self, X):
        """앙상블 예측"""
        predictions = []
        
        for name, model in self.models.items():
            if self.weights[name] > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred * self.weights[name])
                except:
                    continue
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(X.shape[0])

class GNNAlignedPostprocessor:
    """GNN 결과에 정렬된 후처리기"""
    
    def __init__(self, gnn_data):
        self.gnn_values = gnn_data['ASK1_IC50_nM'].values
        self.gnn_mean = np.mean(self.gnn_values)
        self.gnn_std = np.std(self.gnn_values)
        
        print(f"GNN 통계: 평균={self.gnn_mean:.2f}, 표준편차={self.gnn_std:.2f}")
        
    def process_predictions(self, predictions):
        """GNN에 맞춘 후처리"""
        print("GNN 정렬 후처리 중...")
        
        # 1. 기본 클리핑
        predictions = np.clip(predictions, 0.1, 50.0)
        
        # 2. 순위 기반 매핑
        # 예측값의 순위를 GNN 값의 순위와 매핑
        pred_ranks = stats.rankdata(predictions)
        gnn_ranks = stats.rankdata(self.gnn_values)
        
        # 순위를 백분위로 변환
        pred_percentiles = pred_ranks / len(pred_ranks)
        
        # GNN 값을 정렬해서 백분위에 따라 매핑
        gnn_sorted = np.sort(self.gnn_values)
        
        aligned_predictions = []
        for p in pred_percentiles:
            # 백분위에 해당하는 GNN 값 찾기
            idx = int(p * (len(gnn_sorted) - 1))
            aligned_predictions.append(gnn_sorted[idx])
        
        aligned_predictions = np.array(aligned_predictions)
        
        # 3. 원본 예측과 블렌딩 (70% 정렬, 30% 원본)
        blend_ratio = 0.7
        final_predictions = (aligned_predictions * blend_ratio + 
                           predictions * (1 - blend_ratio))
        
        # 4. 통계적 조정
        current_mean = np.mean(final_predictions)
        current_std = np.std(final_predictions)
        
        if current_std > 0:
            normalized = (final_predictions - current_mean) / current_std
            adjusted = normalized * self.gnn_std + self.gnn_mean
        else:
            adjusted = final_predictions
        
        # 5. 최종 범위 조정
        adjusted = np.clip(adjusted, np.min(self.gnn_values), np.max(self.gnn_values))
        
        print(f"후처리 완료: 평균={np.mean(adjusted):.2f}, 표준편차={np.std(adjusted):.2f}")
        return adjusted

def load_training_data():
    """훈련 데이터 로드"""
    print("훈련 데이터 로드 중...")
    
    # ChEMBL 데이터
    try:
        chembl = pd.read_csv("Data/ChEMBL_ASK1(IC50).csv", sep=';')
        chembl.columns = chembl.columns.str.strip().str.replace('"', '')
        chembl = chembl[chembl['Standard Type'] == 'IC50']
        chembl = chembl[['Smiles', 'Standard Value']].rename(
            columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}
        ).dropna()
        chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
        chembl = chembl.dropna()
        print(f"  ChEMBL: {len(chembl)} 행")
    except Exception as e:
        print(f"  ChEMBL 오류: {e}")
        chembl = pd.DataFrame()
    
    # PubChem 데이터
    try:
        pubchem = pd.read_csv("Data/Pubchem_ASK1.csv")
        pubchem = pubchem[['SMILES', 'Activity_Value']].rename(
            columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}
        ).dropna()
        pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
        pubchem = pubchem.dropna()
        pubchem['ic50_nM'] = pubchem['ic50_nM'] * 1000  # μM -> nM
        print(f"  PubChem: {len(pubchem)} 행")
    except Exception as e:
        print(f"  PubChem 오류: {e}")
        pubchem = pd.DataFrame()
    
    # 데이터 결합
    if len(chembl) > 0 and len(pubchem) > 0:
        combined = pd.concat([chembl, pubchem], ignore_index=True)
    elif len(chembl) > 0:
        combined = chembl
    elif len(pubchem) > 0:
        combined = pubchem
    else:
        raise ValueError("훈련 데이터가 없습니다.")
    
    # 전처리
    combined = combined.drop_duplicates(subset='smiles')
    combined = combined[combined['ic50_nM'] > 0]
    
    # 생물학적 범위 필터링
    combined = combined[(combined['ic50_nM'] >= 0.1) & (combined['ic50_nM'] <= 100000)]
    
    # pIC50 변환
    combined['pIC50'] = IC50_to_pIC50(combined['ic50_nM'])
    
    # 유효한 SMILES 확인
    combined['mol'] = combined['smiles'].apply(Chem.MolFromSmiles)
    combined = combined.dropna(subset=['mol']).reset_index(drop=True)
    
    print(f"  전처리 완료: {len(combined)} 행")
    return combined

def main():
    """메인 함수"""
    print("ASK1 IC50 예측 파이프라인 - GNN 직접 매칭 버전")
    print("=" * 60)
    
    # 데이터 로드
    train_data = load_training_data()
    test_data = pd.read_csv("Data/test.csv")
    gnn_data = pd.read_csv("gnn_pytorch.csv")
    
    print(f"훈련 데이터: {len(train_data)} 행")
    print(f"테스트 데이터: {len(test_data)} 행")
    print(f"GNN 참조 데이터: {len(gnn_data)} 행")
    
    # 특성 추출
    print("\\n특성 추출 중...")
    featurizer = OptimizedMolecularFeaturizer()
    
    X_train = featurizer.fit_transform(train_data['smiles'].tolist(), train_data['pIC50'].values)
    y_train = train_data['pIC50'].values
    
    X_test = featurizer.transform(test_data['Smiles'].tolist())
    
    # 모델 훈련
    print("\\n모델 훈련 중...")
    ensemble = PowerfulEnsemble()
    ensemble.fit(X_train, y_train)
    
    # 예측
    print("\\n예측 수행 중...")
    pIC50_pred = ensemble.predict(X_test)
    ic50_pred = pIC50_to_IC50(pIC50_pred)
    
    # 후처리
    print("\\n후처리 중...")
    postprocessor = GNNAlignedPostprocessor(gnn_data)
    final_predictions = postprocessor.process_predictions(ic50_pred)
    
    # 결과 저장
    print("\\n결과 저장 중...")
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'ASK1_IC50_nM': final_predictions
    })
    
    submission.to_csv('submission_gnn_aligned.csv', index=False)
    print("결과 저장 완료: submission_gnn_aligned.csv")
    
    # 분석
    print("\\n결과 분석:")
    print(f"  평균: {np.mean(final_predictions):.2f} nM")
    print(f"  표준편차: {np.std(final_predictions):.2f} nM")
    print(f"  범위: {np.min(final_predictions):.2f} - {np.max(final_predictions):.2f} nM")
    
    # GNN과 상관계수
    correlation = np.corrcoef(final_predictions, gnn_data['ASK1_IC50_nM'].values)[0, 1]
    print(f"  GNN 상관계수: {correlation:.3f}")
    
    # 분포 분석
    very_potent = np.mean(final_predictions < 1)
    potent = np.mean((final_predictions >= 1) & (final_predictions < 10))
    moderate = np.mean((final_predictions >= 10) & (final_predictions < 100))
    weak = np.mean(final_predictions >= 100)
    
    print(f"\\n  활성 분포:")
    print(f"    < 1 nM: {very_potent:.1%}")
    print(f"    1-10 nM: {potent:.1%}")
    print(f"    10-100 nM: {moderate:.1%}")
    print(f"    > 100 nM: {weak:.1%}")
    
    # 모델 저장
    print("\\n모델 저장 중...")
    os.makedirs("Models", exist_ok=True)
    joblib.dump(featurizer, "Models/featurizer_gnn_aligned.pkl")
    joblib.dump(ensemble, "Models/ensemble_gnn_aligned.pkl")
    print("모델 저장 완료")
    
    print("\\n" + "=" * 60)
    print("GNN 직접 매칭 파이프라인 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
