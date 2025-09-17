#!/usr/bin/env python3
"""
JUMP AI 경진대회 - 하이브리드 ASK1 IC50 예측 파이프라인
실제 훈련 데이터 + GNN 패턴 매칭 통합 버전

버전 3.0: 실제 데이터 기반 학습 + GNN 분포 매칭
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class HybridJumpAIPredictor:
    def __init__(self):
        self.gnn_reference = None
        self.test_molecules = None
        self.training_data = None
        self.similarity_matrix = None
        self.ml_models = {}
        self.scalers = {}
        
    def load_data(self):
        """모든 데이터 로드 및 초기화"""
        print("데이터 로드 중...")
        
        # 기존 데이터 로드
        self.gnn_reference = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/gnn_pytorch.csv')
        self.test_molecules = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/Data/test.csv')
        
        # 실제 훈련 데이터 로드
        self.load_training_data()
        
        # ID 정렬
        self.gnn_reference = self.gnn_reference.sort_values('ID')
        self.test_molecules = self.test_molecules.sort_values('ID')
        
        print(f"GNN 참조 데이터: {len(self.gnn_reference)} 개 샘플")
        print(f"테스트 데이터: {len(self.test_molecules)} 개 분자")
        print(f"훈련 데이터: {len(self.training_data)} 개 화합물")
        
        # GNN 통계 계산
        self.gnn_stats = self.calculate_gnn_stats()
        
    def load_training_data(self):
        """실제 훈련 데이터 로드 및 전처리"""
        print("실제 훈련 데이터 로드 중...")
        
        # ChEMBL 데이터 로드
        try:
            chembl_df = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/Data/ChEMBL_ASK1(IC50).csv', 
                                  sep=';', quotechar='"')
            
            # 필요한 컬럼만 선택
            chembl_clean = chembl_df[['Smiles', 'Standard Value', 'Standard Units']].copy()
            chembl_clean = chembl_clean.dropna()
            
            # nM 단위로 변환
            chembl_clean['IC50_nM'] = chembl_clean['Standard Value']
            chembl_clean = chembl_clean[chembl_clean['Standard Units'] == 'nM']
            
            # 유효한 SMILES만 유지
            valid_smiles = []
            valid_ic50 = []
            
            for idx, row in chembl_clean.iterrows():
                mol = Chem.MolFromSmiles(row['Smiles'])
                if mol is not None and row['IC50_nM'] > 0:
                    valid_smiles.append(row['Smiles'])
                    valid_ic50.append(row['IC50_nM'])
                    
            chembl_data = pd.DataFrame({
                'SMILES': valid_smiles,
                'IC50_nM': valid_ic50,
                'Source': 'ChEMBL'
            })
            
            print(f"ChEMBL 데이터: {len(chembl_data)} 개 화합물")
            
        except Exception as e:
            print(f"ChEMBL 데이터 로드 실패: {e}")
            chembl_data = pd.DataFrame()
        
        # PubChem 데이터 로드 (샘플링)
        try:
            pubchem_df = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/Data/Pubchem_ASK1.csv')
            
            # IC50 데이터만 선택
            pubchem_ic50 = pubchem_df[pubchem_df['Activity_Type'] == 'IC50'].copy()
            pubchem_ic50 = pubchem_ic50.dropna(subset=['Activity_Value', 'SMILES'])
            
            # 너무 많으므로 샘플링 (상위 1000개)
            pubchem_ic50 = pubchem_ic50.head(1000)
            
            # 단위 변환 (대부분 μM이므로 nM으로 변환)
            pubchem_ic50['IC50_nM'] = pubchem_ic50['Activity_Value'] * 1000  # μM to nM
            
            # 유효한 데이터만 유지
            valid_pubchem = []
            for idx, row in pubchem_ic50.iterrows():
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is not None and row['IC50_nM'] > 0:
                    valid_pubchem.append({
                        'SMILES': row['SMILES'],
                        'IC50_nM': row['IC50_nM'],
                        'Source': 'PubChem'
                    })
                    
            pubchem_data = pd.DataFrame(valid_pubchem)
            print(f"PubChem 데이터: {len(pubchem_data)} 개 화합물")
            
        except Exception as e:
            print(f"PubChem 데이터 로드 실패: {e}")
            pubchem_data = pd.DataFrame()
        
        # 데이터 통합
        training_datasets = [df for df in [chembl_data, pubchem_data] if not df.empty]
        
        if training_datasets:
            self.training_data = pd.concat(training_datasets, ignore_index=True)
            
            # 중복 제거
            self.training_data = self.training_data.drop_duplicates(subset=['SMILES'])
            
            # 이상치 제거 (IC50 값이 너무 극단적인 경우)
            q1 = self.training_data['IC50_nM'].quantile(0.01)
            q99 = self.training_data['IC50_nM'].quantile(0.99)
            self.training_data = self.training_data[
                (self.training_data['IC50_nM'] >= q1) & 
                (self.training_data['IC50_nM'] <= q99)
            ]
            
            print(f"통합 훈련 데이터: {len(self.training_data)} 개 화합물")
            
        else:
            print("훈련 데이터 로드 실패 - GNN 패턴 매칭만 사용")
            self.training_data = pd.DataFrame()
            
    def calculate_gnn_stats(self):
        """GNN 출력 통계 계산"""
        values = self.gnn_reference['ASK1_IC50_nM'].values
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'q10': np.percentile(values, 10),
            'q90': np.percentile(values, 90)
        }
        
    def calculate_comprehensive_features(self, mol):
        """포괄적인 분자 특성 계산"""
        if mol is None:
            return np.zeros(60)  # 기본값
            
        features = []
        
        # 기본 물리화학적 특성
        features.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumSaturatedRings(mol)
        ])
        
        # 원자 수 특성
        atom_counts = [0] * 10
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'C':
                atom_counts[0] += 1
            elif symbol == 'N':
                atom_counts[1] += 1
            elif symbol == 'O':
                atom_counts[2] += 1
            elif symbol == 'F':
                atom_counts[3] += 1
            elif symbol == 'Cl':
                atom_counts[4] += 1
            elif symbol == 'Br':
                atom_counts[5] += 1
            elif symbol == 'I':
                atom_counts[6] += 1
            elif symbol == 'S':
                atom_counts[7] += 1
            elif symbol == 'P':
                atom_counts[8] += 1
            else:
                atom_counts[9] += 1
        
        features.extend(atom_counts)
        
        # 구조적 특성
        features.extend([
            Descriptors.BertzCT(mol),
            Descriptors.MolMR(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.Ipc(mol)
        ])
        
        # 결합 특성
        features.extend([
            mol.GetNumBonds(),
            len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]),
            len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]),
            len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]),
            len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]),
            len([b for b in mol.GetBonds() if b.IsInRing()])
        ])
        
        # 전자적 특성
        features.extend([
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            Descriptors.MaxAbsEStateIndex(mol),
            Descriptors.MinAbsEStateIndex(mol),
            Descriptors.qed(mol),
            Descriptors.SlogP_VSA1(mol),
            Descriptors.SlogP_VSA2(mol),
            Descriptors.SMR_VSA1(mol),
            Descriptors.SMR_VSA2(mol),
            Descriptors.PEOE_VSA1(mol)
        ])
        
        # 추가 특성 (패딩)
        while len(features) < 60:
            features.append(0.0)
            
        return np.array(features[:60])
        
    def build_ml_models(self):
        """실제 데이터 기반 머신러닝 모델 구축"""
        if self.training_data.empty:
            print("훈련 데이터가 없어 ML 모델 구축 생략")
            return
            
        print("머신러닝 모델 구축 중...")
        
        # 특성 추출
        X_train = []
        y_train = []
        
        for idx, row in self.training_data.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is not None:
                features = self.calculate_comprehensive_features(mol)
                X_train.append(features)
                y_train.append(np.log10(row['IC50_nM']))  # log 변환
                
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"훈련 특성 행렬 크기: {X_train.shape}")
        
        # 스케일러 훈련
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        
        # 여러 모델 훈련
        models = {
            'rf': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'gbm': GradientBoostingRegressor(n_estimators=200, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
            'xgb': xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        }
        
        # 교차 검증 및 모델 훈련
        cv_scores = {}
        for name, model in models.items():
            try:
                # 교차 검증
                cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
                cv_scores[name] = -cv_score.mean()
                
                # 전체 데이터로 훈련
                model.fit(X_train_scaled, y_train)
                self.ml_models[name] = model
                
                print(f"{name.upper()} 모델 CV MSE: {cv_scores[name]:.4f}")
                
            except Exception as e:
                print(f"{name} 모델 훈련 실패: {e}")
        
        print("머신러닝 모델 구축 완료")
        
    def calculate_multiple_similarities(self, mol1, mol2):
        """다중 유사성 지표 계산"""
        if mol1 is None or mol2 is None:
            return np.zeros(5)
            
        similarities = []
        
        # 1. Morgan 지문 (반지름 2)
        try:
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            similarities.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        except:
            similarities.append(0.0)
            
        # 2. Morgan 지문 (반지름 3)
        try:
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
            similarities.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        except:
            similarities.append(0.0)
            
        # 3. MACCS 키
        try:
            maccs1 = rdMolDescriptors.GetMACCSKeysFingerprint(mol1)
            maccs2 = rdMolDescriptors.GetMACCSKeysFingerprint(mol2)
            similarities.append(DataStructs.TanimotoSimilarity(maccs1, maccs2))
        except:
            similarities.append(0.0)
            
        # 4. 토폴로지 지문
        try:
            fp1 = FingerprintMols.FingerprintMol(mol1)
            fp2 = FingerprintMols.FingerprintMol(mol2)
            similarities.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        except:
            similarities.append(0.0)
            
        # 5. 특성 기반 유사성
        try:
            feat1 = self.calculate_comprehensive_features(mol1)
            feat2 = self.calculate_comprehensive_features(mol2)
            feat_sim = cosine_similarity([feat1], [feat2])[0, 0]
            similarities.append(feat_sim)
        except:
            similarities.append(0.0)
            
        return np.array(similarities)
        
    def build_similarity_matrix(self):
        """유사성 행렬 구축"""
        print("유사성 행렬 구축 중...")
        
        n_molecules = len(self.test_molecules)
        similarity_matrices = []
        
        for sim_idx in range(5):
            sim_matrix = np.zeros((n_molecules, n_molecules))
            
            for i, smiles_i in enumerate(self.test_molecules['Smiles']):
                mol_i = Chem.MolFromSmiles(smiles_i)
                
                for j, smiles_j in enumerate(self.test_molecules['Smiles']):
                    if i <= j:
                        mol_j = Chem.MolFromSmiles(smiles_j)
                        similarities = self.calculate_multiple_similarities(mol_i, mol_j)
                        sim_matrix[i, j] = similarities[sim_idx]
                        sim_matrix[j, i] = similarities[sim_idx]
                        
                if (i + 1) % 30 == 0:
                    print(f"유사성 지표 {sim_idx + 1}/5, 진행률: {i+1}/{n_molecules}")
                    
            similarity_matrices.append(sim_matrix)
            
        self.similarity_matrix = np.mean(similarity_matrices, axis=0)
        print("유사성 행렬 구축 완료")
        
    def ml_based_prediction(self, test_smiles):
        """머신러닝 기반 예측"""
        if not self.ml_models:
            print("ML 모델이 없어 ML 예측 생략")
            return np.full(len(test_smiles), self.gnn_stats['median'])
            
        print("머신러닝 기반 예측 수행 중...")
        
        # 테스트 특성 추출
        X_test = []
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            features = self.calculate_comprehensive_features(mol)
            X_test.append(features)
            
        X_test = np.array(X_test)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # 앙상블 예측
        predictions = np.zeros(len(test_smiles))
        
        for name, model in self.ml_models.items():
            try:
                pred = model.predict(X_test_scaled)
                # log 변환 되돌리기
                pred = 10 ** pred
                predictions += pred
            except Exception as e:
                print(f"{name} 모델 예측 실패: {e}")
                
        predictions /= len(self.ml_models)
        
        return predictions
        
    def gnn_similarity_prediction(self, test_smiles):
        """GNN 유사성 기반 예측"""
        print("GNN 유사성 기반 예측 수행 중...")
        
        predictions = np.zeros(len(test_smiles))
        
        for i in range(len(test_smiles)):
            similarities = self.similarity_matrix[i]
            similarities[i] = 0  # 자기 자신 제외
            
            # 상위 k개 유사한 분자 선택
            k = 15
            top_k_indices = np.argsort(similarities)[-k:]
            top_k_similarities = similarities[top_k_indices]
            
            if np.max(top_k_similarities) > 0.1:
                gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
                top_k_values = gnn_values[top_k_indices]
                
                weights = top_k_similarities / np.sum(top_k_similarities)
                predictions[i] = np.sum(weights * top_k_values)
            else:
                predictions[i] = self.gnn_stats['median']
                
        return predictions
        
    def hybrid_prediction(self, test_smiles):
        """하이브리드 예측 (ML + GNN 유사성)"""
        print("하이브리드 예측 수행 중...")
        
        # 각 방법별 예측
        ml_predictions = self.ml_based_prediction(test_smiles)
        gnn_predictions = self.gnn_similarity_prediction(test_smiles)
        
        # 가중 평균 (ML 모델이 있으면 더 높은 가중치)
        if self.ml_models:
            hybrid_predictions = 0.6 * ml_predictions + 0.4 * gnn_predictions
        else:
            hybrid_predictions = gnn_predictions
            
        return hybrid_predictions, ml_predictions, gnn_predictions
        
    def distribution_alignment(self, predictions):
        """분포 정렬"""
        print("분포 정렬 수행 중...")
        
        # 분위수 매칭
        pred_sorted = np.sort(predictions)
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        gnn_sorted = np.sort(gnn_values)
        
        aligned_predictions = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            percentile = (np.searchsorted(pred_sorted, pred) / len(pred_sorted)) * 100
            percentile = np.clip(percentile, 0, 100)
            gnn_value = np.percentile(gnn_sorted, percentile)
            aligned_predictions[i] = gnn_value
            
        # 통계 매칭
        current_mean = np.mean(aligned_predictions)
        current_std = np.std(aligned_predictions)
        
        if current_std > 0:
            standardized = (aligned_predictions - current_mean) / current_std
            final_predictions = standardized * self.gnn_stats['std'] + self.gnn_stats['mean']
        else:
            final_predictions = aligned_predictions
            
        final_predictions = np.clip(final_predictions, 
                                   self.gnn_stats['min'], 
                                   self.gnn_stats['max'])
        
        return final_predictions
        
    def predict(self):
        """최종 예측 수행"""
        print("\n=== 하이브리드 예측 수행 ===")
        
        # 머신러닝 모델 구축
        self.build_ml_models()
        
        # 유사성 행렬 구축
        self.build_similarity_matrix()
        
        # 하이브리드 예측
        test_smiles = self.test_molecules['Smiles'].values
        hybrid_preds, ml_preds, gnn_preds = self.hybrid_prediction(test_smiles)
        
        # 분포 정렬
        final_predictions = self.distribution_alignment(hybrid_preds)
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame({
            'ID': self.test_molecules['ID'],
            'ASK1_IC50_nM': final_predictions,
            'ML_Prediction': ml_preds,
            'GNN_Prediction': gnn_preds,
            'Hybrid_Raw': hybrid_preds
        })
        
        return result_df
        
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("=" * 70)
        print("JUMP AI 경진대회 - 하이브리드 ASK1 IC50 예측 파이프라인")
        print("=" * 70)
        
        # 데이터 로드
        self.load_data()
        
        # 예측 수행
        result_df = self.predict()
        
        # 결과 저장
        output_file = '/Users/skku_aws28/Documents/Jump_Team_Project/submission_hybrid.csv'
        final_df = result_df[['ID', 'ASK1_IC50_nM']].copy()
        final_df.to_csv(output_file, index=False)
        
        # 상세 결과 저장
        detail_file = '/Users/skku_aws28/Documents/Jump_Team_Project/submission_hybrid_detail.csv'
        result_df.to_csv(detail_file, index=False)
        
        print(f"\n예측 완료!")
        print(f"제출 파일: {output_file}")
        print(f"상세 결과: {detail_file}")
        
        # 성능 평가
        self.evaluate_performance(result_df)
        
        return result_df
        
    def evaluate_performance(self, result_df):
        """성능 평가"""
        print("\n=== 성능 평가 ===")
        
        pred_values = result_df['ASK1_IC50_nM'].values
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        
        # 기본 통계
        print(f"최종 예측 평균: {np.mean(pred_values):.3f}")
        print(f"GNN 참조 평균: {np.mean(gnn_values):.3f}")
        print(f"최종 예측 표준편차: {np.std(pred_values):.3f}")
        print(f"GNN 참조 표준편차: {np.std(gnn_values):.3f}")
        
        # 상관관계
        pearson_corr = np.corrcoef(pred_values, gnn_values)[0, 1]
        spearman_corr = stats.spearmanr(pred_values, gnn_values)[0]
        
        print(f"피어슨 상관관계: {pearson_corr:.3f}")
        print(f"스피어만 상관관계: {spearman_corr:.3f}")
        
        # 오차 지표
        mse = np.mean((pred_values - gnn_values)**2)
        mae = np.mean(np.abs(pred_values - gnn_values))
        print(f"MSE: {mse:.3f}")
        print(f"MAE: {mae:.3f}")
        
        # 개별 예측 성능 비교
        if 'ML_Prediction' in result_df.columns:
            ml_corr = np.corrcoef(result_df['ML_Prediction'].values, gnn_values)[0, 1]
            gnn_corr = np.corrcoef(result_df['GNN_Prediction'].values, gnn_values)[0, 1]
            print(f"\n개별 성능:")
            print(f"ML 예측 상관관계: {ml_corr:.3f}")
            print(f"GNN 유사성 상관관계: {gnn_corr:.3f}")
            print(f"하이브리드 개선도: {pearson_corr - max(ml_corr, gnn_corr):.3f}")
        
        # 활성 분포
        print("\n=== 활성 분포 ===")
        for name, values in [('하이브리드', pred_values), ('GNN 참조', gnn_values)]:
            highly_active = np.sum(values < 1.0)
            active = np.sum((values >= 1.0) & (values < 10.0))
            moderate = np.sum((values >= 10.0) & (values < 100.0))
            weak = np.sum(values >= 100.0)
            print(f"{name}: 고활성={highly_active}, 활성={active}, 중간={moderate}, 저활성={weak}")

def main():
    """메인 실행 함수"""
    predictor = HybridJumpAIPredictor()
    result_df = predictor.run_pipeline()
    
    print("\n하이브리드 파이프라인 실행 완료!")
    print("submission_hybrid.csv 파일을 제출하세요.")

if __name__ == "__main__":
    main()
