#!/usr/bin/env python3
"""
JUMP AI 경진대회 - 고도화된 ASK1 IC50 예측 파이프라인
GNN 출력 직접 매칭 및 분자 유사성 극대화

버전 2.0: 향상된 GNN 패턴 매칭
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedJumpAIPredictor:
    def __init__(self):
        self.gnn_reference = None
        self.test_molecules = None
        self.similarity_matrix = None
        self.feature_matrix = None
        
    def load_data(self):
        """데이터 로드 및 초기화"""
        print("데이터 로드 중...")
        
        # 데이터 로드
        self.gnn_reference = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/gnn_pytorch.csv')
        self.test_molecules = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/Data/test.csv')
        
        # ID 정렬
        self.gnn_reference = self.gnn_reference.sort_values('ID')
        self.test_molecules = self.test_molecules.sort_values('ID')
        
        print(f"GNN 참조 데이터: {len(self.gnn_reference)} 개 샘플")
        print(f"테스트 데이터: {len(self.test_molecules)} 개 분자")
        
        # GNN 통계 계산
        self.gnn_stats = self.calculate_gnn_stats()
        
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
            return np.zeros(50)  # 기본값
            
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
        atom_counts = [0] * 10  # C, N, O, F, Cl, Br, I, S, P 등
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
        
        # 추가 특성 (패딩)
        while len(features) < 50:
            features.append(0.0)
            
        return np.array(features[:50])
        
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
        
        for sim_idx in range(5):  # 5가지 유사성 지표
            sim_matrix = np.zeros((n_molecules, n_molecules))
            
            for i, smiles_i in enumerate(self.test_molecules['Smiles']):
                mol_i = Chem.MolFromSmiles(smiles_i)
                
                for j, smiles_j in enumerate(self.test_molecules['Smiles']):
                    if i <= j:
                        mol_j = Chem.MolFromSmiles(smiles_j)
                        similarities = self.calculate_multiple_similarities(mol_i, mol_j)
                        sim_matrix[i, j] = similarities[sim_idx]
                        sim_matrix[j, i] = similarities[sim_idx]
                        
                if (i + 1) % 20 == 0:
                    print(f"유사성 지표 {sim_idx + 1}/5, 진행률: {i+1}/{n_molecules}")
                    
            similarity_matrices.append(sim_matrix)
            
        # 유사성 행렬들을 결합
        self.similarity_matrix = np.mean(similarity_matrices, axis=0)
        print("유사성 행렬 구축 완료")
        
    def advanced_neighbor_prediction(self, target_idx, k=10):
        """고도화된 이웃 기반 예측"""
        # 해당 분자의 유사성 점수
        similarities = self.similarity_matrix[target_idx]
        
        # 자기 자신 제외
        similarities[target_idx] = 0
        
        # 가장 유사한 k개 분자 선택
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_similarities = similarities[top_k_indices]
        
        # 유사성이 너무 낮으면 전체 평균 사용
        if np.max(top_k_similarities) < 0.1:
            return self.gnn_stats['median']
            
        # 가중 평균 계산
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        top_k_values = gnn_values[top_k_indices]
        
        # 유사성 가중치 정규화
        weights = top_k_similarities / np.sum(top_k_similarities)
        
        # 가중 평균 예측
        prediction = np.sum(weights * top_k_values)
        
        return prediction
        
    def gaussian_process_smoothing(self, initial_predictions):
        """가우시안 프로세스 스무딩"""
        print("가우시안 프로세스 스무딩 적용 중...")
        
        smoothed_predictions = initial_predictions.copy()
        
        for i in range(len(initial_predictions)):
            # 주변 분자들의 유사성 가중 평균
            similarities = self.similarity_matrix[i]
            
            # 유사성 임계값 이상인 분자들만 고려
            threshold = 0.3
            similar_indices = np.where(similarities > threshold)[0]
            
            if len(similar_indices) > 1:
                similar_values = initial_predictions[similar_indices]
                similar_weights = similarities[similar_indices]
                
                # 가중 평균
                weighted_mean = np.sum(similar_weights * similar_values) / np.sum(similar_weights)
                
                # 기존 예측값과 가중 평균의 혼합
                smoothed_predictions[i] = 0.7 * initial_predictions[i] + 0.3 * weighted_mean
                
        return smoothed_predictions
        
    def distribution_alignment(self, predictions):
        """분포 정렬 고도화"""
        print("분포 정렬 수행 중...")
        
        # 1단계: 분위수 매칭
        pred_sorted = np.sort(predictions)
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        gnn_sorted = np.sort(gnn_values)
        
        # 분위수 매핑
        aligned_predictions = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            # 현재 예측값의 분위수 찾기
            percentile = (np.searchsorted(pred_sorted, pred) / len(pred_sorted)) * 100
            percentile = np.clip(percentile, 0, 100)
            
            # 해당 분위수의 GNN 값으로 매핑
            gnn_value = np.percentile(gnn_sorted, percentile)
            aligned_predictions[i] = gnn_value
            
        # 2단계: 미세 조정
        # 평균과 표준편차 매칭
        current_mean = np.mean(aligned_predictions)
        current_std = np.std(aligned_predictions)
        
        target_mean = self.gnn_stats['mean']
        target_std = self.gnn_stats['std']
        
        # 표준화 후 타겟 분포로 변환
        if current_std > 0:
            standardized = (aligned_predictions - current_mean) / current_std
            final_predictions = standardized * target_std + target_mean
        else:
            final_predictions = aligned_predictions
            
        # 범위 제한
        final_predictions = np.clip(final_predictions, 
                                   self.gnn_stats['min'], 
                                   self.gnn_stats['max'])
        
        return final_predictions
        
    def ensemble_prediction(self):
        """앙상블 예측 수행"""
        print("앙상블 예측 수행 중...")
        
        n_molecules = len(self.test_molecules)
        predictions = np.zeros(n_molecules)
        
        # 각 분자에 대해 예측
        for i in range(n_molecules):
            # 이웃 기반 예측 (여러 k 값 사용)
            pred_k5 = self.advanced_neighbor_prediction(i, k=5)
            pred_k10 = self.advanced_neighbor_prediction(i, k=10)
            pred_k15 = self.advanced_neighbor_prediction(i, k=15)
            
            # 앙상블 예측
            predictions[i] = np.mean([pred_k5, pred_k10, pred_k15])
            
            if (i + 1) % 20 == 0:
                print(f"예측 진행률: {i+1}/{n_molecules}")
                
        return predictions
        
    def predict(self):
        """최종 예측 수행"""
        print("\n=== 최종 예측 수행 ===")
        
        # 유사성 행렬 구축
        self.build_similarity_matrix()
        
        # 앙상블 예측
        initial_predictions = self.ensemble_prediction()
        
        # 가우시안 프로세스 스무딩
        smoothed_predictions = self.gaussian_process_smoothing(initial_predictions)
        
        # 분포 정렬
        final_predictions = self.distribution_alignment(smoothed_predictions)
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame({
            'ID': self.test_molecules['ID'],
            'ASK1_IC50_nM': final_predictions
        })
        
        return result_df
        
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("JUMP AI 경진대회 - 고도화된 ASK1 IC50 예측 파이프라인")
        print("=" * 60)
        
        # 데이터 로드
        self.load_data()
        
        # 예측 수행
        result_df = self.predict()
        
        # 결과 저장
        output_file = '/Users/skku_aws28/Documents/Jump_Team_Project/submission_advanced.csv'
        result_df.to_csv(output_file, index=False)
        
        print(f"\n예측 완료! 결과 저장: {output_file}")
        
        # 성능 평가
        self.evaluate_performance(result_df)
        
        return result_df
        
    def evaluate_performance(self, result_df):
        """성능 평가"""
        print("\n=== 성능 평가 ===")
        
        pred_values = result_df['ASK1_IC50_nM'].values
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        
        # 기본 통계
        print(f"예측 평균: {np.mean(pred_values):.3f}")
        print(f"GNN 평균: {np.mean(gnn_values):.3f}")
        print(f"예측 표준편차: {np.std(pred_values):.3f}")
        print(f"GNN 표준편차: {np.std(gnn_values):.3f}")
        
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
        
        # 활성 분포
        print("\n=== 활성 분포 ===")
        for name, values in [('예측', pred_values), ('GNN', gnn_values)]:
            highly_active = np.sum(values < 1.0)
            active = np.sum((values >= 1.0) & (values < 10.0))
            moderate = np.sum((values >= 10.0) & (values < 100.0))
            weak = np.sum(values >= 100.0)
            print(f"{name}: 고활성={highly_active}, 활성={active}, 중간={moderate}, 저활성={weak}")

def main():
    """메인 실행 함수"""
    predictor = AdvancedJumpAIPredictor()
    result_df = predictor.run_pipeline()
    
    print("\n고도화된 파이프라인 실행 완료!")
    print("submission_advanced.csv 파일을 제출하세요.")

if __name__ == "__main__":
    main()
