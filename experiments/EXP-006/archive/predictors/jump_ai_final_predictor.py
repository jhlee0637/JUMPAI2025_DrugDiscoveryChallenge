#!/usr/bin/env python3
"""
JUMP AI 경진대회 - 최종 ASK1 IC50 예측 파이프라인
GNN 출력 분석 및 분자 유사성 기반 예측

규칙:
1. GNN 출력 분석을 통한 패턴 학습
2. 분자 유사성 기반 예측
3. 앙상블 휴리스틱 접근법
4. 생물학적 타당성 검증
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Fragments
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import DataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class JumpAIPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.gnn_reference = None
        self.test_molecules = None
        
    def load_data(self):
        """데이터 로드"""
        print("데이터 로드 중...")
        
        # GNN 참조 데이터 로드
        self.gnn_reference = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/gnn_pytorch.csv')
        print(f"GNN 참조 데이터: {len(self.gnn_reference)} 개 샘플")
        
        # 테스트 데이터 로드
        self.test_molecules = pd.read_csv('/Users/skku_aws28/Documents/Jump_Team_Project/Data/test.csv')
        print(f"테스트 데이터: {len(self.test_molecules)} 개 분자")
        
        # GNN 출력 통계 분석
        self.analyze_gnn_patterns()
        
    def analyze_gnn_patterns(self):
        """GNN 출력 패턴 분석"""
        print("\n=== GNN 출력 패턴 분석 ===")
        
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        
        print(f"평균: {np.mean(gnn_values):.3f}")
        print(f"표준편차: {np.std(gnn_values):.3f}")
        print(f"최솟값: {np.min(gnn_values):.3f}")
        print(f"최댓값: {np.max(gnn_values):.3f}")
        print(f"중앙값: {np.median(gnn_values):.3f}")
        
        # 활성 범위 분석
        highly_active = np.sum(gnn_values < 1.0)
        active = np.sum((gnn_values >= 1.0) & (gnn_values < 10.0))
        moderate = np.sum((gnn_values >= 10.0) & (gnn_values < 100.0))
        weak = np.sum(gnn_values >= 100.0)
        
        print(f"\n활성 분포:")
        print(f"고활성 (<1 nM): {highly_active} ({highly_active/len(gnn_values)*100:.1f}%)")
        print(f"활성 (1-10 nM): {active} ({active/len(gnn_values)*100:.1f}%)")
        print(f"중간활성 (10-100 nM): {moderate} ({moderate/len(gnn_values)*100:.1f}%)")
        print(f"저활성 (>100 nM): {weak} ({weak/len(gnn_values)*100:.1f}%)")
        
        # 분포 특성 저장
        self.gnn_stats = {
            'mean': np.mean(gnn_values),
            'std': np.std(gnn_values),
            'min': np.min(gnn_values),
            'max': np.max(gnn_values),
            'median': np.median(gnn_values),
            'q25': np.percentile(gnn_values, 25),
            'q75': np.percentile(gnn_values, 75)
        }
        
    def calculate_molecular_features(self, smiles):
        """분자 특성 계산"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        features = {}
        
        # 기본 분자 특성
        features['MW'] = Descriptors.MolWt(mol)
        features['LogP'] = Descriptors.MolLogP(mol)
        features['HBD'] = Descriptors.NumHDonors(mol)
        features['HBA'] = Descriptors.NumHAcceptors(mol)
        features['RotBonds'] = Descriptors.NumRotatableBonds(mol)
        features['TPSA'] = Descriptors.TPSA(mol)
        features['NumAtoms'] = mol.GetNumAtoms()
        features['NumBonds'] = mol.GetNumBonds()
        
        # 링 특성
        features['NumRings'] = Descriptors.RingCount(mol)
        features['NumAromRings'] = Descriptors.NumAromaticRings(mol)
        features['NumAliphRings'] = Descriptors.NumAliphaticRings(mol)
        features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        
        # 헤테로사이클 수 계산 (수동)
        ri = mol.GetRingInfo()
        num_hetero_rings = 0
        for ring in ri.AtomRings():
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if any(atom.GetAtomicNum() != 6 for atom in atoms):
                num_hetero_rings += 1
        features['NumHeterocycles'] = num_hetero_rings
        
        # 복합 특성
        features['BertzCT'] = Descriptors.BertzCT(mol)
        features['MolMR'] = Descriptors.MolMR(mol)
        
        # FractionCsp3 안전하게 계산
        try:
            features['FractionCsp3'] = Descriptors.FractionCsp3(mol)
        except:
            features['FractionCsp3'] = 0.0
            
        features['Chi0v'] = Descriptors.Chi0v(mol)
        features['Chi1v'] = Descriptors.Chi1v(mol)
        
        # 원자 수
        features['NumC'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        features['NumN'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'])
        features['NumO'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])
        features['NumF'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        features['NumCl'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl'])
        features['NumBr'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br'])
        features['NumI'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'I'])
        features['NumS'] = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'S'])
        
        # 구조 특성 (안전하게 계산)
        try:
            features['NumSP'] = len([atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP])
            features['NumSP2'] = len([atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2])
            features['NumSP3'] = len([atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3])
        except:
            features['NumSP'] = 0
            features['NumSP2'] = 0
            features['NumSP3'] = 0
        
        return features
        
    def calculate_molecular_similarity(self, smiles1, smiles2):
        """분자 유사성 계산"""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
            
        # Morgan 지문 유사성
        fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
        
    def structure_based_prediction(self, test_smiles):
        """구조 기반 예측"""
        print("\n구조 기반 예측 수행 중...")
        
        predictions = []
        
        for i, smiles in enumerate(test_smiles):
            similarities = []
            gnn_values = []
            
            # GNN 참조 분자들과 유사성 계산
            for j, ref_row in self.gnn_reference.iterrows():
                ref_smiles = self.test_molecules.loc[self.test_molecules['ID'] == ref_row['ID'], 'Smiles'].values[0]
                similarity = self.calculate_molecular_similarity(smiles, ref_smiles)
                similarities.append(similarity)
                gnn_values.append(ref_row['ASK1_IC50_nM'])
            
            similarities = np.array(similarities)
            gnn_values = np.array(gnn_values)
            
            # 가중 평균 예측
            if np.max(similarities) > 0:
                weights = similarities / np.sum(similarities)
                prediction = np.sum(weights * gnn_values)
            else:
                prediction = self.gnn_stats['median']
                
            predictions.append(prediction)
            
            if (i + 1) % 20 == 0:
                print(f"진행률: {i+1}/{len(test_smiles)}")
                
        return np.array(predictions)
        
    def feature_based_prediction(self, test_smiles):
        """분자 특성 기반 예측"""
        print("\n분자 특성 기반 예측 수행 중...")
        
        # 테스트 분자 특성 계산
        test_features = []
        for smiles in test_smiles:
            features = self.calculate_molecular_features(smiles)
            if features:
                test_features.append(features)
            else:
                # 기본값 사용
                test_features.append({key: 0.0 for key in ['MW', 'LogP', 'HBD', 'HBA', 'RotBonds', 'TPSA', 'NumAtoms', 'NumBonds']})
        
        test_df = pd.DataFrame(test_features)
        
        # GNN 값을 기반으로 특성별 상관관계 추정
        predictions = []
        
        for _, row in test_df.iterrows():
            # 분자량 기반 예측
            mw_pred = self.predict_by_molecular_weight(row['MW'])
            
            # LogP 기반 예측
            logp_pred = self.predict_by_logp(row['LogP'])
            
            # 복합성 기반 예측
            complexity_pred = self.predict_by_complexity(row)
            
            # 앙상블 예측
            ensemble_pred = np.mean([mw_pred, logp_pred, complexity_pred])
            predictions.append(ensemble_pred)
            
        return np.array(predictions)
        
    def predict_by_molecular_weight(self, mw):
        """분자량 기반 예측"""
        # 일반적으로 분자량이 클수록 활성이 감소하는 경향
        if mw < 300:
            return self.gnn_stats['q25']  # 높은 활성
        elif mw < 400:
            return self.gnn_stats['median']  # 중간 활성
        elif mw < 500:
            return self.gnn_stats['q75']  # 낮은 활성
        else:
            return self.gnn_stats['max']  # 매우 낮은 활성
            
    def predict_by_logp(self, logp):
        """LogP 기반 예측"""
        # 적절한 지용성이 중요
        if 1.0 <= logp <= 4.0:
            return self.gnn_stats['q25']  # 좋은 활성
        elif 0.0 <= logp < 1.0 or 4.0 < logp <= 5.0:
            return self.gnn_stats['median']  # 중간 활성
        else:
            return self.gnn_stats['q75']  # 낮은 활성
            
    def predict_by_complexity(self, features):
        """복합성 기반 예측"""
        # 적절한 복합성이 중요
        complexity_score = 0
        
        # 수소결합 공여체/수용체
        if 1 <= features['HBD'] <= 3:
            complexity_score += 1
        if 2 <= features['HBA'] <= 6:
            complexity_score += 1
            
        # 회전 가능 결합
        if features['RotBonds'] <= 7:
            complexity_score += 1
            
        # 극성 표면적
        if 20 <= features['TPSA'] <= 130:
            complexity_score += 1
            
        # 링 구조
        if 1 <= features['NumRings'] <= 4:
            complexity_score += 1
            
        # 복합성 점수에 따른 예측
        if complexity_score >= 4:
            return self.gnn_stats['q25']  # 높은 활성
        elif complexity_score >= 3:
            return self.gnn_stats['median']  # 중간 활성
        elif complexity_score >= 2:
            return self.gnn_stats['q75']  # 낮은 활성
        else:
            return self.gnn_stats['max']  # 매우 낮은 활성
            
    def pharmacophore_based_prediction(self, test_smiles):
        """약물동력학적 특성 기반 예측"""
        print("\n약물동력학적 특성 기반 예측 수행 중...")
        
        predictions = []
        
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                predictions.append(self.gnn_stats['median'])
                continue
                
            # Lipinski's Rule of Five 평가
            lipinski_score = self.evaluate_lipinski(mol)
            
            # 약물 유사성 평가
            drug_likeness = self.evaluate_drug_likeness(mol)
            
            # 종합 점수 계산
            combined_score = (lipinski_score + drug_likeness) / 2
            
            # 점수에 따른 예측
            if combined_score >= 0.8:
                pred = self.gnn_stats['q25']
            elif combined_score >= 0.6:
                pred = self.gnn_stats['median']
            elif combined_score >= 0.4:
                pred = self.gnn_stats['q75']
            else:
                pred = self.gnn_stats['max']
                
            predictions.append(pred)
            
        return np.array(predictions)
        
    def evaluate_lipinski(self, mol):
        """Lipinski's Rule of Five 평가"""
        score = 0
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        if mw <= 500:
            score += 0.25
        if logp <= 5:
            score += 0.25
        if hbd <= 5:
            score += 0.25
        if hba <= 10:
            score += 0.25
            
        return score
        
    def evaluate_drug_likeness(self, mol):
        """약물 유사성 평가"""
        score = 0
        
        # 회전 가능 결합
        if Descriptors.NumRotatableBonds(mol) <= 7:
            score += 0.2
            
        # 극성 표면적
        if Descriptors.TPSA(mol) <= 140:
            score += 0.2
            
        # 링 구조
        if 1 <= Descriptors.RingCount(mol) <= 6:
            score += 0.2
            
        # 방향족 링
        if 1 <= Descriptors.NumAromaticRings(mol) <= 3:
            score += 0.2
            
        # 헤테로원자
        num_hetero = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6]])
        if 1 <= num_hetero <= 8:
            score += 0.2
            
        return score
        
    def ensemble_prediction(self, test_smiles):
        """앙상블 예측"""
        print("\n앙상블 예측 수행 중...")
        
        # 각 방법별 예측
        structure_preds = self.structure_based_prediction(test_smiles)
        feature_preds = self.feature_based_prediction(test_smiles)
        pharma_preds = self.pharmacophore_based_prediction(test_smiles)
        
        # 가중 평균 (구조 기반을 더 높은 가중치로)
        ensemble_preds = (
            0.5 * structure_preds + 
            0.3 * feature_preds + 
            0.2 * pharma_preds
        )
        
        return ensemble_preds
        
    def apply_biological_constraints(self, predictions):
        """생물학적 제약 조건 적용"""
        print("\n생물학적 제약 조건 적용 중...")
        
        # 최솟값/최댓값 제한
        predictions = np.clip(predictions, 0.1, 10000)
        
        # 극값 스무딩
        q99 = np.percentile(predictions, 99)
        q01 = np.percentile(predictions, 1)
        
        predictions = np.where(predictions > q99, q99, predictions)
        predictions = np.where(predictions < q01, q01, predictions)
        
        return predictions
        
    def match_gnn_distribution(self, predictions):
        """GNN 분포에 맞게 조정"""
        print("\n분포 매칭 수행 중...")
        
        # 현재 예측값의 분포를 GNN 분포에 맞게 조정
        current_mean = np.mean(predictions)
        current_std = np.std(predictions)
        
        target_mean = self.gnn_stats['mean']
        target_std = self.gnn_stats['std']
        
        # 표준화 후 타겟 분포로 변환
        standardized = (predictions - current_mean) / current_std
        adjusted = standardized * target_std + target_mean
        
        # 생물학적 범위 내로 제한
        adjusted = np.clip(adjusted, self.gnn_stats['min'], self.gnn_stats['max'])
        
        return adjusted
        
    def predict(self):
        """최종 예측 수행"""
        print("\n=== 최종 예측 수행 ===")
        
        test_smiles = self.test_molecules['Smiles'].values
        
        # 앙상블 예측
        predictions = self.ensemble_prediction(test_smiles)
        
        # 생물학적 제약 조건 적용
        predictions = self.apply_biological_constraints(predictions)
        
        # GNN 분포 매칭
        predictions = self.match_gnn_distribution(predictions)
        
        # 결과 저장
        result_df = pd.DataFrame({
            'ID': self.test_molecules['ID'],
            'ASK1_IC50_nM': predictions
        })
        
        return result_df
        
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("JUMP AI 경진대회 - 최종 ASK1 IC50 예측 파이프라인")
        print("=" * 60)
        
        # 데이터 로드
        self.load_data()
        
        # 예측 수행
        result_df = self.predict()
        
        # 결과 저장
        output_file = '/Users/skku_aws28/Documents/Jump_Team_Project/submission_final.csv'
        result_df.to_csv(output_file, index=False)
        
        print(f"\n예측 완료! 결과 저장: {output_file}")
        
        # 결과 통계
        print("\n=== 예측 결과 통계 ===")
        pred_values = result_df['ASK1_IC50_nM'].values
        print(f"평균: {np.mean(pred_values):.3f}")
        print(f"표준편차: {np.std(pred_values):.3f}")
        print(f"최솟값: {np.min(pred_values):.3f}")
        print(f"최댓값: {np.max(pred_values):.3f}")
        print(f"중앙값: {np.median(pred_values):.3f}")
        
        # GNN과 비교
        print("\n=== GNN 대비 통계 ===")
        gnn_values = self.gnn_reference['ASK1_IC50_nM'].values
        correlation = np.corrcoef(pred_values, gnn_values)[0, 1]
        print(f"상관관계: {correlation:.3f}")
        
        # 활성 분포 비교
        print("\n=== 활성 분포 비교 ===")
        pred_highly_active = np.sum(pred_values < 1.0)
        pred_active = np.sum((pred_values >= 1.0) & (pred_values < 10.0))
        pred_moderate = np.sum((pred_values >= 10.0) & (pred_values < 100.0))
        pred_weak = np.sum(pred_values >= 100.0)
        
        print(f"예측 - 고활성: {pred_highly_active}, 활성: {pred_active}, 중간: {pred_moderate}, 저활성: {pred_weak}")
        
        gnn_highly_active = np.sum(gnn_values < 1.0)
        gnn_active = np.sum((gnn_values >= 1.0) & (gnn_values < 10.0))
        gnn_moderate = np.sum((gnn_values >= 10.0) & (gnn_values < 100.0))
        gnn_weak = np.sum(gnn_values >= 100.0)
        
        print(f"GNN - 고활성: {gnn_highly_active}, 활성: {gnn_active}, 중간: {gnn_moderate}, 저활성: {gnn_weak}")
        
        return result_df

def main():
    """메인 실행 함수"""
    predictor = JumpAIPredictor()
    result_df = predictor.run_pipeline()
    
    print("\n파이프라인 실행 완료!")
    print("submission_final.csv 파일을 제출하세요.")

if __name__ == "__main__":
    main()
