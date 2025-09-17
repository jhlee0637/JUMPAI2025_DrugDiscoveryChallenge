import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 데이터 로드
df = pd.read_csv('filtered_bioactivity_data.csv')

# ECFP 생성 함수 (Morgan fingerprints)
def smiles_to_ecfp(smiles, radius=2, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            return np.array(fp)
        else:
            return None
    except:
        return None

# 진행 상황 표시 함수
def process_with_progress(df):
    total = len(df)
    fingerprints = []
    
    print(f"총 {total}개의 SMILES 처리 중...")
    
    for i, smiles in enumerate(df['canonical_smiles']):
        if (i+1) % 100 == 0 or i+1 == total:
            print(f"{i+1}/{total} 완료")
        
        fp = smiles_to_ecfp(smiles)
        if fp is not None:
            fingerprints.append(fp)
        else:
            fingerprints.append(np.zeros(1024))  # 실패한 경우 0으로 채운 벡터 추가
    
    return fingerprints

# ECFP fingerprints 생성
fingerprints = process_with_progress(df)

# 결과를 DataFrame으로 변환
fp_df = pd.DataFrame(fingerprints)

# 컬럼 이름 지정
fp_columns = [f'ECFP_{i}' for i in range(fp_df.shape[1])]
fp_df.columns = fp_columns

# 원래 데이터와 fingerprints 결합
result_df = pd.concat([df, fp_df], axis=1)

# 결과 저장
result_df.to_csv('bioactivity_data_with_ecfp.csv', index=False)
print("ECFP fingerprints 생성 완료. 'bioactivity_data_with_ecfp.csv'에 저장되었습니다.")
