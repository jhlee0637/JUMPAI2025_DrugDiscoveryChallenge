# 환경: conda install -c conda-forge rdkit lightgbm scikit-learn optuna

import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import joblib, os

DATA_DIR = "/Users/junu/Documents/Project/Jump_Team_Project/Data"
SEED     = 42
NFOLD    = 5

# 1. 데이터 로드
train = pd.read_csv(os.path.join(DATA_DIR, "ChEMBL_ASK1(IC50).csv"), sep=";")
pub   = pd.read_csv(os.path.join(DATA_DIR, "PubChem_ASK1.csv"))
train = pd.concat([train, pub], ignore_index=True)
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# 2. Feature Engineering
def featurize(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048 + 6)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
    ]
    return np.concatenate([np.array(fp), np.array(desc)])

X_train = np.vstack(train["Smiles"].apply(featurize).values)
y_train = np.log1p(train["IC50_nM"].values)      # 자연로그 변환
X_test  = np.vstack(test["Smiles"].apply(featurize).values)

# 3. K-Fold + LightGBM
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
oof, preds = np.zeros(len(y_train)), np.zeros(len(X_test))

params = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=256,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression",
    random_state=SEED,
)

for i, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    model = LGBMRegressor(**params)
    model.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])],
              eval_metric="rmse",
              verbose=False,
              early_stopping_rounds=80)
    oof[val_idx] = model.predict(X_train[val_idx])
    preds += model.predict(X_test) / NFOLD
    joblib.dump(model, f"lgbm_fold{i}.pkl")

rmse = mean_squared_error(y_train, oof, squared=False)
print(f"CV RMSE = {rmse:.4f}")

# 4. 예측값 역변환 & 제출
submission = pd.DataFrame({
    "ID": test["ID"],
    "ASK1_IC50_nM": np.expm1(preds)    # 역변환
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
