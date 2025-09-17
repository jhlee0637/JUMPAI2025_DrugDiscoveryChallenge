import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 시드 고정
seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### IC50 to pIC50 변환 함수
def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

### 데이터 로드 및 전처리
chembl = pd.read_csv("/Users/skku_aws24/Desktop/AIdrug/open/ChEMBL_ASK1(IC50).csv", sep=';')
pubchem = pd.read_csv("/Users/skku_aws24/Desktop/AIdrug/open/Pubchem_ASK1.csv")

chembl.columns = chembl.columns.str.strip().str.replace('\"', '')
chembl = chembl[chembl['Standard Type'] == 'IC50']
chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')

pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')

total = pd.concat([chembl, pubchem], ignore_index=True)
total = total.drop_duplicates(subset='smiles')
total['pIC50'] = IC50_to_pIC50(total['ic50_nM'])
total = total[total['ic50_nM'] > 0].dropna(subset=['smiles', 'pIC50'])

# 유효한 SMILES만 필터링
total['mol'] = total['smiles'].apply(Chem.MolFromSmiles)
total = total.dropna(subset=['mol']).reset_index(drop=True)

### RDKit Mol -> PyG Graph 변환 함수
def mol_to_graph_data_obj(mol, y=None):
    # 원자 수
    num_atoms = mol.GetNumAtoms()

    # 노드 특성: 원자 번호 (atomic number)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    # 엣지 (결합) 정보 추출
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 무방향 그래프이므로 양쪽 다 추가
        edge_index.append([i, j])
        edge_index.append([j, i])

        # 결합 타입 원-핫 인코딩 (예: 단일, 이중, 삼중, 방향족)
        bond_type = bond.GetBondType()
        bond_feat = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
        ]
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    if y is not None:
        y = torch.tensor([y], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

### 전체 데이터셋 PyG 형식으로 변환
graph_list = []
for idx, row in total.iterrows():
    graph = mol_to_graph_data_obj(row['mol'], row['pIC50'])
    graph_list.append(graph)

### 데이터셋 분할 (간단하게 랜덤으로)
num_total = len(graph_list)
num_train = int(num_total * 0.8)
num_valid = num_total - num_train

random.shuffle(graph_list)
train_dataset = graph_list[:num_train]
valid_dataset = graph_list[num_train:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

### GNN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)   # input feature dim = 1 (atomic number)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)  # output regression

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)  # graph-level representation

        x = self.lin(x)

        return x.squeeze()

### 모델, 옵티마이저, 손실함수
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

### 학습 함수
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

### 검증 함수
def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

### 학습 루프
epochs = 100
for epoch in range(1, epochs+1):
    train_loss = train()
    valid_loss = evaluate(valid_loader)
    print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

### 테스트 데이터 예측 (테스트 데이터도 graph로 변환 필요)
test_df = pd.read_csv("/Users/skku_aws24/Desktop/AIdrug/open/test.csv")
test_df['mol'] = test_df['Smiles'].apply(Chem.MolFromSmiles)
test_df = test_df.dropna(subset=['mol']).reset_index(drop=True)

test_graphs = [mol_to_graph_data_obj(mol) for mol in test_df['mol']]
test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        predictions.extend(output.cpu().numpy())

def pIC50_to_IC50_pred(pIC50):
    return 10 ** (9 - pIC50)

test_df['pIC50_pred'] = predictions
test_df['ASK1_IC50_nM'] = test_df['pIC50_pred'].apply(pIC50_to_IC50_pred)

submission = pd.read_csv("/Users/skku_aws24/Desktop/AIdrug/open/sample_submission.csv")
submission['ASK1_IC50_nM'] = test_df['ASK1_IC50_nM']
submission.to_csv("/Users/skku_aws24/Desktop/AIdrug/output/gnn_pytorch.csv", index=False)

print("Submission file created successfully with PyTorch Geometric GNN model!")
