#!/usr/bin/env python3
"""
JUMP AI Competition: ASK1 IC50 Prediction with CAS MAP3K5 Data
=============================================================

This script creates a comprehensive ASK1 IC50 prediction model using:
1. CAS KPBMA MAP3K5 IC50 data as the core reference (Excel file)
2. ChEMBL ASK1 IC50 data 
3. PubChem ASK1 data
4. Advanced molecular fingerprinting and modeling techniques

Key Features:
- Uses CAS MAP3K5 data as primary training source (high-quality kinase data)
- Integrates multiple data sources with proper weighting
- Advanced molecular descriptors and fingerprints
- Ensemble modeling with cross-validation
- Biologically plausible IC50 range enforcement
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class JumpAICASPredictor:
    def __init__(self):
        """Initialize the predictor with CAS data as core reference"""
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.models = {}
        self.weights = {}
        self.feature_names = []
        
    def load_and_prepare_data(self):
        """Load and prepare all data sources with CAS as primary reference"""
        print("Loading CAS MAP3K5 IC50 data...")
        
        # Load CAS MAP3K5 IC50 data (primary source)
        cas_ic50 = pd.read_excel('../Data/CAS_KPBMA_MAP3K5_IC50s.xlsx', 
                                sheet_name='MAP3K5 Ligand IC50s', skiprows=1)
        
        # Filter and process CAS data
        cas_valid = cas_ic50.dropna(subset=['SMILES', 'pX Value'])
        cas_valid = cas_valid[cas_valid['pX Value'] > 0]  # Remove invalid pX values
        
        # Average multiple measurements for same compound
        cas_grouped = cas_valid.groupby('SMILES').agg({
            'pX Value': 'mean',
            'Single Value (Parsed)': 'mean',
            'Substance Name': 'first'
        }).reset_index()
        
        print(f"CAS data: {len(cas_grouped)} unique compounds")
        
        # Load ChEMBL ASK1 data (secondary source)
        print("Loading ChEMBL ASK1 data...")
        try:
            chembl_data = pd.read_csv('../Data/ChEMBL_ASK1(IC50).csv')
            chembl_valid = chembl_data.dropna(subset=['Smiles', 'pIC50'])
            chembl_valid = chembl_valid[chembl_valid['pIC50'] > 0]
            chembl_valid = chembl_valid.rename(columns={'Smiles': 'SMILES', 'pIC50': 'pX Value'})
            print(f"ChEMBL data: {len(chembl_valid)} compounds")
        except:
            print("ChEMBL data not available")
            chembl_valid = pd.DataFrame()
        
        # Load PubChem ASK1 data (tertiary source)
        print("Loading PubChem ASK1 data...")
        try:
            pubchem_data = pd.read_csv('../Data/Pubchem_ASK1.csv')
            pubchem_valid = pubchem_data.dropna(subset=['smiles', 'pic50'])
            pubchem_valid = pubchem_valid[pubchem_valid['pic50'] > 0]
            pubchem_valid = pubchem_valid.rename(columns={'smiles': 'SMILES', 'pic50': 'pX Value'})
            print(f"PubChem data: {len(pubchem_valid)} compounds")
        except:
            print("PubChem data not available")
            pubchem_valid = pd.DataFrame()
        
        # Combine all data sources with quality weights
        all_data = []
        
        # CAS data (highest weight - kinase-specific, high quality)
        cas_grouped['source'] = 'CAS'
        cas_grouped['weight'] = 1.0
        all_data.append(cas_grouped[['SMILES', 'pX Value', 'source', 'weight']])
        
        # ChEMBL data (medium weight - ASK1 specific but mixed quality)
        if not chembl_valid.empty:
            chembl_valid['source'] = 'ChEMBL'
            chembl_valid['weight'] = 0.7
            all_data.append(chembl_valid[['SMILES', 'pX Value', 'source', 'weight']])
        
        # PubChem data (lower weight - mixed sources and quality)
        if not pubchem_valid.empty:
            pubchem_valid['source'] = 'PubChem'
            pubchem_valid['weight'] = 0.5
            all_data.append(pubchem_valid[['SMILES', 'pX Value', 'source', 'weight']])
        
        # Combine all sources
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Handle duplicates by weighted averaging
        def weighted_mean(group):
            return np.average(group['pX Value'], weights=group['weight'])
        
        final_data = combined_data.groupby('SMILES').apply(
            lambda x: pd.Series({
                'pX Value': weighted_mean(x),
                'weight': x['weight'].max(),  # Use highest quality source weight
                'source_count': len(x),
                'sources': ','.join(x['source'].unique())
            })
        ).reset_index()
        
        print(f"Combined training data: {len(final_data)} unique compounds")
        print(f"Source distribution: {final_data['sources'].value_counts()}")
        
        # Load test data
        print("Loading test data...")
        test_data = pd.read_csv('../Data/test.csv')
        print(f"Test data: {len(test_data)} compounds")
        
        return final_data, test_data
    
    def calculate_molecular_features(self, smiles_list):
        """Calculate comprehensive molecular features"""
        print(f"Calculating molecular features for {len(smiles_list)} compounds...")
        
        features = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Basic molecular descriptors
            feature_dict = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
            }
            
            # Lipinski rule of 5 compliance
            feature_dict['Lipinski_violations'] = (
                (feature_dict['MolWt'] > 500) + 
                (feature_dict['LogP'] > 5) + 
                (feature_dict['NumHDonors'] > 5) + 
                (feature_dict['NumHAcceptors'] > 10)
            )
            
            # Advanced descriptors (use safer alternatives)
            try:
                feature_dict['BertzCT'] = Descriptors.BertzCT(mol)  # Complexity
            except:
                feature_dict['BertzCT'] = 0
            try:
                feature_dict['Kappa1'] = Descriptors.Kappa1(mol)   # Shape index
            except:
                feature_dict['Kappa1'] = 0
            try:
                feature_dict['Kappa2'] = Descriptors.Kappa2(mol)   # Shape index
            except:
                feature_dict['Kappa2'] = 0
            try:
                feature_dict['Chi0v'] = Descriptors.Chi0v(mol)     # Connectivity index
            except:
                feature_dict['Chi0v'] = 0
            try:
                feature_dict['Chi1v'] = Descriptors.Chi1v(mol)     # Connectivity index
            except:
                feature_dict['Chi1v'] = 0
            try:
                feature_dict['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            except:
                feature_dict['HallKierAlpha'] = 0
            
            # Pharmacophore features (important for kinase binding)
            feature_dict['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
            feature_dict['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
            feature_dict['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
            
            # Morgan fingerprint (circular fingerprints)
            morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            for j in range(1024):
                feature_dict[f'Morgan_{j}'] = morgan_fp[j]
            
            features.append(feature_dict)
            valid_indices.append(i)
        
        feature_df = pd.DataFrame(features)
        print(f"Calculated {len(feature_df.columns)} features for {len(feature_df)} valid molecules")
        
        return feature_df, valid_indices
    
    def train_ensemble_models(self, X_train, y_train, sample_weights=None):
        """Train ensemble of models with cross-validation"""
        print("Training ensemble models...")
        
        # Define models
        models_config = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'elastic': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=2000,
                random_state=42
            )
        }
        
        # Cross-validation scores
        cv_scores = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_scores[name] = -scores.mean()
            print(f"{name} CV MAE: {cv_scores[name]:.4f} (±{scores.std():.4f})")
            
            # Train on full data
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            self.models[name] = model
        
        # Calculate ensemble weights based on performance (inverse of error)
        total_inverse_error = sum(1/score for score in cv_scores.values())
        for name in models_config.keys():
            self.weights[name] = (1/cv_scores[name]) / total_inverse_error
        
        print("\nEnsemble weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return cv_scores
    
    def predict_ensemble(self, X_test):
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            predictions[name] = pred
        
        # Weighted average
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred, predictions
    
    def enforce_biological_constraints(self, predictions):
        """Enforce biologically plausible IC50 ranges"""
        # Convert pIC50 to IC50 (µM) for constraint checking
        ic50_values = 10 ** (6 - predictions)  # Convert pIC50 to µM
        
        # Constraint: IC50 should be between 0.001 µM and 1000 µM
        ic50_constrained = np.clip(ic50_values, 0.001, 1000)
        
        # Convert back to pIC50
        predictions_constrained = 6 - np.log10(ic50_constrained)
        
        return predictions_constrained
    
    def run_prediction_pipeline(self):
        """Run complete prediction pipeline"""
        print("=== JUMP AI CAS-Based ASK1 IC50 Prediction Pipeline ===\n")
        
        # Load data
        train_data, test_data = self.load_and_prepare_data()
        
        # Calculate features for training data
        train_features, train_valid_idx = self.calculate_molecular_features(train_data['SMILES'])
        
        # Filter training data to valid molecules
        train_data_valid = train_data.iloc[train_valid_idx].reset_index(drop=True)
        y_train = train_data_valid['pX Value'].values
        weights_train = train_data_valid['weight'].values
        
        # Feature selection
        print("Performing feature selection...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(500, len(train_features.columns)))
        X_train_selected = self.feature_selector.fit_transform(train_features, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        
        # Train models
        cv_scores = self.train_ensemble_models(X_train_scaled, y_train, weights_train)
        
        # Calculate features for test data
        test_features, test_valid_idx = self.calculate_molecular_features(test_data['Smiles'])
        
        # Apply same preprocessing
        X_test_selected = self.feature_selector.transform(test_features)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Make predictions
        print("\nMaking predictions...")
        ensemble_pred, individual_preds = self.predict_ensemble(X_test_scaled)
        
        # Apply biological constraints
        ensemble_pred_constrained = self.enforce_biological_constraints(ensemble_pred)
        
        # Convert pIC50 to nM (competition format requirement)
        # pIC50 = -log10(IC50_M) = 9 - log10(IC50_nM)
        # Therefore: IC50_nM = 10^(9 - pIC50)
        ic50_nM = 10**(9 - ensemble_pred_constrained)
        
        # Create submission dataframe in correct format
        test_data_valid = test_data.iloc[test_valid_idx].reset_index(drop=True)
        submission = pd.DataFrame({
            'ID': test_data_valid['ID'],  # Uppercase ID as required
            'ASK1_IC50_nM': ic50_nM       # nM values as required
        })
        
        # Add predictions for any failed molecules (use median value)
        if len(test_valid_idx) < len(test_data):
            missing_ids = test_data.loc[~test_data.index.isin(test_valid_idx), 'ID']
            median_pred_nM = np.median(ic50_nM)
            
            missing_df = pd.DataFrame({
                'ID': missing_ids,
                'ASK1_IC50_nM': median_pred_nM
            })
            
            submission = pd.concat([submission, missing_df], ignore_index=True)
        
        # Sort by ID to ensure proper order
        submission = submission.sort_values('ID').reset_index(drop=True)
        
        # Save results
        submission.to_csv('submission_cas.csv', index=False)
        
        # Save detailed results
        detailed_results = test_data.copy()
        detailed_results['predicted_pIC50'] = np.nan
        detailed_results.loc[test_valid_idx, 'predicted_pIC50'] = ensemble_pred_constrained
        
        # Fill missing predictions
        detailed_results['predicted_pIC50'].fillna(np.median(ensemble_pred_constrained), inplace=True)
        
        # Add individual model predictions
        for name, pred in individual_preds.items():
            detailed_results[f'{name}_pIC50'] = np.nan
            detailed_results.loc[test_valid_idx, f'{name}_pIC50'] = pred
        
        detailed_results.to_csv('submission_cas_detailed.csv', index=False)
        
        # Print summary
        print("\n=== Prediction Summary ===")
        print(f"Training compounds: {len(train_data_valid)}")
        print(f"Test compounds predicted: {len(test_data)}")
        print(f"Valid molecular structures: {len(test_valid_idx)}/{len(test_data)}")
        print(f"Predicted pIC50 range: {ensemble_pred_constrained.min():.2f} - {ensemble_pred_constrained.max():.2f}")
        print(f"Predicted IC50 range: {ic50_nM.min():.3f} - {ic50_nM.max():.3f} nM")
        print(f"Mean predicted IC50: {ic50_nM.mean():.3f} nM")
        print(f"Cross-validation MAE: {np.mean(list(cv_scores.values())):.4f}")
        
        print("\nResults saved to:")
        print("  - submission_cas.csv (competition format)")
        print("  - submission_cas_detailed.csv (detailed results)")
        
        return submission, detailed_results

if __name__ == "__main__":
    predictor = JumpAICASPredictor()
    submission, detailed_results = predictor.run_prediction_pipeline()
