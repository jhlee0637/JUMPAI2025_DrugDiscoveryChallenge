# Final Submission - ASK1 IC50 Prediction

## Usage
```bash
python jump_ai_final_optimized.py
```

## Files
- `jump_ai_final_optimized.py` - Main prediction script
- `optimized_ensemble_model.pkl` - Trained ensemble model (500KB)
- `requirements.txt` - Dependencies
- `configs/best_hyperparameters.json` - Optimized parameters
- `results/final_submission.csv` - Competition submission

## Model
- Ensemble: RandomForest + GradientBoosting + LightGBM + XGBoost
- Features: RDKit molecular descriptors + Morgan fingerprints
- Validation: 5-fold cross-validation
- Performance: Optimized for ASK1 IC50 prediction

Co-authored-by: @junu6542
