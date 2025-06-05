# Random Forest Model Performance Results

## Model Parameters
The best parameters found during hyperparameter optimization:
- **Bootstrap**: True
- **Class Weight**: balanced
- **Criterion**: entropy
- **Max Depth**: 30
- **Max Features**: sqrt
- **Min Samples Leaf**: 1
- **Min Samples Split**: 2
- **N Estimators**: 500

## Performance Metrics
### Overall Metrics
- **Accuracy**: 0.914
- **ROC AUC**: 0.968
- **Average Precision**: 0.973
- **F1 Score**: 0.913

### Classification Report
```
              precision    recall  f1-score   support

           0       0.91      0.92      0.91      7919
           1       0.92      0.91      0.91      7919

    accuracy                           0.91     15838
   macro avg       0.91      0.91      0.91     15838
weighted avg       0.91      0.91      0.91     15838
```

### Cross-Validation Results
- **Best F1 Score (CV)**: 0.901
- **Individual CV F1 Scores**: [0.852, 0.845, 0.860, 0.860, 0.860]
- **Mean CV F1 Score**: 0.856 (+/- 0.012)

## Model Files
- **Model File**: random_forest_model_20250604_125952.joblib
- **Metadata File**: model_metadata_20250604_125952.joblib
- **Visualizations**: Available in the `plots` directory
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance Plot

## Notes
- The model shows strong performance with balanced precision and recall for both classes
- High ROC AUC (0.968) indicates excellent discrimination ability
- Cross-validation scores show consistent performance across different folds
- The model was trained on a balanced dataset with 15,838 samples (7,919 per class) 