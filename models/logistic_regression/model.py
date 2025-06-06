import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Get the current directory (logistic_regression) and parent directory (models)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)

# Create necessary directories
os.makedirs(os.path.join(CURRENT_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, 'results'), exist_ok=True)

def engineer_features(df):
    """Engineer additional features from the dataset."""
    print("Engineering features...")
    
    df_eng = df.copy()
    
    # Text-based features
    if 'text_combined' in df_eng.columns:
        text_col = 'text_combined'
    elif 'body' in df_eng.columns:
        text_col = 'body'
    else:
        text_col = None
    
    if text_col:
        # Basic text features
        df_eng['text_length'] = df_eng[text_col].fillna('').str.len()
        df_eng['word_count'] = df_eng[text_col].fillna('').str.split().str.len()
        df_eng['unique_word_ratio'] = df_eng[text_col].fillna('').apply(
            lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
        )
        
        # Character-based features
        df_eng['capital_ratio'] = df_eng[text_col].fillna('').apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        df_eng['digit_ratio'] = df_eng[text_col].fillna('').apply(
            lambda x: sum(1 for c in x if c.isdigit()) / len(x) if len(x) > 0 else 0
        )
        df_eng['punctuation_ratio'] = df_eng[text_col].fillna('').apply(
            lambda x: sum(1 for c in x if c in '.,!?;:') / len(x) if len(x) > 0 else 0
        )
        
        # Special character features
        df_eng['special_char_ratio'] = df_eng[text_col].fillna('').apply(
            lambda x: sum(1 for c in x if c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`') / len(x) if len(x) > 0 else 0
        )
        
        # Word-based features
        df_eng['avg_word_length'] = df_eng[text_col].fillna('').apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        df_eng['max_word_length'] = df_eng[text_col].fillna('').apply(
            lambda x: max([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # URL and email features
        df_eng['url_count'] = df_eng[text_col].fillna('').str.count('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        df_eng['email_count'] = df_eng[text_col].fillna('').str.count(r'[\w\.-]+@[\w\.-]+\.\w+')
    
    # Drop original text columns
    text_cols = ['text_combined', 'body', 'subject']
    df_eng = df_eng.drop(columns=[col for col in text_cols if col in df_eng.columns])
    
    # Fill any remaining NaN values with 0
    df_eng = df_eng.fillna(0)
    
    # Ensure all columns are numeric
    for col in df_eng.columns:
        if col != 'label':
            df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce')
    
    # Fill any NaN values that might have been created during conversion
    df_eng = df_eng.fillna(0)
    
    return df_eng

def load_and_prepare_data():
    """Load and prepare the dataset for modeling."""
    print("Loading and preparing data...")
    
    # Load the full dataset
    print("Loading full dataset...")
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'phishing_email.csv'))
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    print(df['label'].value_counts())
    
    # Engineer features
    print("\nEngineering features...")
    df_eng = engineer_features(df)
    
    # Separate majority and minority classes
    df_majority = df_eng[df_eng['label'] == 0]
    df_minority = df_eng[df_eng['label'] == 1]
    
    # Upsample minority class to match majority class
    print("\nUpsampling minority class...")
    df_minority_upsampled = resample(df_minority,
                                   replace=True,
                                   n_samples=len(df_majority),
                                   random_state=42)
    
    # Combine majority and upsampled minority
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal class distribution:")
    print(df_balanced['label'].value_counts())
    
    # Print feature information
    print("\nFeatures used in the model:")
    feature_names = [col for col in df_balanced.columns if col != 'label']
    for col in feature_names:
        print(f"- {col}")
    
    # Scale the features using RobustScaler
    print("\nScaling features...")
    scaler = RobustScaler()
    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaler and feature names for later use
    joblib.dump({
        'scaler': scaler,
        'feature_names': X.columns
    }, os.path.join(CURRENT_DIR, 'preprocessor.joblib'))
    
    return X_scaled, y, X.columns

def select_features(X, y, feature_names):
    """Select the most important features using Recursive Feature Elimination."""
    print("\nSelecting important features...")
    
    # Calculate the number of features to select (80% of available features)
    n_features = X.shape[1]
    n_features_to_select = max(1, int(n_features * 0.8))
    
    # Create a Logistic Regression estimator for feature selection
    base_estimator = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    )
    
    # Initialize RFE
    selector = RFE(
        estimator=base_estimator,
        n_features_to_select=n_features_to_select,
        step=1
    )
    
    # Fit the selector
    selector.fit(X, y)
    
    # Get selected feature indices and names
    selected_features = feature_names[selector.support_]
    
    # Transform the data to keep only selected features
    X_selected = selector.transform(X)
    
    print(f"\nSelected {len(selected_features)} features:")
    for feature in selected_features:
        print(f"- {feature}")
    
    return X_selected, selected_features

def optimize_logistic_regression(X_train, y_train):
    """Perform hyperparameter optimization for Logistic Regression."""
    print("\nPerforming hyperparameter optimization...")
    
    # Define the parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']
    }
    
    # Initialize base model
    base_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Perform grid search
    print("\nStarting grid search...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest F1 score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model and generate comprehensive metrics and visualizations."""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CURRENT_DIR, 'plots', 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(CURRENT_DIR, 'plots', 'roc_curve.png'))
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(CURRENT_DIR, 'plots', 'precision_recall_curve.png'))
    plt.close()
    
    # Feature Coefficients
    coef_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': abs(model.coef_[0])
    })
    coef_importance = coef_importance.sort_values('coefficient', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='coefficient', y='feature', data=coef_importance)
    plt.title('Feature Importance (Absolute Coefficients)')
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'plots', 'feature_importance.png'))
    plt.close()
    
    # Save feature importance to CSV
    coef_importance.to_csv(os.path.join(CURRENT_DIR, 'feature_importance.csv'), index=False)
    
    # Additional cross-validation evaluation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1_score': f1,
        'cv_scores': cv_scores,
        'feature_importance': coef_importance
    }

def generate_results_markdown(metrics, best_params, timestamp):
    """Generate markdown file with model performance results."""
    content = f"""# Logistic Regression Model Performance Results

## Timestamp
{timestamp}

## Performance Metrics
- Accuracy: {metrics['accuracy']:.3f}
- ROC AUC: {metrics['roc_auc']:.3f}
- Average Precision: {metrics['avg_precision']:.3f}
- F1 Score: {metrics['f1_score']:.3f}
- Mean CV F1 Score: {metrics['cv_scores'].mean():.3f} (±{metrics['cv_scores'].std() * 2:.3f})

## Best Model Parameters
"""
    
    for param, value in best_params.items():
        content += f"- {param}: {value}\n"
    
    # Save the markdown file
    with open(os.path.join(CURRENT_DIR, 'results', f'model_performance_{timestamp}.md'), 'w') as f:
        f.write(content)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data()
    
    # Select features
    X_selected, selected_features = select_features(X, y, feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimize the model
    best_model = optimize_logistic_regression(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(best_model, X_test, y_test, selected_features)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model
    model_path = os.path.join(CURRENT_DIR, f'logistic_regression_model_{timestamp}.joblib')
    joblib.dump(best_model, model_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'metrics': metrics,
        'best_params': best_model.get_params(),
        'selected_features': list(selected_features)
    }
    joblib.dump(metadata, os.path.join(CURRENT_DIR, f'model_metadata_{timestamp}.joblib'))
    
    # Generate results markdown
    generate_results_markdown(metrics, best_model.get_params(), timestamp)
    
    print(f"\nModel saved to: {model_path}")
