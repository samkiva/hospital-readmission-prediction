"""
Hospital Readmission Prediction - Model Training & Evaluation
==============================================================

This script implements:
- LightGBM model training with hyperparameter tuning
- Comprehensive evaluation (AUC, precision, recall, F1, confusion matrix)
- Fairness auditing across demographic subgroups
- SHAP interpretability analysis
- Overfitting mitigation strategies

Author: Kivairu Samuel
Date: 12th November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import shap
import warnings
import os
import joblib
import importlib.util
warnings.filterwarnings('ignore')

# Import from 01_data_generation.py using importlib
spec = importlib.util.spec_from_file_location("data_generation", "01_data_generation.py")
data_generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_generation)
HospitalReadmissionPreprocessor = data_generation.HospitalReadmissionPreprocessor
prepare_train_val_test_split = data_generation.prepare_train_val_test_split

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# SECTION 1: MODEL TRAINING
# ============================================================================

class ReadmissionModel:
    """
    LightGBM-based hospital readmission prediction model with:
    - Hyperparameter tuning
    - Early stopping
    - Regularization to prevent overfitting
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True):
        """
        Train LightGBM model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data for early stopping
        tune_hyperparameters : bool
            Whether to perform grid search for hyperparameters
        """
        
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        if tune_hyperparameters:
            print("\n[Step 1/3] Hyperparameter Tuning...")
            self._tune_hyperparameters(X_train, y_train)
        else:
            # Use default parameters with regularization
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'num_leaves': 31,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 0.1,  # L2 regularization
            }
        
        print("\n[Step 2/3] Training Final Model...")
        print(f"Best parameters: {self.best_params}")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.best_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        print(f"\n[OK] Training complete!")
        print(f"[OK] Best iteration: {self.model.best_iteration}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\n[Step 3/3] Top 10 Most Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))
        
        return self
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Grid search for optimal hyperparameters."""
        
        # Parameter grid (limited for demonstration)
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [300, 500],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [20, 50],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        }
        
        # Use smaller grid for speed
        param_grid_small = {
            'max_depth': [6],
            'learning_rate': [0.05],
            'n_estimators': [500],
            'num_leaves': [31],
            'min_child_samples': [50],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1]
        }
        
        base_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1
        )
        
        # 3-fold cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid_small,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        print(f"   Best cross-validation AUC: {grid_search.best_score_:.4f}")
    
    def predict_proba(self, X):
        """Predict readmission probabilities."""
        return self.model.predict(X)
    
    def predict(self, X, threshold=0.5):
        """Predict readmission (binary) using threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# ============================================================================
# SECTION 2: MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics and visualizations."""
    
    def __init__(self, model, X_test, y_test, X_train=None, y_train=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred_proba = model.predict_proba(X_test)
        self.y_pred = model.predict(X_test, threshold=0.5)
        
    def evaluate_all(self):
        """Run all evaluation analyses."""
        
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        self.print_performance_metrics()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.check_overfitting()
        self.find_optimal_threshold()
        
    def print_performance_metrics(self):
        """Calculate and display key performance metrics."""
        
        print("\n[PERFORMANCE METRICS]")
        print("-" * 70)
        
        # Calculate metrics
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # Number needed to evaluate
        nne = 1 / precision if precision > 0 else float('inf')
        
        print(f"\nClassification Metrics (Threshold = 0.5):")
        print(f"  AUC-ROC:       {auc:.4f}")
        print(f"  Accuracy:      {accuracy:.4f}")
        print(f"  Precision:     {precision:.4f}  (PPV - Of predicted high-risk, % truly high-risk)")
        print(f"  Recall:        {recall:.4f}  (Sensitivity - % of actual readmissions caught)")
        print(f"  Specificity:   {specificity:.4f}  (% of non-readmissions correctly identified)")
        print(f"  F1-Score:      {f1:.4f}  (Harmonic mean of precision & recall)")
        print(f"  NNE:           {nne:.2f}  (Number Needed to Evaluate)")
        
        # Confusion matrix values
        print(f"\nConfusion Matrix Values:")
        print(f"  True Negatives:  {tn:,}  (Correctly predicted no readmission)")
        print(f"  False Positives: {fp:,}  (Incorrectly predicted readmission)")
        print(f"  False Negatives: {fn:,}  (Missed readmissions - concerning!)")
        print(f"  True Positives:  {tp:,}  (Correctly predicted readmission)")
        
        # Clinical interpretation
        print(f"\nClinical Interpretation:")
        print(f"  • Model catches {recall*100:.1f}% of patients who will be readmitted")
        print(f"  • {100-recall*100:.1f}% of readmissions are missed (false negatives)")
        print(f"  • {precision*100:.1f}% of high-risk predictions are correct")
        print(f"  • {100-precision*100:.1f}% of high-risk flags are false alarms")
        print(f"  • Need to intervene on {nne:.1f} patients to prevent 1 readmission")
        
        return {
            'auc': auc, 'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1, 'specificity': specificity,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    
    def plot_confusion_matrix(self):
        """Visualize confusion matrix."""
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['No Readmission', 'Readmission'],
                   yticklabels=['No Readmission', 'Readmission'])
        plt.title('Confusion Matrix - Hospital Readmission Prediction', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                plt.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")
        plt.close()
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'LightGBM (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curve - Readmission Prediction Model', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved to 'roc_curve.png'")
        plt.close()
    
    def plot_precision_recall_curve(self):
        """Plot precision-recall curve."""
        
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, color='#2E86AB')
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (PPV)', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Mark current operating point (threshold=0.5)
        current_precision = precision_score(self.y_test, self.y_pred)
        current_recall = recall_score(self.y_test, self.y_pred)
        plt.plot(current_recall, current_precision, 'ro', markersize=10, 
                label=f'Current (threshold=0.5): P={current_precision:.3f}, R={current_recall:.3f}')
        
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Precision-recall curve saved to 'precision_recall_curve.png'")
        plt.close()
    
    def check_overfitting(self):
        """Check for overfitting by comparing train vs test performance."""
        
        if self.X_train is None or self.y_train is None:
            print("\n⚠ Training data not provided, skipping overfitting check")
            return
        
        print("\n[OVERFITTING CHECK]")
        print("-" * 70)
        
        # Calculate AUC on training data
        y_train_pred_proba = self.model.predict_proba(self.X_train)
        train_auc = roc_auc_score(self.y_train, y_train_pred_proba)
        test_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        auc_gap = train_auc - test_auc
        
        print(f"\nAUC Comparison:")
        print(f"  Training AUC:   {train_auc:.4f}")
        print(f"  Test AUC:       {test_auc:.4f}")
        print(f"  Gap:            {auc_gap:.4f}")
        
        if auc_gap < 0.05:
            print("  ✓ Status: Good generalization (gap < 0.05)")
        elif auc_gap < 0.10:
            print("  ⚠ Status: Slight overfitting (gap 0.05-0.10)")
        else:
            print("  ✗ Status: Significant overfitting (gap > 0.10)")
            print("    Recommendation: Increase regularization or reduce model complexity")
    
    def find_optimal_threshold(self):
        """Find optimal classification threshold."""
        
        print("\n[THRESHOLD OPTIMIZATION]")
        print("-" * 70)
        
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0.1, 0.9, 17)
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            precision = precision_score(self.y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(self.y_test, y_pred_thresh)
            f1 = f1_score(self.y_test, y_pred_thresh)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        results_df = pd.DataFrame(results)
        
        # Find threshold that maximizes F1
        best_f1_idx = results_df['f1'].idxmax()
        best_threshold = results_df.loc[best_f1_idx, 'threshold']
        
        print(f"\nCurrent threshold: 0.5")
        print(f"Optimal threshold (max F1): {best_threshold:.2f}")
        print(f"\nMetrics at different thresholds:")
        print(results_df.to_string(index=False))
        
        # Plot threshold analysis
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['threshold'], results_df['precision'], 
                marker='o', label='Precision', linewidth=2)
        plt.plot(results_df['threshold'], results_df['recall'], 
                marker='s', label='Recall', linewidth=2)
        plt.plot(results_df['threshold'], results_df['f1'], 
                marker='^', label='F1-Score', linewidth=2)
        plt.axvline(best_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold ({best_threshold:.2f})', linewidth=2)
        plt.xlabel('Classification Threshold', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title('Performance Metrics vs. Classification Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
        print("\n✓ Threshold analysis saved to 'threshold_optimization.png'")
        plt.close()


# ============================================================================
# SECTION 3: FAIRNESS AUDITING
# ============================================================================

class FairnessAuditor:
    """Audit model for bias across demographic subgroups."""
    
    def __init__(self, model, X_test, y_test, demographics_df):
        """
        Parameters:
        -----------
        model : Trained model
        X_test : Test features
        y_test : Test labels
        demographics_df : DataFrame with demographic columns (race, gender, age_group)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.demographics = demographics_df
        self.y_pred_proba = model.predict_proba(X_test)
        self.y_pred = model.predict(X_test, threshold=0.5)
        
    def audit_fairness(self):
        """Perform comprehensive fairness audit."""
        
        print("\n" + "="*70)
        print("FAIRNESS AUDIT")
        print("="*70)
        
        print("\n[DEMOGRAPHIC PARITY CHECK]")
        print("-" * 70)
        
        # Analyze by race
        if 'race' in self.demographics.columns:
            print("\nPerformance by Race:")
            self._audit_by_group('race')
        
        # Analyze by gender
        if 'gender' in self.demographics.columns:
            print("\nPerformance by Gender:")
            self._audit_by_group('gender')
        
        # Analyze by age group
        if 'age_group' in self.demographics.columns:
            print("\nPerformance by Age Group:")
            self._audit_by_group('age_group')
        
    def _audit_by_group(self, group_col):
        """Calculate metrics for each subgroup."""
        
        results = []
        
        for group in self.demographics[group_col].unique():
            if pd.isna(group):
                continue
                
            mask = (self.demographics[group_col] == group)
            n = mask.sum()
            
            if n < 10:  # Skip small groups
                continue
            
            y_true_group = self.y_test[mask]
            y_pred_group = self.y_pred[mask]
            y_pred_proba_group = self.y_pred_proba[mask]
            
            # Calculate metrics
            auc = roc_auc_score(y_true_group, y_pred_proba_group)
            precision = precision_score(y_true_group, y_pred_group, zero_division=0)
            recall = recall_score(y_true_group, y_pred_group, zero_division=0)
            
            # False positive and false negative rates
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Prediction rate (% flagged as high-risk)
            pred_rate = y_pred_group.mean()
            
            results.append({
                'Group': group,
                'N': n,
                'AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'FPR': fpr,
                'FNR': fnr,
                'Pred_Rate': pred_rate
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Check for disparities
        if len(results_df) > 1:
            auc_gap = results_df['AUC'].max() - results_df['AUC'].min()
            recall_gap = results_df['Recall'].max() - results_df['Recall'].min()
            fpr_gap = results_df['FPR'].max() - results_df['FPR'].min()
            
            print(f"\nDisparity Analysis:")
            print(f"  AUC gap:          {auc_gap:.3f}")
            print(f"  Recall gap:       {recall_gap:.3f}")
            print(f"  FPR gap:          {fpr_gap:.3f}")
            
            if auc_gap < 0.05 and recall_gap < 0.10 and fpr_gap < 0.10:
                print(f"  ✓ Status: Fair (all gaps within acceptable range)")
            else:
                print(f"  ⚠ Status: Potential bias detected")
                print(f"    Recommendation: Review feature selection and consider fairness constraints")


# ============================================================================
# SECTION 4: INTERPRETABILITY (SHAP)
# ============================================================================

def explain_model_predictions(model, X_test, feature_names, num_samples=100):
    """Generate SHAP explanations for model predictions."""
    
    print("\n" + "="*70)
    print("MODEL INTERPRETABILITY (SHAP)")
    print("="*70)
    
    print(f"\nCalculating SHAP values for {num_samples} test samples...")
    print("(This may take a few minutes...)")
    
    # Use TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(model.model)
    
    # Calculate SHAP values for subset of test data
    X_subset = X_test.iloc[:num_samples]
    shap_values = explainer.shap_values(X_subset)
    
    # If binary classification, shap_values is a list [class_0, class_1]
    # We want class 1 (readmission)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print("✓ SHAP values calculated!")
    
    # Plot 1: Global feature importance (bar plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_subset, plot_type="bar", show=False)
    plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ SHAP feature importance saved to 'shap_feature_importance.png'")
    plt.close()
    
    # Plot 2: Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_subset, show=False, max_display=20)
    plt.title('SHAP Summary Plot (Feature Impact)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    print("✓ SHAP summary plot saved to 'shap_summary.png'")
    plt.close()
    
    # Print top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\nTop 15 Most Important Features (by SHAP):")
    print(feature_importance_shap.head(15).to_string(index=False))
    
    return shap_values, explainer


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("HOSPITAL READMISSION PREDICTION - MODEL TRAINING & EVALUATION")
    print("="*70)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    X_val = pd.read_csv('X_val.csv')
    y_val = pd.read_csv('y_val.csv').values.ravel()
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values.ravel()
    
    print(f"✓ Data loaded successfully")
    print(f"  Training: {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Train model
    model = ReadmissionModel(random_state=42)
    model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=False)
    
    # Evaluate model
    evaluator = ModelEvaluator(model, X_test, y_test, X_train, y_train)
    metrics = evaluator.evaluate_all()
    
    # Fairness audit (load demographics from original data)
    raw_data = pd.read_csv('synthetic_patient_data.csv')
    test_indices = X_test.index
    demographics_test = raw_data.iloc[test_indices][['race', 'gender', 'age']]
    demographics_test['age_group'] = pd.cut(
        demographics_test['age'], 
        bins=[0, 50, 65, 80, 100], 
        labels=['<50', '50-65', '65-80', '80+']
    )
    
    fairness_auditor = FairnessAuditor(model, X_test, y_test, demographics_test)
    fairness_auditor.audit_fairness()
    
    # SHAP interpretability
    shap_values, explainer = explain_model_predictions(
        model, X_test, X_test.columns.tolist(), num_samples=100
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.model.save_model('models/readmission_model.txt')
    print("\n✓ Model saved to 'models/readmission_model.txt'")
    joblib.dump(model, 'models/readmission_model.pkl')
    print("✓ Model saved to 'models/readmission_model.pkl'")

    # Save preprocessor
    preprocessor = HospitalReadmissionPreprocessor()
    raw_data = pd.read_csv('synthetic_patient_data.csv')
    preprocessor.fit_transform(raw_data)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("✓ Preprocessor saved to 'models/preprocessor.pkl'")

    print("\n" + "="*70)
    print("✓ MODEL TRAINING & EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - threshold_optimization.png")
    print("  - shap_feature_importance.png")
    print("  - shap_summary.png")
    print("\nModel ready for deployment!")
