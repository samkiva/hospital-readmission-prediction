"""
Hospital Readmission Prediction - Visualization Generator
==========================================================

This script generates all visualizations needed for:
- Model evaluation
- Fairness auditing
- PDF reports
- Presentations

Author: Kivairu Samuel
Date: 12th November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class VisualizationGenerator:
    """Generate all visualizations for the readmission prediction project."""
    
    def __init__(self, output_dir='visualizations'):
        """Initialize visualization generator."""
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Visualizations will be saved to: {output_dir}/")
    
    def generate_all(self, y_true, y_pred, y_pred_proba, 
                    demographics=None, feature_importance=None):
        """
        Generate all visualizations.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like
            Predicted probabilities
        demographics : pd.DataFrame, optional
            Demographics data for fairness analysis
        feature_importance : pd.DataFrame, optional
            Feature importance scores
        """
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Model Performance Visualizations
        self.plot_confusion_matrix_detailed(y_true, y_pred)
        self.plot_roc_curve(y_true, y_pred_proba)
        self.plot_precision_recall_curve(y_true, y_pred_proba)
        self.plot_threshold_analysis(y_true, y_pred_proba)
        self.plot_metrics_summary(y_true, y_pred, y_pred_proba)
        
        # Feature Importance
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance)
        
        # Fairness Visualizations
        if demographics is not None:
            self.plot_fairness_audit(y_true, y_pred, y_pred_proba, demographics)
            self.plot_fairness_metrics_grid(y_true, y_pred, y_pred_proba, demographics)
        
        # Risk Distribution
        self.plot_risk_distribution(y_true, y_pred_proba)
        
        # Calibration Plot
        self.plot_calibration_curve(y_true, y_pred_proba)
        
        print(f"\n✓ All visualizations generated successfully!")
        print(f"✓ Saved to: {self.output_dir}/")
    
    def plot_confusion_matrix_detailed(self, y_true, y_pred):
        """Enhanced confusion matrix with multiple views."""
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                   xticklabels=['No Readmit', 'Readmit'],
                   yticklabels=['No Readmit', 'Readmit'],
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'})
        axes[0].set_xlabel('Predicted', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=13, fontweight='bold')
        axes[0].set_title('Confusion Matrix\n(Counts)', fontsize=14, fontweight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                axes[0].text(j+0.5, i+0.75, f'({pct:.1f}%)', 
                           ha='center', va='center', fontsize=11, color='gray')
        
        # Plot 2: Normalized (by true class)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Oranges', cbar=False, ax=axes[1],
                   xticklabels=['No Readmit', 'Readmit'],
                   yticklabels=['No Readmit', 'Readmit'],
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'})
        axes[1].set_xlabel('Predicted', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Actual', fontsize=13, fontweight='bold')
        axes[1].set_title('Confusion Matrix\n(Normalized by Row)', fontsize=14, fontweight='bold')
        
        # Plot 3: Metrics breakdown
        tn, fp, fn, tp = cm.ravel()
        metrics_data = {
            'True\nNegative': tn,
            'False\nPositive': fp,
            'False\nNegative': fn,
            'True\nPositive': tp
        }
        colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
        bars = axes[2].bar(metrics_data.keys(), metrics_data.values(), color=colors, 
                          edgecolor='black', linewidth=2, alpha=0.8)
        axes[2].set_ylabel('Count', fontsize=13, fontweight='bold')
        axes[2].set_title('Confusion Matrix\n(Component Breakdown)', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Confusion matrix (detailed)")
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """ROC curve with confidence interval."""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=3, label=f'Model (AUC = {auc:.3f})', color='#2E86AB')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)', alpha=0.7)
        ax.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
        
        # Mark optimal point (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12, 
               label=f'Optimal Threshold ({optimal_threshold:.2f})')
        
        # Add reference lines
        ax.axhline(y=0.7, color='red', linestyle=':', alpha=0.5, label='Target Recall (0.70)')
        ax.axvline(x=0.1, color='green', linestyle=':', alpha=0.5, label='Acceptable FPR (0.10)')
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Recall/Sensitivity)', fontsize=13, fontweight='bold')
        ax.set_title('ROC Curve - Hospital Readmission Prediction', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ ROC curve")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Precision-Recall curve."""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, linewidth=3, color='#A23B72')
        ax.fill_between(recall, precision, alpha=0.3, color='#A23B72')
        
        # Mark optimal F1 point
        ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=12,
               label=f'Optimal F1 ({f1_scores[optimal_idx]:.3f})')
        
        # Mark current operating point (threshold=0.5)
        current_precision = precision_score(y_true, (y_pred_proba >= 0.5).astype(int))
        current_recall = recall_score(y_true, (y_pred_proba >= 0.5).astype(int))
        ax.plot(current_recall, current_precision, 'gs', markersize=12,
               label=f'Current (thresh=0.5): P={current_precision:.3f}, R={current_recall:.3f}')
        
        # Baseline (prevalence)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Baseline (Prevalence = {baseline:.3f})')
        
        ax.set_xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=13, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Precision-Recall curve")
    
    def plot_threshold_analysis(self, y_true, y_pred_proba):
        """Analyze performance at different thresholds."""
        
        thresholds = np.linspace(0.1, 0.9, 17)
        metrics = {'precision': [], 'recall': [], 'f1': [], 'specificity': []}
        
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            
            metrics['precision'].append(precision_score(y_true, y_pred_thresh, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred_thresh, zero_division=0))
            metrics['f1'].append((2 * metrics['precision'][-1] * metrics['recall'][-1]) / 
                                (metrics['precision'][-1] + metrics['recall'][-1] + 1e-10))
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            metrics['specificity'].append(tn / (tn + fp))
        
        # Find optimal thresholds
        best_f1_idx = np.argmax(metrics['f1'])
        best_f1_thresh = thresholds[best_f1_idx]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(thresholds, metrics['precision'], marker='o', linewidth=2, 
               markersize=8, label='Precision', color='#3498db')
        ax.plot(thresholds, metrics['recall'], marker='s', linewidth=2, 
               markersize=8, label='Recall', color='#e74c3c')
        ax.plot(thresholds, metrics['f1'], marker='^', linewidth=2, 
               markersize=8, label='F1-Score', color='#2ecc71')
        ax.plot(thresholds, metrics['specificity'], marker='d', linewidth=2, 
               markersize=8, label='Specificity', color='#9b59b6', linestyle='--')
        
        ax.axvline(best_f1_thresh, color='red', linestyle='--', linewidth=2,
                  label=f'Optimal F1 Threshold ({best_f1_thresh:.2f})', alpha=0.7)
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=2,
                  label='Current Threshold (0.50)', alpha=0.5)
        
        ax.set_xlabel('Classification Threshold', fontsize=13, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
        ax.set_title('Performance Metrics vs. Classification Threshold', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Threshold analysis")
    
    def plot_metrics_summary(self, y_true, y_pred, y_pred_proba):
        """Summary dashboard of key metrics."""
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
        
        # Metric gauges
        metrics_data = [
            ('AUC-ROC', auc, 0.75, axes[0, 0]),
            ('Precision', precision, 0.60, axes[0, 1]),
            ('Recall', recall, 0.70, axes[0, 2]),
            ('F1-Score', f1, 0.65, axes[1, 0]),
            ('Specificity', specificity, 0.80, axes[1, 1]),
            ('Accuracy', accuracy, 0.80, axes[1, 2])
        ]
        
        for name, value, target, ax in metrics_data:
            # Create gauge
            self._draw_gauge(ax, name, value, target)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Metrics summary dashboard")
    
    def _draw_gauge(self, ax, name, value, target):
        """Helper function to draw a gauge chart."""
        
        # Determine color based on performance
        if value >= target:
            color = '#2ecc71'  # Green
            status = '✓'
        elif value >= target * 0.9:
            color = '#f39c12'  # Orange
            status = '⚠'
        else:
            color = '#e74c3c'  # Red
            status = '✗'
        
        # Draw gauge
        ax.barh([0], [value], height=0.5, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.barh([0], [1-value], left=value, height=0.5, color='lightgray', alpha=0.3)
        
        # Add target line
        ax.axvline(target, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 0.5])
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(f'{name}\n{status} {value:.3f} (Target: {target:.2f})', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value label
        ax.text(value/2, 0, f'{value:.1%}', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance."""
        
        top_features = feature_importance.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=colors, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=11)
        ax.set_xlabel('Importance (Gain)', fontsize=13, fontweight='bold')
        ax.set_title('Top 20 Most Important Features', fontsize=15, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
            ax.text(val, i, f'  {val:.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Feature importance")
    
    def plot_fairness_audit(self, y_true, y_pred, y_pred_proba, demographics):
        """Fairness metrics across demographic groups."""
        
        # Calculate metrics by race
        results = []
        for race in demographics['race'].unique():
            if pd.isna(race):
                continue
            
            mask = (demographics['race'] == race)
            if mask.sum() < 10:
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            y_pred_proba_group = y_pred_proba[mask]
            
            if len(np.unique(y_true_group)) < 2:
                continue
            
            auc = roc_auc_score(y_true_group, y_pred_proba_group)
            prec = precision_score(y_true_group, y_pred_group, zero_division=0)
            rec = recall_score(y_true_group, y_pred_group, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            fpr = fp / (fp + tn)
            
            results.append({
                'Race': race,
                'N': mask.sum(),
                'AUC': auc,
                'Precision': prec,
                'Recall': rec,
                'FPR': fpr
            })
        
        if not results:
            print("  ⚠ Insufficient data for fairness analysis")
            return
        
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fairness Audit: Performance by Race/Ethnicity', 
                    fontsize=16, fontweight='bold')
        
        # AUC by race
        bars = axes[0, 0].bar(results_df['Race'], results_df['AUC'], 
                             color='skyblue', edgecolor='black', linewidth=2)
        axes[0, 0].axhline(y=0.75, color='red', linestyle='--', linewidth=2, 
                          label='Target (0.75)')
        axes[0, 0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('AUC by Race', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0.6, 1.0])
        for bar, val in zip(bars, results_df['AUC']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                          f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Recall by race
        bars = axes[0, 1].bar(results_df['Race'], results_df['Recall'], 
                             color='lightcoral', edgecolor='black', linewidth=2)
        axes[0, 1].axhline(y=0.70, color='red', linestyle='--', linewidth=2, 
                          label='Target (0.70)')
        axes[0, 1].set_ylabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Recall by Race', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim([0.5, 1.0])
        for bar, val in zip(bars, results_df['Recall']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                          f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # FPR by race
        bars = axes[1, 0].bar(results_df['Race'], results_df['FPR'], 
                             color='lightgreen', edgecolor='black', linewidth=2)
        axes[1, 0].set_ylabel('False Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('FPR by Race', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0, 0.3])
        for bar, val in zip(bars, results_df['FPR']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                          f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Disparity gaps
        gaps = {
            'AUC Gap': results_df['AUC'].max() - results_df['AUC'].min(),
            'Recall Gap': results_df['Recall'].max() - results_df['Recall'].min(),
            'FPR Gap': results_df['FPR'].max() - results_df['FPR'].min()
        }
        
        colors_gap = ['green' if v < 0.05 else 'orange' if v < 0.10 else 'red' for v in gaps.values()]
        bars = axes[1, 1].bar(gaps.keys(), gaps.values(), color=colors_gap, 
                             edgecolor='black', linewidth=2, alpha=0.7)
        axes[1, 1].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, 
                          label='Warning (0.05)')
        axes[1, 1].axhline(y=0.10, color='red', linestyle='--', linewidth=2, 
                          label='Critical (0.10)')
        axes[1, 1].set_ylabel('Gap Magnitude', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Disparity Gaps Across Groups', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        for bar, (key, val) in zip(bars, gaps.items()):
            status = '✓' if val < 0.05 else '⚠' if val < 0.10 else '✗'
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                          f'{status}\n{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fairness_audit.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Fairness audit")
    
    def plot_fairness_metrics_grid(self, y_true, y_pred, y_pred_proba, demographics):
        """Fairness metrics heatmap."""
        
        # Calculate metrics for race and gender
        metrics_matrix = []
        groups = []
        
        for demographic in ['race', 'gender']:
            if demographic not in demographics.columns:
                continue
            
            for group in demographics[demographic].unique():
                if pd.isna(group):
                    continue
                
                mask = (demographics[demographic] == group)
                if mask.sum() < 10:
                    continue
                
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                y_pred_proba_group = y_pred_proba[mask]
                
                if len(np.unique(y_true_group)) < 2:
                    continue
                
                try:
                    auc = roc_auc_score(y_true_group, y_pred_proba_group)
                    prec = precision_score(y_true_group, y_pred_group, zero_division=0)
                    rec = recall_score(y_true_group, y_pred_group, zero_division=0)
                    
                    metrics_matrix.append([auc, prec, rec])
                    groups.append(f"{demographic.capitalize()}: {group}")
                except:
                    continue
        
        if not metrics_matrix:
            return
        
        metrics_df = pd.DataFrame(metrics_matrix, 
                                 index=groups,
                                 columns=['AUC', 'Precision', 'Recall'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.7, vmin=0.5, vmax=1.0,
                   cbar_kws={'label': 'Metric Value'},
                   linewidths=2, linecolor='black', ax=ax)
        ax.set_title('Fairness Metrics Heatmap\n(Darker Green = Better Performance)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Demographic Groups', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fairness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Fairness heatmap")
    
    def plot_risk_distribution(self, y_true, y_pred_proba):
        """Distribution of predicted risks."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram by true class
        axes[0].hist([y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]],
                    bins=30, label=['Not Readmitted', 'Readmitted'], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[0].set_xlabel('Predicted Readmission Risk', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Risk Score Distribution by True Class', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        data_box = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
        bp = axes[1].boxplot(data_box, labels=['Not Readmitted', 'Readmitted'],
                            patch_artist=True, showmeans=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
        axes[1].axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (0.5)')
        axes[1].set_ylabel('Predicted Readmission Risk', fontsize=12, fontweight='bold')
        axes[1].set_title('Risk Score Distribution (Box Plot)', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Risk distribution")
    
    def plot_calibration_curve(self, y_true, y_pred_proba, n_bins=10):
        """Calibration plot to assess prediction reliability."""
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate observed frequency in each bin
        observed_freq = []
        counts = []
        
        for i in range(n_bins):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
            if mask.sum() > 0:
                observed_freq.append(y_true[mask].mean())
                counts.append(mask.sum())
            else:
                observed_freq.append(np.nan)
                counts.append(0)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calibration curve
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        axes[0].plot(bin_centers, observed_freq, 'o-', linewidth=2, markersize=10,
                    color='#2E86AB', label='Model Calibration')
        
        # Add bin sizes as text
        for x, y, count in zip(bin_centers, observed_freq, counts):
            if not np.isnan(y) and count > 0:
                axes[0].text(x, y + 0.02, f'n={count}', ha='center', fontsize=8)
        
        axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Calibration Curve', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # Distribution of predictions
        axes[1].hist(y_pred_proba, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribution of Predicted Probabilities', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Calibration curve")


# Example usage
if __name__ == "__main__":
    # Load test data (assuming you've run the preprocessing pipeline)
    try:
        import pandas as pd
        
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').values.ravel()
        
        # Load model and generate predictions
        import lightgbm as lgb
        model = lgb.Booster(model_file='models/readmission_model.txt')
        
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Load demographics
        df_raw = pd.read_csv('synthetic_patient_data.csv')
        test_indices = X_test.index
        demographics = df_raw.iloc[test_indices][['race', 'gender', 'age']].copy()
        
        # Load feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        # Generate all visualizations
        viz = VisualizationGenerator(output_dir='visualizations')
        viz.generate_all(y_test, y_pred, y_pred_proba, demographics, feature_importance)
        
        print("\n✓ All visualizations generated successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Please run the data generation and model training scripts first.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed and data files exist.")