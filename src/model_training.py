"""
Model training module for churn prediction.
Implements multiple ML algorithms and model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model."""
        print("Training Logistic Regression...")
        
        # Parameter grid for GridSearch
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['Logistic_Regression'] = grid_search.best_estimator_
        print(f"Best LR parameters: {grid_search.best_params_}")
        print(f"Best LR CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model."""
        print("Training Random Forest...")
        
        # Parameter grid for GridSearch
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['Random_Forest'] = grid_search.best_estimator_
        print(f"Best RF parameters: {grid_search.best_params_}")
        print(f"Best RF CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        print("Training XGBoost...")
        
        # Parameter grid for GridSearch
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        print(f"Best XGB parameters: {grid_search.best_params_}")
        print(f"Best XGB CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, feature_names):
        """Train all models and select the best one."""
        self.feature_names = feature_names
        
        # Train individual models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        print("\nAll models trained successfully!")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models."""
        results = {}
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n{name} Results:")
            print(f"ROC-AUC Score: {auc_score:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        # Select best model based on AUC score
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        return results
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance for the best model."""
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = abs(self.best_model.coef_[0])
        else:
            print("Model doesn't support feature importance")
            return
        
        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {type(self.best_model).__name__}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_df
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def save_models(self, filepath_prefix='models/churn_model'):
        """Save all trained models."""
        for name, model in self.models.items():
            filename = f"{filepath_prefix}_{name.lower()}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save best model separately
        if self.best_model:
            best_filename = f"{filepath_prefix}_best.joblib"
            joblib.dump(self.best_model, best_filename)
            print(f"Saved best model to {best_filename}")
    
    def load_model(self, filepath):
        """Load a saved model."""
        model = joblib.load(filepath)
        return model
