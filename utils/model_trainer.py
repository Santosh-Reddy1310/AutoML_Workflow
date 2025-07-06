import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Any
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.is_classification = None
    
    def get_models(self, problem_type: str) -> Dict:
        """Get available models based on problem type"""
        if problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'K-Neighbors': KNeighborsClassifier()
            }
        else:  # regression
            return {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'K-Neighbors': KNeighborsRegressor()
            }
    
    def determine_problem_type(self, y: pd.Series) -> str:
        """Determine if the problem is classification or regression"""
        if y.dtype == 'object' or len(np.unique(y)) < 10:
            return 'classification'
        else:
            return 'regression'
    
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train multiple models and return results"""
        self.is_classification = self.determine_problem_type(y_train)
        models = self.get_models(self.is_classification)
        
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Training {name}...')
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                if self.is_classification == 'classification':
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    
                    results[name] = {
                        'model': model,
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'y_pred_train': y_pred_train,
                        'y_pred_test': y_pred_test,
                        'classification_report': classification_report(y_test, y_pred_test),
                        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
                    }
                else:  # regression
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    results[name] = {
                        'model': model,
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_mae': test_mae,
                        'y_pred_train': y_pred_train,
                        'y_pred_test': y_pred_test
                    }
                
                # Store model
                self.models[name] = model
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text('Training completed!')
        self.results = results
        
        # Find best model
        self.find_best_model()
        
        return results
    
    def find_best_model(self):
        """Find the best performing model"""
        if not self.results:
            return
        
        if self.is_classification == 'classification':
            best_score = 0
            metric = 'test_accuracy'
        else:
            best_score = float('inf')
            metric = 'test_mse'
        
        for name, result in self.results.items():
            if metric in result:
                if self.is_classification == 'classification':
                    if result[metric] > best_score:
                        best_score = result[metric]
                        self.best_model = result['model']
                        self.best_model_name = name
                else:
                    if result[metric] < best_score:
                        best_score = result[metric]
                        self.best_model = result['model']
                        self.best_model_name = name
    
    def create_visualizations(self, y_test: pd.Series) -> Dict:
        """Create visualizations for model results"""
        visualizations = {}
        
        if self.is_classification == 'classification':
            # Model comparison chart
            model_names = list(self.results.keys())
            accuracies = [self.results[name]['test_accuracy'] for name in model_names]
            
            fig = px.bar(x=model_names, y=accuracies, 
                        title='Model Accuracy Comparison',
                        labels={'x': 'Models', 'y': 'Accuracy'})
            fig.update_layout(xaxis_tickangle=-45)
            visualizations['model_comparison'] = fig
            
            # Confusion Matrix for best model
            if self.best_model_name:
                cm = self.results[self.best_model_name]['confusion_matrix']
                fig_cm = px.imshow(cm, 
                                  title=f'Confusion Matrix - {self.best_model_name}',
                                  labels=dict(x="Predicted", y="Actual"),
                                  color_continuous_scale='Blues')
                visualizations['confusion_matrix'] = fig_cm
        
        else:  # regression
            # Model comparison chart
            model_names = list(self.results.keys())
            r2_scores = [self.results[name]['test_r2'] for name in model_names]
            
            fig = px.bar(x=model_names, y=r2_scores, 
                        title='Model R² Score Comparison',
                        labels={'x': 'Models', 'y': 'R² Score'})
            fig.update_layout(xaxis_tickangle=-45)
            visualizations['model_comparison'] = fig
            
            # Prediction vs Actual for best model
            if self.best_model_name:
                y_pred = self.results[self.best_model_name]['y_pred_test']
                fig_pred = px.scatter(x=y_test, y=y_pred, 
                                    title=f'Predictions vs Actual - {self.best_model_name}',
                                    labels={'x': 'Actual', 'y': 'Predicted'})
                fig_pred.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                                 x1=y_test.max(), y1=y_test.max(), 
                                 line=dict(color="red", dash="dash"))
                visualizations['prediction_vs_actual'] = fig_pred
        
        return visualizations
    
    def save_best_model(self, filepath: str):
        """Save the best model to disk"""
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            return True
        return False
    
    def get_model_summary(self) -> Dict:
        """Get summary of all trained models"""
        if not self.results:
            return {}
        
        summary = {}
        for name, result in self.results.items():
            if self.is_classification == 'classification':
                summary[name] = {
                    'Train Accuracy': f"{result['train_accuracy']:.4f}",
                    'Test Accuracy': f"{result['test_accuracy']:.4f}",
                    'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.4f}"
                }
            else:
                summary[name] = {
                    'Train R²': f"{result['train_r2']:.4f}",
                    'Test R²': f"{result['test_r2']:.4f}",
                    'Test MSE': f"{result['test_mse']:.4f}",
                    'Test MAE': f"{result['test_mae']:.4f}"
                }
        
        return summary