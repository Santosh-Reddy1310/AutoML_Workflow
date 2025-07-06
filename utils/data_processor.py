import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from typing import Tuple, Dict, List

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
    
    def load_data(self, file) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def explore_data(self, df: pd.DataFrame) -> Dict:
        """Generate data exploration insights"""
        exploration = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'summary_stats': df.describe().to_dict()
        }
        return exploration
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for machine learning"""
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].dtype in ['object']:
                df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        
        # Encode categorical variables
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and col != target_column:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Handle target variable encoding if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
        
        # Store column information
        self.feature_columns = X.columns.tolist()
        self.target_column = target_column
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None)
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)