import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.pdf_extractor import PDFDataExtractor
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AutoML Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🤖 AutoML Workflow Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gemini API Key
        gemini_api_key = st.text_input("🔑 Gemini API Key", type="password", 
                                      help="Enter your Google Gemini API key")
        
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key to use PDF extraction features.")
        
        st.markdown("---")
        
        # Model Configuration
        st.subheader("🎯 Model Settings")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", 1, 100, 42)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "🔍 Data Exploration", "🤖 Model Training", "📊 Results"])
    
    with tab1:
        st.markdown('<h2 class="section-header">📁 Data Upload & Extraction</h2>', unsafe_allow_html=True)
        
        # File upload
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Upload PDF")
            pdf_file = st.file_uploader("Choose a PDF file", type=['pdf'])
            
            if pdf_file and gemini_api_key:
                if st.button("🔍 Extract Data from PDF", key="pdf_extract"):
                    with st.spinner("Extracting data from PDF..."):
                        try:
                            extractor = PDFDataExtractor(gemini_api_key)
                            result = extractor.process_pdf(pdf_file)
                            
                            if result['processed_data'] is not None:
                                st.session_state.data = result['processed_data']
                                st.success("✅ Data extracted successfully!")
                                st.dataframe(st.session_state.data.head())
                            else:
                                st.error("❌ No tabular data found in PDF")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {str(e)}")
        
        with col2:
            st.subheader("📊 Upload Dataset")
            data_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
            
            if data_file:
                if st.button("📤 Load Dataset", key="dataset_load"):
                    with st.spinner("Loading dataset..."):
                        try:
                            processor = DataProcessor()
                            data = processor.load_data(data_file)
                            if data is not None:
                                st.session_state.data = data
                                st.success("✅ Dataset loaded successfully!")
                                st.dataframe(st.session_state.data.head())
                        except Exception as e:
                            st.error(f"❌ Error loading dataset: {str(e)}")
    
    with tab2:
        st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            processor = DataProcessor()
            exploration = processor.explore_data(st.session_state.data)
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 Rows", exploration['shape'][0])
            with col2:
                st.metric("📊 Columns", exploration['shape'][1])
            with col3:
                st.metric("🔢 Numeric Columns", len(exploration['numeric_columns']))
            
            # Data preview
            st.subheader("👁️ Data Preview")
            st.dataframe(st.session_state.data.head(10))
            
            # Data types and missing values
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏷️ Data Types")
                dtype_df = pd.DataFrame(list(exploration['dtypes'].items()), 
                                      columns=['Column', 'Type'])
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("❓ Missing Values")
                missing_df = pd.DataFrame(list(exploration['missing_values'].items()), 
                                        columns=['Column', 'Missing Count'])
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                if not missing_df.empty:
                    st.dataframe(missing_df)
                else:
                    st.success("✅ No missing values found!")
            
            # Visualizations
            st.subheader("📈 Data Visualizations")
            
            if exploration['numeric_columns']:
                # Correlation heatmap
                numeric_data = st.session_state.data[exploration['numeric_columns']]
                if len(numeric_data.columns) > 1:
                    fig = px.imshow(numeric_data.corr(), 
                                  title="Correlation Matrix",
                                  color_continuous_scale='RdBu')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first in the Data Upload tab.")
    
    with tab3:
        st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            # Target column selection
            target_column = st.selectbox("🎯 Select Target Column", 
                                       options=st.session_state.data.columns.tolist())
            
            if target_column:
                # Show target distribution with detailed analysis
                st.subheader("🎯 Target Distribution Analysis")
                
                # Create two columns for visualization and summary
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.session_state.data[target_column].dtype == 'object':
                        # Categorical target
                        fig = px.histogram(st.session_state.data, x=target_column, 
                                         title=f"Distribution of {target_column}")
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Numerical target
                        fig = px.histogram(st.session_state.data, x=target_column, 
                                         title=f"Distribution of {target_column}",
                                         nbins=30)
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**📊 Distribution Summary**")
                    
                    # Get distribution statistics
                    if st.session_state.data[target_column].dtype == 'object':
                        # Categorical analysis
                        value_counts = st.session_state.data[target_column].value_counts()
                        total_count = len(st.session_state.data)
                        unique_values = st.session_state.data[target_column].nunique()
                        
                        st.metric("📈 Total Records", total_count)
                        st.metric("🏷️ Unique Classes", unique_values)
                        st.metric("📊 Most Common", value_counts.index[0])
                        st.metric("📉 Least Common", value_counts.index[-1])
                        
                        # Class distribution percentages
                        st.markdown("**Class Distribution:**")
                        for class_name, count in value_counts.head(5).items():
                            percentage = (count / total_count) * 100
                            st.write(f"• {class_name}: {count} ({percentage:.1f}%)")
                        
                        # Balance analysis
                        max_class_pct = (value_counts.max() / total_count) * 100
                        min_class_pct = (value_counts.min() / total_count) * 100
                        
                        st.markdown("**⚖️ Balance Analysis:**")
                        if max_class_pct > 70:
                            st.warning("⚠️ **Imbalanced Dataset**")
                            st.write("The dataset is highly imbalanced. Consider using techniques like SMOTE or class weighting.")
                        elif max_class_pct > 60:
                            st.info("ℹ️ **Moderately Imbalanced**")
                            st.write("The dataset shows some imbalance. Monitor model performance carefully.")
                        else:
                            st.success("✅ **Well Balanced**")
                            st.write("The dataset is well balanced across classes.")
                    
                    else:
                        # Numerical analysis
                        stats = st.session_state.data[target_column].describe()
                        
                        st.metric("📊 Mean", f"{stats['mean']:.2f}")
                        st.metric("📏 Std Dev", f"{stats['std']:.2f}")
                        st.metric("📈 Min", f"{stats['min']:.2f}")
                        st.metric("📉 Max", f"{stats['max']:.2f}")
                        
                        # Distribution shape analysis
                        from scipy import stats as scipy_stats
                        skewness = scipy_stats.skew(st.session_state.data[target_column].dropna())
                        kurtosis = scipy_stats.kurtosis(st.session_state.data[target_column].dropna())
                        
                        st.markdown("**📐 Distribution Shape:**")
                        st.write(f"• Skewness: {skewness:.2f}")
                        st.write(f"• Kurtosis: {kurtosis:.2f}")
                        
                        # Interpretation
                        st.markdown("**📝 Interpretation:**")
                        if abs(skewness) < 0.5:
                            st.success("✅ **Normally Distributed**")
                            st.write("Data is approximately normal.")
                        elif abs(skewness) < 1:
                            st.info("ℹ️ **Moderately Skewed**")
                            st.write("Data shows moderate skewness.")
                        else:
                            st.warning("⚠️ **Highly Skewed**")
                            st.write("Data is highly skewed. Consider transformation.")
                        
                        # Outlier detection
                        Q1 = stats['25%']
                        Q3 = stats['75%']
                        IQR = Q3 - Q1
                        outliers = st.session_state.data[
                            (st.session_state.data[target_column] < (Q1 - 1.5 * IQR)) | 
                            (st.session_state.data[target_column] > (Q3 + 1.5 * IQR))
                        ]
                        
                        st.markdown("**🎯 Outlier Analysis:**")
                        if len(outliers) > 0:
                            outlier_pct = (len(outliers) / len(st.session_state.data)) * 100
                            st.write(f"• Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
                            if outlier_pct > 10:
                                st.warning("⚠️ High outlier percentage detected")
                        else:
                            st.success("✅ No outliers detected")
                
                # Problem type prediction
                st.subheader("🔍 Problem Type Analysis")
                if st.session_state.data[target_column].dtype == 'object':
                    unique_values = st.session_state.data[target_column].nunique()
                    if unique_values == 2:
                        st.info("🎯 **Binary Classification Problem**")
                        st.write("This appears to be a binary classification problem with 2 classes.")
                    else:
                        st.info("🎯 **Multi-class Classification Problem**")
                        st.write(f"This appears to be a multi-class classification problem with {unique_values} classes.")
                else:
                    st.info("🎯 **Regression Problem**")
                    st.write("This appears to be a regression problem with continuous target values.")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                recommendations = []
                
                if st.session_state.data[target_column].dtype == 'object':
                    value_counts = st.session_state.data[target_column].value_counts()
                    if (value_counts.max() / len(st.session_state.data)) > 0.7:
                        recommendations.append("⚖️ Consider using class weighting or resampling techniques for imbalanced data")
                    if st.session_state.data[target_column].nunique() > 10:
                        recommendations.append("🔢 High number of classes detected - consider grouping rare classes")
                else:
                    if abs(skewness) > 1:
                        recommendations.append("📊 Consider log transformation or other normalization techniques")
                    if len(outliers) > len(st.session_state.data) * 0.1:
                        recommendations.append("🎯 Consider outlier treatment before training")
                
                if len(st.session_state.data) < 100:
                    recommendations.append("📈 Small dataset detected - results may vary significantly")
                elif len(st.session_state.data) < 1000:
                    recommendations.append("📊 Medium dataset - consider cross-validation for better estimates")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(f"• {rec}")
                else:
                    st.success("✅ Your data looks good for machine learning!")
                
                # Training button
                if st.button("🚀 Start Training", key="start_training"):
                    with st.spinner("Training models..."):
                        try:
                            # Initialize components
                            processor = DataProcessor()
                            trainer = ModelTrainer()
                            
                            # Preprocess data
                            X, y = processor.preprocess_data(st.session_state.data, target_column)
                            
                            # Split data
                            X_train, X_test, y_train, y_test = processor.split_data(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            # Scale features
                            X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
                            
                            # Train models
                            results = trainer.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
                            
                            # Store results
                            st.session_state.results = results
                            st.session_state.trainer = trainer
                            st.session_state.y_test = y_test
                            st.session_state.models_trained = True
                            
                            st.success("✅ Training completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
        else:
            st.warning("⚠️ Please upload data first in the Data Upload tab.")
    
    with tab4:
        st.markdown('<h2 class="section-header">📊 Results & Metrics</h2>', unsafe_allow_html=True)
        
        if st.session_state.models_trained and st.session_state.results:
            trainer = st.session_state.trainer
            
            # Best model info
            st.subheader("🏆 Best Model")
            if trainer.best_model_name:
                st.success(f"🎉 Best Model: **{trainer.best_model_name}**")
            
            # Model comparison
            st.subheader("📊 Model Performance Comparison")
            summary = trainer.get_model_summary()
            if summary:
                summary_df = pd.DataFrame(summary).T
                st.dataframe(summary_df, use_container_width=True)
            
            # Visualizations
            st.subheader("📈 Performance Visualizations")
            visualizations = trainer.create_visualizations(st.session_state.y_test)
            
            # Display visualizations
            if 'model_comparison' in visualizations:
                st.plotly_chart(visualizations['model_comparison'], use_container_width=True)
            
            if 'confusion_matrix' in visualizations:
                st.plotly_chart(visualizations['confusion_matrix'], use_container_width=True)
            
            if 'prediction_vs_actual' in visualizations:
                st.plotly_chart(visualizations['prediction_vs_actual'], use_container_width=True)
            
            # Detailed results
            st.subheader("📋 Detailed Results")
            selected_model = st.selectbox("Select Model for Details", 
                                        options=list(st.session_state.results.keys()))
            
            if selected_model:
                result = st.session_state.results[selected_model]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Metrics**")
                    if trainer.is_classification == 'classification':
                        st.metric("Train Accuracy", f"{result['train_accuracy']:.4f}")
                        st.metric("Test Accuracy", f"{result['test_accuracy']:.4f}")
                        st.metric("Overfitting", f"{result['train_accuracy'] - result['test_accuracy']:.4f}")
                    else:
                        st.metric("Train R²", f"{result['train_r2']:.4f}")
                        st.metric("Test R²", f"{result['test_r2']:.4f}")
                        st.metric("Test MSE", f"{result['test_mse']:.4f}")
                        st.metric("Test MAE", f"{result['test_mae']:.4f}")
                
                with col2:
                    if trainer.is_classification == 'classification':
                        st.markdown("**📈 Classification Report**")
                        st.text(result['classification_report'])
            
            # Model download
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Download Best Model**")
                if trainer.best_model:
                    if st.button("📥 Download Best Model", key="download_model"):
                        try:
                            # Create models directory if it doesn't exist
                            os.makedirs('models', exist_ok=True)
                            
                            # Save model
                            model_path = f"models/best_model_{trainer.best_model_name.replace(' ', '_')}.joblib"
                            success = trainer.save_best_model(model_path)
                            
                            if success:
                                st.success(f"✅ Model saved as {model_path}")
                                
                                # Display model info
                                st.info(f"""
                                **Model Details:**
                                - Algorithm: {trainer.best_model_name}
                                - Performance: {trainer.results[trainer.best_model_name].get('test_accuracy', trainer.results[trainer.best_model_name].get('test_r2', 'N/A')):.4f}
                                - File: {model_path}
                                """)
                            else:
                                st.error("❌ Failed to save model")
                        except Exception as e:
                            st.error(f"❌ Error saving model: {str(e)}")
            
            with col2:
                st.markdown("**📊 Download Test Results**")
                if st.button("📥 Download Test Results", key="download_results"):
                    try:
                        # Prepare comprehensive results
                        results_data = []
                        
                        # Get actual vs predicted for all models
                        for model_name, result in st.session_state.results.items():
                            y_pred = result['y_pred_test']
                            
                            # Create a dataframe with actual vs predicted
                            for i, (actual, predicted) in enumerate(zip(st.session_state.y_test, y_pred)):
                                results_data.append({
                                    'Test_Index': i,
                                    'Actual_Value': actual,
                                    f'Predicted_{model_name.replace(" ", "_")}': predicted,
                                    'Model': model_name
                                })
                        
                        # Convert to DataFrame
                        results_df = pd.DataFrame(results_data)
                        
                        # Pivot to get all predictions in one row per test sample
                        pivot_df = results_df.pivot_table(
                            index=['Test_Index', 'Actual_Value'], 
                            columns='Model', 
                            values=[col for col in results_df.columns if col.startswith('Predicted_')],
                            aggfunc='first'
                        ).reset_index()
                        
                        # Flatten column names
                        pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns.values]
                        
                        # Add performance metrics summary
                        summary_data = []
                        for model_name, result in st.session_state.results.items():
                            if trainer.is_classification == 'classification':
                                summary_data.append({
                                    'Model': model_name,
                                    'Train_Accuracy': result['train_accuracy'],
                                    'Test_Accuracy': result['test_accuracy'],
                                    'Overfitting_Score': result['train_accuracy'] - result['test_accuracy']
                                })
                            else:
                                summary_data.append({
                                    'Model': model_name,
                                    'Train_R2': result['train_r2'],
                                    'Test_R2': result['test_r2'],
                                    'Test_MSE': result['test_mse'],
                                    'Test_MAE': result['test_mae']
                                })
                        
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Create Excel file with multiple sheets
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            # Test results sheet
                            pivot_df.to_excel(writer, sheet_name='Test_Predictions', index=False)
                            
                            # Performance summary sheet
                            summary_df.to_excel(writer, sheet_name='Model_Performance', index=False)
                            
                            # Best model details
                            best_model_info = pd.DataFrame([{
                                'Best_Model': trainer.best_model_name,
                                'Problem_Type': trainer.is_classification,
                                'Test_Size': test_size,
                                'Random_State': random_state,
                                'Total_Samples': len(st.session_state.data),
                                'Features_Used': len(st.session_state.data.columns) - 1
                            }])
                            best_model_info.to_excel(writer, sheet_name='Experiment_Info', index=False)
                        
                        # Prepare download
                        processed_data = output.getvalue()
                        
                        # Create download button
                        st.download_button(
                            label="📥 Download Complete Results (Excel)",
                            data=processed_data,
                            file_name=f"automl_results_{trainer.best_model_name.replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Results file prepared for download!")
                        
                        # Show what's included
                        st.info("""
                        **📋 Download includes:**
                        - 📊 Test Predictions: Actual vs Predicted for all models
                        - 📈 Model Performance: Detailed metrics comparison
                        - ⚙️ Experiment Info: Configuration and dataset details
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ Error preparing results: {str(e)}")
                        
                # Additional download option for CSV
                if st.button("📥 Download Results (CSV)", key="download_csv"):
                    try:
                        # Simple CSV with best model results
                        csv_data = []
                        best_result = st.session_state.results[trainer.best_model_name]
                        
                        for i, (actual, predicted) in enumerate(zip(st.session_state.y_test, best_result['y_pred_test'])):
                            csv_data.append({
                                'Test_Index': i,
                                'Actual_Value': actual,
                                'Predicted_Value': predicted,
                                'Model_Used': trainer.best_model_name
                            })
                        
                        csv_df = pd.DataFrame(csv_data)
                        csv_string = csv_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Best Model Results (CSV)",
                            data=csv_string,
                            file_name=f"best_model_results_{trainer.best_model_name.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error preparing CSV: {str(e)}")
            
            # Feature importance (if available)
            if (trainer.best_model and 
                hasattr(trainer.best_model, 'feature_importances_') and 
                'trainer' in st.session_state):
                
                st.subheader("🎯 Feature Importance")
                try:
                    # Get feature names from processor
                    processor = DataProcessor()
                    if hasattr(processor, 'feature_columns'):
                        feature_names = processor.feature_columns
                    else:
                        feature_names = [f"Feature_{i}" for i in range(len(trainer.best_model.feature_importances_))]
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': trainer.best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df.head(10), 
                               x='Importance', y='Feature', 
                               orientation='h',
                               title='Top 10 Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
        else:
            st.warning("⚠️ Please train models first in the Model Training tab.")

if __name__ == "__main__":
    main()