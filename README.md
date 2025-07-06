# AutoML Workflow Website 🤖

A comprehensive AutoML platform built with Streamlit that enables users to upload datasets or PDFs, automatically extract data, train multiple machine learning models, and visualize results with minimal effort.

## 🚀 Features

- **Multi-format Data Input**: Support for CSV, Excel, and PDF files
- **AI-Powered PDF Extraction**: Uses Google Gemini 1.5 Flash to intelligently extract tabular data from PDFs
- **Automated Data Processing**: Handles missing values, categorical encoding, and feature scaling
- **Multiple ML Algorithms**: Trains and compares 5-6 different models automatically
- **Smart Model Selection**: Automatically determines classification vs regression tasks
- **Interactive Visualizations**: Beautiful charts and graphs using Plotly
- **Model Performance Metrics**: Comprehensive evaluation with accuracy, confusion matrices, R² scores
- **User-Friendly Interface**: Clean, intuitive web interface with progress indicators

## 📁 Project Structure

```
automl_webapp/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── data_processor.py  # Data processing utilities
│   ├── model_trainer.py   # ML model training and evaluation
│   └── pdf_extractor.py   # PDF data extraction with Gemini AI
├── models/               # Directory for saved models
├── uploads/              # Directory for uploaded files
└── README.md            # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (for PDF extraction)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd automl_webapp
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv automl_env

# Activate virtual environment
# On Windows:
automl_env\Scripts\activate
# On macOS/Linux:
source automl_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create Directory Structure
```bash
mkdir models uploads
```

### Step 5: Get Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Keep it secure - you'll need it to use PDF extraction features

## 🚀 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Platform

#### 1. **Configuration** (Sidebar)
- Enter your Gemini API key for PDF extraction
- Adjust model settings:
  - Test size (percentage for testing)
  - Random state (for reproducibility)

#### 2. **Data Upload Tab**
- **PDF Upload**: Upload PDF files containing tabular data
  - AI will automatically extract and structure the data
  - Works with reports, tables, and structured documents
- **Dataset Upload**: Upload CSV or Excel files directly
  - Supports .csv, .xlsx, .xls formats

#### 3. **Data Exploration Tab**
- View dataset statistics and information
- Explore data types and missing values
- Visualize correlations and distributions
- Get insights before training

#### 4. **Model Training Tab**
- Select target column for prediction
- View target variable distribution
- Start automated training process
- Models trained include:
  - **Classification**: Random Forest, Logistic Regression, SVM, Naive Bayes, Decision Tree, K-Neighbors
  - **Regression**: Random Forest, Linear Regression, SVM, Decision Tree, K-Neighbors

#### 5. **Results Tab**
- View best performing model
- Compare all models side-by-side
- Interactive visualizations:
  - Model performance comparison charts
  - Confusion matrices (classification)
  - Prediction vs Actual plots (regression)
- Download trained models

## 🔧 Technical Details

### Supported File Formats
- **CSV**: Standard comma-separated values
- **Excel**: .xlsx and .xls files
- **PDF**: Documents with tabular data

### Machine Learning Pipeline
1. **Data Preprocessing**:
   - Missing value imputation
   - Categorical variable encoding
   - Feature scaling with StandardScaler
   - Automatic train-test split

2. **Model Training**:
   - Automatic problem type detection (classification/regression)
   - Multiple algorithm comparison
   - Cross-validation and performance evaluation
   - Best model selection based on metrics

3. **Evaluation Metrics**:
   - **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   - **Regression**: R² Score, MSE, MAE, Prediction plots

### AI Integration
- **Gemini 1.5 Flash**: Analyzes PDF content and suggests data structure
- **Intelligent Extraction**: Identifies tabular data patterns
- **Data Quality Assessment**: Provides insights on extracted data

## 📊 Model Performance

The platform automatically trains and compares multiple models:

### Classification Models
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine
- Naive Bayes
- Decision Tree
- K-Neighbors Classifier

### Regression Models
- Random Forest Regressor
- Linear Regression
- Support Vector Regression
- Decision Tree Regressor
- K-Neighbors Regressor

## 🎯 Use Cases

- **Business Analytics**: Extract insights from reports and datasets
- **Academic Research**: Quick model prototyping and comparison
- **Data Science Projects**: Automated baseline model creation
- **PDF Data Mining**: Convert PDF tables to ML-ready datasets
- **Rapid Prototyping**: Fast model development and testing

## 📈 Example Workflow

1. **Upload Data**: Drop a CSV file or PDF report
2. **Explore**: Check data quality and distributions
3. **Train**: Select target variable and start training
4. **Analyze**: Compare models and view performance metrics
5. **Deploy**: Use the best model for predictions

## 🔒 Security & Privacy

- API keys are handled securely (input type="password")
- No data is stored permanently on servers
- All processing happens in your session
- Files are temporarily stored during processing only

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'utils'"**
   - Ensure you're in the correct directory
   - Check that `utils/__init__.py` exists

2. **PDF extraction fails**
   - Verify Gemini API key is correct
   - Check if PDF contains tabular data
   - Some PDFs may have complex layouts

3. **Model training errors**
   - Ensure dataset has enough samples
   - Check that target column is selected
   - Verify data types are compatible

4. **Import errors**
   - Reinstall requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **Google Gemini**: For AI-powered PDF extraction
- **Scikit-learn**: For machine learning algorithms
- **Plotly**: For interactive visualizations
- **Pandas**: For data manipulation

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue in the repository
4. Contact the development team

## 🔄 Updates

- **v1.0.0**: Initial release with basic AutoML functionality
- **v1.1.0**: Added PDF extraction with Gemini AI
- **v1.2.0**: Enhanced visualizations and model comparison

---

**Happy AutoML! 🎉**