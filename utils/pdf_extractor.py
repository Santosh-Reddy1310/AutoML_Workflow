import pandas as pd
import PyPDF2
import pdfplumber
import tabula
import google.generativeai as genai
import streamlit as st
import io
import re
from typing import Optional, List, Dict

class PDFDataExtractor:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract raw text from PDF using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_file) -> List[pd.DataFrame]:
        """Extract tables from PDF using tabula-py and pdfplumber"""
        tables = []
        
        try:
            # Try tabula-py first
            df_list = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
            tables.extend(df_list)
        except Exception as e:
            st.warning(f"Tabula extraction failed: {str(e)}")
        
        try:
            # Try pdfplumber as backup
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)
        except Exception as e:
            st.warning(f"PDFPlumber extraction failed: {str(e)}")
        
        return tables
    
    def analyze_with_gemini(self, text: str) -> Dict:
        """Use Gemini AI to analyze text and suggest data structure"""
        prompt = f"""
        Analyze this text and identify if it contains tabular data that can be used for machine learning:
        
        {text[:5000]}  # Limit text to avoid token limits
        
        Please provide:
        1. Whether this contains structured data suitable for ML
        2. Suggested column names if data is present
        3. Data types for each column
        4. Any patterns or relationships you notice
        
        Format your response as JSON with keys: 'has_ml_data', 'columns', 'data_types', 'insights'
        """
        
        try:
            response = self.model.generate_content(prompt)
            return {"analysis": response.text, "success": True}
        except Exception as e:
            return {"analysis": f"Error analyzing with Gemini: {str(e)}", "success": False}
    
    def process_pdf(self, pdf_file) -> Dict:
        """Main method to process PDF and extract data"""
        result = {
            "text": "",
            "tables": [],
            "gemini_analysis": {},
            "processed_data": None
        }
        
        # Extract text
        result["text"] = self.extract_text_from_pdf(pdf_file)
        
        # Extract tables
        result["tables"] = self.extract_tables_from_pdf(pdf_file)
        
        # Analyze with Gemini
        if result["text"]:
            result["gemini_analysis"] = self.analyze_with_gemini(result["text"])
        
        # Process the best table for ML
        if result["tables"]:
            # Select the largest table
            largest_table = max(result["tables"], key=len)
            result["processed_data"] = self.clean_dataframe(largest_table)
        
        return result
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame for ML"""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Remove non-numeric columns with too many unique values
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.5:
                df = df.drop(columns=[col])
        
        return df