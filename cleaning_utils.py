# cleaning_utils.py

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple, Dict
import chardet
import logging
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Enhanced CSV processor with robust error handling"""
    
    ENCODINGS = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1', 'utf-16', 'utf-32']
    NUMERIC_COLS = ['Leads', 'Spend', 'Goals', 'Cost Per Lead', 'Impressions']
    TEXT_COLS = ['Partner', 'Locations', 'Primary', 'Secondary', 'Ad']
    
    @classmethod
    def detect_encoding(cls, file_path: Path) -> str:
        """Detect file encoding using chardet and verification"""
        try:
            # Read raw bytes
            with open(file_path, 'rb') as file:
                raw_data = file.read()
            
            # Use chardet for initial detection
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            
            # Verify encoding works
            try:
                with open(file_path, encoding=encoding) as f:
                    f.read()
                return encoding
            except UnicodeDecodeError:
                pass
            
            # Try common encodings
            for enc in cls.ENCODINGS:
                try:
                    with open(file_path, encoding=enc) as f:
                        f.read()
                    return enc
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("No suitable encoding found")
        except Exception as e:
            logger.error(f"Encoding detection error: {str(e)}")
            return 'utf-8'  # Default fallback

    @classmethod
    def clean_numeric_value(cls, value: any) -> float:
        """Clean and convert numeric values safely"""
        try:
            if pd.isna(value):
                return 0.0
            if isinstance(value, (int, float)):
                return float(np.clip(value, -1e30, 1e30))
            
            # Clean string value
            clean_value = str(value).strip()
            for char in ['₹', '$', ',', '_', '%']:
                clean_value = clean_value.replace(char, '')
            
            if clean_value.lower() in ['nan', 'none', '', '#div/0!', '#value!']:
                return 0.0
                
            return float(np.clip(float(clean_value), -1e30, 1e30))
        except:
            return 0.0

    @classmethod
    def read_csv_safely(cls, file_path: Path) -> pd.DataFrame:
        """Read CSV with robust encoding detection and error handling"""
        encoding = cls.detect_encoding(file_path)
        
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.title()
            
            # Convert date with proper error handling
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
            
            # Clean numeric columns
            for col in cls.NUMERIC_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(cls.clean_numeric_value)
            
            # Clean text columns
            for col in cls.TEXT_COLS:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(['nan', 'None', 'NaN'], '')
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            raise

    @classmethod
    def process_and_split_data(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Process and split data into general and attributed dataframes"""
        try:
            # Validate required columns
            required_cols = ['Date', 'Partner', 'Leads', 'Spend', 'Goals']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Clean general dataframe
            general_df = df.copy()
            
            # Process attributed data
            attributed_data = {}
            for col in ['Locations', 'Primary', 'Secondary']:
                if col in df.columns:
                    # Split and process
                    split_df = (df[['Date', 'Partner', col, 'Leads', 'Spend', 'Goals']]
                              .assign(**{col: lambda x: x[col].str.split(',')})
                              .explode(col))
                    
                    # Clean split data
                    split_df[col] = split_df[col].str.strip()
                    split_df = split_df[split_df[col].notna() & (split_df[col] != '')]
                    
                    # Clean numeric values
                    for num_col in ['Leads', 'Spend', 'Goals']:
                        split_df[num_col] = split_df[num_col].apply(cls.clean_numeric_value)
                    
                    attributed_data[col] = split_df
            
            return general_df, attributed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    @classmethod
    def save_data(cls, general_df: pd.DataFrame, attributed_data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files"""
        try:
            # Save general data
            general_df.to_csv(Config.GENERAL_DATA_FILE, index=False, encoding='utf-8')
            
            # Save attributed data
            for col, df in attributed_data.items():
                file_path = Config.ATTRIBUTED_DATA_DIR / f"attributed_{col.lower()}.csv"
                df.to_csv(file_path, index=False, encoding='utf-8')
                
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    @classmethod
    def process_csv_file(cls, file_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Main method to process CSV file"""
        try:
            # Use sample data if no file provided
            if file_path is None or not file_path.exists():
                logger.info("Using sample data")
                df = pd.DataFrame(Config.SAMPLE_DATA)
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                logger.info(f"Processing file: {file_path}")
                df = cls.read_csv_safely(file_path)
            
            # Process and split data
            general_df, attributed_data = cls.process_and_split_data(df)
            
            # Save processed data
            cls.save_data(general_df, attributed_data)
            
            return general_df, attributed_data
            
        except Exception as e:
            logger.error(f"Error in CSV processing: {str(e)}")
            raise