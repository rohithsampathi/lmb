from datetime import datetime
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
import logging
from config import Config

logger = logging.getLogger(__name__)

class DataCache:
    """Cache manager for DataFrame"""
    _general_data: Optional[pd.DataFrame] = None
    _attributed_data: Optional[Dict[str, pd.DataFrame]] = None
    _last_loaded: Optional[datetime] = None
    
    @staticmethod
    def _clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric values to ensure JSON compatibility"""
        df = df.copy()
        numeric_cols = ['Leads', 'Spend', 'Goals']
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, replacing errors with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Clip values to safe range
                df[col] = df[col].clip(-1e30, 1e30)
        return df

    @classmethod
    def clear_cache(cls):
        """Clear the cached data"""
        cls._general_data = None
        cls._attributed_data = None
        cls._last_loaded = None
        logger.info("Cache cleared")

    @classmethod
    def should_reload(cls) -> bool:
        """Check if data should be reloaded"""
        return (cls._last_loaded is None or 
                (datetime.now() - cls._last_loaded).seconds > Config.CACHE_EXPIRY)

    @classmethod
    def clean_and_save_data(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Clean and save data"""
        try:
            # Clean numeric values
            df = cls._clean_numeric_values(df)
            
            # Save general data
            df.to_csv(Config.GENERAL_DATA_FILE, index=False)
            
            # Process attributed data
            attributed_data = {}
            for col in ['Locations', 'Primary', 'Secondary']:
                if col in df.columns:
                    # Split the data
                    split_data = (df[['Date', 'Partner', col, 'Leads', 'Spend', 'Goals']]
                                .assign(**{col: lambda x: x[col].str.split(',')})
                                .explode(col))
                    
                    # Clean split data
                    split_data = cls._clean_numeric_values(split_data)
                    split_data[col] = split_data[col].str.strip()
                    split_data = split_data[split_data[col].notna() & (split_data[col] != '')]
                    
                    # Save attributed data
                    file_path = Config.ATTRIBUTED_DATA_DIR / f"attributed_{col.lower()}.csv"
                    split_data.to_csv(file_path, index=False)
                    attributed_data[col] = split_data

            cls._general_data = df
            cls._attributed_data = attributed_data
            cls._last_loaded = datetime.now()
            
            return df, attributed_data

        except Exception as e:
            logger.error(f"Error cleaning and saving data: {str(e)}")
            raise

    @classmethod
    def get_data(cls) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Get data from cache or load from file"""
        if cls.should_reload():
            try:
                if not Config.GENERAL_DATA_FILE.exists():
                    # Initialize with sample data
                    sample_df = pd.DataFrame(Config.SAMPLE_DATA)
                    sample_df['Date'] = pd.to_datetime(sample_df['Date'])
                    return cls.clean_and_save_data(sample_df)

                # Load and clean general data
                cls._general_data = pd.read_csv(Config.GENERAL_DATA_FILE)
                cls._general_data = cls._clean_numeric_values(cls._general_data)
                cls._general_data['Date'] = pd.to_datetime(cls._general_data['Date'])

                # Load and clean attributed data
                cls._attributed_data = {}
                for col in ['Locations', 'Primary', 'Secondary']:
                    file_path = Config.ATTRIBUTED_DATA_DIR / f"attributed_{col.lower()}.csv"
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        df = cls._clean_numeric_values(df)
                        df['Date'] = pd.to_datetime(df['Date'])
                        cls._attributed_data[col] = df

                cls._last_loaded = datetime.now()

            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                raise

        return cls._general_data, cls._attributed_data