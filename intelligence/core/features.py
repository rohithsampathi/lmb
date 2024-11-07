# intelligence/core/features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Advanced feature processing for marketing data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract rich temporal features from time series data"""
        try:
            df = df.copy()
            
            # Ensure datetime index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Time-based features
            df['day'] = df.index.day
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Rolling statistics
            for window in [3, 7, 14, 30]:
                # Leads analysis
                df[f'leads_ma_{window}'] = df['Leads'].rolling(window).mean()
                df[f'leads_std_{window}'] = df['Leads'].rolling(window).std()
                
                # Cost efficiency
                df[f'cost_per_lead_ma_{window}'] = (
                    df['Spend'] / df['Leads'].replace(0, np.nan)
                ).rolling(window).mean()
                
                # Goal conversion
                df[f'goal_rate_ma_{window}'] = (
                    df['Goals'] / df['Leads'].replace(0, np.nan)
                ).rolling(window).mean()
            
            # Periodic features
            for period in [7, 14, 30]:
                for harm in range(3):
                    df[f'sin_{period}_{harm}'] = np.sin(2 * np.pi * harm * df.index.dayofyear / period)
                    df[f'cos_{period}_{harm}'] = np.cos(2 * np.pi * harm * df.index.dayofyear / period)
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return df
