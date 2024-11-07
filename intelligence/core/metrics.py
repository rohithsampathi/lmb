# intelligence/core/metrics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Calculate sophisticated performance metrics"""
    
    @staticmethod
    def calculate_strength_score(data: pd.DataFrame) -> float:
        """Calculate comprehensive strength score"""
        try:
            metrics = {}
            
            # Lead Generation Effectiveness
            metrics['lead_volume'] = np.clip(data['Leads'].mean() / data['Leads'].max(), 0, 1)
            
            # Cost Efficiency
            cost_per_lead = data['Spend'] / data['Leads'].replace(0, np.nan)
            metrics['cost_efficiency'] = 1 / (1 + cost_per_lead.mean() / cost_per_lead.min())
            
            # Goal Conversion
            metrics['goal_rate'] = (data['Goals'] / data['Leads'].replace(0, np.nan)).mean()
            
            # Consistency
            metrics['consistency'] = 1 / (1 + data['Leads'].std() / data['Leads'].mean())
            
            # Growth Trajectory
            metrics['growth'] = data['Leads'].pct_change().mean() + 1
            
            # Weighted combination
            weights = {
                'lead_volume': 0.25,
                'cost_efficiency': 0.25,
                'goal_rate': 0.2,
                'consistency': 0.15,
                'growth': 0.15
            }
            
            strength_score = sum(metrics[k] * v for k, v in weights.items()) * 100
            return np.clip(strength_score, 0, 100)
            
        except Exception as e:
            logger.error(f"Error calculating strength score: {str(e)}")
            return 0.0