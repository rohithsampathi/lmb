# intelligence/main.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from .core.features import FeatureProcessor
from .core.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class MarketingIntelligence:
    """Main interface for marketing intelligence"""
    
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.update_frequency = timedelta(hours=1)
        self.last_update = None
        
    def analyze_performance(self,
                          data: pd.DataFrame,
                          entity_type: Optional[str] = None,
                          entity_value: Optional[str] = None) -> Dict:
        """Analyze performance metrics and generate insights"""
        try:
            if self._should_update():
                self._update_models(data)
                
            # Process and analyze data
            processed_data = self.feature_processor.extract_temporal_features(data)
            
            # Filter for specific entity if provided
            if entity_type and entity_value:
                processed_data = processed_data[processed_data[entity_type] == entity_value]
                
            # Calculate metrics
            strength_score = PerformanceMetrics.calculate_strength_score(processed_data)
            
            return {
                'current_performance': {
                    'strength_score': strength_score,
                    'metrics': self._calculate_basic_metrics(processed_data),
                    'trends': self._calculate_trends(processed_data)
                },
                'predictions': self._generate_predictions(processed_data),
                'insights': self._extract_insights(processed_data),
                'anomalies': self._detect_anomalies(processed_data)
            }
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return {}

    def _should_update(self) -> bool:
        """Check if models should be updated"""
        if not self.last_update:
            return True
        return datetime.now() - self.last_update >= self.update_frequency
    
    def _update_models(self, data: pd.DataFrame):
        """Update models with new data"""
        try:
            # Future model updates will go here
            self.last_update = datetime.now()
            logger.info("Models updated successfully")
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate basic performance metrics"""
        try:
            return {
                'total_leads': data['Leads'].sum(),
                'total_spend': data['Spend'].sum(),
                'total_goals': data['Goals'].sum(),
                'average_cost_per_lead': data['Spend'].sum() / data['Leads'].sum() if data['Leads'].sum() > 0 else 0,
                'conversion_rate': data['Goals'].sum() / data['Leads'].sum() * 100 if data['Leads'].sum() > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}
    
    def _calculate_trends(self, data: pd.DataFrame) -> Dict:
        """Calculate performance trends"""
        try:
            return {
                'lead_trend': data['Leads'].pct_change().mean() * 100,
                'cost_trend': data['Spend'].pct_change().mean() * 100,
                'goal_trend': data['Goals'].pct_change().mean() * 100
            }
        except Exception as e:
            logger.error(f"Error calculating trends: {str(e)}")
            return {}
    
    def _generate_predictions(self, data: pd.DataFrame) -> Dict:
        """Generate performance predictions"""
        try:
            # Simplified predictions for now
            recent_avg = data['Leads'].tail(7).mean()
            return {
                'leads': [recent_avg] * 7,  # 7-day forecast
                'confidence_interval': {
                    'lower': recent_avg * 0.8,
                    'upper': recent_avg * 1.2
                }
            }
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {}
    
    def _extract_insights(self, data: pd.DataFrame) -> List[str]:
        """Extract key insights from the data"""
        try:
            insights = []
            
            # Lead generation insights
            lead_trend = data['Leads'].pct_change().mean() * 100
            if abs(lead_trend) > 5:
                trend_direction = "increasing" if lead_trend > 0 else "decreasing"
                insights.append(
                    f"Lead generation is {trend_direction} by {abs(lead_trend):.1f}% on average"
                )
            
            # Cost efficiency insights
            recent_cpl = data['Spend'].sum() / data['Leads'].sum() if data['Leads'].sum() > 0 else 0
            if recent_cpl > 0:
                insights.append(f"Current cost per lead is ₹{recent_cpl:.2f}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return []
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in performance metrics"""
        try:
            anomalies = []
            
            # Simple threshold-based anomaly detection
            mean = data['Leads'].mean()
            std = data['Leads'].std()
            threshold = 2
            
            for date, value in data['Leads'].items():
                if abs(value - mean) > threshold * std:
                    anomalies.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'type': 'Lead Volume Anomaly',
                        'description': f"Unusual lead volume: {value:.0f} (Expected: {mean:.0f}±{std:.0f})"
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []