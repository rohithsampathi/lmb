# intelligence/core/optimizer.py

class CombinationOptimizer:
    """Optimize marketing combinations using advanced algorithms"""
    
    def __init__(self, feature_processor: FeatureProcessor):
        self.feature_processor = feature_processor
        self.performance_cache = {}
        
    def evaluate_combination(self, 
                           data: pd.DataFrame,
                           location: str,
                           primary: str,
                           secondary: str) -> Dict:
        """Evaluate performance of a specific combination"""
        try:
            combo_key = f"{location}_{primary}_{secondary}"
            
            if combo_key in self.performance_cache:
                return self.performance_cache[combo_key]
            
            filtered_data = data[
                (data['Locations'] == location) &
                (data['Primary'] == primary) &
                (data['Secondary'] == secondary)
            ]
            
            if len(filtered_data) == 0:
                return {
                    'score': 0,
                    'confidence': 0,
                    'metrics': {},
                    'insights': {}
                }
            
            # Process features
            processed_data = self.feature_processor.extract_temporal_features(filtered_data)
            
            # Calculate base metrics
            strength_score = PerformanceMetrics.calculate_strength_score(processed_data)
            
            # Calculate trend and seasonality
            trend = processed_data['Leads'].pct_change().mean()
            seasonality = self._detect_seasonality(processed_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(processed_data)
            
            result = {
                'score': strength_score,
                'confidence': confidence,
                'metrics': {
                    'avg_leads': processed_data['Leads'].mean(),
                    'avg_cost': processed_data['Spend'].mean(),
                    'conversion_rate': (processed_data['Goals'] / processed_data['Leads']).mean(),
                    'trend': trend
                },
                'insights': {
                    'seasonality': seasonality,
                    'best_days': self._find_best_days(processed_data),
                    'optimal_spend': self._calculate_optimal_spend(processed_data)
                }
            }
            
            self.performance_cache[combo_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating combination: {str(e)}")
            return {
                'score': 0,
                'confidence': 0,
                'metrics': {},
                'insights': {}
            }
    
    def _detect_seasonality(self, data: pd.DataFrame) -> Dict:
        """Detect seasonal patterns in the data"""
        patterns = {}
        for period in [7, 14, 30]:
            acf = pd.Series(data['Leads']).autocorr(lag=period)
            patterns[f'{period}_day'] = abs(acf) > 0.3
        return patterns
    
    def _find_best_days(self, data: pd.DataFrame) -> List[int]:
        """Find best performing days of the week"""
        daily_avg = data.groupby('day_of_week')['Leads'].mean()
        return daily_avg.nlargest(3).index.tolist()
    
    def _calculate_optimal_spend(self, data: pd.DataFrame) -> float:
        """Calculate optimal spend based on ROI analysis"""
        spend_lead_ratio = data.groupby('Spend')['Leads'].mean()
        return spend_lead_ratio.idxmax() if not spend_lead_ratio.empty else 0