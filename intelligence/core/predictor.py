# intelligence/core/predictor.py

class MarketingPredictor:
    """Advanced prediction system using ensemble of models"""
    
    def __init__(self, feature_processor: FeatureProcessor):
        self.feature_processor = feature_processor
        self.models = {
            'time_series': TimeSeriesEncoder(input_dim=64),
            'trend': TrendAnalyzer(),
            'pattern': PatternMatcher()
        }
        self.prediction_history = []
        self.model_weights = {
            'time_series': 0.5,
            'trend': 0.3,
            'pattern': 0.2
        }
    
    def predict_performance(self, 
                          historical_data: pd.DataFrame,
                          forecast_days: int = 7,
                          confidence_interval: bool = True) -> Dict:
        """Generate performance predictions with confidence intervals"""
        try:
            # Process features
            processed_data = self.feature_processor.extract_temporal_features(historical_data)
            
            predictions = {
                'leads': [],
                'spend': [],
                'goals': [],
                'confidence_intervals': [],
                'contributing_factors': []
            }
            
            # Generate predictions from each model
            model_predictions = {}
            for model_name, model in self.models.items():
                model_pred = model.predict(processed_data, forecast_days)
                model_predictions[model_name] = model_pred
            
            # Ensemble predictions with dynamic weights
            for day in range(forecast_days):
                day_pred = {
                    'leads': 0,
                    'spend': 0,
                    'goals': 0,
                    'factors': []
                }
                
                # Combine predictions using weights
                for model_name, preds in model_predictions.items():
                    weight = self.model_weights[model_name]
                    day_pred['leads'] += preds['leads'][day] * weight
                    day_pred['spend'] += preds['spend'][day] * weight
                    day_pred['goals'] += preds['goals'][day] * weight
                    
                    # Collect contributing factors
                    if preds.get('factors'):
                        day_pred['factors'].extend(preds['factors'][day])
                
                predictions['leads'].append(day_pred['leads'])
                predictions['spend'].append(day_pred['spend'])
                predictions['goals'].append(day_pred['goals'])
                predictions['contributing_factors'].append(day_pred['factors'])
                
                if confidence_interval:
                    ci = self._calculate_confidence_intervals(
                        day_pred,
                        [p['variance'][day] for p in model_predictions.values()]
                    )
                    predictions['confidence_intervals'].append(ci)
            
            # Update model weights based on performance
            self._update_model_weights(model_predictions, historical_data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {}
    
    def _calculate_confidence_intervals(self, 
                                     prediction: Dict,
                                     variances: List[float],
                                     confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals for predictions"""
        try:
            from scipy import stats
            
            intervals = {}
            for metric in ['leads', 'spend', 'goals']:
                mean = prediction[metric]
                pooled_variance = np.mean(variances)
                
                # Calculate confidence interval
                ci = stats.norm.interval(
                    confidence,
                    loc=mean,
                    scale=np.sqrt(pooled_variance)
                )
                
                intervals[metric] = {
                    'lower': max(0, ci[0]),
                    'upper': ci[1]
                }
            
            return intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return {}
    
    def _update_model_weights(self, 
                            predictions: Dict[str, Dict],
                            actual_data: pd.DataFrame):
        """Update model weights based on prediction accuracy"""
        try:
            from sklearn.metrics import mean_squared_error
            
            errors = {}
            for model_name, pred in predictions.items():
                # Calculate error for each metric
                metric_errors = []
                for metric in ['leads', 'spend', 'goals']:
                    actual = actual_data[metric].values[-len(pred[metric]):]
                    predicted = np.array(pred[metric])
                    mse = mean_squared_error(actual, predicted)
                    metric_errors.append(mse)
                
                # Average error across metrics
                errors[model_name] = np.mean(metric_errors)
            
            # Convert errors to weights (lower error = higher weight)
            total_error = sum(1/e for e in errors.values())
            for model_name, error in errors.items():
                self.model_weights[model_name] = (1/error) / total_error
            
        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")