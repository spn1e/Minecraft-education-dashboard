# src/analysis/time_series.py (Fixed version)
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import ruptures as rpt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesEducationAnalyzer:
    """Advanced time series analysis for educational data"""
    
    def __init__(self):
        self.decomposition = None
        self.forecasts = {}
        
    def analyze_engagement_patterns(self, data: pd.DataFrame, 
                                  date_col: str, value_col: str,
                                  student_id: Optional[str] = None) -> Dict:
        """Comprehensive time series analysis of engagement patterns"""
        
        # Prepare time series
        if student_id:
            data = data[data['student_id'] == student_id]
        
        ts_data = data.set_index(pd.to_datetime(data[date_col]))[value_col]
        # Fixed: Use ffill() instead of fillna(method='ffill')
        ts_data = ts_data.resample('D').mean().ffill()
        
        results = {
            'summary_statistics': self._calculate_summary_stats(ts_data),
            'stationarity': self._test_stationarity(ts_data),
            'seasonality': self._analyze_seasonality(ts_data),
            'trend': self._analyze_trend(ts_data),
            'forecast': self._generate_forecast(ts_data),
            'anomalies': self._detect_anomalies(ts_data)
        }
        
        return results
    
    def detect_learning_changepoints(self, student_data: pd.DataFrame,
                                   metric_col: str,
                                   min_size: int = 5) -> Dict:
        """Advanced changepoint detection for learning phases"""
        
        # Prepare data
        values = student_data[metric_col].values
        
        # Multiple changepoint detection methods
        results = {}
        
        # PELT (Pruned Exact Linear Time)
        try:
            algo_pelt = rpt.Pelt(model="rbf", min_size=min_size).fit(values)
            pelt_points = algo_pelt.predict(pen=10)
            results['pelt'] = {
                'changepoints': pelt_points[:-1],
                'n_segments': len(pelt_points),
                'method': 'PELT with RBF kernel'
            }
        except Exception:
            results['pelt'] = {'error': 'PELT detection failed'}
        
        # Binary Segmentation
        try:
            algo_binseg = rpt.Binseg(model="l2", min_size=min_size).fit(values)
            binseg_points = algo_binseg.predict(n_bkps=3)
            results['binseg'] = {
                'changepoints': binseg_points[:-1],
                'n_segments': len(binseg_points),
                'method': 'Binary Segmentation'
            }
        except Exception:
            results['binseg'] = {'error': 'Binary segmentation failed'}
        
        # Window-based detection
        try:
            algo_window = rpt.Window(width=10, model="l2").fit(values)
            window_points = algo_window.predict(n_bkps=3)
            results['window'] = {
                'changepoints': window_points[:-1],
                'n_segments': len(window_points),
                'method': 'Window-based'
            }
        except Exception:
            results['window'] = {'error': 'Window detection failed'}
        
        # Analyze segments
        if 'changepoints' in results.get('pelt', {}):
            segments = self._analyze_segments(values, results['pelt']['changepoints'])
            results['segments'] = segments
            results['learning_phases'] = self._identify_learning_phases(segments)
            results['interpretation'] = self._interpret_learning_journey(segments)
        
        return results
    
    def analyze_cohort_patterns(self, data: pd.DataFrame,
                              cohort_col: str, metric_col: str,
                              date_col: str) -> Dict:
        """Analyze patterns across different student cohorts"""
        
        cohorts = data[cohort_col].unique()
        cohort_results = {}
        
        for cohort in cohorts:
            cohort_data = data[data[cohort_col] == cohort]
            ts_data = cohort_data.set_index(pd.to_datetime(cohort_data[date_col]))[metric_col]
            ts_data = ts_data.resample('D').mean()
            
            cohort_results[cohort] = {
                'mean': ts_data.mean(),
                'std': ts_data.std(),
                'trend': np.polyfit(range(len(ts_data)), ts_data.fillna(0), 1)[0],
                'volatility': ts_data.pct_change().std(),
                'peak_day': ts_data.idxmax(),
                'trough_day': ts_data.idxmin()
            }
        
        # Comparative analysis
        comparison = pd.DataFrame(cohort_results).T
        
        return {
            'cohort_metrics': cohort_results,
            'comparison': comparison.to_dict(),
            'best_performing': comparison['mean'].idxmax(),
            'most_improved': comparison['trend'].idxmax(),
            'most_consistent': comparison['volatility'].idxmin()
        }
    
    def forecast_student_performance(self, historical_data: pd.DataFrame,
                                   student_id: str, metric: str,
                                   days_ahead: int = 7) -> Dict:
        """Forecast individual student performance"""
        
        student_data = historical_data[historical_data['student_id'] == student_id]
        ts_data = student_data.set_index(pd.to_datetime(student_data['timestamp']))[metric]
        # Fixed: Use ffill() instead of fillna(method='ffill')
        ts_data = ts_data.resample('D').mean().ffill()
        
        # Try multiple models
        models_results = {}
        
        # ARIMA
        try:
            arima_model = ARIMA(ts_data, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=days_ahead)
            
            models_results['arima'] = {
                'forecast': arima_forecast.tolist(),
                'confidence_intervals': arima_fit.get_forecast(steps=days_ahead).conf_int().values.tolist(),
                'aic': arima_fit.aic,
                'model': 'ARIMA(1,1,1)'
            }
        except Exception:
            models_results['arima'] = {'error': 'ARIMA failed to converge'}
        
        # Simple Exponential Smoothing
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        try:
            exp_model = ExponentialSmoothing(ts_data, seasonal_periods=7, 
                                            trend='add', seasonal='add')
            exp_fit = exp_model.fit()
            exp_forecast = exp_fit.forecast(steps=days_ahead)
            
            models_results['exponential_smoothing'] = {
                'forecast': exp_forecast.tolist(),
                'model': 'Holt-Winters Exponential Smoothing'
            }
        except Exception:
            models_results['exponential_smoothing'] = {'error': 'Exponential smoothing failed'}
        
        # Moving Average
        ma_forecast = [ts_data.rolling(window=7).mean().iloc[-1]] * days_ahead
        models_results['moving_average'] = {
            'forecast': ma_forecast,
            'model': '7-day Moving Average'
        }
        
        # Trend projection
        x = np.arange(len(ts_data))
        y = ts_data.values
        z = np.polyfit(x, y, 2)  # Quadratic fit
        p = np.poly1d(z)
        future_x = np.arange(len(ts_data), len(ts_data) + days_ahead)
        trend_forecast = p(future_x)
        
        models_results['trend_projection'] = {
            'forecast': trend_forecast.tolist(),
            'model': 'Quadratic Trend Projection'
        }
        
        # Risk assessment
        current_value = ts_data.iloc[-1]
        risk_threshold = ts_data.quantile(0.25)  # Bottom quartile
        
        risk_assessment = {
            'current_performance': current_value,
            'risk_threshold': risk_threshold,
            'at_risk': current_value < risk_threshold,
            'days_until_risk': self._calculate_days_to_risk(models_results, risk_threshold)
        }
        
        return {
            'models': models_results,
            'best_model': self._select_best_model(models_results),
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_recommendations(risk_assessment, models_results)
        }
    
    # Helper methods
    def _calculate_summary_stats(self, ts_data: pd.Series) -> Dict:
        """Calculate comprehensive summary statistics"""
        return {
            'mean': float(ts_data.mean()),
            'std': float(ts_data.std()),
            'min': float(ts_data.min()),
            'max': float(ts_data.max()),
            'range': float(ts_data.max() - ts_data.min()),
            'cv': float(ts_data.std() / ts_data.mean()) if ts_data.mean() != 0 else float('inf'),
            'skewness': float(ts_data.skew()),
            'kurtosis': float(ts_data.kurtosis()),
            'trend_direction': 'increasing' if ts_data.iloc[-7:].mean() > ts_data.iloc[:7].mean() else 'decreasing'
        }
    
    def _test_stationarity(self, ts_data: pd.Series) -> Dict:
        """Test for stationarity using ADF test"""
        adf_result = adfuller(ts_data.dropna())
        
        return {
            'adf_statistic': float(adf_result[0]),
            'p_value': float(adf_result[1]),
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05,
            'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
        }
    
    def _analyze_seasonality(self, ts_data: pd.Series) -> Dict:
        """Analyze seasonal patterns"""
        if len(ts_data) < 14:  # Need at least 2 weeks
            return {'error': 'Insufficient data for seasonality analysis'}
        
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=7)
            self.decomposition = decomposition
            
            seasonal_strength = float(decomposition.seasonal.std() / decomposition.resid.std())
            
            return {
                'has_weekly_pattern': seasonal_strength > 0.5,
                'seasonal_strength': seasonal_strength,
                'peak_day': int(decomposition.seasonal.groupby(decomposition.seasonal.index.dayofweek).mean().idxmax()),
                'trough_day': int(decomposition.seasonal.groupby(decomposition.seasonal.index.dayofweek).mean().idxmin())
            }
        except Exception:
            return {'error': 'Seasonal decomposition failed'}
    
    def _analyze_trend(self, ts_data: pd.Series) -> Dict:
        """Analyze trend component"""
        x = np.arange(len(ts_data))
        # Fixed: Use ffill() instead of fillna(method='ffill')
        y = ts_data.ffill().values
        
        # Linear trend
        linear_coef = np.polyfit(x, y, 1)
        linear_trend = float(linear_coef[0])
        
        # Quadratic trend
        quad_coef = np.polyfit(x, y, 2)
        
        # R-squared for linear fit
        linear_fit = np.polyval(linear_coef, x)
        ss_res = np.sum((y - linear_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        return {
            'linear_slope': linear_trend,
            'trend_strength': abs(linear_trend),
            'r_squared': r_squared,
            'acceleration': float(quad_coef[0] * 2),  # Second derivative
            'trend_type': 'accelerating' if quad_coef[0] > 0.01 else 'decelerating' if quad_coef[0] < -0.01 else 'linear'
        }
    
    def _generate_forecast(self, ts_data: pd.Series, periods: int = 7) -> Dict:
        """Generate forecast using best available method"""
        try:
            model = ARIMA(ts_data, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
            conf_int = model_fit.get_forecast(steps=periods).conf_int()
            
            return {
                'values': forecast.tolist(),
                'lower_bound': conf_int.iloc[:, 0].tolist(),
                'upper_bound': conf_int.iloc[:, 1].tolist(),
                'method': 'ARIMA(1,1,1)'
            }
        except Exception:
            # Fallback to simple moving average
            ma_forecast = [float(ts_data.rolling(window=min(7, len(ts_data))).mean().iloc[-1])] * periods
            return {
                'values': ma_forecast,
                'method': 'Moving Average (fallback)'
            }
    
    def _detect_anomalies(self, ts_data: pd.Series) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        # Calculate rolling statistics
        rolling_mean = ts_data.rolling(window=7, center=True).mean()
        rolling_std = ts_data.rolling(window=7, center=True).std()
        
        # Z-score method
        z_scores = np.abs((ts_data - rolling_mean) / rolling_std)
        anomalies = []
        
        for idx, (date, zscore) in enumerate(z_scores.items()):
            if zscore > 2.5:  # Threshold for anomaly
                anomalies.append({
                    'date': str(date),
                    'value': float(ts_data[date]),
                    'z_score': float(zscore),
                    'type': 'spike' if ts_data[date] > rolling_mean[date] else 'drop'
                })
        
        return anomalies
    
    def _analyze_segments(self, values: np.ndarray, changepoints: List[int]) -> List[Dict]:
        """Analyze characteristics of each segment"""
        segments = []
        start = 0
        
        for cp in changepoints + [len(values)]:
            if cp > start:
                segment_data = values[start:cp]
                
                # Calculate segment metrics
                segment = {
                    'start_idx': int(start),
                    'end_idx': int(cp),
                    'length': int(cp - start),
                    'mean': float(np.mean(segment_data)),
                    'std': float(np.std(segment_data)),
                    'trend': self._calculate_segment_trend(segment_data),
                    'stability': float(1 / (np.std(segment_data) + 1)),  # Higher = more stable
                    'improvement': float(segment_data[-1] - segment_data[0]) if len(segment_data) > 1 else 0.0
                }
                segments.append(segment)
                start = cp
        
        return segments
    
    def _calculate_segment_trend(self, segment_data: np.ndarray) -> str:
        """Calculate trend within a segment"""
        if len(segment_data) < 2:
            return 'stable'
        
        x = np.arange(len(segment_data))
        slope = np.polyfit(x, segment_data, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _identify_learning_phases(self, segments: List[Dict]) -> List[str]:
        """Map segments to learning phases"""
        phases = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                phases.append("Initial Exploration")
            elif segment['mean'] > segments[i-1]['mean'] * 1.2:
                phases.append("Rapid Growth")
            elif segment['mean'] > segments[i-1]['mean'] * 1.05:
                phases.append("Steady Progress")
            elif abs(segment['mean'] - segments[i-1]['mean']) < segments[i-1]['mean'] * 0.05:
                phases.append("Plateau")
            elif segment['mean'] < segments[i-1]['mean'] * 0.95:
                phases.append("Struggle/Regression")
            else:
                phases.append("Consolidation")
        
        return phases
    
    def _interpret_learning_journey(self, segments: List[Dict]) -> str:
        """Provide educational interpretation of the learning journey"""
        if len(segments) <= 1:
            return "Limited data - continue monitoring student progress"
        
        # Calculate overall trajectory
        first_mean = segments[0]['mean']
        last_mean = segments[-1]['mean']
        overall_change = (last_mean - first_mean) / first_mean if first_mean != 0 else 0
        
        # Check recent trend
        recent_trend = segments[-1]['trend']
        
        # Generate interpretation
        if overall_change > 0.3 and recent_trend == 'increasing':
            return "Excellent progress! Student shows strong learning trajectory with continued improvement"
        elif overall_change > 0.1 and recent_trend in ['stable', 'increasing']:
            return "Good progress. Student is developing skills steadily"
        elif overall_change > 0 but recent_trend == 'decreasing':
            return "Overall progress but recent decline - may need refresher or motivation boost"
        elif abs(overall_change) < 0.1:
            return "Student progress has plateaued - consider new challenges or different approaches"
        else:
            return "Student struggling - recommend immediate intervention and support"
    
    def _calculate_days_to_risk(self, forecast_results: Dict, threshold: float) -> Optional[int]:
        """Calculate days until performance drops below risk threshold"""
        for model_name, model_data in forecast_results.items():
            if 'forecast' in model_data and not isinstance(model_data.get('forecast'), str):
                forecast_values = model_data['forecast']
                for day, value in enumerate(forecast_values):
                    if value < threshold:
                        return day + 1
        return None
    
    def _select_best_model(self, models_results: Dict) -> str:
        """Select best model based on available metrics"""
        # Prefer models with AIC/BIC metrics
        models_with_metrics = [(name, data) for name, data in models_results.items() 
                              if 'aic' in data]
        
        if models_with_metrics:
            return min(models_with_metrics, key=lambda x: x[1]['aic'])[0]
        
        # Otherwise prefer ARIMA if available
        if 'arima' in models_results and 'forecast' in models_results['arima']:
            return 'arima'
        
        # Fallback
        return 'moving_average'
    
    def _generate_recommendations(self, risk_assessment: Dict, forecast_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if risk_assessment['at_risk']:
            recommendations.append("⚠️ URGENT: Student is currently at risk - immediate intervention recommended")
        
        days_to_risk = risk_assessment.get('days_until_risk')
        if days_to_risk and days_to_risk <= 7:
            recommendations.append(f"📊 Performance predicted to drop below threshold in {days_to_risk} days")
        
        # Model-specific recommendations
        best_model = self._select_best_model(forecast_results)
        if best_model in forecast_results and 'forecast' in forecast_results[best_model]:
            forecast = forecast_results[best_model]['forecast']
            if len(forecast) > 0:
                trend = 'improving' if forecast[-1] > forecast[0] else 'declining'
                if trend == 'declining':
                    recommendations.append("📉 Declining trend detected - consider:")
                    recommendations.append("   • One-on-one mentoring session")
                    recommendations.append("   • Adjusting difficulty level")
                    recommendations.append("   • Peer collaboration opportunities")
                else:
                    recommendations.append("📈 Positive trend - maintain current approach")
                    recommendations.append("   • Consider additional challenges")
                    recommendations.append("   • Showcase student work")
        
        return recommendations if recommendations else ["✅ Student performing well - continue monitoring"]
