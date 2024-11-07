# dashboard.py

import streamlit as st
import requests
import base64
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache

FASTAPI_URL = "http://127.0.0.1:8000"
REFRESH_INTERVAL = 300

class Dashboard:
    """LeadMirror Analytics Dashboard"""
    
    STYLES = """
    <style>
    .metric-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 10px 0;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
    }
    .metric-item {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .metric-item:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        color: #6B7280;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-value {
        color: #111827;
        font-size: 24px;
        font-weight: 600;
    }
    .metric-type-leads { border-left-color: #2ecc71; }
    .metric-type-spend { border-left-color: #e74c3c; }
    .metric-type-cost { border-left-color: #f1c40f; }
    .metric-type-rate { border-left-color: #9b59b6; }
    .metric-type-goals { border-left-color: #3498db; }
    </style>
    """

    def __init__(self):
        st.set_page_config(page_title="LeadMirror", layout="wide", page_icon="📊")
        st.markdown(self.STYLES, unsafe_allow_html=True)
    
    @staticmethod
    @st.cache_data(ttl=REFRESH_INTERVAL)
    def _fetch_data(endpoint: str, params: Dict = None) -> Dict:
        """Fetch data from API with caching"""
        try:
            response = requests.get(f"{FASTAPI_URL}/{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return {}
    
    def _get_metric_type(self, key: str) -> str:
        """Get metric type for styling"""
        if 'leads' in key.lower(): return 'leads'
        if 'spend' in key.lower(): return 'spend'
        if 'cost' in key.lower(): return 'cost'
        if 'rate' in key.lower(): return 'rate'
        if 'goals' in key.lower(): return 'goals'
        return ''
    
    def _get_metric_icon(self, key: str) -> str:
        """Get icon for metric type"""
        icons = {
            'leads': '📈',
            'spend': '💰',
            'cost': '💎',
            'rate': '🎯',
            'goals': '🏆',
            'conversion': '✨'
        }
        return next((v for k, v in icons.items() if k in key.lower()), '📊')
    
    def _format_metric_value(self, value: Any, key: str) -> str:
        """Format metric value based on type"""
        try:
            if isinstance(value, str): return value
            return (
                f"₹{value:,.2f}" if any(x in key.lower() for x in ['cost', 'spend'])
                else f"{value:.1f}%" if any(x in key.lower() for x in ['rate', 'trend'])
                else f"{value:,.0f}"
            )
        except:
            return str(value)
    
    def _display_chart(self, chart_data: str, title: str):
        """Display chart with error handling"""
        if not chart_data:
            st.info(f"No data for {title}")
            return
        try:
            fig = go.Figure(json.loads(base64.b64decode(chart_data)))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(title)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
    
    def _display_metrics(self, metrics: Dict[str, Any]):
        """Display metrics in modern cards"""
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        for key, value in metrics.items():
            metric_type = self._get_metric_type(key)
            icon = self._get_metric_icon(key)
            label = key.replace('_', ' ').upper()
            formatted_value = self._format_metric_value(value, key)
            
            metric_html = f"""
                <div class="metric-item metric-type-{metric_type}">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _plot_predictions(self, predictions: Dict):
        """Create and display prediction chart"""
        fig = go.Figure()
        
        if actual := predictions.get('actual'):
            recent_dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") 
                          for i in range(len(actual))[::-1]]
            fig.add_trace(go.Scatter(
                x=recent_dates,
                y=actual,
                name='Actual',
                line=dict(color='#1f77b4', width=3)
            ))
        
        fig.add_trace(go.Scatter(
            x=predictions['dates'],
            y=predictions['predicted'],
            name='Predicted',
            line=dict(color='#2ca02c', width=3, dash='dot')
        ))
        
        if ci := predictions.get('confidence_intervals'):
            fig.add_trace(go.Scatter(
                x=predictions['dates'] + predictions['dates'][::-1],
                y=ci['upper'] + ci['lower'][::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title="7-Day Forecast",
            xaxis_title='Date',
            yaxis_title='Leads',
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", y=1.1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_section_analysis(self, section: str, charts: Dict, analysis: Dict):
        """Display section analysis"""
        if section_data := analysis.get(f"{section}", {}):
            if summary := section_data.get("summary", {}):
                metrics = {
                    "Total Leads": summary.get("total_leads", 0),
                    "Total Spend": summary.get("total_spend", 0),
                    "Cost Per Lead": summary.get("avg_cost_per_lead", 0)
                }
                
                # Only display metrics if there's actual data
                if any(metrics.values()):
                    self._display_metrics(metrics)
                
                if best := summary.get("best_performer"):
                    st.success(f"""
                    🏆 **Best {section.title()}:** {best.get(section.lower())}
                    - Leads: **{best.get('leads', 0):,.0f}**
                    - CPL: **₹{best.get('cost', 0):,.2f}**
                    - Goals: **{best.get('goals', 0):,.0f}**
                    """)
        
        section_charts = charts.get(section, {}) if section != "ads" else {"ads": charts.get("ads")}
        
        if "general" in section_charts:
            st.subheader("Overview")
            self._display_chart(section_charts.get("general"), f"{section.title()} Analysis")
            st.markdown("---")
        
        if top_chart := section_charts.get("top", section_charts.get("ads")):
            st.subheader("Top Performance")
            self._display_chart(top_chart, f"Top {section.title()}")
    
    def _show_insights(self, partner: str):
        """Display insights section"""
        st.subheader("📈 Performance Forecast")
        predictions = self._fetch_data(
            "performance_prediction",
            {"partner": partner, "forecast_days": 7}
        )
        if pred_data := predictions.get('predictions'):
            self._plot_predictions(pred_data)
        
        st.subheader("📊 Performance Metrics")
        analysis = self._fetch_data("entity_analysis", {"partner": partner})
        
        if metrics := analysis.get('metrics'):
            self._display_metrics({
                "Total Leads": metrics.get("total_leads", 0),
                "Total Spend": metrics.get("total_spend", 0),
                "Cost per Lead": metrics.get("avg_cost_per_lead", 0),
                "Conversion Rate": metrics.get("conversion_rate", 0)
            })
        
        if trends := analysis.get('trends'):
            st.subheader("📈 Trends")
            trend_metrics = {
                k.replace('_', ' ').title(): f"{v:+.1f}%" 
                for k, v in trends.items()
            }
            self._display_metrics(trend_metrics)
        
        st.markdown("---")
        recommendations = self._fetch_data(
            "optimization_recommendations",
            {"partner": partner}
        )
        
        if recs := recommendations.get('recommendations'):
            st.subheader("🎯 Optimization Suggestions")
            for i, rec in enumerate(recs, 1):
                with st.expander(f"Suggestion {i}: {rec['title']}", expanded=i==1):
                    st.write("📋 **Details:**", rec['description'])
                    
                    if metrics := rec.get('metrics'):
                        st.write("📊 **Impact Metrics:**")
                        for metric, value in metrics.items():
                            formatted = self._format_metric_value(value, metric)
                            st.write(f"- {metric.title()}: {formatted}")
                    
                    if actions := rec.get('actions'):
                        st.write("⚡ **Actions:**")
                        cols = st.columns(len(actions))
                        for col, action in zip(cols, actions):
                            with col:
                                if st.button(
                                    action['label'],
                                    key=f"action_{i}_{action['label']}"[:100]
                                ):
                                    st.success(f"Action '{action['label']}' initiated!")
    
    def show_analysis(self, partner: str, start_date: datetime, end_date: datetime):
        """Show main analysis dashboard"""
        data = self._fetch_data("generate_charts", {
            "partner": partner,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        })
        
        charts = data.get("charts", {})
        analysis = data.get("analysis", {})
        
        tabs = st.tabs(["📍 Location", "🎯 Targeting", "📢 Ads", "🧠 Insights"])
        
        with tabs[0]:
            st.header("Location Analysis")
            self._show_section_analysis("locations", charts, analysis)
        
        with tabs[1]:
            st.header("Targeting Analysis")
            self._show_section_analysis("primary", charts.get("targeting", {}), analysis)
            st.markdown("---")
            self._show_section_analysis("secondary", charts.get("targeting", {}), analysis)
        
        with tabs[2]:
            st.header("Ad Analysis")
            self._show_section_analysis("ads", charts, analysis)
        
        with tabs[3]:
            self._show_insights(partner)
    
    def setup_sidebar(self) -> Optional[Tuple[str, datetime, datetime]]:
        """Setup sidebar controls"""
        with st.sidebar:
            st.header("🎯 Settings")
            
            if st.button("Process Data"):
                with st.spinner("Processing..."):
                    try:
                        requests.get(f"{FASTAPI_URL}/process_csv/").raise_for_status()
                        st.success("Success!")
                        self._fetch_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return None
            
            partners = self._fetch_data("get_partners").get("partners", [])
            if not partners:
                st.warning("No partners available")
                return None
            
            end_date = datetime.today()
            return (
                st.selectbox("Partner", partners),
                st.date_input("Start Date", end_date - timedelta(days=30)),
                st.date_input("End Date", end_date)
            )

def main():
    dashboard = Dashboard()
    st.title("📊 LeadMirror Analytics")
    if params := dashboard.setup_sidebar():
        dashboard.show_analysis(*params)

if __name__ == "__main__":
    main()