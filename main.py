# main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# Local imports
from config import Config
from cache import DataCache
from charts import ChartGenerator
from cleaning_utils import CSVProcessor  # Added this import

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LeadMirror Analytics API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

class Analytics:
    @staticmethod
    def clean_float(value) -> float:
        """Clean float values for JSON serialization"""
        try:
            if pd.isna(value) or np.isinf(value):
                return 0.0
            return float(np.clip(value, -1e30, 1e30))
        except:
            return 0.0
    
    @staticmethod
    def calculate_metrics(data: pd.DataFrame) -> Dict:
        """Calculate core performance metrics"""
        try:
            total_leads = data['Leads'].sum()
            total_spend = data['Spend'].sum()
            total_goals = data['Goals'].sum()
            
            return {
                'total_leads': Analytics.clean_float(total_leads),
                'total_spend': Analytics.clean_float(total_spend),
                'avg_cost_per_lead': Analytics.clean_float(
                    total_spend / total_leads if total_leads > 0 else 0),
                'conversion_rate': Analytics.clean_float(
                    total_goals / total_leads * 100 if total_leads > 0 else 0),
                'total_goals': Analytics.clean_float(total_goals)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'total_leads': 0.0,
                'total_spend': 0.0,
                'avg_cost_per_lead': 0.0,
                'conversion_rate': 0.0,
                'total_goals': 0.0
            }

# File Processing Endpoints
@app.get("/process_csv/")
async def process_csv():
    """Process CSV file and save cleaned data"""
    try:
        # Ensure directories exist
        Config.DATA_DIR.mkdir(exist_ok=True)
        Config.UPLOADS_DIR.mkdir(exist_ok=True)
        Config.CLEANED_DIR.mkdir(exist_ok=True)
        Config.ATTRIBUTED_DATA_DIR.mkdir(exist_ok=True)
        
        # Process the CSV file
        general_df, attributed_data = CSVProcessor.process_csv_file(Config.RAW_DATA_FILE)
        
        # Clear the cache to force reload
        DataCache.clear_cache()
        
        return {
            "status": "success",
            "message": "Data processed successfully",
            "rows_processed": len(general_df),
            "last_date": general_df['Date'].max().strftime('%Y-%m-%d')
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing CSV: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV: {error_msg}"
        )

@app.get("/get_partners/")
async def get_partners():
    """Get unique partners list"""
    try:
        general_df, _ = DataCache.get_data()
        return {"partners": sorted(general_df['Partner'].unique().tolist())}
    except Exception as e:
        logger.error(f"Error getting partners: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_charts/")
async def generate_charts(partner: str, start_date: str, end_date: str):
    """Generate all charts and analysis for a partner"""
    try:
        # Get data
        general_df, attributed_data = DataCache.get_data()
        
        # Verify partner exists
        if partner not in general_df['Partner'].unique():
            raise HTTPException(status_code=404, detail=f"Partner not found: {partner}")

        # Convert dates
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Please use YYYY-MM-DD format: {str(e)}"
            )

        # Filter data
        filtered_df = general_df[
            (general_df['Date'] >= start) &
            (general_df['Date'] <= end) &
            (general_df['Partner'] == partner)
        ].copy()
        
        # Filter attributed data
        filtered_attributed = {}
        for key, df in attributed_data.items():
            filtered_attributed[key] = df[
                (df['Partner'] == partner) &
                (df['Date'] >= start) &
                (df['Date'] <= end)
            ].copy()

        if filtered_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for partner {partner} in the specified date range"
            )

        # Generate charts
        chart_gen = ChartGenerator(filtered_df, filtered_attributed, partner, start, end)
        
        return {
            "charts": {
                "locations": {
                    "top": chart_gen.generate_top_locations_chart(),
                    "general": chart_gen.generate_general_locations_chart()
                },
                "targeting": {
                    "primary": {
                        "top": chart_gen.generate_top_primary_targeting_chart(),
                        "general": chart_gen.generate_general_primary_targeting_chart()
                    },
                    "secondary": {
                        "top": chart_gen.generate_top_secondary_targeting_chart(),
                        "general": chart_gen.generate_general_secondary_targeting_chart()
                    }
                },
                "ads": chart_gen.generate_top_ads_chart()
            },
            "analysis": {
                "locations": chart_gen.generate_locations_analysis(),
                "primary": chart_gen.generate_primary_analysis(),
                "secondary": chart_gen.generate_secondary_analysis(),
                "ads": chart_gen.generate_ad_analysis()
            },
            "metadata": {
                "partner": partner,
                "date_range": {
                    "start": start.isoformat(),
                    "end": end.isoformat()
                },
                "total_records": len(filtered_df),
                "generated_at": datetime.now().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance_prediction/")
async def get_prediction(partner: str, forecast_days: int = 7):
    """Generate performance predictions"""
    try:
        general_df, _ = DataCache.get_data()
        partner_data = general_df[general_df['Partner'] == partner]
        
        if partner_data.empty:
            raise HTTPException(status_code=404, detail="Partner not found")
        
        recent = partner_data.sort_values('Date').tail(30)
        avg = Analytics.clean_float(recent['Leads'].mean())
        std = Analytics.clean_float(recent['Leads'].std())
        
        predictions = [avg] * forecast_days
        upper = [Analytics.clean_float(avg + 2*std)] * forecast_days
        lower = [Analytics.clean_float(max(0, avg - 2*std))] * forecast_days
        actuals = [Analytics.clean_float(x) for x in recent['Leads'].tolist()]
        
        dates = [(recent['Date'].max() + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(1, forecast_days + 1)]
        
        return {
            'predictions': {
                'dates': dates,
                'predicted': predictions,
                'actual': actuals,
                'confidence_intervals': {
                    'upper': upper,
                    'lower': lower
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimization_recommendations/")
async def get_recommendations(partner: str):
    """Get optimization recommendations"""
    try:
        general_df, _ = DataCache.get_data()
        partner_data = general_df[general_df['Partner'] == partner]
        
        if partner_data.empty:
            raise HTTPException(status_code=404, detail="Partner not found")
        
        metrics = Analytics.calculate_metrics(partner_data)
        trend = Analytics.clean_float(partner_data['Leads'].pct_change().mean() * 100)
        
        recommendations = [
            {
                'title': 'Lead Generation Optimization',
                'description': f"Current performance: {metrics['total_leads']:.0f} leads at ₹{metrics['avg_cost_per_lead']:.2f} per lead",
                'actions': [
                    {'label': 'Review Top Performing Channels'},
                    {'label': 'Optimize Budget Allocation'}
                ],
                'metrics': metrics
            },
            {
                'title': 'Conversion Rate Enhancement',
                'description': f"Current conversion rate: {metrics['conversion_rate']:.1f}%",
                'actions': [
                    {'label': 'Analyze Success Patterns'},
                    {'label': 'Implement A/B Testing'}
                ],
                'metrics': {
                    'trend': trend,
                    'potential_improvement': Analytics.clean_float(metrics['conversion_rate'] * 1.2)
                }
            }
        ]
        
        return {'recommendations': recommendations}
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entity_analysis/")
async def analyze_entity(partner: str):
    """Get entity performance analysis"""
    try:
        general_df, _ = DataCache.get_data()
        partner_data = general_df[general_df['Partner'] == partner]
        
        if partner_data.empty:
            raise HTTPException(status_code=404, detail="Partner not found")
        
        return {
            'metrics': Analytics.calculate_metrics(partner_data),
            'trends': {
                'leads': Analytics.clean_float(partner_data['Leads'].pct_change().mean() * 100),
                'spend': Analytics.clean_float(partner_data['Spend'].pct_change().mean() * 100)
            }
        }
    except Exception as e:
        logger.error(f"Error in entity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)