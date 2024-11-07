# charts.py

import pandas as pd
import plotly.graph_objects as go
from typing import Dict
import logging
import numpy as np
import base64

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Class to generate charts for the dashboard.
    """

    def __init__(
        self,
        general_df: pd.DataFrame,
        attributed_data: Dict[str, pd.DataFrame],
        partner: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        self.general_df = general_df.copy()
        self.attributed_data = attributed_data  # Dict containing attributed data for each targeting column
        self.partner = partner
        self.start_date = start_date
        self.end_date = end_date

    def _figure_to_base64(self, fig) -> str:
        """Serialize Plotly figure to base64 string using JSON."""
        fig_json = fig.to_json()
        fig_base64 = base64.b64encode(fig_json.encode("utf-8")).decode("utf-8")
        return fig_base64

    def shorten_label(self, label, max_length=15):
        """Shorten labels that exceed max_length and add ellipsis."""
        label = str(label)
        return label if len(label) <= max_length else label[:max_length] + "..."

    def generate_top_entities_chart(self, entity_column: str, title: str) -> str:
        """
        Generate Top 10 entities chart with bar for Leads and lines for Cost Per Lead and Goals.

        Args:
            entity_column (str): The column to analyze.
            title (str): The chart title.

        Returns:
            str: Base64-encoded JSON of the Plotly figure.
        """
        try:
            if entity_column == "Ad":
                data = self.general_df.copy()
                data = data[
                    (data["Partner"] == self.partner)
                    & (data["Date"] >= self.start_date)
                    & (data["Date"] <= self.end_date)
                ]
                # Group data by the entity column
                data = (
                    data.groupby(entity_column)
                    .agg({"Leads": "sum", "Spend": "sum", "Goals": "sum"})
                    .reset_index()
                )
            else:
                # Get attributed data for the column
                data = self.attributed_data.get(entity_column, None)
                if data is None or data.empty:
                    logger.warning(f"No attributed data available for {entity_column} chart")
                    return ""
                # Data is already filtered by partner and date range in main.py

                # Group data by the entity column to sum leads for unique variables
                data = (
                    data.groupby(entity_column)
                    .agg({"Leads": "sum", "Spend": "sum", "Goals": "sum"})
                    .reset_index()
                )

            if data.empty:
                logger.warning(f"No data available for {entity_column} chart for partner {self.partner}")
                return ""

            # Calculate Cost Per Lead safely
            data["Cost_Per_Lead"] = np.where(
                data["Leads"] > 0,
                data["Spend"] / data["Leads"],
                np.nan
            )

            # Get Top 10 entities by Leads
            top_entities = data.nlargest(10, "Leads")

            if top_entities.empty:
                logger.warning(f"No data available for {entity_column} chart after filtering top entities")
                return ""

            # Shorten labels
            top_entities["short_label"] = top_entities[entity_column].apply(
                lambda x: self.shorten_label(x, max_length=15)
            )

            fig = go.Figure()

            # Bar chart for Leads
            fig.add_trace(
                go.Bar(
                    x=top_entities["short_label"],
                    y=top_entities["Leads"],
                    name="Leads",
                    marker_color="rgb(55, 83, 109)",
                    hovertemplate=f"{entity_column}: %{{customdata}}<br>Leads: %{{y:.0f}}<extra></extra>",
                    customdata=top_entities[entity_column],
                )
            )

            # Line chart for Cost Per Lead
            fig.add_trace(
                go.Scatter(
                    x=top_entities["short_label"],
                    y=top_entities["Cost_Per_Lead"],
                    name="Cost per Lead",
                    mode="lines+markers",
                    marker=dict(color="rgb(26, 118, 255)"),
                    yaxis="y2",
                    hovertemplate=f"{entity_column}: %{{customdata}}<br>Cost per Lead: ₹%{{y:.2f}}<extra></extra>",
                    customdata=top_entities[entity_column],
                )
            )

            # Line chart for Goals
            fig.add_trace(
                go.Scatter(
                    x=top_entities["short_label"],
                    y=top_entities["Goals"],
                    name="Goals",
                    mode="lines+markers",
                    marker=dict(color="rgb(50, 205, 50)"),
                    yaxis="y2",
                    hovertemplate=f"{entity_column}: %{{customdata}}<br>Goals: %{{y:.0f}}<extra></extra>",
                    customdata=top_entities[entity_column],
                )
            )

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=entity_column,
                yaxis=dict(title="Number of Leads"),
                yaxis2=dict(title="Cost per Lead (₹) / Goals", overlaying="y", side="right"),
                legend=dict(x=0, y=1.1, orientation="h"),
                xaxis_tickangle=-45,
                xaxis=dict(automargin=True),
                margin=dict(l=50, r=50, t=80, b=150),
                height=600,
                template="plotly_white",
                barmode="group",  # Ensure bars are displayed separately
            )

            return self._figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating {entity_column} chart: {str(e)}")
            return ""

    def generate_general_entities_chart(self, entity_column: str, title: str) -> str:
        """
        Generate general analysis chart with combinations.

        Args:
            entity_column (str): The column to analyze.
            title (str): The chart title.

        Returns:
            str: Base64-encoded JSON of the Plotly figure.
        """
        try:
            # Copy data for the selected partner and date range
            data = self.general_df[
                (self.general_df["Partner"] == self.partner)
                & (self.general_df["Date"] >= self.start_date)
                & (self.general_df["Date"] <= self.end_date)
            ].copy()

            if data.empty:
                logger.warning(f"No data available for general {entity_column} chart")
                return ""

            # Ensure that the entity_column is a string
            data[entity_column] = data[entity_column].astype(str)

            # Group by the exact combinations (no normalization)
            data = (
                data.groupby(entity_column)
                .agg({"Leads": "sum", "Spend": "sum", "Goals": "sum"})
                .reset_index()
            )

            # Calculate Cost Per Lead safely
            data["Cost_Per_Lead"] = np.where(
                data["Leads"] > 0,
                data["Spend"] / data["Leads"],
                np.nan
            )

            if data.empty:
                logger.warning(f"No data available after processing for general {entity_column} chart")
                return ""

            # Shorten labels
            data["short_label"] = data[entity_column].apply(lambda x: self.shorten_label(x, max_length=15))

            # Ensure uniqueness by adding an index
            data.reset_index(drop=True, inplace=True)
            data['unique_label'] = data.index.astype(str) + '. ' + data['short_label']

            fig = go.Figure()

            # Bar chart for Leads
            fig.add_trace(
                go.Bar(
                    x=data["unique_label"],
                    y=data["Leads"],
                    name="Leads",
                    marker_color="rgb(55, 83, 109)",
                    hovertemplate=f"{entity_column}: %{{customdata}}<br>Leads: %{{y:.0f}}<extra></extra>",
                    customdata=data[entity_column],
                )
            )

            # Line chart for Cost Per Lead
            fig.add_trace(
                go.Scatter(
                    x=data["unique_label"],
                    y=data["Cost_Per_Lead"],
                    name="Cost per Lead",
                    mode="lines+markers",
                    marker=dict(color="rgb(26, 118, 255)"),
                    yaxis="y2",
                    hovertemplate=f"{entity_column}: %{{customdata}}<br>Cost per Lead: ₹%{{y:.2f}}<extra></extra>",
                    customdata=data[entity_column],
                )
            )

            # Line chart for Goals
            fig.add_trace(
                go.Scatter(
                    x=data["unique_label"],
                    y=data["Goals"],
                    name="Goals",
                    mode="lines+markers",
                    marker=dict(color="rgb(50, 205, 50)"),
                    yaxis="y2",
                    hovertemplate=f"{entity_column}: %{{customdata}}<br>Goals: %{{y:.0f}}<extra></extra>",
                    customdata=data[entity_column],
                )
            )

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=entity_column,
                yaxis=dict(title="Number of Leads"),
                yaxis2=dict(title="Cost per Lead (₹) / Goals", overlaying="y", side="right"),
                legend=dict(x=0, y=1.1, orientation="h"),
                xaxis_tickangle=-45,
                xaxis=dict(automargin=True),
                margin=dict(l=50, r=50, t=80, b=150),
                height=600,
                template="plotly_white",
                barmode="group",  # Ensure bars are displayed separately
            )

            return self._figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating general {entity_column} chart: {str(e)}")
            return ""

    def generate_entity_analysis(self, entity_column: str) -> Dict:
        """
        Generate performance analysis summary for a given entity.

        Args:
            entity_column (str): The column to analyze.

        Returns:
            Dict: Summary of performance.
        """
        try:
            # Filter data for the selected partner and date range
            filtered_data = self.general_df[
                (self.general_df["Partner"] == self.partner)
                & (self.general_df["Date"] >= self.start_date)
                & (self.general_df["Date"] <= self.end_date)
            ].copy()

            if filtered_data.empty:
                logger.warning(f"No data available for analysis of {entity_column}")
                return {}

            # Ensure that the entity_column is a string
            filtered_data[entity_column] = filtered_data[entity_column].astype(str)

            # Calculate total leads and total spend before grouping
            total_leads = filtered_data["Leads"].sum()
            total_spend = filtered_data["Spend"].sum()

            # Group by the entity_column
            data = (
                filtered_data.groupby(entity_column)
                .agg({"Leads": "sum", "Spend": "sum", "Goals": "sum"})
                .reset_index()
            )

            # Calculate cost per lead safely
            data["Cost_Per_Lead"] = np.where(
                data["Leads"] > 0,
                data["Spend"] / data["Leads"],
                np.nan
            )

            if data.empty:
                logger.warning(f"No data available after processing for {entity_column} analysis")
                return {}

            # Calculate average cost per lead as total spend divided by total leads
            avg_cost_per_lead = total_spend / total_leads if total_leads > 0 else None

            # Get best performer (highest number of leads)
            best_performer = data.loc[data["Leads"].idxmax()]

            # Create summary
            summary = {
                'total_leads': float(total_leads),
                'total_spend': float(total_spend),
                'avg_cost_per_lead': float(avg_cost_per_lead) if avg_cost_per_lead is not None else None,
                'best_performer': {
                    entity_column.lower(): best_performer[entity_column],
                    'leads': float(best_performer["Leads"]),
                    'cost': float(best_performer["Cost_Per_Lead"]) if not np.isnan(best_performer["Cost_Per_Lead"]) else None,
                    'goals': float(best_performer["Goals"]),
                },
            }

            return {'summary': summary}

        except Exception as e:
            logger.error(f"Error generating {entity_column} analysis: {str(e)}")
            return {}

    def generate_primary_analysis(self) -> Dict:
        return self.generate_entity_analysis('Primary')

    def generate_secondary_analysis(self) -> Dict:
        return self.generate_entity_analysis('Secondary')

    def generate_locations_analysis(self) -> Dict:
        return self.generate_entity_analysis('Locations')

    def generate_ad_analysis(self) -> Dict:
        """
        Generate AD performance analysis summary.

        Returns:
            Dict: Summary of AD performance.
        """
        try:
            # Filter data for the selected partner and date range
            filtered_data = self.general_df[
                (self.general_df["Partner"] == self.partner)
                & (self.general_df["Date"] >= self.start_date)
                & (self.general_df["Date"] <= self.end_date)
            ].copy()

            if filtered_data.empty:
                logger.warning("No AD data available for analysis")
                return {}

            # Calculate total leads and total spend before grouping
            total_leads = filtered_data["Leads"].sum()
            total_spend = filtered_data["Spend"].sum()

            # Group by AD
            data = (
                filtered_data.groupby("Ad")
                .agg({"Leads": "sum", "Spend": "sum", "Goals": "sum"})
                .reset_index()
            )

            # Calculate cost per lead safely
            data["Cost_Per_Lead"] = np.where(
                data["Leads"] > 0,
                data["Spend"] / data["Leads"],
                np.nan
            )

            if data.empty:
                logger.warning("No AD data available after processing")
                return {}

            # Calculate average cost per lead as total spend divided by total leads
            avg_cost_per_lead = total_spend / total_leads if total_leads > 0 else None

            # Get best performer (highest number of leads)
            best_performer = data.loc[data["Leads"].idxmax()]

            # Create summary
            summary = {
                'total_leads': float(total_leads),
                'total_spend': float(total_spend),
                'avg_cost_per_lead': float(avg_cost_per_lead) if avg_cost_per_lead is not None else None,
                'best_performer': {
                    'ad': best_performer["Ad"],
                    'leads': float(best_performer["Leads"]),
                    'cost': float(best_performer["Cost_Per_Lead"]) if not np.isnan(best_performer["Cost_Per_Lead"]) else None,
                    'goals': float(best_performer["Goals"]),
                },
            }

            return {'summary': summary}

        except Exception as e:
            logger.error(f"Error generating AD analysis: {str(e)}")
            return {}

    # Specific methods for each analysis

    def generate_top_primary_targeting_chart(self) -> str:
        return self.generate_top_entities_chart("Primary", f"Top 10 Primary Targeting - {self.partner}")

    def generate_top_secondary_targeting_chart(self) -> str:
        return self.generate_top_entities_chart("Secondary", f"Top 10 Secondary Targeting - {self.partner}")

    def generate_top_locations_chart(self) -> str:
        return self.generate_top_entities_chart("Locations", f"Top 10 Locations - {self.partner}")

    def generate_general_primary_targeting_chart(self) -> str:
        return self.generate_general_entities_chart("Primary", f"Primary Targeting Combinations - {self.partner}")

    def generate_general_secondary_targeting_chart(self) -> str:
        return self.generate_general_entities_chart("Secondary", f"Secondary Targeting Combinations - {self.partner}")

    def generate_general_locations_chart(self) -> str:
        return self.generate_general_entities_chart("Locations", f"Location Combinations - {self.partner}")

    def generate_top_ads_chart(self) -> str:
        # Ads don't need splitting
        return self.generate_top_entities_chart("Ad", f"Top 10 Ads - {self.partner}")
