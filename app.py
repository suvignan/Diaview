import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = dash.Dash(__name__)
app.title = "DiaView: Type 1 Diabetes Dashboard"

# Data (unchanged)
timestamps = pd.date_range(start="2023-11-16 00:00", end="2023-11-30 23:55", freq="5min")
glucose = np.random.normal(loc=120, scale=30, size=len(timestamps))
glucose = np.clip(glucose, 40, 300)
df = pd.DataFrame({"timestamp": timestamps, "glucose": glucose, "carbs": 0, "bolus_insulin": 0, "basal_insulin": 1.0})
for hour in [8, 12, 18]:
    meal_times = df["timestamp"].dt.hour == hour
    df.loc[meal_times, "carbs"] = np.random.uniform(20, 60, sum(meal_times))
    df.loc[meal_times, "bolus_insulin"] = np.random.uniform(2, 6, sum(meal_times))
interruption = (df["timestamp"] >= "2023-11-26 01:00") & (df["timestamp"] <= "2023-11-26 08:00")
df.loc[interruption, "glucose"] += 50
df.loc[interruption, "basal_insulin"] = 0.0
hypo_indices = np.random.choice(df.index, size=10, replace=False)
for idx in hypo_indices:
    end_idx = min(idx + 12, len(df) - 1)
    if end_idx - idx + 1 >= 13:
        df.loc[df.index[idx:end_idx + 1], "glucose"] = np.linspace(80, 50, end_idx - idx + 1)
    else:
        df.loc[df.index[idx:], "glucose"] = np.linspace(80, 50, len(df) - idx)

# AGP (unchanged)
def compute_agp(df):
    hourly = df.groupby(df["timestamp"].dt.hour)["glucose"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return hourly.unstack()

def create_agp_plot(df):
    percentiles = compute_agp(df)
    fig = go.Figure()
    colors = ["#FF9999", "#FF6666", "#FF0000", "#6666FF", "#9999FF"]
    for i, p in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
        fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles[p], mode="lines", 
                                 name=f"{int(p*100)}th", line=dict(color=colors[i])))
    hourly_carbs = df.groupby(df["timestamp"].dt.hour)["carbs"].sum()
    hourly_insulin = df.groupby(df["timestamp"].dt.hour)["bolus_insulin"].sum()
    fig.add_trace(go.Bar(x=hourly_carbs.index, y=hourly_carbs, name="Carbs (g)", yaxis="y2", opacity=0.5))
    fig.add_trace(go.Bar(x=hourly_insulin.index, y=hourly_insulin, name="Bolus Insulin (U)", yaxis="y2", opacity=0.5))
    fig.update_layout(
        title="Ambulatory Glucose Profile (AGP) - 14 Days",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Glucose (mg/dL)", range=[0, 300]),
        yaxis2=dict(title="Carbs/Insulin", overlaying="y", side="right", range=[0, max(hourly_carbs.max(), hourly_insulin.max()) * 1.2]),
        barmode="overlay",
        template="plotly_white"
    )
    return fig

# Multi Horizon Graphs (unchanged)
def create_multi_horizon_graphs(df, selected_dates):
    if not selected_dates or not isinstance(selected_dates, list) or not selected_dates:
        return go.Figure().update_layout(title="No dates selected")
    baseline = 120
    fig = go.Figure()
    days = len(selected_dates)
    subplot_height = 0.9 / days if days > 0 else 0.9
    for i, date in enumerate(selected_dates):
        daily_df = df[df["timestamp"].dt.date == pd.to_datetime(date).date()]
        if daily_df.empty:
            continue
        above = (daily_df["glucose"] - baseline).clip(lower=0)
        below = (daily_df["glucose"] - baseline).clip(upper=0)
        fig.add_trace(go.Scatter(x=daily_df["timestamp"], y=above, fill="tozeroy", mode="none", 
                                 name=f"{date}: Above Target", line_color="red", showlegend=(i == 0), yaxis=f"y{i+1}"))
        fig.add_trace(go.Scatter(x=daily_df["timestamp"], y=below, fill="tozeroy", mode="none", 
                                 name=f"{date}: Below Target", line_color="blue", showlegend=(i == 0), yaxis=f"y{i+1}"))
        carbs = daily_df[daily_df["carbs"] > 0]
        insulin = daily_df[daily_df["bolus_insulin"] > 0]
        fig.add_trace(go.Scatter(x=carbs["timestamp"], y=carbs["carbs"], mode="markers", 
                                 name=f"{date}: Carbs (g)", marker=dict(color="green", size=8), showlegend=(i == 0), yaxis=f"y{i+2}"))
        fig.add_trace(go.Scatter(x=insulin["timestamp"], y=insulin["bolus_insulin"], mode="markers", 
                                 name=f"{date}: Insulin (U)", marker=dict(color="purple", size=8), showlegend=(i == 0), yaxis=f"y{i+2}"))
    fig.update_layout(
        title=f"Multi-Day Horizon Graphs ({days} Days Selected)",
        xaxis_title="Time",
        height=max(200 * days, 200),
        grid={"rows": days, "columns": 1, "pattern": "independent"},
        template="plotly_white"
    )
    for i in range(days):
        fig.update_layout({
            f"yaxis{i*2 + 1}": dict(title="Glucose Deviation (mg/dL)" if i == 0 else "", range=[-100, 100], 
                                    domain=[1 - (i + 1) * subplot_height, 1 - i * subplot_height], anchor="x"),
            f"yaxis{i*2 + 2}": dict(title="Carbs/Insulin" if i == 0 else "", range=[0, 60], overlaying=f"y{i*2 + 1}", 
                                    side="right", showgrid=False)
        })
    return fig

# Pattern Detection (unchanged)
def detect_patterns(df, pattern_type="meal", window_hours=3, n_clusters=3):
    try:
        time_series = []
        if pattern_type == "meal":
            event_times = df[df["carbs"] > 0]["timestamp"].drop_duplicates()
            for event_time in event_times:
                window = df[(df["timestamp"] >= event_time) & 
                            (df["timestamp"] <= event_time + pd.Timedelta(hours=window_hours))]
                if len(window) >= 36:
                    time_series.append(window["glucose"].values[:36])
        elif pattern_type == "hypoglycemia":
            hypo_starts = df[(df["glucose"] < 70) & (df["glucose"].shift(1) >= 70)]["timestamp"]
            for event_time in hypo_starts:
                window = df[(df["timestamp"] >= event_time) & 
                            (df["timestamp"] <= event_time + pd.Timedelta(hours=window_hours))]
                if len(window) >= 36:
                    time_series.append(window["glucose"].values[:36])
        if not time_series:
            return None, None, f"No valid {pattern_type} windows found"
        time_series = np.array(time_series)
        scaled_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
        labels = model.fit_predict(scaled_series)
        return scaled_series, labels, None
    except Exception as e:
        print(f"Error in detect_patterns with pattern_type={pattern_type}: {str(e)}")
        return None, None, str(e)

def plot_patterns(scaled_series, labels, pattern_type="meal", n_clusters=3):
    if scaled_series is None or labels is None:
        return go.Figure().update_layout(title=f"No {pattern_type.capitalize()} Patterns Detected")
    fig = go.Figure()
    colors = ["#FF0000", "#00FF00", "#0000FF"]
    for cluster in range(n_clusters):
        cluster_series = scaled_series[labels == cluster]
        mean_series = cluster_series.mean(axis=0).flatten()
        fig.add_trace(go.Scatter(
            x=np.arange(len(mean_series)) * 5,
            y=mean_series,
            mode="lines",
            name=f"Cluster {cluster} ({len(cluster_series)} events)",
            line=dict(color=colors[cluster], width=4)  # Increased line thickness
        ))
    fig.update_layout(
        title=f"{pattern_type.capitalize()} Response Patterns (3-Hour Window)",
        xaxis_title="Minutes Post-Event",
        yaxis_title="Scaled Glucose",
        yaxis_range=[-2, 2],
        template="plotly_white"
    )
    return fig

def export_patterns(scaled_series, labels, pattern_type, n_clusters=3):
    if scaled_series is None or labels is None:
        return pd.DataFrame()
    data = {}
    for cluster in range(n_clusters):
        cluster_series = scaled_series[labels == cluster]
        if len(cluster_series) > 0:
            mean_series = cluster_series.mean(axis=0).flatten()
            data[f"Cluster {cluster} ({len(cluster_series)} events)"] = mean_series
    df_export = pd.DataFrame(data, index=[f"{i*5} min" for i in range(len(mean_series))])
    df_export.index.name = "Time Post-Event"
    return df_export

# Diary Content (unchanged)
def create_diary(df, date):
    try:
        daily_df = df[df["timestamp"].dt.date == pd.to_datetime(date).date()].copy()
        if daily_df.empty:
            return html.Div("No data for selected date")

        mean_glucose = daily_df["glucose"].mean()
        ea1c = (mean_glucose + 46.7) / 28.7
        cv = (daily_df["glucose"].std() / mean_glucose) * 100 if mean_glucose > 0 else 0
        tir_70_180 = len(daily_df[(daily_df["glucose"] >= 70) & (daily_df["glucose"] <= 180)]) / len(daily_df) * 100
        time_below_70 = len(daily_df[daily_df["glucose"] < 70]) / len(daily_df) * 100
        time_below_54 = len(daily_df[daily_df["glucose"] < 54]) / len(daily_df) * 100
        time_180_250 = len(daily_df[(daily_df["glucose"] >= 180) & (daily_df["glucose"] <= 250)]) / len(daily_df) * 100
        time_above_250 = len(daily_df[daily_df["glucose"] > 250]) / len(daily_df) * 100
        total_time = len(daily_df) * 5 / 60
        time_below_54_hm = f"{int(time_below_54 / 100 * total_time):d}h {int((time_below_54 / 100 * total_time % 1) * 60):02d}m"
        time_54_70_hm = f"{int((time_below_70 - time_below_54) / 100 * total_time):d}h {int(((time_below_70 - time_below_54) / 100 * total_time % 1) * 60):02d}m"
        time_70_180_hm = f"{int(tir_70_180 / 100 * total_time):d}h {int((tir_70_180 / 100 * total_time % 1) * 60):02d}m"
        time_180_250_hm = f"{int(time_180_250 / 100 * total_time):d}h {int((time_180_250 / 100 * total_time % 1) * 60):02d}m"
        time_above_250_hm = f"{int(time_above_250 / 100 * total_time):d}h {int((time_above_250 / 100 * total_time % 1) * 60):02d}m"
        total_carbs = daily_df["carbs"].sum()
        total_bolus = daily_df["bolus_insulin"].sum()
        total_basal = (daily_df["basal_insulin"].mean() * (len(daily_df) * 5 / 60))
        total_insulin = total_bolus + total_basal
        bolus_percent = (total_bolus / total_insulin * 100) if total_insulin > 0 else 0
        basal_percent = (total_basal / total_insulin * 100) if total_insulin > 0 else 0

        colors = {
            "very_low": "#FF0000", "low": "#FF4136", "target": "#2ECC40", 
            "high": "#B10DC9", "very_high": "#6B48FF"
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_df["timestamp"], y=daily_df["glucose"], mode="lines", 
                                 name="Glucose (mg/dL)", line=dict(color="blue")))
        carbs = daily_df[daily_df["carbs"] > 0]
        insulin = daily_df[daily_df["bolus_insulin"] > 0]
        fig.add_trace(go.Scatter(x=carbs["timestamp"], y=carbs["carbs"], mode="markers", 
                                 name="Carbs (g)", yaxis="y2", marker=dict(color="green", size=10)))
        fig.add_trace(go.Scatter(x=insulin["timestamp"], y=insulin["bolus_insulin"], mode="markers", 
                                 name="Bolus Insulin (U)", yaxis="y2", marker=dict(color="purple", size=10)))
        fig.add_trace(go.Scatter(x=daily_df["timestamp"], y=daily_df["basal_insulin"] * 60, 
                                 name="Basal Insulin (U/hour)", yaxis="y2", line=dict(color="orange", dash="dash")))
        fig.update_layout(
            title=f"Detailed View: {date}",
            xaxis_title="Time",
            yaxis=dict(title="Glucose (mg/dL)", range=[0, 300]),
            yaxis2=dict(title="Carbs/Insulin", overlaying="y", side="right", range=[0, 60]),
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            template="plotly_white"
        )

        return html.Div([
            html.Div([
                html.Button("⬅", id="prev-day-diary", n_clicks=0, style={"fontSize": "20px", "width": "50px", "height": "50px", "margin": "5px"}),
                html.Span(date, style={"fontSize": "24px", "margin": "0 20px"}),
                html.Button("➡", id="next-day-diary", n_clicks=0, style={"fontSize": "20px", "width": "50px", "height": "50px", "margin": "5px"})
            ], style={"textAlign": "center", "margin": "10px"}),
            html.Div([
                html.Div([
                    html.Div("eA1c", style={"fontSize": "16px", "color": "#666"}),
                    html.Div(f"{ea1c:.1f}%", style={"fontSize": "24px", "fontWeight": "bold"})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("Average", style={"fontSize": "16px", "color": "#666"}),
                    html.Div(f"{mean_glucose:.0f}mg/dL", style={"fontSize": "24px", "fontWeight": "bold"})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("CV", style={"fontSize": "16px", "color": "#666"}),
                    html.Div(f"{cv:.0f}%", style={"fontSize": "24px", "fontWeight": "bold"})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"})
            ], style={"display": "flex", "justifyContent": "space-around"}),
            html.H3("Time in Range", style={"textAlign": "center", "margin": "10px"}),
            html.Div([
                html.Div([
                    html.Div("Very Low", style={"fontSize": "14px", "color": "#666"}),
                    html.Div("< 54 mg/dL", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(time_below_54_hm, style={"fontSize": "18px", "color": colors["very_low"]})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("Low", style={"fontSize": "14px", "color": "#666"}),
                    html.Div("54 - 70 mg/dL", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(time_54_70_hm, style={"fontSize": "18px", "color": colors["low"]})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("Target", style={"fontSize": "14px", "color": "#666"}),
                    html.Div("70 - 180 mg/dL", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(time_70_180_hm, style={"fontSize": "18px", "color": colors["target"]})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"})
            ], style={"display": "flex", "justifyContent": "space-around"}),
            html.Div([
                html.Div([
                    html.Div("High", style={"fontSize": "14px", "color": "#666"}),
                    html.Div("180 - 250 mg/dL", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(time_180_250_hm, style={"fontSize": "18px", "color": colors["high"]})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("Very High", style={"fontSize": "14px", "color": "#666"}),
                    html.Div("> 250 mg/dL", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(time_above_250_hm, style={"fontSize": "18px", "color": colors["very_high"]})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"})
            ], style={"display": "flex", "justifyContent": "space-around"}),
            html.Div([
                html.Div(style={"width": f"{time_below_54}%", "height": "20px", "backgroundColor": colors["very_low"]}),
                html.Div(style={"width": f"{time_below_70 - time_below_54}%", "height": "20px", "backgroundColor": colors["low"]}),
                html.Div(style={"width": f"{tir_70_180}%", "height": "20px", "backgroundColor": colors["target"]}),
                html.Div(style={"width": f"{time_180_250}%", "height": "20px", "backgroundColor": colors["high"]}),
                html.Div(style={"width": f"{time_above_250}%", "height": "20px", "backgroundColor": colors["very_high"]})
            ], style={"display": "flex", "width": "100%", "margin": "10px 0"}),
            html.H3("Treatments", style={"textAlign": "center", "margin": "10px"}),
            html.Div([
                html.Div([
                    html.Div("Carbs", style={"fontSize": "16px", "color": "#666"}),
                    html.Div(f"{total_carbs:.0f}g", style={"fontSize": "24px", "fontWeight": "bold"})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("Insulin (U)", style={"fontSize": "16px", "color": "#666"}),
                    html.Div(f"{total_bolus:.0f}U", style={"fontSize": "24px", "fontWeight": "bold"}),
                    html.Div(f"({bolus_percent:.0f}%)", style={"fontSize": "14px", "color": "#0074D9"})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"}),
                html.Div([
                    html.Div("Basal Insulin", style={"fontSize": "16px", "color": "#666"}),
                    html.Div(f"{total_basal:.0f}U", style={"fontSize": "24px", "fontWeight": "bold"}),
                    html.Div(f"({basal_percent:.0f}%)", style={"fontSize": "14px", "color": "#0074D9"})
                ], style={"display": "inline-block", "width": "30%", "textAlign": "center", "background": "#F5F5F5", "padding": "10px", "margin": "5px", "borderRadius": "10px"})
            ], style={"display": "flex", "justifyContent": "space-around"}),
            html.H3("Detailed View", style={"textAlign": "center", "margin": "10px"}),
            dcc.Graph(figure=fig)
        ], style={"border": "1px solid #ddd", "padding": "20px", "borderRadius": "10px", "background": "#fff", "maxWidth": "800px", "margin": "0 auto"})
    except Exception as e:
        print(f"Error in create_diary with date {date}: {str(e)}")
        return html.Div(f"Error loading diary content: {str(e)}")

# Insights Generation (enhanced UI for recommendations)
def generate_insights(df, date, hypo_threshold=70, start_date=None, end_date=None, goal_range=None, profile=None):
    try:
        if start_date and end_date:
            date_range = df[(df["timestamp"].dt.date >= pd.to_datetime(start_date).date()) & 
                           (df["timestamp"].dt.date <= pd.to_datetime(end_date).date())].copy()
        else:
            date_range = df[df["timestamp"].dt.date == pd.to_datetime(date).date()].copy()

        if date_range.empty:
            return html.Div("No data for selected date range", className="text-red-600 text-center p-4")

        if profile is None:
            profile = {"age": 30, "weight": 70, "activity_level": "moderate"}
        age, weight, activity_level = profile.get("age", 30), profile.get("weight", 70), profile.get("activity_level", "moderate")
        print(f"Processing insights for date(s): {date if not start_date else f'{start_date} to {end_date}'}, activity_level: {activity_level}, goal_range: {goal_range}")

        mean_glucose = date_range["glucose"].mean()
        total_carbs = date_range["carbs"].sum()
        total_bolus = date_range["bolus_insulin"].sum()
        total_basal = (date_range["basal_insulin"].mean() * (len(date_range) * 5 / 60))
        total_insulin = total_bolus + total_basal

        recommendations = []

        if len(date_range) > 1:
            trend = date_range["glucose"].iloc[-1] - date_range["glucose"].iloc[0] if len(date_range["glucose"]) > 1 else 0
            if trend > 20:
                recommendations.append(html.P("Consider increasing insulin dosage to manage rising glucose levels.", className="text-gray-700"))
            elif trend < -20:
                recommendations.append(html.P("Consider increasing carb intake or reducing insulin to address falling glucose levels.", className="text-gray-700"))

        hypo_events = date_range[date_range["glucose"] < hypo_threshold].shape[0]
        hyper_events = date_range[date_range["glucose"] > 250].shape[0]
        if hypo_events > 0:
            if activity_level == "high":
                carb_adjust = 15
            elif activity_level == "moderate":
                carb_adjust = 20
            else:
                carb_adjust = 25
            recommendations.append(html.P(f"Increase carb intake by ~{carb_adjust-5}-{carb_adjust}g or decrease insulin by ~1-2U to prevent {hypo_events} hypoglycemia (<{hypo_threshold} mg/dL) event(s).", className="text-gray-700"))
        if hyper_events > 0:
            if activity_level == "high":
                insulin_adjust = 1.5
            elif activity_level == "moderate":
                insulin_adjust = 1.0
            else:
                insulin_adjust = 0.5
            recommendations.append(html.P(f"Increase insulin by ~{insulin_adjust}-{insulin_adjust+0.5}U or reduce carb intake by ~15-30g to address {hyper_events} hyperglycemia (>250 mg/dL) event(s).", className="text-gray-700"))

        if not date_range[date_range["carbs"] > 0].empty and not date_range[date_range["bolus_insulin"] > 0].empty:
            carb_glucose_corr = date_range["carbs"].corr(date_range["glucose"].shift(-1)) if len(date_range) > 1 else 0
            insulin_glucose_corr = date_range["bolus_insulin"].corr(date_range["glucose"].shift(-1)) if len(date_range) > 1 else 0
            if abs(carb_glucose_corr) > 0.5:
                recommendations.append(html.P(f"Adjust carb intake based on a correlation of {carb_glucose_corr:.2f} with glucose changes.", className="text-gray-700"))
            if abs(insulin_glucose_corr) > 0.5:
                recommendations.append(html.P(f"Adjust insulin dosage based on a correlation of {insulin_glucose_corr:.2f} with glucose changes.", className="text-gray-700"))

        if mean_glucose < 70:
            recommendations.append(html.P("Increase overall carb intake or reduce insulin to maintain glucose in the 70-180 mg/dL range.", className="text-gray-700"))
        elif mean_glucose > 180:
            recommendations.append(html.P("Increase insulin or decrease carb intake to maintain glucose in the 70-180 mg/dL range.", className="text-gray-700"))

        if start_date and end_date:
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            total_hypo = date_range[date_range["glucose"] < hypo_threshold].shape[0]
            total_hyper = date_range[date_range["glucose"] > 250].shape[0]
            if days > 1:
                recommendations.append(html.P(f"Over {days} days, {total_hypo} hypoglycemia and {total_hyper} hyperglycemia events detected. Consider reviewing insulin or carb adjustments with a healthcare provider.", className="text-gray-700"))

        if goal_range and len(goal_range) == 2 and all(x is not None for x in goal_range):
            goal_min, goal_max = goal_range
            tir_goal = len(date_range[(date_range["glucose"] >= goal_min) & (date_range["glucose"] <= goal_max)]) / len(date_range) * 100
            if tir_goal < 70:
                gap = 70 - tir_goal
                if activity_level == "high":
                    insulin_adjust = 0.3
                    carb_adjust = 5
                elif activity_level == "moderate":
                    insulin_adjust = 0.5
                    carb_adjust = 10
                else:
                    insulin_adjust = 0.7
                    carb_adjust = 15
                recommendations.append(html.P(f"Only {tir_goal:.1f}% time in {goal_min}-{goal_max} mg/dL range. Increase insulin by ~{insulin_adjust}U or reduce carbs by ~{carb_adjust}g to improve by ~{gap:.1f}%.", className="text-gray-700"))

        if len(date_range) > 10:
            X = np.arange(len(date_range["glucose"])).reshape(-1, 1)
            y = date_range["glucose"].values
            model = LinearRegression()
            model.fit(X, y)
            next_time = len(date_range)
            predicted_glucose = model.predict([[next_time]])[0]
            if predicted_glucose < 70:
                recommendations.append(html.P(f"Glucose may drop to ~{predicted_glucose:.0f} mg/dL tomorrow. Consider increasing carbs by ~10g.", className="text-gray-700"))
            elif predicted_glucose > 180:
                recommendations.append(html.P(f"Glucose may rise to ~{predicted_glucose:.0f} mg/dL tomorrow. Consider increasing insulin by ~0.5U.", className="text-gray-700"))

        # Wrap recommendations in styled cards
        recommendation_cards = [
            html.Div([
                html.P("Recommendation:", className="text-sm font-medium text-blue-600"),
                rec
            ], className="bg-gray-100 p-4 rounded-lg mb-2 shadow-sm")
            for rec in recommendations
        ]

        return html.Div([
            html.H3("Personalized Insights", className="text-xl font-semibold text-gray-800 mb-4"),
            html.Div(recommendation_cards, className="space-y-2")
        ], className="p-4 bg-white rounded-lg shadow-lg")
    except Exception as e:
        print(f"Error in generate_insights with date {date}: {str(e)}")
        return html.Div(f"Error generating insights: {str(e)}", className="text-red-600 text-center p-4")

# Dropdown options (unchanged)
unique_dates = [str(date) for date in sorted(df["timestamp"].dt.date.unique())]
default_dates = unique_dates[-5:]
default_detail_date_vis = "2023-11-26"
default_detail_date_diary = "2023-11-26"
hypo_threshold_options = [50, 60, 70, 80]
activity_level_options = ["low", "moderate", "high"]

# Layout (updated Insights tab)
app.layout = html.Div([
    # Include Tailwind CSS
    html.Link(
        href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css",
        rel="stylesheet"
    ),
    html.H1("DiaView: Type 1 Diabetes Visualization", className="text-3xl font-bold text-center text-blue-600 mb-6"),
    dcc.Tabs(id="tabs", value="visualizations", className="border-b border-gray-200", children=[
        dcc.Tab(
            label="Visualizations",
            value="visualizations",
            className="px-4 py-2 text-blue-600 hover:text-blue-800",
            selected_className="border-b-2 border-blue-600",
            children=[
                html.Div([
                    dcc.Graph(id="agp-plot", figure=create_agp_plot(df)),
                    html.Div([
                        html.Div([
                            html.Label("Select Days for Horizon Graphs:"),
                            dcc.Dropdown(id="date-dropdown", options=[{"label": date, "value": date} for date in unique_dates], 
                                         value=default_dates, multi=True),
                            dcc.Graph(id="horizon-graphs"),
                            html.Label("Select Pattern Type:"),
                            dcc.Dropdown(id="pattern-type-dropdown", 
                                         options=[{"label": "Meal", "value": "meal"}, {"label": "Hypoglycemia", "value": "hypoglycemia"}],
                                         value="meal", multi=False),
                            html.Label("Select Day for Pattern:"),
                            dcc.Dropdown(id="detail-date-dropdown-vis", options=[{"label": date, "value": date} for date in unique_dates], 
                                         value=default_detail_date_vis, multi=False)
                        ], style={"width": "80%", "padding": "10px"}),
                        html.Div([
                            html.Button("Export Statistics", id="btn-export-stats"),
                            dcc.Download(id="download-stats"),
                            html.Button("Export Patterns", id="btn-export-patterns"),
                            dcc.Download(id="download-patterns")
                        ], style={"width": "20%", "padding": "10px"})
                    ], style={"display": "flex", "flexDirection": "row"}),
                    dcc.Graph(id="pattern-plot")
                ], style={"padding": "20px"})
            ]
        ),
        dcc.Tab(
            label="Diary",
            value="diary",
            className="px-4 py-2 text-blue-600 hover:text-blue-800",
            selected_className="border-b-2 border-blue-600",
            children=[
                html.Div([
                    html.Label("Select Day for Detailed View:"),
                    dcc.Dropdown(id="detail-date-dropdown-diary", options=[{"label": date, "value": date} for date in unique_dates], 
                                 value=default_detail_date_diary, multi=False),
                    html.Div(id="diary-content"),
                    html.Div([
                        html.Button("⬅", id="prev-day-diary", n_clicks=0, style={"fontSize": "20px", "width": "50px", "height": "50px", "margin": "5px"}),
                        html.Button("➡", id="next-day-diary", n_clicks=0, style={"fontSize": "20px", "width": "50px", "height": "50px", "margin": "5px"})
                    ], style={"textAlign": "center", "margin": "10px"})
                ], style={"padding": "20px"})
            ]
        ),
        dcc.Tab(
            label="Insights",
            value="insights",
            className="px-4 py-2 text-blue-600 hover:text-blue-800",
            selected_className="border-b-2 border-blue-600",
            children=[
                html.Div([
                    html.H2("Personalized Diabetes Insights", className="text-2xl font-bold text-gray-800 mb-6 text-center"),
                    html.Div([
                        html.Label(
                            "Select Day for Insights",
                            className="block text-sm font-medium text-gray-700 mb-2",
                            title="Choose a date to generate personalized insights"
                        ),
                        dcc.Dropdown(
                            id="insight-date-dropdown",
                            options=[{"label": date, "value": date} for date in unique_dates],
                            value=default_detail_date_diary,
                            multi=False,
                            className="w-full p-2 border rounded-md mb-4"
                        ),
                        html.Label(
                            "Select Date Range for Summary",
                            className="block text-sm font-medium text-gray-700 mb-2",
                            title="Choose a range to summarize glucose trends"
                        ),
                        dcc.DatePickerRange(
                            id="date-range-picker",
                            start_date=unique_dates[0],
                            end_date=unique_dates[-1],
                            display_format="YYYY-MM-DD",
                            className="mb-4 w-full"
                        ),
                        html.Label(
                            "Hypoglycemia Threshold (mg/dL)",
                            className="block text-sm font-medium text-gray-700 mb-2",
                            title="Set the glucose level for hypoglycemia detection"
                        ),
                        dcc.Dropdown(
                            id="hypo-threshold-dropdown",
                            options=[{"label": str(th), "value": th} for th in hypo_threshold_options],
                            value=70,
                            multi=False,
                            className="w-full p-2 border rounded-md mb-4"
                        ),
                        html.H3(
                            "Goal Range (mg/dL)",
                            className="text-lg font-semibold text-gray-800 mb-4 text-center"
                        ),
                        html.Div([
                            dcc.Input(
                                id="goal-min",
                                type="number",
                                value=70,
                                min=40,
                                max=300,
                                className="p-2 border rounded-md w-24 mx-2",
                                placeholder="Min"
                            ),
                            dcc.Input(
                                id="goal-max",
                                type="number",
                                value=180,
                                min=40,
                                max=300,
                                className="p-2 border rounded-md w-24 mx-2",
                                placeholder="Max"
                            )
                        ], className="flex justify-center mb-4"),
                        html.H3(
                            "User Profile",
                            className="text-lg font-semibold text-gray-800 mb-4 text-center"
                        ),
                        html.Div([
                            html.Label(
                                "Age",
                                className="block text-sm font-medium text-gray-700 mb-2",
                                title="Enter your age for personalized recommendations"
                            ),
                            dcc.Input(
                                id="age-input",
                                type="number",
                                value=30,
                                min=1,
                                max=120,
                                className="w-full p-2 border rounded-md mb-4"
                            ),
                            html.Label(
                                "Weight (kg)",
                                className="block text-sm font-medium text-gray-700 mb-2",
                                title="Enter your weight for tailored insights"
                            ),
                            dcc.Input(
                                id="weight-input",
                                type="number",
                                value=70,
                                min=1,
                                max=200,
                                className="w-full p-2 border rounded-md mb-4"
                            ),
                            html.Label(
                                "Activity Level",
                                className="block text-sm font-medium text-gray-700 mb-2",
                                title="Select your activity level"
                            ),
                            dcc.Dropdown(
                                id="activity-level-dropdown",
                                options=[{"label": al, "value": al} for al in activity_level_options],
                                value="moderate",
                                className="w-full p-2 border rounded-md mb-4"
                            )
                        ], className="space-y-2"),
                    ], className="w-full md:w-1/2 mx-auto"),
                    html.Div(id="insights-content", className="mt-6")
                ], className="container mx-auto p-4 max-w-4xl")
            ]
        ),
        dcc.Tab(
            label="Learn",
            value="learn",
            className="px-4 py-2 text-blue-600 hover:text-blue-800",
            selected_className="border-b-2 border-blue-600",
            children=[
                html.Div([
                    html.H2("Diabetes Management Tips", style={"textAlign": "center", "margin": "20px"}),
                    html.Div([
                        html.H3("What to Do During Hypoglycemia", style={"margin": "10px"}),
                        html.Ul([
                            html.Li("Check blood sugar if possible; aim for below 70 mg/dL."),
                            html.Li("Consume 15-20g of fast-acting carbs (e.g., glucose tablets, juice, or candy)."),
                            html.Li("Recheck blood sugar after 15 minutes; repeat if still low."),
                            html.Li("If it doesn’t improve, seek medical help or contact a healthcare provider."),
                        ]),
                        html.P(html.A("Learn more at American Diabetes Association", href="https://www.diabetes.org/healthy-living/medication-treatments/blood-glucose-testing-and-control/hypoglycemia", target="_blank")),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("How Insulin Works", style={"margin": "10px"}),
                        html.P("Insulin is a hormone that helps regulate blood glucose by allowing cells to absorb glucose for energy or storage. There are two main types:"),
                        html.Ul([
                            html.Li("Basal insulin: Provides a steady release to manage background glucose levels."),
                            html.Li("Bolus insulin: Taken at meals to handle carbohydrate intake and correct high glucose."),
                        ]),
                        html.P("Dosing depends on factors like carb intake, activity level, and individual sensitivity. Consult your healthcare provider for personalized plans."),
                        html.P(html.A("More details at JDRF", href="https://www.jdrf.org/t1d-resources/about/insulin/", target="_blank")),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("Carbohydrate Counting Basics", style={"margin": "10px"}),
                        html.P("Carb counting helps match insulin to food intake. Key tips:"),
                        html.Ul([
                            html.Li("Read food labels to find total carbs per serving."),
                            html.Li("Use the 15g rule: 1 serving = 15g carbs (e.g., 1 slice of bread)."),
                            html.Li("Adjust insulin based on your carb-to-insulin ratio (consult your doctor)."),
                        ]),
                        html.P(html.A("More at ADA Carb Counting", href="https://www.diabetes.org/healthy-living/recipes-nutrition/eating-well/carb-counting", target="_blank")),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("Exercise and Blood Sugar", style={"margin": "10px"}),
                        html.P("Exercise can lower or raise blood sugar depending on intensity and timing:"),
                        html.Ul([
                            html.Li("Check glucose before exercising; avoid if below 100 mg/dL."),
                            html.Li("Have 15g carbs ready for lows during activity."),
                            html.Li("Monitor for 24 hours post-exercise as levels may drop."),
                        ]),
                        html.P(html.A("Learn more at CDC", href="https://www.cdc.gov/diabetes/managing/active.html", target="_blank")),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("Stress and Glucose Management", style={"margin": "10px"}),
                        html.P("Stress can raise blood sugar by triggering hormone release. Tips:"),
                        html.Ul([
                            html.Li("Practice deep breathing or meditation for 5-10 minutes."),
                            html.Li("Monitor glucose during stressful periods."),
                            html.Li("Adjust insulin if needed with medical guidance."),
                        ]),
                        html.P(html.A("More at NIDDK", href="https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes/stress", target="_blank")),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("Healthy Meal Planning", style={"margin": "10px"}),
                        html.P("Balanced meals help stabilize glucose. Include:"),
                        html.Ul([
                            html.Li("Carbs (e.g., whole grains, 45-60g per meal)."),
                            html.Li("Protein (e.g., chicken, 15-20g)."),
                            html.Li("Healthy fats (e.g., nuts, 1-2 tbsp)."),
                        ]),
                        html.P("Example: Oatmeal with berries and almonds."),
                        html.P(html.A("ADA Meal Planning Guide", href="https://www.diabetes.org/healthy-living/recipes-nutrition/meal-planning", target="_blank")),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("Hypoglycemia Preparedness Checklist", style={"margin": "10px"}),
                        dcc.Checklist(
                            options=[
                                {"label": "I have a glucose meter or CGM", "value": "meter"},
                                {"label": "I carry 15-20g fast-acting carbs", "value": "carbs"},
                                {"label": "I know my low blood sugar signs", "value": "signs"},
                                {"label": "I have an emergency contact plan", "value": "plan"},
                            ],
                            value=[],
                            id="hypo-checklist",
                            labelStyle={"display": "block"}
                        ),
                        html.P("Check off items to assess your readiness. Consult your doctor to address gaps."),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                    html.Div([
                        html.H3("Additional Resources", style={"margin": "10px"}),
                        html.Ul([
                            html.Li(html.A("CDC Diabetes Basics", href="https://www.cdc.gov/diabetes/basics/diabetes.html", target="_blank")),
                            html.Li(html.A("National Institute of Diabetes and Digestive and Kidney Diseases", href="https://www.niddk.nih.gov/health-information/diabetes", target="_blank")),
                        ]),
                    ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px", "margin": "10px"}),
                ], style={"maxWidth": "800px", "margin": "0 auto"})
            ]
        )
    ])
], className="font-sans")

# Callbacks (unchanged)
@app.callback(
    Output("detail-date-dropdown-diary", "value"),
    [Input("prev-day-diary", "n_clicks"), Input("next-day-diary", "n_clicks")],
    State("detail-date-dropdown-diary", "value"),
    State("detail-date-dropdown-diary", "options")
)
def update_date_diary(prev_clicks, next_clicks, current_date, date_options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_date
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    dates = [opt["value"] for opt in date_options]
    current_idx = dates.index(current_date) if current_date in dates else 0
    if trigger_id == "prev-day-diary" and current_idx > 0:
        return dates[current_idx - 1]
    elif trigger_id == "next-day-diary" and current_idx < len(dates) - 1:
        return dates[current_idx + 1]
    return current_date

@app.callback(
    [Output("horizon-graphs", "figure"), Output("pattern-plot", "figure")],
    [Input("date-dropdown", "value"), Input("detail-date-dropdown-vis", "value"), Input("pattern-type-dropdown", "value")]
)
def update_visualizations(selected_dates, detail_date_vis, pattern_type):
    try:
        print(f"update_visualizations called with selected_dates={selected_dates}, detail_date_vis={detail_date_vis}, pattern_type={pattern_type}")
        if not selected_dates or not isinstance(selected_dates, list):
            selected_dates = [detail_date_vis] if detail_date_vis and detail_date_vis in unique_dates else [df["timestamp"].dt.date.max().strftime('%Y-%m-%d')]
        elif not selected_dates:
            selected_dates = [df["timestamp"].dt.date.max().strftime('%Y-%m-%d')]
        print(f"Processed selected_dates={selected_dates}")

        date_filter = df["timestamp"].dt.date.isin([pd.to_datetime(d).date() for d in selected_dates])
        filtered_df = df[date_filter].copy()
        print(f"Filtered_df shape for horizon: {filtered_df.shape}")

        horizon_fig = create_multi_horizon_graphs(filtered_df, selected_dates)
        print("Horizon graphs generated")

        if not detail_date_vis or detail_date_vis not in unique_dates:
            detail_date_vis = df["timestamp"].dt.date.max().strftime('%Y-%m-%d')
        pattern_df = df[df["timestamp"].dt.date == pd.to_datetime(detail_date_vis).date()].copy()
        print(f"Pattern_df shape for {detail_date_vis}: {pattern_df.shape}")

        scaled_series, labels, error = detect_patterns(pattern_df, pattern_type)
        if error:
            print(f"Pattern detection error: {error}")
            pattern_fig = go.Figure().update_layout(title=f"Error: {error}")
        else:
            pattern_fig = plot_patterns(scaled_series, labels, pattern_type)
        print("Pattern plot generated")

        return horizon_fig, pattern_fig
    except Exception as e:
        print(f"Error in update_visualizations callback: {str(e)}")
        return (go.Figure().update_layout(title="Error loading horizon graphs"),
                go.Figure().update_layout(title="Error loading pattern plot"))

@app.callback(
    [Output("diary-content", "children")],
    [Input("detail-date-dropdown-diary", "value"), Input("prev-day-diary", "n_clicks"), Input("next-day-diary", "n_clicks")],
    State("detail-date-dropdown-diary", "value"),
    State("detail-date-dropdown-diary", "options")
)
def update_diary(detail_date_diary, prev_clicks, next_clicks, current_date, date_options):
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_value = current_date
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        dates = [opt["value"] for opt in date_options]
        current_idx = dates.index(current_date) if current_date in dates else 0
        if trigger_id == "prev-day-diary" and current_idx > 0:
            trigger_value = dates[current_idx - 1]
        elif trigger_id == "next-day-diary" and current_idx < len(dates) - 1:
            trigger_value = dates[current_idx + 1]
        else:
            trigger_value = detail_date_diary if detail_date_diary else current_date

    if not trigger_value or trigger_value not in unique_dates:
        trigger_value = df["timestamp"].dt.date.max().strftime('%Y-%m-%d')
    print(f"Using detail_date_diary={trigger_value}")
    diary_content = create_diary(df, trigger_value)
    print("Diary content generated")
    return [diary_content]

@app.callback(
    Output("insights-content", "children"),
    [Input("insight-date-dropdown", "value"), Input("date-range-picker", "start_date"), Input("date-range-picker", "end_date"),
     Input("hypo-threshold-dropdown", "value"), Input("goal-min", "value"), Input("goal-max", "value"),
     Input("age-input", "value"), Input("weight-input", "value"), Input("activity-level-dropdown", "value")]
)
def update_insights(selected_date, start_date, end_date, hypo_threshold, goal_min, goal_max, age, weight, activity_level):
    profile = {"age": age, "weight": weight, "activity_level": activity_level} if all(v is not None for v in [age, weight, activity_level]) else None
    goal_range = [goal_min, goal_max] if all(v is not None for v in [goal_min, goal_max]) else None
    return generate_insights(df, selected_date, hypo_threshold, start_date, end_date, goal_range, profile)

@app.callback(
    Output("download-stats", "data"),
    Input("btn-export-stats", "n_clicks"),
    State("date-dropdown", "value"),
    prevent_initial_call=True
)
def export_stats(n_clicks, selected_dates):
    try:
        date_filter = df["timestamp"].dt.date.isin([pd.to_datetime(d).date() for d in selected_dates])
        filtered_df = df[date_filter]
        stats_df = pd.DataFrame([{
            "Time in Range (70-180 mg/dL) (%)": len(filtered_df[(filtered_df["glucose"] >= 70) & (filtered_df["glucose"] <= 180)]) / len(filtered_df) * 100 if not filtered_df.empty else 0,
            "Time Below 70 mg/dL (%)": len(filtered_df[filtered_df["glucose"] < 70]) / len(filtered_df) * 100 if not filtered_df.empty else 0,
            "Time Below 54 mg/dL (%)": len(filtered_df[filtered_df["glucose"] < 54]) / len(filtered_df) * 100 if not filtered_df.empty else 0,
            "Mean Glucose (mg/dL)": filtered_df["glucose"].mean() if not filtered_df.empty else 0,
            "Estimated A1c (%)": (filtered_df["glucose"].mean() + 46.7) / 28.7 if filtered_df["glucose"].mean() > 0 else 0,
            "Glucose Variability (SD)": filtered_df["glucose"].std() if not filtered_df.empty else 0
        }])
        return dcc.send_data_frame(stats_df.to_csv, "marjorie_statistics.csv")
    except Exception as e:
        print(f"Error in export_stats: {str(e)}")
        return dcc.send_data_frame(pd.DataFrame(), "marjorie_statistics.csv")

@app.callback(
    Output("download-patterns", "data"),
    Input("btn-export-patterns", "n_clicks"),
    State("date-dropdown", "value"),
    State("pattern-type-dropdown", "value"),
    prevent_initial_call=True
)
def export_pattern_data(n_clicks, selected_dates, pattern_type):
    try:
        date_filter = df["timestamp"].dt.date.isin([pd.to_datetime(d).date() for d in selected_dates])
        filtered_df = df[date_filter]
        scaled_series, labels, error = detect_patterns(filtered_df, pattern_type)
        patterns_df = export_patterns(scaled_series, labels, pattern_type)
        return dcc.send_data_frame(patterns_df.to_csv, f"marjorie_{pattern_type}_patterns.csv")
    except Exception as e:
        print(f"Error in export_pattern_data: {str(e)}")
        return dcc.send_data_frame(pd.DataFrame(), f"marjorie_{pattern_type}_patterns.csv")

if __name__ == "__main__":
    app.run(debug=True)