import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import streamlit.components.v1 as components

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(page_title="Thailand Oil Anomaly Dashboard", layout="wide")

# --- 2. DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Monthly_import.csv")
        df['date'] = pd.to_datetime(df['year_month_CE'])
        df['crude_logdiff'] = np.log(df['crude_oil_ML']).diff()
        df = df.fillna(0)
        
        # --- Model 1: Z-Score ---
        df['rolling_mean'] = df['crude_oil_ML'].rolling(window=24).mean()
        df['rolling_std'] = df['crude_oil_ML'].rolling(window=24).std()
        df['z_score'] = (df['crude_oil_ML'] - df['rolling_mean']) / df['rolling_std']
        df['z_anomaly'] = df['z_score'].apply(lambda x: 1 if abs(x) > 2 else 0)
        
        # --- Model 2: Isolation Forest ---
        model = IsolationForest(contamination=0.07, random_state=42)
        model.fit(df[['crude_oil_ML', 'crude_logdiff']])
        
        # Anomaly Score (0-1) สำหรับพล็อตกราฟที่ 3
        raw_scores = model.decision_function(df[['crude_oil_ML', 'crude_logdiff']])
        df['if_score'] = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        
        # Anomaly Detection (Binary)
        df['if_anomaly'] = model.predict(df[['crude_oil_ML', 'crude_logdiff']])
        df['if_anomaly'] = df['if_anomaly'].apply(lambda x: 1 if x == -1 else 0)
        
        # Decision Threshold
        if df['if_anomaly'].any():
            threshold_value = df[df['if_anomaly'] == 1]['if_score'].min()
        else:
            threshold_value = 0.8
            
        # --- Signal Logic สำหรับกราฟที่ 4 ---
        def get_signal_status(row):
            if row['z_anomaly'] == 1 and row['if_anomaly'] == 1: return 'Both agree'
            if row['z_anomaly'] == 1: return 'Z-Score only'
            if row['if_anomaly'] == 1: return 'IF only'
            return 'Normal'
        
        df['signal_status'] = df.apply(get_signal_status, axis=1)
        df['agreement'] = ((df['z_anomaly'] == 1) & (df['if_anomaly'] == 1)).astype(int)
        
        return df, threshold_value
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), 0.8

df, if_threshold = load_data()

if not df.empty:
    # --- 3. CRISIS DEFINITIONS (For Chart Shading) ---
    crisis_events = [
        {"n": "GFC 2008", "s": "2008-09", "e": "2009-03", "c": "rgba(231, 76, 60, 0.15)"},
        {"n": "Oil Crash 2014-15", "s": "2014-07", "e": "2015-06", "c": "rgba(241, 196, 15, 0.15)"},
        {"n": "COVID-19 2020", "s": "2020-02", "e": "2020-08", "c": "rgba(52, 152, 219, 0.15)"},
        {"n": "Ukraine War 2022", "s": "2022-02", "e": "2022-06", "c": "rgba(46, 204, 113, 0.15)"}
    ]

    st.title("📊 Anomaly Detection Comparison: Z-Score vs Isolation Forest")
    st.markdown("### Thailand Crude Oil Imports | 2008-2023")

    # --- 4. MULTI-LAYER CHART ---
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.35, 0.18, 0.22, 0.25],
        subplot_titles=("A. Crude Oil Import Volume (ML)", 
                        "B. Z-Score Anomaly (Model 1)", 
                        "C. Isolation Forest Score (Model 2)", 
                        "D. Signal Agreement Timeline")
    )

    # --- Row 1: Import Volume ---
    fig.add_trace(go.Scatter(x=df['date'], y=df['crude_oil_ML'], name="Import", line=dict(color='#088F8F', width=2)), row=1, col=1)
    for c in crisis_events:
        fig.add_vrect(x0=c['s'], x1=c['e'], fillcolor=c['c'], line_width=0, layer="below",
                     annotation_text=c['n'], annotation_position="top left",
                     annotation=dict(font_size=10, font_color="#444"), row=1, col=1)

    # --- Row 2: Z-Score ---
    fig.add_hrect(y0=-2, y1=2, fillcolor="rgba(173, 216, 230, 0.25)", line_width=0, layer="below", row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['z_score'], name="Z-Score", line=dict(color='#008080', width=1.5)), row=2, col=1)
    for val, color in [(2, "#E0A030"), (-2, "#E0A030"), (3, "#C06060"), (-3, "#C06060")]:
        fig.add_hline(y=val, line_dash="dash", line_color=color, line_width=1, row=2, col=1)
    z_anom = df[df['z_anomaly'] == 1]
    fig.add_trace(go.Scatter(x=z_anom['date'], y=z_anom['z_score'], mode='markers', 
                             marker=dict(color='#E0A030', size=6), name="Z-Anom"), row=2, col=1)

    # --- Row 3: Isolation Forest  ---
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['if_score'], mode='lines', name="IF Score", 
        line=dict(color='#8860D0', width=1.5), fill='tozeroy', fillcolor='rgba(136, 96, 208, 0.15)'
    ), row=3, col=1)
    fig.add_hline(y=if_threshold, line_dash="dash", line_color="#8860D0", line_width=1, row=3, col=1)
    if_anom = df[df['if_anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=if_anom['date'], y=if_anom['if_score'], mode='markers', name="IF Anom", 
        marker=dict(color='#8860D0', size=8, symbol='diamond')
    ), row=3, col=1)
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=3, col=1)

    # --- Row 4: Signal Timeline ---
    color_map = {
        'Normal': '#408E91',
        'Z-Score only': '#D4A017',
        'IF only': '#8E44AD',
        'Both agree': '#4F7942'
    }
    for status, color in color_map.items():
        mask = df['signal_status'] == status
        fig.add_trace(go.Bar(
            x=df.loc[mask, 'date'], 
            y=[1] * mask.sum(),
            name=status,
            marker_color=color,
            showlegend=True,
            width=2500000000
        ), row=4, col=1)

    fig.update_yaxes(range=[0, 1], showticklabels=False, row=4, col=1)
    fig.update_layout(
        height=950, template="plotly_white", barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(t=100)
    )
    st.plotly_chart(fig, use_container_width=True)

    
    st.subheader("📋 Crisis Detection Scorecard")
    
    crisis_data_final = [
        {
            "event": "GFC 2008",
            "window": "2008-09 — 2009-03",
            "months": 7,
            "z_score": "N/A*",
            "if_score": "0/7 (0%)",
            "both": "0/7 (0%)",
            "rate": "0%",
            "color": "#e74c3c", "bg": "rgba(231,76,60,0.1)"
        },
        {
            "event": "Oil Crash 2014-15",
            "window": "2014-07 — 2015-06",
            "months": 12,
            "z_score": "2/12 (17%)",
            "if_score": "2/12 (17%)",
            "both": "2/12 (17%)",
            "rate": "100%",
            "color": "#27ae60", "bg": "rgba(39,174,96,0.1)"
        },
        {
            "event": "COVID-19 2020",
            "window": "2020-02 — 2020-08",
            "months": 7,
            "z_score": "1/7 (14%)",
            "if_score": "4/7 (57%)",
            "both": "1/7 (14%)",
            "rate": "25%",
            "color": "#f1c40f", "bg": "rgba(241,196,15,0.1)"
        },
        {
            "event": "Ukraine War 2022",
            "window": "2022-02 — 2022-06",
            "months": 5,
            "z_score": "1/5 (20%)",
            "if_score": "2/5 (40%)",
            "both": "1/5 (20%)",
            "rate": "50%",
            "color": "#27ae60", "bg": "rgba(39,174,96,0.1)"
        }
    ]

    table_rows = ""
    for d in crisis_data_final:
        
        z_style = "color:#c0392b;" if d['z_score'] != "N/A*" else "color:grey; font-style:italic;"
        
        if_style = "color:#27ae60;" if d['event'] in ["COVID-19 2020", "Ukraine War 2022"] else "color:#c0392b;"
        both_style = "color:#c0392b;"

        table_rows += f"""
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding:12px; text-align:left;'><b>{d['event']}</b></td>
            <td style='padding:12px; color:grey;'>{d['window']}</td>
            <td style='padding:12px;'>{d['months']}</td>
            <td style='padding:12px; {z_style}'>{d['z_score']}</td>
            <td style='padding:12px; {if_style}'>{d['if_score']}</td>
            <td style='padding:12px; {both_style}'>{d['both']}</td>
            <td style='padding:12px;'>
                <div style='background:{d['bg']}; border:1px solid {d['color']}; color:{d['color']}; 
                padding:4px; border-radius:8px; font-weight:bold; width:60px; margin:auto;'>
                    {d['rate']}
                </div>
            </td>
        </tr>
        """

    html_table = f"""
    <table style='width:100%; border-collapse:collapse; text-align:center; font-family:sans-serif;'>
        <thead><tr style='background:#1e3d43; color:white;'>
            <th style='padding:15px; text-align:left;'>Crisis Event</th>
            <th>Window</th><th>Months</th><th>Z-Score</th><th>Isolation Forest</th><th>Both Agree</th><th>Agreement Rate</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    <p style='font-size:11px; color:grey; margin-top:10px;'>* Z-Score requires 24-month warm-up period</p>
    """
    components.html(html_table, height=350)
else:
    st.error("No data found. Please check your Monthly_import.csv file.")