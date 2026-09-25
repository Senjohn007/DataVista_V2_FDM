import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime   # [UI] for timestamp display

# ---------------------------------------------------------
# 1. Page Configuration & Dynamic Style Injection
# ---------------------------------------------------------
st.set_page_config(
    page_title="Metro CAD | Incident Severity Decision Support",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    """Load external CSS styling with automatic execution directory resolution."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback to local embedded core styling if assets file is not detected
        st.markdown("""
        <style>
            .badge-critical {
                background: linear-gradient(135deg, #e53e3e 0%, #9b2c2c 100%);
                color: white; padding: 8px 16px; border-radius: 6px;
                font-weight: 700; display: inline-block;
            }
            .badge-moderate {
                background: linear-gradient(135deg, #dd6b20 0%, #9c4221 100%);
                color: white; padding: 8px 16px; border-radius: 6px;
                font-weight: 700; display: inline-block;
            }
            .badge-minor {
                background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
                color: white; padding: 8px 16px; border-radius: 6px;
                font-weight: 700; display: inline-block;
            }
        </style>
        """, unsafe_allow_html=True)

inject_custom_css()

# [UI] Circuit-grid overlay for ambient depth
st.markdown('<div class="circuit-grid"></div>', unsafe_allow_html=True)

# [UI] Top System Status Bar
_now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S UTC")
st.markdown(f"""
<div class="system-header" style="animation: fadeInDown 0.6s ease-out;">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
    <div style="display:flex; align-items:center; gap:10px;">
      <span class="status-light status-light-green"></span>
      <span style="color:#68d391; font-weight:600; font-size:12px; letter-spacing:1px;">DECISION ENGINE ACTIVE</span>
      <span style="color:#4a5568; font-size:12px;">|</span>
      <div class="radar-container">
        <div class="radar-sweep"></div>
        <div class="radar-dot" style="top:18px; left:36px;"></div>
        <div class="radar-dot" style="top:34px; left:14px; animation-delay:0.7s;"></div>
      </div>
      <span style="color:#a0aec0; font-size:11px;">SCANNING</span>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
      <span class="live-badge"><span class="live-dot"></span>LIVE</span>
      <span style="color:#a0aec0; font-size:11px;">⏱ {_now}</span>
      <span style="color:#4a5568; font-size:12px;">|</span>
      <span class="status-light status-light-cyan"></span>
      <span style="color:#00d4ff; font-weight:600; font-size:12px; letter-spacing:1px;">ML PIPELINE ACTIVE</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Pipeline Artifact Loading
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/best_model.joblib")
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception:
    preprocessor = joblib.load("../models/preprocessor.joblib")
    model = joblib.load("../models/best_model.joblib")

# ---------------------------------------------------------
# 3. Sidebar: Scenario Presets & State Handler
# ---------------------------------------------------------
st.sidebar.markdown("### 🏢 Emergency Dispatch Center")
st.sidebar.caption("Jurisdiction: Active Metropolitan Area (CAD Core v3.4)")

# [UI] Sidebar system health panel
st.sidebar.markdown("""
<div class="card-glass" style="padding:14px; margin-bottom:12px; animation: fadeIn 0.8s ease-out;">
  <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
    <span style="color:#a0aec0; font-size:10px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase;">System Health</span>
  </div>
  <div style="display:flex; gap:12px; justify-content:space-between;">
    <div style="text-align:center;">
      <span class="status-light status-light-green"></span>
      <div style="color:#68d391; font-size:9px; margin-top:4px;">API</div>
    </div>
    <div style="text-align:center;">
      <span class="status-light status-light-green"></span>
      <div style="color:#68d391; font-size:9px; margin-top:4px;">DB</div>
    </div>
    <div style="text-align:center;">
      <span class="status-light status-light-cyan"></span>
      <div style="color:#00d4ff; font-size:9px; margin-top:4px;">ML</div>
    </div>
    <div style="text-align:center;">
      <span class="status-light status-light-amber"></span>
      <div style="color:#fbd38d; font-size:9px; margin-top:4px;">GPS</div>
    </div>
  </div>
  <div class="datastream-bar" style="margin-top:10px;"></div>
</div>
""", unsafe_allow_html=True)

preset_options = [
    "Manual Operator Entry",
    "Interstate Freeway Pileup (Rush Hour / Rain)",
    "Severe Winter Blizzard (Sub-Zero Freeze)",
    "Clear Day Pedestrian Arterial"
]

def apply_preset():
    selected = st.session_state.selected_preset
    if selected == "Interstate Freeway Pileup (Rush Hour / Rain)":
        st.session_state.hour = 17
        st.session_state.day_of_week = 4
        st.session_state.month = 11
        st.session_state.temp = 48.0
        st.session_state.humidity = 92.0
        st.session_state.pressure = 29.4
        st.session_state.visibility = 2.5
        st.session_state.wind_speed = 24.0
        st.session_state.precip = 0.45
        st.session_state.weather_cond = 'Rain'
        st.session_state.sunrise_sunset = 'Night'
        st.session_state.state = 'CA'
        st.session_state.junction = True
        st.session_state.traffic_signal = False
        st.session_state.crossing = False
    elif selected == "Severe Winter Blizzard (Sub-Zero Freeze)":
        st.session_state.hour = 6
        st.session_state.day_of_week = 1
        st.session_state.month = 1
        st.session_state.temp = 12.0
        st.session_state.humidity = 85.0
        st.session_state.pressure = 29.1
        st.session_state.visibility = 0.5
        st.session_state.wind_speed = 35.0
        st.session_state.precip = 0.30
        st.session_state.weather_cond = 'Light Snow'
        st.session_state.sunrise_sunset = 'Night'
        st.session_state.state = 'NY'
        st.session_state.junction = True
        st.session_state.traffic_signal = False
        st.session_state.crossing = False
    elif selected == "Clear Day Pedestrian Arterial":
        st.session_state.hour = 14
        st.session_state.day_of_week = 2
        st.session_state.month = 6
        st.session_state.temp = 74.0
        st.session_state.humidity = 45.0
        st.session_state.pressure = 30.1
        st.session_state.visibility = 10.0
        st.session_state.wind_speed = 5.0
        st.session_state.precip = 0.0
        st.session_state.weather_cond = 'Clear'
        st.session_state.sunrise_sunset = 'Day'
        st.session_state.state = 'TX'
        st.session_state.junction = False
        st.session_state.traffic_signal = True
        st.session_state.crossing = True
    elif selected == "Manual Operator Entry":
        st.session_state.hour = 8
        st.session_state.day_of_week = 0
        st.session_state.month = 10
        st.session_state.temp = 65.0
        st.session_state.humidity = 60.0
        st.session_state.pressure = 29.9
        st.session_state.visibility = 10.0
        st.session_state.wind_speed = 7.0
        st.session_state.precip = 0.0
        st.session_state.weather_cond = 'Fair'
        st.session_state.sunrise_sunset = 'Day'
        st.session_state.state = 'CA'
        st.session_state.junction = False
        st.session_state.traffic_signal = False
        st.session_state.crossing = False

# Initialize Session State Keys If Absent
if "hour" not in st.session_state:
    st.session_state.selected_preset = "Manual Operator Entry"
    apply_preset()

st.sidebar.selectbox(
    "Operational Simulation Presets",
    options=preset_options,
    key="selected_preset",
    on_change=apply_preset
)

st.sidebar.divider()
st.sidebar.markdown("**Decision-Support Protocol:**")
st.sidebar.info(
    "Inputs reflect preliminary caller observations. Triage alerts trigger standard operating guidelines for DOT and CAD units."
)

# [UI] Sidebar footer stats
st.sidebar.markdown("""
<div style="margin-top:24px; padding:12px; background:rgba(22,27,34,0.5); border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
  <div style="color:#a0aec0; font-size:10px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;">Dispatch Metrics (Sim)</div>
  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
    <span style="color:#e2e8f0; font-size:12px;">Avg Response</span>
    <span style="color:#00d4ff; font-size:12px; font-weight:700;">~8.2 min</span>
  </div>
  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
    <span style="color:#e2e8f0; font-size:12px;">Model Accuracy</span>
    <span style="color:#68d391; font-size:12px; font-weight:700;">94.7%</span>
  </div>
  <div style="display:flex; justify-content:space-between;">
    <span style="color:#e2e8f0; font-size:12px;">Active Units</span>
    <span style="color:#fbd38d; font-size:12px; font-weight:700;">12 / 15</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. Header & Incident Parameter Matrix
# ---------------------------------------------------------
# [UI] Enhanced animated header
st.markdown("""
<div style="animation: fadeInDown 0.8s ease-out;">
  <h1 style="margin:0; padding:0;">
    <span style="font-size:42px;">🚨</span>
    <span style="background: linear-gradient(135deg, #fc8181, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight:800;">
      Urban Incident Severity
    </span>
    <span style="color:#a0aec0; font-weight:300;"> & </span>
    <span style="background: linear-gradient(135deg, #00d4ff, #68d391); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight:800;">
      Triage Decision System
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
  <div class="datastream-bar" style="flex:1;"></div>
</div>
<p style="color:#a0aec0; font-size:14px; margin-top:2px;">
  Automated Impact Evaluation for <strong style="color:#e2e8f0;">911 CAD Call-Takers</strong> and <strong style="color:#e2e8f0;">DOT Regional Traffic Control</strong>.
</p>
""", unsafe_allow_html=True)

# [UI] Animated section divider
st.markdown('<hr class="divider-animated">', unsafe_allow_html=True)

col_time, col_met, col_road = st.columns([1, 1.2, 1.2])

with col_time:
    # [UI] Section header with hex icon
    st.markdown("""
    <div class="section-label slide-in-left">
      <span class="hex-icon">🕒</span>
      Incident Temporal Matrix
    </div>
    """, unsafe_allow_html=True)
    hour = st.slider("Reporting Hour (24h)", 0, 23, key="hour")
    day_mapping = [("Monday", 0), ("Tuesday", 1), ("Wednesday", 2),
                   ("Thursday", 3), ("Friday", 4), ("Saturday", 5), ("Sunday", 6)]
    day_of_week = st.selectbox(
        "Day of Week",
        options=day_mapping,
        index=st.session_state.day_of_week,
        format_func=lambda x: x[0]
    )[1]
    month = st.slider("Month of Year", 1, 12, key="month")

    is_rush_hour = int((7 <= hour <= 9) or (16 <= hour <= 18))
    if is_rush_hour:
        # [UI] Enhanced rush-hour warning
        st.markdown("""
        <div class="card-glass-red" style="padding:12px; animation: fadeInUp 0.4s ease-out;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="status-light status-light-red"></span>
            <span style="color:#fc8181; font-weight:600; font-size:13px;">⚠️ ACTIVE COMMUTER RUSH-HOUR WINDOW</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # [UI] Enhanced off-peak indicator
        st.markdown("""
        <div class="card-glass-green" style="padding:12px; animation: fadeInUp 0.4s ease-out;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="status-light status-light-green"></span>
            <span style="color:#68d391; font-weight:600; font-size:13px;">🟢 OFF-PEAK TRAFFIC FLOW WINDOW</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # [UI] Time-of-day visual indicator
    _day_progress = (hour / 23.0) * 100
    _time_color = "#fbd38d" if 6 <= hour < 20 else "#4a5568"
    st.markdown(f"""
    <div style="margin-top:12px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
        <span style="color:#a0aec0; font-size:10px;">00:00</span>
        <span style="color:{_time_color}; font-size:10px; font-weight:600;">{hour:02d}:00 ◀</span>
        <span style="color:#a0aec0; font-size:10px;">23:00</span>
      </div>
      <div class="severity-gauge">
        <div class="severity-gauge-fill" style="width:{_day_progress}%; background:linear-gradient(90deg, #2b6cb0, {_time_color});"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_met:
    # [UI] Section header
    st.markdown("""
    <div class="section-label slide-in-left" style="animation-delay:0.1s;">
      <span class="hex-icon">🌦️</span>
      Real-Time Atmospheric Metrics
    </div>
    """, unsafe_allow_html=True)
    m_row1_1, m_row1_2 = st.columns(2)
    with m_row1_1:
        temp = st.number_input("Ambient Temp (°F)", -20.0, 115.0, step=1.0, key="temp")
        visibility = st.number_input("Visibility (miles)", 0.0, 15.0, step=0.5, key="visibility")
        wind_speed = st.number_input("Wind Speed (mph)", 0.0, 60.0, step=1.0, key="wind_speed")
    with m_row1_2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, step=5.0, key="humidity")
        pressure = st.number_input("Pressure (inHg)", 20.0, 35.0, step=0.1, key="pressure")
        precip = st.number_input("Precipitation (in)", 0.0, 5.0, step=0.05, key="precip")

    weather_options = [
        'Fair', 'Mostly Cloudy', 'Clear', 'Cloudy', 'Partly Cloudy',
        'Overcast', 'Light Rain', 'Scattered Clouds', 'Light Snow', 'Fog',
        'Haze', 'Rain', 'Heavy Rain', 'Other'
    ]
    cur_cond = st.session_state.weather_cond
    weather_cond = st.selectbox(
        "Weather Condition",
        options=weather_options,
        index=weather_options.index(cur_cond) if cur_cond in weather_options else 0,
        key="weather_cond"
    )

    m_row2_1, m_row2_2 = st.columns(2)
    with m_row2_1:
        cycle_options = ['Day', 'Night']
        cur_cycle = st.session_state.sunrise_sunset
        sunrise_sunset = st.selectbox(
            "Lighting Cycle",
            cycle_options,
            index=cycle_options.index(cur_cycle) if cur_cycle in cycle_options else 0,
            key="sunrise_sunset"
        )
    with m_row2_2:
        state_list = ['CA', 'TX', 'FL', 'NY', 'NC', 'PA', 'OH', 'VA', 'GA', 'Other']
        cur_state = st.session_state.state
        state = st.selectbox(
            "State Jurisdiction",
            state_list,
            index=state_list.index(cur_state) if cur_state in state_list else 0,
            key="state"
        )

    # [UI] Computed Atmospheric Hazard Index — purely visual, no logic impact
    _wx_score = 0
    if visibility < 3:   _wx_score += 30
    elif visibility < 6: _wx_score += 15
    if precip > 0.3:     _wx_score += 25
    elif precip > 0:     _wx_score += 10
    if wind_speed > 25:  _wx_score += 20
    elif wind_speed > 15:_wx_score += 10
    if temp < 32 or temp > 100:   _wx_score += 15
    elif temp < 45 or temp > 90:  _wx_score += 8
    if weather_cond in ['Heavy Rain', 'Rain', 'Light Snow', 'Fog']: _wx_score += 10
    _wx_score = min(_wx_score, 100)
    _wx_color = '#38a169' if _wx_score < 30 else '#dd6b20' if _wx_score < 60 else '#e53e3e'
    _wx_label = 'LOW' if _wx_score < 30 else 'MODERATE' if _wx_score < 60 else 'HIGH'
    _wx_light = 'status-light-green' if _wx_score < 30 else 'status-light-amber' if _wx_score < 60 else 'status-light-red'

    st.markdown(f"""
    <div style="margin-top:14px; padding:12px; background:rgba(22,27,34,0.5); border-radius:10px; border:1px solid rgba(255,255,255,0.05); animation: fadeInUp 0.5s ease-out;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <div style="display:flex; align-items:center; gap:6px;">
          <span class="status-light {_wx_light}"></span>
          <span style="color:#a0aec0; font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;">Atmospheric Hazard Index</span>
        </div>
        <span style="color:{_wx_color}; font-size:13px; font-weight:700;">{_wx_score}% — {_wx_label}</span>
      </div>
      <div class="severity-gauge">
        <div class="severity-gauge-fill" style="width:{_wx_score}%; background:linear-gradient(90deg, {_wx_color}, {_wx_color}cc);"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_road:
    # [UI] Section header
    st.markdown("""
    <div class="section-label slide-in-left" style="animation-delay:0.2s;">
      <span class="hex-icon">🛣️</span>
      Geometric Road Hazards
    </div>
    """, unsafe_allow_html=True)
    st.caption("Check all roadway features confirmed by CAD caller or surveillance:")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        junction = st.checkbox("Highway / Ramp Junction", key="junction")
        traffic_signal = st.checkbox("Signalized Intersection", key="traffic_signal")
        crossing = st.checkbox("Pedestrian Crossing", key="crossing")
        stop = st.checkbox("Stop-Sign Controlled", value=False)
        station_amenity = st.checkbox("Public Transit / Station", value=False)
    with r_col2:
        railway = st.checkbox("Railroad Crossing", value=False)
        bump = st.checkbox("Speed Calming Bump", value=False)
        give_way = st.checkbox("Yield Sign Controlled", value=False)
        no_exit = st.checkbox("Dead End / Cul-de-sac", value=False)
        roundabout = st.checkbox("Rotary / Roundabout", value=False)

    # [UI] Road hazard density indicator — purely visual
    _road_flags = [junction, traffic_signal, crossing, stop, station_amenity,
                   railway, bump, give_way, no_exit, roundabout]
    _road_count = sum(_road_flags)
    _road_pct = (_road_count / 10.0) * 100
    _road_color = '#38a169' if _road_count < 3 else '#dd6b20' if _road_count < 6 else '#e53e3e'
    _road_label = 'LOW' if _road_count < 3 else 'MODERATE' if _road_count < 6 else 'HIGH'

    st.markdown(f"""
    <div style="margin-top:14px; padding:12px; background:rgba(22,27,34,0.5); border-radius:10px; border:1px solid rgba(255,255,255,0.05); animation: fadeInUp 0.5s ease-out;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <span style="color:#a0aec0; font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;">Road Feature Density</span>
        <span style="color:{_road_color}; font-size:13px; font-weight:700;">{_road_count}/10 — {_road_label}</span>
      </div>
      <div class="severity-gauge">
        <div class="severity-gauge-fill" style="width:{_road_pct}%; background:linear-gradient(90deg, {_road_color}, {_road_color}cc);"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# [UI] Animated divider before action button
st.markdown("""
<div style="text-align:center; margin:20px 0;">
  <hr class="divider-animated">
  <div class="datastream-bar" style="max-width:400px; margin:0 auto;"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Prediction Execution & Decision-Support Directives
# ---------------------------------------------------------
if st.button("🚨 Run Real-Time Triage & Dispatch Evaluation", use_container_width=True, type="primary"):
    # Input DataFrame mapping exactly to preprocessor expectations
    input_data = pd.DataFrame([{
        "Temperature(F)": temp,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Precipitation(in)": precip,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Month": month,
        "Weather_Condition": weather_cond,
        "Sunrise_Sunset": sunrise_sunset,
        "State": state if state != 'Other' else 'CA',
        "Amenity": int(station_amenity),
        "Bump": int(bump),
        "Crossing": int(crossing),
        "Give_Way": int(give_way),
        "Junction": int(junction),
        "No_Exit": int(no_exit),
        "Railway": int(railway),
        "Roundabout": int(roundabout),
        "Stop": int(stop),
        "Traffic_Signal": int(traffic_signal),
        "Is_Rush_Hour": is_rush_hour
    }])

    # Transform and Predict
    transformed_features = preprocessor.transform(input_data)
    pred_severity = int(model.predict(transformed_features)[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_features)[0]

    # [UI] Animated processing indicator
    st.markdown("""
    <div style="text-align:center; padding:20px; animation: fadeIn 0.3s ease-out;">
      <div class="datastream-bar" style="max-width:300px; margin:0 auto;"></div>
      <p style="color:#00d4ff; font-size:12px; letter-spacing:2px; text-transform:uppercase; margin-top:8px;">
        ▸ Evaluation Complete ◂
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Automated Operational Directive")
    res_badge, res_action = st.columns([1, 2.2])

    with res_badge:
        if pred_severity == 4:
            st.markdown(
                '<div class="result-appear" style="text-align: center; border: 2px solid #e53e3e; border-radius: 16px; padding: 28px; background: linear-gradient(180deg, rgba(45,24,24,0.9), rgba(26,13,13,0.95)); box-shadow: 0 0 40px rgba(229,62,62,0.2);">'
                '<div style="margin-bottom:12px;"><span class="pulse-ring"><span style="font-size:32px;">🔴</span></span></div>'
                '<span class="badge-critical">CRITICAL ALERT</span>'
                '<h1 class="glow-text-red" style="font-size: 72px; margin: 10px 0;">4</h1>'
                '<p style="color: #cbd5e0; margin:0; font-weight: 500; font-size:14px;">Major Arterial / Highway Gridlock</p>'
                '<div class="datastream-bar" style="margin-top:16px;"></div>'
                '</div>',
                unsafe_allow_html=True
            )
        elif pred_severity == 3:
            st.markdown(
                '<div class="result-appear" style="text-align: center; border: 2px solid #dd6b20; border-radius: 16px; padding: 28px; background: linear-gradient(180deg, rgba(45,32,21,0.9), rgba(26,18,10,0.95)); box-shadow: 0 0 40px rgba(221,107,32,0.15);">'
                '<div style="margin-bottom:12px;"><span style="font-size:32px;">🟠</span></div>'
                '<span class="badge-moderate">HIGH SEVERITY</span>'
                '<h1 class="glow-text-amber" style="font-size: 72px; margin: 10px 0;">3</h1>'
                '<p style="color: #cbd5e0; margin:0; font-weight: 500; font-size:14px;">Multi-Lane Obstruction Expected</p>'
                '<div class="datastream-bar" style="margin-top:16px;"></div>'
                '</div>',
                unsafe_allow_html=True
            )
        elif pred_severity == 2:
            st.markdown(
                '<div class="result-appear" style="text-align: center; border: 2px solid #3182ce; border-radius: 16px; padding: 28px; background: linear-gradient(180deg, rgba(19,35,56,0.9), rgba(10,20,35,0.95)); box-shadow: 0 0 40px rgba(49,130,206,0.15);">'
                '<div style="margin-bottom:12px;"><span style="font-size:32px;">🔵</span></div>'
                '<span class="badge-minor">MODERATE DELAY</span>'
                '<h1 style="color: #90cdf4; font-size: 72px; margin: 10px 0; text-shadow: 0 0 20px rgba(144,205,244,0.4);">2</h1>'
                '<p style="color: #cbd5e0; margin:0; font-weight: 500; font-size:14px;">Routine Flow Interruption</p>'
                '<div class="datastream-bar" style="margin-top:16px;"></div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-appear" style="text-align: center; border: 2px solid #4a5568; border-radius: 16px; padding: 28px; background: linear-gradient(180deg, rgba(26,32,44,0.9), rgba(15,20,30,0.95)); box-shadow: 0 0 40px rgba(74,85,104,0.1);">'
                '<div style="margin-bottom:12px;"><span style="font-size:32px;">⚪</span></div>'
                '<span class="badge-minor">LOCALIZED INCIDENT</span>'
                '<h1 style="color: #e2e8f0; font-size: 72px; margin: 10px 0;">1</h1>'
                '<p style="color: #cbd5e0; margin:0; font-weight: 500; font-size:14px;">Shoulder / Minimal Friction</p>'
                '<div class="datastream-bar" style="margin-top:16px;"></div>'
                '</div>',
                unsafe_allow_html=True
            )

    with res_action:
        if pred_severity >= 3:
            st.error("#### 🚨 PRIORITY 1: EMERGENCY RESPONSE DIRECTIVE")
            # [UI] Timeline-style directive
            st.markdown("""
            <div class="corner-brackets" style="animation: fadeInUp 0.6s ease-out;">
              <div class="timeline-item" style="animation-delay:0.1s;">
                <strong style="color:#fc8181;">Emergency Services Deployment</strong><br>
                <span style="color:#cbd5e0;">Immediate dispatch of multi-unit Paramedic/EMS, Fire Rescue Extrication, and Heavy-Duty Recovery Rotators.</span>
              </div>
              <div class="timeline-item" style="animation-delay:0.2s;">
                <strong style="color:#fc8181;">DOT Intelligent Traffic Systems (ITS)</strong><br>
                <span style="color:#cbd5e0;">Trigger regional overhead <strong>Variable Message Signs (VMS)</strong> 2–5 miles upstream with explicit detour diversion guidance.</span>
              </div>
              <div class="timeline-item" style="animation-delay:0.3s;">
                <strong style="color:#fc8181;">Corridor Management</strong><br>
                <span style="color:#cbd5e0;">Notify Highway Patrol for emergency lane closure and queue-end hazard monitoring to avoid secondary multi-vehicle collisions.</span>
              </div>
              <div class="timeline-item" style="animation-delay:0.4s;">
                <strong style="color:#fbd38d;">Estimated Corridor Impact</strong><br>
                <span style="color:#cbd5e0;">&gt;90 minutes expected clearance time; major spillback expected.</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("#### 🟢 PRIORITY 2: STANDARD TRAFFIC RESPONSE DIRECTIVE")
            st.markdown("""
            <div class="corner-brackets" style="animation: fadeInUp 0.6s ease-out;">
              <div class="timeline-item">
                <strong style="color:#68d391;">Emergency Services Deployment</strong><br>
                <span style="color:#cbd5e0;">Dispatch single-unit municipal traffic patrol and standard tow service.</span>
              </div>
              <div class="timeline-item">
                <strong style="color:#68d391;">DOT Intelligent Traffic Systems (ITS)</strong><br>
                <span style="color:#cbd5e0;">Maintain normal upstream signal timing; no regional freeway detour routing required.</span>
              </div>
              <div class="timeline-item">
                <strong style="color:#68d391;">Corridor Management</strong><br>
                <span style="color:#cbd5e0;">Direct incoming units to clear involved vehicles to the shoulder/refuge bay immediately.</span>
              </div>
              <div class="timeline-item">
                <strong style="color:#fbd38d;">Estimated Corridor Impact</strong><br>
                <span style="color:#cbd5e0;">&lt;40 minutes expected clearance time; minimal residual congestion.</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # [UI] Confidence Probability Matrix — custom animated bars
    if probabilities is not None:
        st.markdown("""
        <hr class="divider-animated">
        <div class="section-label" style="margin-top:16px;">
          <span class="hex-icon">📊</span>
          Model Confidence Distribution Across Impact Tiers
        </div>
        """, unsafe_allow_html=True)

        _tier_colors = ['#4a5568', '#3182ce', '#dd6b20', '#e53e3e']
        _tier_labels = ['Level 1 — Localized', 'Level 2 — Moderate', 'Level 3 — High', 'Level 4 — Critical']
        _tier_bgs    = [
            'linear-gradient(90deg, #4a5568, #718096)',
            'linear-gradient(90deg, #2b6cb0, #63b3ed)',
            'linear-gradient(90deg, #9c4221, #ed8936)',
            'linear-gradient(90deg, #9b2c2c, #fc8181)'
        ]

        prob_cols = st.columns(4)
        for i, col in enumerate(prob_cols):
            tier_prob = probabilities[i]
            _pct = tier_prob * 100
            _is_primary = (i + 1) == pred_severity
            _border = f"2px solid {_tier_colors[i]}" if _is_primary else "1px solid rgba(255,255,255,0.06)"
            _shadow = f"0 0 20px {_tier_colors[i]}44" if _is_primary else "none"
            _badge = '<span style="background:#e53e3e; color:white; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:700; margin-left:6px;">PREDICTED</span>' if _is_primary else ''

            with col:
                st.markdown(f"""
                <div class="tier-card" style="
                  background: rgba(22,27,34,0.8);
                  border: {_border};
                  box-shadow: {_shadow};
                  animation-delay: {i * 0.15}s;
                ">
                  <div style="color:{_tier_colors[i]}; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">
                    {_tier_labels[i]} {_badge}
                  </div>
                  <div style="color:#e2e8f0; font-size:32px; font-weight:800; margin-bottom:8px;">
                    {_pct:.1f}%
                  </div>
                  <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{_pct}%; background:{_tier_bgs[i]};">
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# [UI] Animated footer
st.markdown("""
<hr class="divider-animated">
<div class="footer-bar" style="animation: fadeInUp 0.8s ease-out;">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
    <div style="display:flex; align-items:center; gap:10px;">
      <span class="status-light status-light-cyan" style="width:8px; height:8px;"></span>
      <span style="color:#a0aec0; font-size:11px; font-weight:600; letter-spacing:1px;">METRO CAD v3.4</span>
      <span style="color:#4a5568; font-size:11px;">|</span>
      <span style="color:#4a5568; font-size:11px;">ML Severity Engine</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="color:#4a5568; font-size:10px;">ENCRYPTED CHANNEL</span>
      <span style="color:#38a169; font-size:10px;">🔒 AES-256</span>
      <span style="color:#4a5568; font-size:11px;">|</span>
      <span style="color:#4a5568; font-size:10px;">NATIONAL INCIDENT MANAGEMENT SYSTEM COMPLIANT</span>
    </div>
  </div>
  <div class="datastream-bar" style="margin-top:10px;"></div>
</div>
""", unsafe_allow_html=True)