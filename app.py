import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. ENTERPRISE ENGINEERING HUD (UI STYLE)
# ==========================================
st.set_page_config(page_title="Predictive Maintenance HUD", layout="wide")

# Custom Dark Cyberpunk Theme for Mechanical Labs
st.markdown("""
    <style>
    .main { background-color: #030712; color: #f3f4f6; }
    h1, h2, h3 { color: #00ffaa !important; font-family: 'Courier New', monospace; }
    .stNumberInput label { color: #9ca3af !important; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #00ffaa !important; font-family: 'Courier New', monospace; }
    .status-card { padding: 20px; border-radius: 10px; border: 1px solid #1f2937; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ TELEMETRY DIAGNOSTIC COMPONENT BENCHMARKER")
st.subheader("Automated Predictive Maintenance Engine v2.4 (Multi-Model Evaluation)")

# ==========================================
# 2. DATA PIPELINE (PHYSICS SYNTHESIS)
# ==========================================
@st.cache_data
def generate_telemetry_data():
    np.random.seed(42)
    n_samples = 6000
    
    # Simulating standard operational limits of a milling spindle
    air_temp = np.random.normal(298, 3, n_samples)
    process_temp = air_temp + np.random.normal(10, 1.5, n_samples)
    rot_speed = np.random.normal(1500, 200, n_samples)
    torque = np.random.normal(40, 12, n_samples)
    tool_wear = np.random.randint(0, 250, n_samples)
    
    # Establish strict physical failure boundaries (Torque Overload or Thermal Runaway)
    failure = np.zeros(n_samples)
    for i in range(n_samples):
        if torque[i] > 65.5 or air_temp[i] > 305 or tool_wear[i] > 210:
            if np.random.rand() > 0.10:  # 90% true physics failure correlation
                failure[i] = 1
                
    df = pd.DataFrame({
        'Air_Temp_K': air_temp, 'Process_Temp_K': process_temp,
        'Rotational_Speed_RPM': rot_speed, 'Torque_Nm': torque,
        'Tool_Wear_Min': tool_wear, 'Machine_Failure': failure
    })
    return df

df = generate_telemetry_data()
X = df[['Air_Temp_K', 'Process_Temp_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_Min']]
y = df['Machine_Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. CONCURRENT MACHINE LEARNING ENGINES
# ==========================================
@st.cache_resource
def build_virtual_test_benches():
    # Model 1: White-Box Decision Tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    
    # Model 2: Multi-Tree Ensemble Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    
    # Model 3: Linear Feature Boundary (Logistic Regression)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    return dt, rf, lr

dt_bench, rf_bench, lr_bench = build_virtual_test_benches()

def process_metrics(model):
    preds = model.predict(X_test)
    return {
        'Acc': accuracy_score(y_test, preds),
        'Prec': precision_score(y_test, preds, zero_division=0),
        'Rec': recall_score(y_test, preds, zero_division=0)
    }

m_dt = process_metrics(dt_bench)
m_rf = process_metrics(rf_bench)
m_lr = process_metrics(lr_bench)

# ==========================================
# 4. HIGH-TECH PERFORMANCE HUD DISPLAY
# ==========================================
st.write("---")
st.header("📊 ALGORITHM EFFICIENCY MATRIX")
st.write("Real-time telemetry evaluation metrics across independent diagnostic engines.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌲 Single-Node Decision Tree")
    st.metric(label="System Accuracy", value=f"{m_dt['Acc']*100:.2f}%")
    st.markdown(f"**Precision:** {m_dt['Prec']*100:.1f}% | **Recall:** {m_dt['Rec']*100:.1f}%")
    st.caption("Fast execution, follows auditable engineering logic steps.")

with col2:
    st.markdown("### 🚀 Multi-Tree Random Forest (🏆 Top-Rated)")
    st.metric(label="System Accuracy", value=f"{m_rf['Acc']*100:.2f}%")
    st.markdown(f"**Precision:** {m_rf['Prec']*100:.1f}% | **Recall:** {m_rf['Rec']*100:.1f}%")
    st.caption("Ensemble architecture. Extremely robust against sensor anomalies.")

with col3:
    st.markdown("### 📈 Linear Logistic Regression")
    st.metric(label="System Accuracy", value=f"{m_lr['Acc']*100:.2f}%")
    st.markdown(f"**Precision:** {m_lr['Prec']*100:.1f}% | **Recall:** {m_lr['Rec']*100:.1f}%")
    st.caption("Standard gradient boundary. Missing intricate physics intersections.")

# ==========================================
# 5. REAL-TIME SENSOR CONTROL BOARD
# ==========================================
st.write("---")
st.header("🎛️ SIMULATED TELEMETRY CONTROL PANEL")
st.write("Manipulate physical parameters below to execute an immediate stress-test diagnostic across all engines.")

pane1, pane2 = st.columns(2)

with pane1:
    torque = st.number_input("🔬 Torque (Nm)", min_value=-50.0, max_value=600.0, value=42.0, step=0.5)
    air_temp = st.number_input("🌡️ Air Ambient Temperature (K)", min_value=100.0, max_value=500.0, value=298.15, step=0.1)
    process_temp = st.number_input("🔥 Process Tool Temperature (K)", min_value=100.0, max_value=500.0, value=308.15, step=0.1)

with pane2:
    speed = st.number_input("⚙️ Spindle Rotational Speed (RPM)", min_value=0.0, max_value=6000.0, value=1500.0, step=25.0)
    wear = st.number_input("⏳ Cumulative Tool Wear Time (Minutes)", min_value=0, max_value=500, value=45, step=1)

# Format the input telemetry stream
telemetry_stream = np.array([[air_temp, process_temp, speed, torque, wear]])

# ==========================================
# 6. INSTANT TRI-MODEL DIAGNOSIS & VERDICT
# ==========================================
st.write("---")
if st.button("🔴 RUN REAL-TIME LIVE CROSS-MODEL EVALUATION"):
    st.subheader("⚙️ Live Diagnostic Streams")
    
    out1, out2, out3 = st.columns(3)
    
    p_dt = dt_bench.predict(telemetry_stream)[0]
    p_rf = rf_bench.predict(telemetry_stream)[0]
    p_lr = lr_bench.predict(telemetry_stream)[0]
    
    def render_hud_banner(pred):
        if pred == 0:
            return "<div style='color:#00ffaa; font-weight:bold; font-size:20px;'>🟢 SYSTEM CRITICAL STATUS: OPERATIONAL</div>"
        else:
            return "<div style='color:#ff3333; font-weight:bold; font-size:20px;'>🔴 ALERT: STRUCTURAL FAILURE RISK CAUGHT</div>"

    with out1:
        st.markdown(f"<div class='status-card'><b>Decision Tree Node Verdict:</b><br><br>{render_hud_banner(p_dt)}</div>", unsafe_allow_html=True)
    with out2:
        st.markdown(f"<div class='status-card'><b>Random Forest Node Verdict:</b><br><br>{render_hud_banner(p_rf)}</div>", unsafe_allow_html=True)
    with out3:
        st.markdown(f"<div class='status-card'><b>Linear Regression Verdict:</b><br><br>{render_hud_banner(p_lr)}</div>", unsafe_allow_html=True)

    # High-Tech Recommendation Summary Engine
    st.write("### 🧠 Automated Engineering Recommendation:")
    votes = [p_dt, p_rf, p_lr]
    if sum(votes) >= 2:
        st.error("🚨 CRITICAL MAINTENANCE VERDICT: Multi-engine consensus confirms mechanical or thermal parameters have breached threshold boundaries. Schedule structural teardown and component replacement immediately.")
    else:
        st.success("✅ OPTIMAL PERFORMANCE VERDICT: Core telemetry remains within calculated safe physics bounds. Machine can maintain full operational load.")
