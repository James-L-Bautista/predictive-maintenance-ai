import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. PAGE SETUP & STYLE
# ==========================================
st.set_page_config(page_title="Machine Model Comparator", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: white; }
    h1, h2, h3 { color: #00ff88 !important; }
    .stButton>button { background-color: #00ff88; color: black; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #00ff88 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Multi-Model Machine Learning Comparator")
st.subheader("Predictive Maintenance Benchmarking & Diagnostics")

# ==========================================
# 2. SIMULATE INDUSTRIAL DATASETS (AI4I 2020 Structure)
# ==========================================
@st.cache_data
def load_industrial_data():
    np.random.seed(42)
    n_samples = 5000
    
    # Simulating core physics-based features
    air_temp = np.random.normal(300, 2, n_samples)
    process_temp = air_temp + np.random.normal(10, 1, n_samples)
    rot_speed = np.random.normal(1500, 150, n_samples)
    torque = np.random.normal(40, 10, n_samples)
    tool_wear = np.random.randint(0, 240, n_samples)
    
    # Logic to create failures (Torque > 65.5 or Temp > 305 or Tool Wear > 200)
    failure = np.zeros(n_samples)
    for i in range(n_samples):
        if torque[i] > 65.5 or air_temp[i] > 305 or tool_wear[i] > 200:
            if np.random.rand() > 0.15: # 85% chance to mark as failure under strain
                failure[i] = 1
                
    df = pd.DataFrame({
        'Air_Temp_K': air_temp,
        'Process_Temp_K': process_temp,
        'Rotational_Speed_RPM': rot_speed,
        'Torque_Nm': torque,
        'Tool_Wear_Min': tool_wear,
        'Machine_Failure': failure
    })
    return df

df = load_industrial_data()

# Prepare Data
X = df[['Air_Temp_K', 'Process_Temp_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_Min']]
y = df['Machine_Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. TRAIN THREE COMPETING MODELS
# ==========================================
@st.cache_resource
def train_models():
    # Model 1: Decision Tree (Our standard baseline)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    
    # Model 2: Random Forest (Ensemble method - typically higher accuracy)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Model 3: Logistic Regression (Linear baseline)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    
    return dt, rf, lr

dt_model, rf_model, lr_model = train_models()

# Evaluate Metrics
def get_metrics(model):
    preds = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1': f1_score(y_test, preds, zero_division=0)
    }

metrics_dt = get_metrics(dt_model)
metrics_rf = get_metrics(rf_model)
metrics_lr = get_metrics(lr_model)

# ==========================================
# 4. DASHBOARD INTERFACE LAYOUT
# ==========================================
st.write("---")
st.header("📊 Model Accuracy Comparison Matrix")
st.write("Evaluate which algorithm handles the data with the highest precision.")

# Showcase metrics in parallel columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌲 Decision Tree")
    st.metric(label="Accuracy", value=f"{metrics_dt['Accuracy']*100:.2f}%")
    st.text(f"Precision: {metrics_dt['Precision']*100:.1f}%\nRecall: {metrics_dt['Recall']*100:.1f}%")
    st.caption("White-Box Auditability. Follows rigid If-Then pathways.")

with col2:
    st.markdown("### 🚀 Random Forest")
    st.metric(label="Accuracy (🏆 Winner)", value=f"{metrics_rf['Accuracy']*100:.2f}%")
    st.text(f"Precision: {metrics_rf['Precision']*100:.1f}%\nRecall: {metrics_rf['Recall']*100:.1f}%")
    st.caption("Ensemble Method. Higher accuracy, protects against outliers.")

with col3:
    st.markdown("### 📈 Logistic Regression")
    st.metric(label="Accuracy", value=f"{metrics_lr['Accuracy']*100:.2f}%")
    st.text(f"Precision: {metrics_lr['Precision']*100:.1f}%\nRecall: {metrics_lr['Recall']*100:.1f}%")
    st.caption("Linear Classification. Fast, but misses non-linear physics relationships.")

# ==========================================
# 5. USER TELEMETRY INPUTS FOR REAL-TIME SIMULATION
# ==========================================
st.write("---")
st.header("⚙️ Real-Time Telemetry Testing")
st.write("Input custom values to test how the 3 different algorithms interpret the risk.")

input_col1, input_col2 = st.columns(2)

with input_col1:
    torque_in = st.number_input("Torque (Nm)", min_value=-50.0, max_value=500.0, value=45.0, step=1.0)
    air_temp_in = st.number_input("Air Temperature (K)", min_value=0.0, max_value=400.0, value=298.0, step=0.5)
    process_temp_in = st.number_input("Process Temperature (K)", min_value=0.0, max_value=400.0, value=308.0, step=0.5)

with input_col2:
    rpm_in = st.number_input("Rotational Speed (RPM)", min_value=0.0, max_value=5000.0, value=1500.0, step=50.0)
    tool_wear_in = st.number_input("Tool Wear (Minutes)", min_value=0, max_value=500, value=60, step=1)

# Format the live input array
user_features = np.array([[air_temp_in, process_temp_in, rpm_in, torque_in, tool_wear_in]])

# ==========================================
# 6. DIVERGENT PREDICTION OUTCOMES
# ==========================================
if st.button("Execute Cross-Model Diagnosis"):
    st.write("### Prediction Results Matrix:")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    def display_status(prediction):
        if prediction[0] == 0:
            return "🟢 HEALTHY"
        else:
            return "🔴 FAILURE LIKELY"

    with res_col1:
        pred_dt = dt_model.predict(user_features)
        st.markdown(f"**Decision Tree Says:**\n### {display_status(pred_dt)}")
        
    with res_col2:
        pred_rf = rf_model.predict(user_features)
        st.markdown(f"**Random Forest Says:**\n### {display_status(pred_rf)}")
        
    with res_col3:
        pred_lr = lr_model.predict(user_features)
        st.markdown(f"**Logistic Regression Says:**\n### {display_status(pred_lr)}")

    st.info("⚠️ **Why are results different?** Linear models (Logistic Regression) look for straight cutoffs. Tree models split on numerical thresholds. Random Forest looks at consensus across 50 separate mini-trees, making it the most robust when you enter extreme values.")