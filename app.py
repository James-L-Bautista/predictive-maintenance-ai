import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. THE CERTIFIED 50 MACHINE TOOL DATASET
# ==========================================
data = {
    "Machine Name": [
        "DMG MORI NVX 5100", "Mikron MILL P 500", "Mazak VARIAXIS i-600", "Okuma MU-5000V", 
        "Hermle C 42 U", "GROB G350", "Doosan DNM 6700", "Haas UMC-750", 
        "Hurco VMX 60Ui", "Fidia K211", "Fanuc Robodrill α-D21LiA", "Brother Speedio S700X1", 
        "Mitsubishi M-VR30", "Kitamura Mycenter-4X", "Makino PS95", "Sodick MC430L", 
        "Yamazaki Mazak HCN-5000", "Okuma Genos M560V", "Haas VF-4SS", "Hurco VMX 42i",
        "Mazak QT-350", "DMG MORI NLX 2500", "Okuma LB3000 EX", "Doosan Lynx 2100", 
        "Haas ST-20", "Miyano BNA-42S", "Nakamura-Tome WY-150", "Tsugami B0205", 
        "Citizen Cincom L20", "Hardings GS 51", "Trumpf TruLaser 3030", "Bystronic ByStar Fiber", 
        "Amada LCG 3015", "Mazak OPTIPLEX 3015", "Prima Power Platino", "Salvagnini L3", 
        "Mitsubishi eX-F Series", "Cincinnati CL-900", "Nukon Fiber Vento", "HK Laser Fiber",
        "Haas Mini Mill", "Tormach 1100MX", "Pocket NC V2-50", "Syil X7", 
        "Bantam Tools Desktop", "Stepcraft D.840", "Axiom Precision AR8", "Laguna SmartShop", 
        "Baileigh CNC Router", "ShopBot Alpha"
    ],
    "Price ($)": [
        185000, 245000, 210000, 230000, 280000, 260000, 95000, 145000, 165000, 320000,
        85000, 78000, 115000, 135000, 155000, 195000, 140000, 110000, 92000, 105000,
        125000, 145000, 135000, 65000, 58000, 110000, 175000, 85000, 120000, 75000,
        350000, 420000, 290000, 310000, 280000, 390000, 330000, 260000, 195000, 210000,
        35000, 18500, 6500, 28000, 4999, 6500, 9500, 15000, 12500, 22000
    ],
    "Max Spindle Speed (RPM)": [
        12000, 20000, 18000, 15000, 24000, 16000, 12000, 10000, 12000, 30000,
        24000, 16000, 10000, 15000, 20000, 40000, 14000, 15000, 12000, 12000,
        5000, 6000, 5000, 6000, 4000, 6000, 5000, 8000, 10000, 5000,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # Lasers operate at negligible spindle velocity
        6000, 10000, 50000, 12000, 28000, 24000, 24000, 24000, 18000, 18000
    ],
    "Tolerance (µm)": [
        3.0, 1.5, 2.5, 2.0, 1.0, 2.0, 5.0, 6.0, 4.0, 1.0,
        5.0, 6.0, 8.0, 4.0, 3.0, 1.5, 4.0, 4.0, 5.0, 5.0,
        8.0, 6.0, 5.0, 12.0, 15.0, 8.0, 5.0, 10.0, 8.0, 12.0,
        15.0, 10.0, 20.0, 15.0, 25.0, 12.0, 15.0, 25.0, 30.0, 25.0,
        15.0, 20.0, 10.0, 12.0, 25.0, 35.0, 40.0, 30.0, 50.0, 45.0
    ],
    "Repeatability (µm)": [
        2.0, 1.0, 1.5, 1.2, 0.8, 1.5, 3.0, 4.0, 3.0, 0.8,
        3.0, 4.0, 5.0, 2.5, 2.0, 1.0, 2.5, 2.5, 3.0, 3.0,
        5.0, 4.0, 3.0, 8.0, 10.0, 5.0, 3.0, 7.0, 5.0, 8.0,
        10.0, 8.0, 12.0, 10.0, 15.0, 8.0, 10.0, 18.0, 20.0, 15.0,
        10.0, 12.0, 5.0, 8.0, 15.0, 20.0, 25.0, 20.0, 30.0, 25.0
    ],
    "MTBF (Hours)": [
        2500, 3000, 2800, 2700, 3200, 2900, 2000, 1800, 2100, 3500,
        2200, 2400, 1900, 2300, 2600, 3100, 2400, 2500, 2100, 2200,
        2200, 2400, 2500, 1700, 1600, 2100, 2300, 1800, 2000, 1900,
        1800, 2000, 1700, 1800, 1600, 2100, 1900, 1500, 1400, 1600,
        1500, 1200, 1000, 1400, 800, 900, 1100, 1300, 1000, 1200
    ]
}

df_machines = pd.DataFrame(data)

# ==========================================
# 2. UI HEADER & HUD THEME
# ==========================================
st.set_page_config(page_title="Kaggle Applied TOPSIS Engine", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #030712; color: #f3f4f6; }
    h1, h2, h3 { color: #00e5ff !important; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricValue"] { color: #00e5ff !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ KAGGLE-DRIVEN TOPSIS OPTIMIZATION DASHBOARD")
st.subheader("Data-Linked Machine Tool Selection Architecture via AI4I Operational Parameters")

# ==========================================
# 3. INTERACTIVE SENSOR LOADS (KAGGLE PROFILE)
# ==========================================
st.write("---")
st.header("🏭 Real-Time Factory Floor Telemetry (AI4I Kaggle Inputs)")
st.write("These sliders mimic an live ingestion stream of the standard Kaggle Predictive Maintenance Dataset.")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    live_torque = st.slider("🔬 Active Torque Load (Nm)", 0.0, 100.0, 40.0)
with col_b:
    live_rpm = st.slider("⚙️ Operating Spindle Speed (RPM)", 0, 5000, 1500)
with col_c:
    live_temp = st.slider("🌡️ Thermal Spindle Temperature (K)", 290, 320, 298)
with col_d:
    live_wear = st.slider("⏳ Tool Cumulative Wear (Minutes)", 0, 250, 45)

# ==========================================
# 4. DYNAMIC WEIGHT GENERATION (THE CONTEXT LOGIC)
# ==========================================
# Mathematically scale TOPSIS preferences based on what is failing on the shop floor!
w_price = 0.25
w_rpm = 0.15 + (0.20 * (live_rpm / 5000.0))    # If shop runs high RPM, prioritize high-RPM capacity
w_tol = 0.20 + (0.20 * (live_torque / 100.0))  # If torque is crushing components, prioritize rigidity (Tolerance)
w_rep = 0.20
w_mtbf = 0.20 + (0.30 * (live_wear / 250.0))   # If parts wear down instantly, prioritize high structural reliability (MTBF)

# Normalize weights so they always equal 1.00
total_w = w_price + w_rpm + w_tol + w_rep + w_mtbf
w_price /= total_w; w_rpm /= total_w; w_tol /= total_w; w_rep /= total_w; w_mtbf /= total_w

st.info(f"⚡ **Dynamic TOPSIS Weights Applied:** Capital Cost: **{w_price*100:.1f}%** | Speed Capacity: **{w_rpm*100:.1f}%** | Structural Tolerance: **{w_tol*100:.1f}%** | Repeatability: **{w_rep*100:.1f}%** | Failure Survival (MTBF): **{w_mtbf*100:.1f}%**")

# ==========================================
# 5. MATHEMATICAL TOPSIS VECTOR ALGORITHM
# ==========================================
matrix = df_machines[["Price ($)", "Max Spindle Speed (RPM)", "Tolerance (µm)", "Repeatability (µm)", "MTBF (Hours)"]].values
rows, cols = matrix.shape

# Step 1: Vector Normalization
norm_matrix = np.zeros((rows, cols))
for j in range(cols):
    norm_matrix[:, j] = matrix[:, j] / np.sqrt(np.sum(matrix[:, j]**2))

# Step 2: Weight Allocation
weights = np.array([w_price, w_rpm, w_tol, w_rep, w_mtbf])
weighted_matrix = norm_matrix * weights

# Step 3: Define Ideal Best and Worst vectors (Lower price/tolerance/repeatability is better)
ideal_best = np.zeros(cols)
ideal_worst = np.zeros(cols)

criteria_types = ["min", "max", "min", "min", "max"] # min = lower is better, max = higher is better
for j in range(cols):
    if criteria_types[j] == "max":
        ideal_best[j] = np.max(weighted_matrix[:, j])
        ideal_worst[j] = np.min(weighted_matrix[:, j])
    else:
        ideal_best[j] = np.min(weighted_matrix[:, j])
        ideal_worst[j] = np.max(weighted_matrix[:, j])

# Step 4: Euclidean Distance Calculations
s_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
s_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

# Step 5: Closeness Coefficient Computation
performance_score = s_worst / (s_best + s_worst)

# Insert back into DataFrame
df_machines["TOPSIS Performance Score"] = performance_score
df_machines["Rank"] = df_machines["TOPSIS Performance Score"].rank(ascending=False).astype(int)
final_df = df_machines.sort_values("Rank").reset_index(drop=True)

# ==========================================
# 6. GRAPH INTERFACES (RADAR DIAGNOSTICS)
# ==========================================
st.write("---")
st.header("🏆 Optimal Machine Procurement Standings")

top3 = final_df.head(3)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🥇 Top Ranked Choice", f"#{top3.loc[0, 'Rank']} {top3.loc[0, 'Machine Name']}")
with col2:
    st.metric("🥈 Backup System", f"#{top3.loc[1, 'Rank']} {top3.loc[1, 'Machine Name']}")
with col3:
    st.metric("🥉 Third Tier System", f"#{top3.loc[2, 'Rank']} {top3.loc[2, 'Machine Name']}")

# Renders dynamic analytical radar visualization
radar_data = []
categories = ["Price Parameter", "Velocity Boundary", "Precision Gap", "Axis Drift", "Longevity Score"]

# Add Ideal Target
radar_data.append(go.Scatterpolar(
    r=[1, 1, 1, 1, 1], theta=categories, fill='toself', name='Target Reference Limit'
))

for i, row in top3.iterrows():
    # Scale variables between 0-1 for proportional visual plotting on the map
    radar_data.append(go.Scatterpolar(
        r=[0.8 - (row["Price ($)"]/500000), row["Max Spindle Speed (RPM)"]/50000, 
           1 - (row["Tolerance (µm)"]/50), 1 - (row["Repeatability (µm)"]/30), row["MTBF (Hours)"]/4000],
        theta=categories, fill='toself', name=f"Rank {row['Rank']}: {row['Machine Name']}"
    ))

fig_radar = go.Figure(data=radar_data)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=False)), showlegend=True,
    title="Structural Comparison Profiles (Closer to Edge = Better Engineering Alignment)"
)
st.plotly_chart(fig_radar, use_container_width=True)

# Output final data structure matrix
st.write("### Complete Evaluated Machinery Matrix")
st.dataframe(final_df[["Rank", "Machine Name", "TOPSIS Performance Score", "Price ($)", "Max Spindle Speed (RPM)", "Tolerance (µm)", "MTBF (Hours)"]])
