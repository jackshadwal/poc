import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from antenna import UniformRectangularArray
from coverage_vis import create_coverage_figure
# Configure Streamlit Page
st.set_page_config(page_title="NTN Beamforming Dashboard", layout="wide")

st.title("🛰️ NTN Beamforming Interactive Dashboard")
st.markdown("This system computes mathematical beamforming responses and simulates directivity patterns for NTN environments, satisfying strict REQ 1 functional constraints.")

tab_beam, tab_topo = st.tabs(["📶 Antenna Beamforming", "🌐 Network Topology"])

# Sidebar Controls for Full Configuration
st.sidebar.header("1. Antenna Configuration")
num_x = st.sidebar.number_input("Elements (X-Axis)", min_value=2, max_value=32, value=8)
num_y = st.sidebar.number_input("Elements (Y-Axis)", min_value=2, max_value=32, value=8)
freq_ghz = st.sidebar.number_input("Frequency (GHz)", min_value=1.0, max_value=100.0, value=28.0)

st.sidebar.header("2. Steering Mode")
steering_mode = st.sidebar.radio("Select Mode", ["Manual Steering", "Beam Hopping"])

azimuth = 0.0
elevation = 0.0
hopping_config = None

if steering_mode == "Manual Steering":
    st.sidebar.subheader("Manual Controls")
    azimuth = st.sidebar.slider("Azimuth Angle (deg)", -90.0, 90.0, 0.0, 1.0)
    elevation = st.sidebar.slider("Elevation (Deflection from Nadir)", 0.0, 90.0, 0.0, 1.0)
else:
    st.sidebar.subheader("Hopping Configuration")
    num_beams = st.sidebar.number_input("Number of Beams", min_value=1, max_value=8, value=2)
    pattern_length = st.sidebar.number_input("Pattern Length", min_value=1, max_value=32, value=8)
    time_granularity = st.sidebar.number_input("Time Granularity (s/hop)", min_value=0.1, max_value=10.0, value=1.0)
    
    st.sidebar.markdown("**Hopping Patterns (Cell IDs 0-47)**")
    patterns = []
    # Provide sensible defaults
    defaults = [
        "1, 2, 3, 4, 2, 1, 3, 4",
        "5, 7, 6, 8, 7, 7, 7, 7",
        "10, 11, 12, 13, 14, 15, 16, 17",
        "40, 41, 42, 43, 42, 41, 40, 41"
    ]
    for b in range(int(num_beams)):
        def_val = defaults[b % len(defaults)]
        seq_str = st.sidebar.text_input(f"Beam {b+1} Sequence", value=def_val)
        # Parse the string into a list of ints safely, modulo 48 for safety
        try:
            seq = [int(x.strip()) % 48 for x in seq_str.split(',') if x.strip().isdigit()]
            if len(seq) == 0: seq = [0]
        except Exception:
            seq = [0]
        patterns.append(seq)
        
    hopping_config = {
        "num_beams": int(num_beams),
        "pattern_length": int(pattern_length),
        "time_granularity": float(time_granularity),
        "patterns": patterns
    }

st.sidebar.header("3. Satellite Trajectory")
time_t = st.sidebar.slider("Mission Time (s)", -120.0, 120.0, 0.0, 1.0)

# Initialize Antenna Mathematics dynamically based on UI inputs
fc = freq_ghz * 1e9 
array = UniformRectangularArray(int(num_x), int(num_y), fc)

# Calculate Array Factor over a spherical grid for 3D
phi = np.linspace(-np.pi, np.pi, 45)
theta = np.linspace(0, np.pi/2, 45)
Phi, Theta = np.meshgrid(phi, theta)
AF_mag_3d = array.calculate_array_factor(Theta, Phi, azimuth, elevation)

# Convert to Cartesian for 3D plotting
X = AF_mag_3d * np.sin(Theta) * np.cos(Phi)
Y = AF_mag_3d * np.sin(Theta) * np.sin(Phi)
Z = AF_mag_3d * np.cos(Theta)

# Calculate distinct 2D Frontal cut
phi_1d = np.linspace(-np.pi, np.pi, 360)
theta_1d = np.full_like(phi_1d, np.pi/2) # Simplified Azimuth Horizon cut
AF_mag_2d = array.calculate_array_factor(np.array([theta_1d]), np.array([phi_1d]), azimuth, elevation).flatten()

# Split page into layout columns
with tab_beam:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("3D Volume Visualization")
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(
            autosize=False, height=600, margin=dict(l=0, r=0, b=0, t=20),
            scene=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[0, 1]))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("2D Azimuth Pattern")
        fig_2d = go.Figure(go.Scatterpolar(r=AF_mag_2d, theta=np.degrees(phi_1d), mode='lines'))
        fig_2d.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_2d, use_container_width=True)
        
        st.subheader("Export Results Data")
        df = pd.DataFrame({"Azimuth_Deg": np.degrees(phi_1d), "Gain_Normalized": AF_mag_2d})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download 2D Pattern (CSV)", data=csv, file_name="pattern_export.csv", mime="text/csv")

with tab_topo:
    st.subheader("Satellite Spot Beam Coverage Architecture")
    fig_topo = create_coverage_figure(time_t=time_t, steer_az=azimuth, steer_el=elevation, freq_ghz=freq_ghz, hopping_config=hopping_config)
    st.plotly_chart(fig_topo, use_container_width=True)
