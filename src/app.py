import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from antenna import UniformRectangularArray
from coverage_vis import (
    create_coverage_figure,
    create_animated_coverage_figure,
    compute_frame_metrics,
)
from scenario import Scenario, REQUIRED_COLS, OPTIONAL_TARGETING_COLS
# Configure Streamlit Page
st.set_page_config(page_title="NTN Beamforming Dashboard", layout="wide")

st.title("🛰️ NTN Beamforming Interactive Dashboard")
st.markdown("This system computes mathematical beamforming responses and simulates directivity patterns for NTN environments.")

tab_beam, tab_topo, tab_anim = st.tabs([
    "📶 Antenna Beamforming",
    "🌐 Network Topology",
    "🎬 Multi-Sat Animation",
])

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
    
    st.sidebar.markdown("**Hopping Patterns (Cell IDs 0-119)**")
    patterns = []
    # Defaults walk adjacent cells in the 10x12 grid so the hop is visually
    # obvious — each sequence stays inside one local region and ping-pongs.
    defaults = [
        "48, 49, 50, 49",      # row 4, cols 0-2 ping-pong
        "60, 61, 62, 61",      # row 5, cols 0-2
        "55, 56, 57, 56",      # row 4, cols 7-9
        "67, 68, 69, 68",      # row 5, cols 7-9
        "24, 25, 26, 38, 37, 36",  # 2x3 snake
        "29, 30, 31, 43, 42, 41",
        "84, 85, 86, 98, 97, 96",
        "89, 90, 91, 103, 102, 101",
    ]
    for b in range(int(num_beams)):
        def_val = defaults[b % len(defaults)]
        seq_str = st.sidebar.text_input(f"Beam {b+1} Sequence", value=def_val)
        # Parse the string into a list of ints safely, modulo 48 for safety
        try:
            seq = [int(x.strip()) % 120 for x in seq_str.split(',') if x.strip().isdigit()]
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
    fig_topo, metrics_data = create_coverage_figure(time_t=time_t, steer_az=azimuth, steer_el=elevation, freq_ghz=freq_ghz, hopping_config=hopping_config)
    
    if metrics_data:
        active_cells = [str(m['cell_id']) for m in metrics_data]
        if steering_mode == "Beam Hopping" and hopping_config is not None:
            hop_step = int(abs(time_t) // hopping_config['time_granularity']) % hopping_config['pattern_length']
            st.success(f"**📡 Live Network Status** &nbsp;&nbsp;|&nbsp;&nbsp; **Mode:** Beam Hopping &nbsp;&nbsp;|&nbsp;&nbsp; **Current Hop Step:** {hop_step + 1} of {hopping_config['pattern_length']} &nbsp;&nbsp;|&nbsp;&nbsp; **Active Cells:** [{', '.join(active_cells)}]")
        else:
            st.success(f"**📡 Live Network Status** &nbsp;&nbsp;|&nbsp;&nbsp; **Mode:** Manual Steering &nbsp;&nbsp;|&nbsp;&nbsp; **Active Cell:** {active_cells[0] if active_cells else 'None'}")
            
        st.markdown("### 📡 Live Link Metrics")
        cols = st.columns(len(metrics_data))
        for idx, m in enumerate(metrics_data):
            with cols[idx]:
                st.markdown(f"**Beam #{m['beam_num']} (Cell {m['cell_id']})**")
                sub_cols = st.columns(3)
                sub_cols[0].metric("Distance", f"{m['distance_km']:.1f} km")
                sub_cols[1].metric("Path Loss", f"{m['fspl_db']:.2f} dB")
                sub_cols[2].metric("Doppler Shift", f"{m['doppler_khz']:.2f} kHz")

    st.plotly_chart(fig_topo, use_container_width=True)

# ======================================================================
# Tab 3 — Multi-Satellite Animated Scenario (CSV-driven)
# ======================================================================
with tab_anim:
    st.subheader("Multi-Satellite Scenario Player")
    st.markdown(
        "Upload a CSV describing **N satellites over time** and watch the "
        "simulation play as a continuous animation — no slider scrubbing needed. "
        "Each timestep becomes one frame; play / pause / scrub from the controls "
        "below the 3D scene."
    )

    with st.expander("📄 CSV format reference"):
        st.markdown(
            f"""
**Required columns:** `{', '.join(REQUIRED_COLS)}`
**Optional columns:** `name`, `{', '.join(OPTIONAL_TARGETING_COLS)}`

| Column | Type | Required | Meaning |
| --- | --- | :-: | --- |
| `time_sec` | float | ✅ | Simulation time in seconds (can be negative) |
| `sat_id` | int | ✅ | Unique satellite identifier |
| `x_km` | float | ✅ | Ground-track X position in km (East+) |
| `y_km` | float | ✅ | Ground-track Y position in km (North+) |
| `altitude_km` | float | ✅ | Satellite altitude above ground in km |
| `name` | str | ◻︎ | Display name shown on satellite marker |
| `beam_idx` | int | ◻︎ | Beam index (0..N-1) within the satellite. Multiple rows with same `(time_sec, sat_id)` and distinct `beam_idx` encode multi-beam. **Hopping** is encoded by changing `target_cell_id` across time for the same `beam_idx`. |
| `target_cell_id` | int | ◻︎ | Explicit hex cell to point this beam at (0–119) |
| `target_x_km` | float | ◻︎ | Explicit ground X target (used if `target_cell_id` absent) |
| `target_y_km` | float | ◻︎ | Explicit ground Y target |

**Target resolution priority** per beam row:
`target_cell_id` → `(target_x_km, target_y_km)` → nadir (sub-sat point).

Velocity, slant range and radial velocity are derived automatically from
consecutive timesteps (per-beam Doppler is recomputed against each beam's
actual target). All scene parameters — number of satellites, number of
beams per satellite, altitudes, ground tracks, hopping patterns — are
inferred from the data; the figure adapts to whatever you upload.
            """
        )

    col_a, col_b = st.columns([3, 1.2])
    with col_a:
        uploaded = st.file_uploader(
            "Upload scenario CSV",
            type=['csv'],
            help="One row per (timestamp, satellite, beam). See the format expander above.",
        )
    with col_b:
        st.markdown(" "); st.markdown(" ")
        use_sample = st.button(
            "📦 Load bundled sample",
            use_container_width=True,
            help="Pre-generated demo with satellites carrying different numbers of beams.",
        )

    # Persist the user's choice across Streamlit reruns
    if use_sample:
        st.session_state['scenario_source'] = 'sample'
    if uploaded is not None:
        st.session_state['scenario_source'] = 'upload'

    scenario = None
    err = None
    src = st.session_state.get('scenario_source')

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(repo_root, 'data', 'sample_scenario.csv')

    if src == 'upload' and uploaded is not None:
        try:
            scenario = Scenario.from_csv(uploaded)
        except Exception as e:
            err = f"Failed to load uploaded CSV: {e}"
    elif src == 'sample':
        try:
            scenario = Scenario.from_csv(sample_path)
        except Exception as e:
            err = f"Could not load bundled sample at {sample_path}: {e}"

    if err:
        st.error(err)

    if scenario is None:
        st.info(
            "👉 Upload a CSV above, or click **Load bundled sample** to play a "
            "pre-generated scenario. Every parameter — satellite count, "
            "beams per satellite, altitudes, ground tracks, hopping patterns — "
            "is read directly from the CSV; the visualization adapts automatically."
        )
    else:
        t_min, t_max = scenario.time_range()
        n_sats = scenario.num_satellites()
        n_frames = len(scenario.get_timestamps())

        total_beams = scenario.total_beam_slots()
        info_cols = st.columns(5)
        info_cols[0].metric("Satellites", n_sats)
        info_cols[1].metric("Total beam slots", total_beams)
        info_cols[2].metric("Frames", n_frames)
        info_cols[3].metric("Time span", f"{t_min:.0f} → {t_max:.0f} s")
        info_cols[4].metric("Carrier freq", f"{freq_ghz:.1f} GHz")

        # Render the animated 3D figure
        anim_fig = create_animated_coverage_figure(scenario, freq_ghz=freq_ghz)
        st.plotly_chart(anim_fig, use_container_width=True)

        # Per-satellite link metrics table + plots
        with st.expander("📊 Per-satellite link metrics (across all timesteps)"):
            metrics_df = compute_frame_metrics(scenario, freq_ghz=freq_ghz)

            st.markdown("**Summary by satellite**")
            summary = metrics_df.groupby(['sat_id', 'name']).agg(
                min_dist_km=('distance_km', 'min'),
                max_dist_km=('distance_km', 'max'),
                min_fspl_db=('fspl_db', 'min'),
                max_fspl_db=('fspl_db', 'max'),
                min_doppler_khz=('doppler_khz', 'min'),
                max_doppler_khz=('doppler_khz', 'max'),
            ).reset_index()
            st.dataframe(summary, use_container_width=True)

            st.markdown("**Doppler shift over time (per satellite)**")
            doppler_fig = go.Figure()
            for sat_id, sub in metrics_df.groupby('sat_id'):
                doppler_fig.add_trace(go.Scatter(
                    x=sub['time_sec'], y=sub['doppler_khz'],
                    mode='lines', name=str(sub['name'].iloc[0]),
                ))
            doppler_fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Doppler shift (kHz)",
                height=350, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(doppler_fig, use_container_width=True)

            st.markdown("**Path loss over time (per satellite)**")
            fspl_fig = go.Figure()
            for sat_id, sub in metrics_df.groupby('sat_id'):
                fspl_fig.add_trace(go.Scatter(
                    x=sub['time_sec'], y=sub['fspl_db'],
                    mode='lines', name=str(sub['name'].iloc[0]),
                ))
            fspl_fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Free Space Path Loss (dB)",
                height=350, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fspl_fig, use_container_width=True)

            # Download the computed metrics as CSV
            csv_out = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download all-frames link metrics (CSV)",
                data=csv_out,
                file_name="scenario_link_metrics.csv",
                mime="text/csv",
            )
