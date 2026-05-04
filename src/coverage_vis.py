import numpy as np
import plotly.graph_objects as go
from trajectory import LEOSatellite
from channel import NTNChannel

def get_hexagon_vertices(center_x, center_y, radius):
    """Generate vertices for a flat-topped hexagon."""
    angles_deg = np.array([0, 60, 120, 180, 240, 300, 360])
    angles_rad = np.deg2rad(angles_deg)
    x = center_x + radius * np.cos(angles_rad)
    y = center_y + radius * np.sin(angles_rad)
    return x, y

def generate_hex_grid(radius, rows, cols):
    """Generate center coordinates for a hexagon grid."""
    centers = []
    width = radius * 2
    height = radius * np.sqrt(3)
    
    for r in range(rows):
        for c in range(cols):
            x = c * width * 0.75
            y = r * height
            if c % 2 == 1:
                y += height / 2
            centers.append((x, y))
    
    centers = np.array(centers)
    cx, cy = centers[:, 0].mean(), centers[:, 1].mean()
    centers[:, 0] -= cx
    centers[:, 1] -= cy
    
    return centers

def create_coverage_figure(time_t=0.0, steer_az=0.0, steer_el=0.0, freq_ghz=28.0, hopping_config=None):
    """Builds and returns a 3D Plotly figure representing the dynamic satellite coverage."""
    fig = go.Figure()
    
    # Initialize Physics Models
    leo = LEOSatellite(altitude_km=600.0)
    channel = NTNChannel(carrier_frequency=freq_ghz * 1e9)
    
    # Math to Visual Mapping: 600km maps to 12 units. So 1 unit = 50km
    SCALE = 50000.0
    
    # Sat 1 Dynamic Position Calculations
    # Based on trajectory.py calculation (moving along X axis)
    x_real = leo.velocity * time_t
    z_real = leo.altitude 
    
    sat1 = (x_real / SCALE, 0.0, z_real / SCALE)
    sat2 = (6, 0, 12) # Leave Sat 2 static
    
    radius = 1.0
    centers = generate_hex_grid(radius, rows=6, cols=8)
    z = np.zeros(len(centers))
    
    # Dynamic Beam Steering Calculations
    active_beams = []
    colors = ['rgba(255, 69, 0, 0.8)', 'rgba(50, 205, 50, 0.8)', 'rgba(255, 215, 0, 0.8)', 'rgba(255, 20, 147, 0.8)', 'rgba(0, 255, 255, 0.8)', 'rgba(255, 140, 0, 0.8)', 'rgba(138, 43, 226, 0.8)', 'rgba(255, 0, 255, 0.8)']
    
    if hopping_config is None:
        # Manual Steering Mode
        theta_rad = np.deg2rad(steer_el)
        phi_rad = np.deg2rad(steer_az)
        
        r_ground = sat1[2] * np.tan(theta_rad)
        target_x = sat1[0] + r_ground * np.cos(phi_rad)
        target_y = sat1[1] + r_ground * np.sin(phi_rad)
        
        distances = np.sqrt((centers[:, 0] - target_x)**2 + (centers[:, 1] - target_y)**2)
        active_hex_idx = np.argmin(distances)
        active_beams.append({'idx': active_hex_idx, 'color': colors[0], 'beam_num': 1})
    else:
        # Beam Hopping Mode
        granularity = hopping_config.get('time_granularity', 1.0)
        p_len = hopping_config.get('pattern_length', 8)
        num_beams = hopping_config.get('num_beams', 2)
        patterns = hopping_config.get('patterns', [])
        
        hop_index = int(abs(time_t) // granularity) % p_len
        for b in range(num_beams):
            if b < len(patterns):
                pat = patterns[b]
                cell_idx = pat[hop_index % len(pat)] if len(pat) > 0 else 0
                cell_idx = cell_idx % len(centers) # Safety bound
                active_beams.append({'idx': cell_idx, 'color': colors[b % len(colors)], 'beam_num': b + 1})

    # Utility function to compute channel metrics for hovering
    def get_cell_metrics(idx):
        target_real_x = centers[idx][0] * SCALE
        target_real_y = centers[idx][1] * SCALE
        dist_real_m = np.sqrt((x_real - target_real_x)**2 + (0 - target_real_y)**2 + (z_real - 0)**2)
        vector_x = target_real_x - x_real
        v_radial = leo.velocity * (vector_x / dist_real_m)
        fspl = channel.free_space_path_loss(dist_real_m)
        doppler_shift = channel.calculate_doppler_shift(v_radial)
        return dist_real_m, fspl, doppler_shift
    
    # Draw Hexagons
    active_indices = {b['idx']: b for b in active_beams}
    
    for i, (cx, cy) in enumerate(centers):
        x, y = get_hexagon_vertices(cx, cy, radius)
        
        fill_color = "rgba(240, 240, 240, 0.4)"
        line_color = "black"
        width = 1
        hover_text = ""
        
        # Color specific static beams for Sat2
        if i in [36, 44]:
            fill_color = "rgba(138, 43, 226, 0.6)" # Purpleish
        elif i in [39, 47]:
            fill_color = "rgba(30, 144, 255, 0.6)" # Blueish
            
        # Highlight dynamic target for Sat 1
        if i in active_indices:
            beam = active_indices[i]
            fill_color = beam['color']
            line_color = "red" if hopping_config is None else "white"
            width = 3
            dist_real_m, fspl, doppler_shift = get_cell_metrics(i)
            hop_text = ""
            if hopping_config is not None:
                p_len = hopping_config.get('pattern_length', 8)
                granularity = hopping_config.get('time_granularity', 1.0)
                hop_idx_str = int(abs(time_t) // granularity) % p_len
                hop_text = f"Hop Step: {hop_idx_str}<br>Beam: #{beam['beam_num']}<br>"
            hover_text = (f"<b>Live Illuminated Cell #{i}</b><br>"
                          f"{hop_text}"
                          f"Distance: {dist_real_m/1000:.1f} km<br>"
                          f"Free Space Path Loss: {fspl:.2f} dB<br>"
                          f"Doppler Shift: {doppler_shift/1e3:.2f} kHz")
            
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            surfaceaxis=2, 
            surfacecolor=fill_color,
            line=dict(color=line_color, width=width),
            text=hover_text,
            hoverinfo='text' if hover_text else 'skip',
            showlegend=False
        ))
        
    # Gateway Coordinates
    gw1 = (-14, -5, 0)
    gw2 = (14, -5, 0)
    
    fig.add_trace(go.Scatter3d(
        x=[gw1[0], gw2[0]], y=[gw1[1], gw2[1]], z=[gw1[2], gw2[2]],
        mode='markers+text',
        marker=dict(symbol='diamond', size=15, color='gray', line=dict(width=2, color='white')),
        text=["GW1", "GW2"],
        textposition="bottom center",
        name="Gateways"
    ))
    
    # Satellites
    sat_hover = (f"<b>Sat 1 (Dynamic LEO)</b><br>"
                 f"Mission Time: +{time_t}s<br>"
                 f"Altitude: {leo.altitude/1000} km<br>"
                 f"Velocity: {leo.velocity/1000} km/s")
                 
    fig.add_trace(go.Scatter3d(
        x=[sat1[0], sat2[0]], y=[sat1[1], sat2[1]], z=[sat1[2], sat2[2]],
        mode='markers+text',
        marker=dict(symbol='square', size=10, color=['red', '#1f77b4'], line=dict(width=2, color='white')),
        text=["Sat 1", "Sat 2"],
        textposition="top center",
        hovertext=[sat_hover, "Static Reference"],
        hoverinfo='text',
        name="Satellites"
    ))
    
    # Orbit Path (Dashed arc)
    t = np.linspace(-18, 18, 100)
    orbit_z = 12 - (t**2)*0.015
    fig.add_trace(go.Scatter3d(
        x=t, y=np.zeros(100), z=orbit_z,
        mode='lines',
        line=dict(color='gray', dash='dash', width=2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Feeder Links
    fig.add_trace(go.Scatter3d(
        x=[gw1[0], sat1[0]], y=[gw1[1], sat1[1]], z=[gw1[2], sat1[2]],
        mode='lines', line=dict(color='gray', width=3), name="Feeder-Link"
    ))
    fig.add_trace(go.Scatter3d(
        x=[gw2[0], sat2[0]], y=[gw2[1], sat2[1]], z=[gw2[2], sat2[2]],
        mode='lines', line=dict(color='gray', width=3), showlegend=False
    ))
    
    # Active Dynamic Beam Sat 1
    for beam in active_beams:
        fig.add_trace(go.Scatter3d(
            x=[sat1[0], centers[beam['idx']][0]], y=[sat1[1], centers[beam['idx']][1]], z=[sat1[2], 0],
            mode='lines', line=dict(color=beam['color'], width=6), showlegend=False, name=f"Beam {beam['beam_num']}"
        ))
    if hopping_config is None and len(fig.data) > 0:
        fig.data[-1].name = "Active Spot Beam"
        fig.data[-1].showlegend = True

    # Static Beams Sat 2
    target_sat2_purple = centers[44]
    target_sat2_blue = centers[39]
    for beam_color, target in [('rgba(138, 43, 226, 0.8)', target_sat2_purple), 
                               ('rgba(30, 144, 255, 0.8)', target_sat2_blue)]:
        fig.add_trace(go.Scatter3d(
            x=[sat2[0], target[0]], y=[sat2[1], target[1]], z=[sat2[2], 0],
            mode='lines', line=dict(color=beam_color, width=4), showlegend=False
        ))

    # Coverage Area Ellipses
    t_ellipse = np.linspace(0, 2*np.pi, 100)
    # Circle 1 dynamically tracking Sat 1
    ex1 = sat1[0] + 6*np.cos(t_ellipse)
    ey1 = sat1[1] + 6*np.sin(t_ellipse)
    fig.add_trace(go.Scatter3d(
        x=ex1, y=ey1, z=np.zeros(100),
        mode='lines', line=dict(color='rgba(255, 69, 0, 0.4)', width=3, dash='dot'), showlegend=False
    ))
    
    # Circle 2 static under Sat 2
    ex2 = 6 + 6*np.cos(t_ellipse)
    ey2 = 0 + 6*np.sin(t_ellipse)
    fig.add_trace(go.Scatter3d(
        x=ex2, y=ey2, z=np.zeros(100),
        mode='lines', line=dict(color='deepskyblue', width=3, dash='dot'), showlegend=False
    ))
    
    # Layout Config
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-20, 20], showbackground=False, visible=False),
            yaxis=dict(range=[-10, 10], showbackground=False, visible=False),
            zaxis=dict(range=[-1, 14], showbackground=False, visible=False),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.8),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.2),
                eye=dict(x=0, y=-2.5, z=1.2) 
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        title=dict(text=f"<b>Live Beam Illumination (T={time_t}s)</b>", x=0.5, y=0.95),
        height=800,
        legend=dict(x=0.05, y=0.9, bgcolor="rgba(255, 255, 255, 0.5)")
    )
    
    return fig
