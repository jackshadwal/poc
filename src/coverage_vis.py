import numpy as np
import plotly.graph_objects as go
from trajectory import LEOSatellite
from channel import NTNChannel

# Display scale used by both static and animated figures.
# 1 display unit = 50 km of real distance.
SCALE_KM_PER_UNIT = 50.0

# Color palette used to distinguish satellites in the animated view.
SAT_COLORS = [
    'rgba(255, 69, 0, 0.95)',    # 1 — orange-red
    'rgba(30, 144, 255, 0.95)',  # 2 — dodger blue
    'rgba(50, 205, 50, 0.95)',   # 3 — lime green
    'rgba(255, 215, 0, 0.95)',   # 4 — gold
    'rgba(255, 20, 147, 0.95)',  # 5 — deep pink
    'rgba(138, 43, 226, 0.95)',  # 6 — purple
    'rgba(0, 255, 255, 0.95)',   # 7 — cyan
    'rgba(255, 140, 0, 0.95)',   # 8 — dark orange
]

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
    centers = generate_hex_grid(radius, rows=10, cols=12)
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
    metrics_data = []
    
    for i, (cx, cy) in enumerate(centers):
        x, y = get_hexagon_vertices(cx, cy, radius)
        
        fill_color = "rgba(240, 240, 240, 0.4)"
        line_color = "black"
        width = 1
        hover_text = ""
        
        # Color specific static beams for Sat2 (cells near its sub-sat point
        # at display x=6 in the new 10x12 grid)
        if i in [58, 70]:
            fill_color = "rgba(138, 43, 226, 0.6)" # Purpleish
        elif i in [59, 71]:
            fill_color = "rgba(30, 144, 255, 0.6)" # Blueish
            
        # Highlight dynamic target for Sat 1
        if i in active_indices:
            beam = active_indices[i]
            fill_color = beam['color']
            line_color = "red" if hopping_config is None else "white"
            width = 3
            dist_real_m, fspl, doppler_shift = get_cell_metrics(i)
            metrics_data.append({
                'beam_num': beam.get('beam_num', 1),
                'cell_id': i,
                'distance_km': dist_real_m / 1000,
                'fspl_db': fspl,
                'doppler_khz': doppler_shift / 1e3
            })
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

    # Static Beams Sat 2 (cells near display x=6 in the new 10x12 grid)
    target_sat2_purple = centers[70]
    target_sat2_blue = centers[71]
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
            xaxis=dict(range=[-22, 22], showbackground=False, visible=False),
            yaxis=dict(range=[-13, 13], showbackground=False, visible=False),
            zaxis=dict(range=[-1, 14], showbackground=False, visible=False),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1.2, z=0.8),
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
    
    return fig, metrics_data


# ======================================================================
# Animated multi-satellite figure (data-driven)
# ======================================================================

def _build_static_traces(scenario):
    """
    Build the traces that are identical across all animation frames:
    hex grid outlines, gateways, and per-satellite orbital paths.
    """
    hex_radius = 1.0
    centers = generate_hex_grid(hex_radius, rows=10, cols=12)

    # All hex outlines combined into a single trace using None separators
    hex_x, hex_y, hex_z = [], [], []
    for cx, cy in centers:
        xs, ys = get_hexagon_vertices(cx, cy, hex_radius)
        hex_x.extend(xs.tolist() + [None])
        hex_y.extend(ys.tolist() + [None])
        hex_z.extend([0.0] * len(xs) + [None])

    hex_trace = go.Scatter3d(
        x=hex_x, y=hex_y, z=hex_z, mode='lines',
        line=dict(color='rgba(120,120,120,0.55)', width=1),
        hoverinfo='skip', showlegend=False, name='Ground Cells',
    )

    # Gateways
    gw1 = (-14, -5, 0)
    gw2 = (14, -5, 0)
    gateway_trace = go.Scatter3d(
        x=[gw1[0], gw2[0]], y=[gw1[1], gw2[1]], z=[gw1[2], gw2[2]],
        mode='markers+text',
        marker=dict(symbol='diamond', size=12, color='gray',
                    line=dict(width=2, color='white')),
        text=['GW1', 'GW2'], textposition='bottom center',
        name='Gateways', hoverinfo='text',
    )

    # Orbital paths — one combined trace, all sats, with None separators
    orbit_x, orbit_y, orbit_z = [], [], []
    for sat_id in scenario.get_satellite_ids():
        sat_df = scenario.df[scenario.df['sat_id'] == sat_id].sort_values('time_sec')
        orbit_x.extend((sat_df['x_km'] / SCALE_KM_PER_UNIT).tolist() + [None])
        orbit_y.extend((sat_df['y_km'] / SCALE_KM_PER_UNIT).tolist() + [None])
        orbit_z.extend((sat_df['altitude_km'] / SCALE_KM_PER_UNIT).tolist() + [None])

    orbit_trace = go.Scatter3d(
        x=orbit_x, y=orbit_y, z=orbit_z, mode='lines',
        line=dict(color='rgba(150,150,150,0.45)', dash='dot', width=2),
        hoverinfo='skip', showlegend=False, name='Orbital Paths',
    )

    return [hex_trace, gateway_trace, orbit_trace], centers


def _resolve_beam_target(beam_row, sub_x_unit, sub_y_unit, centers):
    """
    Resolve a beam's ground target.
    Priority: target_cell_id > (target_x_km, target_y_km) > nadir.
    Returns (cell_idx_or_None, target_x_unit, target_y_unit).
    """
    n_cells = len(centers)

    # Priority 1: explicit cell ID
    cell_id = beam_row.get('target_cell_id', None) if hasattr(beam_row, 'get') else None
    try:
        if cell_id is not None and not (isinstance(cell_id, float) and np.isnan(cell_id)):
            cid = int(cell_id) % n_cells
            tx, ty = centers[cid]
            return cid, float(tx), float(ty)
    except (TypeError, ValueError):
        pass

    # Priority 2: explicit ground target in km
    tx_km = beam_row.get('target_x_km', None) if hasattr(beam_row, 'get') else None
    ty_km = beam_row.get('target_y_km', None) if hasattr(beam_row, 'get') else None
    try:
        if (tx_km is not None and ty_km is not None
                and not (isinstance(tx_km, float) and np.isnan(tx_km))
                and not (isinstance(ty_km, float) and np.isnan(ty_km))):
            tx_u = float(tx_km) / SCALE_KM_PER_UNIT
            ty_u = float(ty_km) / SCALE_KM_PER_UNIT
            # Find closest cell for the active-cell marker
            dists = np.sqrt((centers[:, 0] - tx_u) ** 2 + (centers[:, 1] - ty_u) ** 2)
            cid = int(np.argmin(dists))
            return cid, tx_u, ty_u
    except (TypeError, ValueError):
        pass

    # Priority 3: nadir — closest hex to sub-sat point
    dists = np.sqrt((centers[:, 0] - sub_x_unit) ** 2 + (centers[:, 1] - sub_y_unit) ** 2)
    cid = int(np.argmin(dists))
    tx, ty = centers[cid]
    return cid, float(tx), float(ty)


def _build_dynamic_traces(scenario, t, centers, channel, sat_names,
                          sat_ids, beam_slot_map):
    """
    Build the per-frame traces. Trace count is fixed across frames so that
    Plotly's frame protocol works cleanly. Schema (in order):
        - 1 trace : satellite markers (all sats in one trace)
        - 1 trace : active-cell markers (one point per active beam target)
        - N traces: per-satellite footprint ellipses
        - B traces: per-(sat, beam_slot) beam lines, where B = total beam slots
    `beam_slot_map[sat_id]` -> number of beam slots reserved for that sat.
    """
    state = scenario.get_state_at(t)
    ellipse_t = np.linspace(0, 2 * np.pi, 50)
    footprint_radius = 4.0  # display units → ~200 km footprint

    # Per-satellite scratch
    sx_arr, sy_arr, sz_arr = [], [], []
    sat_labels, sat_hover, sat_marker_colors = [], [], []
    ellipse_traces = []

    # Per-beam scratch
    active_cx, active_cy, active_cz = [], [], []
    active_marker_colors = []

    # Beam line traces will be appended in (sat_id, beam_slot) order
    beam_line_traces = []

    state_by_sat = {int(sid): grp for sid, grp in state.groupby('sat_id')}

    for slot, sat_id in enumerate(sat_ids):
        base_color = SAT_COLORS[slot % len(SAT_COLORS)]
        name = sat_names.get(sat_id, f'Sat-{sat_id}')
        n_beam_slots = beam_slot_map.get(sat_id, 1)

        sat_rows = state_by_sat.get(sat_id)
        if sat_rows is None or len(sat_rows) == 0:
            # Empty placeholders to preserve trace structure
            ellipse_traces.append(go.Scatter3d(
                x=[], y=[], z=[], mode='lines',
                line=dict(color=base_color, width=2, dash='dot'),
                hoverinfo='skip', showlegend=False, name=f'{name} footprint',
            ))
            for b in range(n_beam_slots):
                beam_line_traces.append(go.Scatter3d(
                    x=[], y=[], z=[], mode='lines',
                    line=dict(color=base_color, width=5),
                    hoverinfo='skip', showlegend=False,
                    name=f'{name} beam {b}',
                ))
            continue

        # Position is consistent across beam rows — read it once
        first = sat_rows.iloc[0]
        sx = float(first['x_km']) / SCALE_KM_PER_UNIT
        sy = float(first['y_km']) / SCALE_KM_PER_UNIT
        sz = float(first['altitude_km']) / SCALE_KM_PER_UNIT
        sx_arr.append(sx); sy_arr.append(sy); sz_arr.append(sz)
        sat_labels.append(name)
        sat_marker_colors.append(base_color)

        # Coverage footprint ring (one per sat)
        ex = sx + footprint_radius * np.cos(ellipse_t)
        ey = sy + footprint_radius * np.sin(ellipse_t)
        ellipse_traces.append(go.Scatter3d(
            x=ex.tolist(), y=ey.tolist(), z=[0.0] * len(ex),
            mode='lines',
            line=dict(color=base_color, width=2, dash='dot'),
            hoverinfo='skip', showlegend=False, name=f'{name} footprint',
        ))

        # Hover summary for sat marker — aggregates beams
        beam_summary_lines = []

        # Build beam traces: iterate from 0 to n_beam_slots, filling the slot
        # if there's a row with matching beam_idx, else leaving it empty.
        rows_by_beam = {int(r['beam_idx']): r for _, r in sat_rows.iterrows()}
        for b in range(n_beam_slots):
            beam_color = _beam_shade(base_color, b, n_beam_slots)
            if b not in rows_by_beam:
                beam_line_traces.append(go.Scatter3d(
                    x=[], y=[], z=[], mode='lines',
                    line=dict(color=beam_color, width=5),
                    hoverinfo='skip', showlegend=False,
                    name=f'{name} beam {b}',
                ))
                continue

            row = rows_by_beam[b]
            cell_idx, tx_u, ty_u = _resolve_beam_target(row, sx, sy, centers)
            target_cell_real = np.array(
                [tx_u * SCALE_KM_PER_UNIT, ty_u * SCALE_KM_PER_UNIT, 0.0]
            ) * 1000.0
            sat_real_xyz = np.array(
                [float(row['x_km']), float(row['y_km']), float(row['altitude_km'])]
            ) * 1000.0
            sat_to_target = target_cell_real - sat_real_xyz
            slant_m = float(np.linalg.norm(sat_to_target))
            fspl_db = channel.free_space_path_loss(slant_m)

            # Per-beam Doppler: project sat velocity onto sat→target LOS.
            # v vector in m/s (ground-plane only; altitude is constant).
            v_xyz = np.array([float(row['vx_kms']) * 1000.0,
                              float(row['vy_kms']) * 1000.0, 0.0])
            los_unit = sat_to_target / (slant_m if slant_m > 0 else 1.0)
            # Closing-speed convention: positive when approaching target
            v_r_mps = -float(np.dot(v_xyz, los_unit))
            doppler_hz = channel.calculate_doppler_shift(v_r_mps)

            beam_line_traces.append(go.Scatter3d(
                x=[sx, tx_u], y=[sy, ty_u], z=[sz, 0.0],
                mode='lines',
                line=dict(color=beam_color, width=5),
                hoverinfo='skip', showlegend=False,
                name=f'{name} beam {b}',
            ))

            active_cx.append(tx_u); active_cy.append(ty_u); active_cz.append(0.05)
            active_marker_colors.append(beam_color)

            beam_summary_lines.append(
                f"  Beam {b} → Cell {cell_idx}: "
                f"{slant_m/1000:.0f} km, "
                f"{fspl_db:.1f} dB, "
                f"{doppler_hz/1e3:+.1f} kHz"
            )

        hover = (
            f"<b>{name}</b> (Sat {sat_id})<br>"
            f"T = {t:.0f} s<br>"
            f"Altitude: {float(first['altitude_km']):.0f} km<br>"
            f"Ground pos: ({float(first['x_km']):.0f}, {float(first['y_km']):.0f}) km<br>"
            f"Speed: {float(first['speed_kms']):.2f} km/s<br>"
            f"Active beams: {len(beam_summary_lines)}<br>"
            + "<br>".join(beam_summary_lines)
        )
        sat_hover.append(hover)

    sat_marker_trace = go.Scatter3d(
        x=sx_arr, y=sy_arr, z=sz_arr,
        mode='markers+text',
        marker=dict(symbol='square', size=11,
                    color=sat_marker_colors,
                    line=dict(width=2, color='white')),
        text=sat_labels, textposition='top center',
        hovertext=sat_hover, hoverinfo='text',
        name='Satellites', showlegend=False,
    )

    active_cell_trace = go.Scatter3d(
        x=active_cx, y=active_cy, z=active_cz,
        mode='markers',
        marker=dict(symbol='circle', size=13,
                    color=active_marker_colors,
                    line=dict(width=2, color='white')),
        hoverinfo='skip', showlegend=False, name='Active Cells',
    )

    return [sat_marker_trace, active_cell_trace] + ellipse_traces + beam_line_traces


def _beam_shade(base_rgba: str, beam_idx: int, total_beams: int) -> str:
    """Return a lighter variant of base_rgba for higher beam indices."""
    # Parse rgba(r,g,b,a) → tuple
    try:
        inside = base_rgba[base_rgba.index('(') + 1: base_rgba.rindex(')')]
        parts = [p.strip() for p in inside.split(',')]
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a = float(parts[3]) if len(parts) > 3 else 1.0
    except Exception:
        return base_rgba
    if total_beams <= 1:
        return base_rgba
    # Lighten toward white as beam_idx grows
    blend = beam_idx / max(total_beams - 1, 1) * 0.55
    r = int(r + (255 - r) * blend)
    g = int(g + (255 - g) * blend)
    b = int(b + (255 - b) * blend)
    return f'rgba({r},{g},{b},{a:.2f})'


def create_animated_coverage_figure(scenario, freq_ghz: float = 28.0,
                                     frame_duration_ms: int = 120):
    """
    Build an animated 3D Plotly figure from a Scenario object.

    Each timestamp in the scenario becomes one animation frame. The figure
    comes with native play/pause controls and a scrubber slider — no
    Streamlit reruns required during playback.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    channel = NTNChannel(carrier_frequency=freq_ghz * 1e9)
    sat_names = scenario.get_satellite_names()
    sat_ids = scenario.get_satellite_ids()
    timestamps = scenario.get_timestamps()
    if not timestamps:
        raise ValueError("Scenario contains no timestamps")

    static_traces, centers = _build_static_traces(scenario)
    beam_slot_map = scenario.max_beams_per_satellite()

    # Build frames
    frames = []
    for t in timestamps:
        dyn = _build_dynamic_traces(scenario, t, centers, channel, sat_names,
                                     sat_ids, beam_slot_map)
        frames.append(go.Frame(
            data=static_traces + dyn,
            name=f"{t:.0f}",
        ))

    # Initial figure = first frame's traces
    fig = go.Figure(data=frames[0].data, frames=frames)

    # Layout: animation buttons + slider. Sized large so individual beams,
    # cells and satellite markers are clearly readable at presentation distance.
    # Scene extents accommodate the 10x12 hex grid plus ground tracks
    # reaching ±1140 km (≈ ±22.8 display units).
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-25, 25], showbackground=False, visible=False),
            yaxis=dict(range=[-25, 25], showbackground=False, visible=False),
            zaxis=dict(range=[-1, 26], showbackground=False, visible=False),
            aspectmode='manual',
            aspectratio=dict(x=1.7, y=1.7, z=0.9),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.12),
                eye=dict(x=1.35, y=-1.7, z=1.0),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=dict(
            text=(f"<b>Multi-Satellite Animated Scenario</b> &nbsp;|&nbsp; "
                  f"{scenario.num_satellites()} satellites &nbsp;|&nbsp; "
                  f"{scenario.total_beam_slots()} beams &nbsp;|&nbsp; "
                  f"{len(timestamps)} frames"),
            x=0.5, y=0.98,
            font=dict(size=18),
        ),
        height=950,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x': 0.05, 'y': 0.05, 'xanchor': 'left', 'yanchor': 'bottom',
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': frame_duration_ms, 'redraw': True},
                        'transition': {'duration': 0},
                        'fromcurrent': True,
                        'mode': 'immediate',
                    }],
                },
                {
                    'label': '❚❚ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    }],
                },
            ],
        }],
        sliders=[{
            'active': 0,
            'x': 0.12, 'y': 0.05, 'len': 0.82, 'xanchor': 'left', 'yanchor': 'bottom',
            'currentvalue': {
                'prefix': 'Mission Time: ',
                'suffix': ' s',
                'visible': True,
                'xanchor': 'right',
            },
            'steps': [
                {
                    'label': f"{t:.0f}",
                    'method': 'animate',
                    'args': [[f"{t:.0f}"], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0},
                    }],
                }
                for t in timestamps
            ],
        }],
    )

    return fig




def compute_frame_metrics(scenario, freq_ghz: float = 28.0):
    """
    Pre-compute per-frame, per-(satellite, beam) link metrics for tabular
    display alongside the animation. Doppler and slant range are computed
    against each beam's actual target (resolved via target_cell_id /
    target_x_km, target_y_km / nadir-fallback).
    """
    import pandas as pd
    channel = NTNChannel(carrier_frequency=freq_ghz * 1e9)
    sat_names = scenario.get_satellite_names()
    hex_radius = 1.0
    centers = generate_hex_grid(hex_radius, rows=10, cols=12)

    rows = []
    for t in scenario.get_timestamps():
        state = scenario.get_state_at(t)
        for _, r in state.iterrows():
            sat_id = int(r['sat_id'])
            sx_unit = float(r['x_km']) / SCALE_KM_PER_UNIT
            sy_unit = float(r['y_km']) / SCALE_KM_PER_UNIT
            cell_idx, tx_u, ty_u = _resolve_beam_target(r, sx_unit, sy_unit, centers)

            target_real = np.array(
                [tx_u * SCALE_KM_PER_UNIT, ty_u * SCALE_KM_PER_UNIT, 0.0]
            ) * 1000.0
            sat_real = np.array(
                [float(r['x_km']), float(r['y_km']), float(r['altitude_km'])]
            ) * 1000.0
            sat_to_target = target_real - sat_real
            slant_m = float(np.linalg.norm(sat_to_target))
            fspl_db = channel.free_space_path_loss(slant_m)

            v_xyz = np.array([float(r['vx_kms']) * 1000.0,
                              float(r['vy_kms']) * 1000.0, 0.0])
            los_unit = sat_to_target / (slant_m if slant_m > 0 else 1.0)
            v_r_mps = -float(np.dot(v_xyz, los_unit))
            doppler_hz = channel.calculate_doppler_shift(v_r_mps)

            rows.append({
                'time_sec': float(t),
                'sat_id': sat_id,
                'beam_idx': int(r.get('beam_idx', 0) or 0),
                'name': sat_names.get(sat_id, f'Sat-{sat_id}'),
                'target_cell_id': int(cell_idx),
                'distance_km': slant_m / 1000.0,
                'fspl_db': fspl_db,
                'doppler_khz': doppler_hz / 1e3,
                'altitude_km': float(r['altitude_km']),
                'speed_kms': float(r['speed_kms']),
            })
    return pd.DataFrame(rows)
