"""
Scenario loader for multi-satellite, time-stepped NTN simulations.

Accepts a CSV (or DataFrame) with one row per (timestamp, satellite) and
exposes per-timestep state plus derived kinematic quantities (velocity,
slant distance, radial velocity) used by the channel and visualization
modules.

CSV format
----------
Required columns:
    time_sec      : float   simulation time in seconds (can be negative)
    sat_id        : int     unique satellite identifier
    x_km          : float   ground-track X position in km (East+)
    y_km          : float   ground-track Y position in km (North+)
    altitude_km   : float   satellite altitude above ground in km

Optional columns:
    name           : str   human-readable satellite name
    beam_idx       : int   beam index within the satellite (default 0).
                           Multiple rows with the same (time_sec, sat_id) but
                           different beam_idx encode multi-beam scenarios.
                           Changing beam_idx targets over time → beam hopping.
    target_cell_id : int   explicit hex cell to point this beam at (0–47).
    target_x_km    : float explicit ground-target X (km). Used only if
                           target_cell_id is absent.
    target_y_km    : float explicit ground-target Y (km).

Target-resolution priority per beam row:
    target_cell_id  >  (target_x_km, target_y_km)  >  nadir (sub-sat point)

Any other columns are preserved but unused.
"""

import numpy as np
import pandas as pd


REQUIRED_COLS = ['time_sec', 'sat_id', 'x_km', 'y_km', 'altitude_km']
OPTIONAL_TARGETING_COLS = ['beam_idx', 'target_cell_id', 'target_x_km', 'target_y_km']


class Scenario:
    """Multi-satellite, time-stepped NTN scenario."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate()
        self._compute_derived()

    @classmethod
    def from_csv(cls, path_or_buffer) -> 'Scenario':
        df = pd.read_csv(path_or_buffer)
        return cls(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'Scenario':
        return cls(df)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_timestamps(self) -> list:
        return sorted(self.df['time_sec'].unique().tolist())

    def get_satellite_ids(self) -> list:
        return sorted(self.df['sat_id'].unique().tolist())

    def get_satellite_names(self) -> dict:
        """Map sat_id -> display name (falls back to 'Sat-{id}')."""
        names = {}
        for sat_id in self.get_satellite_ids():
            sub = self.df[self.df['sat_id'] == sat_id]
            if 'name' in sub.columns and sub['name'].notna().any():
                names[sat_id] = str(sub['name'].iloc[0])
            else:
                names[sat_id] = f'Sat-{sat_id}'
        return names

    def get_state_at(self, time_sec: float) -> pd.DataFrame:
        """Return all satellites' state at a given timestamp."""
        return self.df[self.df['time_sec'] == time_sec].reset_index(drop=True)

    def num_satellites(self) -> int:
        return len(self.get_satellite_ids())

    def time_range(self) -> tuple:
        return (float(self.df['time_sec'].min()), float(self.df['time_sec'].max()))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate(self):
        missing = set(REQUIRED_COLS) - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Scenario CSV missing required columns: {sorted(missing)}. "
                f"Required: {REQUIRED_COLS}"
            )
        for col in ['time_sec', 'x_km', 'y_km', 'altitude_km']:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        if len(self.df) == 0:
            raise ValueError("Scenario CSV is empty")

    def _compute_derived(self):
        df = self.df.sort_values(['sat_id', 'time_sec']).reset_index(drop=True)

        # Default beam_idx for rows missing it
        if 'beam_idx' not in df.columns:
            df['beam_idx'] = 0
        else:
            df['beam_idx'] = df['beam_idx'].fillna(0).astype(int)

        # Kinematics are computed on per-(sat, time) positions only —
        # multi-beam rows share the same position, so we deduplicate first
        # then merge derived columns back.
        positions = (
            df.drop_duplicates(subset=['sat_id', 'time_sec'])
              [['sat_id', 'time_sec', 'x_km', 'y_km', 'altitude_km']]
              .sort_values(['sat_id', 'time_sec'])
              .reset_index(drop=True)
        )

        positions['vx_kms'] = np.nan
        positions['vy_kms'] = np.nan
        for sat_id, group in positions.groupby('sat_id'):
            idx = group.index
            dt = group['time_sec'].diff()
            vx = group['x_km'].diff() / dt
            vy = group['y_km'].diff() / dt
            if len(group) > 1:
                vx.iloc[0] = vx.iloc[1]
                vy.iloc[0] = vy.iloc[1]
            else:
                vx.iloc[0] = 0.0
                vy.iloc[0] = 0.0
            positions.loc[idx, 'vx_kms'] = vx.values
            positions.loc[idx, 'vy_kms'] = vy.values

        positions['distance_km'] = np.sqrt(
            positions['x_km'] ** 2 + positions['y_km'] ** 2 + positions['altitude_km'] ** 2
        )

        # Radial velocity relative to ground origin (used as a fallback
        # Doppler proxy; per-beam Doppler is computed against the actual
        # beam target at visualization time).
        ux = positions['x_km'] / positions['distance_km']
        uy = positions['y_km'] / positions['distance_km']
        v_outward = positions['vx_kms'] * ux + positions['vy_kms'] * uy
        positions['radial_velocity_kms'] = -v_outward

        positions['speed_kms'] = np.sqrt(positions['vx_kms'] ** 2 + positions['vy_kms'] ** 2)

        # Merge derived columns back onto all rows (including multi-beam duplicates)
        derived_cols = ['sat_id', 'time_sec', 'vx_kms', 'vy_kms',
                        'distance_km', 'radial_velocity_kms', 'speed_kms']
        df = df.merge(positions[derived_cols], on=['sat_id', 'time_sec'], how='left')

        # Sort canonically: sat then time then beam
        df = df.sort_values(['sat_id', 'time_sec', 'beam_idx']).reset_index(drop=True)

        self.df = df

    # ------------------------------------------------------------------
    # Multi-beam helpers
    # ------------------------------------------------------------------
    def max_beams_per_satellite(self) -> dict:
        """Return {sat_id: max_beam_count} — informs trace slot allocation."""
        out = {}
        for sat_id, group in self.df.groupby('sat_id'):
            # Per-timestep beam count, then take the max
            per_t = group.groupby('time_sec').size()
            out[sat_id] = int(per_t.max()) if len(per_t) else 1
        return out

    def total_beam_slots(self) -> int:
        return int(sum(self.max_beams_per_satellite().values()))


# ----------------------------------------------------------------------
# Sample scenario generator (used to bundle a demo CSV)
# ----------------------------------------------------------------------
def generate_sample_scenario(num_satellites: int = 3,
                              t_start: float = -120.0,
                              t_end: float = 120.0,
                              dt: float = 2.0) -> pd.DataFrame:
    """
    Generate a synthetic multi-satellite scenario for demos.
    Each satellite flies a straight line at constant altitude and speed
    so the math is transparent and easy to verify.
    """
    times = np.arange(t_start, t_end + dt, dt)
    rows = []

    # (name, altitude_km, start_pos_km, unit_direction, velocity_kms)
    # Designed so each sat crosses near the origin at t = (t_start+t_end)/2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    sat_defs = [
        ("LEO-Alpha",   600.0, (-900.0,    0.0), ( 1.0,       0.0),       7.5),
        ("LEO-Bravo",   550.0, ( 250.0, -900.0), ( 0.0,       1.0),       7.5),
        ("LEO-Charlie", 700.0, (-636.0, -636.0), ( inv_sqrt2, inv_sqrt2), 7.5),
        ("LEO-Delta",   650.0, ( 900.0,  400.0), (-1.0,       0.0),       7.5),
        ("LEO-Echo",    500.0, (-200.0,  800.0), ( 0.0,      -1.0),       7.5),
    ]

    for sat_idx in range(min(num_satellites, len(sat_defs))):
        name, alt, start_pos, direction, v = sat_defs[sat_idx]
        sat_id = sat_idx + 1
        for t in times:
            # Position evolves linearly from start_pos at t = t_start
            elapsed = t - t_start
            x = start_pos[0] + direction[0] * v * elapsed
            y = start_pos[1] + direction[1] * v * elapsed
            rows.append({
                'time_sec': float(t),
                'sat_id': sat_id,
                'name': name,
                'x_km': x,
                'y_km': y,
                'altitude_km': alt,
            })

    return pd.DataFrame(rows)


def generate_sample_scenario_with_hopping(num_satellites: int = 3,
                                            beams_per_sat: int = 3,
                                            t_start: float = -120.0,
                                            t_end: float = 120.0,
                                            dt: float = 2.0,
                                            hop_period_sec: float = 8.0,
                                            num_cells: int = 48) -> pd.DataFrame:
    """
    Same satellite trajectories as `generate_sample_scenario`, but each sat
    has multiple beams that hop through configured target_cell_id patterns
    over time — useful for demonstrating multi-beam + hopping in the
    animated viewer.
    """
    base = generate_sample_scenario(num_satellites=num_satellites,
                                    t_start=t_start, t_end=t_end, dt=dt)

    # Per-satellite hopping patterns (cell IDs)
    hop_patterns_per_sat = [
        # Sat 1 beams
        [[10, 11, 12, 19, 18, 17, 10, 11],
         [20, 21, 22, 23, 22, 21, 20, 21],
         [30, 31, 32, 33, 32, 31, 30, 31]],
        # Sat 2 beams
        [[ 1,  2,  3,  4,  5,  6,  7,  8],
         [ 9, 17, 25, 33, 41, 40, 32, 24],
         [44, 45, 46, 47, 46, 45, 44, 45]],
        # Sat 3 beams
        [[ 0,  8, 16, 24, 32, 40, 32, 24],
         [13, 14, 15, 22, 21, 20, 13, 14],
         [37, 38, 39, 46, 45, 44, 37, 38]],
        # Sat 4 / Sat 5 fall back to slot 0–2 if more sats are requested
        [[ 5, 13, 21, 29, 37, 45, 37, 29],
         [ 2, 10, 18, 26, 34, 42, 34, 26],
         [ 6, 14, 22, 30, 38, 46, 38, 30]],
        [[ 4, 12, 20, 28, 36, 44, 36, 28],
         [ 7, 15, 23, 31, 39, 47, 39, 31],
         [ 3, 11, 19, 27, 35, 43, 35, 27]],
    ]

    rows = []
    for _, base_row in base.iterrows():
        sat_idx = int(base_row['sat_id']) - 1
        beams = hop_patterns_per_sat[sat_idx % len(hop_patterns_per_sat)]
        # Hop step decided by simulation time (so all sats hop in sync)
        # Step length: hop_period_sec
        hop_idx = int((base_row['time_sec'] - t_start) // hop_period_sec)

        for b in range(min(beams_per_sat, len(beams))):
            pattern = beams[b]
            cell_id = pattern[hop_idx % len(pattern)] % num_cells
            rows.append({
                **base_row.to_dict(),
                'beam_idx': b,
                'target_cell_id': cell_id,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Allow regenerating a basic single-beam sample from the command line:
    #     python src/scenario.py > data/sample_scenario.csv
    # (The bundled sample at data/sample_scenario.csv is hand-crafted with
    # realistic multi-beam hopping sequences — keep that one unless you
    # specifically want to replace it.)
    import sys
    df = generate_sample_scenario(num_satellites=3)
    df.to_csv(sys.stdout, index=False)
