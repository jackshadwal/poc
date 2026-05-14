import os
import sys
from io import StringIO

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from scenario import Scenario, REQUIRED_COLS, generate_sample_scenario


def _toy_csv(extra_cols: str = "") -> StringIO:
    """Two satellites, three timesteps each, moving linearly in +x."""
    header = "time_sec,sat_id,x_km,y_km,altitude_km" + extra_cols
    return StringIO(
        header + "\n"
        "0,1,0,0,600\n"
        "1,1,7.5,0,600\n"
        "2,1,15,0,600\n"
        "0,2,100,0,550\n"
        "1,2,107.5,0,550\n"
        "2,2,115,0,550\n"
    )


def test_loads_minimal_csv():
    scn = Scenario.from_csv(_toy_csv())
    assert scn.num_satellites() == 2
    assert scn.get_timestamps() == [0, 1, 2]
    assert scn.get_satellite_ids() == [1, 2]


def test_derived_velocity_and_distance():
    scn = Scenario.from_csv(_toy_csv())
    # At t=1 sat 1 should have vx_kms = 7.5 (it just moved 7.5 km in 1 s)
    state = scn.get_state_at(1.0)
    sat1 = state[state['sat_id'] == 1].iloc[0]
    assert np.isclose(sat1['vx_kms'], 7.5)
    assert np.isclose(sat1['vy_kms'], 0.0)
    # Slant distance at t=1: sqrt(7.5^2 + 0 + 600^2)
    assert np.isclose(sat1['distance_km'], np.sqrt(7.5 ** 2 + 600 ** 2))


def test_radial_velocity_sign_convention():
    """Sat moving away from ground origin → radial_velocity_kms should be negative."""
    scn = Scenario.from_csv(_toy_csv())
    state = scn.get_state_at(2.0)
    sat1 = state[state['sat_id'] == 1].iloc[0]
    # At (15, 0, 600) moving with vx=+7.5 → moving outward in x → away from origin
    assert sat1['radial_velocity_kms'] < 0


def test_missing_required_column_raises():
    bad = StringIO("time_sec,sat_id,x_km,y_km\n0,1,0,0\n")  # altitude_km missing
    with pytest.raises(ValueError, match="missing required columns"):
        Scenario.from_csv(bad)


def test_empty_csv_raises():
    empty = StringIO("time_sec,sat_id,x_km,y_km,altitude_km\n")
    with pytest.raises(ValueError, match="empty"):
        Scenario.from_csv(empty)


def test_sample_scenario_generator_shape():
    df = generate_sample_scenario(num_satellites=3, t_start=-10, t_end=10, dt=1)
    # 21 timesteps × 3 sats
    assert len(df) == 21 * 3
    assert set(REQUIRED_COLS).issubset(df.columns)
    assert df['sat_id'].nunique() == 3


def test_bundled_sample_csv_loads():
    """The shipped data/sample_scenario.csv must load cleanly."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(repo_root, 'data', 'sample_scenario.csv')
    if not os.path.exists(path):
        pytest.skip(f"sample file not present at {path}")
    scn = Scenario.from_csv(path)
    assert scn.num_satellites() >= 1
    assert len(scn.get_timestamps()) > 1


def test_bundled_sample_has_varying_beams_per_satellite():
    """The shipped sample is designed so different sats have different beam counts.
    This guards the 'simulation adapts to the data' property visually demonstrable
    on first load: we expect the per-sat beam-count map to have more than one
    distinct value."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(repo_root, 'data', 'sample_scenario.csv')
    if not os.path.exists(path):
        pytest.skip(f"sample file not present at {path}")
    scn = Scenario.from_csv(path)
    beam_counts = list(scn.max_beams_per_satellite().values())
    assert len(set(beam_counts)) > 1, (
        f"Bundled sample should showcase varied beams per sat; got {beam_counts}"
    )


# ----------------------------------------------------------------------
# Multi-beam / explicit-targeting tests
# ----------------------------------------------------------------------
def _multi_beam_csv() -> StringIO:
    """Two sats, two beams per sat, two timesteps. Positions duplicated."""
    return StringIO(
        "time_sec,sat_id,x_km,y_km,altitude_km,beam_idx,target_cell_id\n"
        "0,1,0,0,600,0,10\n"
        "0,1,0,0,600,1,25\n"
        "1,1,7.5,0,600,0,11\n"
        "1,1,7.5,0,600,1,26\n"
        "0,2,200,0,550,0,5\n"
        "0,2,200,0,550,1,40\n"
        "1,2,207.5,0,550,0,5\n"
        "1,2,207.5,0,550,1,41\n"
    )


def test_multi_beam_rows_load():
    scn = Scenario.from_csv(_multi_beam_csv())
    # 2 sats, 2 beams each, 2 timesteps = 8 rows
    assert len(scn.df) == 8
    assert scn.max_beams_per_satellite() == {1: 2, 2: 2}
    assert scn.total_beam_slots() == 4


def test_kinematics_deduplicated_across_beams():
    """Velocity must be computed from unique positions, not beam-duplicated rows."""
    scn = Scenario.from_csv(_multi_beam_csv())
    state_t1 = scn.get_state_at(1.0)
    sat1_rows = state_t1[state_t1['sat_id'] == 1]
    # Both beam rows must share the same derived velocity
    assert sat1_rows['vx_kms'].nunique() == 1
    assert np.isclose(sat1_rows['vx_kms'].iloc[0], 7.5)


def test_beam_idx_missing_defaults_to_zero():
    """Backward compat: CSVs without beam_idx still load with beam_idx=0."""
    scn = Scenario.from_csv(_toy_csv())
    assert (scn.df['beam_idx'] == 0).all()
    assert scn.max_beams_per_satellite() == {1: 1, 2: 1}


def test_target_columns_preserved():
    """Optional targeting columns are preserved on the loaded DataFrame."""
    scn = Scenario.from_csv(_multi_beam_csv())
    assert 'target_cell_id' in scn.df.columns
    # Sat 1, beam 1, t=0 should target cell 25
    row = scn.df[(scn.df['sat_id'] == 1) & (scn.df['beam_idx'] == 1) &
                 (scn.df['time_sec'] == 0)].iloc[0]
    assert int(row['target_cell_id']) == 25


