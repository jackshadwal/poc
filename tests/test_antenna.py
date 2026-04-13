import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from antenna import UniformRectangularArray

def test_antenna_initialization():
    fc = 28e9
    array = UniformRectangularArray(8, 8, fc)
    assert array.num_x == 8
    assert array.num_y == 8
    assert array.frequency == fc
    
    # Assert lambda and spacing calculated correctly
    assert array.c == 3e8
    assert np.isclose(array.lambda_, 3e8 / 28e9)
    assert np.isclose(array.spacing, array.lambda_ / 2.0)
    
def test_steering_vector_dimensions():
    array = UniformRectangularArray(4, 4, 30e9)
    weights = array.generate_steering_vector(0, 0)
    # Validate the phase matrix evaluates to the size of the array
    assert weights.shape == (4, 4)
    
def test_array_factor_normalization():
    array = UniformRectangularArray(4, 4, 28e9)
    theta = np.array([[0, np.pi/4]])
    phi = np.array([[0, 0]])
    af = array.calculate_array_factor(theta, phi, 0, 0)
    
    # Validates output tensors exist and don't overflow the normalization boundary
    assert af is not None
    assert np.max(af) <= 1.0001
