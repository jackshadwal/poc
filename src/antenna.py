import numpy as np

class UniformRectangularArray:
    """
    Simulates a Phased Uniform Rectangular Array (URA) for NTN environments.
    """
    def __init__(self, num_x: int, num_y: int, frequency: float):
        self.num_x = num_x
        self.num_y = num_y
        self.frequency = frequency
        
        self.c = 3e8 # Speed of light in m/s
        self.lambda_ = self.c / self.frequency
        self.spacing = self.lambda_ / 2.0 # Half-wavelength spacing avoids grating lobes
        
    def generate_steering_vector(self, azimuth: float, elevation: float) -> np.ndarray:
        """
        Calculates the phase shifts (weights) required to steer the beam.
        """
        az_rad = np.radians(azimuth)
        # Convert elevation from standard (-90 to 90) to zenith (0 to 180) for spherical physics
        zenith = 90.0 - elevation
        ze_rad = np.radians(zenith)
        
        k = 2 * np.pi / self.lambda_
        weights = np.zeros((self.num_x, self.num_y), dtype=complex)
        
        for m in range(self.num_x):
            for n in range(self.num_y):
                # Using negative phase for the transmit steering weights
                phase = -k * self.spacing * (m * np.sin(ze_rad) * np.cos(az_rad) + 
                                             n * np.sin(ze_rad) * np.sin(az_rad))
                weights[m, n] = np.exp(1j * phase)
                
        return weights

    def calculate_array_factor(self, theta_grid: np.ndarray, phi_grid: np.ndarray, steer_az: float, steer_el: float) -> np.ndarray:
        """
        Calculates the normalized Array Factor (magnitude) over a 2D spherical grid.
        theta_grid: Zenith angle grid in radians (typically 0 to pi/2)
        phi_grid: Azimuth angle grid in radians (typically -pi to pi)
        """
        weights = self.generate_steering_vector(steer_az, steer_el)
        k = 2 * np.pi / self.lambda_
        AF = np.zeros_like(theta_grid, dtype=complex)
        
        for m in range(self.num_x):
            for n in range(self.num_y):
                # Steering phase across the physical geometric space
                phase = k * self.spacing * (m * np.sin(theta_grid) * np.cos(phi_grid) + 
                                            n * np.sin(theta_grid) * np.sin(phi_grid))
                AF += weights[m, n] * np.exp(1j * phase)
                
        # Normalize the field
        AF_mag = np.abs(AF)
        return AF_mag / np.max(AF_mag)

