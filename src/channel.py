import numpy as np

class NTNChannel:
    """
    Mathematical simulator for Non-Terrestrial Network Channel impairments.
    """
    def __init__(self, carrier_frequency: float):
        self.fc = carrier_frequency
        self.c = 3e8 # Speed of light m/s

    def calculate_doppler_shift(self, relative_velocity: float) -> float:
        """
        Calculate Doppler Shift in Hz based on radial speed.
        relative_velocity: Relative radial velocity in m/s (positive means closing in)
        """
        doppler = (relative_velocity / self.c) * self.fc
        return doppler

    def free_space_path_loss(self, distance: float) -> float:
        """
        Calculate standard Free Space Path Loss (FSPL) in dB.
        distance: Distance in meters
        """
        # FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4*pi / c)
        fspl = 20 * np.log10(distance) + 20 * np.log10(self.fc) + 20 * np.log10(4 * np.pi / self.c)
        return fspl
    
    def apply_channel_impact(self, transmitted_signal, distance, relative_velocity):
        """
        Phase 4 placeholder: Will apply phase shifts and attenuation directly to a complex signal tensor.
        """
        pass
