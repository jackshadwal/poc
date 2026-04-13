import numpy as np

class LEOSatellite:
    """
    Models the orbital trajectory of a Low Earth Orbit satellite to generate 
    moving target coordinates for the Beamformer to track.
    """
    def __init__(self, altitude_km: float = 600.0):
        self.altitude = altitude_km * 1000.0 # Convert to meters
        self.earth_radius = 6371e3
        
        # Approximate LEO velocity = sqrt(G*M / r) which is consistently ~7.5 km/s
        self.velocity = 7500.0 # m/s
        
    def get_tracking_data(self, time_sec: float) -> tuple:
        """
        Simulates an overhead pass.
        At time_sec = 0, the satellite is exactly Zenith (90 deg elevation, directly overhead).
        
        Returns:
            azimuth (deg), elevation (deg), distance (m), radial_velocity (m/s)
        """
        # Distance horizontally along the pass (x-axis)
        x = self.velocity * time_sec
        y = self.altitude
        
        distance = np.sqrt(x**2 + y**2)
        
        # Elevation: Angle from the horizon up to the satellite
        elevation = np.degrees(np.arctan2(y, np.abs(x)))
        
        # Azimuth: 0 if departing along the axis, 180 if approaching
        if x > 0:
            azimuth = 0.0 # Moving away
        else:
            azimuth = 180.0 # Approaching
            
        # Radial velocity (component of velocity directed at the user)
        # v_r = x * v / d  (Negative means velocity is pointing away from user)
        radial_velocity = -self.velocity * (x / distance)
        
        return azimuth, elevation, distance, radial_velocity
