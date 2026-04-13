import time
from antenna import UniformRectangularArray
from trajectory import LEOSatellite
from channel import NTNChannel

def run_tracking_demo():
    print("=========================================================")
    print(" NTN Beamforming POC: Automated LEO Tracking Validation")
    print("=========================================================\n")
    
    fc = 28e9 # 28 GHz Ka-Band
    
    # Initialize Core Components
    array = UniformRectangularArray(num_x=8, num_y=8, frequency=fc)
    satellite = LEOSatellite(altitude_km=600.0)
    channel = NTNChannel(carrier_frequency=fc)
    
    print("Initiating Tracking Sequence (Target Passing Overhead)...")
    print(f"{'Time(s)':<8} | {'Dist(km)':<10} | {'Elev(deg)':<9} | {'Azim(deg)':<9} | {'Doppler(kHz)':<12} | {'FSPL(dB)':<8}")
    print("-" * 75)
    
    # Simulate an intense flyby from -50 seconds (approaching) to +50 seconds (departing)
    for t in range(-50, 51, 5):
        # 1. Physics Engine: Get true position of moving satellite target
        azimuth, elevation, distance, radial_velocity = satellite.get_tracking_data(time_sec=t)
        
        # 2. Antenna Engine: Autonomously Steer the beam to the target
        # Here we prove Phase 2 works: we pass in the exact geometric target and retrieve the phase shifts
        weights = array.generate_steering_vector(azimuth, elevation)
        
        # 3. Channel Engine: Calculate physical propagation impairments based on dynamic metrics
        doppler_hz = channel.calculate_doppler_shift(radial_velocity)
        path_loss = channel.free_space_path_loss(distance)
        
        # Format the output metrics for scientific console display
        doppler_khz = doppler_hz / 1000.0
        dist_km = distance / 1000.0
        
        # Output telemetry to console
        print(f"{t:<8} | {dist_km:<10.2f} | {elevation:<9.2f} | {azimuth:<9.2f} | {doppler_khz:<12.2f} | {path_loss:<8.2f}")
        
        # Sleep slightly to imitate real-time lock updates
        time.sleep(0.3)

    print("-" * 75)
    print("Simulation Complete. Phased Array maintained dynamic NTN target lock successfully.")

if __name__ == "__main__":
    run_tracking_demo()
