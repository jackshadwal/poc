import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from antenna import UniformRectangularArray

def main():
    print("Initializing NTN Beamforming POC (Python Edition)...")
    
    # Step 1: Define NTN Parameters
    fc = 28e9 # Ka-Band frequency
    
    # Step 2: Initialize Antenna Array (8x8 URA)
    print("Generating 8x8 Uniform Rectangular Array math models...")
    array = UniformRectangularArray(8, 8, fc)
    
    # Grids for 3D Projection (Zenith 0->pi/2 for upper hemisphere)
    phi = np.linspace(-np.pi, np.pi, 50)
    theta = np.linspace(0, np.pi/2, 50)
    Phi, Theta = np.meshgrid(phi, theta)
    
    # Initial beam pointing
    init_az = 0.0
    init_el = 0.0
    
    # Step 3: Create Programmatic GUI for Visualization using Matplotlib
    print("Launching Interactive Beamforming Dashboard...")
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title('NTN Beamforming Dashboard')
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    # Function to draw the Array Factor surface
    def get_surface_coords(az, el):
        # Calculate mathematical array factor (AF) magnitude
        AF_mag = array.calculate_array_factor(Theta, Phi, az, el)
        # Convert spherical to Cartesian based on the AF magnitude scaling
        X = AF_mag * np.sin(Theta) * np.cos(Phi)
        Y = AF_mag * np.sin(Theta) * np.sin(Phi)
        Z = AF_mag * np.cos(Theta)
        return X, Y, Z
    
    # Draw initial plot
    X, Y, Z = get_surface_coords(init_az, init_el)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=False, alpha=0.9)
    ax.set_title(f"3D Beam Directivity (Az: {init_az}°, El: {init_el}°)")
    
    # Lock axes limits so the plot doesn't dynamically jump around
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Build Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_az = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_el = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    s_az = Slider(ax_az, 'Azimuth', -90.0, 90.0, valinit=init_az)
    s_el = Slider(ax_el, 'Elevation', 0.0, 90.0, valinit=init_el) # Elevation typically 0-90
    
    # Global state variable for the surface object to enable efficient removal
    plot_state = {'surf': surf}
    
    # The callback for dynamic redrawing
    def update(val):
        az = s_az.val
        el = s_el.val
        
        # Remove old surface
        plot_state['surf'].remove()
        
        # Calculate new beam pattern mathematically
        X, Y, Z = get_surface_coords(az, el)
        
        # Re-plot
        plot_state['surf'] = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=False, alpha=0.9)
        ax.set_title(f"3D Beam Directivity (Az: {az:.1f}°, El: {el:.1f}°)")
        fig.canvas.draw_idle()
        
    s_az.on_changed(update)
    s_el.on_changed(update)
    
    print("Dashboard operational. Drag the sliders to see the phase arrays dynamically redirect the RF beam!")
    plt.show()

if __name__ == "__main__":
    main()
