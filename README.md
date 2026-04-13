# Simulation and Visualization of Beamforming for NTN Communication

## 📖 Overview
This repository contains the Proof of Concept (POC) for simulating and visualizing Non-Terrestrial Network (NTN) beamforming. It is designed to model the physical layer dynamics of a satellite communication system, focusing on Ka-Band (28 GHz) transmissions and Low Earth Orbit (LEO) trajectories.

Our goal is to dynamically generate phase-shift steering arrays, simulate severe physical channel impairments (Doppler shift and Free Space Path Loss), and provide high-fidelity 3D visual representations.

<img width="1470" height="798" alt="Screenshot 2026-04-13 at 7 47 08 PM" src="https://github.com/user-attachments/assets/3e067bee-791a-4b7a-8d48-589f3fe3b4ea" />


## 🚀 Key Features

This project is separated into three core mathematical simulation engines:

1. **Antenna Engine (`src/antenna.py`)**:
   - Implements a Phased Uniform Rectangular Array (URA) configured for 64 elements (8x8).
   - Dynamically computes complex phase-shift weights to mathematically steer beams across 3D spherical space without moving hardware.
   - Computes real-time 3D Array Factor directivity patterns using `numpy`.

2. **NTN Channel Engine (`src/channel.py`)**:
   - Computes foundational Free Space Path Loss (FSPL) in dB over intense distances.
   - Generates severe Doppler shifts mathematically tied to the rapid velocity of a LEO object relative to a stationary ground terminal.

3. **Trajectory Engine (`src/trajectory.py`)**:
   - Models continuous overhead flybys.
   - Outputs the exact geometric Azimuth, Elevation, line-of-sight Distance, and Radial Velocity required at any given timestamp for the beam hardware to successfully lock and track.

---

## 🏗️ High-Level Design (HLD)
The system is built on a modular three-tier architecture that cleanly separates the physical channel modeling from the user interface presentation.

1. **Presentation Layer**: A Streamlit/Plotly interface that accepts user steering inputs and renders the 3D physics outcomes in an isolated state loop.
2. **Integration Controller**: A central simulation bus (`app.py` & `demo_tracking.py`) that acts as the timeline coordinator, passing mathematical physics states from the trajectory generator directly to the antenna matrix.
3. **Core Physics Engines**: A suite of decoupled Python modules handling highly-specific math operations (Antenna Phasing, Orbital Telemetry, and Signal Path Impairment).

### *Architecture Diagrams & UI Screenshots*

![hld_diagram](https://github.com/user-attachments/assets/8c7cee43-2f4f-4f9f-a2aa-32ebf36444fa)

---

## 🔍 Low-Level Design (LLD)
The backend physics engines are designed around strict, typed Object-Oriented boundaries:

### 1. `UniformRectangularArray` Class (`antenna.py`)
- **Initialization**: `num_x` (int), `num_y` (int), `frequency` (float)
- **Core Method - `calculate_array_factor`**: 
  - Takes a `numpy.meshgrid` of spherical parameters `Theta` (zenith) and `Phi` (azimuth).
  - Determines the target look-angle and generates the $64$-element phase shift weights array ($W_{m,n}$).
  - Computes the spatial electromagnetic interference pattern and normalizes the output tensor for 3D mapping.

### 2. `LEOSatellite` Class (`trajectory.py`)
- **Initialization State**: Maintains static orbital descriptors (e.g., $600$km altitude, $v = 7500$ m/s).
- **Core Method - `get_tracking_data`**: 
  - Takes `time_sec` ($t$) as the only dynamic variable.
  - Uses spherical trigonometric formulations to calculate relative geometric positions.
  - **Returns**: `(azimuth, elevation, distance, radial_velocity)`.

### 3. `NTNChannel` Class (`channel.py`)
- **Initialization State**: Maintains carrier frequency context ($f_c = 28$ GHz).
- **Core Methods**:
  - `calculate_doppler_shift`: Computes frequency variance $f_d = (\frac{v_r}{c}) \cdot f_c$.
  - `free_space_path_loss`: Computes the classic Friis transmission equation.

### *Technical Flowcharts & Console Output*
![lld_diagram](https://github.com/user-attachments/assets/c7e0a3ad-be5b-4853-9c8d-7f873bfdff37)


---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Mathematics**: NumPy (for tensor and complex vector phase calculations)
- **Visualization (CLI Base)**: Matplotlib
- **Visualization (Web/Dashboard)**: Streamlit & Plotly

---

## 🏃 Getting Started & How to Run

### Prerequisites
Install all dependencies using pip. It is highly recommended to operate within an isolated virtual environment (`venv`).

```bash
pip install -r requirements.txt
```

### Option 1: Interactive Streamlit Web Dashboard
Run the professional web dashboard interface to manually steer the beam using sliders. This mode heavily utilizes the Plotly engine to render beautiful, responsive 3D graphs natively in your browser.

```bash
streamlit run src/app.py
```
*This will open the dashboard automatically in your default internet browser at `http://localhost:8501`.*

### Option 2: High-Speed Validation Demo (Console)
Execute the autonomous simulation engine. This script mimics a satellite flying directly overhead at 7.5 kilometers-per-second, and forces the Antenna module to successfully track and point the beam at it while outputting the physical disruption metrics (Path loss and Doppler shift) continuously.

```bash
python src/demo_tracking.py
```
*Outputs real-time tracking console data directly to your terminal.*

---

## 📅 Architecture & Future Roadmap
- **3GPP NTN Integration**: Transition from FSPL to realistic TR 38.811 Standardized Models (Including Rain Attenuation / Atmospheric Oxygen Absorption).
- **Multi-Beam Steering**: Incorporate Zero-Forcing (ZF) or MMSE precoding equations for distinct Multi-User tracking simultaneously.
- **Waveform Interference**: Mathematically apply the theoretical Doppler shifts and Path Loss directly to digital baseband communication signal tensors alongside massive AWGN interference generation.
