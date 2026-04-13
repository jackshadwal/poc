# Project Plan: Python / Open-Source Pathway

## Overview
This plan details the execution of the Beamforming for NTN Communication POC using an open-source Python stack. This pathway offers extreme flexibility, zero licensing costs, and great integration with web-based visualization dashboards, though it requires more foundational coding for the channel models.

### Tech Stack
* **Language**: Python 3.10+
* **Physical Layer / Simulation**: NVIDIA Sionna (TensorFlow-based 5G/6G simulation natively supporting ray tracing and channels) and NumPy/SciPy.
* **Visualization/GUI**: 
  * Dash / Streamlit (for a browser-based interactive GUI)
  * Plotly / Matplotlib (for rendering the 3D antenna beam patterns)

---

## 25-Day Execution Timeline

### Step 1: Requirement (3 Days) - Formal Documentation
**Goal**: Collect and document all requirements.
* **Day 1**: Define the NTN structural parameters (Orbital altitude, frequency, user distribution). 
* **Day 2**: Determine the exact mathematical models required (since Python lacks some out-of-the-box 3GPP one-liners, the team must identify equations for free-space path loss, array factors, etc.).
* **Day 3**: Define GUI expectations (e.g., will it be a local desktop app or a web dashboard hosted via Streamlit?).
* **Outcome**: Formal Requirements (REQ) Specification Document.

### Step 2: Requirement (2 Days) - Tooling Closeout
**Goal**: Finalize tool availability and environment setup.
* **Day 4**: Standardize the development environments (e.g., setting up `conda` or `venv`). 
* **Day 5**: Assess GPU availability (if using NVIDIA Sionna for heavy simulations, a CUDA-capable GPU is highly recommended). Finalize the GUI framework choice (Streamlit is recommended for speed).
* **Outcome**: Conda `environment.yml` / `requirements.txt` finalized.

### Step 3: Design (5 Days) - System Architecture
**Goal**: Design software architecture and mathematical models.
* **Day 6**: **Object-Oriented Design**: Outline the Python classes (`Satellite`, `UserEquipment`, `AntennaArray`, `Channel`).
* **Day 7**: **Math Design**: Formulate the steering vector calculations and precoding matrix matrices logic using NumPy.
* **Day 8**: **Channel Design**: Design the custom NTN channel block. Model how the Doppler shift and large delay spreads will be programmatically applied to the signal tensors.
* **Day 9**: **UI Wireframes**: Design the layout of the Streamlit/Dash dashboard. 
* **Day 10**: Consolidate architecture. Ensure data bridges cleanly from NumPy tensors into Plotly 3D scatter/surface objects.
* **Outcome**: Formal Design Document outlining classes and math.

### Step 4: Implementation (6 Days) - Coding
**Goal**: Write the POC code.
* **Day 11**: Code the 3D spatial geometry algorithms (Coordinate conversions from ECEF to Cartesian, tracking satellite position over time).
* **Day 12**: Build the `AntennaArray` class logic to generate the array factor and directivity arrays.
* **Day 13**: Implement the beamforming weights computation using NumPy linear algebra libraries.
* **Day 14**: Build the custom NTN channel simulator (applying time delays and frequency shifts to the transmitted signal).
* **Day 15**: Develop the GUI (e.g., `app.py` for Streamlit) and define the interactive widgets (sliders, inputs).
* **Day 16**: Connect the backend simulation data to Plotly rendering endpoints for live 3D surface plot updates.
* **Outcome**: Pre-alpha Python codebase and runnable UI server.

### Step 5: Testing (4 Days) - Dummy Data
**Goal**: Verify pure mathematical correctness.
* **Day 17-18**: Test the array steering matrices. If the GUI slider is set to 45-degrees, ensure the Numpy maximum gain calculations occur precisely at the 45-degree tensor index.
* **Day 19-20**: Test without fading. Hardcode a dummy stationary satellite and stationary user. Ensure signal strength falls off according to the inverse-square law correctly.
* **Outcome**: Phase 1 Test Results.

### Step 6: Testing (3 Days) - Real Fading Data
**Goal**: Validate the system under realistic communication conditions.
* **Day 21**: Activate advanced channel properties (Rain fading models, Doppler velocity profiles).
* **Day 22**: Simulate dynamic moving satellites. Test if the tracking algorithm updates the beamforming weights fast enough to maintain the link.
* **Day 23**: Performance profiling. Ensure the Python loop doesn't bottleneck and the Streamlit UI frame rate remains smooth during updates.
* **Outcome**: Final Test Results and Profiling Matrix.

### Step 7: Demo (2 Days) - Final Polish
**Goal**: Stakeholder presentation.
* **Day 24**: UI/UX polish. Add informative tooltips to the Streamlit app, enhance Plotly render colors/lighting for maximum aesthetic impact.
* **Day 25**: Present the interactive web dashboard / local execution to the team. 
* **Outcome**: Successful Demo.
