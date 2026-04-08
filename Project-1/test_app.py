import streamlit as st
import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="10-Qubit Quantum Explorer", layout="wide")
st.title("⚛ 10-Qubit Quantum Simulator")

# --- Sidebar Controls ---
st.sidebar.header("Circuit Configuration")
preset = st.sidebar.selectbox("Select Preset", ["Teleportation", "Hubbard-inspired Ring"])

if preset == "Teleportation":
    st.sidebar.info("Simulating a circuit to teleport quantum information from Alice to Bob.")
    theta_x = st.sidebar.slider("Initial Rotation About X Axis (θ_x)", 0.0, 6.28, math.pi/2)
    theta_z = st.sidebar.slider("Initial Rotation About Z Axis (θ_z)", 0.0, 6.28, math.pi)
    param = [theta_x, theta_z]
else:
    st.sidebar.info("Simulating a 1D Hubbard-like interaction chain (XY-model logic).")
    param = st.sidebar.slider("Interaction Strength (J/U)", 0.0, 10.0, 1.0)


# --- Circuit Generation Logic ---
def generate_circuit(mode, val):
    qc = QuantumCircuit(3)

    if mode == "Teleportation":
        # Prepare an initial state on q0 --> arbitrary superposition of |0> and |1>
        qc.ry(val[0], 0)
        qc.rz(val[1], 0)

        #Prepare Bell pair on q1 and q2
        qc.h(1)
        qc.cx(1,2)

        # Teleportation protocol
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(1,2)
        qc.cz(0,2)

        qc.measure_all()

    else:  # Hubbard-inspired
        # Initialize with alternating states
        for i in range(0, 10, 2):
            qc.x(i)
        qc.barrier()
        # Interaction layer
        for i in range(9):
            qc.rxx(val, i, i + 1)
            qc.ryy(val, i, i + 1)

        qc.measure_all()

    return qc


# --- Execution ---
circuit = generate_circuit(preset, param)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Circuit Diagram")
    # Using 'mpl' for a clean look; falls back to text if matplotlib is missing
    fig = circuit_drawer(circuit, output='mpl', style='iqp')
    st.pyplot(fig)

with col2:
    st.subheader("Simulation Results")

    # Using AerSimulator as requested
    backend = AerSimulator()
    job = backend.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()

    # Displaying the "Histogram" as data
    st.write("**Raw Counts (Subroutine Output):**")
    st.dataframe(counts, use_container_width=True)

    # Optional: Small bar chart of the top 10 results for clarity
    top_counts = dict(sorted(counts.items(), key=lambda item: item, reverse=True)[:10])
    st.bar_chart(top_counts)

# --- Stats Summary ---
st.divider()
st.caption(f"Backend: AerSimulator | Qubits: 10 | Shots: 1024 | Parameter: {param:.2f}")