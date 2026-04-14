"""
backend.py — Qiskit-Aer quantum simulation backend for P452 Project 1
Implements all circuits required by Q1.2 – Q3.3
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

# ─── Global simulator ────────────────────────────────────────────────────────────
SIMULATOR = AerSimulator()


def run_circuit(qc: QuantumCircuit, shots: int = 1024) -> dict:
    """Execute a QuantumCircuit and return a measurement counts dict."""
    # Add measurements to all qubits if not already present
    if qc.num_clbits == 0:
        qc = qc.copy()
        qc.measure_all()
    result = SIMULATOR.run(qc, shots=shots).result()
    return result.get_counts()


def get_statevector(qc: QuantumCircuit) -> Statevector:
    """Return the ideal statevector (no measurement, uses save_statevector)."""
    qc2 = qc.copy()
    qc2.save_statevector()
    result = SIMULATOR.run(qc2).result()
    return result.get_statevector()


def plot_histogram(counts: dict, title: str = "") -> plt.Figure:
    """Plot a measurement histogram with dark theme."""
    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]
    total = sum(values)
    probs = [v / total for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4), facecolor="#07090f")
    ax.set_facecolor("#07090f")
    bars = ax.bar(range(len(labels)), probs, color="#3b6ef8", edgecolor="#1e2d50", linewidth=0.8)
    for bar, prob in zip(bars, probs):
        if prob > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{prob:.2f}", ha="center", va="bottom", fontsize=8, color="#e2e8f7")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color="#e2e8f7")
    ax.set_ylabel("Probability", color="#e2e8f7")
    ax.set_title(title, color="#e2e8f7", fontsize=11)
    ax.tick_params(colors="#e2e8f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d50")
    ax.set_ylim(0, min(1.15, max(probs) + 0.15))
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────────
#  Q1.2 — Parameter Control: Ry(θ) on q0, CNOT q0→q1
# ────────────────────────────────────────────────────────────────────────────────

def make_param_control_circuit(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ────────────────────────────────────────────────────────────────────────────────
#  Q1.3 — GHZ State on n qubits
# ────────────────────────────────────────────────────────────────────────────────

def make_ghz_circuit(n: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.barrier()
    qc.measure(range(n), range(n))
    return qc


# ────────────────────────────────────────────────────────────────────────────────
#  Q1.4 — Unitarity: prepare superposition, apply CNOT chain, reverse
# ────────────────────────────────────────────────────────────────────────────────

def _prepare_superposition(qc: QuantumCircuit) -> None:
    """
    Prepare (1/sqrt(2))(|201> + |425>) using explicit primitive gates.
    Avoids initialize() which dumps the full state vector into the circuit label.

    Bit decomposition (little-endian, q0 = LSB):
      |201>: q0=1, q1=0, q2=0, q3=1, q4=0, q5=0, q6=1, q7=1, q8=0, q9=0
      |425>: q0=1, q1=0, q2=0, q3=1, q4=0, q5=1, q6=0, q7=1, q8=1, q9=0

    Shared bits: q0=1, q3=1, q7=1
    Differing bits: q5 (0 vs 1), q6 (1 vs 0), q8 (0 vs 1)
    """
    qc.x(0);
    qc.x(3);
    qc.x(7)  # shared '1' bits
    qc.h(5)  # superposition over differing qubit
    qc.x(6)  # q6=1 for |201> branch
    qc.cx(5, 6)  # flip q6->0 in |425> branch
    qc.cx(5, 8)  # set q8=1 in |425> branch


def _apply_cnot_chain(qc: QuantumCircuit, reverse: bool = False) -> None:
    """Apply or reverse 9 CNOT gates q0→q1, q1→q2, …, q8→q9."""
    pairs = [(i, i + 1) for i in range(9)]
    if reverse:
        pairs = list(reversed(pairs))
    for ctrl, tgt in pairs:
        qc.cx(ctrl, tgt)


def make_unitarity_circuit():
    """
    Returns a single labeled circuit and statevectors for all three stages of Q1.4.
    The circuit uses barrier(label=...) to divide:
      - State Preparation
      - Gate Operations
      - Reverse Operations
    """
    # Full labeled circuit for display
    qc_full = QuantumCircuit(10)
    _prepare_superposition(qc_full)
    qc_full.barrier(label="State Preparation")
    _apply_cnot_chain(qc_full, reverse=False)
    qc_full.barrier(label="Gate Operations")
    _apply_cnot_chain(qc_full, reverse=True)
    qc_full.barrier(label="Reverse Operations")

    # Statevectors at each stage (separate circuits)
    qc_init = QuantumCircuit(10)
    _prepare_superposition(qc_init)
    sv_init = Statevector(qc_init)

    qc_forward = QuantumCircuit(10)
    _prepare_superposition(qc_forward)
    _apply_cnot_chain(qc_forward, reverse=False)
    sv_after = Statevector(qc_forward)

    qc_reverse = QuantumCircuit(10)
    _prepare_superposition(qc_reverse)
    _apply_cnot_chain(qc_reverse, reverse=False)
    _apply_cnot_chain(qc_reverse, reverse=True)
    sv_recovered = Statevector(qc_reverse)

    return qc_full, sv_init, sv_after, sv_recovered


# ────────────────────────────────────────────────────────────────────────────────
#  Q2.1 — Quantum Teleportation
# ────────────────────────────────────────────────────────────────────────────────

# Default angles to produce (2|0> + |1>)/sqrt(5):
#   Ry(theta_y)|0> = cos(theta_y/2)|0> + sin(theta_y/2)|1>
#   We need cos(theta_y/2) = 2/sqrt(5), so theta_y = 2*arccos(2/sqrt(5))
#   Rz(0) leaves relative phase unchanged → theta_z = 0
ALICE_DEFAULT_THETA_Y = 2 * np.arccos(2 / np.sqrt(5))
ALICE_DEFAULT_THETA_Z = 0.0


def make_teleportation_circuit(
        theta_y: float = ALICE_DEFAULT_THETA_Y,
        theta_z: float = ALICE_DEFAULT_THETA_Z,
) -> QuantumCircuit:
    """
    3-qubit teleportation circuit with a general Alice state.
    q0: Alice message qubit  — prepared as Rz(theta_z) Ry(theta_y) |0>
    q1: Alice Bell-pair qubit
    q2: Bob qubit (receives teleported state)

    Default angles produce (2|0> + |1>)/sqrt(5).
    """
    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)

    # -- Alice's state preparation --
    qc.ry(theta_y, 0)
    qc.rz(theta_z, 0)
    qc.barrier(label="Alice's State")

    # -- Bell State Preparation (q1, q2) --
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell State Preparation")

    # -- Bell Measurement (q0, q1) --
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Bell Measurement")
    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])

    # -- Bob's Correction (classically controlled) --
    with qc.if_test((cr[1], 1)):
        qc.x(2)
    with qc.if_test((cr[0], 1)):
        qc.z(2)

    return qc


# ────────────────────────────────────────────────────────────────────────────────
#  Q2.2 — Long-Distance CNOT via SWAP chain
# ────────────────────────────────────────────────────────────────────────────────

def make_long_distance_cnot_circuit():
    """
    Perform CNOT q0 → q4 in a linear chain (0-1-2-3-4)
    using only adjacent SWAPS and CNOTs. Each SWAP = 3 CNOTs.

    Strategy: SWAP q4 toward q0, do CNOT, SWAP back.
    Specifically: SWAP(3,4), SWAP(2,3), SWAP(1,2), CNOT(0,1),
                  SWAP(1,2), SWAP(2,3), SWAP(3,4)
    Total CNOTs = 3*6 + 1 = 19
    """
    qc = QuantumCircuit(5)
    n_cnots = 0
    n_swaps = 0

    def swap_adjacent(a, b):
        nonlocal n_swaps
        qc.swap(a, b)
        n_swaps += 1

    # Move q4 adjacent to q0
    swap_adjacent(3, 4)
    swap_adjacent(2, 3)
    swap_adjacent(1, 2)
    qc.barrier(label="Swap q4 → q1")


    # The actual CNOT
    qc.cx(0, 1)
    n_cnots += 1
    qc.barrier(label="CNOT q0 → q1")

    # Restore positions
    swap_adjacent(1, 2)
    swap_adjacent(2, 3)
    swap_adjacent(3, 4)
    qc.barrier(label="Swap q1 → q4")

    n_cnots += n_swaps*3    # 3 CNOTs per SWAP gate

    return qc, n_cnots


# ────────────────────────────────────────────────────────────────────────────────
#  Q3 — Fermi-Hubbard Model via Trotterization + Jordan-Wigner
# ────────────────────────────────────────────────────────────────────────────────

def _trotter_step(qc: QuantumCircuit, J: float, U: float, dt: float) -> None:
    """
    One Trotter step using RXX, RYY, and RZZ gates.

    Args:
        dt: The individual time step size derived from tau / n_trotter.
    """
    # ── Tunneling terms for spin up - XX + YY on q0,q2 ─────────────────────────────
    qc.rxx(-J * dt / 2, 0, 2)
    qc.ryy(-J * dt / 2, 0, 2)
    qc.barrier(label = "Tunneling term for spin up")

    # ── Tunneling terms for spin down - XX + YY on q1,q3 ───────────────────────────
    qc.rxx(-J * dt / 2, 1, 3)
    qc.ryy(-J * dt / 2, 1, 3)
    qc.barrier(label = "Tunneling term for spin down")

    # ── Interaction: site 1 (q0, q1) and site 2 (q2, q3) ────────────────────────
    # Angle u_phase simulates the Coulomb repulsion energy cost
    u_phase = U * dt / 4
    for site_pair in [(0, 1), (2, 3)]:
        i, j = site_pair
        qc.rz(-u_phase, i)
        qc.rz(-u_phase, j)
        qc.rzz(u_phase, i, j)
    qc.barrier(label = "Interaction term")


def make_hubbard_circuit(
        J: float = 1.0,
        U: float = 0.0,
        tau: float = 3.1415,
        n_trotter: int = 20,
        init_state: str = "1000"
) -> QuantumCircuit:
    """
    Build a Trotterized Hubbard circuit with constant total time tau.

    Args:
        tau: Constant total evolution time (e.g., pi for Rabi transfer)
        n_trotter: Variable number of steps to adjust Trotter error.
    """
    qc = QuantumCircuit(4)

    # Initialize state (e.g., |1000> for Q3.2 or |1100> for Q3.3)
    for i, bit in enumerate(reversed(init_state)):
        if bit == "1":
            qc.x(i)
    qc.barrier(label = "State Preparation")

    # Proportional time step calculation
    dt = tau / n_trotter

    for _ in range(n_trotter):
        _trotter_step(qc, J=J, U=U, dt=dt)

    return qc