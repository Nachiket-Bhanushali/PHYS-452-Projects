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
    qc.measure(range(n), range(n))
    return qc


# ────────────────────────────────────────────────────────────────────────────────
#  Q1.4 — Unitarity: prepare superposition, apply CNOT chain, reverse
# ────────────────────────────────────────────────────────────────────────────────

def _prepare_superposition(qc: QuantumCircuit) -> None:
    """
    Prepare (1/√2)(|201⟩ + |425⟩) in a 10-qubit register.
    |201⟩ = |0011001001⟩  (binary of 201, q0 is LSB)
    |425⟩ = |0110101001⟩  (binary of 425, q0 is LSB)

    In Qiskit little-endian: qubit i = bit i of the integer.
    |201⟩: 201 = 11001001b → q0=1,q1=0,q2=0,q3=1,q4=0,q5=0,q6=1,q7=1,q8=0,q9=0
    |425⟩: 425 = 110101001b → q0=1,q1=0,q2=0,q3=1,q4=0,q5=1,q6=0,q7=1,q8=1,q9=0
    """
    # Use initialize to set arbitrary superposition
    n = 10
    sv = np.zeros(2 ** n, dtype=complex)
    sv[201] = 1 / np.sqrt(2)
    sv[425] = 1 / np.sqrt(2)
    qc.initialize(sv, range(n))


def _apply_cnot_chain(qc: QuantumCircuit, reverse: bool = False) -> None:
    """Apply or reverse 9 CNOT gates q0→q1, q1→q2, …, q8→q9."""
    pairs = [(i, i + 1) for i in range(9)]
    if reverse:
        pairs = list(reversed(pairs))
    for ctrl, tgt in pairs:
        qc.cx(ctrl, tgt)


def make_unitarity_circuit():
    """
    Returns circuits and statevectors for all three stages of Q1.4.
    """
    # Stage 1: Initial state
    qc_init = QuantumCircuit(10)
    _prepare_superposition(qc_init)
    sv_init = Statevector(qc_init)

    # Stage 2: After CNOT chain
    qc_forward = QuantumCircuit(10)
    _prepare_superposition(qc_forward)
    _apply_cnot_chain(qc_forward, reverse=False)
    sv_after = Statevector(qc_forward)

    # Stage 3: Reverse (recover)
    qc_reverse = QuantumCircuit(10)
    _prepare_superposition(qc_reverse)
    _apply_cnot_chain(qc_reverse, reverse=False)
    _apply_cnot_chain(qc_reverse, reverse=True)
    sv_recovered = Statevector(qc_reverse)

    return qc_init, qc_forward, qc_reverse, sv_init, sv_after, sv_recovered


# ────────────────────────────────────────────────────────────────────────────────
#  Q2.1 — Quantum Teleportation
# ────────────────────────────────────────────────────────────────────────────────

def make_teleportation_circuit(alice_state: str = "custom") -> QuantumCircuit:
    """
    3-qubit teleportation circuit.
    q0: Alice's message qubit
    q1: Alice's Bell pair qubit
    q2: Bob's qubit

    alice_state: "custom" → |q0⟩ = (2|0⟩+|1⟩)/√5
                 "zero"   → |q0⟩ = |0⟩
    """
    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(2, "c")  # measure Alice's q0,q1
    qc = QuantumCircuit(qr, cr)

    # ── Prepare Alice's message qubit ──
    if alice_state == "custom":
        # Ry + Rz to get (2|0⟩+|1⟩)/√5
        # |0⟩ → Ry(2*arccos(2/√5))|0⟩ gives (2|0⟩+|1⟩)/√5 up to global phase
        theta = 2 * np.arccos(2 / np.sqrt(5))
        qc.ry(theta, 0)
    # else: leave as |0⟩

    qc.barrier(label="Alice's state")

    # ── Bell State Preparation (q1, q2) ──
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell State Preparation")

    # ── Bell Measurement (q0, q1) ──
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Bell Measurement")
    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])

    # ── Bob's Correction (classically controlled) ──
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
    using only adjacent CNOTs. Each SWAP = 3 CNOTs.

    Strategy: SWAP q4 toward q0, do CNOT, SWAP back.
    Specifically: SWAP(3,4), SWAP(2,3), SWAP(1,2), CNOT(0,1),
                  SWAP(1,2), SWAP(2,3), SWAP(3,4)
    Total CNOTs = 3*6 + 1 = 19
    """
    qc = QuantumCircuit(5)
    n_cnots = 0

    def swap_adjacent(a, b):
        nonlocal n_cnots
        qc.cx(a, b); qc.cx(b, a); qc.cx(a, b)
        n_cnots += 3

    # Move q4 adjacent to q0
    swap_adjacent(3, 4)
    swap_adjacent(2, 3)
    swap_adjacent(1, 2)

    # The actual CNOT
    qc.cx(0, 1)
    n_cnots += 1

    # Restore positions
    swap_adjacent(1, 2)
    swap_adjacent(2, 3)
    swap_adjacent(3, 4)

    return qc, n_cnots


# ────────────────────────────────────────────────────────────────────────────────
#  Q3 — Fermi-Hubbard Model via Trotterization + Jordan-Wigner
# ────────────────────────────────────────────────────────────────────────────────

def _trotter_step(qc: QuantumCircuit, J: float, U: float, dt: float) -> None:
    """
    One Trotter step for the 2-site Fermi-Hubbard model on 4 qubits.

    Qubit layout:
        q0: site 1 spin-up  (1↑)
        q1: site 1 spin-down (1↓)
        q2: site 2 spin-up  (2↑)
        q3: site 2 spin-down (2↓)

    JW-mapped terms:
      Hopping spin-up (q0↔q2):
        H_J↑ = J/2 * (X0 X2 + Y0 Y2)   [with JW Z-string on q1]
        e^{-i dt H_J↑} = RXX(-J*dt) composed with JW correction

        For non-adjacent (q0,q2) with JW string on q1:
          c†_{0}c_{2} → σ+_0 Z_1 σ-_2
          hopping = -J/2 (X0 Z1 X2 + Y0 Z1 Y2)

      Hopping spin-down (q1↔q3, nearest neighbor, no Z-string needed beyond q1 itself):
        H_J↓ = J/2 * (X1 X3 + Y1 Y3)   [with JW Z-string on q2]
        For q1,q3 non-adjacent:
          hopping = -J/2 (X1 Z2 X3 + Y1 Z2 Y3)

      Interaction terms H_U (on-site):
        Site 1: U/4 * (I - Z0 - Z1 + Z0 Z1)
        Site 2: U/4 * (I - Z2 - Z3 + Z2 Z3)
    """

    # ── Spin-up hopping: q0 ↔ q2 with JW Z-string on q1 ──
    # e^{-i dt * (-J/2)(X0 Z1 X2 + Y0 Z1 Y2)}
    # Decompose using: XZX = -Y on middle, YZY = ... 
    # Standard decomposition for XZX + YZY Pauli strings:
    # Use CNOT-based implementation

    # XZX term: e^{i J dt/2 * X0 Z1 X2}
    t = J * dt / 2  # note sign: H = -J * hopping → exponent picks up -i dt * (-J/2) = +i J dt/2

    # e^{i t X0 Z1 X2}: change basis, apply RZZ-like circuit
    # CNOT(0,1), CNOT(2,1) → ZZZ on middle, then Rz, then uncompute
    qc.h(0); qc.h(2)           # X basis on 0,2
    qc.cx(0, 1); qc.cx(2, 1)   # parity into q1
    qc.rz(-2 * t, 1)            # phase
    qc.cx(0, 1); qc.cx(2, 1)
    qc.h(0); qc.h(2)
    qc.barrier()

    # e^{i t Y0 Z1 Y2}
    qc.sdg(0); qc.h(0)         # Y basis on 0
    qc.sdg(2); qc.h(2)         # Y basis on 2
    qc.cx(0, 1); qc.cx(2, 1)
    qc.rz(-2 * t, 1)
    qc.cx(0, 1); qc.cx(2, 1)
    qc.h(0); qc.s(0)
    qc.h(2); qc.s(2)
    qc.barrier()

    # ── Spin-down hopping: q1 ↔ q3 with JW Z-string on q2 ──
    qc.h(1); qc.h(3)
    qc.cx(1, 2); qc.cx(3, 2)
    qc.rz(-2 * t, 2)
    qc.cx(1, 2); qc.cx(3, 2)
    qc.h(1); qc.h(3)
    qc.barrier()

    qc.sdg(1); qc.h(1)
    qc.sdg(3); qc.h(3)
    qc.cx(1, 2); qc.cx(3, 2)
    qc.rz(-2 * t, 2)
    qc.cx(1, 2); qc.cx(3, 2)
    qc.h(1); qc.s(1)
    qc.h(3); qc.s(3)
    qc.barrier()

    # ── Interaction terms: site 1 (q0,q1) and site 2 (q2,q3) ──
    # H_U = U/4 (I - Z0 - Z1 + Z0Z1) + U/4 (I - Z2 - Z3 + Z2Z3)
    # e^{-i dt H_U}:
    #   global phase from I terms: ignored (unobservable)
    #   Z0 term → Rz(+U*dt/2, q0) [e^{-i dt (-U/4)Z0} = e^{+i U dt/4 Z0} = Rz(-U*dt/2,q0) 
    #              note Rz(λ)|ψ⟩ = e^{-iλ/2 Z}|ψ⟩ → Rz(2*U*dt/4) = Rz(U*dt/2)]
    u_phase = U * dt / 2
    # Single-Z terms (chemical potential shifts)
    qc.rz(u_phase, 0)
    qc.rz(u_phase, 1)
    qc.rz(u_phase, 2)
    qc.rz(u_phase, 3)
    # ZZ terms (entangling)
    # e^{-i dt (U/4) Z0 Z1} = RZZ(U*dt/2)
    qc.rzz(U * dt / 2, 0, 1)
    qc.rzz(U * dt / 2, 2, 3)
    qc.barrier()


def make_hubbard_circuit(
    J: float = 1.0,
    U: float = 0.0,
    tau: float = 0.1,
    n_trotter: int = 10,
    init_state: str = "1000",
) -> QuantumCircuit:
    """
    Build a Trotterized Hubbard circuit.

    init_state: "1000" → q0=1 (spin-up at site 1)
                "1100" → q0=1,q1=1 (both spins at site 1)
    """
    qc = QuantumCircuit(4)

    # ── Initialize state ──
    for i, bit in enumerate(reversed(init_state)):  # reversed: q0 is rightmost
        if bit == "1":
            qc.x(i)
    qc.barrier()

    # ── Trotterize ──
    dt = tau / n_trotter
    for _ in range(n_trotter):
        _trotter_step(qc, J=J, U=U, dt=dt)

    return qc
