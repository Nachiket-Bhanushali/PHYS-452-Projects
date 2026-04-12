"""
P452 Project 1: Universal Quantum Computer Simulator
Streamlit frontend — connects to Qiskit-Aer backend
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from backend import (
    run_circuit,
    make_ghz_circuit,
    make_teleportation_circuit,
    make_hubbard_circuit,
    make_param_control_circuit,
    make_unitarity_circuit,
    make_long_distance_cnot_circuit,
    plot_histogram,
    get_statevector,
)

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q-Simulator | P452",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #07090f;
    color: #e2e8f7;
}
.stApp { background: #07090f; }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
code, pre { font-family: 'Space Mono', monospace; }

.block-container { padding-top: 2rem; }

.metric-card {
    background: linear-gradient(135deg, #0f1524 0%, #131b30 100%);
    border: 1px solid #1e2d50;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
.stButton>button {
    background: linear-gradient(90deg, #3b6ef8, #6a3bf8);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }

section[data-testid="stSidebar"] {
    background: #0b0e1a;
    border-right: 1px solid #1a2540;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ P452 Quantum Simulator")
    st.markdown("---")
    page = st.radio(
        "Select Module",
        [
            "🏠 Home",
            "Q1.2 — Parameter Control",
            "Q1.3 — GHZ State (10-qubit)",
            "Q1.4 — Unitarity & State Recovery",
            "Q2.1 — Teleportation",
            "Q2.2 — Long-Distance CNOT",
            "Q2.3 — Teleport Statistics",
            "Q3.1 — Hubbard: Circuit",
            "Q3.2 — Hubbard: Non-Interacting",
            "Q3.3 — Hubbard: Mott Physics",
        ],
    )
    st.markdown("---")
    shots = st.slider("Simulation Shots", 256, 8192, 1024, 256)

# ─── Home ────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("Universal Quantum Computer Simulator")
    st.markdown(
        """
        **P452 Project 1 · Cheng Chin · University of Chicago**

        This app provides a complete 10-qubit Qiskit-Aer simulation environment covering:
        - Quantum circuit construction & visualization
        - Quantum teleportation
        - Long-distance CNOT via SWAP chains
        - Fermi-Hubbard model Trotterized time evolution

        Use the sidebar to navigate between modules.
        """,
        unsafe_allow_html=True,
    )
    st.info("🔧 Backend: Qiskit-Aer `AerSimulator`  |  📦 Frontend: Streamlit")

# ─── Q1.2 Parameter Control ──────────────────────────────────────────────────────
elif page == "Q1.2 — Parameter Control":
    st.header("Q1.2 — Parameter Control Loop")
    st.markdown(
        """
        **Circuit:** `Ry(θ) → q0`, then `CNOT q0 → q1`.

        When `θ = π`, `Ry(π)|0⟩ = |1⟩`, so after the CNOT both qubits are `|1⟩`,
        giving `|11⟩` with ~100 % probability — proving the slider value is passed correctly.
        """
    )

    theta = st.slider("Rotation angle θ (radians)", 0.0, 2 * np.pi, np.pi, 0.01,
                      format="%.3f")
    st.latex(rf"\theta = {theta:.4f} \approx {theta/np.pi:.3f}\pi")

    qc = make_param_control_circuit(theta)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Circuit Diagram")
        fig_circ = qc.draw("mpl", style="iqp")
        st.pyplot(fig_circ)
        plt.close()

    with col2:
        st.subheader("Measurement Histogram")
        counts = run_circuit(qc, shots=shots)
        fig_hist = plot_histogram(counts, title=f"θ = {theta:.3f}")
        st.pyplot(fig_hist)
        plt.close()

    with st.expander("Logic Check"):
        st.markdown(
            r"""
            **Why does |11⟩ with ~100 % probability confirm the slider is working?**

            - `Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩`
            - At `θ = π`: `Ry(π)|0⟩ = |1⟩`
            - The CNOT flips q1 only when q0 = 1, so the state after CNOT is `|11⟩`
            - Any incorrect θ would produce a different distribution,
              so a histogram peaked at `|11⟩` uniquely identifies `θ = π` being received.
            """
        )

# ─── Q1.3 GHZ State ──────────────────────────────────────────────────────────────
elif page == "Q1.3 — GHZ State (10-qubit)":
    st.header("Q1.3 — 10-Qubit GHZ State")
    st.latex(r"|GHZ\rangle = \frac{|0\rangle + |1023\rangle}{\sqrt{2}}")
    st.markdown("Prepared via one Hadamard on q0 followed by nine cascaded CNOT gates.")

    qc = make_ghz_circuit(10)

    st.subheader("Circuit Diagram (10 wires)")
    fig_circ = qc.draw("mpl", style="iqp", fold=-1)
    fig_circ.set_size_inches(16, 5)
    st.pyplot(fig_circ)
    plt.close()

    st.subheader("Measurement Histogram")
    counts = run_circuit(qc, shots=shots)
    fig_hist = plot_histogram(counts, title="10-Qubit GHZ")
    st.pyplot(fig_hist)
    plt.close()

    st.markdown(
        "As expected, only `|0000000000⟩` (= |0⟩) and `|1111111111⟩` (= |1023⟩) appear."
    )

# ─── Q1.4 Unitarity ──────────────────────────────────────────────────────────────
elif page == "Q1.4 — Unitarity & State Recovery":
    st.header("Q1.4 — Unitarity & State Recovery (10 qubits)")
    st.markdown(
        r"""
        **Target initial state:** $\frac{1}{\sqrt{2}}(|201\rangle + |425\rangle)$

        where $|201\rangle = |0011001001\rangle$ and $|425\rangle = |0110101001\rangle$ (little-endian Qiskit ordering).

        **Protocol:**
        1. Prepare the superposition state.
        2. Apply a chain of 9 CNOT gates (q0→q1, q1→q2, …, q8→q9).
        3. Reverse the gate sequence to recover the original state.
        """
    )

    qc_init, qc_forward, qc_reverse, sv_init, sv_after, sv_recovered = make_unitarity_circuit()

    col1, col2, col3 = st.columns(3)
    stages = [
        ("Step 1: Initial State", qc_init, sv_init),
        ("Step 2: After CNOT Chain", qc_forward, sv_after),
        ("Step 3: Recovered State", qc_reverse, sv_recovered),
    ]
    for col, (title, qc_s, sv) in zip([col1, col2, col3], stages):
        with col:
            st.subheader(title)
            # show non-zero amplitudes
            amps = {
                format(i, "010b"): round(float(np.abs(amp) ** 2), 6)
                for i, amp in enumerate(sv.data)
                if abs(amp) > 1e-6
            }
            st.json(amps)

    st.subheader("Full Circuit")
    fig_circ = qc_reverse.draw("mpl", style="iqp", fold=40)
    fig_circ.set_size_inches(18, 6)
    st.pyplot(fig_circ)
    plt.close()

    with st.expander("Analysis: Unitarity Confirmed"):
        st.markdown(
            r"""
            - Quantum gates are **unitary matrices** (U†U = I), so every operation is reversible.
            - After applying the CNOT chain, the state vector changes but the information is preserved.
            - Applying the inverse gate sequence (reversed CNOTs) exactly recovers the original amplitudes.
            - The final probabilities being identical to Step 1 is direct experimental confirmation of unitarity.
            """
        )

# ─── Q2.1 Teleportation ─────────────────────────────────────────────────────────
elif page == "Q2.1 — Teleportation":
    st.header("Q2.1 — Quantum Teleportation")
    st.markdown(
        r"""
        **Alice's state:** $|q_0\rangle = \frac{1}{\sqrt{5}}(2|0\rangle + |1\rangle)$

        **Protocol (3 qubits):**
        - q0: Alice's message qubit
        - q1: Alice's half of the Bell pair
        - q2: Bob's qubit (receives teleported state)
        """
    )

    qc = make_teleportation_circuit()

    st.subheader("Teleportation Circuit")
    fig_circ = qc.draw("mpl", style="iqp")
    fig_circ.set_size_inches(14, 4)
    st.pyplot(fig_circ)
    plt.close()

    with st.expander("Protocol Details"):
        st.markdown(
            r"""
            **Bell State Preparation (q1, q2):** H on q1, then CNOT q1→q2 creates $|\Phi^+\rangle$.

            **Bell Measurement (q0, q1):**
            1. CNOT q0→q1
            2. H on q0
            3. Measure q0, q1

            **Bob's Correction (q2):**
            - If q1 = 1: apply X to q2
            - If q0 = 1: apply Z to q2

            After corrections, q2 holds exactly Alice's original state.
            """
        )

# ─── Q2.2 Long-Distance CNOT ────────────────────────────────────────────────────
elif page == "Q2.2 — Long-Distance CNOT":
    st.header("Q2.2 — Long-Distance CNOT (q0 → q4)")
    st.markdown(
        """
        Hardware constraint: only adjacent CNOTs allowed (q_i ↔ q_{i+1}).
        To perform CNOT q0→q4, we SWAP q4 towards q1 step by step,
        execute the CNOT, then SWAP back.

        Each SWAP = 3 CNOT gates.
        """
    )

    qc, n_cnots = make_long_distance_cnot_circuit()

    st.subheader("Circuit Diagram")
    fig_circ = qc.draw("mpl", style="iqp")
    fig_circ.set_size_inches(16, 4)
    st.pyplot(fig_circ)
    plt.close()

    st.metric("Total CNOT gates used", n_cnots)

    with st.expander("Gate Count Breakdown"):
        st.markdown(
            r"""
            To CNOT q0→q4 in a linear chain (0-1-2-3-4):

            - SWAP(3,4): 3 CNOTs  → brings q4 to position 3
            - SWAP(2,3): 3 CNOTs  → brings it to position 2
            - SWAP(1,2): 3 CNOTs  → brings it to position 1
            - CNOT(0,1): 1 CNOT   → the actual gate
            - SWAP(1,2): 3 CNOTs  → restore
            - SWAP(2,3): 3 CNOTs
            - SWAP(3,4): 3 CNOTs

            **Total = 3×6 + 1 = 19 CNOT gates**
            """
        )

# ─── Q2.3 Teleport Statistics ───────────────────────────────────────────────────
elif page == "Q2.3 — Teleport Statistics":
    st.header("Q2.3 — Teleportation Statistics (|0⟩ input, 1024 shots)")
    st.markdown(
        "Alice starts with `|0⟩`. After full teleportation, Bob's qubit should also be `|0⟩`."
    )

    qc = make_teleportation_circuit(alice_state="zero")
    counts = run_circuit(qc, shots=1024)

    # Bob's qubit is q2 (rightmost in little-endian)
    bob_zero = sum(v for k, v in counts.items() if k[-1] == "0")
    bob_one  = sum(v for k, v in counts.items() if k[-1] == "1")
    total    = bob_zero + bob_one
    prob_zero = bob_zero / total

    col1, col2 = st.columns(2)
    col1.metric("Bob finds |0⟩", f"{prob_zero*100:.1f}%")
    col2.metric("Bob finds |1⟩", f"{(1-prob_zero)*100:.1f}%")

    fig_hist = plot_histogram(counts, title="Teleportation (|0⟩ input, 1024 shots)")
    st.pyplot(fig_hist)
    plt.close()

    with st.expander("Analysis"):
        st.markdown(
            r"""
            **Expected:** Bob measures |0⟩ with 100% probability.

            **Deviations** can arise from:
            - **Shot noise**: finite sampling (N=1024) introduces statistical fluctuations ~1/√N ≈ 3%
            - **Classical feed-forward** in simulation: Qiskit's `c_if` gates are ideal; no hardware noise here.

            Ideal statevector simulation gives exactly 100%; shot-based simulation will hover near 100%.
            """
        )

# ─── Q3.1 Hubbard Circuit ───────────────────────────────────────────────────────
elif page == "Q3.1 — Hubbard: Circuit":
    st.header("Q3.1 — Fermi-Hubbard Circuit (1 Trotter Step)")
    st.markdown(
        r"""
        **Qubit mapping (Jordan-Wigner):**

        | Qubit | Site | Spin |
        |-------|------|------|
        | q0 | 1 | ↑ |
        | q1 | 1 | ↓ |
        | q2 | 2 | ↑ |
        | q3 | 2 | ↓ |

        **Hamiltonian:**
        $$H = -J\sum_\sigma(c^\dagger_{1\sigma}c_{2\sigma} + h.c.) + U\sum_i n_{i\uparrow}n_{i\downarrow}$$

        After JW mapping:
        $$H_J = \frac{J}{2}(X_j X_{j+1} + Y_j Y_{j+1}), \quad H_U = \frac{U}{4}(I - Z_{j\uparrow} - Z_{j\downarrow} + Z_{j\uparrow}Z_{j\downarrow})$$
        """
    )

    J = st.slider("Hopping amplitude J", 0.1, 2.0, 1.0, 0.1)
    U = st.slider("Interaction U", 0.0, 20.0, 5.0, 0.5)
    tau = st.slider("Trotter step τ", 0.01, 1.0, 0.1, 0.01)

    qc = make_hubbard_circuit(J=J, U=U, tau=tau, n_trotter=1)

    st.subheader("One Trotter Step Circuit")
    fig_circ = qc.draw("mpl", style="iqp")
    fig_circ.set_size_inches(14, 4)
    st.pyplot(fig_circ)
    plt.close()

    with st.expander("Gate Labels"):
        st.markdown(
            r"""
            - **RXX / RYY gates** (blue/teal): Hopping term $e^{-i\frac{J\tau}{2}(XX+YY)}$
            - **RZZ gate** (orange): ZZ interaction part of $H_U$
            - **Rz gates** (green): Local Z rotations from $-Z_{j\uparrow}$ and $-Z_{j\downarrow}$ terms
            - **Z-string**: The Zj factor from the JW string appears as a single-qubit Z sandwiched
              between the XX+YY rotations for the non-nearest-neighbor spin-up hop (q0→q2).
            """
        )

# ─── Q3.2 Non-Interacting ───────────────────────────────────────────────────────
elif page == "Q3.2 — Hubbard: Non-Interacting":
    st.header("Q3.2 — Non-Interacting Dynamics (U = 0, J = 1)")
    st.markdown(
        r"""
        **Initial state:** $|1000\rangle$ (one ↑ electron at Site 1).

        **Expected:** Rabi oscillation — electron tunnels to Site 2 and back.
        Transfer complete at $\tau = \pi/2$ (half the Rabi period),
        matching a two-level system with coupling J: $\Omega_R = 2J$.
        """
    )

    n_points = 60
    taus = np.linspace(0, np.pi, n_points)
    n_trotter = st.slider("Trotter steps per τ", 1, 20, 10)

    probs_site2 = []
    for tau in taus:
        qc = make_hubbard_circuit(J=1.0, U=0.0, tau=tau, n_trotter=n_trotter,
                                  init_state="1000")
        sv = get_statevector(qc)
        # |0010⟩ in Qiskit little-endian = index where q2=1, others 0
        # binary: q3q2q1q0 = 0100 = 4
        idx = int("0100", 2)
        probs_site2.append(float(np.abs(sv.data[idx]) ** 2))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#07090f")
    ax.set_facecolor("#07090f")
    ax.plot(taus / np.pi, probs_site2, color="#3b6ef8", lw=2.5, label="P(Site 2 ↑)")
    ax.axvline(0.5, color="#f8a83b", ls="--", lw=1.5, label=r"τ = π/2 (expected transfer)")
    ax.set_xlabel(r"τ / π", color="#e2e8f7")
    ax.set_ylabel("Probability", color="#e2e8f7")
    ax.set_title(r"Electron Transfer Probability: Site 1 → Site 2 (U=0, J=1)", color="#e2e8f7")
    ax.tick_params(colors="#e2e8f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d50")
    ax.legend(facecolor="#0f1524", labelcolor="#e2e8f7")
    st.pyplot(fig)
    plt.close()

    with st.expander("Discussion"):
        st.markdown(
            r"""
            For a two-level system with tunneling J, the Rabi frequency is $\Omega_R = 2J$.
            The period is $T = 2\pi / \Omega_R = \pi / J$.
            At $J = 1$, transfer is complete at $\tau = T/2 = \pi/2 \approx 1.57$.

            The plotted curve shows $P(\text{Site 2}) = \sin^2(J\tau)$, peaking at $\tau = \pi/2$,
            in agreement with analytic prediction.
            """
        )

# ─── Q3.3 Mott Physics ──────────────────────────────────────────────────────────
elif page == "Q3.3 — Hubbard: Mott Physics":
    st.header("Q3.3 — Strong Interactions & Mott Physics (U = 10, J = 1)")
    st.markdown(
        r"""
        **Initial state:** $|1100\rangle$ (both electrons at Site 1).

        With large U, double-occupancy is energetically costly → tunneling is suppressed.
        This is the **Mott insulator** regime.
        """
    )

    n_points = 60
    taus = np.linspace(0, np.pi, n_points)
    n_trotter = st.slider("Trotter steps per τ", 1, 20, 10)

    probs_1100, probs_0011 = [], []
    for tau in taus:
        qc = make_hubbard_circuit(J=1.0, U=10.0, tau=tau, n_trotter=n_trotter,
                                  init_state="1100")
        sv = get_statevector(qc)
        # |1100⟩ q3q2q1q0 little-endian: q0=0,q1=0,q2=1,q3=1 → binary 1100 → index 12? 
        # Qiskit: state index = q_{n-1}...q1 q0 where q0 is LSB
        # |1100⟩ means q0=1,q1=1,q2=0,q3=0 → index = 0b0011 = 3
        idx_1100 = 0b0011  # q0=1,q1=1
        # |0011⟩ means q0=0,q1=0,q2=1,q3=1 → index = 0b1100 = 12
        idx_0011 = 0b1100
        probs_1100.append(float(np.abs(sv.data[idx_1100]) ** 2))
        probs_0011.append(float(np.abs(sv.data[idx_0011]) ** 2))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#07090f")
    ax.set_facecolor("#07090f")
    ax.plot(taus / np.pi, probs_1100, color="#3b6ef8", lw=2.5, label=r"P(|1100⟩) — both at Site 1")
    ax.plot(taus / np.pi, probs_0011, color="#f83b6e", lw=2.5, label=r"P(|0011⟩) — doublon at Site 2")
    ax.set_xlabel(r"τ / π", color="#e2e8f7")
    ax.set_ylabel("Probability", color="#e2e8f7")
    ax.set_title(r"Mott Physics: U=10, J=1  — Suppressed Tunneling", color="#e2e8f7")
    ax.tick_params(colors="#e2e8f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d50")
    ax.legend(facecolor="#0f1524", labelcolor="#e2e8f7")
    st.pyplot(fig)
    plt.close()

    with st.expander("Discussion: Mott Insulator"):
        st.markdown(
            r"""
            When $U \gg J$, tunneling of a doublon (double-occupancy) is energetically penalized
            by the cost $U$. The effective tunneling rate is suppressed as $\sim J^2/U$ (second-order
            perturbation theory), leading to an extremely slow oscillation.

            This is the hallmark of a **Mott insulator**: despite having mobile electrons,
            the strong on-site repulsion U locks them in place — the material becomes insulating
            not because of a lack of carriers, but because of strong correlations.

            Compare to Q3.2 (U=0): the tunneling was fast ($T = \pi$). Here, even at $\tau = \pi$,
            the probability of the doublon transferring to Site 2 remains near zero.
            """
        )
