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
    st.markdown("## ⚛ P452 Project 1 - 10 Qubit Simulator")
    st.markdown("---")
    page = st.radio(
        "Select Module",
        [
            "🏠 Home",
            "Q1.2 — Parameter Control Loop",
            "Q1.3 — GHZ State (10-qubit)",
            "Q1.4 — Unitarity & State Recovery",
            "Q2.1/2.3 — Teleportation",
            "Q2.2 — Long-Distance CNOT",
            "Q3 — Hubbard Model",
        ],
    )
    st.markdown("---")
    shots = st.slider("Simulation Shots", 256, 8192, 1024, 256)

# ─── Home ────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("⚛ 10 Qubit Quantum Simulator")
    st.markdown(
        """
        **P452 Project 1 · Nachiket Bhanushali**

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

# ─── Q1.2 Parameter Control Loop ──────────────────────────────────────────────────────
elif page == "Q1.2 — Parameter Control Loop":
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
    st.markdown("Prepared via one Hadamard on q0 followed by nine cascaded CNOT gates. Hadamard on q0 prepares")
    st.latex(r"\frac{|0\rangle + |1\rangle}{\sqrt{2}}")
    st.markdown("The first CNOT prepares a Bell pair with q0 and q1:")
    st.latex(r"\frac{|00\rangle + |11\rangle}{\sqrt{2}}")
    st.markdown(" and each subsequent CNOT entangles the next qubit with the previous ones. Circuit has 10 qubits and "
                "10 classical bits to store measurement results.")

    qc = make_ghz_circuit(10)

    st.subheader("Circuit Diagram")
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
        "As expected, only `|0000000000⟩` (= |0⟩) and `|1111111111⟩` (= |1023⟩) appear, with ~50% probability each."
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

    qc_full, sv_init, sv_after, sv_recovered = make_unitarity_circuit()

    st.subheader("Circuit Diagram")
    fig_circ = qc_full.draw("mpl", style="iqp", fold=-1)
    fig_circ.set_size_inches(22, 6)
    st.pyplot(fig_circ)
    plt.close()

    st.subheader("State Vectors")
    col1, col2, col3 = st.columns(3)
    stages = [
        ("State Preparation", sv_init),
        ("After Gate Operations", sv_after),
        ("After Reverse Operations", sv_recovered),
    ]
    for col, (title, sv) in zip([col1, col2, col3], stages):
        with col:
            st.markdown(f"**{title}**")
            amps = {
                format(i, "010b"): round(float(np.abs(amp) ** 2), 6)
                for i, amp in enumerate(sv.data)
                if abs(amp) > 1e-6
            }
            st.json(amps)

    with st.expander("Analysis: Unitarity Confirmed"):
        st.markdown(
            r"""
            - Quantum gates are **unitary matrices** (U†U = I), so every operation is reversible.
            - After applying the CNOT chain, the state vector changes but the information is preserved.
            - Applying the inverse gate sequence (reversed CNOTs) exactly recovers the original amplitudes.
            - The final probabilities being identical to Step 1 is direct experimental confirmation of unitarity.
            """
        )

# ─── Q2.1 / Q2.3 Combined — Teleportation ────────────────────────
elif page == "Q2.1/2.3 — Teleportation":
    st.header("Q2.1 & Q2.3 — Teleportation")

    # ── Alice's state controls ───────────────────────────────────────────────────
    st.subheader("Alice's Initial State")
    st.markdown(
        r"""
        Alice's qubit is prepared as $R_z(\phi)\,R_y(\theta)|0\rangle$.
        Adjust the sliders to choose any single-qubit state.
        The default produces $\frac{1}{\sqrt{5}}(2|0\rangle + |1\rangle)$.
        """
    )

    DEFAULT_THETA_Y = float(2 * np.arccos(2 / np.sqrt(5)))  # ≈ 0.9273 rad
    DEFAULT_THETA_Z = 0.0

    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        theta_y = st.slider(
            "Ry angle θ (radians)",
            min_value=0.0, max_value=float(2 * np.pi),
            value=DEFAULT_THETA_Y, step=0.01, format="%.4f",
        )
    with col_sl2:
        theta_z = st.slider(
            "Rz angle φ (radians)",
            min_value=float(-np.pi), max_value=float(np.pi),
            value=DEFAULT_THETA_Z, step=0.01, format="%.4f",
        )

    alpha = np.cos(theta_y / 2)
    beta = np.exp(1j * theta_z) * np.sin(theta_y / 2)
    col_a, col_b, col_state = st.columns([1, 1, 2])
    col_a.metric("α = ⟨0|ψ⟩", f"{alpha:.4f}")
    col_b.metric("|β| = |⟨1|ψ⟩|", f"{abs(beta):.4f}")
    col_state.latex(
        rf"|\psi\rangle = {alpha:.3f}|0\rangle"
        rf"+ {abs(beta):.3f}\,e^{{i({theta_z:.3f})}}\,|1\rangle"
    )

    st.divider()

    # ── Q2.1: Teleportation circuit ──────────────────────────────────────────────
    st.subheader("Q2.1 — Teleportation Circuit")
    st.markdown(
        "**Protocol (3 qubits):** "
        "q0 = Alice's message qubit · q1 = Alice's Bell-pair qubit · q2 = Bob's qubit"
    )

    qc_tel = make_teleportation_circuit(theta_y=theta_y, theta_z=theta_z)
    fig_tel = qc_tel.draw("mpl", style="iqp")
    fig_tel.set_size_inches(14, 4)
    st.pyplot(fig_tel)
    plt.close()

    with st.expander("Protocol Details"):
        st.markdown(
            r"""
            **Bell State Preparation (q1, q2):** H on q1, then CNOT q1→q2 creates $|\Phi^+\rangle$.

            **Bell Measurement (q0, q1):**
            1. CNOT q0→q1  2. H on q0  3. Measure q0, q1

            **Bob's Correction (q2):**
            - If c1 = 1: apply X · If c0 = 1: apply Z

            After corrections, q2 holds exactly Alice's original state $R_z(\phi)R_y(\theta)|0\rangle$.
            """
        )

    st.divider()

    # ── Q2.3: Teleportation statistics ───────────────────────────────────────────
    st.subheader("Q2.3 — Teleportation Statistics")
    st.markdown(
        f"Run the teleportation **{shots} times** with the state set above and measure Bob's qubit."
    )

    if st.button("▶ Run Simulation"):
        qc_run = make_teleportation_circuit(theta_y=theta_y, theta_z=theta_z)
        counts = run_circuit(qc_run, shots=shots)

        # Bob is q2; in the 2-bit classical register, c[0]=q0 measurement, c[1]=q1 measurement.
        # Bob's qubit is NOT in the classical register — we need a 3rd bit.
        # Re-run with an extra classical bit for Bob.
        from qiskit import QuantumRegister, ClassicalRegister

        qr_s = QuantumRegister(3, "q")
        cr_s = ClassicalRegister(3, "c")
        qc_stat = QuantumCircuit(qr_s, cr_s)
        qc_stat.ry(theta_y, 0)
        if theta_z != 0.0:
            qc_stat.rz(theta_z, 0)
        qc_stat.barrier()
        qc_stat.h(1);
        qc_stat.cx(1, 2)
        qc_stat.barrier()
        qc_stat.cx(0, 1);
        qc_stat.h(0)
        qc_stat.measure(qr_s[0], cr_s[0])
        qc_stat.measure(qr_s[1], cr_s[1])
        with qc_stat.if_test((cr_s[1], 1)):
            qc_stat.x(2)
        with qc_stat.if_test((cr_s[0], 1)):
            qc_stat.z(2)
        qc_stat.measure(qr_s[2], cr_s[2])

        counts_stat = run_circuit(qc_stat, shots=shots)
        # MSB of 3-bit string is q2 (Bob)
        bob_0 = sum(v for k, v in counts_stat.items() if k[0] == "0")
        bob_1 = sum(v for k, v in counts_stat.items() if k[0] == "1")
        total = bob_0 + bob_1

        # Expected probabilities from the input state
        p_expect_0 = float(np.abs(alpha) ** 2)
        p_expect_1 = float(np.abs(beta) ** 2)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Bob |0⟩ (measured)", f"{bob_0 / total * 100:.1f}%")
        mc2.metric("Bob |1⟩ (measured)", f"{bob_1 / total * 100:.1f}%")
        mc3.metric("Bob |0⟩ (expected)", f"{p_expect_0 * 100:.1f}%")
        mc4.metric("Bob |1⟩ (expected)", f"{p_expect_1 * 100:.1f}%")

        fig_hist = plot_histogram(counts_stat, title=f"Teleportation ({shots} shots)")
        st.pyplot(fig_hist)
        plt.close()

        with st.expander("Analysis"):
            st.markdown(
                rf"""
                **Expected from input state:**
                $P(|0\rangle) = |\alpha|^2 = {p_expect_0:.4f}$,
                $P(|1\rangle) = |\beta|^2 = {p_expect_1:.4f}$

                Any deviation from these values is due to **shot noise** (~$1/\sqrt{{N}}$ ≈ {1 / np.sqrt(shots):.3f}).
                Ideal statevector simulation gives exact agreement; finite sampling introduces fluctuations.
                """
            )
    else:
        st.info("Press **▶ Run Simulation** to execute the teleportation and measure Bob's qubit.")

# ─── Q2.2 Long-Distance CNOT ────────────────────────────────────────────────────
elif page == "Q2.2 — Long-Distance CNOT":
    st.header("Q2.2 — Long-Distance CNOT (q0 → q4)")
    st.markdown(
        """
        Hardware constraint: only adjacent CNOTs and SWAPS allowed (q_i ↔ q_{i+1}).
        To perform CNOT q0→q4, we SWAP q4 step-by-step toward q0,
        execute the CNOT, then SWAP back. Each SWAP decomposes into 3 CNOTs.
        """
    )

    qc_ld, n_cnots = make_long_distance_cnot_circuit()

    fig_ld = qc_ld.draw("mpl", style="iqp")
    fig_ld.set_size_inches(12, 3)
    st.pyplot(fig_ld)
    plt.close()

    st.metric("Total CNOT gates used", n_cnots)

    with st.expander("Gate Count Breakdown"):
        st.markdown(
            r"""
            To CNOT q0→q4 in a linear chain (0–1–2–3–4):

            - SWAP(3,4): 3 CNOTs — brings q4 to position 3
            - SWAP(2,3): 3 CNOTs — brings it to position 2
            - SWAP(1,2): 3 CNOTs — brings it to position 1
            - CNOT(0,1): 1 CNOT  — the actual gate
            - SWAP(1,2): 3 CNOTs — final 3 SWAP gates to restore q1 back to position 4
            - SWAP(2,3): 3 CNOTs
            - SWAP(3,4): 3 CNOTs

            **Total = 3×6 + 1 = 19 CNOT gates**
            """
        )

# ─── Q3 — Hubbard Model (combined) ─────────────────────────────────────────────
elif page == "Q3 — Hubbard Model":
    st.header("Q3 — Fermi-Hubbard Model")
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
        $$H = -J\sum_\sigma(c^\dagger_{1\sigma}c_{2\sigma} + \text{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow}$$

        After Jordan-Wigner mapping:
        
        Tunneling term: $$ H_J = - \tfrac{J}{2}(X_j X_{j+1} + Y_j Y_{j+1}) $$
        
        On-site Interaction term: $$ H_U = \tfrac{U}{4}(I - Z_{j\uparrow} - Z_{j\downarrow} + Z_{j\uparrow}Z_{j\downarrow})$$
        """
    )

    st.divider()

    # ── Shared parameter controls ─────────────────────────────────────────────────
    st.subheader("Parameters")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        J = st.slider("Hopping amplitude J", 0.1, 10.0, 1.0, 0.1)
    with pcol2:
        U = st.slider("Interaction U", 0.0, 100.0, 0.0, 0.5)

    init_state = st.radio(
        "Initial state",
        ["1000 — one ↑ electron at Site 1", "1100 — both spins at Site 1"],
        horizontal=True,
    )
    init_str = "1000" if init_state.startswith("1000") else "1100"

    st.divider()

    # ── Q3.1: Circuit diagram (1 Trotter step) ────────────────────────────────────
    st.subheader("Q3.1 — Circuit Diagram (1 Trotter Step)")

    tau_circ = st.slider("Trotter step size dt (for circuit display only)", 0.01, 1.0, 0.1, 0.01)
    qc_circ = make_hubbard_circuit(J=J, U=U, tau=tau_circ, n_trotter=1, init_state=init_str)
    fig_circ = qc_circ.draw("mpl", style="iqp")
    fig_circ.set_size_inches(14, 4)
    st.pyplot(fig_circ)
    plt.close()

    with st.expander("Gate Labels"):
        st.markdown(
            r"""
            - **XZX / YZY blocks**: Hopping term $e^{-i\frac{J\tau}{2}(XZX + YZY)}$ — spin-up hop q0↔q2 with JW Z-string on q1; same structure for spin-down q1↔q3
            - **Rz gates**: Local Z rotations from the $-Z_{j\uparrow}$ and $-Z_{j\downarrow}$ chemical-potential terms in $H_U$
            - **RZZ gate**: Entangling ZZ term $e^{-i\frac{U\tau}{8}Z_{j\uparrow}Z_{j\downarrow}}$ — on-site interaction
            - **Z-string**: Single-qubit Z between the two CNOT pairs enforces fermionic anti-commutation for non-adjacent hops
            """
        )

    st.divider()

    # ── Q3.2 / Q3.3: Time-evolution plots ────────────────────────────────────────
    st.subheader("Q3.2 / Q3.3 — Time Evolution")

    pcol3, pcol4 = st.columns(2)
    with pcol3:
        n_points = st.slider("Time points", 20, 100, 40, 10)
    with pcol4:
        n_trotter = st.slider("Trotter steps per time point", 1, 40, 20, 1)

    taus = np.linspace(0, np.pi, n_points)

    # State indices (Qiskit little-endian: q0 = LSB)
    # |1000⟩ → q0=0,q1=0,q2=0,q3=1 → index 0b1000 = 8
    # |0010⟩ → q0=0,q1=1,q2=0,q3=0 → index 0b0010 = 2
    # |1100⟩ → q0=0,q1=0,q2=1,q3=1 → index 0b1100 = 12
    # |0011⟩ → q0=1,q1=1,q2=0,q3=0 → index 0b0011 = 3
    # |0110⟩: q0=0,q1=1,q2=1,q3=0 → index = 0b0110 = 6
    # |1001⟩: q0=1,q1=0,q2=0,q3=1 → index = 0b0110 = 13
    idx_map = {
        "1000": 0b1000,
        "0010": 0b0010,
        "1100": 0b1100,
        "0011": 0b0011,
        "0110": 0b0110,
        "1001": 0b1001,
    }

    with st.spinner("Running Trotterized time evolution…"):
        if init_str == "1000":
            p_init, p_transfer = [], []
            for tau in taus:
                qc_t = make_hubbard_circuit(J=J, U=U, tau=tau,
                                            n_trotter=n_trotter, init_state="1000")
                sv = Statevector(qc_t)
                p_init.append(abs(sv.data[idx_map["1000"]]) ** 2)
                p_transfer.append(abs(sv.data[idx_map["0010"]]) ** 2)

            fig, ax = plt.subplots(figsize=(9, 4), facecolor="#07090f")
            ax.set_facecolor("#07090f")
            ax.plot(taus / np.pi, p_init, color="#3b6ef8", lw=2.5,
                    label=r"$P(|1000\rangle)$ — ↑ at Site 1")
            ax.plot(taus / np.pi, p_transfer, color="#f8a83b", lw=2.5,
                    label=r"$P(|0010\rangle)$ — ↑ at Site 2")
            title = rf"Single ↑ Electron Dynamics  |  J={J}, U={U}"

        else:  # 1100
            p1100, p0011, p0110, p1001 = [], [], [], []
            for tau in taus:
                qc_t = make_hubbard_circuit(J=J, U=U, tau=tau,
                                            n_trotter=n_trotter, init_state="1100")
                sv = Statevector(qc_t)
                p1100.append(abs(sv.data[idx_map["1100"]]) ** 2)
                p0110.append(abs(sv.data[idx_map["0110"]]) ** 2)
                p0011.append(abs(sv.data[idx_map["0011"]]) ** 2)
                p1001.append(abs(sv.data[idx_map["1001"]]) ** 2)

            fig, ax = plt.subplots(figsize=(9, 4), facecolor="#07090f")
            ax.set_facecolor("#07090f")
            ax.plot(taus / np.pi, p1100, color="#3b6ef8", lw=2.5,
                    label=r"$P(|1100\rangle)$ — doublon at Site 1")
            ax.plot(taus / np.pi, p0011, color="#f83b6e", lw=2.5,
                    label=r"$P(|0011\rangle)$ — doublon at Site 2")
            ax.plot(taus / np.pi, p0110, color="#f8a83b", lw=2.5,
                    label=r"$P(|0110\rangle)$ — ↓ at Site 1, ↑ at Site 2")
            ax.plot(taus / np.pi, p1001, color="#3bf86e", linestyle="--", lw=2.5,
                     label=r"$P(|1001\rangle)$ — ↑ at Site 1, ↓ at Site 2")
            title = rf"Doublon Dynamics  |  J={J}, U={U}"

    ax.set_xlabel(r"$\tau\,/\,\pi$", color="#e2e8f7")
    ax.set_ylabel("Probability", color="#e2e8f7")
    ax.set_title(title, color="#e2e8f7")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors="#e2e8f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d50")
    ax.legend(facecolor="#0f1524", labelcolor="#e2e8f7", framealpha=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("Discussion"):
        if init_str == "1000":
            st.markdown(
                rf"""
                **Initial state:** $|1000\rangle$ — one spin-up electron at Site 1.

                For a two-level system with hopping $J$, the Rabi frequency is $\Omega_R = 2J$
                and the transfer time is $\tau_{{transfer}} = \pi/(2J)$.
                At $J={J}$: $\tau_{{transfer}} = {np.pi / (2 * J):.3f}$ rad $= {0.5 / J:.3f}\,\pi$.

                {"**U = 0 (non-interacting):** The electron undergoes perfect Rabi oscillations between the two sites, with $P(\\text{Site 2}) = \\sin^2(J\\tau)$." if U == 0 else
                f"**U = {U} (interacting):** Finite U mixes in other charge sectors, modifying the oscillation amplitude and frequency compared to the pure two-level case."}
                """
            )
        else:
            st.markdown(
                rf"""
                **Initial state:** $|1100\rangle$ — both spins at Site 1 (doublon).

                The doublon must pay an energy cost $U$ to hop to Site 2.
                The effective tunneling amplitude is suppressed as $\sim J^2/U$ (second-order perturbation theory).

                {"**U = 0:** No energy penalty — fast doublon transfer, period $\\sim \\pi/J$." if U == 0 else
                f"**U = {U} ≫ J = {J}:** Strong suppression of doublon hopping — hallmark of a **Mott insulator**. The system remains frozen near $|1100\\rangle$ throughout the evolution."}
                """
            )
