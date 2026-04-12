# P452 Project 1 — Universal Quantum Computer Simulator

**Lecturer:** Cheng Chin | **Due:** April 13, 2026

A full-stack 10-qubit quantum simulation environment built with **Qiskit-Aer** (backend) and **Streamlit** (frontend).

---

## 📁 File Structure

```
quantum_simulator/
├── app.py              ← Streamlit frontend (all UI and page logic)
├── backend.py          ← Qiskit-Aer backend (all circuit construction & simulation)
├── P452_Project1.ipynb ← Jupyter/Colab notebook with all checkpoint answers
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🚀 Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

### GitHub

1. Create a new public GitHub repository
2. Push this folder:
   ```bash
   git init
   git add .
   git commit -m "P452 Project 1: Quantum Simulator"
   git remote add origin https://github.com/YOUR_USERNAME/p452-quantum-sim.git
   git push -u origin main
   ```

### Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo and set `Main file path: app.py`
4. Click **Deploy**

---

## 📋 Checkpoint Summary

| Q | Topic | Key Result |
|---|-------|------------|
| Q1.1 | Deployment | GitHub + Streamlit Cloud URLs |
| Q1.2 | Parameter Control | `Ry(π)→CNOT` gives `\|11⟩` with ~100% |
| Q1.3 | GHZ State | Only `\|0⟩` and `\|1023⟩` appear |
| Q1.4 | Unitarity | CNOT chain reversed → exact state recovery |
| Q2.1 | Teleportation | 3-qubit circuit with Bell prep & measurement |
| Q2.2 | Long CNOT | 19 total CNOT gates (6 SWAPs + 1 direct) |
| Q2.3 | Teleport Stats | Bob finds `\|0⟩` ~100% (shot noise ~3%) |
| Q3.1 | Hubbard Circuit | JW-mapped hopping + interaction gates |
| Q3.2 | Non-Interacting | Rabi oscillation, transfer at `τ = π/2` |
| Q3.3 | Mott Physics | Doublon tunneling suppressed by large U |

---

## 🧮 Physics Notes

### Jordan-Wigner Mapping (Fermi-Hubbard)

**Hopping (spin-up, q0↔q2 with JW Z-string on q1):**
```
H_J↑ = -J/2 (X₀Z₁X₂ + Y₀Z₁Y₂)
```
Implemented via: CNOT(0,1)·CNOT(2,1)·Rz·CNOT(0,1)·CNOT(2,1) in X and Y bases.

**Interaction (site 1):**
```
H_U = U/4 (I - Z₀ - Z₁ + Z₀Z₁)
```
Implemented via: Rz rotations + RZZ gate.

### Teleportation Protocol
1. Prepare Alice's qubit: `|q₀⟩ = (2|0⟩+|1⟩)/√5`
2. Bell pair: H(q1), CNOT(q1,q2)
3. Bell measurement: CNOT(q0,q1), H(q0), measure q0 and q1
4. Bob's correction: X if c1=1, Z if c0=1

### Long-Distance CNOT (q0→q4)
- SWAP(3,4), SWAP(2,3), SWAP(1,2) → bring q4 adjacent to q0
- CNOT(0,1)
- SWAP(1,2), SWAP(2,3), SWAP(3,4) → restore
- Each SWAP = 3 CNOTs → **Total: 19 CNOTs**
