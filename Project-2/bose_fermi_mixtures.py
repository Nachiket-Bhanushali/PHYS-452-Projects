import numpy as np
import matplotlib.pyplot as plt

def solve_mixture_rigorous(aB_a0, aBF_a0, mB_amu, mF_amu, freq_Hz, NB, NF, num_pts=5000):
    # --- Physical Constants ---
    amu = 1.660539e-27  # kg
    a0 = 5.291772e-11  # Bohr radius (m)
    hbar = 1.054571e-34  # J*s
    um = 1e-6  # Micron scale

    omega = 2 * np.pi * freq_Hz
    mB, mF = mB_amu * amu, mF_amu * amu
    aB, aBF = aB_a0 * a0, aBF_a0 * a0

    # --- Calculate Couplings (SI units) ---
    gB_si = (4 * np.pi * hbar ** 2 * aB) / mB
    mu_red = (mB * mF) / (mB + mF)
    gBF_si = (2 * np.pi * hbar ** 2 * aBF) / mu_red

    # Grid out to 50 microns
    r = np.linspace(0, 50 * um, num_pts)
    dV = 4 * np.pi * r ** 2

    VB = 0.5 * mB * omega ** 2 * r ** 2
    VF = 0.5 * mF * omega ** 2 * r ** 2

    nB, nF = np.zeros_like(r), np.zeros_like(r)

    # Iterative solver with Bisection for Rigorous Normalization
    for i in range(500):
        # 1. Solve for muB given current nF
        def get_nB(mu):
            return np.maximum(0, (mu - VB - gBF_si * nF) / gB_si)

        low, high = 0, 1e-25
        for _ in range(40):
            mid = (low + high) / 2
            if np.trapezoid(get_nB(mid) * dV, r) < NB:
                low = mid
            else:
                high = mid
        nB_new = get_nB(high)

        # 2. Solve for muF given current nB
        def get_nF(mu):
            eff_mu = np.maximum(0, mu - VF - gBF_si * nB_new)
            return (1 / (6 * np.pi ** 2)) * (2 * mF / hbar ** 2 * eff_mu) ** 1.5

        low, high = 0, 1e-25
        for _ in range(40):
            mid = (low + high) / 2
            if np.trapezoid(get_nF(mid) * dV, r) < NF:
                low = mid
            else:
                high = mid
        nF_new = get_nF(high)

        # 3. Damping for smooth convergence
        delta = np.max(np.abs(nB - nB_new))
        nB = 0.9 * nB + 0.1 * nB_new
        nF = 0.9 * nF + 0.1 * nF_new

        if delta < 1e-45 and i > 50: break

    # Returns r in microns and real density in particles per um^3
    return r / um, nB * (um ** 3), nF * (um ** 3)


# --- Parameters (85Rb and 40K) ---
mB, mF = 85.0, 40.0
trap_freq = 50.0  # Hz
NB, NF = 500000, 500000  # Using high atom counts for visible real density
aB_a0 = 100.0

regimes = [
    (0.0, "Non-interacting ($a_{BF}=0$)"),
    (-150.0, "Attractive ($a_{BF} < 0$)"),
    (10.0, "Weak Repulsion ($a_{BF} > 0$)"),
    (150.0, "Phase Separation ($a_{BF} \gg 0$)")
]

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (aBF_val, title) in enumerate(regimes):
    r_um, nb_real, nf_real = solve_mixture_rigorous(aB_a0, aBF_val, mB, mF, trap_freq, NB, NF)

    axes[i].plot(r_um, nb_real, color='blue', lw=2.5, label=r'Bosons $^{85}$Rb')
    axes[i].plot(r_um, nf_real, color='red', ls='--', lw=2.5, label=r'Fermions $^{40}$K')
    axes[i].set_title(title, fontweight='bold', fontsize=14)
    axes[i].set_xlabel(r'Radial Distance $r$ [$\mu$m]')
    axes[i].set_ylabel(r'Real Density [particles/$\mu$m$^3$]')
    axes[i].legend(frameon=False)
    axes[i].grid(alpha=0.3)
    axes[i].set_xlim(0, 50)
    axes[i].set_ylim(0, 200)

plt.suptitle(f"Real Density Profiles: $^{85}$Rb - $^{40}$K Mixture ($f = {trap_freq}$ Hz)\n"
             f"$a_B={aB_a0} a_0$, $N_B={NB}$, $N_F={NF}$", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


def solve_mixture_peak_normalized(aB_a0, aBF_a0, mB_amu, mF_amu, freq_Hz, NB, NF, num_pts=5000):
    # --- Physical Constants & Unit Conversion ---
    amu = 1.660539e-27  # kg
    a0 = 5.291772e-11  # Bohr radius (meters)
    hbar = 1.054571e-34  # J*s
    um = 1e-6  # Micron conversion

    omega = 2 * np.pi * freq_Hz
    mB, mF = mB_amu * amu, mF_amu * amu
    aB, aBF = aB_a0 * a0, aBF_a0 * a0

    # Calculate Couplings from Scattering Lengths
    gB_si = (4 * np.pi * hbar ** 2 * aB) / mB
    mu_red = (mB * mF) / (mB + mF)
    gBF_si = (2 * np.pi * hbar ** 2 * aBF) / mu_red

    # Define radial grid out to 50 microns
    r = np.linspace(0, 50 * um, num_pts)
    dV = 4 * np.pi * r ** 2

    # Potentials and coupled initial states
    VB = 0.5 * mB * omega ** 2 * r ** 2
    VF = 0.5 * mF * omega ** 2 * r ** 2
    nB, nF = np.zeros_like(r), np.zeros_like(r)

    # Iterative solver with Bisection for Rigorous Normalization
    for i in range(500):
        # Solve for muB (Boson TF limit)
        def get_nB(mu):
            return np.maximum(0, (mu - VB - gBF_si * nF) / gB_si)

        low, high = 0, 1e-24
        for _ in range(40):
            mid = (low + high) / 2
            if np.trapezoid(get_nB(mid) * dV, r) < NB:
                low = mid
            else:
                high = mid
        nB_new = get_nB(high)

        # Solve for muF (Fermion LDA)
        def get_nF(mu):
            eff_mu = np.maximum(0, mu - VF - gBF_si * nB_new)
            return (1 / (6 * np.pi ** 2)) * (2 * mF / hbar ** 2 * eff_mu) ** 1.5

        low, high = 0, 1e-24
        for _ in range(40):
            mid = (low + high) / 2
            if np.trapezoid(get_nF(mid) * dV, r) < NF:
                low = mid
            else:
                high = mid
        nF_new = get_nF(high)

        delta = np.max(np.abs(nB - nB_new))
        nB, nF = 0.9 * nB + 0.1 * nB_new, 0.9 * nF + 0.1 * nF_new
        if delta < 1e-45 and i > 50: break

    # Peak Normalization: Divide by max value to set peak to 1
    nB_peak = nB / np.max(nB) if np.max(nB) > 0 else nB
    nF_peak = nF / np.max(nF) if np.max(nF) > 0 else nF

    return r / um, nB_peak, nF_peak


# --- Run and Plot ---
# Parameters for 85Rb and 40K
mB, mF, freq, NB, NF, aB = 85.0, 40.0, 50.0, 500000, 300000, 100.0
regimes = [
    (0.0, "Non-interacting ($a_{BF}=0$)"),
    (-150.0, "Attractive ($a_{BF} < 0$)"),
    (10.0, "Weak Repulsion ($a_{BF} > 0$)"),
    (150.0, "Phase Separation ($a_{BF} \gg 0$)")
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (aBF_val, title) in enumerate(regimes):
    r_um, nb, nf = solve_mixture_peak_normalized(aB, aBF_val, mB, mF, freq, NB, NF)
    axes[i].plot(r_um, nb, color='blue', lw=2.5, label='Bosons $^{85}$Rb')
    axes[i].plot(r_um, nf, color='red', ls='--', lw=2.5, label='Fermions $^{40}$K')
    axes[i].set_title(title, fontweight='bold', fontsize=14)
    axes[i].set_xlabel(r'Radial Distance $r$ [$\mu$m]')
    axes[i].set_ylabel('Peak-Normalized Density')
    axes[i].legend(frameon=False);
    axes[i].grid(alpha=0.3)
    axes[i].set_xlim(0, 50)

plt.suptitle(f"Peak-Normalized Profiles for Phase 2 Verification\n"
             f"$N_B={NB}$, $N_F={NF}$, $f={freq}$ Hz", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
plt.show()