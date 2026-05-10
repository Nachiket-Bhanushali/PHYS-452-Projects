import numpy as np
import matplotlib.pyplot as plt


def solve_mixture(gB, gBF, mB, mF, wB, wF, NB, NF, num_pts=5000):
    # Constants
    hbar = 1.0  # Dimensionless units
    r = np.linspace(0, 15, num_pts)

    # Potentials
    VB = 0.5 * mB * wB ** 2 * r ** 2
    VF = 0.5 * mF * wF ** 2 * r ** 2

    # Initial guess (non-interacting TF profiles)
    muB = (gB * NB * 3 / (4 * np.pi) * (mB * wB ** 2) ** 1.5) ** (2 / 5)  # Rough estimate
    muF = hbar * wF * (6 * NF) ** (1 / 3)

    nB = np.maximum(0, (muB - VB) / gB)
    nF = (2 * mF / hbar ** 2 * np.maximum(0, muF - VF)) ** 1.5 / (6 * np.pi ** 2)


    # Iterative solver for coupled densities
    for _ in range(10000):
        # Update mu to conserve particle number
        muB = gB * NB / (4 * np.pi * np.trapezoid(nB * r ** 2, r)) * muB  # Simple scaling
        # (In a rigorous solver, we find muB and muF such that integral(n)=N)

        nB_new = np.maximum(0, (muB - VB - gBF * nF) / gB)
        # nF_new = np.maximum(0,
        #                     (1 / (6 * np.pi ** 2)) * (np.maximum(0, 2 * mF / hbar ** 2 * (muF - VF - gBF * nB))) ** 1.5)

        # Clip the effective potential term at 0 first
        effective_mu = np.maximum(0, muF - VF - gBF * nB)
        nF_new = (1 / (6 * np.pi ** 2)) * (2 * mF / hbar ** 2 * effective_mu) ** 1.5

        nB, nF = 0.5 * (nB + nB_new), 0.5 * (nF + nF_new)  # Damping for stability

    return r, nB, nF


# Parameters: (gB, gBF, mB, mF, wB, wF, NB, NF)
params_nonint = (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1000, 1000)
params_strong_attr = (1.0, -1, 1.0, 1.0, 1.0, 1.0, 1000, 1000)
params_weak_rep = (1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1000, 1000)
params_strong_rep = (1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1000, 1000)

plt.figure(figsize=(12, 4))
for i, (p, title) in enumerate(zip([params_nonint, params_strong_attr, params_weak_rep, params_strong_rep],
                                   ["Non-interacting","Attractive", "Weak Repulsion", "Phase Separation"])):
    r, nb, nf = solve_mixture(*p)
    plt.subplot(1, 4, i + 1)
    plt.plot(r, nb, label='Bosons ($n_B$)', color='blue', linewidth=2)
    plt.plot(r, nf, label='Fermions ($n_F$)', color='red', linewidth=2)

    # Axis Labeling
    plt.xlabel(r'Radial Distance $r$ [arb. units]', fontsize=10)
    if i == 0:
        plt.ylabel(r'Density $n(r)$ [arb. units]', fontsize=10)

    plt.title(title, fontsize=12)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()