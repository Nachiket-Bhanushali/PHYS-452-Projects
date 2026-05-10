import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from tqdm import tqdm


def get_bonds_pbc(Lx, Ly):
    """Generate nearest-neighbor bonds for a 2D square lattice with PBC."""
    bonds = set()
    for x in range(Lx):
        for y in range(Ly):
            i = x + y * Lx
            x_right = (x + 1) % Lx
            i_right = x_right + y * Lx
            if i != i_right: bonds.add(tuple(sorted((i, i_right))))

            y_top = (y + 1) % Ly
            i_top = x + y_top * Lx
            if i != i_top: bonds.add(tuple(sorted((i, i_top))))
    return list(bonds)


def get_sector_minimums(Lx, Ly, J_val):
    N = Lx * Ly
    bonds = get_bonds_pbc(Lx, Ly)
    J_list = [[J_val, i, j] for i, j in bonds]
    static_heis = [["xx", J_list], ["yy", J_list], ["zz", J_list]]

    candidates = []

    for Nup in tqdm(range(N + 1), desc=f"Diagonalizing {Lx}x{Ly} (N={N})"):
        # THE FIX: Explicitly set pauli=0 to use true Spin-1/2 operators!
        basis = spin_basis_1d(L=N, Nup=Nup, pauli=0)
        if basis.Ns == 0:
            continue

        H_heis = hamiltonian(static_heis, [], basis=basis, dtype=np.float64,
                             check_herm=False, check_pcon=False)

        if basis.Ns <= 3:
            E_min = np.min(H_heis.eigvalsh())
        else:
            try:
                E_min = H_heis.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
            except:
                E_min = np.min(H_heis.eigvalsh())

        Sz_total = Nup - (N / 2.0)
        candidates.append((Sz_total, E_min))

    return np.array(candidates)


if __name__ == "__main__":
    J = 1.0
    # Change H from 0 to 10*J
    H_vals = np.linspace(0, 10, 10000)

    # --- 2x2 Magnetization ---
    N_2x2 = 4
    candidates_2x2 = get_sector_minimums(2, 2, J)

    Sz_sec_2x2 = candidates_2x2[:, 0][:, np.newaxis]
    E_heis_sec_2x2 = candidates_2x2[:, 1][:, np.newaxis]
    energies_sec_2x2 = E_heis_sec_2x2 + H_vals * Sz_sec_2x2
    gs_indices_2x2 = np.argmin(energies_sec_2x2, axis=0)
    gs_Sz_2x2 = Sz_sec_2x2[gs_indices_2x2, 0]
    Mz_2x2 = np.abs(gs_Sz_2x2 / N_2x2)

    plt.figure(figsize=(9, 5.5))
    plt.plot(H_vals / J, Mz_2x2, label='2x2 Lattice', color='C0', drawstyle='steps-mid')
    plt.title('2x2 Average Magnetization vs Magnetic Field')
    plt.xlabel('H/J')
    plt.ylabel('Magnetization |<M_z>|')
    plt.axhline(0.5, color='gray', linestyle='--', label='Saturation (|M_z| = 0.5)')
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # --- 4x4 Magnetization ---
    N_4x4 = 16
    candidates_4x4 = get_sector_minimums(4, 4, J)

    Sz_sec_4x4 = candidates_4x4[:, 0][:, np.newaxis]
    E_heis_sec_4x4 = candidates_4x4[:, 1][:, np.newaxis]
    energies_sec_4x4 = E_heis_sec_4x4 + H_vals * Sz_sec_4x4
    gs_indices_4x4 = np.argmin(energies_sec_4x4, axis=0)
    gs_Sz_4x4 = Sz_sec_4x4[gs_indices_4x4, 0]
    Mz_4x4 = np.abs(gs_Sz_4x4 / N_4x4)

    plt.figure(figsize=(9, 5.5))
    plt.plot(H_vals / J, Mz_4x4, label='4x4 Lattice', color='C3', drawstyle='steps-mid')
    plt.title('4x4 Average Magnetization vs Magnetic Field')
    plt.xlabel('H/J')
    plt.ylabel('Magnetization |<M_z>|')
    plt.axhline(0.5, color='gray', linestyle='--', label='Saturation (|M_z| = 0.5)')
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()