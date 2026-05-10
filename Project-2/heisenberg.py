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
            if i != i_right:
                bonds.add(tuple(sorted((i, i_right))))

            y_top = (y + 1) % Ly
            i_top = x + y_top * Lx
            if i != i_top:
                bonds.add(tuple(sorted((i, i_top))))

    return list(bonds)


def get_candidate_states(Lx, Ly, J_val, max_states_per_sector=None):
    """
    Finds the zero-field Heisenberg energies and Sz for the eigenstates.
    Returns a numpy array of pairs: (Sz_total, E_heis), sorted by E_heis.
    """
    N = Lx * Ly
    bonds = get_bonds_pbc(Lx, Ly)

    J_list = [[J_val, i, j] for i, j in bonds]
    static_heis = [["xx", J_list], ["yy", J_list], ["zz", J_list]]

    candidates = []

    for Nup in tqdm(range(N + 1), desc=f"Diagonalizing {Lx}x{Ly} (N={N})"):
        # THE FIX: pauli=0 forces true Spin-1/2 operators instead of Pauli matrices
        basis = spin_basis_1d(L=N, Nup=Nup, pauli=0)
        if basis.Ns == 0:
            continue

        H_heis = hamiltonian(static_heis, [], basis=basis, dtype=np.float64,
                             check_herm=False, check_pcon=False)

        if max_states_per_sector is None or basis.Ns <= max_states_per_sector + 2:
            E_sector = H_heis.eigvalsh()
            if max_states_per_sector is not None:
                E_sector = E_sector[:max_states_per_sector]
        else:
            try:
                E_sector = H_heis.eigsh(k=max_states_per_sector, which="SA", return_eigenvectors=False)
            except:
                E_sector = H_heis.eigvalsh()[:max_states_per_sector]

        Sz_total = Nup - (N / 2.0)
        for E in E_sector:
            candidates.append((Sz_total, E))

    candidates_array = np.array(candidates)
    sorted_indices = np.argsort(candidates_array[:, 1])
    return candidates_array[sorted_indices]


# --- Execution and Plotting ---
if __name__ == "__main__":
    J = 1.0
    # H/J goes from 0 to 10
    H_vals = np.linspace(0, 10, 2000)

    # ==========================================
    # 1. Plot Full Spectrum for 2x2 Lattice
    # ==========================================
    print("\nCalculating 2x2 Full Spectrum...")
    candidates_2x2 = get_candidate_states(2, 2, J, max_states_per_sector=None)

    Sz_2x2 = candidates_2x2[:, 0][:, np.newaxis]
    E_heis_2x2 = candidates_2x2[:, 1][:, np.newaxis]
    energies_2x2 = E_heis_2x2 + H_vals * Sz_2x2

    plt.figure(figsize=(9, 6))
    for i in range(energies_2x2.shape[0]):
        sz_val = int(Sz_2x2[i, 0])
        plt.plot(H_vals / J, energies_2x2[i] / J, label=f'State {i} ($S^z={sz_val}$)')

    plt.title('Heisenberg Model: 2x2 Lattice (Full Spectrum)')
    plt.xlabel('$H/J$')
    plt.ylabel('Energy ($E/J$)')
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="H=0 Rank")
    plt.tight_layout()

    # ==========================================
    # 2. Plot Lowest State per Sz Sector for 4x4 Lattice
    # ==========================================
    print("\nCalculating 4x4 Lowest States by Sector...")
    candidates_4x4_sectors = get_candidate_states(4, 4, J, max_states_per_sector=1)

    Sz_4x4_sec = candidates_4x4_sectors[:, 0][:, np.newaxis]
    E_heis_4x4_sec = candidates_4x4_sectors[:, 1][:, np.newaxis]
    energies_4x4_sec = E_heis_4x4_sec + H_vals * Sz_4x4_sec

    plt.figure(figsize=(9, 6))
    for i in range(energies_4x4_sec.shape[0]):
        sz_val = int(Sz_4x4_sec[i, 0])
        plt.plot(H_vals / J, energies_4x4_sec[i] / J, label=f'Lowest $S^z={sz_val}$')

    plt.title('Heisenberg Model: 4x4 Lattice (Lowest State per $S^z$ Sector)')
    plt.xlabel('$H/J$')
    plt.ylabel('Energy ($E/J$)')
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Sector Minimums")
    plt.tight_layout()

    # ==========================================
    # 3. Plot Lowest 6 Energy Levels at each H/J for 4x4
    # ==========================================
    print("\nCalculating 4x4 Full Spectrum (This may take a minute)...")
    candidates_4x4_all = get_candidate_states(4, 4, J, max_states_per_sector=None)

    Sz_4x4_all = candidates_4x4_all[:, 0][:, np.newaxis]
    E_heis_4x4_all = candidates_4x4_all[:, 1][:, np.newaxis]

    energies_4x4_all = E_heis_4x4_all + H_vals * Sz_4x4_all
    lowest_6_at_each_H = np.sort(energies_4x4_all, axis=0)[:6, :]

    plt.figure(figsize=(9, 5.5))
    for i in range(6):
        plt.plot(H_vals / J, lowest_6_at_each_H[i] / J, label=f'E{i}')

    plt.title('4x4 low-energy levels by S_z sector')
    plt.xlabel('H/J')
    plt.ylabel('Energy (units of J)')
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()

    # Show all three figures
    print("\nDone! Rendering plots...")
    plt.show()