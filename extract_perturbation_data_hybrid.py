import os
import re
import csv
import logging
import math
from pathlib import Path
import tempfile

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# ── Issue #2 Fix ──────────────────────────────────────────────────────────────
# Use a whitelist of real element symbols instead of a generic [A-Za-z]{1,2}
# pattern. This prevents NBO orbital labels (BD, LP, etc.) from being
# misinterpreted as atom identifiers.
KNOWN_ELEMENTS = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
)
_elem_pattern = '|'.join(sorted(KNOWN_ELEMENTS, key=len, reverse=True))
ATOM_RE = re.compile(rf'\b({_elem_pattern})\s+(\d+)\b')


def extract_atoms_from_nbo_str(text):
    """Return a list of (symbol, index) tuples for every real atom mention
    in an NBO donor/acceptor string, using the element whitelist."""
    return [(sym, int(idx)) for sym, idx in ATOM_RE.findall(text)]


def get_target_atoms(gjf_path):
    """Parse a .gjf file to find the user-specified C-B bond indices."""
    c_idx, b_idx = None, None
    atom_map = {}
    with open(gjf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    in_molecule_spec = False
    atom_counter = 1
    for line in lines:
        line_s = line.strip()
        if not in_molecule_spec and len(atom_map) == 0 and re.match(r'^-?\d+\s+\d+$', line_s):
            in_molecule_spec = True
            continue
        if in_molecule_spec:
            if line_s == '':
                in_molecule_spec = False
                continue
            parts = line_s.split()
            if len(parts) >= 4 and parts[0].isalpha():
                atom_map[atom_counter] = parts[0].upper()
                atom_counter += 1
        elif len(atom_map) > 0 and line_s != '':
            m = re.match(r'^(?:[A-Za-z]\s+)?(\d+)\s+(\d+)(?:\s+[A-Za-z]+)?$', line_s)
            if m:
                idx1, idx2 = int(m.group(1)), int(m.group(2))
                if idx1 in atom_map and idx2 in atom_map:
                    sym1, sym2 = atom_map[idx1], atom_map[idx2]
                    if (sym1 == 'C' and sym2 == 'B') or (sym1 == 'B' and sym2 == 'C'):
                        c_idx = idx1 if sym1 == 'C' else idx2
                        b_idx = idx1 if sym1 == 'B' else idx2
                        break
    return c_idx, b_idx


def get_coords(lines):
    """Extract the last Standard/Input orientation coordinate block."""
    coords = {}
    start_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if 'Standard orientation:' in lines[i] or 'Input orientation:' in lines[i]:
            start_idx = i
            break
    if start_idx == -1:
        return coords

    ATOMIC_NUM_TO_SYM = {
        1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
        14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
    }

    i = start_idx + 5
    while i < len(lines) and '-------' not in lines[i]:
        parts = lines[i].split()
        if len(parts) >= 6:
            try:
                idx = int(parts[0])
                atomic_num = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                sym = ATOMIC_NUM_TO_SYM.get(atomic_num, 'X')
                coords[idx] = (sym, x, y, z)
            except ValueError:
                pass
        i += 1
    return coords


def dist(a, b):
    """Euclidean distance between two coordinate tuples (sym, x, y, z)."""
    return math.sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2)


def get_rdkit_neighbors(coords_dict, c_idx, b_idx):
    """Use RDKit to infer connectivity from XYZ and return neighbor lists."""
    if not coords_dict:
        return None, None

    keys = sorted(coords_dict.keys())

    fd, temp_xyz = tempfile.mkstemp(suffix='.xyz')
    with os.fdopen(fd, 'w') as f:
        f.write(f"{len(keys)}\nTemp\n")
        for k in keys:
            sym, x, y, z = coords_dict[k]
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")
    try:
        mol = Chem.MolFromXYZFile(temp_xyz)
        if mol is None:
            raise ValueError("RDKit parse failed")
        rdDetermineBonds.DetermineConnectivity(mol)

        c_atom = mol.GetAtomWithIdx(c_idx - 1)
        c_neighbors = []
        for nbr in c_atom.GetNeighbors():
            n_idx = nbr.GetIdx() + 1
            if n_idx != b_idx:
                c_neighbors.append((nbr.GetSymbol(), n_idx))

        b_atom = mol.GetAtomWithIdx(b_idx - 1)
        b_neighbors = []
        for nbr in b_atom.GetNeighbors():
            n_idx = nbr.GetIdx() + 1
            if nbr.GetSymbol() == 'O':
                b_neighbors.append(('O', n_idx))

        os.remove(temp_xyz)
        return c_neighbors, b_neighbors
    except Exception:
        if os.path.exists(temp_xyz):
            os.remove(temp_xyz)
        return None, None


def process_nbo_log(log_path):
    """Read a Gaussian NBO log, returning coords, perturbation lines, warnings,
    and a flag indicating whether the job terminated normally."""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # ── Issue #3 Fix ──────────────────────────────────────────────────────
    # Check for normal termination.  We return the flag so the caller can
    # decide whether to skip or mark the data as unreliable.
    normal_term = False
    has_perturbation = False
    for line in lines[-500:]:
        if "Normal termination" in line:
            normal_term = True
            break
    for line in lines:
        if "Second Order Perturbation Theory Analysis of Fock Matrix" in line:
            has_perturbation = True

    file_warnings = []
    if not normal_term:
        file_warnings.append("Abnormal termination detected — data may be unreliable")
    if not has_perturbation:
        file_warnings.append("No perturbation section found in log")

    coords = get_coords(lines)
    if not coords:
        file_warnings.append("Could not find atomic coordinates in log")

    # Collect all Second Order Perturbation lines
    perturb_lines = []
    in_perturb_section = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis" in line:
            in_perturb_section = True
            i += 5  # skip table headers
            continue
        if in_perturb_section:
            if ("NATURAL LOCALIZED MOLECULAR ORBITAL" in line
                    or "NBO summary" in line
                    or line.strip() == "1|1|UNPC"
                    or "Job cpu time" in line):
                in_perturb_section = False
            else:
                perturb_lines.append(line.rstrip('\n'))
        i += 1

    return coords, perturb_lines, file_warnings, normal_term


def find_bpin_bridge_carbons(coords, b_idx, o_indices):
    """Identify the two C atoms on the Bpin ring that are directly bonded to the
    two O atoms (one C per O).  These are needed so that the C–C bond between
    them can be classified into Σ E(Bpin).

    Strategy: for each O in o_indices, find the nearest C atom (excluding the
    target B's own C neighbor which is on the skeleton side)."""
    bridge_c = []
    for o_idx in o_indices:
        if o_idx not in coords:
            continue
        o_coord = coords[o_idx]
        best_dist = 999.0
        best_c = None
        for k, v in coords.items():
            if v[0] == 'C' and k != b_idx:
                d = dist(o_coord, v)
                # O-C covalent bond ~ 1.4 Å; use generous 1.6 Å cutoff
                if d < best_dist and d < 1.6:
                    best_dist = d
                    best_c = k
        if best_c is not None:
            bridge_c.append(('C', best_c))
    return bridge_c


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    base_dir = Path(".")
    atoms_dir = base_dir / "Atoms"

    if not atoms_dir.exists():
        logging.error("Atoms folder not found in current directory")
        return

    csv_data = []
    warning_data = []
    txt_classified = []
    txt_high_energy = []

    # Only process .log files inside subdirectories (not the root dir)
    log_paths = [
        p for p in base_dir.rglob("*")
        if p.suffix.lower() == '.log'
        and p.parent != base_dir
        and "Atoms" not in p.parts
        and "1" not in p.parts
        and "old" not in p.parts
    ]

    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', str(text))]

    log_paths.sort(key=lambda p: natural_keys(p.parent.name))

    for log_path in log_paths:
        structure_name = log_path.parent.name

        # Locate matching .gjf
        gjf_dir = atoms_dir / structure_name
        gjf_file = None
        if gjf_dir.exists():
            for f in gjf_dir.glob("*.gjf"):
                gjf_file = f
                break

        c_idx, b_idx = None, None
        if not gjf_file:
            warning_data.append({
                'Structure': structure_name, 'File': log_path.name,
                'Warning': "Matching .gjf file not found"
            })
        else:
            c_idx, b_idx = get_target_atoms(gjf_file)
            if c_idx is None or b_idx is None:
                warning_data.append({
                    'Structure': structure_name, 'File': log_path.name,
                    'Warning': f"Failed to extract C-B indices from {gjf_file.name}"
                })

        coords, perturb_lines, file_warnings, normal_term = process_nbo_log(log_path)

        for w in file_warnings:
            warning_data.append({
                'Structure': structure_name, 'File': log_path.name,
                'Warning': w
            })

        # ── Issue #3 extended fix ─────────────────────────────────────────
        # Skip files that did not terminate normally — extracted data is
        # likely incomplete or corrupt.
        if not normal_term:
            warning_data.append({
                'Structure': structure_name, 'File': log_path.name,
                'Warning': "SKIPPED: abnormal termination, data unreliable"
            })
            continue

        if not coords:
            continue

        # ── Issue #5 Fix ──────────────────────────────────────────────────
        # Cross-validate that the atom indices from the .gjf actually match
        # the expected element symbols in the coordinate table.  Gaussian
        # may reorder atoms, causing a C-index to point at a different
        # element in the log.
        if c_idx is not None and b_idx is not None:
            c_sym_in_coords = coords.get(c_idx, ('?',))[0]
            b_sym_in_coords = coords.get(b_idx, ('?',))[0]
            if c_sym_in_coords != 'C' or b_sym_in_coords != 'B':
                warning_data.append({
                    'Structure': structure_name, 'File': log_path.name,
                    'Warning': (
                        f"Index mismatch: gjf says C={c_idx}/B={b_idx}, but "
                        f"log coords show atom {c_idx}={c_sym_in_coords}, "
                        f"atom {b_idx}={b_sym_in_coords}. "
                        "Falling back to coordinate-distance search."
                    )
                })
                # Invalidate the indices so the distance fallback triggers
                c_idx, b_idx = None, None

        # Build the list of C-B bonds to analyse
        all_found_bonds = []
        if c_idx is not None and b_idx is not None:
            all_found_bonds.append({'c_idx': c_idx, 'b_idx': b_idx})
        else:
            # Fallback: find all C-B bonds by coordinate distance < 1.8 Å
            for k1, v1 in coords.items():
                if v1[0] == 'C':
                    for k2, v2 in coords.items():
                        if v2[0] == 'B' and dist(v1, v2) < 1.8:
                            all_found_bonds.append({'c_idx': k1, 'b_idx': k2})
            if not all_found_bonds:
                warning_data.append({
                    'Structure': structure_name, 'File': log_path.name,
                    'Warning': "Fallback failed: no C-B bonds found by distance"
                })
                continue
            else:
                warning_data.append({
                    'Structure': structure_name, 'File': log_path.name,
                    'Warning': (f"Distance fallback: {len(all_found_bonds)} "
                                f"C-B bond(s) identified")
                })

        # ─────────────────────────────────────────────────────────────────
        for bond_i, b_info in enumerate(all_found_bonds):
            suffix = f"_{bond_i + 1}" if len(all_found_bonds) > 1 else ""
            struct_display = f"{structure_name}{suffix}"

            curr_c = b_info['c_idx']
            curr_b = b_info['b_idx']

            if curr_c not in coords or curr_b not in coords:
                warning_data.append({
                    'Structure': struct_display, 'File': log_path.name,
                    'Warning': "Target C or B index not found in coordinates"
                })
                continue

            # ── Primary method: Euclidean distance ────────────────────────
            dists_c = []
            for k, v in coords.items():
                if k != curr_c and k != curr_b:
                    dists_c.append((dist(coords[curr_c], v), k, v[0]))
            dists_c.sort()
            c_neighbors = [(sym, k) for d, k, sym in dists_c[:3]]

            dists_b_o = []
            for k, v in coords.items():
                if v[0] == 'O':
                    dists_b_o.append((dist(coords[curr_b], v), k, v[0]))
            dists_b_o.sort()
            b_neighbors = [('O', k) for d, k, sym in dists_b_o[:2]]

            # ── Validation method: RDKit topology ─────────────────────────
            c_nbr_rdkit, b_nbr_rdkit = get_rdkit_neighbors(coords, curr_c, curr_b)
            if c_nbr_rdkit is None:
                warning_data.append({
                    'Structure': struct_display, 'File': log_path.name,
                    'Warning': 'RDKit inference failed; using distance method only'
                })
            else:
                if set(c_nbr_rdkit) != set(c_neighbors):
                    warning_data.append({
                        'Structure': struct_display, 'File': log_path.name,
                        'Warning': (f"Topology discrepancy: C-neighbors "
                                    f"RDKit={set(c_nbr_rdkit)}, "
                                    f"Distance={set(c_neighbors)}")
                    })
                if set(b_nbr_rdkit) != set(b_neighbors):
                    warning_data.append({
                        'Structure': struct_display, 'File': log_path.name,
                        'Warning': (f"Topology discrepancy: B-neighbors "
                                    f"RDKit={set(b_nbr_rdkit)}, "
                                    f"Distance={set(b_neighbors)}")
                    })

            if len(c_neighbors) != 3:
                warning_data.append({
                    'Structure': struct_display, 'File': log_path.name,
                    'Warning': f"C neighbor count != 3 (got {len(c_neighbors)})"
                })
            if len(b_neighbors) != 2:
                warning_data.append({
                    'Structure': struct_display, 'File': log_path.name,
                    'Warning': f"B-O neighbor count != 2 (got {len(b_neighbors)})"
                })

            # ── Issue #1 Fix ──────────────────────────────────────────────
            # Identify the two C atoms on the Bpin ring that are bonded to
            # the two O atoms.  Donors involving these C atoms (e.g. the
            # C–C bond bridging the two Bpin ring carbons) should be
            # classified into Σ E(Bpin), not left uncategorised.
            o_indices = [idx for (_, idx) in b_neighbors]
            bpin_bridge_c = find_bpin_bridge_carbons(coords, curr_b, o_indices)

            skel_atoms_set = {('C', curr_c)} | set(c_neighbors)
            bpin_atoms_set = set(b_neighbors) | set(bpin_bridge_c)

            skel_lines_buf = []
            bpin_lines_buf = []
            high_energy_buf = []

            sum_e_skel = 0.0
            sum_e_bpin = 0.0
            f_skel_list = []

            for line in perturb_lines:
                m = re.match(
                    r"^\s*\d+\.\s*(.+?)\s*/\s*\d+\.\s*(.+?)"
                    r"\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)$",
                    line.strip()
                )
                if not m:
                    continue
                donor_str = m.group(1)
                acceptor_str = m.group(2)
                e2_val = float(m.group(3))
                f_val = float(m.group(5))

                # Exclude core (CR) and Rydberg (RY) orbitals
                if ("CR" in donor_str or "CR" in acceptor_str
                        or "RY" in donor_str or "RY" in acceptor_str):
                    continue

                # Acceptor must reference the target B atom
                acc_atoms = extract_atoms_from_nbo_str(acceptor_str)
                if ('B', curr_b) not in acc_atoms:
                    continue

                # Donor must NOT reference the target B atom
                don_atoms = extract_atoms_from_nbo_str(donor_str)
                if ('B', curr_b) in don_atoms:
                    continue

                if e2_val > 3.0:
                    high_energy_buf.append(line)

                in_skel = any(a in skel_atoms_set for a in don_atoms)
                # Require ALL donor atoms to be within the Bpin fragment
                # so that only the intra-ring C–C bond (e.g. C5–C8) and
                # pure O/O–C donors qualify, NOT bonds like C5–C6 where
                # one atom extends outside the Bpin ring.
                in_bpin = all(a in bpin_atoms_set for a in don_atoms)

                if in_skel:
                    skel_lines_buf.append(line)
                    sum_e_skel += e2_val
                    f_skel_list.append(f_val)
                elif in_bpin:
                    bpin_lines_buf.append(line)
                    sum_e_bpin += e2_val

            # Build classified output
            txt_classified.append(
                f"====== Structure: {struct_display} | "
                f"File: {log_path.name} ======"
            )
            txt_classified.append(
                f"--- Skel Group (C + {len(c_neighbors)} skeleton neighbors) ---"
            )
            txt_classified.extend(skel_lines_buf)
            txt_classified.append(
                f"--- Bpin Group ({len(b_neighbors)} O + "
                f"{len(bpin_bridge_c)} bridge-C neighbors) ---"
            )
            txt_classified.extend(bpin_lines_buf)
            txt_classified.append("\n")

            if high_energy_buf:
                txt_high_energy.append(
                    f"====== Structure: {struct_display} | "
                    f"File: {log_path.name} ======"
                )
                txt_high_energy.extend(high_energy_buf)
                txt_high_energy.append("")

            max_f_skel = max(f_skel_list) if f_skel_list else 0.0
            avg_f_skel = (sum(f_skel_list) / len(f_skel_list)
                          if f_skel_list else 0.0)

            # Pad/trim neighbor display strings
            c_n_str = [f"{s}{i}" for s, i in c_neighbors]
            while len(c_n_str) < 3:
                c_n_str.append("N/A")
            c_n_str = c_n_str[:3]

            b_n_str = [f"{s}{i}" for s, i in b_neighbors]
            while len(b_n_str) < 2:
                b_n_str.append("N/A")
            b_n_str = b_n_str[:2]

            # Extra columns: Bpin bridge C atoms for transparency
            bridge_str = [f"{s}{i}" for s, i in bpin_bridge_c]
            while len(bridge_str) < 2:
                bridge_str.append("N/A")
            bridge_str = bridge_str[:2]

            csv_data.append({
                'Structure': struct_display,
                'Target_C': f"C{curr_c}",
                'Target_B': f"B{curr_b}",
                'C_Neighbor_1': c_n_str[0],
                'C_Neighbor_2': c_n_str[1],
                'C_Neighbor_3': c_n_str[2],
                'B_Neighbor_O1': b_n_str[0],
                'B_Neighbor_O2': b_n_str[1],
                'Bpin_Bridge_C1': bridge_str[0],
                'Bpin_Bridge_C2': bridge_str[1],
                'Sum_E_Skel': round(sum_e_skel, 4),
                'Sum_E_Bpin': round(sum_e_bpin, 4),
                'Sum_E_Total': round(sum_e_skel + sum_e_bpin, 4),
                'Max_F_Skel': round(max_f_skel, 4),
                'Average_F_Skel': round(avg_f_skel, 4),
            })

    # ── Write output files ────────────────────────────────────────────────

    csv_file_path = base_dir / "perturbation_summary.csv"
    with open(csv_file_path, "w", newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Structure', 'Target_C', 'Target_B',
            'C_Neighbor_1', 'C_Neighbor_2', 'C_Neighbor_3',
            'B_Neighbor_O1', 'B_Neighbor_O2',
            'Bpin_Bridge_C1', 'Bpin_Bridge_C2',
            'Sum_E_Skel', 'Sum_E_Bpin', 'Sum_E_Total',
            'Max_F_Skel', 'Average_F_Skel',
        ])
        writer.writeheader()
        writer.writerows(csv_data)

    with open(base_dir / "perturbation_classified.txt", "w",
              encoding='utf-8') as f:
        f.write("\n".join(txt_classified))

    with open(base_dir / "perturbation_high_energy.txt", "w",
              encoding='utf-8') as f:
        f.write("\n".join(txt_high_energy))

    # ── Issue #4 Fix ──────────────────────────────────────────────────────
    # Always write the warnings file.  If there are no warnings, delete any
    # stale file left from a previous run to avoid confusion.
    warnings_csv_path = base_dir / "warnings_perturbation.csv"
    if warning_data:
        with open(warnings_csv_path, "w", newline='',
                  encoding='utf-8-sig') as f:
            writer = csv.DictWriter(
                f, fieldnames=['Structure', 'File', 'Warning']
            )
            writer.writeheader()
            writer.writerows(warning_data)
    elif warnings_csv_path.exists():
        warnings_csv_path.unlink()

    print("Data extraction and compilation successfully finished.")


if __name__ == "__main__":
    main()
