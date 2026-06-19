#!/usr/bin/env python3
"""
auto_nbo_choose.py
-----------------
Automatically generate $CHOOSE keylists for NBO analysis in Gaussian .gjf files.

This script:
  1. Reads atomic coordinates from .gjf files in subdirectories (excluding Atoms/).
  2. Determines bonding connectivity using distance-based neighbour finding
     (reusing the distance/fallback logic from extract_perturbation_data_hybrid.py).
  3. Cross-validates bond types (S/D/T) against the corresponding .log NBO output.
  4. Generates a complete $CHOOSE keylist (ALPHA block) with LONE and BOND
     specifications.
  5. Writes a new .gjf file with the $CHOOSE block inserted.

Valence rules for common elements (distance-based connectivity determines
the actual bond count; listed LP counts are defaults that may be adjusted):
    H  : 1 bond,  0 LP       B  : 3 bonds, 0 LP
    C  : 4 bonds, 0 LP       N  : 3 bonds, 1 LP  (or 4+0, auto-detected)
    O  : 2 bonds, 2 LP       F  : 1 bond,  3 LP
    Si : 4 bonds, 0 LP       P  : 3 bonds, 1 LP  (or 4+0 / 5+0)
    S  : 2 bonds, 2 LP       (or 4+1 / 6+0, auto-detected)

The coordinate-reading, distance-calculation, and fallback patterns are
intentionally kept consistent with extract_perturbation_data_hybrid.py.

Requirements: Python 3.7+ (standard library only; no RDKit dependency for
basic operation, though it can be optionally enabled for validation).
"""

import os
import re
import math
import logging
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Logging configuration  (mirrors extract_perturbation_data_hybrid.py)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------------------------------------------------------------
# Known-element whitelist  (mirrors extract_perturbation_data_hybrid.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Element valence rules
#   Each entry: (expected_valence_electrons, default_lone_pairs)
#   "expected_valence_electrons" = total bonds * 2 (since each bond = 2 e⁻)
#   For atoms with variable valence we list the most common form first.
# ---------------------------------------------------------------------------
ELEMENT_VALENCE = {
    'H':  (2,  0),     # 1 bond  * 2 e⁻ per bond
    'B':  (6,  0),     # 3 bonds * 2
    'C':  (8,  0),     # 4 bonds * 2
    'N':  (6,  1),     # 3 bonds + 1 LP (amine default; ammonium = 8,0)
    'O':  (4,  2),     # 2 bonds + 2 LP
    'F':  (2,  3),     # 1 bond  + 3 LP
    'Si': (8,  0),     # 4 bonds
    'P':  (6,  1),     # 3 bonds + 1 LP (phosphine default)
    'S':  (4,  2),     # 2 bonds + 2 LP (sulfide default)
}

# Alternative valence configurations (checked when default doesn't match):
# priority order: later entries are tried after earlier ones fail
ALT_VALENCES = {
    'N': [(8, 0)],                 # ammonium / quaternary N: 4 bonds, 0 LP
    'P': [(8, 0), (10, 0)],        # phosphonium (4 bonds) / phosphorane (5 bonds)
    'S': [(8, 1), (12, 0)],        # sulfoxide (4 bonds, 1 LP) / sulfone (6 bonds, 0 LP)
}

# ---------------------------------------------------------------------------
# Distance cutoffs (Angstrom) for bond detection.
# These are deliberately generous to capture all bonded neighbours;
# element-pair-specific cutoffs can be added as needed.
# ---------------------------------------------------------------------------
MAX_BOND_DIST = {
    # H-X bonds are short
    ('H', 'H'): 0.9,
    # Default for most bonds involving hydrogen
    'H': 1.25,
    # First-row element bonds
    'first_row': 1.85,
    # Second-row (and heavier) element bonds — longer
    'heavy': 2.40,
    # Special: B-O bonds can be up to ~1.50 Å (boronate)
    ('B', 'O'): 1.65,
    ('O', 'B'): 1.65,
}

FIRST_ROW = {'H', 'B', 'C', 'N', 'O', 'F'}
HEAVY_ROW = {'Si', 'P', 'S', 'Cl', 'Br', 'I'}


def get_bond_cutoff(sym1, sym2):
    """Return the maximum distance (Angstrom) for which *sym1*–*sym2*
    is considered a covalent bond.  Reuses the hierarchical lookup
    pattern from extract_perturbation_data_hybrid.py."""

    key = (sym1, sym2)
    if key in MAX_BOND_DIST:
        return MAX_BOND_DIST[key]
    if sym1 == 'H' or sym2 == 'H':
        return MAX_BOND_DIST['H']
    if sym1 in HEAVY_ROW or sym2 in HEAVY_ROW:
        return MAX_BOND_DIST['heavy']
    return MAX_BOND_DIST['first_row']


# ---------------------------------------------------------------------------
# Distance helpers  (identical to extract_perturbation_data_hybrid.py)
# ---------------------------------------------------------------------------
def dist(a, b):
    """Euclidean distance between two (sym, x, y, z) tuples."""
    return math.sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2)


# ===========================================================================
# 1.  Coordinate reading from .gjf files
# ===========================================================================
def read_gjf_coords(gjf_path):
    """Parse a .gjf file and return:
        coords : dict  {atom_index: (symbol, x, y, z)}
        charge  : int
        multiplicity : int
        header_lines : list of str  (everything before the charge/multiplicity line)
        title_line  : str           (the title line)
    Raises ValueError on parse failure.
    """
    with open(gjf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Locate the charge / multiplicity line.
    # It is the first line that consists solely of two integers (possibly signed)
    # AFTER the route line and title line(s).
    charge_mult_re = re.compile(r'^\s*(-?\d+)\s+(\d+)\s*$')

    charge = None
    mult = None
    charge_mult_idx = -1
    title_idx = -1

    # Find the route line first (starts with '#')
    route_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            route_idx = i
            break
    if route_idx == -1:
        raise ValueError("No route line (#p ...) found")

    # Title is the next non-empty line after route
    for i in range(route_idx + 1, len(lines)):
        if lines[i].strip() and not lines[i].strip().startswith('%'):
            title_idx = i
            break
    if title_idx == -1:
        raise ValueError("No title line found after route line")

    # Charge/multiplicity line after title
    for i in range(title_idx + 1, len(lines)):
        m = charge_mult_re.match(lines[i].strip())
        if m:
            charge = int(m.group(1))
            mult = int(m.group(2))
            charge_mult_idx = i
            break
    if charge_mult_idx == -1:
        raise ValueError("Could not find charge/multiplicity line")

    # Header = everything up to and including the charge/multiplicity line
    header_lines = lines[:charge_mult_idx + 1]

    # Coordinate section starts after charge/multiplicity line
    coord_lines = lines[charge_mult_idx + 1:]

    coords = {}
    atom_idx = 1
    for line in coord_lines:
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines in coordinate section
        # Stop when we hit a line that looks like an NBO keylist
        if stripped.startswith('$'):
            break
        parts = stripped.split()
        if len(parts) >= 4:
            sym = parts[0]
            # Validate it's a real element symbol
            if re.match(r'^[A-Z][a-z]?$', sym) and sym in KNOWN_ELEMENTS:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coords[atom_idx] = (sym, x, y, z)
                    atom_idx += 1
                except ValueError:
                    pass  # not a coordinate line

    if not coords:
        raise ValueError("No atomic coordinates found in .gjf")

    return coords, charge, mult, header_lines, lines[title_idx].strip()


# ===========================================================================
# 2.  Distance-based neighbour finding
#     (mirrors the primary distance method in extract_perturbation_data_hybrid.py)
# ===========================================================================
def find_neighbours_by_distance(coords, atom_idx, atom_sym, exclude=None):
    """Return a list of (neighbour_symbol, neighbour_index, distance)
    for all atoms within bonding distance of *atom_idx*, sorted by distance.

    *exclude* is an optional set of atom indices to skip.
    """
    if exclude is None:
        exclude = set()
    exclude.add(atom_idx)
    neighbours = []
    a_coord = coords[atom_idx]
    for other_idx, other_coord in coords.items():
        if other_idx in exclude:
            continue
        cutoff = get_bond_cutoff(atom_sym, other_coord[0])
        d = dist(a_coord, other_coord)
        if d < cutoff:
            neighbours.append((other_coord[0], other_idx, d))
    neighbours.sort(key=lambda x: x[2])
    return neighbours


# ===========================================================================
# 3.  NBO log parsing — extract bond types (S / D / T) from NBO output
# ===========================================================================
def parse_nbo_bonds_from_log(log_path):
    """Extract bond-type information from an NBO log file.

    Returns:
        bond_map : dict  {(atom_i, atom_j): bond_type_char}
            where bond_type_char is 'S', 'D', or 'T'.
            The atom indices are sorted so that the smaller index comes first.

    Also returns:
        all_nbo_lines : list of str — raw NBO bond orbital lines for debugging.
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find the NBO bond orbital section.
    # It starts after "Structure accepted:" or the occupancy summary table
    # and contains lines like:
    #    N. (X.XXXXX) BD ( 1) C  13- C  14
    in_nbo_section = False
    bond_map = {}
    all_nbo_lines = []

    # Pattern for BD (bond) lines:  N. (occ) BD ( N) SYM idx - SYM idx
    bd_re = re.compile(
        r'^\s*\d+\.\s*\(\s*[\d.]+\s*\)\s*'
        r'BD\s*\(\s*(\d+)\s*\)\s*'    # BD serial number (1=σ, 2=π, 3=2nd π)
        r'(\w+)\s+(\d+)\s*-\s*'        # first atom
        r'(\w+)\s+(\d+)'               # second atom
    )

    # Pattern to detect the NBO bond orbital section start
    bond_orbital_header = re.compile(
        r'\(Occupancy\)\s+Bond orbital/ Coefficients/ Hybrids'
    )

    for i, line in enumerate(lines):
        if bond_orbital_header.search(line):
            in_nbo_section = True
            continue
        if in_nbo_section:
            # Detect end of bond-orbital section
            if ('NATURAL BOND ORBITAL ANALYSIS' in line
                    or 'Second Order Perturbation' in line
                    or 'NBO summary' in line
                    or line.strip().startswith('1|1|UNPC')
                    or 'Natural Bond Orbitals (Summary)' in line):
                in_nbo_section = False
                continue

            m = bd_re.match(line.rstrip())
            if m:
                bd_num = int(m.group(1))
                sym1, idx1 = m.group(2), int(m.group(3))
                sym2, idx2 = m.group(4), int(m.group(5))
                # Normalise order: smaller index first
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                    sym1, sym2 = sym2, sym1
                key = (idx1, idx2)
                all_nbo_lines.append(line.rstrip())

                # BD(1) = σ → 'S' bond type
                # BD(2) = π → indicates this atom pair also has a π bond,
                #           so we upgrade 'S' → 'D'
                # BD(3) = 2nd π → upgrade 'D' → 'T'
                if bd_num == 3 and key in bond_map and bond_map[key] == 'D':
                    bond_map[key] = 'T'
                elif bd_num == 2 and key in bond_map and bond_map[key] == 'S':
                    bond_map[key] = 'D'
                elif bd_num == 2:
                    bond_map[key] = 'D'
                elif bd_num == 1 and key not in bond_map:
                    bond_map[key] = 'S'

    # Also check Summary section for any bonds we might have missed
    in_summary = False
    for line in lines:
        if 'Natural Bond Orbitals (Summary):' in line:
            in_summary = True
            continue
        if in_summary:
            if ('Total unit' in line or 'Charge unit' in line
                    or '=======' in line):
                break
            m = bd_re.match(line.rstrip())
            if m:
                bd_num = int(m.group(1))
                sym1, idx1 = m.group(2), int(m.group(3))
                sym2, idx2 = m.group(4), int(m.group(5))
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                key = (idx1, idx2)
                if key not in bond_map:
                    if bd_num == 2:
                        bond_map[key] = 'D'
                    elif bd_num == 1:
                        bond_map[key] = 'S'

    return bond_map, all_nbo_lines


def check_log_termination(log_path):
    """Return True if the log file shows 'Normal termination'."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f.readlines()[-100:]:
                if 'Normal termination' in line:
                    return True
    except Exception:
        return False
    return False


# ===========================================================================
# 4.  Bond-type determination for each atom pair
# ===========================================================================
def determine_bond_type(idx1, idx2, nbo_bond_map):
    """Determine the bond type character ('S', 'D', or 'T') for a given
    atom pair, using the .log NBO data when available; otherwise default
    to 'S'.

    Parameters
    ----------
    idx1, idx2 : int
        Atom indices (1-based).
    nbo_bond_map : dict
        {(i, j): 'S'|'D'|'T'} from parse_nbo_bonds_from_log().

    Returns
    -------
    str : 'S', 'D', or 'T'
    """
    key = (min(idx1, idx2), max(idx1, idx2))
    return nbo_bond_map.get(key, 'S')


# ===========================================================================
# 5.  Valence / lone-pair logic
# ===========================================================================
def get_valence_info(sym, actual_bond_count, total_bond_order):
    """Return (expected_valence_electrons, lone_pairs) for an element,
    trying the default configuration first and falling back to alternative
    configurations if the total bond order doesn't match.

    *actual_bond_count* = number of bonded neighbours.
    *total_bond_order*  = sum of bond orders (S=1, D=2, T=3).

    The function selects the valence configuration whose expected valence
    electrons best match *total_bond_order * 2*.
    """
    candidates = [ELEMENT_VALENCE.get(sym, (total_bond_order * 2, 0))]
    candidates.extend(ALT_VALENCES.get(sym, []))

    target = total_bond_order * 2  # each bond unit = 2 valence electrons

    best = candidates[0]
    best_diff = abs(candidates[0][0] - target)
    for cand in candidates:
        diff = abs(cand[0] - target)
        if diff < best_diff:
            best_diff = diff
            best = cand

    return best  # (valence_electrons, lone_pairs)


# ===========================================================================
# 6.  Main bonding analysis for a molecule
# ===========================================================================
def analyze_bonding(coords, nbo_bond_map):
    """Determine complete bonding connectivity for all atoms.

    **Processing order matters**: atoms are processed in descending order
    of expected valence (C/Si=4, B=3, N/P/S=variable, O=2, H/F=1).
    This ensures that multi-valent atoms claim their neighbours first.

    Each atom independently claims its closest *available* neighbours
    up to its valence limit.  The ``self_assigned`` set tracks only
    which neighbours *this atom* has already bonded to during its own
    processing loop, preventing the same atom from claiming the same
    neighbour twice.  Cross-atom duplicates are resolved during the
    final deduplication pass — so we do NOT cross-mark ``assigned``
    on the target atom (doing so would cause later phases to skip
    legitimate bonding partners that had already been claimed by an
    earlier, higher-valence atom, leading to systematic under-counting
    of bonds).

    Parameters
    ----------
    coords : dict  {idx: (sym, x, y, z)}
    nbo_bond_map : dict  {(i,j): 'S'|'D'|'T'}  from .log NBO analysis

    Returns
    -------
    bonds : list of (idx1, idx2, bond_type)
        All bonds in the molecule, with idx1 < idx2.
    lone_pairs : dict  {idx: n_lp}
        Number of lone pairs on each atom.
    warnings : list of str
    """
    warnings = []
    atom_lone_pairs = {}       # idx → n_lp
    atom_bonds_selected = {}   # idx → list of (other_idx, bond_type, bond_order)

    # ---- Phase 0: find all distance-based neighbours for every atom ----
    all_neighbours = {}
    for idx, (sym, x, y, z) in coords.items():
        all_neighbours[idx] = find_neighbours_by_distance(coords, idx, sym)

    # Helper: select neighbours up to a bond-order budget.
    # During connectivity determination we treat every bond as single (bo=1);
    # the final bond types (S/D/T) are assigned later by the valence-driven
    # post-processor.  This avoids circular dependencies where bond-type
    # knowledge would be needed to determine which atoms are connected.
    def _select_neighbours(idx, sym, max_bo, nbrs):
        """Return (taken_list, bo_used) where *taken_list* is a list of
        (other_idx, bond_type='S', bond_order=1) and *bo_used* is the
        number of neighbours taken.  All bonds are provisional 'S' at
        this stage."""
        taken = []
        bo_used = 0
        for other_sym, other_idx, d in nbrs:
            if bo_used >= max_bo:
                break
            if other_idx in self_assigned[idx]:
                continue
            taken.append((other_idx, 'S', 1))
            self_assigned[idx].add(other_idx)
            bo_used += 1
        return taken, bo_used

    self_assigned = defaultdict(set)

    # ---- Phase 1: C and Si (4 neighbours max) ----
    # Bond-order warnings are deferred to Phase 6 (valence-driven
    # assignment) which handles S/D/T upgrades correctly; connectivity
    # at this stage treats all bonds as single.
    for idx, (sym, _, _, _) in coords.items():
        if sym not in ('C', 'Si'):
            continue
        nbrs = all_neighbours[idx]
        taken, bo_used = _select_neighbours(idx, sym, 4, nbrs)
        atom_bonds_selected[idx] = taken
        atom_lone_pairs[idx] = 0

    # ---- Phase 2: B (3 neighbours max) ----
    for idx, (sym, _, _, _) in coords.items():
        if sym != 'B':
            continue
        nbrs = all_neighbours[idx]
        taken, bo_used = _select_neighbours(idx, sym, 3, nbrs)
        atom_bonds_selected[idx] = taken
        atom_lone_pairs[idx] = 0

    # ---- Phase 3: N, P, S (variable valence) ----
    # During connectivity we take up to a generous maximum of neighbours
    # (all treated as single bonds).  The final bond types and valence
    # assignment are determined by the valence-driven post-processor.
    NPS_MAX_NEIGHBOURS = {'N': 4, 'P': 5, 'S': 6}
    for idx, (sym, _, _, _) in coords.items():
        if sym not in ('N', 'P', 'S'):
            continue
        nbrs = all_neighbours[idx]
        max_n = NPS_MAX_NEIGHBOURS.get(sym, 4)
        available = [(os, oi, d) for os, oi, d in nbrs
                     if oi not in self_assigned[idx]]
        taken = []
        for other_sym, other_idx, d in available:
            if len(taken) >= max_n:
                break
            taken.append((other_idx, 'S', 1))
            self_assigned[idx].add(other_idx)
        # Find best valence config based on neighbour count
        _, default_lp = get_valence_info(sym, len(taken), len(taken))
        atom_bonds_selected[idx] = taken
        atom_lone_pairs[idx] = default_lp
        if len(taken) == 0:
            warnings.append(f'Atom {sym}{idx} has no bonded neighbours')

    # ---- Phase 4: O (2 bonds) ----
    for idx, (sym, _, _, _) in coords.items():
        if sym != 'O':
            continue
        nbrs = all_neighbours[idx]
        taken, bo_used = _select_neighbours(idx, sym, 2, nbrs)
        atom_bonds_selected[idx] = taken
        atom_lone_pairs[idx] = 2
        if len(taken) == 0:
            warnings.append(f'Atom O{idx} has no bonded neighbours')
        elif len(taken) == 1:
            # May be a carbonyl O (C=O, already counted as D with 2 bond
            # orders) or a genuine alkoxide where the second expected
            # neighbour is beyond the distance cutoff.  Inspect the
            # generated $CHOOSE block to confirm.
            warnings.append(
                f'Atom O{idx} has only 1 sigma neighbour (expected 2) '
                f'— may be C=O (D bond covers 2nd bond order) or alkoxide')

    # ---- Phase 5: H and F (1 bond) ----
    for idx, (sym, _, _, _) in coords.items():
        if sym not in ('H', 'F'):
            continue
        nbrs = all_neighbours[idx]
        taken, bo_used = _select_neighbours(idx, sym, 1, nbrs)
        atom_bonds_selected[idx] = taken
        atom_lone_pairs[idx] = 0 if sym == 'H' else 3
        if len(taken) == 0:
            warnings.append(f'Atom {sym}{idx} has no bonded neighbours')

    # ---- Dedup: collect all unique bonds (all S at this stage) ----
    seen = set()
    all_bonds = []
    for idx, taken_list in atom_bonds_selected.items():
        for entry in taken_list:
            other_idx = entry[0]
            key = (min(idx, other_idx), max(idx, other_idx))
            if key not in seen:
                seen.add(key)
                all_bonds.append((key[0], key[1], 'S'))
    all_bonds.sort(key=lambda x: (x[0], x[1]))

    # ---- Phase 6: valence-driven bond-type assignment -----------------
    # All bonds start as single ('S').  We compute each atom's bond-order
    # deficit against its expected valence and iteratively upgrade shared
    # bonds (S→D, D→T) where both atoms have remaining deficit.  This is
    # the PRIMARY determinant of bond types; the .log NBO data is NOT
    # consulted here (it is unreliable for polar/dative bonds such as B-O
    # and for some carbonyls).
    all_bonds, atom_lone_pairs, valence_warnings = _assign_bond_types_valence(
        all_bonds, coords, atom_lone_pairs)
    warnings.extend(valence_warnings)

    # ---- Phase 7: phenyl-ring Kekule alternation ----------------------
    all_bonds, atom_lone_pairs, flip_warnings = _fix_phenyl_kekule(
        all_bonds, coords, atom_lone_pairs)
    warnings.extend(flip_warnings)

    return all_bonds, atom_lone_pairs, warnings


# ---------------------------------------------------------------------------
# Valence-driven bond-type assignment
# ---------------------------------------------------------------------------
def _assign_bond_types_valence(all_bonds, coords, atom_lone_pairs):
    """Assign S/D/T bond types based on valence requirements.

    Algorithm:
      1. All bonds start as 'S' (bond order = 1).
      2. For each atom, compute:
             current_bo = sum(bond_orders)
             expected_bo = expected valence (C=4, B=3, O=2, Si=4, etc.)
             deficit = expected_bo - current_bo (>0 means under-bonded)
      3. Collect all bonds (i,j) where deficit[i] > 0 AND deficit[j] > 0.
         Upgrading such a bond benefits both atoms.
      4. Upgrade bonds iteratively (S→D and D→T), updating deficits.
      5. Remaining single-sided deficits become informational warnings
         (common cases: isocyanide C needs LP, N in N₂, etc.).

    Returns (updated_bonds, updated_lone_pairs, warnings).
    """
    # Bond-type lookup
    bond_map = {}
    for i, j, bt in all_bonds:
        bond_map[(i, j)] = bt

    # Compute current bond order and expected bond order per atom
    def _bo(bt):
        return 1 if bt == 'S' else (2 if bt == 'D' else 3)

    def _expected_bo(sym, n_neighbours):
        """Return the expected total bond order for an element given
        *n_neighbours* connected atoms.  This encodes simple valence
        rules: C always 4, B always 3, O always 2, H/F always 1,
        Si always 4.  For variable-valence atoms (N, P, S) we use
        the neighbour count as a floor."""
        if sym in ('H', 'F'):
            return 1
        if sym in ('B',):
            return 3
        if sym in ('C', 'Si'):
            return 4
        if sym in ('O',):
            return 2
        # For N, P, S: expected bond order is at least the neighbour
        # count; the deficit is resolved by LP assignment if the
        # neighbour count is less than the element's normal valence.
        if sym == 'N':
            return max(n_neighbours, 3)  # amine: 3 bonds; ammonium: 4
        if sym == 'P':
            return max(n_neighbours, 3)  # phosphine: 3; phosphonium: 4
        if sym == 'S':
            return max(n_neighbours, 2)  # sulfide: 2; sulfoxide: 4; sulfone: 6
        return n_neighbours

    # Build adjacency
    adj = defaultdict(set)
    for i, j, bt in all_bonds:
        adj[i].add(j)
        adj[j].add(i)

    # Compute initial deficits
    deficit = {}
    for idx, (sym, _, _, _) in coords.items():
        current = sum(_bo(bond_map.get((min(idx, nb), max(idx, nb)), 'S'))
                      for nb in adj[idx])
        expected = _expected_bo(sym, len(adj[idx]))
        deficit[idx] = expected - current

    # Collect upgradeable bonds: bonds where BOTH atoms have deficit > 0
    # Sorted by the sum of deficits (largest first) to prioritise the
    # most under-bonded pairs.
    candidates = []
    for (i, j), bt in bond_map.items():
        if deficit.get(i, 0) > 0 and deficit.get(j, 0) > 0:
            candidates.append((deficit[i] + deficit[j], i, j, bt))

    # Upgrade iteratively from largest deficit sum to smallest.
    # Revisit each bond while both atoms still have a deficit so that a
    # bond can progress through both S -> D and D -> T when required.
    candidates.sort(key=lambda x: -x[0])
    for _, i, j, _ in candidates:
        while deficit.get(i, 0) > 0 and deficit.get(j, 0) > 0:
            current_bt = bond_map[(i, j)]
            if current_bt == 'S':
                new_bt = 'D'
            elif current_bt == 'D':
                new_bt = 'T'
            else:
                break  # already at maximum
            bond_map[(i, j)] = new_bt
            deficit[i] -= 1
            deficit[j] -= 1

    # Generate warnings for remaining deficits
    valence_warnings = []
    for idx, d in sorted(deficit.items()):
        if d > 0:
            sym = coords[idx][0]
            nb_count = len(adj[idx])
            valence_warnings.append(
                f'Atom {sym}{idx} has bond-order deficit {d} '
                f'(neighbours={nb_count}). '
                f'May need lone pair(s) or bond-type upgrade '
                f'(e.g. isocyanide C, nitro N, sulfoxide S).')

    # Update LP assignments based on remaining valence electrons
    _update_lone_pairs_from_valence(coords, adj, bond_map, atom_lone_pairs,
                                     deficit)

    # Rebuild bond list
    updated = [(a, b, bond_map[(a, b)]) for a, b, _ in all_bonds]
    return updated, atom_lone_pairs, valence_warnings


def _update_lone_pairs_from_valence(coords, adj, bond_map, atom_lone_pairs,
                                     deficit):
    """Adjust lone-pair counts based on valence accounting.

    For atoms whose bond-order deficit cannot be resolved by upgrading
    shared bonds (e.g. isocyanide C where the deficit is one-sided, or
    anionic O⁻), we assign the deficit as lone pairs.
    """
    # Standard valence-electron counts (total valence e⁻ for neutral atom)
    VALENCE_ELECTRONS = {
        'H': 1, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
        'Si': 4, 'P': 5, 'S': 6,
    }

    def _bo(bt):
        return 1 if bt == 'S' else (2 if bt == 'D' else 3)

    for idx, (sym, _, _, _) in coords.items():
        if sym not in VALENCE_ELECTRONS:
            continue
        ve_total = VALENCE_ELECTRONS[sym]
        # Electrons used in bonds (2 per bond order)
        e_bond = sum(2 * _bo(bond_map.get((min(idx, nb), max(idx, nb)), 'S'))
                     for nb in adj[idx])
        e_remaining = ve_total - e_bond
        # If remaining > 0, they become lone pairs
        lp = max(0, e_remaining // 2)
        # For elements not already handled, update based on valence accounting.
        # C is allowed LP in special cases (isocyanide :C≡N-R, carbenes, CO).
        if sym in ('H', 'B', 'Si'):
            continue  # essentially never carry LP in normal organic molecules
        if sym == 'F':
            continue  # already set to 3
        if sym == 'C':
            # C normally has 0 LP, but valence accounting may demand LP for
            # isocyanide, carbene, or carbonyl-like C atoms.  Only assign LP
            # when valence accounting unambiguously requires it (lp > 0).
            if lp > 0:
                atom_lone_pairs[idx] = lp
            continue
        if sym in ('N', 'P', 'S', 'O'):
            # Use the better of the two estimates
            current_lp = atom_lone_pairs.get(idx, 0)
            atom_lone_pairs[idx] = max(current_lp, lp)


# ---------------------------------------------------------------------------
# Post-processing: phenyl-ring Kekule alternation flip
# ---------------------------------------------------------------------------
def _fix_phenyl_kekule(all_bonds, coords, atom_lone_pairs):
    """Detect 6-membered carbon rings and, when beneficial, flip the
    Kekule S/D alternation so that ring carbons connected to the
    skeleton (ipso carbons without a hydrogen neighbour) receive a
    double bond, giving them 4 bond orders.

    Returns (updated_bonds, updated_lone_pairs, warnings).
    """
    # Build adjacency and bond-type lookup
    adj = defaultdict(set)
    bond_type = {}
    for i, j, bt in all_bonds:
        adj[i].add(j)
        adj[j].add(i)
        bond_type[(min(i, j), max(i, j))] = bt

    # Find all 6-membered rings of carbon atoms via DFS
    carbon_idxs = {idx for idx, v in coords.items() if v[0] == 'C'}
    rings = _find_c6_rings(carbon_idxs, adj)

    if not rings:
        return all_bonds, atom_lone_pairs, []

    warnings = []
    bonds_modified = set()

    for ring in rings:
        # ring is a tuple of 6 indices in cyclic order
        # Compute current bond-order per ring carbon
        bo_per_c = {}
        for c in ring:
            bo = 0
            for nb in adj[c]:
                bt = bond_type.get((min(c, nb), max(c, nb)), 'S')
                bo += 1 if bt == 'S' else (2 if bt == 'D' else 3)
            bo_per_c[c] = bo

        # If all ring carbons already have 4, skip
        if all(b == 4 for b in bo_per_c.values()):
            continue

        # Determine the best Kekule pattern: try both possible
        # alternating S/D assignments around the ring and pick the one
        # with the fewest under-bonded carbons.
        n = len(ring)
        # Ring bonds: (ring[0],ring[1]), (ring[1],ring[2]), ..., (ring[5],ring[0])
        ring_bond_pairs = [(ring[k], ring[(k + 1) % n]) for k in range(n)]

        # Pattern 0: even-index bonds are D, odd-index are S
        # Pattern 1: odd-index bonds are D, even-index are S
        best_pattern = None
        best_deficit = 999

        for pat in (0, 1):
            # Simulate bond order per carbon under this pattern
            sim_bo = {}
            for c in ring:
                bo = 0
                for nb in adj[c]:
                    pair = (min(c, nb), max(c, nb))
                    # Check if this pair is a ring bond
                    pair_idx = None
                    for k, (ra, rb) in enumerate(ring_bond_pairs):
                        if (min(ra, rb), max(ra, rb)) == pair:
                            pair_idx = k
                            break
                    if pair_idx is not None:
                        # Ring bond: use pattern
                        is_d = (pair_idx % 2 == pat)
                        bo += 2 if is_d else 1
                    else:
                        # External bond: keep original type
                        bt = bond_type.get(pair, 'S')
                        bo += 1 if bt == 'S' else (2 if bt == 'D' else 3)
                sim_bo[c] = bo
            deficit = sum(max(0, 4 - b) for b in sim_bo.values())
            if deficit < best_deficit:
                best_deficit = deficit
                best_pattern = pat

        # If the best pattern is an improvement over current, apply it
        current_deficit = sum(max(0, 4 - b) for b in bo_per_c.values())
        if best_deficit < current_deficit:
            for k, (ra, rb) in enumerate(ring_bond_pairs):
                key = (min(ra, rb), max(ra, rb))
                is_d = (k % 2 == best_pattern)
                new_bt = 'D' if is_d else 'S'
                old_bt = bond_type.get(key, 'S')
                if new_bt != old_bt:
                    bond_type[key] = new_bt
                    bonds_modified.add(key)
            if bonds_modified:
                ring_str = '-'.join(f'C{r}' for r in ring[:3]) + '...'
                warnings.append(
                    f'Flipped Kekule pattern on 6C-ring ({ring_str}): '
                    f'deficit {current_deficit} -> {best_deficit}')

    # Rebuild bond list
    updated_bonds = [(a, b, bond_type[(a, b)]) for a, b, _ in all_bonds]
    # Apply bond_type changes to the output
    for i in range(len(updated_bonds)):
        a, b, _ = updated_bonds[i]
        key = (a, b)
        if key in bonds_modified:
            updated_bonds[i] = (a, b, bond_type[key])

    return updated_bonds, atom_lone_pairs, warnings


def _find_c6_rings(carbon_idxs, adj):
    """Find all 6-membered rings composed entirely of carbon atoms.

    Uses a DFS that prunes paths longer than 6 atoms and requires the
    ring to close back to the starting atom.  Returns a list of tuples,
    each containing 6 atom indices in cyclic order.
    """
    rings = []
    visited_rings = set()

    for start in sorted(carbon_idxs):
        # DFS from this start
        stack = [(start, [start], {start})]
        while stack:
            current, path, path_set = stack.pop()
            if len(path) == 6:
                # Check if we can close back to start
                if start in adj[current] and start not in path_set - {start}:
                    # Found a 6-ring — canonicalise and check uniqueness
                    ring_tuple = tuple(path)
                    # Rotate to put smallest index first
                    min_idx = min(ring_tuple)
                    rot = ring_tuple.index(min_idx)
                    ring_canon = ring_tuple[rot:] + ring_tuple[:rot]
                    # Also check reverse direction
                    ring_rev = (ring_canon[0],) + tuple(reversed(ring_canon[1:]))
                    ring_key = min(ring_canon, ring_rev)
                    if ring_key not in visited_rings:
                        visited_rings.add(ring_key)
                        rings.append(ring_key)
                continue
            if len(path) > 6:
                continue
            for nb in adj[current]:
                if nb not in carbon_idxs:
                    continue
                if nb in path_set:
                    continue
                stack.append((nb, path + [nb], path_set | {nb}))

    return rings


# ===========================================================================
# 7.  Generate $CHOOSE keylist text
# ===========================================================================
def generate_choose_lines(coords, bonds, atom_lone_pairs, indent='  '):
    """Generate the formatted $CHOOSE keylist lines.

    Parameters
    ----------
    coords : dict
    bonds : list of (i, j, bond_type)
    atom_lone_pairs : dict  {idx: n_lp}
    indent : str
        Indentation string for the ALPHA block interior.

    Returns
    -------
    list of str — the complete $CHOOSE block lines (without trailing newline).
    """
    lines = []
    lines.append('$CHOOSE')
    lines.append('ALPHA')

    # ---- LONE section ----
    lone_entries = []
    for idx in sorted(atom_lone_pairs.keys()):
        n_lp = atom_lone_pairs[idx]
        if n_lp > 0:
            lone_entries.append(f'{idx} {n_lp}')
    if lone_entries:
        # Put all LONE entries on one line, END-terminated
        lone_str = '  '.join(lone_entries)
        lines.append(f'{indent}LONE {lone_str} END')
    else:
        lines.append(f'{indent}LONE END')

    # ---- BOND section ----
    lines.append(f'{indent}BOND')
    # Sort bonds for consistent output: by first atom, then second
    sorted_bonds = sorted(bonds, key=lambda x: (x[0], x[1]))

    # Build bond spec lines, grouping by proximity to keep lines readable
    current_line_parts = []
    MAX_PER_LINE = 6  # max bond specifiers per line for readability

    for i, j, bt in sorted_bonds:
        spec = f'{bt} {i:>3} {j:<3}'
        current_line_parts.append(spec)
        if len(current_line_parts) >= MAX_PER_LINE:
            lines.append(f'{indent}  {"  ".join(current_line_parts)}')
            current_line_parts = []

    if current_line_parts:
        lines.append(f'{indent}  {"  ".join(current_line_parts)}')

    lines.append(f'{indent}END')    # terminate BOND list
    lines.append('END')            # terminate ALPHA block
    lines.append('$END')           # terminate $CHOOSE keylist

    return lines


# ===========================================================================
# 8.  Build new .gjf file content
# ===========================================================================
def build_new_gjf(original_lines, coords, bonds, atom_lone_pairs):
    """Construct the full content for a new .gjf file that includes
    the $CHOOSE keylist after the coordinates.

    Parameters
    ----------
    original_lines : list of str
        All lines from the original .gjf file (with trailing newlines).
    coords : dict
    bonds, atom_lone_pairs : from analyze_bonding()

    Returns
    -------
    str — new .gjf file content (ready to write).
    """
    # Strip trailing whitespace/newlines from each line so we can
    # rebuild the file with a single consistent line separator.
    clean_original = [line.rstrip('\n').rstrip('\r') for line in original_lines]

    # Find where the NBO section begins
    nbo_start = -1
    for i, line in enumerate(clean_original):
        if line.strip().startswith('$NBO') or line.strip().startswith('$CHOOSE'):
            nbo_start = i
            break

    if nbo_start == -1:
        # No existing NBO section — keep everything
        prefix = clean_original[:]
    else:
        # Keep everything before the first NBO keylist
        prefix = clean_original[:nbo_start]

    # Collapse consecutive blank lines (artefact of prior buggy runs).
    # We remove ALL purely blank lines from the prefix to restore the
    # tight original format, then re-insert the strategic blank lines
    # shown in the plan-choose.md template:
    #   - one blank after the route line (#p ...)
    #   - one blank after the title line
    #   - one blank before the $NBO section (added below)
    prefix_no_blanks = [line for line in prefix if line.strip() != '']

    # Rebuild prefix with strategic blanks: locate route, title, and
    # charge/multiplicity lines by their content.
    prefix_clean = []
    charge_mult_re = re.compile(r'^\s*(-?\d+)\s+(\d+)\s*$')
    route_found = False
    title_found = False
    for i, line in enumerate(prefix_no_blanks):
        prefix_clean.append(line)
        # After the route line (starts with '#'), insert one blank
        if not route_found and line.strip().startswith('#'):
            route_found = True
            prefix_clean.append('')
            continue
        # Title is the first non-empty line after the route blank that
        # is not the charge/multiplicity line.
        if route_found and not title_found:
            if charge_mult_re.match(line.strip()):
                # charge/multiplicity — title already passed
                title_found = True
            elif line.strip() and not line.strip().startswith('%'):
                # This is the title line — insert blank after it
                title_found = True
                prefix_clean.append('')
    prefix = prefix_clean

    # Ensure exactly one blank line before the NBO section
    prefix.append('')

    # Build the NBO + CHOOSE block
    nbo_choose_lines = []
    nbo_choose_lines.append('$NBO $END')
    nbo_choose_lines.extend(generate_choose_lines(coords, bonds, atom_lone_pairs))

    # Ensure exactly one trailing blank line at end of file
    nbo_choose_lines.append('')

    # Combine prefix and NBO/CHOOSE, with single newline between lines
    result = '\n'.join(prefix + nbo_choose_lines)
    return result


# ===========================================================================
# 9.  Audit helpers
# ===========================================================================
def _extract_choose_block(content):
    """Return the $CHOOSE ... $END block from *content*, or '' if none."""
    m = re.search(r'\$CHOOSE\n(.*?)\n\$END', content, re.DOTALL)
    return m.group(0) if m else ''


def _parse_choose_block(block):
    """Parse a $CHOOSE block into {atom_idx: n_lp} and set of (i,j,bt)."""
    lp_map = {}
    bonds = set()
    # Parse LONE line
    lone_m = re.search(r'LONE\s+(.+?)\s+END', block)
    if lone_m:
        nums = lone_m.group(1).split()
        for k in range(0, len(nums), 2):
            idx = int(nums[k])
            nlp = int(nums[k + 1])
            lp_map[idx] = nlp
    # Parse BOND lines between BOND and END
    bond_section = re.search(r'BOND\s*\n(.*?)\n\s*END', block, re.DOTALL)
    if bond_section:
        for line in bond_section.group(1).split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            k = 0
            while k < len(parts):
                bt = parts[k]
                if bt in ('S', 'D', 'T') and k + 2 < len(parts):
                    i = int(parts[k + 1])
                    j = int(parts[k + 2])
                    bonds.add((min(i, j), max(i, j), bt))
                    k += 3
                else:
                    k += 1
    return lp_map, bonds


def _diff_choose_blocks(disk_block, script_block, coords):
    """Compare two $CHOOSE blocks and return a list of human-readable diffs."""
    diffs = []
    disk_lp, disk_bonds = _parse_choose_block(disk_block)
    script_lp, script_bonds = _parse_choose_block(script_block)

    # Compare LONE pairs
    all_lp_atoms = set(disk_lp.keys()) | set(script_lp.keys())
    for idx in sorted(all_lp_atoms):
        d_lp = disk_lp.get(idx, 0)
        s_lp = script_lp.get(idx, 0)
        sym = coords.get(idx, ('?',))[0] if idx in coords else '?'
        if d_lp != s_lp:
            diffs.append(f"LP {sym}{idx}: disk={d_lp} vs script={s_lp}")

    # Compare bonds
    disk_bo = {(i, j): bt for i, j, bt in disk_bonds}
    script_bo = {(i, j): bt for i, j, bt in script_bonds}
    all_pairs = set(disk_bo.keys()) | set(script_bo.keys())
    for (i, j) in sorted(all_pairs):
        d_bt = disk_bo.get((i, j))
        s_bt = script_bo.get((i, j))
        sym_i = coords.get(i, ('?',))[0] if i in coords else '?'
        sym_j = coords.get(j, ('?',))[0] if j in coords else '?'
        if d_bt is None:
            diffs.append(f"BOND {sym_i}{i}-{sym_j}{j}: disk=MISSING vs script={s_bt}")
        elif s_bt is None:
            diffs.append(f"BOND {sym_i}{i}-{sym_j}{j}: disk={d_bt} vs script=MISSING")
        elif d_bt != s_bt:
            diffs.append(f"BOND {sym_i}{i}-{sym_j}{j}: disk={d_bt} vs script={s_bt}")

    if not diffs:
        diffs.append('(differences found but too subtle for line-by-line diff)')
    return diffs[:15]  # cap at 15 to avoid log spam


# ===========================================================================
# 10. Main driver
# ===========================================================================
def main():
    """Main driver.  Supports an --audit flag:

        python auto_nbo_choose.py          # normal: write .gjf files
        python auto_nbo_choose.py --audit  # compare only, no writes
    """
    import sys
    audit_mode = '--audit' in sys.argv

    base_dir = Path('.')
    atoms_dir = base_dir / 'Atoms'

    # Collect all .gjf files in subdirectories (exclude Atoms/ and root dir)
    gjf_files = []
    for p in base_dir.rglob('*.gjf'):
        if p.parent == base_dir:
            continue  # skip root-level .gjf
        if 'Atoms' in p.parts:
            continue
        gjf_files.append(p)

    if not gjf_files:
        logging.error("No .gjf files found in subdirectories (excluding Atoms/)")
        return

    logging.info(f"Found {len(gjf_files)} .gjf files to process")

    # Track results
    warnings_all = []
    processed = 0
    skipped_no_log = 0
    skipped_no_termination = 0
    failed = 0

    for gjf_path in sorted(gjf_files):
        struct_name = gjf_path.parent.name
        gjf_name = gjf_path.name

        # ---- Find corresponding .log file ----
        log_dir = gjf_path.parent
        log_files = list(log_dir.glob('*.log'))
        log_path = log_files[0] if log_files else None

        if log_path is None:
            msg = f"No .log file found for {struct_name}/{gjf_name}"
            logging.warning(msg)
            warnings_all.append({'Structure': struct_name, 'File': gjf_name,
                                 'Warning': msg})
            skipped_no_log += 1
            continue

        if not check_log_termination(log_path):
            msg = f"Log file did not terminate normally — skipping"
            logging.warning(f"{struct_name}/{gjf_name}: {msg}")
            warnings_all.append({'Structure': struct_name, 'File': gjf_name,
                                 'Warning': msg})
            skipped_no_termination += 1
            continue

        # ---- Read .gjf ----
        try:
            coords, charge, mult, header_lines, title = read_gjf_coords(gjf_path)
        except Exception as e:
            msg = f"Failed to parse .gjf: {e}"
            logging.error(f"{struct_name}/{gjf_name}: {msg}")
            warnings_all.append({'Structure': struct_name, 'File': gjf_name,
                                 'Warning': msg})
            failed += 1
            continue

        logging.info(f"Processing {struct_name}/{gjf_name}: "
                     f"{len(coords)} atoms, charge={charge}, mult={mult}")

        # ---- Parse NBO bond data from .log ----
        nbo_bond_map, nbo_lines = parse_nbo_bonds_from_log(log_path)
        if not nbo_bond_map:
            msg = "No NBO bond data found in log file"
            logging.warning(f"{struct_name}/{gjf_name}: {msg}")
            warnings_all.append({'Structure': struct_name, 'File': gjf_name,
                                 'Warning': msg})
            # Continue anyway — will use distance-based S defaults

        logging.info(f"  Parsed {len(nbo_bond_map)} bond types from NBO log "
                     f"({sum(1 for v in nbo_bond_map.values() if v == 'D')} D, "
                     f"{sum(1 for v in nbo_bond_map.values() if v == 'S')} S)")

        # ---- Analyze bonding ----
        bonds, atom_lone_pairs, bond_warnings = analyze_bonding(coords, nbo_bond_map)

        for w in bond_warnings:
            warnings_all.append({'Structure': struct_name, 'File': gjf_name,
                                 'Warning': w})

        # ---- Validate: check for disconnected fragments ----
        # Simple BFS to see if all atoms are connected
        if bonds:
            adj = defaultdict(set)
            for i, j, _ in bonds:
                adj[i].add(j)
                adj[j].add(i)
            visited = set()
            stack = [min(coords.keys())]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                for nb in adj.get(v, set()):
                    if nb not in visited:
                        stack.append(nb)
            if len(visited) != len(coords):
                msg = (f"Disconnected fragments: {len(visited)}/{len(coords)} "
                       f"atoms connected")
                logging.warning(f"{struct_name}/{gjf_name}: {msg}")
                warnings_all.append({'Structure': struct_name, 'File': gjf_name,
                                     'Warning': msg})

        # ---- Generate new .gjf ----
        with open(gjf_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        new_content = build_new_gjf(original_lines, coords, bonds, atom_lone_pairs)

        # Build summary line
        lone_summary = {idx: n for idx, n in atom_lone_pairs.items() if n > 0}
        d_count = sum(1 for _, _, bt in bonds if bt == 'D')
        s_count = sum(1 for _, _, bt in bonds if bt == 'S')
        t_count = sum(1 for _, _, bt in bonds if bt == 'T')
        summary = (f"{len(bonds)} bonds ({s_count} S, {d_count} D, {t_count} T), "
                   f"{len(lone_summary)} atoms with LP: "
                   f"{dict(sorted(lone_summary.items()))}")

        if audit_mode:
            # ---- Audit mode: compare script-generated $CHOOSE with on-disk ----
            on_disk = ''.join(original_lines)
            # Extract the $CHOOSE block from both for comparison
            script_choose = _extract_choose_block(new_content)
            disk_choose = _extract_choose_block(on_disk)

            if script_choose != disk_choose:
                diff_info = _diff_choose_blocks(disk_choose, script_choose, coords)
                logging.warning(f"  AUDIT: {struct_name}/{gjf_name} — USER-MODIFIED")
                for d in diff_info:
                    logging.warning(f"    {d}")
            else:
                logging.info(f"  → {summary}  [audit: MATCH]")
            processed += 1
        else:
            # ---- Normal mode: write back to same file ----
            with open(gjf_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logging.info(f"  → {summary}")
            processed += 1

    # ---- Write warnings file ----
    if warnings_all:
        import csv
        warnings_path = base_dir / ('warnings_audit.csv' if audit_mode else 'warnings_choose_generation_v2.csv')
        with open(warnings_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Structure', 'File', 'Warning'])
            writer.writeheader()
            writer.writerows(warnings_all)
        logging.info(f"Warnings written to {warnings_path} ({len(warnings_all)} entries)")

    # ---- Final summary ----
    print("\n" + "=" * 60)
    mode_str = "AUDIT MODE (no files written)" if audit_mode else "WRITE MODE"
    print(f"  auto_nbo_choose.py — Summary  [{mode_str}]")
    print("=" * 60)
    print(f"  Total .gjf files found : {len(gjf_files)}")
    print(f"  Processed successfully  : {processed}")
    print(f"  Skipped (no .log)       : {skipped_no_log}")
    print(f"  Skipped (abnormal term) : {skipped_no_termination}")
    print(f"  Failed (parse error)    : {failed}")
    print(f"  Warnings                : {len(warnings_all)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
