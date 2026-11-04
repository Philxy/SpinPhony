"""
Named hybrid modes and the pump specification of the hybrid dynamics.

A hybrid band index is only meaningful at one k (bands are energy-sorted per k), so modes are named by a
fingerprint instead: k-point, symmetry label (Gamma only, needs the structure file), phonon helicity and energy, e.g.

    Eu(-)@25.32      phonon-like, irrep Eu, L^z < 0, 25.32 meV
    mag@19.82        magnon-like mode

Pump specification (command line  --pump "<k>:<selectors>" , several specs separated by ';' or by repeating --pump):

    k          G | Gamma | idx=<grid index> | a,b,c   (fractional coordinates, multiples of 1/mesh)
    selectors  comma separated, all optional:
                 b=<int>        hybrid band index at that k (0-based, as in the CSV outputs)
                 E=<meV>        pick the mode closest to this energy (after the other filters)
                 L+ / L- / L0   sign of the phonon angular momentum (|L^z| > 0.1 counts as +/-)
                 ph / mag       phonon-like (w_mag <= 0.5) / magnon-like
                 Eu, Ag, ...    irrep label (Gamma, needs `structure` in the config); 'E' / 'A' match both parities
    examples   "G:Eu,L-,E=25.3"     "G:b=22"     "0.5,0,0:ph,E=12"     "G:mag,E=19.8"

An unresolved or ambiguous spec is an error that lists the candidates, so a bad name fails before the expensive scan.

Populations only: a degenerate doublet that the Hamiltonian does not split (e.g. Eg here, split by ~1 ueV) has no
helicity in a diagonal Boltzmann equation. Pump both members (L0 / both b=) or use the Eu doublets, which the
magnon coupling splits into one hybridised and one unhybridised helicity.
"""
import math
import re
import numpy as np

HEL_THRESHOLD = 0.1       # |L^z| above this counts as a definite helicity
E_TOLERANCE = 1.0         # meV: E=<value> must be closer than this to the nearest candidate
AMBIGUITY = 0.1           # meV: two candidates closer than this (after filtering) are indistinguishable by E
_IRREP = re.compile(r"^[ABE][gu]?$")


# ------------------------------------------------------------------------------------------ symmetry (Gamma)
def _read_structure(path):
    """Header of a phonopy-style yaml (lattice, points, ...) without the phonon data. Those files are tens of MB with
    the whole band structure on one line, which yaml.safe_load would parse for minutes, so that line is skipped."""
    import yaml
    keep = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("phonon", "path")) or len(line) > 100000:
                continue
            keep.append(line)
    return yaml.safe_load("".join(keep))


_OPS_CACHE = {}


def _structure_operators(structure_yaml, tol=0.02):
    if structure_yaml not in _OPS_CACHE:
        _OPS_CACHE[structure_yaml] = _structure_operators_uncached(structure_yaml, tol)
    return _OPS_CACHE[structure_yaml]


def _structure_operators_uncached(structure_yaml, tol=0.02):
    """Cartesian representation matrices of inversion and C3(z) on the 3*N_atoms displacement space, or None."""
    st = _read_structure(structure_yaml)
    lat = np.array(st["lattice"], dtype=float)
    pts = st["points"]
    frac = np.array([p["coordinates"] for p in pts], dtype=float)
    sym = [p["symbol"] for p in pts]
    cart = frac @ lat
    inv_lat = np.linalg.inv(lat)
    n = len(pts)

    def perm_of(op):
        out = []
        for i in range(n):
            f = op(cart[i]) @ inv_lat
            hit = [j for j in range(n) if sym[j] == sym[i]
                   and np.abs((f - frac[j]) - np.round(f - frac[j])).max() < tol]
            if len(hit) != 1:
                return None
            out.append(hit[0])
        return out

    inv_perm = None
    for i in range(n):
        for j in range(i, n):
            if sym[i] == sym[j]:
                c = 0.5 * (cart[i] + cart[j])
                inv_perm = perm_of(lambda r: 2 * c - r)
                if inv_perm is not None:
                    break
        if inv_perm is not None:
            break
    a = 2 * math.pi / 3
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    c3_perm = None
    for i in range(n):
        c = cart[i].copy()
        c3_perm = perm_of(lambda r: Rz @ (r - c) + c)
        if c3_perm is not None:
            break
    if inv_perm is None or c3_perm is None:
        return None

    def rep(perm, R):
        D = np.zeros((3 * n, 3 * n), dtype=complex)
        for i, j in enumerate(perm):
            D[3 * j:3 * j + 3, 3 * i:3 * i + 3] = R
        return D

    return rep(inv_perm, -np.eye(3)), rep(c3_perm, Rz)


def _is_gamma(cd, q_idx):
    return not np.any(np.asarray(cd.q_grid[q_idx]) % np.asarray(cd.mesh))


# --------------------------------------------------------------------------------------------- catalogue
def mode_catalogue(cd, q_idx, props, structure=None, min_energy=0.2):
    """List of dicts, one per hybrid band at grid point q_idx.  props = cd.extract_full_grid_hybrid_properties()."""
    phon_ch, mag_ch, phon_am, spin_am = props
    Np, Nm = cd.phon_branches, cd.n_mag_branches
    Nb = Np + Nm
    Q = cd.Qmatrix[q_idx]
    eps = cd.w_hyb[q_idx]

    ops = None
    if structure and _is_gamma(cd, q_idx):
        try:
            ops = _structure_operators(structure)
        except Exception as exc:                       # missing yaml / pyyaml: names fall back to energy + helicity
            print(f" mode_id: no symmetry labels ({exc})")
        if ops is None and structure:
            print(" mode_id: could not find inversion and C3(z) in the structure; no irrep labels")
    E = cd.eig_phon[q_idx].reshape(Np, Np) if ops is not None else None

    cat = []
    for mu in range(Nb):
        wm = float(mag_ch[q_idx, mu])
        lz = float(phon_am[q_idx, mu])
        kind = "mag" if wm > 0.5 else "ph"
        hel = "+" if lz > HEL_THRESHOLD else ("-" if lz < -HEL_THRESHOLD else "0")
        irrep, c3 = None, float("nan")
        if ops is not None and kind == "ph":
            D_I, D_C3 = ops
            P = Q[:Np, mu] + Q[Nb:Nb + Np, mu]         # displacement amplitude of the mode in the bare branch basis
            u = P @ E
            nrm = np.vdot(u, u).real
            if nrm > 1e-12:
                c3 = float(np.angle(np.vdot(u, D_C3 @ u) / nrm) / (2 * math.pi / 3))
                par = np.vdot(u, D_I @ u).real / nrm
                irrep = ("A" if abs(c3) < 0.3 else "E") + ("g" if par > 0 else "u")
        if kind == "mag":
            label = f"mag@{eps[mu]:.2f}"
        else:
            label = f"{irrep if irrep else 'ph'}({hel})@{eps[mu]:.2f}"
        cat.append(dict(band=mu, eps=float(eps[mu]), w_mag=wm, kind=kind, Lz=lz, Sz=float(spin_am[q_idx, mu]),
                        hel=hel, irrep=irrep, c3=c3, label=label, active=bool(eps[mu] >= min_energy)))
    return cat


def print_catalogue(cd, q_idx, cat):
    f = cd.q_grid[q_idx] / cd.mesh
    print(f"\n Mode catalogue at grid point {q_idx}  (q_frac = {np.round(f, 4).tolist()})")
    print(f" {'band':>4} {'eps/meV':>9} {'w_mag':>7} {'L^z':>6} {'S^z':>6} {'C3':>5}  name")
    for m in cat:
        c3 = f"{m['c3']:+.1f}" if np.isfinite(m["c3"]) else "   -"
        print(f" {m['band']:4d} {m['eps']:9.3f} {m['w_mag']:7.3f} {m['Lz']:+6.2f} {m['Sz']:+6.2f} {c3:>5}  "
              f"{m['label']}{'' if m['active'] else '   (inactive: below the scattering cutoff)'}")
    print("  C3: phase of <C3> in units of 2pi/3 (+-1 circular state; +-1.5 linear mix of a doublet)\n")


# ------------------------------------------------------------------------------------------ spec parsing
def parse_k(kspec, cd):
    k = kspec.strip()
    if k.lower() in ("g", "gamma", "0"):
        return int(cd.grid_map[0, 0, 0])
    if k.lower().startswith("idx="):
        idx = int(k[4:])
        if not 0 <= idx < cd.N:
            raise SystemExit(f"--pump: grid index {idx} outside 0..{cd.N - 1}")
        return idx
    try:
        fr = [float(x) for x in k.split(",")]
        assert len(fr) == 3
    except Exception:
        raise SystemExit(f"--pump: cannot read k-point '{kspec}' (use G, idx=<n>, or three fractional coordinates 'a,b,c')")
    mesh = np.asarray(cd.mesh)
    g = np.asarray(fr) * mesh
    if np.abs(g - np.round(g)).max() > 1e-6:
        raise SystemExit(f"--pump: k = {fr} is not on the {mesh.tolist()} mesh (needs integer multiples of 1/mesh)")
    gi = np.round(g).astype(int) % mesh
    return int(cd.grid_map[gi[0], gi[1], gi[2]])


def resolve_spec(spec, cd, props, structure=None, min_energy=0.2, verbose=True):
    """One '<k>:<selectors>' spec -> dict(q_idx, band, label, eps)."""
    kspec, _, sel = spec.partition(":")
    q_idx = parse_k(kspec, cd)
    cat = mode_catalogue(cd, q_idx, props, structure, min_energy)
    cand = list(cat)
    E_want = None
    for tok in [t.strip() for t in sel.split(",") if t.strip()]:
        if tok.lower().startswith("b="):
            b = int(tok[2:])
            cand = [m for m in cand if m["band"] == b]
        elif tok.lower().startswith("e="):
            E_want = float(tok[2:])
        elif tok in ("L+", "L-", "L0"):
            cand = [m for m in cand if m["hel"] == tok[1]]
        elif tok.lower() in ("ph", "mag"):
            cand = [m for m in cand if m["kind"] == tok.lower()]
        elif _IRREP.match(tok):
            if all(m["irrep"] is None for m in cat):
                raise SystemExit(f"--pump: irrep '{tok}' requested but no symmetry labels exist at this k-point "
                                 f"(labels exist at Gamma and need `structure = ...` in the material's config)")
            cand = [m for m in cand if m["irrep"] is not None and (m["irrep"] == tok or m["irrep"][0] == tok)]
        else:
            raise SystemExit(f"--pump: unknown selector '{tok}' in '{spec}' (b=, E=, L+, L-, L0, ph, mag, irrep)")
    cand = [m for m in cand if m["active"]]
    listing = lambda ms: "\n".join(f"      band {m['band']:3d}  {m['label']:18s} w_mag={m['w_mag']:.3f} L^z={m['Lz']:+.2f}" for m in ms)
    if not cand:
        raise SystemExit(f"--pump: '{spec}' matches no active mode. Modes at that k-point:\n{listing([m for m in cat if m['active']])}")
    if E_want is not None:
        cand.sort(key=lambda m: abs(m["eps"] - E_want))
        if abs(cand[0]["eps"] - E_want) > E_TOLERANCE:
            raise SystemExit(f"--pump: nearest mode to E={E_want} meV in '{spec}' is {cand[0]['eps']:.2f} meV (> {E_TOLERANCE} meV). "
                             f"Candidates:\n{listing(cand[:6])}")
        cand = [m for m in cand if abs(m["eps"] - cand[0]["eps"]) < AMBIGUITY]
    if len(cand) != 1:
        raise SystemExit(f"--pump: '{spec}' is ambiguous ({len(cand)} modes); add L+/L-, b=<band> or E=:\n{listing(cand[:8])}")
    m = cand[0]
    if verbose:
        print(f" pump '{spec}' -> q_idx {q_idx}, band {m['band']}: {m['label']}  (eps = {m['eps']:.3f} meV, "
              f"w_mag = {m['w_mag']:.3f}, L^z = {m['Lz']:+.2f})")
    return dict(q_idx=q_idx, band=m["band"], label=m["label"], eps=m["eps"], w_mag=m["w_mag"], Lz=m["Lz"], spec=spec)


def parse_pump(specs, cd, props, structure=None, min_energy=0.2):
    """--pump arguments (repeatable, ';'-separated) -> list of resolved modes (no duplicates)."""
    flat = [s for a in specs for s in a.split(";") if s.strip()]
    modes, seen = [], set()
    for s in flat:
        m = resolve_spec(s, cd, props, structure, min_energy)
        key = (m["q_idx"], m["band"])
        if key in seen:
            raise SystemExit(f"--pump: mode q_idx {key[0]} band {key[1]} selected twice")
        seen.add(key)
        modes.append(m)
    return modes


def pump_occupation(modes, n_points, E_abs=None, dn=None):
    """
    Adds the pump to each mode:  dn_i = N_cells * E_abs / (len(modes) * eps_i)   (E_abs in meV per cell, shared equally)
    or the same dn quanta in every mode.  Returns the list of dn.
    """
    if (E_abs is None) == (dn is None):
        raise SystemExit("--pump needs exactly one of --pump_E (meV absorbed per cell) or --pump_dn (quanta per mode)")
    out = []
    for m in modes:
        out.append(float(dn) if dn is not None else float(n_points) * float(E_abs) / (len(modes) * m["eps"]))
    return out
