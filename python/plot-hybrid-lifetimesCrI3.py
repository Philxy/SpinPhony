import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# CrI3 version of plot-hybrid-lifetimesbccFe.py / plot-hybrid-lifetimesfccNi.py.
#
# Differences to the Fe script:
#   * default cell volume: bulk CrI3 rhombohedral primitive cell, 8 atoms (2 Cr + 6 I),
#     V = 281.794 A^3 (|det| of `lattice` in Inputs/CrI3/band.h5). Override with --cell_volume
#     (e.g. for a monolayer: area * effective thickness).
#   * the bcc-Fe literature points are NOT used. Optional literature rates are read from
#     two-column files (energy_meV, gamma_ps-1; comma separated, header optional):
#         --lit_mag  <file>   --lit_phon <file>
#     Without them the literature overlay and the literature G_mp are skipped.
#   * reference group velocities: Inputs/CrI3/band.h5 (phonopy `group_velocities`
#     and `frequencies` stored along G-K-M-G; THz*A = A/ps).
#   * G_mp is only meaningful for a full-BZ run (--full_bz); on a path N and the
#     weights are not a BZ sample.

# Old (hybrid_equilibrium_lifetimes.csv) -> new (hybrid_path_lifetimes.csv) column names.
# Columns are normalised to the NEW names so both file formats work.
COLUMN_ALIASES = {
    'mag_char': 'mag_character',
    'phon_char': 'phon_character',
    'phon_AM': 'phon_AM_z_hbar',
    'spin_AM': 'spin_AM_z_hbar',
}

V_CRI3 = 281.794            # A^3, bulk rhombohedral primitive cell (8 atoms), from band.h5 `lattice`


def load_spinphony_csv(path):
    # comment='#' skips the "# path_labels: ..." first line of the new format;
    # without it pandas takes that line as the header and no column is found.
    df = pd.read_csv(path, comment='#')
    df = df.rename(columns={old: new for old, new in COLUMN_ALIASES.items() if old in df.columns})

    required = ['energy_meV', 'tau_ps', 'mag_character', 'phon_character']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Error: {path} is missing column(s) {missing}.\n"
                         f"Columns found: {list(df.columns)}")
    return df


def load_literature(path):
    """Two columns (energy_meV, gamma_ps-1), comma/whitespace separated, optional header."""
    if path is None:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"Error: literature file '{path}' not found.")
    df = pd.read_csv(path, sep=r'[,\s]+', engine='python', comment='#', header=None)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()   # drops a text header line if present
    df = df.iloc[:, :2]
    df.columns = ['energy_meV', 'gamma_ps-1']
    df = df[(df['energy_meV'] > 0) & (df['gamma_ps-1'] > 0)].copy()
    df['tau_ps'] = 1.0 / df['gamma_ps-1']
    return df


# ---------------------------------------------------------------------------
# Reference (bare phonon) group velocities
# ---------------------------------------------------------------------------
THZ_TO_MEV = 4.135667696
VASP_TO_THZ = 15.633302     # sqrt(eV/(A^2 amu)) -> THz, same factor as SpinPhony.py
HBAR_MEV_PS = 0.6582119569
DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                           "Inputs", "CrI3", "band.h5")


def _reference_from_yaml(yaml_path):
    """Phonopy's own frequencies (THz -> meV) and group velocities (THz*A = A/ps)."""
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader
    with open(yaml_path) as f:
        cfg = yaml.load(f, Loader=Loader)
    ph = cfg['phonon']
    E = np.array([[b['frequency'] for b in p['band']] for p in ph], float) * THZ_TO_MEV
    v = np.array([[b['group_velocity'] for b in p['band']] for p in ph], float)
    q = np.array([p['q-position'] for p in ph], float)
    return q, E, v


def _reference_from_h5(h5_path):
    """
    Group velocities straight from the h5 if it carries them; otherwise
    Hellmann-Feynman on the grid:  dw_n/df_i = <e_n| dD/df_i |e_n> / (2 w_n),
    with dD/df from a 4th-order central difference of D(q) on the periodic mesh
    (error ~ h^4, h = 1/N), then f -> Cartesian via q_cart = f @ (2 pi B).
    Degenerate bands (Gamma, high-symmetry lines) are basis-ambiguous in this
    scheme - the yaml velocities from phonopy do not have that problem.
    """
    import h5py
    with h5py.File(h5_path, 'r') as h:
        q = np.array(h['q_positions'], float)
        if 'group_velocities' in h and 'frequencies' in h:
            return q, np.array(h['frequencies'], float) * THZ_TO_MEV, np.array(h['group_velocities'], float), 'h5'
        mesh = np.array(h['mesh'], int)
        B = np.array(h['reciprocal_lattice'], float)
        D = np.array(h['dynamical_matrices'])

    D = D * (VASP_TO_THZ * THZ_TO_MEV) ** 2                          # meV^2
    grid = np.rint(q * mesh).astype(int) % mesh
    gmap = np.full(tuple(mesh), -1, int)
    gmap[grid[:, 0], grid[:, 1], grid[:, 2]] = np.arange(len(q))
    if (gmap < 0).any():
        raise SystemExit("Reference h5 is not a complete q mesh; cannot differentiate on the grid.")

    def shifted(axis, s):
        g = grid.copy()
        g[:, axis] = (g[:, axis] + s) % mesh[axis]
        return gmap[g[:, 0], g[:, 1], g[:, 2]]

    ev, U = np.linalg.eigh(D)
    E = np.sqrt(np.clip(ev, 0.0, None))
    grad_f = np.zeros((len(q), D.shape[1], 3))
    for ax in range(3):
        dD = (8.0 * (D[shifted(ax, 1)] - D[shifted(ax, -1)])
              - (D[shifted(ax, 2)] - D[shifted(ax, -2)])) / 12.0 * mesh[ax]
        diag = np.einsum('qin,qij,qjn->qn', U.conj(), dD, U).real
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_f[:, :, ax] = np.where(E > 0.05, diag / (2.0 * E), 0.0)
    M_inv = np.linalg.inv(2.0 * np.pi * B)
    v = np.einsum('ai,qni->qna', M_inv, grad_f) / HBAR_MEV_PS
    return q, E, v, 'h5-FD'


def load_reference(path):
    """Priority: h5 velocity dataset > sibling phonopy yaml (same name without .h5) > grid finite difference."""
    if path is None or not os.path.exists(path):
        print(f"No reference file at {path}: reference velocities skipped.")
        return None
    yaml_path = path[:-3] if path.endswith('.h5') else path
    if path.endswith('.h5') and os.path.exists(yaml_path):
        try:
            q, E, v = _reference_from_yaml(yaml_path)
            return dict(q=q, E=E, v=v, source='phonopy yaml')
        except Exception as e:
            print(f" -> could not read {yaml_path} ({e}); falling back to {path}")
    if path.endswith('.h5'):
        q, E, v, src = _reference_from_h5(path)
        return dict(q=q, E=E, v=v, source='h5 (grid FD)' if src == 'h5-FD' else 'h5')
    q, E, v = _reference_from_yaml(path)
    return dict(q=q, E=E, v=v, source='phonopy yaml')


# ---------------------------------------------------------------------------
# G_mp = (1/N) sum_{q,lam} w * E * gamma * dn_BE/dT      [see calc_Gmp.py]
# ---------------------------------------------------------------------------
KB_MEV = 0.08617333262            # meV/K
MEV_PER_PS_TO_W = 1.602176634e-10 # 1 meV/ps in W
ANG3_TO_M3 = 1e-30


def dn_dT(E, T):
    """dn_BE/dT = E/(kB T^2) n(n+1), overflow-safe via n(n+1) = 1/(4 sinh^2(x/2))."""
    x = np.asarray(E, float) / (KB_MEV * T)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out


def compute_Gmp(E, gamma, w, T, N, V_m3):
    """Returns G_mp per cell [meV/(ps K)] and volumetric [W/(m^3 K)]."""
    g = np.where(np.isfinite(gamma), gamma, 0.0)
    G_cell = np.sum(w * E * g * dn_dT(E, T)) / N
    return G_cell, G_cell * MEV_PER_PS_TO_W / V_m3


def infer_temperature(df):
    if 'n_occ' not in df.columns:
        return None
    E, n = df['energy_meV'].to_numpy(float), df['n_occ'].to_numpy(float)
    ok = (E > 0.5) & (n > 1e-8) & np.isfinite(n)
    if not ok.any():
        return None
    return float(np.median(E[ok] / (KB_MEV * np.log1p(1.0 / n[ok]))))


def literature_gamma_of_E(E_lit, g_lit, E_query, n_bins=25, mean="arith"):
    """
    The literature points are a scatter over several branches, not a BZ sample.
    Bin them in energy, average gamma per bin, and interpolate gamma(E) onto
    the SpinPhony phonon energies, which then supply the BZ weighting.
    Arithmetic mean = right estimator for a SUM over branches; geometric mean
    is reported as a lower bound since the data span several decades.
    Energies outside the literature range get gamma = 0.
    """
    order = np.argsort(E_lit)
    E_lit, g_lit = E_lit[order], g_lit[order]
    n_bins = max(1, min(n_bins, len(E_lit)))
    edges = np.linspace(E_lit.min(), E_lit.max(), n_bins + 1)
    idx = np.clip(np.digitize(E_lit, edges) - 1, 0, n_bins - 1)
    centers, vals = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            centers.append(E_lit[sel].mean())
            vals.append(g_lit[sel].mean() if mean == "arith" else np.exp(np.log(g_lit[sel]).mean()))
    centers, vals = np.array(centers), np.array(vals)
    g = np.exp(np.interp(E_query, centers, np.log(vals)))     # log-interp: positive, smooth
    inside = (E_query >= E_lit.min()) & (E_query <= E_lit.max())
    return np.where(inside, g, 0.0), inside


def save_and_maybe_show(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f" -> Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot BZ lifetimes (and group velocities, if present) "
                                                 "for hybrid magnon-polarons in CrI3")
    parser.add_argument("filepath", type=str,
                        help="hybrid_path_lifetimes.csv (new) or hybrid_equilibrium_lifetimes.csv (old)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to save plots (default: same folder as the CSV)")
    parser.add_argument("--show", action="store_true", help="Also open interactive windows")
    parser.add_argument("--linear_x", action="store_true", help="Linear instead of log energy axis")
    parser.add_argument("--T", type=float, default=None,
                        help="Temperature for G_mp (K). Default: inferred from n_occ")
    parser.add_argument("--cell_volume", type=float, default=V_CRI3,
                        help="Primitive-cell volume (A^3), default: bulk CrI3 rhombohedral cell")
    parser.add_argument("--lit_mag", type=str, default=None,
                        help="Optional literature magnon rates: energy_meV, gamma_ps-1")
    parser.add_argument("--lit_phon", type=str, default=None,
                        help="Optional literature phonon rates: energy_meV, gamma_ps-1")
    parser.add_argument("--ref", type=str, default=DEFAULT_REF,
                        help="Reference phonon file: band.h5 (with group_velocities) or combined_*.h5 (a sibling phonopy yaml of the same name "
                             "without .h5 is preferred if present) or a yaml. Overlaid on the group velocity plot.")
    parser.add_argument("--no_ref", action="store_true", help="Do not overlay reference group velocities")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' not found.")
        return

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.filepath))
    os.makedirs(out_dir, exist_ok=True)
    xscale = 'linear' if args.linear_x else 'log'

    print(f"Loading data from {args.filepath}...")
    df = load_spinphony_csv(args.filepath)

    df_lit_mag = load_literature(args.lit_mag)
    df_lit_phon = load_literature(args.lit_phon)
    if df_lit_mag is None and df_lit_phon is None:
        print("No literature files given (--lit_mag / --lit_phon): literature overlay and "
              "literature G_mp are skipped.")

    tau = pd.to_numeric(df['tau_ps'], errors='coerce')
    valid = np.isfinite(tau) & (tau > 0) & (df['energy_meV'] > 0)
    df_valid = df[valid].copy()
    print(f" -> {len(df_valid):,} of {len(df):,} modes have finite lifetimes "
          f"({len(df) - len(df_valid):,} dropped: tau = inf or E <= 0)")
    if df_valid.empty:
        raise SystemExit("Nothing to plot.")

    # Net character: -1 pure phonon, +1 pure magnon
    character = df_valid['mag_character'] - df_valid['phon_character']

    dot_size = 5
    alpha_val = 0.7
    cmap = 'coolwarm'

    def style(ax, ylabel=r'Lifetime $\tau$ (ps)'):
        ax.set_yscale('log')
        ax.set_xscale(xscale)
        ax.set_xlabel('Energy (meV)', fontsize=12, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="--", alpha=0.4)

    # PLOT 1: Lifetime vs Energy (character)
    print("Generating Character Plot...")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sc1 = ax1.scatter(df_valid['energy_meV'], df_valid['tau_ps'], c=character,
                      cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val, edgecolors='none')
    style(ax1)
    ax1.set_title('CrI3: BZ Lifetimes vs Energy (Subsystem Character)', fontsize=14)
    fig1.colorbar(sc1, ax=ax1).set_label('Character (Blue=Phonon, Red=Magnon)', fontsize=12)
    save_and_maybe_show(fig1, os.path.join(out_dir, "lifetimes_vs_energy_character.png"), args.show)

    # PLOT 2: Lifetime vs Energy (PAM and SAM)
    if {'phon_AM_z_hbar', 'spin_AM_z_hbar'} <= set(df_valid.columns):
        print("Generating Angular Momentum Plot...")
        fig2, (ax_pam, ax_sam) = plt.subplots(1, 2, figsize=(14, 6))
        sc_pam = ax_pam.scatter(df_valid['energy_meV'], df_valid['tau_ps'], c=df_valid['phon_AM_z_hbar'],
                                cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val, edgecolors='none')
        style(ax_pam)
        ax_pam.set_title('Phonon Angular Momentum (PAM)', fontsize=14)
        fig2.colorbar(sc_pam, ax=ax_pam).set_label(r'Phonon AM $L_z$ ($\hbar$)', fontsize=12)

        sc_sam = ax_sam.scatter(df_valid['energy_meV'], df_valid['tau_ps'], c=df_valid['spin_AM_z_hbar'],
                                cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val, edgecolors='none')
        style(ax_sam, ylabel=None)
        ax_sam.set_title('Spin Angular Momentum (SAM)', fontsize=14)
        fig2.colorbar(sc_sam, ax=ax_sam).set_label(r'Spin AM $S_z$ ($\hbar$)', fontsize=12)
        save_and_maybe_show(fig2, os.path.join(out_dir, "lifetimes_vs_energy_AM.png"), args.show)
    else:
        print("Skipping Angular Momentum Plot (no AM columns in file).")

    # PLOT 3: Hybrid vs uncoupled literature (only if literature data was supplied)
    if df_lit_mag is not None or df_lit_phon is not None:
        print("Generating Literature Comparison Plot...")
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        sc3 = ax3.scatter(df_valid['energy_meV'], df_valid['tau_ps'], c=character,
                          cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val, edgecolors='none',
                          label='Hybrid Polarons (SpinPhony)')
        if df_lit_mag is not None:
            ax3.scatter(df_lit_mag['energy_meV'], df_lit_mag['tau_ps'], marker='^', s=40,
                        facecolors='none', edgecolors='black', linewidths=1.2, label='Uncoupled Magnons (Lit)')
        if df_lit_phon is not None:
            ax3.scatter(df_lit_phon['energy_meV'], df_lit_phon['tau_ps'], marker='s', s=40,
                        facecolors='none', edgecolors='black', linewidths=1.2, label='Uncoupled Phonons (Lit)')
        style(ax3)
        ax3.set_title('CrI3: Hybrid vs Uncoupled Literature Lifetimes', fontsize=14)
        fig3.colorbar(sc3, ax=ax3).set_label('Character (Blue=Phonon, Red=Magnon)', fontsize=12)
        ax3.legend(loc='lower left', fontsize=10, framealpha=0.9)
        save_and_maybe_show(fig3, os.path.join(out_dir, "lifetimes_comparison_literature.png"), args.show)

    # =========================================================================
    # G_mp (needs a full-BZ run; on a path N and the weights are not a BZ sample)
    # =========================================================================
    if 'q_idx' in df.columns and 'gamma_ps-1' in df.columns:
        print("\nComputing G_mp ...")
        T = args.T if args.T is not None else infer_temperature(df)
        if T is None:
            T = 300.0
            print(" -> no n_occ column; using T = 300 K (override with --T)")
        V_A3 = args.cell_volume
        V_m3 = V_A3 * ANG3_TO_M3
        N = df['q_idx'].nunique()

        E_all = df['energy_meV'].to_numpy(float)
        pc = np.clip(df['phon_character'].to_numpy(float), 0.0, 1.0)
        g_sp = pd.to_numeric(df['gamma_ps-1'], errors='coerce').to_numpy(float)
        keep = E_all > 0

        G_sp_cell, G_sp = compute_Gmp(E_all[keep], g_sp[keep], pc[keep], T, N, V_m3)

        print("=" * 66)
        print(f" T = {T:.1f} K   N = {N:,} q-points   V_cell = {V_A3:.4f} A^3")
        print(f" Phonon weight = phonon character; BZ weighting from the SpinPhony grid")
        print("-" * 66)
        print(f" G_mp SpinPhony           : {G_sp:.4e} W m^-3 K^-1  ({G_sp_cell:.4e} meV/(ps K)/cell)")

        if df_lit_phon is not None:
            E_lit = df_lit_phon['energy_meV'].to_numpy(float)
            g_lit = df_lit_phon['gamma_ps-1'].to_numpy(float)
            g_ar, inside = literature_gamma_of_E(E_lit, g_lit, E_all[keep], mean="arith")
            g_ge, _ = literature_gamma_of_E(E_lit, g_lit, E_all[keep], mean="geo")
            G_lit_cell, G_lit = compute_Gmp(E_all[keep], g_ar, pc[keep], T, N, V_m3)
            G_litg_cell, G_litg = compute_Gmp(E_all[keep], g_ge, pc[keep], T, N, V_m3)
            w_out = pc[keep][~inside].sum() / max(pc[keep].sum(), 1e-30)
            print(f" G_mp literature (arith.) : {G_lit:.4e} W m^-3 K^-1  ({G_lit_cell:.4e} meV/(ps K)/cell)")
            print(f" G_mp literature (geom.)  : {G_litg:.4e} W m^-3 K^-1  (lower bound)")
            if G_lit > 0:
                print(f" Ratio SpinPhony / lit.   : {G_sp / G_lit:.3g}")
            print(f" Phonon weight outside literature energy range "
                  f"[{E_lit.min():.1f}, {E_lit.max():.1f}] meV: {w_out:.1%} (gamma set to 0 there)")
        print("=" * 66)
    else:
        print("Skipping G_mp (file has no q_idx / gamma_ps-1 columns).")

    # PLOT 4: Group velocity magnitude: SpinPhony hybrid modes + reference bare phonons
    has_v = {'vx', 'vy', 'vz'} <= set(df_valid.columns)
    ref = None if args.no_ref else load_reference(args.ref)
    if has_v or ref is not None:
        print("Generating Group Velocity Magnitude Plot...")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        plotted = False
        if ref is not None:
            E_ref = ref['E'].reshape(-1)
            v_ref = np.linalg.norm(ref['v'], axis=-1).reshape(-1)
            ok = (E_ref > 0.05) & (v_ref > 0)
            ax4.scatter(E_ref[ok], v_ref[ok], marker='s', s=10, facecolors='none', edgecolors='0.35',
                        linewidths=0.6, zorder=1, label=f"Reference phonons ({ref['source']})", alpha=0.7)
            plotted = plotted or ok.any()
            print(f" -> reference: {ok.sum():,} phonon modes from {ref['source']}")
        if has_v:
            v = np.sqrt(df_valid['vx']**2 + df_valid['vy']**2 + df_valid['vz']**2)
            pos = v > 0
            if pos.any():
                sc4 = ax4.scatter(df_valid['energy_meV'][pos], v[pos], c=character[pos],
                                  cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val,
                                  edgecolors='none', zorder=2, label='Hybrid Polarons (SpinPhony)')
                fig4.colorbar(sc4, ax=ax4).set_label('Character (Blue=Phonon, Red=Magnon)', fontsize=12, alpha=0.9)
                plotted = True
            else:
                print(" -> all SpinPhony velocities are zero; only the reference is drawn.")
        if plotted:
            style(ax4, ylabel=r'Group Velocity Magnitude |v| ($\AA$/ps)')
            ax4.set_title('CrI3: Group Velocity Magnitude vs Energy', fontsize=14)
            ax4.legend(loc='best', fontsize=10, framealpha=0.9)
            save_and_maybe_show(fig4, os.path.join(out_dir, "group_velocities_vs_energy.png"), args.show)
        else:
            plt.close(fig4)
            print("Skipping Group Velocity Plot (nothing to draw).")
    else:
        print("Skipping Group Velocity Plot (no vx/vy/vz columns and no reference).")


if __name__ == "__main__":
    main()
