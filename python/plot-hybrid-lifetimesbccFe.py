import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import io

literature_scattering_rates = """3.5330366949927345, 0.09102981779915223
7.1097094323124335, 0.21363397911526588
10.706058931542348, 0.3442640548857818
14.470842848333064, 0.4563947703262179
17.656864338394996, 0.5710470594624236
21.17999854447029, 0.6094377562050983
24.976458841488746, 0.7145014915214672
27.510987074215667, 0.8685113737513525
31.712798486156796, 0.8811603465430217
38.91557544867763, 0.9963917920135462
44.60491499417326, 1.2199505557502925
52.00551275510881, 1.2557439628235745
60.29022624209932, 1.6173023172161325
57.937928431613116, 1.2557439628235751
81.95590434970995, 1.7638750529078007
81.95590434971, 1.4199606034664953
101.72024731297093, 1.828800808389492
130.63194185325634, 1.6173023172161325
157.5908014514459, 1.5045063827385736
187.96373839101864, 1.5154225728627575
221.65575534665635, 1.815627230483809
259.9051079393094, 2.606228915488146
294.5342488814524, 3.5054273703936762
330.00347911252817, 4.923882631706749
369.74408456724194, 5.9421053233155
438.50576628007633, 9.785451025760969
547.358170107669, 7.119234215334258
491.3127387829667, 3.7682364621484377
494.1140015119873, 4.7490758412131004
514.1751827683922, 5.731149775014003
391.3745601980384, 7.877516562652843
245.5406814401623, 2.1596327552241537
318.9361179185746, 4.050748891445512
112.68130079648533, 1.3894954943731381
154.92569632334198, 1.3019661288117441
226.7543125870802, 1.5045063827385743
256.9665200810953, 1.8824579107057196
289.5532174388399, 2.389659095901388
320.7545602050025, 2.968434514256669
359.38136638046257, 3.6608276068155456
451.1500269578206, 7.543120063354623
414.27044478476745, 4.680903310144551
446.0491499417328, 5.9852192324297695
491.3127387829667, 7.708505252846041"""

literature_phonon_scatt_rates_per_ps = """4.48783529925765, 0.02417798122601752
5.888869366178223, 0.03018961164785434
7.72715099685385, 0.03592133496112259
9.552576079607517, 0.0448030835375812
11.498811546775325, 0.03797595526262054
2.819945410807872, 0.0018608683755954732
2.205492307903121, 0.0003036605033083713
4.450439241168672, 0.00018995045631324254
6.710237827390981, 0.00014331969094337878
8.861676208828554, 0.00009795239460169727
10.88051628597797, 0.00006848671723329059
12.923347784635489, 0.00004140970996091235
14.461402786530698, 0.00002562047919458797
16.50676813613766, 0.000014399516021145573
17.98638352798355, 0.000006667506035524671
19.34066700750141, 0.00000308653933925039
21.76989381385246, 2.3442804949386288e-7
22.21150529579873, 4.1829323583566974e-7
20.79061045149376, 6.144486797269541e-7
30.53377425516116, 8.068191541300666e-7
23.27042469457957, 7.289455365444654e-7
32.8536022169316, 0.000002171138969964886
32.213529222940345, 0.000003778907373660745
37.274005751938304, 0.000005990992992441711
36.0655132196093, 0.000009466407938056631
37.78360310758767, 0.000014980087765876983
37.29081640863232, 0.000020989644109504375
36.80764831549063, 0.00003742902679541909
36.570319566573744, 0.0000577614265829801
37.307958148953894, 0.00007533254030118384
37.31410521967911, 0.00011910687334609841
38.06740644022548, 0.0001630136886027892
38.58316675921735, 0.00029083172595973267
38.33638427781492, 0.0005186791640562766
38.092171143636584, 0.0009944195336829467
36.37329443616216, 0.0017299518875646077
35.428852387763236, 0.0029389203320260254
33.16789309208566, 0.006825693748558308
32.308921292763465, 0.014062767011255488
29.45071032555504, 0.01967518210022706
26.318682767796464, 0.03257654148508324
23.67299124493301, 0.03757297961500052
20.599497361035734, 0.04330903340752583
17.806219042213936, 0.046431633292903694
16.119485867090166, 0.03076088847880551
14.21178468983274, 0.024113955973496384
16.33122538157716, 0.017671226335935375
19.91918347482425, 0.012059606741113765
21.708065122359866, 0.008618498399364556
24.13122704692414, 0.00533166806358697
26.647290954291396, 0.0031426679144979028
28.09325635747641, 0.0020381870894790404
29.42084696312597, 0.0011716008229174818
31.84940281004941, 0.0007071729593616644
32.91776254713872, 0.000492860869656832
33.3539175298224, 0.00038736352044241063
34.24215638404002, 0.00021210509834114111
36.57602794170704, 0.00008914989749269242
19.404332167011614, 0.02871341014344943
20.72904644147398, 0.016117950083224756
23.662524668714553, 0.010986026640499313
23.822075919770004, 0.014324458929957002
21.716537312924185, 0.02550579992248868
24.791265392849553, 0.02058110602859472
26.307502677228335, 0.00999568438416322
26.47799969148959, 0.0063228355236533
27.737675675906697, 0.008451661734235422
27.01839682163299, 0.015811788347656845
32.73104854405593, 0.006661444307982777
29.631042008063712, 0.00463122482335295
28.861414034076653, 0.007680303434886697
29.25092410564113, 0.011856825789249426
30.628175632789933, 0.004310720343112179
34.27453588945389, 0.0029371086065807963
31.651453304831524, 0.002092567267547945
33.59104698984424, 0.0013904322047930025
34.033761901261435, 0.000901102104394745
36.60045716226939, 0.0005707014406665525
36.354060198752066, 0.00039745346996477237
3.84979904629312, 0.0015810082131400367
4.543593440471047, 0.001970219587118185
5.617144983630647, 0.0027061705676656076
6.8975435207454, 0.0025269796825938503
8.47024281279611, 0.0027269424944705825
9.669834466532574, 0.002423208804549792
10.821709075404096, 0.001954609094920877
12.436657263396148, 0.0018229337431930857
13.554143491507192, 0.0014696899177669414
15.575238314840613, 0.0010263166913905854
17.430567028489264, 0.0008278477428336345
19.637071795532684, 0.0007179379825630434
21.97548553774678, 0.0005258609359733531
23.63340681035368, 0.0003580302303372937
25.08187396214061, 0.0002437032734623337
26.09530218251285, 0.00016582207412149487
27.51179206175554, 0.00011285750539076553
28.06063275278628, 0.00008055514630399777
29.001398630201955, 0.00005349946453265148
29.777341409421236, 0.00004105628637951391
30.369534768897857, 0.000024753897503073283
34.66578706360583, 0.000014956089598947628
4.279903816377769, 0.0012447392010800492
5.881520833116225, 0.0009375491143307623
8.135508671740649, 0.0005549457211472995
10.188717308882618, 0.0003200606549059669
11.631182307179714, 0.0002521104088510457
12.426308903996297, 0.00018010615369414866
13.452510242901413, 0.00011971804789301572
15.152417560616993, 0.000059628897447650664
17.182062208380483, 0.00003780247511765629
7.769882872274368, 0.0016409030411369252
6.2848444959483585, 0.0011661948295607954
6.040716046811293, 0.0016730882800491326
11.557088530759623, 0.00048334988675016944
11.714585154205448, 0.001071323905660601
10.05801269678106, 0.000819416800514051
10.39864529263987, 0.0013604087975356619
8.084173961065334, 0.0012595640935558253
11.406653744902375, 0.0007280583065504387
14.288270816548787, 0.0007311176230224562
13.106621362405011, 0.0004192293155021089
15.779965627873809, 0.0006039771112262869
14.571289919792358, 0.00035482132066059585
13.908706581863724, 0.00022422321010324018
15.462203911393695, 0.00016421430406608902
16.091834222669647, 0.0002598283675638897
16.8585636365404, 0.00042119866034329455
21.539313696074792, 0.00032455233622302634
20.97276422295791, 0.00021017814578466423
18.008856207715798, 0.00021469746469324796
20.287861893361214, 0.00016907456554769148
19.75268784044839, 0.00008813321807783798
20.96112852338841, 0.00004491857822834452
19.22179479862899, 0.000011076327613528422
23.13682665588415, 0.000008123981935882013
23.912308766112634, 0.000005266884799028793
24.876755190031894, 0.000002955048445872974
24.224311255884512, 0.000002377436305876912
23.432411206801298, 0.0000017366588633680271
22.517321000874094, 0.0000013635793171180275
20.390505744080144, 0.0000021006800931885603
20.393511725426162, 0.000003164984466999983
20.798003195230883, 0.000001651231106752081
25.37126872064914, 0.0000017392309502530048
27.469599464048382, 0.0000015816664265930465
24.71378288087261, 0.0000034145909856985495
29.36248405978752, 0.00000468659963601712
27.849983773423585, 0.000006405542665577084
26.588896091042262, 0.000007048003237506703
29.370123259951402, 0.000009660421666986051
26.417226853058562, 0.000010617547104639775
24.55996951387645, 0.000009863276249549688
27.677611325870274, 0.00002037621586662683
26.95544869624374, 0.000024110610486692646
24.084819160301755, 0.000025248769055436327
25.066654350326466, 0.00004506840227041166
22.85230130798525, 0.00009273859881318703
24.257312027490137, 0.000104736716560433
22.698851586050946, 0.0000677766973457672
21.09789077146501, 0.00003205396237418737
23.607804547493878, 0.000017579657558550953
26.067937069913906, 0.000008966400909731527
29.56354493956374, 0.000008161138059648034
28.61074534013592, 0.000022453106567528304
24.739511933481317, 0.0000616441125776425
16.967528504147154, 0.0002538889400689191
18.866598987642874, 0.0003316536519671069
18.493575857403105, 0.00025429632390578267
27.293663999901057, 0.000002753598671205664
28.395966708527638, 0.0000017854145724009856
29.354592307028277, 0.000002219465258234286
31.988343013691466, 0.000001276748458247854
34.18134199076503, 0.000001513351728790208
30.341370543452946, 0.0000018759326350961785
30.75805420076251, 0.000005553051549190054
22.43945678612649, 0.000010404311080699466
33.09578293667771, 0.00001606412040410882
33.76860889233399, 0.000032335872941472364
33.32992113371684, 0.00005236039963279138
33.55523945480777, 0.00007164453691910147"""

# Old (hybrid_equilibrium_lifetimes.csv) -> new (hybrid_path_lifetimes.csv) column names.
# Columns are normalised to the NEW names so both file formats work.
COLUMN_ALIASES = {
    'mag_char': 'mag_character',
    'phon_char': 'phon_character',
    'phon_AM': 'phon_AM_z_hbar',
    'spin_AM': 'spin_AM_z_hbar',
}


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
    is reported as a lower bound since the data span ~5 decades.
    Energies outside the literature range get gamma = 0.
    """
    order = np.argsort(E_lit)
    E_lit, g_lit = E_lit[order], g_lit[order]
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
                                                 "for hybrid magnon-polarons")
    parser.add_argument("filepath", type=str,
                        help="hybrid_path_lifetimes.csv (new) or hybrid_equilibrium_lifetimes.csv (old)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to save plots (default: same folder as the CSV)")
    parser.add_argument("--show", action="store_true", help="Also open interactive windows")
    parser.add_argument("--linear_x", action="store_true", help="Linear instead of log energy axis")
    parser.add_argument("--T", type=float, default=None,
                        help="Temperature for G_mp (K). Default: inferred from n_occ")
    parser.add_argument("--a", type=float, default=2.8665, help="Lattice constant (A), bcc primitive cell")
    parser.add_argument("--cell_volume", type=float, default=None,
                        help="Primitive-cell volume (A^3); overrides --a")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' not found.")
        return

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.filepath))
    os.makedirs(out_dir, exist_ok=True)
    xscale = 'linear' if args.linear_x else 'log'

    print(f"Loading data from {args.filepath}...")
    df = load_spinphony_csv(args.filepath)

    print("Parsing literature rates...")
    df_lit_mag = pd.read_csv(io.StringIO(literature_scattering_rates), names=['energy_meV', 'gamma_ps-1'])
    df_lit_phon = pd.read_csv(io.StringIO(literature_phonon_scatt_rates_per_ps), names=['energy_meV', 'gamma_ps-1'])
    df_lit_mag['tau_ps'] = 1.0 / df_lit_mag['gamma_ps-1']
    df_lit_phon['tau_ps'] = 1.0 / df_lit_phon['gamma_ps-1']

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
    ax1.set_title('BZ Lifetimes vs Energy (Subsystem Character)', fontsize=14)
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

    # PLOT 3: Hybrid vs uncoupled literature
    print("Generating Literature Comparison Plot...")
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    sc3 = ax3.scatter(df_valid['energy_meV'], df_valid['tau_ps'], c=character,
                      cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val, edgecolors='none',
                      label='Hybrid Polarons (SpinPhony)')
    ax3.scatter(df_lit_mag['energy_meV'], df_lit_mag['tau_ps'], marker='^', s=40,
                facecolors='none', edgecolors='black', linewidths=1.2, label='Uncoupled Magnons (Lit)')
    ax3.scatter(df_lit_phon['energy_meV'], df_lit_phon['tau_ps'], marker='s', s=40,
                facecolors='none', edgecolors='black', linewidths=1.2, label='Uncoupled Phonons (Lit)')
    style(ax3)
    ax3.set_title('Comparison: Hybrid vs Uncoupled Literature Lifetimes', fontsize=14)
    fig3.colorbar(sc3, ax=ax3).set_label('Character (Blue=Phonon, Red=Magnon)', fontsize=12)
    ax3.legend(loc='lower left', fontsize=10, framealpha=0.9)
    save_and_maybe_show(fig3, os.path.join(out_dir, "lifetimes_comparison_literature.png"), args.show)


    # =========================================================================
    # G_mp: SpinPhony vs literature phonon rates
    # =========================================================================
    print("\nComputing G_mp ...")
    T = args.T if args.T is not None else infer_temperature(df)
    if T is None:
        T = 300.0
        print(" -> no n_occ column; using T = 300 K (override with --T)")
    V_A3 = args.cell_volume if args.cell_volume is not None else 0.5 * args.a ** 3
    V_m3 = V_A3 * ANG3_TO_M3
    N = df['q_idx'].nunique()

    E_all = df['energy_meV'].to_numpy(float)
    pc = np.clip(df['phon_character'].to_numpy(float), 0.0, 1.0)
    g_sp = pd.to_numeric(df['gamma_ps-1'], errors='coerce').to_numpy(float)
    keep = E_all > 0

    G_sp_cell, G_sp = compute_Gmp(E_all[keep], g_sp[keep], pc[keep], T, N, V_m3)

    E_lit = df_lit_phon['energy_meV'].to_numpy(float)
    g_lit = df_lit_phon['gamma_ps-1'].to_numpy(float)
    g_ar, inside = literature_gamma_of_E(E_lit, g_lit, E_all[keep], mean="arith")
    g_ge, _ = literature_gamma_of_E(E_lit, g_lit, E_all[keep], mean="geo")
    G_lit_cell, G_lit = compute_Gmp(E_all[keep], g_ar, pc[keep], T, N, V_m3)
    G_litg_cell, G_litg = compute_Gmp(E_all[keep], g_ge, pc[keep], T, N, V_m3)

    w_out = pc[keep][~inside].sum() / max(pc[keep].sum(), 1e-30)
    print("=" * 66)
    print(f" T = {T:.1f} K   N = {N:,} q-points   V_cell = {V_A3:.4f} A^3")
    print(f" Phonon weight = phonon character; BZ weighting from the SpinPhony grid")
    print("-" * 66)
    print(f" G_mp SpinPhony           : {G_sp:.4e} W m^-3 K^-1  ({G_sp_cell:.4e} meV/(ps K)/cell)")
    print(f" G_mp literature (arith.) : {G_lit:.4e} W m^-3 K^-1  ({G_lit_cell:.4e} meV/(ps K)/cell)")
    print(f" G_mp literature (geom.)  : {G_litg:.4e} W m^-3 K^-1  (lower bound)")
    if G_lit > 0:
        print(f" Ratio SpinPhony / lit.   : {G_sp / G_lit:.3g}")
    print(f" Phonon weight outside literature energy range "
          f"[{E_lit.min():.1f}, {E_lit.max():.1f}] meV: {w_out:.1%} (gamma set to 0 there)")
    print("=" * 66)

    # PLOT 4: Group velocity magnitude (only if the file has velocities)
    if {'vx', 'vy', 'vz'} <= set(df_valid.columns):
        print("Generating Group Velocity Magnitude Plot...")
        v = np.sqrt(df_valid['vx']**2 + df_valid['vy']**2 + df_valid['vz']**2)
        pos = v > 0
        if pos.any():
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sc4 = ax4.scatter(df_valid['energy_meV'][pos], v[pos], c=character[pos],
                              cmap=cmap, vmin=-1.0, vmax=1.0, s=dot_size, alpha=alpha_val, edgecolors='none')
            style(ax4, ylabel='Group Velocity Magnitude |v| (arb. units)')
            ax4.set_title('Hybrid Polaron Group Velocity Magnitude vs Energy', fontsize=14)
            fig4.colorbar(sc4, ax=ax4).set_label('Character (Blue=Phonon, Red=Magnon)', fontsize=12)
            save_and_maybe_show(fig4, os.path.join(out_dir, "group_velocities_vs_energy.png"), args.show)
        else:
            print("Skipping Group Velocity Plot (all velocities are zero - velocity calc disabled in SpinPhony).")
    else:
        print("Skipping Group Velocity Plot (no vx/vy/vz columns in this file).")


if __name__ == "__main__":
    main()