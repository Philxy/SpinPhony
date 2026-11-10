"""Shared helpers of the convergence post-processing (python/convergence_lifetimes.py, python/convergence_Gmp.py).

Scan layout (written by Submit_convergence/make_runs.py + submit.sh):
    <scan>/runs.csv                       id,run,material,N,mesh,smearing,frac0,phph,fc3,T
    <scan>/<run>/{path_lifetimes.csv, params.txt, status.txt, config.toml, run.log}
"""
import io
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KB_MEV = 0.08617333262
MEV_PER_PS_TO_W = 1.602176634e-10
ANG3_TO_M3 = 1e-30
CELL_VOLUME = {"bccFe": 11.430011, "fccNi": 3.516822176 ** 3 / 4.0}      # A^3, as in python/plot_Gmp_vs_T.py
LIT_SCRIPT = {"bccFe": os.path.join(HERE, "plot-hybrid-lifetimesbccFe.py")}
LIT_FILES = {"fccNi": {"magnon": os.path.join(ROOT, "Inputs/fccNi/lit_magnon_rates.csv"), "phonon": os.path.join(ROOT, "Inputs/fccNi/lit_phonon_rates.csv")}}
COLORS = {"mp": "C0", "phph": "C2", "lit_mag": "C4", "lit_phon": "k"}


def literature(material):
    """{'magnon': DataFrame(E, g) | None, 'phonon': ...}: energy (meV), rate (1/ps). bccFe: the strings of plot-hybrid-lifetimesbccFe.py;
    fccNi: optional two-column files Inputs/fccNi/lit_{magnon,phonon}_rates.csv (none shipped)."""
    out = {"magnon": None, "phonon": None}
    if material in LIT_SCRIPT and os.path.exists(LIT_SCRIPT[material]):
        txt = open(LIT_SCRIPT[material]).read()
        for key, name in (("magnon", "literature_scattering_rates"), ("phonon", "literature_phonon_scatt_rates_per_ps")):
            out[key] = pd.read_csv(io.StringIO(re.search(name + r'\s*=\s*"""(.*?)"""', txt, re.S).group(1)), names=["E", "g"])
    for key, path in LIT_FILES.get(material, {}).items():
        if os.path.exists(path):
            out[key] = pd.read_csv(path, comment="#", header=None, names=["E", "g"], sep=r"[,\s]+", engine="python")
            out[key] = out[key][pd.to_numeric(out[key].E, errors="coerce").notna()].astype(float)
    return out


def load_scan(scan_dir):
    """runs.csv + status of every run; adds path, status, ok."""
    man = pd.read_csv(os.path.join(scan_dir, "runs.csv"))
    st, ok = [], []
    for run in man.run:
        f = os.path.join(scan_dir, run, "status.txt")
        s = open(f).read().strip() if os.path.exists(f) else "missing"
        has = os.path.exists(os.path.join(scan_dir, run, "path_lifetimes.csv"))
        st.append(s)
        ok.append(has and (s.startswith("status=ok") or s == "missing"))
    man["status"], man["ok"] = st, ok
    man["path"] = [os.path.join(scan_dir, r) for r in man.run]
    return man


def load_run(path):
    """path_lifetimes.csv with gamma_mp (magnon-phonon rate: gamma_exch when the file also carries phph, else the total) and gamma_phph (0 if absent)."""
    d = pd.read_csv(os.path.join(path, "path_lifetimes.csv"), comment="#")
    d["gamma_mp"] = d["gamma_exch_ps-1"] if "gamma_exch_ps-1" in d else d["gamma_ps-1"]
    d["gamma_phph"] = d["gamma_phph_ps-1"] if "gamma_phph_ps-1" in d else 0.0
    return d


def dn_dT(E, T):
    x = np.asarray(E, float) / (KB_MEV * T)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 700)
    out[ok] = (x[ok] / T) / (4.0 * np.sinh(0.5 * x[ok]) ** 2)
    return out


def compute_Gmp(E, gamma, w, T, N, V_A3):
    """G_mp = (1/N) sum w E gamma dn_BE/dT: per cell [meV/(ps K)] and volumetric [W/(m^3 K)]."""
    g = np.where(np.isfinite(gamma), gamma, 0.0)
    G_cell = np.sum(w * E * g * dn_dT(E, T)) / N
    return G_cell, G_cell * MEV_PER_PS_TO_W / (V_A3 * ANG3_TO_M3)


def literature_gamma_of_E(E_lit, g_lit, E_query, n_bins=25):
    """arithmetic bin average of the literature scatter, log-interpolated onto E_query (0 outside the literature range); as in plot-hybrid-lifetimesbccFe.py"""
    order = np.argsort(E_lit)
    E_lit, g_lit = E_lit[order], g_lit[order]
    edges = np.linspace(E_lit.min(), E_lit.max(), n_bins + 1)
    idx = np.clip(np.digitize(E_lit, edges) - 1, 0, n_bins - 1)
    c, v = [], []
    for b in range(n_bins):
        s = idx == b
        if s.any():
            c.append(E_lit[s].mean()); v.append(g_lit[s].mean())
    g = np.exp(np.interp(E_query, np.array(c), np.log(np.array(v))))
    return np.where((E_query >= E_lit.min()) & (E_query <= E_lit.max()), g, 0.0)


def gmp_of_run(d, material, T, lit_phonon=None):
    """(G_cell, G_Wm3K, lit_G_Wm3K or nan) of one run, phonon rows only (phonon character 1), magnon-phonon channel only."""
    ph = d[d.particle == "phonon"]
    E = ph.energy_meV.to_numpy(float)
    keep = E > 0
    N = d.q_idx.nunique()
    V = CELL_VOLUME[material]
    G_cell, G = compute_Gmp(E[keep], ph.gamma_mp.to_numpy(float)[keep], np.ones(keep.sum()), T, N, V)
    Gl = np.nan
    if lit_phonon is not None:
        g_lit = literature_gamma_of_E(lit_phonon.E.to_numpy(float), lit_phonon.g.to_numpy(float), E[keep])
        Gl = compute_Gmp(E[keep], g_lit, np.ones(keep.sum()), T, N, V)[1]
    return G_cell, G, Gl
