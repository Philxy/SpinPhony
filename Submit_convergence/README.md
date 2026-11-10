# Convergence scan of the bare lifetimes (grid N x Gaussian broadening)

Cluster, repository root, after `git pull`:

    python3 Submit_convergence/make_runs.py --tag conv_20261110          # writes Outputs/convergence/conv_20261110/runs.csv, prints the sbatch line
    mkdir -p Outputs/convergence/logs
    sbatch --array=0-29%6 Submit_convergence/submit.sh conv_20261110     # one GPU task per run, 6 at a time

Defaults: bccFe N = 12, 20, 40 (with three-phonon scattering, --phph), fccNi N = 20, 32, 40 (no phph), broadening (`--smearing`) 0.05 0.1 0.2 0.3 0.5 meV,
T_mag = T_phon = 300 K, full BZ, `--no_slc --bare_only` (only `path_lifetimes.csv`, no hybrid scan, no dynamics). Other choices:
`make_runs.py --materials bccFe --grids bccFe=12,20 --smearings 0.1 0.2 --T 300`.

`submit.sh` writes `config.toml` per run (mesh of the material block and `anticipated_fraction` replaced), detects a channel-buffer overflow (the scan
only warns "lifetimes are too small"), repeats with the suggested fraction x 1.2 (up to 4 tries) and records `status.txt`
(`ok`, `oom`, `memory_limit(...)`, `error(...)`). Check: `cat Outputs/convergence/conv_20261110/*/status.txt`.

Copy back (local terminal; 2FA):

    rsync -av --exclude run.log --exclude code philipp@login.hpc.arrhenius.naiss.se:/nobackup/proj/disk/oppeneer/personal/philipp/SpinPhony/Outputs/convergence/conv_20261110 Outputs/convergence/

Post-processing (local):

    python python/convergence_lifetimes.py Outputs/convergence/conv_20261110     # per-run figures, N and broadening overviews, binned medians, with literature
    python python/convergence_Gmp.py       Outputs/convergence/conv_20261110     # G_mp vs N and vs broadening + convergence_Gmp.csv

Literature: bccFe from `python/plot-hybrid-lifetimesbccFe.py`; fccNi only if `Inputs/fccNi/lit_magnon_rates.csv` / `lit_phonon_rates.csv`
(two columns: energy meV, rate 1/ps) exist.

## Temperature scan (fixed grid and broadening)

Same scripts, one run per temperature (choose N and the broadening from the convergence scan):

    python3 Submit_convergence/make_runs.py --tag Tscan_20261110 --grids bccFe=20 fccNi=32 --smearings 0.2 \
        --temps 100 200 300 400 500 600 800 1000
    mkdir -p Outputs/convergence/logs
    sbatch --array=0-15%8 Submit_convergence/submit.sh Tscan_20261110            # use the array range printed by make_runs.py

Runs are named `<material>_N<N>_s<smearing>_T<T>K` and carry `params.txt` with `mode=fullbz`, `T_mag`, `scan`, so that
`python/plot_Gmp_vs_T.py` (all materials in one plot) reads them directly:

    python python/plot_Gmp_vs_T.py Outputs/convergence/Tscan_20261110 --source bare              # G_mp vs T/T_C, bccFe and fccNi in one figure
    python python/plot_Gmp_vs_T.py Outputs/convergence/Tscan_20261110 --source bare --absolute_T  # vs T in K

For bccFe runs with phph the bare file holds the channels separately; G_mp uses the magnon-phonon (`gamma_exch_ps-1`) channel only.
