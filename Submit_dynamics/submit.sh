#!/bin/sh
#SBATCH -A naiss2026-1-49-gpu
#SBATCH -J dyn_bccFe
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH -o Outputs/dynamics/logs/dyn_%A_%a.out
#SBATCH -e Outputs/dynamics/logs/dyn_%A_%a.err

# Bare magnon-phonon dynamics of bcc Fe, T_mag 600 K -> T_phon 300 K, 250 ps, with and without three-phonon scattering, at
#   N = 20^3 with broadening 0.2 meV   and   N = 40^3 with broadening 0.1 meV.
# Submit from the repository root (paths are relative):
#     mkdir -p Outputs/dynamics/logs && sbatch Submit_dynamics/submit.sh dyn_20261111
# Array tasks: 0 N20 off | 1 N20 on | 2 N40 off | 3 N40 on.
# Outputs per run (Outputs/dynamics/<tag>/N<N>_s<s>_<off|on>/): observables_dynamics.txt, spectral_dynamics.csv, flux_dynamics.csv,
# final_populations.npz, occupation_snapshots_bare.npz (mode-resolved occupations at 60 log-spaced times, float32; for python/plot_mode_temperatures.py),
# w_mag_grid.csv, w_phon_grid.csv, config.toml, params.txt, status.txt, run.log.
# The grid channel buffer of the dynamics scan is not chunked: an overflow aborts the scan with the fraction to use, the run is then repeated with
# that value x 1.2 (up to 4 tries, memory budget 75 GB).

TAG=${1:-dyn_default}
module purge
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load CUDA/12.9.1
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
source venv/bin/activate
export PYTHONUNBUFFERED=1

# ---- settings ------------------------------------------------------------------------------------------
TMAG=600
TPHON=300
DT=1e-3             # ps
STEPS=250000        # 250 ps
OBS_EVERY=500       # observables / spectra every 0.5 ps
SNAPSHOTS=60        # full occupation snapshots, log-spaced in time (plus t = 0 and the final state)
FC3=Inputs/bccFe/PH3/fc3.hdf5
# ---------------------------------------------------------------------------------------------------------

IDX=${SLURM_ARRAY_TASK_ID:-0}
case $IDX in
    0|1) N=20; SMEAR=0.2; MESH=Inputs/bccFe/combined_band_20x20x20.h5; FRAC=0.03 ;;
    2|3) N=40; SMEAR=0.1; MESH=Inputs/bccFe/grid_40x40x40.h5;          FRAC=0.012 ;;
esac
if [ $((IDX % 2)) -eq 1 ]; then TAGP=on;  FLAG="--phph --phph_fc3 $FC3"; else TAGP=off; FLAG=""; fi

RD=Outputs/dynamics/$TAG/N${N}_s${SMEAR}_${TAGP}
mkdir -p $RD
cp -n python/SpinPhony.py python/phph.py python/phph_gpu.py Outputs/dynamics/$TAG/ 2>/dev/null
write_config() {
    sed -e "s|^anticipated_fraction = [0-9.eE+-]*|anticipated_fraction = $1|" \
        -e "/^\[materials.bccFe\]/,/^\[/ s|^mesh = .*|mesh = \"$MESH\"|" config.toml > $RD/config.toml
}
cat > $RD/params.txt <<PARAMS
material=bccFe
N=$N
smearing=$SMEAR
phph=$TAGP
T_mag=$TMAG
T_phon=$TPHON
dt=$DT
steps=$STEPS
obs_every=$OBS_EVERY
snapshots=$SNAPSHOTS
git_commit=$(git rev-parse --short HEAD 2>/dev/null)
node=$(hostname)
PARAMS
echo "task $IDX: N=$N smearing=$SMEAR phph=$TAGP frac0=$FRAC"
nvidia-smi --query-gpu=name,memory.total --format=csv

TRY=0; STATUS=failed
while [ $TRY -lt 4 ]; do
    TRY=$((TRY + 1))
    write_config $FRAC
    grep -n "^anticipated_fraction" $RD/config.toml; grep -n "^mesh" $RD/config.toml | grep "$MESH"
    python python/SpinPhony.py --config $RD/config.toml --material bccFe --full_bz --dynamics_only $FLAG \
        --tmag $TMAG --tphon $TPHON --smearing $SMEAR --steps $STEPS --dt $DT --obs_every $OBS_EVERY --snapshots $SNAPSHOTS \
        --out_dir $RD > $RD/run.log 2>&1
    RC=$?
    if [ $RC -eq 0 ] && [ -f $RD/occupation_snapshots_bare.npz ]; then STATUS=ok; break; fi
    if grep -q "CUDA_ERROR_OUT_OF_MEMORY" $RD/run.log; then STATUS=oom; break; fi
    if grep -q "Channel buffer overflow" $RD/run.log; then
        NEED=$(grep -o "at least [0-9.]*" $RD/run.log | tail -1 | awk '{print $3}')
        FRAC=$(awk -v x="$NEED" 'BEGIN{printf "%.5f", x * 1.2}')
        if awk -v n=$N -v f=$FRAC 'BEGIN{exit !(36*9*n^6*f/1e9 > 75)}'; then STATUS="memory_limit(needs fraction $FRAC)"; break; fi
        echo "overflow: retry $TRY with anticipated_fraction $FRAC"
        continue
    fi
    STATUS="error(rc=$RC)"; break
done
echo "status=$STATUS tries=$TRY frac=$FRAC" | tee $RD/status.txt
