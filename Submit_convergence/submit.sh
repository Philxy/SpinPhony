#!/bin/sh
#SBATCH -A naiss2026-1-49-gpu
#SBATCH -J conv_bare
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -o Outputs/convergence/logs/conv_%A_%a.out
#SBATCH -e Outputs/convergence/logs/conv_%A_%a.err

# Convergence of the bare magnon-phonon (and, for bcc Fe, three-phonon) lifetimes with the grid size N and the Gaussian broadening.
# One array task per line of Outputs/convergence/<TAG>/runs.csv (written by Submit_convergence/make_runs.py, which also prints the sbatch line):
#     python3 Submit_convergence/make_runs.py --tag conv_20261110
#     mkdir -p Outputs/convergence/logs && sbatch --array=0-29%6 Submit_convergence/submit.sh conv_20261110
# Every run only produces the bare path_lifetimes.csv (--bare_only: no hybrid scans, no dynamics) of the full BZ, T_mag = T_phon = T.
# A channel-buffer overflow (the scan then only warns about truncated lifetimes) is detected, anything above the memory budget is
# reported and skipped, otherwise the run is repeated with the suggested anticipated_fraction x 1.2.
# Output per run: Outputs/convergence/<TAG>/<run>/{config.toml, params.txt, run.log, status.txt, path_lifetimes.csv}
# Post-processing: python/convergence_lifetimes.py and python/convergence_Gmp.py <scan dir>.

TAG=$1
[ -n "$TAG" ] || { echo "usage: sbatch --array=... Submit_convergence/submit.sh TAG"; exit 1; }
ROOT=Outputs/convergence/$TAG

module purge
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
module load CUDA/12.9.1
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
source venv/bin/activate
export PYTHONUNBUFFERED=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $ROOT/runs.csv)
IFS=, read ID RUN MATERIAL N MESH SMEAR FRAC PHPH FC3 T <<RUNLINE
$LINE
RUNLINE
[ -n "$RUN" ] || { echo "no run for task $SLURM_ARRAY_TASK_ID in $ROOT/runs.csv"; exit 1; }
RD=$ROOT/$RUN
mkdir -p $RD $ROOT/code
cp -n python/SpinPhony.py python/phph.py python/phph_gpu.py $ROOT/code/ 2>/dev/null

write_config() {   # $1 = anticipated_fraction ; mesh set only inside the [materials.<MATERIAL>] block
    sed -e "s|^anticipated_fraction = [0-9.eE+-]*|anticipated_fraction = $1|" \
        -e "/^\[materials.$MATERIAL\]/,/^\[/ s|^mesh = .*|mesh = \"$MESH\"|" config.toml > $RD/config.toml
}
cat > $RD/params.txt <<PARAMS
material=$MATERIAL
N=$N
mesh=$MESH
smearing=$SMEAR
T=$T
phph=$PHPH
frac0=$FRAC
git_commit=$(git rev-parse --short HEAD 2>/dev/null)
node=$(hostname)
PARAMS
echo "run $RUN  (task $SLURM_ARRAY_TASK_ID)  material $MATERIAL  N $N  smearing $SMEAR  phph $PHPH  frac0 $FRAC"
nvidia-smi --query-gpu=name,memory.total --format=csv

FLAGS="--full_bz --no_slc --bare_only"
[ "$PHPH" = "1" ] && FLAGS="$FLAGS --phph --phph_fc3 $FC3"

TRY=0
STATUS=failed
while [ $TRY -lt 4 ]; do
    TRY=$((TRY + 1))
    write_config $FRAC
    grep -n "^anticipated_fraction" $RD/config.toml
    grep -n "^mesh" $RD/config.toml | grep "$MESH"
    python python/SpinPhony.py --config $RD/config.toml --material $MATERIAL $FLAGS --tmag $T --tphon $T --smearing $SMEAR \
        --out_dir $RD > $RD/run.log 2>&1
    RC=$?
    if [ $RC -eq 0 ] && [ -f $RD/path_lifetimes.csv ]; then STATUS=ok; break; fi
    if grep -q "CUDA_ERROR_OUT_OF_MEMORY" $RD/run.log; then STATUS=oom; break; fi
    if grep -q "OVERFLOW" $RD/run.log; then
        NEED=$(grep -o "to >= [0-9.]*" $RD/run.log | tail -1 | awk '{print $3}')
        FRAC=$(awk -v x="$NEED" 'BEGIN{printf "%.5f", x * 1.2}')
        if awk -v n=$N -v f=$FRAC 'BEGIN{exit !(36*9*n^6*f/1e9 > 75)}'; then STATUS="memory_limit(needs fraction $FRAC)"; break; fi
        echo "overflow: retry $TRY with anticipated_fraction $FRAC"
        continue
    fi
    STATUS="error(rc=$RC)"; break
done
echo "status=$STATUS tries=$TRY frac=$FRAC" | tee $RD/status.txt
