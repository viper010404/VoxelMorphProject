#!/usr/bin/env bash
# Rebuild data/ from scratch: download, verify, extract, cache, and check that the regenerated
# splits still address the same subjects as the runs already in results/.
#
# That last step is the point. The splits come from a seeded permutation over the subject list,
# so a different subject count or ordering silently remaps every index -- existing results would
# still load and still plot, while comparing different brains. The check recomputes the
# unregistered Dice of the recorded test pairs and requires an exact match.
#
#   ./scripts/restore_dataset.sh 2d      # 24 MB, what the Colab notebook needs
#   ./scripts/restore_dataset.sh 3d      # 6.6 GB download, 17 GB cache
#   ./scripts/restore_dataset.sh both
set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-both}"
PY=./.venv/bin/python
BASE=https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data
# the host 403s curl's default user-agent
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

fetch() {  # fetch <file> <md5>; resumes, so an interrupted 6.6 GB download is not restarted
    local file=$1 want=$2
    for attempt in 1 2 3 4 5 6 7 8; do
        [ -f "data/$file" ] && [ "$(md5sum "data/$file" | cut -d' ' -f1)" = "$want" ] && break
        echo "  fetching $file (attempt $attempt)"
        curl -sSL -C - -A "$UA" --retry 5 --retry-delay 5 --retry-all-errors \
             -o "data/$file" "$BASE/$file" || true
    done
    local got; got=$(md5sum "data/$file" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { echo "  CHECKSUM MISMATCH for $file: $got != $want"; exit 1; }
    echo "  $file checksum OK"
}

verify() {  # verify <cache> <reference run>
    $PY - "$1" "$2" <<'PYEOF'
import json, sys
import numpy as np
from project.data import OasisData, default_label_policy, fixed_pairs
from project.metrics import dice_per_structure, mean_dice

cache, run = sys.argv[1], sys.argv[2]
data = OasisData(cache, device='cpu')
reference = json.load(open(f'results/{run}/eval_test.json'))

recorded = [(r['fixed'], r['moving']) for r in reference['per_pair']]
if fixed_pairs(data, 'test', 100, seed=1234) != recorded:
    raise SystemExit('  FAIL: regenerated pair list differs from the recorded one')

labels = default_label_policy(data, 'test')
worst = 0.0
for row in reference['per_pair']:
    fixed = data.seg_batch([row['fixed']]).squeeze().numpy()
    moving = data.seg_batch([row['moving']]).squeeze().numpy()
    worst = max(worst, abs(mean_dice(dice_per_structure(fixed, moving, labels))
                           - row['dice_initial']))
print(f'  splits verified against {run}: max |dice_initial diff| = {worst:.2e}')
if worst > 1e-9:
    raise SystemExit('  FAIL: indices address different subjects; results are NOT comparable')
PYEOF
}

mkdir -p data

if [ "$WHICH" = 2d ] || [ "$WHICH" = both ]; then
    echo "== 2D =="
    fetch neurite-oasis.2d.v1.0.tar c9ae5864f250c7e4b8d83a104e51ae8e
    [ -d data/OASIS_OAS1_0001_MR1 ] || tar -xf data/neurite-oasis.2d.v1.0.tar -C data
    [ -f data/oasis2d.npz ] || $PY -m project.prepare_data --data-dir data --out data/oasis2d.npz
    verify data/oasis2d.npz 2d_baseline_lam0.25_svf
fi

if [ "$WHICH" = 3d ] || [ "$WHICH" = both ]; then
    echo "== 3D =="
    fetch neurite-oasis.v1.0.tar 081392a8150ff99ab7a64a9ded377835
    if [ ! -d data/oasis3d ]; then
        mkdir -p data/oasis3d
        # template-space files only: 1.2 GB extracted instead of 6.6 GB
        tar -xf data/neurite-oasis.v1.0.tar -C data/oasis3d --wildcards \
            '*/aligned_norm.nii.gz' '*/aligned_seg35.nii.gz' '*/aligned_seg4.nii.gz' \
            'seg35_labels.txt' 'subjects.txt'
    fi
    [ -d data/oasis3d_cache ] || $PY -m project.prepare_data --ndim 3 \
        --data-dir data/oasis3d --out data/oasis3d_cache
    verify data/oasis3d_cache 3d_baseline_lam0.025_svf_s80k
fi

echo "done"
