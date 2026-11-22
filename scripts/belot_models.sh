#!/usr/bin/env bash
# Copies the Bulgarian Belot networks from a belote-bot checkout's checkpoints/
# (gitignored there, not in git here) into the Belot app's assets, under the
# names cpp/belot/belot_api.cpp loads. The default set is the one measured at
# 54.35% +- 1.74 pp of 2000 games against Kostov's Master
# (belote-bot runs/bg_bridge_e2e_bid_dbl_5mspec_1000p.log).
#
#   scripts/belot_models.sh [path/to/belote-bot]     # default ../Belote
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-../Belote}/checkpoints"
DEST=android/belot-app/src/main/assets/models
mkdir -p "$DEST"
copy() { cp "$SRC/$1" "$DEST/$2"; echo "  $2 <- $1"; }
copy bg_models_bid_unmasked_dbl/bg_bid.bin  belot_bid.bin
for pair in suit:bg_models_suit nt:bg_models_nt_latest at:bg_models_at; do
    fam=${pair%%:*}; dir=${pair#*:}
    copy "$dir/bg_fast_policy.bin" "belot_${fam}_fast.bin"
    copy "$dir/bg_belief.bin"      "belot_${fam}_belief.bin"
    copy "$dir/bg_q_gru.bin"       "belot_${fam}_q.bin"
done
echo "Models in $DEST"
