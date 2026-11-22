#!/usr/bin/env bash
# Downloads the Bulgarian Belot networks (not in git: ~40 MB) into the Belot
# app's assets, from the release belot-models-vN of this repository (private:
# needs gh logged in, or GH_TOKEN in CI). scripts/belot_models.sh copies them
# from a belote-bot checkout instead.
#   scripts/fetch_belot_models.sh                 # belot-models-v1
#   BELOT_MODELS_TAG=belot-models-v2 scripts/fetch_belot_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${BELOT_MODELS_TAG:-belot-models-v1}"
DEST=android/belot-app/src/main/assets/models
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
gh release download "$TAG" --repo frigoow/cardtable --pattern belot-app-models.zip --dir "$TMP"
mkdir -p "$DEST"
unzip -o -q "$TMP/belot-app-models.zip" -d "$DEST"
echo "Models in $DEST:"; ls -1 "$DEST"
