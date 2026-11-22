#!/usr/bin/env bash
# CI: checks out the private submodule engines/belote (frigoow/belote-bot)
# with the read-only deploy key in $KEY (secret BELOTE_BOT_DEPLOY_KEY).
# Sets the step output ok=true when it did; without a key it warns and sets
# ok=false, so the Belot steps are skipped rather than failed.
set -euo pipefail
if [ -z "${KEY:-}" ]; then
    echo "::warning::secret BELOTE_BOT_DEPLOY_KEY is not set: Belot is not built"
    echo "ok=false" >> "$GITHUB_OUTPUT"
    exit 0
fi
mkdir -p ~/.ssh
printf '%s\n' "$KEY" > ~/.ssh/belote_bot
chmod 600 ~/.ssh/belote_bot
# the submodule's relative URL resolves to https; read it over SSH with the key
git config --global url."git@github.com:frigoow/belote-bot.git".insteadOf "https://github.com/frigoow/belote-bot.git"
GIT_SSH_COMMAND="ssh -i ~/.ssh/belote_bot -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
    git submodule update --init engines/belote
echo "ok=true" >> "$GITHUB_OUTPUT"
