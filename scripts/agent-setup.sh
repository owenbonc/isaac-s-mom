#!/bin/sh
# Run it once, first: the reset discards anything already on this branch.
set -eu
base="${1:?usage: scripts/agent-setup.sh <base-ref>, e.g. origin/phase-1}"
main="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"

git -c credential.helper= -c credential.helper='!gh auth git-credential' fetch -q origin
git reset -q --hard "$base"
git config user.name "Owen Yan"
git config user.email "100368790+owenbonc@users.noreply.github.com"
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
# Secrets stay in the main checkout; link, never copy, and never print them.
if [ ! -e .env ] && [ -e "$main/.env" ]; then ln -s "$main/.env" .env; fi
