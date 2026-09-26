#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after you create the GitHub repository.
git init -b main
git add .
git commit -m "chore: initialize business entity resolution project"

echo "Next: add your GitHub remote, then run:"
echo "git remote add origin <YOUR_GITHUB_REPO_URL>"
echo "git push -u origin main"
