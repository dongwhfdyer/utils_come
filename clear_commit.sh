#!/bin/bash

# Clear commit script
# Removes all files except single_commit.sh and com_scripts/ directory

set -e

cd /Users/kuhn/Documents/code/transfer_things/utils_come

echo "Clearing repository (keeping only single_commit.sh and com_scripts/)..."

# Remove all files except the ones we want to keep
find . -maxdepth 1 -type f ! -name 'single_commit.sh' ! -name '.gitignore' ! -name 'clear_commit.sh' -delete
find . -maxdepth 1 -type d ! -name '.' ! -name '.git' ! -name 'com_scripts' -exec rm -rf {} +

echo "Files cleared. Creating single commit..."

# Use the single_commit.sh script to commit the cleared state
# ./single_commit.sh "Clear repository - keep only essential files"

echo "Repository cleared and committed!"