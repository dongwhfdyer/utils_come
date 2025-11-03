#!/bin/bash

# Single commit workflow script
# This script maintains only one commit by squashing all history

set -e

# Get commit message from argument or use default
COMMIT_MSG="${1:-Update repository}"

echo "Creating single commit with message: $COMMIT_MSG"

# Add all changes
git add -A

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit"
    exit 0
fi

# Create a new orphan branch
TEMP_BRANCH="temp-single-$(date +%s)"
git checkout --orphan "$TEMP_BRANCH"

# Commit all files
git add -A
git commit -m "$COMMIT_MSG"

# Force replace main branch
git branch -D main 2>/dev/null || true
git branch -m "$TEMP_BRANCH" main

# Force push to remote (if remote exists)
if git remote get-url origin >/dev/null 2>&1; then
    echo "Force pushing to remote..."
    git push -f origin main
fi

echo "Repository now has only one commit: $COMMIT_MSG"