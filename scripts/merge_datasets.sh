#!/bin/bash
# Merge bottle_cup_0212 and piper-all-merged-0212 into piper-all-merged-0212
# Data root: /root/data/piper_dataset (adjust ROOT if your data is elsewhere)

ROOT="${1:-/root/data/piper_dataset}"
TEMP_OUTPUT="piper-all-merged-0212_merged"
FINAL_NAME="piper-all-merged"

cd "$(dirname "$0")/.." || exit 1

# Step 1: Merge to temp folder (avoids overwriting source piper-all-merged)
lerobot-edit-dataset \
    --repo_id "$TEMP_OUTPUT" \
    --root "$ROOT" \
    --operation.type merge \
    --operation.repo_ids "['bottle_cup_0212', 'piper-all-merged']"

# Step 2: Replace old piper-all-merged with merged result
rm -rf "$ROOT/$FINAL_NAME"
mv "$ROOT/$TEMP_OUTPUT" "$ROOT/$FINAL_NAME"

echo "Done. Merged dataset at $ROOT/$FINAL_NAME"
