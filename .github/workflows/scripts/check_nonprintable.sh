#!/bin/bash

BASE_SHA="$1"
HEAD_SHA="$2"

if [ -z "$BASE_SHA" ] || [ -z "$HEAD_SHA" ]; then
    echo "Usage: $0 <base_sha> <head_sha>"
    exit 1
fi

echo "Checking for non-printable characters..."

# Store current directory and change to git root for proper file paths
SCRIPT_DIR="$(pwd)"
cd "$(git rev-parse --show-toplevel)"

# Get added lines from text files only
if [ -s "$SCRIPT_DIR/text_files.txt" ]; then
    # Build file list manually for compatibility
    files=""
    while IFS= read -r file; do
        files="$files $file"
    done < "$SCRIPT_DIR/text_files.txt"

    if [ -n "$files" ]; then
        git diff --no-color --unified=0 "$BASE_SHA..$HEAD_SHA" -- $files | \
          grep '^+' | grep -v '^+++' > "$SCRIPT_DIR/pr_added_lines.txt" || true
    else
        touch "$SCRIPT_DIR/pr_added_lines.txt"
    fi
else
    touch "$SCRIPT_DIR/pr_added_lines.txt"
fi

# Change back to script directory for the rest of the script
cd "$SCRIPT_DIR"

if [ ! -s pr_added_lines.txt ]; then
    echo "No added lines to scan."
    exit 0
fi

# Check for control characters and problematic Unicode whitespace
# Control chars, no-break space, zero-width space, etc.
if LC_ALL=C grep -nP '[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|\xC2\xA0|\xE2\x80\x8B' pr_added_lines.txt > hits.txt 2>/dev/null; then
    echo "Non-printable or problematic characters detected:"
    while IFS= read -r line; do
        echo "$line"
        echo "$line" | sed 's/^[0-9]*://' | hexdump -C
        echo ""
    done < hits.txt
    exit 1
else
    echo "No problematic characters found."
    exit 0
fi
