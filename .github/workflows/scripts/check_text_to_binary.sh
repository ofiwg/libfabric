#!/bin/bash

BASE_SHA="$1"
HEAD_SHA="$2"

if [ -z "$BASE_SHA" ] || [ -z "$HEAD_SHA" ]; then
    echo "Usage: $0 <base_sha> <head_sha>"
    exit 1
fi

echo "Checking for files converted from text to binary between $BASE_SHA and $HEAD_SHA..."

git diff --name-only "$BASE_SHA..$HEAD_SHA" > changed_files.txt

FAILED=0
> text_files.txt
> text_to_binary_files.txt

while IFS= read -r file; do
    if git cat-file -e "$HEAD_SHA:$file" 2>/dev/null; then
        git show "$HEAD_SHA:$file" | file - | grep -q "text\|empty"
        head_is_text=$?

        if git cat-file -e "$BASE_SHA:$file" 2>/dev/null; then
            git show "$BASE_SHA:$file" | file - | grep -q "text\|empty"
            base_is_text=$?

            if [ $base_is_text -eq 0 ] && [ $head_is_text -ne 0 ]; then
                echo "$file" >> text_to_binary_files.txt
                FAILED=1
            fi
        fi

        if [ $head_is_text -eq 0 ]; then
            echo "$file" >> text_files.txt
        fi
    fi
done < changed_files.txt

if [ -s text_to_binary_files.txt ]; then
    echo "Files converted from text to binary:"
    cat text_to_binary_files.txt
fi

exit $FAILED
