#!/bin/bash

BASE_SHA="$1"
HEAD_SHA="$2"

if [ -z "$BASE_SHA" ] || [ -z "$HEAD_SHA" ]; then
    echo "Usage: $0 <base_sha> <head_sha>"
    echo "Example: $0 HEAD~1 HEAD"
    exit 1
fi

echo "Running PR character safety check..."

./check_text_to_binary.sh "$BASE_SHA" "$HEAD_SHA"
TEXT_TO_BINARY=$?

./check_nonprintable.sh "$BASE_SHA" "$HEAD_SHA"
NONPRINTABLE=$?

if [ $TEXT_TO_BINARY -ne 0 ] || [ $NONPRINTABLE -ne 0 ]; then
    echo "❌ PR character safety check failed"
    exit 1
else
    echo "✅ PR character safety check passed"
    exit 0
fi
