#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn

# Run unit tests.  $(CWD) should be writeable.
if [[ $# -eq 0 ]]; then
    ./cxitest --verbose --tap=cxitest.tap
    exit $?
fi
for arg in $@; do
    ./cxitest --verbose --filter="@($arg)" --tap=cxitest.tap || exit $?
done
