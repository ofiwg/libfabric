#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

# Run unit tests.  $(CWD) should be writeable.
./cxitest --verbose --tap=cxitest.tap
# Run tests again with RPut enabled
RDZV_OFFLOAD=1 ./cxitest --verbose --tap=cxitest-rput.tap
