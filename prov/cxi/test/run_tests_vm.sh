#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

# Run unit tests.  $(CWD) should be writeable.
#
# Tests must be run with -j1 to prevent processes from racing to allocate CXI
# resources.
./cxitest -j1 --verbose --tap=cxitest.tap
