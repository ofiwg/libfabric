#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

# Run unit tests.  $(CWD) should be writeable.
./cxitest --verbose --tap=cxitest.tap
