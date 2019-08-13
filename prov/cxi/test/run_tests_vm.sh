#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn

# Run unit tests.  $(CWD) should be writeable.
./cxitest --verbose --tap=cxitest.tap

# Run tests again with RPut enabled
RDZV_OFFLOAD=1 ./cxitest --verbose --tap=cxitest-rput.tap

PYCXI="../../../../pycxi"
CSRUTIL="$PYCXI/utils/csrutil"

# Run tests with RPut and SW Gets
if [ -e $CSRUTIL ]; then
	. $PYCXI/.venv/bin/activate
	$CSRUTIL store csr get_ctrl get_en=0
	RDZV_OFFLOAD=1 ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-rput-swget.tap
	$CSRUTIL store csr get_ctrl get_en=1
else
	echo "No csrutil"
fi
