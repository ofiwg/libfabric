#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn

# Run unit tests.  $(CWD) should be writeable.
./cxitest --verbose --tap=cxitest.tap

# Run tests again with RPut disabled
FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --tap=cxitest-swrdzv.tap

PYCXI="../../../../pycxi"
CSRUTIL="$PYCXI/utils/csrutil"

if [ -e $CSRUTIL ]; then
	# Run tests with RPut and SW Gets
	. $PYCXI/.venv/bin/activate
	$CSRUTIL store csr get_ctrl get_en=0
	./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swget.tap
	$CSRUTIL store csr get_ctrl get_en=1

	# Run tests with constrained LE count
	MAX_ALLOC=`$CSRUTIL dump csr le_pools[63] |grep max_alloc |awk '{print $3}'`
	$CSRUTIL store csr le_pools[] max_alloc=10
	./cxitest --verbose --filter="tagged/fc*" --tap=cxitest-fc.tap
	$CSRUTIL store csr le_pools[] max_alloc=$MAX_ALLOC
else
	echo "No csrutil"
fi
