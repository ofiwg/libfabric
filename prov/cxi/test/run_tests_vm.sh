#!/bin/bash

cd $(dirname $0)
. ./virtualize.sh

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn

# Run unit tests.  $(CWD) should be writeable.
./cxitest --verbose --tap=cxitest.tap

# Re-run messaging tests with RPut disabled
FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swrdzv.tap

PYCXI="../../../../pycxi"
CSRUTIL="$PYCXI/utils/csrutil"

if [ -e $CSRUTIL ]; then
	. $PYCXI/.venv/bin/activate

	# Run tests with RPut and SW Gets
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
