#!/bin/bash
#
# Run CXI unit tests.

DIR=`dirname $0`
cd $DIR

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn

# Run unit tests.  $(CWD) should be writeable.
./cxitest --verbose --tap=cxitest.tap

# Re-run messaging tests with RPut disabled
FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swrdzv.tap

# Run tests with RPut and SW Gets
csrutil store csr get_ctrl get_en=0
./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swget.tap
csrutil store csr get_ctrl get_en=1

# Run tests with constrained LE count
MAX_ALLOC=`csrutil dump csr le_pools[63] |grep max_alloc |awk '{print $3}'`
csrutil store csr le_pools[] max_alloc=10
./cxitest --verbose --filter="tagged/fc*" --tap=cxitest-fc.tap
csrutil store csr le_pools[] max_alloc=$MAX_ALLOC
