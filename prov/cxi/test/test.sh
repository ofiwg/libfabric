#!/bin/bash
#
# Run CXI unit tests.

DIR=`dirname $0`
cd $DIR
TEST_OUTPUT=cxitest.out

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn

if [[ $# -gt 0 ]]; then
    ./cxitest --verbose --filter="@($1)" --tap=cxitest.tap -j2 
    exit $?
fi

# Run unit tests.  $(CWD) should be writeable.
echo "running: ./cxitest --verbose --tap=cxitest.tap -j2 > $TEST_OUTPUT 2>&1"
./cxitest --verbose --tap=cxitest.tap -j2 > $TEST_OUTPUT 2>&1

# Re-run messaging tests with RPut disabled
echo "running: FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --filter=\"@(tagged|msg)/*\" --tap=cxitest-swrdzv.tap -j2 >> $TEST_OUTPUT 2>&1"
FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swrdzv.tap -j2 >> $TEST_OUTPUT 2>&1

# Run tests with RPut and SW Gets
csrutil store csr get_ctrl get_en=0 > /dev/null
echo "running: ./cxitest --verbose --filter=\"@(tagged|msg)/*\" --tap=cxitest-swget.tap -j2 >> $TEST_OUTPUT 2>&1"
./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swget.tap -j2 >> $TEST_OUTPUT 2>&1
csrutil store csr get_ctrl get_en=1 > /dev/null

# Run tests with constrained LE count
MAX_ALLOC=`csrutil dump csr le_pools[63] |grep max_alloc |awk '{print $3}'`
csrutil store csr le_pools[] max_alloc=10 > /dev/null
echo "running; ./cxitest --verbose --filter=\"tagged/fc*\" --tap=cxitest-fc.tap -j2 >> $TEST_OUTPUT 2>&1"
./cxitest --verbose --filter="tagged/fc*" --tap=cxitest-fc.tap -j2 >> $TEST_OUTPUT 2>&1
csrutil store csr le_pools[] max_alloc=$MAX_ALLOC > /dev/null

grep "Tested" $TEST_OUTPUT
