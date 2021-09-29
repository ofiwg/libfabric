#!/bin/bash
#
# Run CXI unit tests.

DIR=`dirname $0`
cd $DIR
TEST_OUTPUT=cxitest.out

export DMA_FAULT_RATE=.1
export MALLOC_FAULT_RATE=.1
export FI_LOG_LEVEL=warn
export FI_CXI_FC_RECOVERY=1

if [[ $# -gt 0 ]]; then
    ./cxitest --verbose --filter="@($1)" --tap=cxitest.tap -j2
    exit $?
fi

# Run unit tests.  $(CWD) should be writeable.
echo "running: ./cxitest --verbose --tap=cxitest.tap -j2 > $TEST_OUTPUT 2>&1"
./cxitest --verbose --tap=cxitest.tap -j2 > $TEST_OUTPUT 2>&1
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Re-run messaging tests with RPut disabled
#echo "running: FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --filter=\"@(tagged|msg)/*\" --tap=cxitest-swrdzv.tap -j2 >> $TEST_OUTPUT 2>&1"
#FI_CXI_RDZV_OFFLOAD=0 ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swrdzv.tap -j2 >> $TEST_OUTPUT 2>&1

# Run tests with RPut and SW Gets
csrutil store csr get_ctrl get_en=0 > /dev/null
echo "running: ./cxitest --verbose --filter=\"@(tagged|msg)/*\" --tap=cxitest-swget.tap -j2 >> $TEST_OUTPUT 2>&1"
./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swget.tap -j2 >> $TEST_OUTPUT 2>&1
cxitest_exit_status=$?
csrutil store csr get_ctrl get_en=1 > /dev/null
if [[ $cxitest_exit_status -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Run tests with constrained LE count
MAX_ALLOC=`csrutil dump csr le_pools[63] |grep max_alloc |awk '{print $3}'`
csrutil store csr le_pools[] max_alloc=10 > /dev/null
echo "running; ./cxitest --verbose --filter=\"tagged/fc*\" --tap=cxitest-fc.tap -j1 >> $TEST_OUTPUT 2>&1"
./cxitest --verbose --filter="tagged/fc*" --tap=cxitest-fc.tap -j1 >> $TEST_OUTPUT 2>&1
cxitest_exit_status=$?
csrutil store csr le_pools[] max_alloc=$MAX_ALLOC > /dev/null
if [[ $cxitest_exit_status -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Verify tag matching with rendezvous
test="FI_CXI_DEVICE_NAME=cxi1,cxi0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --verbose -j2 --filter=\"tagged_directed/*\" --tap=cxitest-hw-rdzv-tag-matching.tap >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

#test="FI_CXI_RDZV_OFFLOAD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --verbose -j2 --filter=\"tagged_directed/*\" --tap=cxitest-sw-rdzv-tag-matching.tap >> $TEST_OUTPUT 2>&1"
#echo "running: $test"
#eval $test

test="FI_CXI_MSG_OFFLOAD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --verbose -j 1 --tap=cxitest-sw-ep-mode.tap >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

test="FI_CXI_DEFAULT_CQ_SIZE=64 FI_CXI_DISABLE_CQ_HUGETLB=1 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --filter=\"tagged/fc_no_eq_space_expected_multi_recv\" --verbose -j 1 --tap=cxitest-fc-eq-space.tap >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

test="FI_CXI_DEVICE_NAME=cxi1 ../../../util/fi_info"
echo "running: $test"
eval $test
if [[ $? -eq 0 ]]; then
    echo "fi_info incorrectly returned fi_info"
    exit 1
fi

grep "Tested" $TEST_OUTPUT
