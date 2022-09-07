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

# Run unit tests. $(CWD) should be writeable.
test="./cxitest --verbose --tap=cxitest.tap -j 1 > $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Disable caching of FI_HMEM_SYSTEM and IOTLB.
test="FI_MR_CACHE_MONITOR=disable FI_CXI_IOTLB=0 ./cxitest --verbose --tap=cxitest-no-cache.tap -j 1 >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Run tests with RPut and SW Gets
csrutil store csr C_LPE_CFG_GET_CTRL get_en=0 > /dev/null
echo "running: FI_CXI_RGET_TC=BULK_DATA ./cxitest --verbose --filter=\"@(tagged|msg)/*\" --tap=cxitest-swget.tap -j 1 >> $TEST_OUTPUT 2>&1"
FI_CXI_RGET_TC=BULK_DATA ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swget.tap -j1 >> $TEST_OUTPUT 2>&1
cxitest_exit_status=$?
csrutil store csr C_LPE_CFG_GET_CTRL get_en=1 > /dev/null
if [[ $cxitest_exit_status -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Run tests with RPut and SW Gets forcing unaligned address for RGet
csrutil store csr C_LPE_CFG_GET_CTRL get_en=0 > /dev/null
echo "running: FI_CXI_RDZV_THRESHOLD=2036 ./cxitest --verbose --filter=\"@(tagged|msg)/*\" --tap=cxitest-swget-unaligned.tap -j 1 >> $TEST_OUTPUT 2>&1"
FI_CXI_RDZV_THRESHOLD=2036 ./cxitest --verbose --filter="@(tagged|msg)/*" --tap=cxitest-swget-unaligned.tap -j1 >> $TEST_OUTPUT 2>&1
cxitest_exit_status=$?
csrutil store csr C_LPE_CFG_GET_CTRL get_en=1 > /dev/null
if [[ $cxitest_exit_status -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Run tests with constrained LE count
MAX_ALLOC=`csrutil dump csr le_pools[63] |grep max_alloc |awk '{print $3}'`
csrutil store csr le_pools[] max_alloc=10 > /dev/null
echo "running: FI_CXI_DEFAULT_CQ_SIZE=16384 ./cxitest --verbose --filter=\"@(tagged|msg)/fc*\" --tap=cxitest-fc.tap -j1 >> $TEST_OUTPUT 2>&1"
FI_CXI_DEFAULT_CQ_SIZE=16384 ./cxitest --verbose --filter="@(tagged|msg)/fc*" --tap=cxitest-fc.tap -j1 >> $TEST_OUTPUT 2>&1
cxitest_exit_status=$?
csrutil store csr le_pools[] max_alloc=$MAX_ALLOC > /dev/null
if [[ $cxitest_exit_status -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Verify tag matching with rendezvous
test="FI_CXI_DEVICE_NAME=cxi1,cxi0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --verbose -j 1 --filter=\"tagged_directed/*\" --tap=cxitest-hw-rdzv-tag-matching.tap >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

test="FI_CXI_RX_MATCH_MODE=\"software\" FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --verbose -j 1 --tap=cxitest-sw-ep-mode.tap >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

test="FI_CXI_DEFAULT_CQ_SIZE=64 FI_CXI_DISABLE_CQ_HUGETLB=1 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --filter=\"msg/fc_no_eq_space_expected_multi_recv\" --verbose -j 1 --tap=cxitest-fc-eq-space.tap >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

test="FI_CXI_CQ_FILL_PERCENT=20 FI_CXI_DEFAULT_CQ_SIZE=64 FI_CXI_DISABLE_CQ_HUGETLB=1 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=2048 ./cxitest --filter=\"msg/fc_no_eq_space_expected_multi_recv\" --verbose -j 1 --tap=cxitest-fc-20%-eq-space.tap >> $TEST_OUTPUT 2>&1"
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

# Unoptimized MR testing
test="FI_CXI_OPTIMIZED_MRS=0 ./cxitest --filter=amo_hybrid_mr_desc/* -j 1 -f --verbose --tap=cxitest-hybrid_mr_desc_unopt_mrs >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# MR_PROV_KEY mr mode bit testing
test="CXIP_TEST_PROV_KEY=1 ./cxitest -j 1 -f --verbose --tap=cxitest-prov_key_mrs >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# MR_PROV_KEY mr mode bit without optimized MRs testing
test="CXIP_TEST_PROV_KEY=1 FI_CXI_OPTIMIZED_MRS=0 ./cxitest --filter=\"@(rma|mr)/*\" -j 1 -f --verbose --tap=cxitest-prov_key_no_opt_mrs >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# MR_PROV_KEY mr mode bit optimized to standard fallback without MR caching
test="CXIP_TEST_PROV_KEY=1 FI_MR_CACHE_MONITOR=disabled ./cxitest --filter=\"mr_resources/opt_fallback\" -j 1 -f --verbose --tap=cxitest-prov_key_opt_to_std >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

# Verify 0 rendezvous eager data with unexpected/expected processing
test="FI_CXI_RDZV_EAGER_SIZE=0 ./cxitest --filter=\"@(tagged|msg)/*\" -j 1 -f --verbose --tap=cxitest-zero-rdzv-eager-size >> $TEST_OUTPUT 2>&1"
echo "running: $test"
eval $test
if [[ $? -ne 0 ]]; then
    echo "cxitest return non-zero exit code. Possible failures in test teardown"
    exit 1
fi

grep "Tested" $TEST_OUTPUT
