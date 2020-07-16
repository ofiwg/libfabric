#!/usr/bin/env bats

load test_helper

# RC                                                                                                                                     
@test "imb_P2P unirandom 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_P2P unirandom"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_P2P unirandom 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_P2P unirandom"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_P2P birandom 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_P2P birandom"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_P2P birandom 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_P2P birandom"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_P2P corandom 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_P2P corandom"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_P2P corandom 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_P2P corandom"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_EXT window 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_EXT window"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_EXT window 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_EXT window"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_EXT accumulate 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_EXT accumulate"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_EXT accumulate 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_EXT accumulate"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_RMA bidir_get 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA bidir_get"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_RMA bidir_get 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA bidir_get"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_RMA bidir_put 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA bidir_put"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_RMA bidir_put 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA bidir_put"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_RMA unidir_get 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA unidir_get"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_RMA unidir_get 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA unidir_get"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_RMA unidir_put 2 ranks, 1 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA unidir_put"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_RMA unidir_put 2 ranks, 1 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 2 1) timeout 300 "$IMB_BUILD_PATH/imb_RMA unidir_put"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iallgather 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgather"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iallgather 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgather"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iallgather_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgather_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iallgather_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgather_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iallgatherv 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgatherv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iallgatherv 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgatherv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iallgatherv_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgatherv_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iallgatherv_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallgatherv_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iallreduce 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallreduce"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iallreduce 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallreduce"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iallreduce_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallreduce_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iallreduce_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iallreduce_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ialltoall 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoall"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ialltoall 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoall"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ialltoall_pure 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoall_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ialltoall_pure 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoall_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ialltoallv 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoallv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ialltoallv 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoallv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ialltoallv_pure 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoallv_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ialltoallv_pure 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_NBC ialltoallv_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ibarrier 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibarrier"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ibarrier 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibarrier"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ibarrier_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibarrier_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ibarrier_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibarrier_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ibcast 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibcast"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ibcast 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibcast"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ibcast_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibcast_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ibcast_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ibcast_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC igather 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igather"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC igather 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igather"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC igather_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igather_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC igather_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igather_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC igatherv 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igatherv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC igatherv 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igatherv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC igatherv_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igatherv_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC igatherv_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC igatherv_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ireduce 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ireduce 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ireduce_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ireduce_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ireduce_scatter 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce_scatter"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ireduce_scatter 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce_scatter"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC ireduce_scatter_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce_scatter_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC ireduce_scatter_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC ireduce_scatter_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iscatter 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatter"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iscatter 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatter"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iscatter_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatter_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iscatter_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatter_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iscatterv 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatterv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iscatterv 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatterv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_NBC iscatterv_pure 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatterv_pure"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_NBC iscatterv_pure 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_NBC iscatterv_pure"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 reduce 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 reduce"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 reduce 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 reduce"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 reduce_scatter 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 reduce_scatter"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 reduce_scatter 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 reduce_scatter"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 reduce_scatter_block 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 reduce_scatter_block"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 reduce_scatter_block 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 reduce_scatter_block"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 allreduce 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 allreduce"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 allreduce 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 allreduce"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 allgather 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 allgather"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 allgather 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 allgather"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 allgatherv 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 allgatherv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 allgatherv 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 allgatherv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 scatter 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 scatter"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 scatter 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 scatter"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 scatterv 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 scatterv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 scatterv 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 scatterv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 gather 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 gather"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 gather 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 gather"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 gatherv 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 gatherv"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 gatherv 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 gatherv"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 alltoall 20 ranks, 5 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 alltoall"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 alltoall 20 ranks, 5 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 20 5) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 alltoall"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 bcast 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 bcast"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 bcast 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 bcast"
        [ "$status" -eq 0 ]
}
# RC                                                                                                                                     
@test "imb_MPI1 barrier 40 ranks, 10 ranks per node using RC verbs" {
        run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 barrier"
        [ "$status" -eq 0 ]
}

# XRC
@test "imb_MPI1 barrier 40 ranks, 10 ranks per node using XRC verbs" {
        FI_OFI_RXM_USE_SRX=1 FI_VERBS_PREFER_XRC=1 run $CONTRIB_BIN/logwrap -w ${BATS_TEST_LOGFILE} -- \
                $(batch_launcher 40 10) timeout 300 "$IMB_BUILD_PATH/imb_MPI1 barrier"
        [ "$status" -eq 0 ]
}
