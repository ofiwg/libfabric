#!/usr/bin/env bats

load test_helper

@test "fi_info" {
    run ${CONTRIB_BIN}/logwrap -w ${BATS_TEST_LOGFILE} -- $(batch_launcher 1 1) ${LIBFABRIC_INSTALL_PATH}/bin/fi_info -p 'verbs;ofi_rxm'
    [ "$status" -eq 0 ]
}  
