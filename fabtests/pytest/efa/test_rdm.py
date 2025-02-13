from efa.efa_common import efa_run_client_server_test
from common import perf_progress_model_cli

import pytest
import copy


@pytest.mark.functional
def test_rdm_efa(cmdline_args, completion_semantic, fabric):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_rdm", completion_semantic=completion_semantic, fabric=fabric)
    test.run()

@pytest.mark.functional
def test_rdm_bw_functional_efa(cmdline_args, completion_semantic, fabric):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_flood -e rdm -v -T 1", completion_semantic=completion_semantic, fabric=fabric)
    test.run()

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_pingpong(cmdline_args, iteration_type, completion_semantic, memory_type_bi_dir, completion_type, fabric):
    command = "fi_rdm_pingpong"  + " " + perf_progress_model_cli
    efa_run_client_server_test(cmdline_args, command, iteration_type,
                               completion_semantic, memory_type_bi_dir, "all",
                               completion_type=completion_type, fabric=fabric)

@pytest.mark.functional
@pytest.mark.serial
def test_mr_exhaustion_rdm_pingpong(cmdline_args, completion_semantic, fabric):
    efa_run_client_server_test(cmdline_args, "fi_efa_exhaust_mr_reg_rdm_pingpong", "short",
                                completion_semantic, "host_to_host", "all", timeout=1000,
                                fabric=fabric)

@pytest.mark.functional
def test_rdm_pingpong_range(cmdline_args, completion_semantic, memory_type_bi_dir, message_size, fabric):
    efa_run_client_server_test(cmdline_args, "fi_rdm_pingpong", "short",
                               completion_semantic, memory_type_bi_dir, message_size, fabric=fabric)

@pytest.mark.functional
def test_rdm_pingpong_no_inject_range(cmdline_args, completion_semantic, inject_message_size, fabric):
    efa_run_client_server_test(cmdline_args, "fi_rdm_pingpong -j 0", "short",
                               completion_semantic, "host_to_host", inject_message_size, fabric=fabric)

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_tagged_pingpong(cmdline_args, iteration_type, completion_semantic, memory_type_bi_dir, completion_type, fabric):
    command = "fi_rdm_tagged_pingpong"  + " " + perf_progress_model_cli
    efa_run_client_server_test(cmdline_args, command, iteration_type,
                               completion_semantic, memory_type_bi_dir, "all", completion_type=completion_type,
                               fabric=fabric)

@pytest.mark.functional
def test_rdm_tagged_pingpong_range(cmdline_args, completion_semantic, memory_type_bi_dir, message_size, fabric):
    efa_run_client_server_test(cmdline_args, "fi_rdm_tagged_pingpong", "short",
                               completion_semantic, memory_type_bi_dir, message_size,
                               fabric=fabric)

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_tagged_bw(cmdline_args, iteration_type, completion_semantic, memory_type, completion_type, fabric):
    command = "fi_rdm_tagged_bw"  + " " + perf_progress_model_cli
    efa_run_client_server_test(cmdline_args, command, iteration_type,
                               completion_semantic, memory_type, "all", completion_type=completion_type,
                               fabric=fabric)

@pytest.mark.functional
def test_rdm_tagged_bw_range(cmdline_args, completion_semantic, memory_type, message_size, fabric):
    efa_run_client_server_test(cmdline_args, "fi_rdm_tagged_bw", "short",
                               completion_semantic, memory_type, message_size, fabric=fabric)

@pytest.mark.functional
def test_rdm_tagged_bw_no_inject_range(cmdline_args, completion_semantic, inject_message_size, fabric):
    efa_run_client_server_test(cmdline_args, "fi_rdm_tagged_bw -j 0", "short",
                               completion_semantic, "host_to_host", inject_message_size, fabric=fabric)

@pytest.mark.functional
@pytest.mark.parametrize("env_vars", [["FI_EFA_TX_SIZE=64"], ["FI_EFA_RX_SIZE=64"], ["FI_EFA_TX_SIZE=64", "FI_EFA_RX_SIZE=64"]])
def test_rdm_tagged_bw_small_tx_rx(cmdline_args, completion_semantic, memory_type, completion_type, env_vars, fabric):
    cmdline_args_copy = copy.copy(cmdline_args)
    for env_var in env_vars:
        cmdline_args_copy.append_environ(env_var)
    # Use a window size larger than tx/rx size
    efa_run_client_server_test(cmdline_args_copy, "fi_rdm_tagged_bw -W 128", "short",
                               completion_semantic, memory_type, "all", completion_type=completion_type,
                               fabric=fabric)

@pytest.mark.functional
def test_rdm_tagged_bw_use_fi_more(cmdline_args, completion_semantic, memory_type, message_size, fabric):
    efa_run_client_server_test(cmdline_args, "fi_rdm_tagged_bw --use-fi-more",
                               "short", completion_semantic, memory_type, message_size, fabric=fabric)

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_rdm_atomic(cmdline_args, iteration_type, completion_semantic, memory_type, fabric):
    from copy import copy

    from common import ClientServerTest

    if "neuron" in memory_type:
        pytest.skip("Neuron does not fully support atomics")

    # the rdm_atomic test's run time has a high variance when running single c6gn instance.
    # the issue is tracked in:  https://github.com/ofiwg/libfabric/issues/7002
    # to mitigate the issue, set the maximum timeout of fi_rdm_atomic to 1800 seconds.
    cmdline_args_copy = copy(cmdline_args)
    command = "fi_rdm_atomic"  + " " + perf_progress_model_cli
    test = ClientServerTest(cmdline_args_copy, "fi_rdm_atomic", iteration_type, completion_semantic,
                            memory_type=memory_type, timeout=1800, fabric=fabric)
    test.run()

@pytest.mark.functional
def test_rdm_tagged_peek(cmdline_args):
    from copy import copy

    from common import ClientServerTest

    test = ClientServerTest(cmdline_args, "fi_rdm_tagged_peek", timeout=1800)
    test.run()

# This test is run in serial mode because it takes a lot of memory
@pytest.mark.serial
@pytest.mark.functional
def test_rdm_pingpong_1G(cmdline_args, completion_semantic, fabric):
    # Default window size is 64 resulting in 128GB being registered, which
    # exceeds max number of registered host pages
    efa_run_client_server_test(cmdline_args, "fi_rdm_pingpong -W 1", 2,
                               completion_semantic=completion_semantic, message_size=1073741824,
                               memory_type="host_to_host", warmup_iteration_type=0, fabric=fabric)

@pytest.mark.functional
def test_rdm_pingpong_zcpy_recv(cmdline_args, memory_type_bi_dir, zcpy_recv_max_msg_size, zcpy_recv_message_size, fabric):
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("no zero copy recv for intra-node communication")
    cmdline_args_copy = copy.copy(cmdline_args)
    cmdline_args_copy.append_environ("FI_EFA_ENABLE_SHM_TRANSFER=0")
    efa_run_client_server_test(cmdline_args_copy, f"fi_rdm_pingpong --max-msg-size {zcpy_recv_max_msg_size}",
                               "short", "transmit_complete", memory_type_bi_dir, zcpy_recv_message_size, fabric=fabric)

@pytest.mark.functional
def test_rdm_bw_zcpy_recv(cmdline_args, memory_type, zcpy_recv_max_msg_size, zcpy_recv_message_size, fabric):
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("no zero copy recv for intra-node communication")
    cmdline_args_copy = copy.copy(cmdline_args)
    cmdline_args_copy.append_environ("FI_EFA_ENABLE_SHM_TRANSFER=0")
    efa_run_client_server_test(cmdline_args_copy, f"fi_rdm_bw --max-msg-size {zcpy_recv_max_msg_size}",
                               "short", "transmit_complete", memory_type, zcpy_recv_message_size, fabric=fabric)

@pytest.mark.functional
def test_rdm_bw_zcpy_recv_use_fi_more(cmdline_args, memory_type, zcpy_recv_max_msg_size, zcpy_recv_message_size, fabric):
    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("no zero copy recv for intra-node communication")
    cmdline_args_copy = copy.copy(cmdline_args)
    cmdline_args_copy.append_environ("FI_EFA_ENABLE_SHM_TRANSFER=0")
    efa_run_client_server_test(cmdline_args_copy, f"fi_rdm_bw --use-fi-more --max-msg-size {zcpy_recv_max_msg_size}",
                               "short", "transmit_complete", memory_type, zcpy_recv_message_size, fabric=fabric)
