import pytest


# This test must be run in serial mode because it checks the hw counter
@pytest.mark.serial
@pytest.mark.functional
def test_efa_device_selection(cmdline_args):
    from efa.efa_common import efa_retrieve_hw_counter_value, get_efa_domain_names
    from common import ClientServerTest

    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("EFA device selection test requires 2 nodes")
        return

    efa_domain_names = get_efa_domain_names(cmdline_args.server_id)

    for efa_domain_name in efa_domain_names:
        if '-rdm' in efa_domain_name:
            assert not('-dgrm') in efa_domain_name
            fabtest_opts = "fi_rdm_pingpong"
        elif '-dgrm' in efa_domain_name:
            assert not('-rdm') in efa_domain_name
            fabtest_opts = "fi_dgram_pingpong -k"

        efa_device_name = efa_domain_name.split('-')[0]

        server_tx_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "tx_bytes", efa_device_name)
        client_tx_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "tx_bytes", efa_device_name)

        executable = "{} -d {}".format(fabtest_opts, efa_domain_name)
        test = ClientServerTest(cmdline_args, executable, message_size="1000", timeout=300)
        test.run()

        server_tx_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "tx_bytes", efa_device_name)
        client_tx_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "tx_bytes", efa_device_name)

        # Verify EFA traffic
        assert server_tx_bytes_before_test < server_tx_bytes_after_test
        assert client_tx_bytes_before_test < client_tx_bytes_after_test
