import copy
import pytest
import subprocess
from efa.efa_common import efa_retrieve_hw_counter_value, get_efa_device_names
from common import ClientServerTest

# This test must be run in serial mode because it checks the hw counter
@pytest.mark.serial
@pytest.mark.functional
@pytest.mark.parametrize("selection_approach", ["domain name", "environment"])
def test_efa_device_selection(cmdline_args, fabric, selection_approach):

    if cmdline_args.server_id == cmdline_args.client_id:
        pytest.skip("EFA device selection test requires 2 nodes")
        return

    server_device_names = get_efa_device_names(cmdline_args.server_id)
    client_device_names = get_efa_device_names(cmdline_args.client_id)

    server_num_devices = len(server_device_names)
    client_num_devices = len(server_device_names)

    for i in range(max(server_num_devices, client_num_devices)):
        server_device_idx = i % server_num_devices
        client_device_idx = i % client_num_devices

        server_device_name = server_device_names[server_device_idx]
        client_device_name = client_device_names[client_device_idx]

        for suffix in ["rdm", "dgrm"]:
            if fabric == "efa-direct" and suffix == "dgrm":
                continue
            server_tx_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "tx_bytes", server_device_name)
            client_tx_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "tx_bytes", client_device_name)

            prefix_type = "wout_prefix"
            strict_fabtests_mode = True
            if suffix == "rdm":
                command = "fi_rdm_pingpong"
            else:
                prefix_type = "with_prefix"  # efa provider requires prefix mode for dgram provider, hence "-k"
                strict_fabtests_mode = False  # # dgram is unreliable
                command = "fi_dgram_pingpong"

            server_domain_name = server_device_name + "-" + suffix
            client_domain_name = client_device_name + "-" + suffix

            cmdline_args_copy = copy.copy(cmdline_args)
            if selection_approach == "domain name":
                cmdline_args_copy.additional_server_arguments = "-d " + server_domain_name
                cmdline_args_copy.additional_client_arguments = "-d " + client_domain_name
            else:
                assert selection_approach == "environment"
                cmdline_args_copy.append_server_environ(f"FI_EFA_IFACE={server_device_name}")
                cmdline_args_copy.append_client_environ(f"FI_EFA_IFACE={client_device_name}")
            cmdline_args_copy.strict_fabtests_mode = strict_fabtests_mode

            test = ClientServerTest(cmdline_args_copy, command, message_size="1000", prefix_type=prefix_type, timeout=300, fabric=fabric, iteration_type="short")
            test.run()

            server_tx_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "tx_bytes", server_device_name)
            client_tx_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "tx_bytes", client_device_name)

            # Verify EFA traffic
            assert server_tx_bytes_before_test < server_tx_bytes_after_test
            assert client_tx_bytes_before_test < client_tx_bytes_after_test

# Verify that fi_getinfo does not return any info objects when FI_EFA_IFACE is set to an invalid value
@pytest.mark.functional
def test_efa_device_selection_negative(cmdline_args, fabric):
    invalid_iface = "r"

    command = cmdline_args.populate_command(f"fi_efa_info_test -f {fabric}", "host", additional_environment=f"FI_EFA_IFACE={invalid_iface}")
    proc = subprocess.run(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 61

# Verify that fi_getinfo returns all NICs when FI_EFA_IFACE is set to all
@pytest.mark.functional
def test_efa_device_selection_all(cmdline_args, fabric):
    num_devices = len(get_efa_device_names(cmdline_args.server_id))

    command = cmdline_args.populate_command(f"fi_efa_info_test -f {fabric}", "host", additional_environment="FI_EFA_IFACE=all")
    proc = subprocess.run(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0

    num_domains = 0
    info_out = proc.stdout.strip().split("\n")
    for info in info_out:
        if "domain" in info:
            num_domains += 1

    assert num_domains == num_devices

# Verify that fi_getinfo returns two NICs when FI_EFA_IFACE is set to two NICs separated by a comma
@pytest.mark.functional
def test_efa_device_selection_comma(cmdline_args, fabric):
    devices = get_efa_device_names(cmdline_args.server_id)

    if len(devices) > 1:
        iface_str = f"{devices[0]},{devices[1]}"
    else:
        iface_str = f"{devices[0]},{devices[0]}"

    command = cmdline_args.populate_command(f"fi_efa_info_test -f {fabric}", "host", additional_environment=f"FI_EFA_IFACE={iface_str}")
    proc = subprocess.run(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0

    num_domains = 0
    info_out = proc.stdout.strip().split("\n")
    for info in info_out:
        if "domain" in info:
            num_domains += 1

    if len(devices) > 1:
        assert num_domains == 2
    else:
        assert num_domains == 1
