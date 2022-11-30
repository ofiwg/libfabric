import subprocess
import pytest

# This test must be run in serial mode because it checks the hw counter
@pytest.mark.serial
@pytest.mark.functional
def test_efa_device_selection(cmdline_args):
    from efa.efa_common import efa_retrieve_hw_counter_value
    from common import ClientServerTest

    command = "ssh {} ibv_devices".format(cmdline_args.server_id)
    process = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if not (process.stdout):
        raise Exception("No EFA devices found")

    # The output of the above process command will be list of efa devices available on the cluster. 
    # The naming of efa devices can be different on different clusters but the output of the command will
    # follow the format below
    #
    #   device                 node GUID
    #   ------              ----------------
    #   rdmap16s27          0000000000000000
    #   rdmap32s27          0000000000000000
    #   rdmap144s27         0000000000000000
    #   rdmap160s27         0000000000000000
    #
    # The above output will result as shown below when .split() is used
    #   ['device', 'node', 'GUID', '------', '----------------', 'rdmap16s27', '0000000000000000', 
    #    'rdmap32s27', '0000000000000000', 'rdmap144s27', '0000000000000000', 'rdmap160s27', '0000000000000000']
    
    # Extract the efa device names 
    efa_device_list = process.stdout.split()[5::2]

    for device_index in range(len(efa_device_list)):
        assert isinstance(efa_device_list[device_index], str)
        server_tx_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "tx_bytes", efa_device_list[device_index])
        client_tx_bytes_before_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "tx_bytes", efa_device_list[device_index])

        executable = "fi_rdm_pingpong -p efa --info-index {}".format(device_index)
        test = ClientServerTest(cmdline_args, executable, message_size="1000", timeout=300)
        test.run()

        server_tx_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.server_id, "tx_bytes", efa_device_list[device_index])
        client_tx_bytes_after_test = efa_retrieve_hw_counter_value(cmdline_args.client_id, "tx_bytes", efa_device_list[device_index])

        # Verify EFA traffic
        assert server_tx_bytes_before_test < server_tx_bytes_after_test 
        assert client_tx_bytes_before_test < client_tx_bytes_after_test