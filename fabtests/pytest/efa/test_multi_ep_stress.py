import pytest
from common import ClientServerTest

@pytest.mark.unstable
def test_multi_ep_stress_standard(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_tagged(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --op-type tagged"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_writedata(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --op-type writedata"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_multi_receivers(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --receiver-workers 4"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()


@pytest.mark.unstable
def test_multi_ep_stress_multi_senders(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --sender-workers 10"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_shared_cq(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --receiver-workers 4 --shared-cq"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_shared_av(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --sender-workers 8 --receiver-workers 2 --shared-av"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_shared_av_and_cq(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --sender-workers 8 --receiver-workers 2 --shared-av --shared-cq"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
def test_multi_ep_stress_transient_client(cmdline_args):
    cmd = f"fi_efa_multi_ep_stress --sender-ep-cycles 5"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()

@pytest.mark.unstable
@pytest.mark.parametrize("configuration", [{"sender_workers": 2, "receiver_workers": 4, "sender_ep_cycles": 10, "receiver_ep_cycles": 20},
                                           {"sender_workers": 4, "receiver_workers": 2, "sender_ep_cycles": 20, "receiver_ep_cycles": 10},
                                           {"sender_workers": 2, "receiver_workers": 4, "sender_ep_cycles": 20, "receiver_ep_cycles": 10}
                                            ])
def test_multi_ep_stress_multi_and_transient_sender_receiver(cmdline_args, configuration):
    sender_workers = configuration["sender_workers"]
    receiver_workers = configuration["receiver_workers"]
    sender_ep_cycles = configuration["sender_ep_cycles"]
    receiver_ep_cycles = configuration["receiver_ep_cycles"]
    cmd = f"fi_efa_multi_ep_stress --sender-workers {sender_workers} --receiver-workers {receiver_workers} --sender-ep-cycles {sender_ep_cycles} --receiver-ep-cycles {receiver_ep_cycles}"
    test = ClientServerTest(cmdline_args, cmd, message_size=1024, fabric="efa", additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()
