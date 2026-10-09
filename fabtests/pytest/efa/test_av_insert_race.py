import pytest
from common import ClientServerTest

# fi_efa_av_insert_race: every worker thread owns an endpoint on an AV shared by
# all workers of its process. Workers insert remote endpoints only when the
# first message from them arrives, or after a random delay, and send to them as
# soon as their own insert returns. That races implicit-to-explicit promotion
# against CQ reads on other endpoints (implicit and explicit AV lookups) and
# against lazy peer creation on the TX path. Between rounds every remote
# address is removed, so each round starts from implicit entries again and
# reuses the AV's freed entries.
#
# 64 workers per side, 128 threads in total.

@pytest.mark.unstable
@pytest.mark.parametrize("threading", ["safe", "completion"])
@pytest.mark.parametrize("message_size", [64, 16384])
def test_av_insert_race(cmdline_args, threading, message_size):
    # Multi-packet messages can complete out of order even with FI_ORDER_SAS,
    # so for those only exactly-once delivery is checked.
    if message_size <= 4096:
        extra = "--msgs 200 --rounds 20"
    else:
        extra = "--msgs 50 --rounds 10 --no-order-check"
    cmd = f"fi_efa_av_insert_race --threading {threading} --threads 64 " \
          f"--peers 5 {extra}"
    test = ClientServerTest(cmdline_args, cmd, message_size=message_size,
                            fabric="efa",
                            additional_env="FI_EFA_ENABLE_SHM_TRANSFER=0")
    test.run()
