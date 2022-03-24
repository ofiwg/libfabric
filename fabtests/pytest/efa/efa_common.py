

def efa_run_client_server_test(cmdline_args, executable, iteration_type,
                               completion_type, memory_type, message_size):
    from common import ClientServerTest
    # It is observed that cuda tests requires larger time-out limit (~240 secs) to test all
    # message sizes for libfabric's debug and mem-poisoning builds, on p4d instances.
    timeout = None
    if "cuda" in memory_type and message_size == "all":
        timeout = 240

    test = ClientServerTest(cmdline_args, executable, iteration_type,
                            completion_type=completion_type,
                            datacheck_type="with_datacheck",
                            message_size=message_size,
                            memory_type=memory_type,
                            timeout=timeout)
    test.run()

