def shm_run_client_server_test(cmdline_args, executable, iteration_type,
                               completion_semantic, memory_type, message_size=None,
                               warmup_iteration_type=None, timeout=None,
                               completion_type="queue", additional_env='',
                               hmem_device_offset=0):
    from common import ClientServerTest

    test = ClientServerTest(cmdline_args, executable, iteration_type,
                            completion_semantic=completion_semantic,
                            datacheck_type="with_datacheck",
                            memory_type=memory_type,
                            message_size=message_size,
                            warmup_iteration_type=warmup_iteration_type,
                            completion_type=completion_type, timeout=timeout,
                            additional_env=additional_env,
                            hmem_device_offset=hmem_device_offset)
    test.run()