=========================
Debugging the CXI drivers
=========================


debugfs
-------

The drivers exposes some information in /sys/kernel/debug. If that
directory is empty, then mount debugfs:

    mount -t debugfs none /sys/kernel/debug

The cxi exposes its files in /sys/kernel/debug/cxi/cxiX, where cxiX is
the adapter (cxi0, cxi1, ...).

.../port
~~~~~~~~

It contains the physical link current information.

.../telemetry and .../dmac-desc-sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Telemetry information.

.../tc_cfg
~~~~~~~~~~

Information about configured traffic classes.

.../services
~~~~~~~~~~~~

Information about the allocated resources

.../uc_log
~~~~~~~~~~

This outputs (and consumes) log messages from the uC.


Dynamic printing
----------------

Dynamic debugging can be enabled if it's configured in the kernel. For
instance:

    echo -n 'module cxi_ss1 +p' > /sys/kernel/debug/dynamic_debug/control
    echo -n 'module cxi_eth +p' > /sys/kernel/debug/dynamic_debug/control

See the kernel documentation in
Documentation/admin-guide/dynamic-debug-howto.rst for more
information.


tracing
-------

The driver can also be traced with the link tracing subsystem. If the /sys/kernel/tracing directory is empty, mount it with:

    mount -t tracefs nodev /sys/kernel/tracing

Example of usage:

    echo function_graph > current_tracer
    echo '*:mod:cxi_ss1' >> set_ftrace_filter
    echo '!cass_read*' >> set_ftrace_filter
    echo '!cass_lmon*' >> set_ftrace_filter

	echo 1 > tracing_on
	.....
	echo 0 > tracing_on

    cat trace

See the extensive kernel documentation in Documentation/trace/ for
more options.


bpftrace
--------

The eBPF can be used to trace some functions. The `bpftrace` package
must be installed. See `scripts/csr_access.bt` and
`scripts/eq_alloc.bt` for examples.

See the bpftrace tutorial, reference and examples at
https://bpftrace.org/ for more information.
