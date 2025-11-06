---
title: CXI_HEATSINK_CHECK(1) Version 2.2.2 | CXI Diagnostics and Utilities
date: 2025-08-05
---

# NAME

cxi_heatsink_check - Cassini NIC thermal stress test


# SYNOPSIS

| **cxi_heatsink_check** [*OPTIONS*]


# DESCRIPTION

**cxi_heatsink_check** is a diagnostic that monitors ASIC temperatures and
power consumption while stressing the chip with a large amount of small RMDA
writes to maximize both bandwidth and packet rate. Users have the option to
allocate source and destination buffers from system memory or from an available
GPU device via calls to the HIP, CUDA, or INTEL Level-zero APIs.

## How to run

To test a single-NIC card, simply run the diagnostic, specifying the NIC device
with **\-\-device**. When testing a dual-NIC card, two copies of the diagnostic
must be run concurrently, one for each NIC. In some instances this could mean
running the diagnostic on two separate nodes. The pass/fail criteria leaves some
room for delay between the start times of each copy.

Running **cxi_heatsink_check** requires root privileges.

## Output

Once running, the diagnostic performs a one second warmup. Then it prints
several columns of data, once every **\-\-interval** seconds.

*Time[s]*
: The elapsed time, not including the warmup period.

*Rate[GB/s]*
: The observed bandwidth of the last measurement interval.

*VDD[W]*
: The VDD power sensor reading at the end of the last measurement interval.

*AVDD[W]*
: The AVDD power sensor reading at the end of the last measurement interval.

*TRVDD[W]*
: The TRVDD power sensor reading at the end of the last measurement interval
(Cassini 2 cards only).

*QSFP[W]*
: The QSFP power sensor reading at the end of the last measurement interval
(Cassini 1 single-NIC cards only).

*ASIC_0[°C]*
: The NIC 0 ASIC temperature sensor reading at the end of the last measurement
interval.

*ASIC_1[°C]*
: The NIC 1 ASIC temperature sensor reading at the end of the last measurement
interval (dual-NIC cards only).

*QSFP_VR[°C]*
: The QSFP VR temperature sensor reading at the end of the last measurement
interval (Cassini 1 single-NIC cards only).

*QSFP_INT[°C]*
: The QSFP AOC internal temperature sensor reading at the end of the last
measurement interval (single-NIC cards only).

When the diagnostic completes, it prints a summary of the pass/fail criteria.


# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to use. When unspecified, \"cxi0\" is used.

**-v, \-\-svc-id**=*SVC_ID*
: The service ID to use. This option will be used when running as a
non-privileged user. When unspecified the default service ID is used.

**-t, \-\-tx-gpu**
: The GPU to use for TX buffer allocation. By default system memory is used.

**-r, \-\-rx-gpu**
: The GPU to use for RX buffer allocation. By default system memory is used.

**-g, \-\-gpu-type**
: The GPU type to use with this test. Valid values are AMD, NVIDIA, or INTEL. By
default the type is determined by discovered GPU files on the system.

**-P, \-\-procs**=*PROCS*
: The number of processes created to generate RDMA writes. By default this is
1/4 of the available processors. This argument is ignored when **\-\-cpu-list**
is supplied.

**-D, \-\-duration**=*DURATION*
: The number of seconds to monitor sensors and generate stress. This excludes
an additional one second of warmup time. By default the diagnostic runs for 600
seconds.

**-c, \-\-cpu-list**=*LIST*
: The processors to use when pinning each write-generating process. Each process
is pinned to one processor from this list in the order given. This option
overrides any value supplied with **\-\-procs**. By default this list is
determined by spreading each socket's processors evenly among all CXI devices
whose NUMA nodes belong to the socket.

**-i, \-\-interval**=*INTERVAL*
: Sensor readings and bandwidth are printed every *INTERVAL* seconds. By default
the interval is 10 seconds.

**-s, \-\-size**=*SIZE*
: The size of the writes performed to generate stress. Valid sizes are 1 through
262144 bytes. By default the write size is 512 bytes.

**-l, \-\-list-size**=*LIST_SIZE*
: The number of write commands added to the TX command queue before ringing the
queue's doorbell to initiate the writes. By default the list size is 256.

**\-\-no-hrp**
: Disable the use of High Rate Puts. When this option is specified, matching
puts are used, and additional unmatching list entries are appended to increase
the workload of the matchers. By default high rate puts are used for sizes up to
2048 bytes.

**\-\-no-idc**
: Disable the use of Immediate Data Commands. By default IDCs are used for sizes
up to 224 bytes for high rate puts or 192 bytes for normal puts. IDCs are
disabled when TX GPU buffers are used.

**\-\-no-ple**
: Disable the use of Persistent List Entries. When this options is specified,
a single use-once list entry is appended for every write. This option should
not be combined with large values for *PROCS* and *LIST_SIZE* as this may
exhaust the available list entries and cause the diagnostic to fail.

**-h, \-\-help**
: Display the help text and exit.

**-V, \-\-version**
: Display the program version and exit.


# EXAMPLES

## Example 1

Testing a dual-NIC card (only one NIC's output shown here).

```
# cxi_heatsink_check
------------------------------------------------------------
    CXI Heatsink Test
Device          : cxi0
TX Mem          : System
RX Mem          : System
Duration        : 600 seconds
Sensor Interval : 10 seconds
TX/RX Processes : 32
Processor List  : 1-31,0
RDMA Write Size : 512
List Size       : 250
HRP             : Enabled
IDC             : Enabled
Persistent LEs  : Enabled
Local Address   : NIC 0x13
Board Type      : Dual-NIC
------------------------------------------------------------
Time[s]  Rate[GB/s]  VDD[W]  AVDD[W]  ASIC_0[°C]  ASIC_1[°C]
     10       21.68      16        6          59          58
     20       21.99      18        6          60          60
     30       21.99      18        6          60          60
     40       21.99      18        6          60          60
< some output omitted >
    560       21.95      18        6          69          69
    570       21.95      18        6          69          69
    580       21.95      18        6          69          69
    590       21.95      18        6          69          69
------------------------------------------------------------
Cassini 0 Temperature (ASIC_0) under 85 °C:  69 °C       PASS
Cassini 1 Temperature (ASIC_1) under 85 °C:  69 °C       PASS
0.85V S0 Power (VDD):                        18 W        ----
0.9V S0 Power (AVDD):                        6 W         ----
Average BW over 19 GB/s:                     21.96 GB/s  PASS
```

## Example 2

Testing a Single-NIC card.

```
# cxi_heatsink_check
------------------------------------------------------------------------------------
    CXI Heatsink Test
Device          : cxi0
TX Mem          : System
RX Mem          : System
Duration        : 600 seconds
Sensor Interval : 10 seconds
TX/RX Processes : 16
Processor List  : 1-15,0
RDMA Write Size : 512
List Size       : 250
HRP             : Enabled
IDC             : Enabled
Persistent LEs  : Enabled
Local Address   : NIC 0xe
Board Type      : Single-NIC
------------------------------------------------------------------------------------
Time[s]  Rate[GB/s]  VDD[W]  AVDD[W]  QSFP[W]  ASIC_0[°C]  QSFP_VR[°C]  QSFP_INT[°C]
     10       20.19      10        3        0          53           48             -
     20       20.45      10        3        0          53           48             -
     30       20.49      10        3        0          53           48             -
     40       20.50      10        3        0          54           48             -
< some output omitted >
    560       20.39      10        3        0          63           54             -
    570       20.40      10        3        0          63           54             -
    580       20.40      10        3        0          64           54             -
    590       20.40      10        3        0          63           54             -
------------------------------------------------------------------------------------
Cassini 0 Temperature (ASIC_0) under 85 °C:        64 °C       PASS
3.3V QSFP VR Temperature (QSFP_VR) under 125 °C:   54 °C       PASS
QSFP Internal Temperature (QSFP_INT) under 70 °C:  -             NA
0.85V S0 Power (VDD):                              10 W        ----
0.9V S0 Power (AVDD):                              3 W         ----
Average BW over 19 GB/s:                           20.43 GB/s  PASS
```


# SEE ALSO

**cxi_diags**(7)
