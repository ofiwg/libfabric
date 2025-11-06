---
title: CXI_STAT(1) Version 1.5.5 | CXI Diagnostics and Utilities
date: 2024-09-10
---

# NAME

cxi_stat - CXI device status utility


# SYNOPSIS

**cxi_stat** [*OPTIONS*]


# DESCRIPTION

**cxi_stat** is a utility which displays a summary of information provided by
the CXI driver.


# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to report. When unspecified, all devices are
reported.

**-l, \-\-list**
: Output a list of all CXI devices

**-m, \-\-mac-list**
: Output a list of all CXI MAC addresses

**-a, \-\-aer**
: Output PCIe AER statistics

**-r, \-\-rates**
: Output codeword rates and pause percentages

**-p, \-\-pause**=*PAUSE*
: Wait *PAUSE* seconds between counter reads when measuring codeword rates.
Default is 4 seconds.

**-h, \-\-help**
: Display the help text and exit.

**-V, \-\-version**
: Display the program version and exit.


# EXAMPLES
```
$ cxi_stat
Device: cxi0
    Description: Sawtooth
    Part Number: 102251001
    Serial Number: HJ20110019
    FW Version: 1.4.285
    Network device: hsn0
    MAC: 00:40:a6:82:ef:02
    NID: 18 (0x00012)
    PID granule: 256
    PCIE speed/width: 16 GT/s x16
    PCIE slot: 0000:21:00.0
        Link layer retry: on
        Link loopback: off
        Link media: electrical
        Link MTU: 2112
        Link speed: BS_200G
        Link state: up
Device: cxi1
    Description: Sawtooth
    Part Number: 102251001
    Serial Number: HJ20110013
    FW Version: 1.4.285
    Network device: hsn1
    MAC: 00:40:a6:82:ee:f6
    NID: 192246 (0x2eef6)
    PID granule: 256
    PCIE speed/width: 16 GT/s x16
    PCIE slot: 0000:a1:00.0
        Link layer retry: on
        Link loopback: off
        Link media: electrical
        Link MTU: 2112
        Link speed: BS_200G
        Link state: up

$ cxi_stat -l
cxi1
cxi0

$ cxi_stat -m
02:00:00:00:08:c2
02:00:00:00:08:82

$ cxi_stat --device cxi1
Device: cxi1
    Description: Sawtooth
    Part Number: 102251001
    Serial Number: HJ20110013
    FW Version: 1.4.285
    Network device: hsn1
    MAC: 00:40:a6:82:ee:f6
    NID: 192246 (0x2eef6)
    PID granule: 256
    PCIE speed/width: 16 GT/s x16
    PCIE slot: 0000:a1:00.0
        Link layer retry: on
        Link loopback: off
        Link media: electrical
        Link MTU: 2112
        Link speed: BS_200G
        Link state: up

$ cxi_stat -r
Device: cxi0
    Description: SS11 200Gb 2P NIC Mezz
    Part Number: P43012-001
    Serial Number: GR21120003
    FW Version: 1.6.0.326
    Network device: hsn0
    MAC: 02:00:00:00:00:12
    NID: 18 (0x00012)
    PID granule: 256
    PCIE speed/width: 16 GT/s x16
    PCIE slot: 0000:21:00.0
        Link layer retry: on
        Link loopback: off
        Link media: electrical
        Link MTU: 2112
        Link speed: BS_200G
        Link state: up
    Rates:
        Good CW: 39065776.00/s
        Corrected CW: 13.50/s
        Uncorrected CW: 0.00/s
        Corrected BER: 6.353e-11
        Uncorrected BER: <1.176e-12
        TX Pause state: pfc/802.1qbb
        RX Pause state: pfc/802.1qbb
            RX Pause PCP 0: 0.0%
            TX Pause PCP 0: 0.0%
            RX Pause PCP 1: 0.0%
            TX Pause PCP 1: 0.0%
            RX Pause PCP 2: 0.0%
            TX Pause PCP 2: 0.0%
            RX Pause PCP 3: 0.0%
            TX Pause PCP 3: 0.0%
            RX Pause PCP 4: 0.0%
            TX Pause PCP 4: 0.0%
            RX Pause PCP 5: 0.0%
            TX Pause PCP 5: 0.0%
            RX Pause PCP 6: 0.0%
            TX Pause PCP 6: 0.0%
            RX Pause PCP 7: 0.0%
            TX Pause PCP 7: 0.0%
```


# SEE ALSO

**cxi_diags**(7)
