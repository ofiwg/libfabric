---
title: CXI_DUMP_CSRS(1) Version 1.0.0 | CXI Diagnostics and Utilities
date: 2022-10-06
---

# NAME

cxi_dump_csrs - Dump Cassini registers into a file


# SYNOPSIS

| **cxi_dump_csrs** [**-d** *DEV*]


# DESCRIPTION

**cxi_dump_csrs** will dump CSRs into a file for further analysis. It
will contain most data of a running board, including command queues
states or counters.

The name of the file generated contains the name of the Cassini
device, and the date, in Unix format, it was generated. For instance:

    cxi0-csrs-1665092478.bin

The content of that file can then be analyzed with `cxiutil`, from the
pycxi package:

    $ cxiutil dump csr sts_rev --file=cxi0-csrs-1665092478.bin
          c_mb_sts_rev    hex       dec.
          --------------  ------  ------
          rev             0x1          1
          device_id       0x501     1281
          vendor_id       0x17db    6107
          proto           0x0          0
          platform        0x0          0

# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to use. When unspecified, \"cxi0\" is used.
