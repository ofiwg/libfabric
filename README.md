[<img alt="libfabric master branch Travis CI status" src="https://travis-ci.org/ofiwg/libfabric.svg?branch=master"/>](https://travis-ci.org/ofiwg/libfabric)
[<img alt="libfabric Coverity scan suild status" src="https://scan.coverity.com/projects/4274/badge.svg"/>](https://scan.coverity.com/projects/4274)
[![libfabric release version](https://img.shields.io/github/release/ofiwg/libfabric.svg)](https://github.com/ofiwg/libfabric/releases/latest)

# libfabric

The Open Fabrics Interfaces (OFI) is a framework focused on exporting fabric
communication services to applications.

See [the OFI web site](http://libfabric.org) for more details, including a
description and overview of the project, and detailed documentation of the
Libfabric APIs.

## Installing pre-built Libfabric packages

On OS X, the latest release of Libfabric can be installed using the
[Homebrew](https://github.com/Homebrew/homebrew) package manager using the
following command:

```
$ brew install libfabric
```

Libfabric pre-built binaries may be available from other sources, such as Linux
distributions.

## Building and installing Libfabric from source

Distribution tarballs are available from the Github
[releases](https://github.com/ofiwg/libfabric/releases) tab.

If you are building Libfabric from a developer Git clone, you must first run
the `autogen.sh` script. This will invoke the GNU Autotools to bootstrap
Libfabric's configuration and build mechanisms. If you are building Libfabric
from an official distribution tarball, there is no need to run `autogen.sh`;
Libfabric distribution tarballs are already bootstrapped for you.

Libfabric currently supports GNU/Linux, Free BSD, and OS X.

### Configure options

The `configure` script has many built in options (see `./configure --help`).
Some useful options are:

```
--prefix=<directory>
```

By default `make install` will place the files in the `/usr` tree.
The `--prefix` option specifies that Libfabric files should be installed into
the tree specified by named `<directory>`. The executables will be located at
`<directory>/bin`.

```
--with-valgrind=<directory>
```

Directory where valgrind is installed. If valgrind is found, then valgrind
annotations are enabled. This may incur a performance penalty.

```
--enable-debug
```

Enable debug code paths. This enables various extra checks and allows for using
the highest verbosity logging output that is normally compiled out in
production builds.

```
--enable-<provider>=[yes|no|auto|dl|<directory>]
--disable-<provider>
```

This enables or disables the provider named `<provider>`. Valid options are:
- auto (This is the default if the `--enable-<provider>` option isn't specified)

  The provider will be enabled if all of its requirements are satisfied. If one
  of the requirements cannot be satisfied, then the provider is disabled.
- yes (This is the default if the `--enable-<provider>` option is specified)

  The configure script will abort if the provider cannot be enabled (e.g., due
  to some of its requirements not being available.
- no

  Disable the provider. This is synonymous with `--disable-<provider>`.
- dl

  Enable the provider and build it as a loadable library.
- \<directory\>

  Enable the provider and use the installation given in `<directory>`.

### Examples

Consider the following example:

```
$ ./configure --prefix=/opt/libfabric --disable-sockets && make -j 32 && sudo make install
```
This will tell Libfabric to disable the `sockets` provider, and install
Libfabric in the `/opt/libfabric` tree. All other providers will be enabled if
possible and all debug features will be disabled.

Alternatively:

```
$ ./configure --prefix=/opt/libfabric --enable-debug --enable-psm=dl && make -j 32 && sudo make install
```

This will tell Libfabric to enable the `psm` provider as a loadable library,
enable all debug code paths, and install Libfabric to the `/opt/libfabric`
tree. All other providers will be enabled if possible.


## Validate installation

The fi_info utility can be used to validate the libfabric and provider
installation, as well as provide details about provider support and available
interfaces.  See `fi_info(1)` man page for details on using the fi_info
utility.  fi_info is installed as part of the libfabric package.

A more comprehensive test package is available via the fabtests package.


## Providers

### gni

***

The `gni` provider runs on Cray XC (TM) systems utilizing the user-space
Generic Network Interface (`uGNI`) which provides low-level access to
the Aries interconnect.  The Aries interconnect is designed for
low-latency one-sided messaging and also includes direct hardware
support for common atomic operations and optimized collectives.

See the `fi_gni(7)` man page for more details.

#### Dependencies

- The `gni` provider requires `gcc` version 4.9 or higher.

### mxm

***

The MXM provider has been deprecated and was removed after the 1.4.0 release.

### psm

***

The `psm` provider runs over the PSM 1.x interface that is currently supported
by the Intel TrueScale Fabric. PSM provides tag-matching message queue
functions that are optimized for MPI implementations.  PSM also has limited
Active Message support, which is not officially published but is quite stable
and well documented in the source code (part of the OFED release). The `psm`
provider makes use of both the tag-matching message queue functions and the
Active Message functions to support a variety of Libfabric data transfer APIs,
including tagged message queue, message queue, RMA, and atomic
operations.

The `psm` provider can work with the `psm2-compat` library, which exposes
a PSM 1.x interface over the Intel Omni-Path Fabric.

See the `fi_psm(7)` man page for more details.

### psm2

***

The `psm2` provider runs over the PSM 2.x interface that is supported
by the Intel Omni-Path Fabric. PSM 2.x has all the PSM 1.x features plus a set
of new functions with enhanced capabilities. Since PSM 1.x and PSM 2.x are not
ABI compatible, the `psm2` provider only works with PSM 2.x and doesn't support
Intel TrueScale Fabric.

See the `fi_psm2(7)` man page for more details.

### rxm

The `ofi_rxm` provider is an utility provider that supports RDM endpoints emulated
over MSG endpoints of a core provider.

See [`fi_rxm`(7)](fi_rxm.7.html) for more information.

### sockets

***

The `sockets` provider is a general purpose provider that can be used on any
system that supports TCP sockets.  The provider is not intended to provide
performance improvements over regular TCP sockets, but rather to allow
developers to write, test, and debug application code even on platforms
that do not have high-performance fabric hardware.  The sockets provider
supports all Libfabric provider requirements and interfaces.

See the `fi_sockets(7)` man page for more details.


### udp

***

The `udp` provider is a basic provider that can be used on any system that
supports UDP sockets.  The provider is not intended to provide performance
improvements over regular UDP sockets, but rather to allow application and
provider developers to write, test, and debug their code.  The `udp` provider
forms the foundation of a utility provider that enables the implementation of
Libfabric features over any hardware.

See the `fi_udp(7)` man page for more details.

### usnic

***

The `usnic` provider is designed to run over the Cisco VIC (virtualized NIC)
hardware on Cisco UCS servers. It utilizes the Cisco usnic (userspace NIC)
capabilities of the VIC to enable ultra low latency and other offload
capabilities on Ethernet networks.

See the `fi_usnic(7)` man page for more details.

#### Dependencies

- The `usnic` provider depends on library files from either `libnl` version 1
  (sometimes known as `libnl` or `libnl1`) or version 3 (sometimes known as
  `libnl3`). If you are compiling Libfabric from source and want to enable
  usNIC support, you will also need the matching `libnl` header files (e.g.,
  if you are building with `libnl` version 3, you need both the header and
  library files from version 3).

#### Configure options

```
--with-libnl=<directory>
```

If specified, look for libnl support. If it is not found then the `usnic`
provider will not be built. If `<directory>` is specified, then check in the
directory and check for `libnl` version 3. If version 3 is not found, then
check for version 1. If no `<directory>` argument is specified, then this
option is redundant with `--with-usnic`.

### verbs

***

The verbs provider enables applications using OFI to be run over any verbs
hardware (Infiniband, iWarp, etc). It uses the Linux Verbs API for network
transport and provides a translation of OFI calls to appropriate verbs API calls.
It uses librdmacm for communication management and libibverbs for other control
and data transfer operations.

See the `fi_verbs(7)` man page for more details.

#### Dependencies

- The verbs provider requires libibverbs (v1.1.8 or newer) and librdmacm (v1.0.16
  or newer). If you are compiling Libfabric from source and want to enable verbs
  support, you will also need the matching header files for the above two libraries.
  If the libraries and header files are not in default paths, specify them in CFLAGS,
  LDFLAGS and LD_LIBRARY_PATH environment variables.

### bgq

***

The `bgq` provider is a native provider that directly utilizes the hardware
interfaces of the Blue Gene/Q system to implement aspects of the libfabric
interface to fully support MPICH3 CH4.

See the `fi_bgq(7)` man page for more details

#### Dependencies

- The `bgq` provider depends on the system programming interfaces (SPI) and
  the hardware interfaces (HWI) located in the Blue Gene/Q driver installation.
  Additionally, the open source Blue Gene/Q system files are required.

#### Configure options

```
--with-bgq-progress=(auto|manual)
```

If specified, set the progress mode enabled in FABRIC_DIRECT (default is FI_PROGRESS_MANUAL).

```
--with-bgq-mr=(basic|scalable)
```

If specified, set the memory registration mode (default is FI_MR_BASIC).

### Network Direct

***

The Network Direct provider enables applications using OFI to be run over
any verbs hardware (Infiniband, iWarp and etc). It uses the Microsoft Network
Direct SPI for network transport and provides a translation of OFI calls to
appropriate Network Direct API calls.
The Network Direct providers allows to OFI-based applications utilize
zero-copy data transfers between applications, kernel-bypass I/O generation and
one-sided data transfer operations on Microsoft Windows OS.
An application is able to use OFI with Network Direct provider enabled on
Windows OS to expose the capabilities of the networking devices if the hardware
vendors of the devices implemented the Network Direct service provider interface
(SPI) for their hardware.

The Network Direct is an experimental provider. The full support of the Network
Direct provider will be added to 1.6 release version of libfabric.

See the `fi_netdir(7)` man page for more details.

#### Dependencies

- The Network Direct provider requires Network Direct SPI. If you are compiling
  Libfabric from source and want to enable Network Direct support, you will also
  need the matching header files for the Network Direct SPI.
  If the libraries and header files are not in default paths (the default path is
  root of provier directory, i.e. \prov\netdir\NetDirect, where NetDirect contains
  the header files), specify them in the configuration properties of the VS project.

### mlx

***

The MLX provider enables applications using OFI to be run over UCX
communication library. It uses libucp for connections control and data transfer operations.
Supported UCP API version: 1.2

See the `fi_mlx(7)` man page for more details.

#### Dependencies

- The MLX provider requires UCP API 1.2 capable libucp and libucs (tested with hpcx v1.8.0, v1.9.7).
  If you are compiling Libfabric from source and want to enable MLX
  support, you will also need the matching header files for UCX.
  If the libraries and header files are not in default paths, specify them using:

```
--with-mlx=<path to local UCX installation>
```

### shm

***

The shm provider enables applications using OFI to be run over shared memory.

See the `fi_shm(7)` man page for more details.

#### Dependencies

- The shared memory provider only works on Linux platforms and makes use of
  kernel support for 'cross-memory attach' (CMA) data copies for large
  transfers.