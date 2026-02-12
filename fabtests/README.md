# fabtests

Fabtests provides a set of examples that uses
[libfabric](http://libfabric.org) -- a high-performance fabric
software library.

## Notes

Note that the fabtests suite is released paired with a specific
version of libfabric.  For example, libfabric v1.4 and fabtests v1.4
were released together.

Using these paired versions is the best way to test a given version of
libfabric.  Using version-mismatched libfabric/fabtests pairs may
produce unexpected results.

## Building fabtests

Distribution tarballs are available from the Github
[releases](https://github.com/ofiwg/libfabric/releases) tab.

If you are building Fabtests from a developer Git clone, you must
first run the `autogen.sh` script. This will invoke the GNU Autotools
to bootstrap Fabtests' configuration and build mechanisms. If you are
building Fabtests from an official distribution tarball, there is no
need to run `autogen.sh`; Fabtests distribution tarballs are already
bootstrapped for you.

Fabtests relies on being able to find an installed version of
Libfabric. In some cases, Libfabric may be in default compiler /
linker search paths, and you don't need to tell Fabtests where to find
it. In other cases, you may need to tell Fabtests where to find the
installed Libfabric's header and library files using the
`--with-libfabric=<directory>` option, described below.

### Configure options

The `configure` script has many built in options (see `./configure
--help`). Some useful options are:

```
--prefix=<directory>
```

By default `make install` will place the files in the `/usr` tree.
The `--prefix` option specifies that the Fabtests files should be
installed into the tree specified by named `<directory>`. The
executables will be located at `<directory>/bin`.

```
--with-libfabric=<directory>
```

Specify the directory where the Libfabric library and header files are
located.  This is necessary if Libfabric was installed in a location
where the compiler and linker will not search by default.  The
Libfabric library will be searched for in `<directory>/lib`, and
headers will be searched for in `<directory>/include`.

```
--with-valgrind=<directory>
```

Directory where valgrind is installed.  If valgrind is found, then
valgrind annotations are enabled. This may incur a performance
penalty.

```
--with-cuda[=DIR]
```

Provide path to where the CUDA development and runtime libraries are installed. This enables CUDA memory support for heterogeneous memory (HMEM) testing.

```
--with-rocr[=DIR]
```

Provide path to where the ROCR development and runtime libraries are installed. This enables ROCr memory support for heterogeneous memory (HMEM) testing.

```
--with-neuron[=DIR]
```

Provide path to where the Neuron development and runtime libraries are installed. This enables Neuron memory support for heterogeneous memory (HMEM) testing.

```
--with-synapseai[=DIR]
```

Enable SynapseAI build and fail if not found. Optional=<Path to where the SynapseAI libraries and headers are installed.> This enables SynapseAI memory support for heterogeneous memory (HMEM) testing.

```
--with-ze[=DIR]
```

Enable Level-Zero (Ze) support and fail if not found. Optional=<Path to where the Level-Zero libraries and headers are installed.> This enables Intel GPU memory support for heterogeneous memory (HMEM) testing.

### Examples

Consider the following example:

```
$ ./configure --with-libfabric=/opt/libfabric --prefix=/opt/fabtests && make -j 32 && sudo make install
```

This will tell the Fabtests to look for Libfabric libraries in the
`/opt/libfabric` tree, and to install the Fabtests in the
`/opt/fabtests` tree.

Alternatively:

```
$ ./configure --prefix=/opt/fabtests && make -j 32 && sudo make install
```

Tells the Fabtests that it should be able to find the Libfabric header
files and libraries in default compiler / linker search paths
(configure will abort if it is not able to find them), and to install
Fabtests in `/opt/fabtests`.

## Installation Location

After running `make install`, the fabtests binaries will be installed to:
- `<prefix>/bin/` - All test executables and scripts (e.g., `fi_rdm_pingpong`, `fi_msg_bw`, `runfabtests.sh`, `runfabtests.py`)
- `<prefix>/share/fabtests/` - Test configuration files and utilities

For detailed documentation of individual test binaries and their options, see [fabtests/man/fabtests.7.md](man/fabtests.7.md) which contains comprehensive man pages for each test binary.

## Running Fabtests

Fabtests can be run in three different ways. **Note**: Ensure `<prefix>/bin` is in your `$PATH` or use absolute paths to the binaries and scripts.

### 1. Direct Binary Execution

Run individual test binaries directly. Each binary supports `-h` for detailed options:
```bash
# Server side (starts server and waits for client)
$ fi_rdm_pingpong -p <provider_name>

# Client side (from another terminal/node)
$ fi_rdm_pingpong -p <provider_name> <server_ip>

# With specific options (-S: transfer size in bytes, -I: number of iterations)
$ fi_rdm_pingpong -p <provider_name> -S 1024 -I 1000

# For providers requiring out-of-band address exchange (e.g., efa)
$ fi_rdm_pingpong -p <provider_name> -E
$ fi_rdm_pingpong -p <provider_name> -E <server_ip>
```

Example provider names (available in libfabric/prov/): `tcp`, `shm`, `efa`, `verbs`, `psm3`, `opx`, `cxi`, `ucx`, etc.

Common test binaries include:
- `fi_msg_pingpong` - MSG endpoint ping-pong latency test
- `fi_rdm_pingpong` - RDM endpoint ping-pong test  
- `fi_msg_bw` - MSG endpoint bandwidth measurement test
- `fi_rdm_tagged_pingpong` - RDM endpoint tagged message ping-pong test
- `fi_rma_pingpong` - RMA ping-pong test
- `fi_rma_bw` - RMA bandwidth test

### 2. Bash Script (runfabtests.sh)

Use the comprehensive test script for automated testing:
```bash
# Run quick test suite with sockets provider in loopback
$ ./runfabtests.sh

# Run with specific provider
$ ./runfabtests.sh tcp

# Run tests between two nodes
$ ./runfabtests.sh tcp <server_ip> <client_ip>

# For providers requiring out-of-band address exchange (e.g., efa)
$ ./runfabtests.sh -b efa <server_ip> <client_ip>

# Run specific test sets
$ ./runfabtests.sh -t standard tcp
$ ./runfabtests.sh -t "quick,functional" tcp

# Exclude specific tests
$ ./runfabtests.sh -e "dgram,rma.*write" tcp

# Verbose output for debugging
$ ./runfabtests.sh -vvv tcp
```

Available test sets: `all`, `quick`, `unit`, `functional`, `standard`, `short`, `complex`, `threaded`

### 3. Python Script (runfabtests.py)

Use the Python-based test runner built on the pytest framework for advanced testing configurations. Use `-h` for detailed help:

**Prerequisites**: Install Python dependencies first:
```bash
$ pip install -r fabtests/pytest/requirements.txt
```

**Usage**:
```bash
# Run quick test suite
$ python runfabtests.py <provider_name> <server_ip> <client_ip>

# For providers requiring out-of-band address exchange (e.g., efa)
$ python runfabtests.py -b <provider_name> <server_ip> <client_ip>

# Run specific test sets
$ python runfabtests.py -t standard <provider_name> <server_ip> <client_ip>
$ python runfabtests.py -t "quick,functional" <provider_name> <server_ip> <client_ip>

# Test with CUDA memory (HMEM)
$ python runfabtests.py -t cuda_memory <provider_name> <server_ip> <client_ip>

# Test with Neuron memory
$ python runfabtests.py -t neuron_memory <provider_name> <server_ip> <client_ip>

# Generate HTML and JUnit XML reports
$ python runfabtests.py --html=report.html --junit-xml=results.xml <provider_name> <server_ip> <client_ip>

# Run with multiple parallel workers
$ python runfabtests.py --nworkers=4 <provider_name> <server_ip> <client_ip>

# Filter tests by expression
$ python runfabtests.py --expression="pingpong" <provider_name> <server_ip> <client_ip>
```

The Python script is well tested with `tcp`, `shm`, and `efa` providers. It includes:
- **Common test items**: Defined in `fabtests/pytest/` directory, applied to any provider
- **Provider-specific tests**: Located in `fabtests/pytest/<provider_name>/` directories
  - Currently implemented for `shm` and `efa` providers
  - We welcome more providers to join this framework!

The Python script leverages the pytest framework, providing advanced testing configurations including:
- Support for heterogeneous memory (HMEM) types like CUDA and Neuron
- Parallel test execution with configurable worker count
- HTML and JUnit XML report generation
- Advanced test filtering and exclusion capabilities
- Comprehensive logging and verbosity controls

Both script methods provide comprehensive testing across multiple providers and configurations, while direct binary execution allows for focused testing of specific scenarios.
