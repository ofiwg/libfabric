# Libfabric CXI Provider Tests

All tests are built under the Criterion tool. See [https://criterion.readthedocs.io/en/master/index.html](url).

Common setup/teardown routines are found in cxip_test_common.c.

Collections of related tests are found in the other files.

The build produces an executable cxitest, which runs the pre-supplied Criterion main() function, and supports selecting launch of individual tests, or the entire test suite.

## Running Tests

See the test.sh file for examples of launching tests with cxitest.

## Running Collective Tests

Collectives are not run as part of the test.sh script, because they require a very specific environment.

First, collective tests can only be run on real hardware (or Z1 emulation) with a minimum of 2 Cassini nodes and one Rosetta.

Second, collective tests must be run under a Work Load Manager (WLM), typically slurm on our test systems, with a minimum of two nodes.

Third, there must be a multicast configuration service running on the System Management Workstation (SMW). This is currently a Python prototype service that uses the Python Flask library to provide a socket-based HTTP server, with a RESTful interface that can accept JSON data via HTTP POST requests, and will generate a multicast address tree in Rosetta and return a multicast address value.

The multicast service can be found in [ssh://git@stash.us.cray.com:7999/sshot/rosetta_ostest.git](url).

You can start the server using:

```
$ cd .../rosetta_ostest/src/tests/rosetta/reductions_scapy/src/configserver
$ ./configserver.py
```
It must be running while you run your tests.

Tests can be started using:

```
cd .../libfabric/prov/cxi/test
srun -N2 -l ./cxitest -j1 --filter "coll_mcast/*"

```

The -j1 is necessary to prevent calls to the multicast service from overlapping.