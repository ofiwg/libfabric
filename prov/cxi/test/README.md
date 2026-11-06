*SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only 
Copyright (c) 2020-2026 Hewlett Packard Enterprise Development LP*

# Libfabric CXI Provider Tests

All tests in this directory are built under the Criterion tool. See [https://criterion.readthedocs.io/en/master/index.html](url).

Common setup/teardown routines are found in cxip_test_common.c.

Collections of related tests are found in the other files.

The build produces an executable cxitest, which runs the pre-supplied Criterion main() function, and supports selecting launch of individual tests, or the entire test suite.

## Running Tests

See the test.sh file for examples of launching tests with cxitest.

Run `cxitest --filter 'rma_writedata_source/shared_cq'` to test source
tracking when an FI_SOURCE endpoint binds CQs opened without FI_SOURCE. A
second endpoint shares the CQs and sends a loopback fi_writedata operation;
the test verifies payload, remote CQ data, and the source address returned by
fi_cq_readfrom.

### KI tests

Run on a compute node with the appropriate service and VNI configuration.
Use `-j 1` because the tests share NIC resources. The KI tests are included
in the full test suite run by `test.sh`. You can run them separately via
the following:

```sh
cxitest --verbose -j 1 --filter="ki_*/*"
```

Use the following to run the tests with provider keys for those tests that
support them:

```sh
CXIP_TEST_PROV_KEY=1 cxitest --verbose -j 1 --filter="ki_*/*"
```
