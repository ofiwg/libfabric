# Libfabric CXI Accelerated Collectives


# Introduction

The libfabric CXI provider supports HPC hardware-accelerated collectives through the libfabric fi_collectives API. This is a subset of collectives, with some extensions.

The libfabric CXI provider does not support non-accelerated collectives.

This document describes the specific features of this API.

# Overview

The accelerated collectives feature uses special multicast trees within the Slingshot fabric to accelerate collective operations.

These multicast trees utilize the fabric switches as reduce-and-hold storage for partial reductions that traverse the multicast tree from leaf endpoints to the root endpoint. Reduction is performed in-tree during the data transfer, and the count of the contributions is maintained.

Leaf endpoints compute their contribution to the collective operation and immediately send it upstream (rootward) on the multicast tree, where it is counted/reduced in switch hardware. The root endpoint computes its contribution to the collective, and then waits for a full count of leaf contributions. When the root endpoint receives the full count of N-1 contributions from the leaves, it completes the reduction in software with its own contribution, and broadcasts the result through the multicast tree downstream (leafward) to all the leaf endpoints.

When there are no complications, each endpoint sends and receives exactly one packet, and only one reduction is performed in software on the root endpoint.

This avoids the delays associated with passing through NICs in a traditional radix-tree implementation of collectives.

The benefit is that these "accelerated" collectives show much better scaling performance as the number of endpoints increases.

# Requirements

- HPE CSM/HPCM environment
	- Cassini NICs
	- Rosetta fabric switches
	- Shasta Fabric Manager REST API (FM API)
	- Supported Workload Manager (WLM)
	- libfabric/cxi libraries
	- libcurl.so library
	- libjson-c.so library

Note: *The libcurl.so and libjson-c.so libraries must be present, but will be dynamically loaded into the collective application at runtime the first time libcurl and libjson routines are needed. Specifically, libcurl.so and libjson-c.so must be present on any endpoint that serves as rank 0 for any call to fi_join_collective(). If they are not present, the join will fail.*

# Basic Application Overview

1. The user application must be started on multiple compute nodes by an appropriate Workload Manager, such as SLURM or PBS/PALS, which is adapted to support accelerated collective requirements. The WLM must:

	- Gain secure access to the fabric manager (HTTPS) prior to job start
	- Generate environment variables needed by the libfabric library
	- Gain secure access to the fabric manager (HTTPS) upon job completion


1. User applications must enable collectives for all CXI endpoints (NICs) to be used in a collective using the `FI_COLLECTIVE` flag when the endpoint is enabled.

1. User applications must create one or more collective groups using `fi_join_collective()`, which will return an mc_obj pointer to each endpoint that identifies the collective group.

1. User applications can now use `fi_barrier()`, `fi_broadcast()`, `fi_reduce()`, and `fi_allreduce()` on these joined collective groups.

1. Upon completion of use, the application should call `fi_close()` on the mc_obj for each collective group. Note that simply exiting the application (cleanly or with an abort) will perform preemptive cleanup of all mc_obj objects.

# Collective Functions

## Collective Join

"Joining" a collective is the process by which a collective group is created. Each endpoint in the collective group must "join" the collective before it can participate in the collective. The join operation itself is a collective, and no endpoint can proceed from the join until all endpoints in that group have joined.

**Note**: *libfabric endpoints in the CXI provider represent NICs, and each NIC can be individually joined to the collective. MPI applications use the term RANK to represent compute processes, and these typically outnumber endpoints. These RANKS must be locally reduced before submitting the partial results to the fabric endpoint.*

The following system-wide considerations apply to joining collectives:

1. Only endpoints included within a WLM JOB can be joined to a collective.
1. Collective groups may overlap (i.e. an endpoint can belong to multiple collective groups).
1. The number of collective groups in a job is limited (see `FI_CXI_HWCOLL_ADDRS_PER_JOB`).
1. Any endpoint can serve as HWRoot for _at most_ one collective group.

### `fi_av`

Any libfabric application requires an `fi_av` structure to convert endpoint hardware addresses to libfabric addresses. There can be multiple `fi_av` structures used for different purposes. It is also common to have a single `fi_av` structure representing all endpoints in a job. This follows the standard libfabric documentation.


### `fi_av_set`

Joining a collective requires an `fi_av_set` structure that defines the endpoints to be included in the collective group, which in turn requires an `fi_av` structure that defines all endpoints to be used in that set. This follows the standard libfabric documentation.

```
int fi_av_set(cxit_av, &attr, &setp, ctx);
```
- `cxit_av` is the `fi_av` structure for this job
- `attr` is a custom attribute (`comm_key`) structure for the endpoints
- `setp` is the `fid_av_set` pointer for the result
- `ctx` is an optional pointer associated with this operation to allow `av_set` creation concurrency, and can be NULL

The only cxi-unique feature for this operation is the `struct cxip_comm_key`. This appears in the `attr` structure, and should be initialized to zero.

```
	// clear comm_key structure
	memset(&comm_key, 0, sizeof(comm_key);

	// attributes to create empty av_set
	struct fi_av_set_attr attr = {
		.count = 0,
		.start_addr = FI_ADDR_NOTAVAIL,
		.end_addr = FI_ADDR_NOTAVAIL,
		.stride = 1,
		.comm_key_size = sizeof(comm_key),
		.comm_key = (void *)&comm_key,
		.flags = 0,
	};

	// create empty av_set
	ret = fi_av_set(cxit_av, &attr, &setp, NULL);
	if (ret) {
		fprintf(stderr, "fi_av_set failed %d\n", ret);
		goto quit;
	}

	// append count addresses to av_set
	for (i = 0; i < count; i++) {
		ret = fi_av_set_insert(setp, fiaddrs[i]);
		if (ret) {
			fprintf(stderr, "fi_av_set_insert failed %d\n", ret);
			goto quit;
		}
	}

```

Note: *The `fi_av_set` endpoints within the structure must be identical and must appear in the same order on all endpoints. If the content or ordering differs, results are undefined.*

### `fi_join_collective()`

Once the `fi_av_set` structure is created, `fi_join_collective()` can be called to create the collective mc_obj that represents the multicast tree.

```
int fi_join_collective(ep, FI_ADDR_NOTAVAIL, avset, 0L, &mc_obj, ctx);
```
- `ep` is the endpoint on which the function is called
- `FI_ADDR_NOTAVAIL` is a mandatory placeholder
- `avset` is the fi_av_set created above
- `flags` are not supported
- `mc_obj` is the return multicast object pointer
- `ctx` is an arbitrary pointer associated with this operation to allow concurrency, and can be NULL

Note: `fi_join_collective()` must be called on all endpoints in the collective with identical av_set structure, or results are undefined.

The join operation is asynchronous, and the application must poll the EQ (Event Queue) to progress the operation and to obtain the result. Joins are non-concurrent and return `FI_EAGAIN` until an active join completes.

Note: Internal resource constraints may cause `fi_join_collective()` to return `-FI_EAGAIN`, and the operation should be retried after polling the EQ at least once to progress the running join operations.

## Collective Operations

All collective operations are asynchronous and must be progressed by polling the CQ (Completion Queue).

Only eight concurrent reductions can be performed on a given multicast tree. Attempts to exceed this limit will result in the `-FI_EAGAIN` error, and the operation should be retried after polling the CQ at least once.

All collective operations below are syntactic variants based on `fi_allreduce()`, which is the only operation supported by accelerated collectives.


### Barrier

```
ssize_t fi_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context)
```
- `ep` is the endpoint on which the function is called
- `coll_addr` is the typecast mc_obj for the collective group
- `context` is a user context pointer

This operation initiates a barrier operation and returns immediately. The user must poll the CQ for a successful completion.

It is implemented as an allreduce with no data.

### Broadcast

```
ssize_t fi_broadcast(struct fid_ep *ep, void *buf, size_t count,
		       void *desc, fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, uint64_t flags,
		       void *context)
```
- `ep` is the endpoint on which the function is called
- `buf` is the buffer to be sent/received
- `count` is the data count
- `desc` is ignored
- `coll_addr` is the typecast mc_obj for the collective group
- `root_addr` is the address of the designated broadcast root
- `datatype` is the data type
- `flags` modify the operation (see below)
- `context` is a user context pointer

This operation initiates delivery of the data supplied by the designated `root_addr` to all endpoints.

It is implemented as an allreduce using the bitwise OR operator. The data provided in `buf` is used on the `root_addr` endpoint, and zero is used on all other endpoints.

Upon completion, `buf` on every endpoint will contain the contents of `buf` from the designated `root_addr`.

Note: `data` is limited to 16 bytes.

### Reduce

```
ssize_t fi_reduce(struct fid_ep *ep, const void *buf, size_t count,
				void *desc, void *result, void *result_desc,
				fi_addr_t coll_addr, fi_addr_t root_addr,
				enum fi_datatype datatype, enum fi_op op,
				uint64_t flags, void *context)
```
- `ep` is the endpoint on which the function is called
- `buf` is the buffer to be sent
- `count` is the data count
- `desc` is ignored
- `result` is the result buffer
- `result_desc` is ignored
- `coll_addr` is the typecast mc_obj for the collective group
- `root_addr` is the address of the result target
- `datatype` is the data type
- `fi_op` is the reduction operator
- `flags` modify the operation (see below)
- `context` is a user context pointer

This operation initiates reduction of the data supplied in `buf` from all endpoints and delivers the `result` in the designated `root_addr`.

It is implemented as an allreduce operation, where the result on all endpoints other than `root_addr` is discarded.

The `result` parameter can be NULL on all endpoints other than the `root_addr` endpoint.

Note: `data` is limited to 16 bytes.

### Allreduce

```
ssize_t fi_allreduce(struct fid_ep *ep, const void *buf, size_t count,
				void *desc, void *result, void *result_desc,
				fi_addr_t coll_addr,
				enum fi_datatype datatype, enum fi_op op,
				uint64_t flags, void *context)
```
- `ep` is the endpoint on which the function is called
- `buf` is the buffer to be sent/received
- `count` is the data count
- `desc` is ignored
- `result` contains the reduced result on completion
- `result_desc` is ignored
- `coll_addr` is the typecast mc_obj for the collective group
- `datatype` is the data type
- `fi_op` is the reduction operator
- `flags` modify the operation (see below)
- `context` is a user context pointer

This operation initiates reduction of the data supplied in `buf` from all endpoints and delivers it to the `result` on all endpoints.

Note: `data` is limited to 16 bytes.

### Collective flags

Calling any reduction function normally submits the reduction to the fabric.

In collective practice, multiple threads are used on a given compute node, each representing a separate reduction rank. One of these ranks is designated the "captain rank," which pre-reduces data from each of the ranks (including itself) before initiating the multi-endpoint reduction.

This local reduction is typically performed using normal C operators, such as sum, multiply, logical operations, or bitwise operations.

Accelerated collectives provide two "novel" operators, the `MINMAXLOC` operator and the `REPSUM` operator.

To allow these functions to be easily used, the `FI_MORE` flag can be specified for any accelerated collective reduction, which -- as the name suggests -- informs the reduction that more data is expected. This reduces data (in software) and holds the reduction data without submitting it to the fabric. This can be repeated any number of times to continue to accumulate results. When a subsequent reduction is then performed without the `FI_MORE` flag, the supplied value is taken as the final contribution, is locally reduced with the existing reduction data, and the result is submitted to the fabric for collective reduction across endpoints.

This mechanism can be used for any operator, such as `FI_SUM`, but this is not generally the most efficient way to do this, since the normal addition operators are available in C.

### Collective operators

The following reduction operators are supported (maximum count in parentheses):

| Operator  | (u)int8/16/32 | int64 | uint64 | double | minmaxloc |
|:----------|:--------------|:------|:-------|:-------|:----------|
| BAND      | yes*          |       | yes(4) |        |           |
| BXOR      | yes*          |       | yes(4) |        |           |
| BOR       | yes*          |       | yes(4) |        |           |
| MIN       |               | yes(4)|        | yes(4) |           |
| MAX       |               | yes(4)|        | yes(4) |           |
| SUM       |               | yes(4)|        | yes(4) |           |
| REPSUM    |               |       |        | yes(1) |           |
| MINMAXLOC |               |       |        |        | yes(1)    |

Note: * `BAND`, `BXOR`, and `BOR` do not test to reject collections of signed 8/16/32 bits, but reduce them as packed collections of up to 4 `uint64_t`.


#### NEW OPERATOR MINMAXLOC

The `minmaxloc` operation performs a minimum and a maximum in a single operation, returning both the minimum and maximum values, along with the index of the endpoint that contributed that minimum or maximum.

It can be used to implement the `MINLOC` or `MAXLOC` operations by simply setting the unwanted fields to zero and ignoring the result

The `minmaxloc` structure is specialized:

| Offset | Field     | Data Type |
|:-------|:----------|:----------|
| 0      | minval    | int64     |
| 4      | minidx    | uint64    |
| 8      | maxval    | int64     |
| 12     | maxidx    | uint64    |


#### NEW OPERATOR REPSUM

The REPSUM operator uses the REPROBLAS algorithm described below:

https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf
Algorithm 7

This algorithm provides extended-precision double precision summation, with associative behavior (summation is order-independent).

Because the summation occurs within a multicast tree that may take different paths through the fabric on different runs based on other jobs that are running and using the fabric, the order of summation within the reduction cannot be generally predicted or controlled. The well-known ordering problem of double-precision floating point can lead to varying results on each run.

The REPSUM algorithm improves on the accuracy of the summation by implicitly adding more bits to the computations, but more importantly, guarantees that all additions are associative, meaning they are order-independent.


## Collective Close

```
int fi_close(struct fid *fid);
```
`fi_close()` can be called on the mc_obj file identifier returned by fi_join_collective.

If the application does not call this before attempting to exit, the application on one or more endpoints will likely throw exceptions and WLM job abort, due to unsynchronized removal of global resources.

The WLM will perform necessary cleanup of global resources.

# Environment variables

## Workload Manager Environment

The following environment variables must be provided by the WLM (Workload Manager) to enable collectives:

| Name						    	 | Format	| Meaning|
|:-----------------------------|:----------|:----------|
| `FI_CXI_COLL_JOB_ID`			 | integer   | WLM job identifier |
| `FI_CXI_COLL_JOB_STEP_ID`    | integer   | WLM job step identifier |
| `FI_CXI_COLL_MCAST_TOKEN`	 | string	   | FM API REST authorization token |
| `FI_CXI_COLL_FABRIC_MGR_URL` | string	   | FM API REST URL |
| `FI_CXI_HWCOLL_ADDRS_PER_JOB`| integer   | maximum quota for mcast addresses |

## User Environment

The following environment variable can be provided by the user application to control collective behavior.

| Name							| Format	| Default | Meaning |
|:-----------------------------|:----------|:-------|:----|
| `FI_CXI_COLL_RETRY_USEC`		| integer	| 32000    | retry period on dropped packets |


# Provider-Specific Error Codes

Provider-specific error codes are supplied through the normal `fi_cq_readerr()` and `fi_eq_readerr()` functions.

A typical optimization is to use `fi_*_read()` with a smaller buffer, and if this fails with -FI_EAVAIL, to use a larger buffer and call `fi_*_readerr()`.

There are two blocks of errors, found in `fi_cxi_ext.h`.

### Reduction Errors

Reduction errors are reported through the CQ, which is polled to detect reduction completion events.


| Erro code  | Value | Meaning |
|:-----------|:------|:--------|
|`FI_CXI_ERRNO_RED_FLT_OVERFLOW`   | 1024 | double precision value overflow |
|`FI_CXI_ERRNO_RED_FLT_INVALID`    | 1025 | double precision sNAN/inf value |
|`FI_CXI_ERRNO_RED_INT_OVERFLOW`   | 1026 | reproducible sum overflow |
|`FI_CXI_ERRNO_RED_CONTR_OVERFLOW` | 1027 | reduction contribution overflow |
|`FI_CXI_ERRNO_RED_OP_MISMATCH`    | 1028 | reduction opcode mismatch |
|`FI_CXI_ERRNO_RED_MC_FAILURE`     | 1029 | unused |
|`FI_CXI_ERRNO_RED_OTHER`          | 1030 | non-specific reduction error, fatal |

### Join Errors

Join errors are reported through the EQ, which is polled to detect collective join completion events.

| Error code  | Value | Meaning |
|:------------|:------|:--------|
|`FI_CXI_ERRNO_JOIN_MCAST_INUSE`    | 2048| endpoint already using mcast address |
|`FI_CXI_ERRNO_JOIN_HWROOT_INUSE`   | 2049| endpoint already serving as HWRoot |
|`FI_CXI_ERRNO_JOIN_MCAST_INVALID`  | 2050| mcast address from FM is invalid  |
|`FI_CXI_ERRNO_JOIN_HWROOT_INVALID` | 2051| HWRoot address from FM is invalid |
|`FI_CXI_ERRNO_JOIN_CURL_FAILED`    | 2052| libcurl initiation failed |
|`FI_CXI_ERRNO_JOIN_CURL_TIMEOUT`   | 2053| libcurl timed out |
|`FI_CXI_ERRNO_JOIN_SERVER_ERR`     | 2054| unhandled CURL response code |
|`FI_CXI_ERRNO_JOIN_FAIL_PTE`       | 2055| libfabric PTE allocation failed |
|`FI_CXI_ERRNO_JOIN_OTHER`          | 2056| non-specific JOIN error, fatal |


