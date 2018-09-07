---
layout:page
title: hpcs(1)
tagline: High-Performance Connection Scaling test framework
---
{% include JB/setup %}

# NAME

hpcs-sendrecv
hpcs-onesided

# SYNOPSIS
```
	hpcs-sendrecv [CORE OPTIONS] -- [PATTERN OPTIONS] -- [TEST OPTIONS]
	hpcs-onesided [CORE OPTIONS] -- [PATTERN OPTIONS] -- [TEST OPTIONS]
```

# DESCRIPTION

HPCS is an OFI-level test framework designed to generate network traffic between an arbitrary number of nodes, to enable large-scale testing at the OFI layer.

MPI is often used for multi-node tests, but failures tend to be opaque and those tests only stress whatever portion of OFI that particular MPI uses.  HPCS allows us to do similar scale-out tests, but with libfabric communication primitives.

HPCS uses MPI for some basic out-of-band startup and communication barriers.  We recommend against using an OFI-enabled MPI in most cases, to ensure that HPCS test runs are only testing those parts of OFI invoked by HPCS directly.

An HPCS application is composed of a core module, a pattern, and a test, which each accept certain arguments.

The HPCS core is the base framework.  It handles initialization and drives callbacks from the other two components.

A pattern defines which nodes send or receive from which other nodes.  Patterns are selected with a command line argument passed to core. The currently available patterns are "self", "alltoall", "alltoone", and "ring".

A test defines the low-level OFI primitives that a sending node will use to initiate traffic and the receiving node will use to receive, along with how data validity is checked and how OFI resources are allocated and released.  Only one test may be compiled in at once and therefore we provide separate binaries for each test (currently "sendrecv" and "onesided").

HPCS determines number of ranks and local rank number directly from MPI.

## CORE OPTIONS

*-p, --prov=\<provider\>*
: Specify provider.

*-w, --window=\<window\>*
: Control how many parallel operations can be outstanding at a time.

*-o, --order=[expected|unexpected|none]*
: Impose ordering between sends and receives.  (Both expected and unexpected use MPI barriers to ensure ordering.)

*-i, --iterations=\<n\>*
: Run the same test/pattern multiple times.

*-p, --pattern=[self|alltoall|alltoone|ring]*
: Select pattern.

*-v, --verbose*

*-h, --help*

## SELF, ALL-TO-ALL PATTERN OPTIONS

The self and all-to-all patterns take no arguments.  As the names suggest, in a self pattern, each rank initiates a transfer with itself, and in all-to-all, every rank initiates a transfer with every other rank including itself.

## ALL-TO-ONE PATTERN OPTIONS

The all-to-one pattern initiates transfers from every rank to a single target rank.

*-t, --target=\<rank\>*
: Send to a particular target rank (default 0).

*-h, --help*

## RING PATTERN OPTIONS

In a ring pattern, a leader sends a message to its immediate rank (N+1)%N peer, which sends to its next peer, and so on until the last peer sends a message back to the leader.  The ring pattern uses triggered ops.

*-l, --leader=\<rank\>*
: Set a particular rank as the leader node.  Setting leader to -1 results in no leader, which is expected to deadlock if triggered ops are working as expected.

*-r, --rings=\<n\>*
: Execute multiple rings concurrently.  Leaders are chosen sequentially.  The number of rings cannot exceed the number of ranks in the job.

*-m, --multi-ring*
: Each rank is the leader of a single ring, and they all execute concurrently.

*-v, --verbose*

*-h, --help*

## SEND-RECV TEST OPTIONS

The send-recv test uses plain tagged message sends and receives.

*-s, --size=\<n\>*
: Message length.

*-w, --workqueue*
: Use new triggered ops experimental workqueue API rather than the stable API.  This only matters for patterns that use triggered ops, such as ring.

*-h, --help*

## ONE-SIDED TEST OPTIONS

The onesided test may use RMA reads, writes, or atomics, though read support may be deprecated, as it relies on using counters in a way that is no longer officially supported.  (Well behaved providers should simply refuse the relevant counter binding, in which case the test prints a message that the test was skipped and HPCS returns 0.)

Onesided tests should be run with "expected" ordering.

*-s, --size=\<n\>*
: Message length.

*-m, --offset-mode=[same-offset|different-offset]*
: Control whether all peers write into the same buffer at the same offset, or at different offsets.  Usually, the latter is preferred for RMA writes but the former can be useful for testing atomics.  (Warning: different-offset doesn't currently work properly with all-to-one and ring patterns.)

*-o, --op=[read|write|add]*
: Set what operation is being tested.  The add test performs an atomic increment on the target buffer.

*-r, --repeat=\<n\>*
: Perform the given operation multiple times.  This is expected to perform better in most cases than increasing  the "--iterations" core argument, as the onesided test does not begin a later iteration until completions from the previous iteration are done whereas repeated operations will be sent eagerly, and HPCS will wait for the whole batch to complete at once.  RMA writes just overwrite the same buffer with the same content multiple times, so atomic adds are preferred if one wants to ensure that the right number of operations completed on the target.

*-w, --workqueue*
: Use new triggered ops experimental workqueue API rather than the stable API.  This only matters for patterns
 that use triggered ops, such as ring.

*-h, --help*

# EXAMPLES

```
	$ mpiexec.hydra -n 2 hpcs_sendrecv --order=unexpected --pattern=alltoall -- -- --size=4
```

Run a basic send/recv unexpected tagged message test between two nodes using Hydra as the MPI launcher.  Both ranks send a single message of four bytes to themselves and the other.

```
	$ mpiexec.hydra -n 3 hpcs_onesided --order=expected --pattern=alltoone -- --target=2 -- --offset-mode=same-offset --op=add --repeat=20
```

Run an atomics test, sending twenty atomic increments each from all three ranks to rank 2.  The same buffer/offset is used on the target, so it should be incremented sixty times.


