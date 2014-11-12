libfabric
=========

The Open Fabrics Interfaces (OFI) is a framework focused on exporting fabric communication services to applications.

Libfabric is a software library instantiation of OFI with the following objectives:

<img src="https://raw.githubusercontent.com/wiki/ofiwg/libfabric/openfabric-interfaces-overview.png">

Libfabric is being developed by the OFI Working Group (OFIWG or "ofee-wig"), a subgroup of the OpenFabrics Alliance.  However, participation in the OFIWG is open to anyone.  The charter of the OFIWG is:

> Develop an extensible, open source framework and interfaces aligned with upper-layer protocols and application needs for high-performance fabric services.

The goal of OFI and libfabric is to define interfaces that enable a tight semantic map between applications and underlying fabric services.  Specifically, libfabric software interfaces have been co-designed with fabric hardware providers and application developers, with a focus on the needs of HPC users.  OFI supports multiple interface semantics, is fabric and vendor agnostic, and leverages and expands the existing RDMA open source community.  A high-level view of the libfabric architecture is shown below.

<img src="https://raw.githubusercontent.com/wiki/ofiwg/libfabric/fabric-interface-groups.png">

Libfabric is designed to minimize the impedance mismatch between applications, including middleware such as MPI, SHMEM, and PGAS, and fabric communication hardware.  Its interfaces target high-bandwidth, low-latency NICs, with a goal to scale to tens of thousands of nodes.

How do I get involved?
======================

First: read the documentation.

The bulk of the libfabric code base is being developed in this GitHub repository.  As part of this repository, a set of Linux man pages are being carefully written to exactly specify the libfabric APIs.  Read through these man pages to get a sense of the libfabric APIs.

Second: become part of the conversation.

[Join the ofiwg mailing list](http://lists.openfabrics.org/mailman/listinfo/ofiwg).  Notices of the Tuesday-weekly OFIWG Webexes are sent on this list; anyone can join the calls to listen and participate in the design of libfabric.
