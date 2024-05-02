---
layout: page
title: fi_lnx(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_lnx \- The LINKx (LNX) Provider

# OVERVIEW

The LNX provider is designed to link two or more providers, allowing
applications to seamlessly use multiple providers or NICs. This provider uses
the libfabric peer infrastructure to aid in the use of the underlying providers.
This version of the provider currently supports linking the libfabric
shared memory provider for intra-node traffic and another provider for
inter-node traffic. Future releases of the provider will allow linking any
number of providers and provide the users with the ability to influence
the way the providers are utilized for traffic load.

# SUPPORTED FEATURES

This release contains an initial implementation of the LNX provider that
offers the following support:

*Endpoint types*
: The provider supports only endpoint type *FI_EP_RDM*.

*Endpoint capabilities*
: LNX is a passthrough layer on the send path. On the receive path LNX
  utilizes the peer infrastructure to create shared receive queues (SRQ).
  Receive requests are placed on the SRQ instead of on the core provider
  receive queue. When the provider receives a message it queries the SRQ for
  a match. If one is found the receive request is completed, otherwise the
  message is placed on the LNX shared unexpected queue (SUQ). Further receive
  requests query the SUQ for matches.
  The first release of the provider only supports tagged and RMA operations.
  Other message types will be supported in future releases.

*Modes*
: The provider does not require the use of any mode bits.

*Progress*
: LNX utilizes the peer infrastructure to provide a shared completion
  queue. Each linked provider still needs to handle its own progress.
  Completion events will however be placed on the shared completion queue,
  which is passed to the application for access.

*Address Format*
: LNX wraps the linked providers addresses in one common binary blob.
  It does not alter or change the linked providers address format. It wraps
  them into a LNX structure which is then flattened and returned to the
  application. This is passed between different nodes. The LNX provider
  is able to parse the flattened format and operate on the different links.
  This assumes that nodes in the same group are all using the same version of
  the provider with the exact same links. IE: you can't have one node linking
  SHM+CXI while another linking SHM+RXM.

*Message Operations*
: LNX is designed to intercept message operations such as fi_tsenddata
  and based on specific criteria forward the operation to the appropriate
  provider. For the first release, LNX will only support linking SHM
  provider for intra-node traffic and another provider (ex: CXI) for inter
  node traffic. LNX send operation looks at the destination and based on
  whether the destination is local or remote it will select the provider to
  forward the operation to. The receive case has been described earlier.

*Using the Provider*
: In order to use the provider the user needs to set FI_LNX_PROV_LINKS
  environment variable to the linked providers in the following format
  shm+<prov>. This will allow LNX to report back to the application in the
  fi_getinfo() call the different links which can be selected. Since there are
  multiple domains per provider LNX reports a permutation of all the
  possible links. For example if there are two CXI interfaces on the machine
  LNX will report back shm+cxi0 and shm+cxi1. The application can then
  select based on its own criteria the link it wishes to use.
  The application typically uses the PCI information in the fi_info
  structure to select the interface to use. A common selection criteria is
  the interface nearest the core the process is bound to. In order to make
  this determination, the application requires the PCI information about the
  interface. For this reason LNX forwards the PCI information for the
  inter-node provider in the link to the application.

# LIMITATIONS AND FUTURE WORK

*Hardware Support*
: LNX doesn't support hardware offload; ex hardware tag matching. This is
  an inherit limitation when using the peer infrastructure. Due to the use
  of a shared receive queue which linked providers need to query when
  a message is received, any hardware offload which requires sending the
  receive buffers to the hardware directly will not work with the shared
  receive queue. The shared receive queue provides two advantages; 1) reduce
  memory usage, 2) coordinate the receive operations. For #2 this is needed
  when receiving from FI_ADDR_UNSPEC. In this case both providers which are
  part of the link can race to gain access to the receive buffer. It is
  a future effort to determine a way to use hardware tag matching and other
  hardware offload capability with LNX

*Limited Linking*
: This release of the provider supports linking SHM provider for intra-node
  operations and another provider which supports the FI_PEER capability for
  inter-node operations. It is a future effort to expand to link any
  multiple sets of providers.

*Memory Registration*
: As part of the memory registration operation, varying hardware can perform
  hardware specific steps such as memory pinning. Due to the fact that
  memory registration APIs do not specify the source or destination
  addresses it is not possible for LNX to determine which provider to
  forward the memory registration to. LNX, therefore, registers the memory
  with all linked providers. This might not be efficient and might have
  unforeseen side effects. A better method is needed to support memory
  registration. One option is to have memory registration cache in lnx
  to avoid expensive operations.

*Operation Types*
: This release of LNX supports tagged and RMA operations only. Future
  releases will expand the support to other operation types.

*Multi-Rail*
: Future design effort is being planned to support utilizing multiple interfaces
  for traffic simultaneously. This can be over homogeneous interfaces or over
  heterogeneous interfaces.

# RUNTIME PARAMETERS

The *LNX* provider checks for the following environment variables:

*FI_LNX_PROV_LINKS*
: This environment variable is used to specify which providers to link. This
  must be set in order for the LNX provider to return a list of fi_info
  blocks in the fi_getinfo() call. The format which must be used is:
  <prov1>+<prov2>+... As mentioned earlier currently LNX supports linking
  only two providers the first of which is SHM followed by one other
  provider for inter-node operations

*FI_LNX_DISABLE_SHM*
: By default this environment variable is set to 0. However, the user can
  set it to one and then the SHM provider will not be used. This can be
  useful for debugging and performance analysis. The SHM provider will
  naturally be used for all intra-node operations. Therefore, to test SHM in
  isolation with LNX, the processes can be limited to the same node only.

*FI_LNX_USE_SRQ*
: Shared Receive Queues are integral part of the peer infrastructure, but
  they have the limitation of not using hardware offload, such as tag
  matching. SRQ is needed to support the FI_ADDR_UNSPEC case. If the application
  is sure this will never be the case, then it can turn off SRQ support by
  setting this environment variable to 0. It is 1 by default.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
