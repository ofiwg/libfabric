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
applications to seamlessly use multiple providers or NICs. This provider
uses the libfabric peer infrastructure to aid in the use of the underlying
providers.  This version of the provider is able to link any libfabric
provider which supports the FI_PEER capability.


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
  The first release of the provider only supports tagged operations.
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
  shm+cxi while another linking shm+rxm.

*Message Operations*
: LNX is designed to intercept message operations such as fi_tsenddata
  and based on specific criteria forward the operation to the appropriate
  provider. LNX can link any provider which supports the FI_PEER
  capability. SHM provider is special cased in the sense that it's always
  used for intra-node communication if it's part of the link. Otherwise
  all other providers in the link are used in round robin.

*Using the Provider*
: In order to use the provider the user needs to set FI_LNX_PROV_LINKS
  environment variable to the linked providers; see syntax under variable
  description. This allows the definition of multiple links. Each link
  defines the provider to add to the link along with an explicit list of
  domains. If only the provider name is specified then all the domains are
  added to the link.
  The provider returns the list of links specified to the application in
  the fi_getinfo() call. Since there could be multiple providers in the
  link, the first domain of the first non-shm provider fi_info is used as
  the link info. This means that if a link is defined as follows:
     - shm+cxi:cxi0,cxi1
  The cxi0 fi_info information is used as the link fi_info with the
  modification of the provider and domain names. This is important since the
  PCI information is part of the fi_info and some applications might use the
  PCI information to decide which link to use. A common selection criteria
  is the interface nearest the core the process is bound to. In order to
  make this determination, the application requires the interface's PCI
  information

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
: This release of LNX supports tagged operations only. Future
  releases will expand the support to other operation types.

*Multi-Rail*
: LNX implements a multi-rail feature that enables messages to be
  transmitted across multiple interfaces, driven by the link configuration.
  It supports configuring multiple provider domains within a single link.
  For example, the following configuration:
       FI_LNX_PROV_LINKS="cxi:cxi0,cxi1"
  Defines a link that includes the cxi0 and cxi1 domains. When an
  application utilizes this link, messages are distributed in a round-robin
  fashion across both domains. Similarly, all available peer addresses are
  used in a round-robin manner, ensuring balanced communication.
  An exception applies to the shm provider: if included in the link, all
  intra-node traffic is routed through the shm provider.
  For the following configuration:
      FI_LNX_PROV_LINKS="tcp;ofi_rxm+cxi:cxi0"
  The link consists of all tcp domains associated with the tcp;ofi_rxm
  provider, along with cxi0. Messages are evenly distributed across all
  domain endpoints and peer addresses. The provider ensures that local
  endpoint types match corresponding remote addresses—for instance, if the
  tcp provider is used, messages are directed to the peer's tcp address.


# RUNTIME PARAMETERS

The *LNX* provider checks for the following environment variables:

*FI_LNX_PROV_LINKS*
: This environment variable is used to specify which providers to link. This
  must be set in order for the LNX provider to return a list of fi_info
  blocks in the fi_getinfo() call. The format which must be used is:
       link ::= group ['|' group]*
       group ::= provider ['+' provider [':' provider_list]]
       provider_list ::= provider [',' provider]*
       provider ::= [a-zA-Z0-9_]+
  Examples:
     - shm+cxi:cxi0
     - shm+cxi:cxi0,cxi1
     - shm+cxi:cxi0|shm+cxi:cxi1
     - shm+cxi
  On a system with four cxi domains, the last example is equivalent to:
     - shm+cxi:cxi0,cxi1,cxi2,cxi3

*FI_LNX_DISABLE_SHM*
: By default this environment variable is set to 0. However, the user can
  set it to one and then the SHM provider will not be used. This can be
  useful for debugging and performance analysis. The SHM provider will
  naturally be used for all intra-node operations. Therefore, to test SHM in
  isolation with LNX, the processes can be limited to the same node only.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)
