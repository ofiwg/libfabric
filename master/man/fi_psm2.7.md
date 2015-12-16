---
layout: page
title: fi_psm2(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The PSM2 Fabric Provider

# OVERVIEW

The *psm2* provider runs over the PSM 2.x interface that is supported
by the Intel Omni-Path Fabric. PSM 2.x has all the PSM 1.x features
plus a set of new functions with enhanced capability. Since PSM 1.x
and PSM 2.x are not ABI compatible the *psm2* provider only works with
PSM 2.x and doesn't support Intel TrueScale Fabric.

# LIMITATIONS

The *psm2* provider doesn't support all the features defined in the
libfabric API. Here are some of the limitations:

Endpoint types
: Only support non-connection based types *FI_DGRAM* and *FI_RDM*

Endpoint capabilities
: Endpoints can support any combination of data transfer capabilities
  *FI_TAGGED*, *FI_MSG*, *FI_ATOMICS*, and *FI_RMA*. These capabilities
  can be further refined by *FI_SEND*, *FI_RECV*, *FI_READ*, *FI_WRITE*,
  *FI_REMOTE_READ*, and *FI_REMOTE_WRITE* to limit the direction of
  operations.

  *FI_MULTI_RECV* is supported for non-tagged message queue only.

  Other supported capabilities include *FI_TRIGGER*, *FI_REMOTE_CQ_DATA*,
  and *FI_SOURCE*.

Modes
: *FI_CONTEXT* is required for the *FI_TAGGED* and *FI_MSG*
  capabilities. That means, any request belonging to these two
  categories that generates a completion must pass as the operation
  context a valid pointer to type *struct fi_context*, and the space
  referenced by the pointer must remain untouched until the request
  has completed. If none of *FI_TAGGED* and *FI_MSG* is asked for,
  the *FI_CONTEXT* mode is not required.
  
Progress
: The *psm2* provider requires manual progress. The application is
  expected to call *fi_cq_read* or *fi_cntr_read* function from time
  to time when no other libfabric function is called to ensure
  progress is made in a timely manner. The provider does support
  auto progress mode. However, the performance is signinficantly
  impacted if the application purely depends on the pprovide to
  make auto progress.

Unsupported features
: These features are unsupported: connection management, 
  scalable endpoint, passive endpoint, shared receive context,
  and send/inject with immediate data over tagged message queue.

# RUNTIME PARAMETERS

The *psm2* provider checks for the following environment variables:

*FI_PSM2_UUID*
: PSM requires that each job has a unique ID (UUID). All the processes
  in the same job need to use the same UUID in order to be able to
  talk to each other. The PSM reference manual advises to keep UUID
  unique to each job. In practice, it generally works fine to reuse
  UUID as long as (1) no two jobs with the same UUID are running at 
  the same time; and (2) previous jobs with the same UUID have exited
  normally. If running into "resource busy" or "connection failure"
  issues with unknown reason, it is advisable to manually set the UUID
  to a value different from the default.

  The default UUID 00FF00FF-0000-0000-0000-00FF0F0F00FF.

*FI_PSM2_NAME_SERVER*
: The *psm2* provider has a simple built-in name server that can be used
  to resolve an IP address or host name into a transport address needed
  by the *fi_av_insert* call. The main purpose of this name server is to
  allow simple client-server type applications (such as those in *fabtest*)
  to be written purely with libfabric, without using any out-of-band
  communication mechanism. For such applications, the server would run first,
  and the client would call *fi_getinfo* with the *node* parameter set to
  the IP address or host name of the server. The resulting *fi_info* structure
  would have the transport address of the server in the *dest_addr* field.

  The name server won't work properly if there are more than one processes
  from the same job (i.e. with the same UUID) running on the same node and
  acting as servers. For such scenario it is recommended to have each
  process getting local transport address with *fi_cm_getname* and exchanging
  the addresses with out-of-band mechanism.

  The name server is on by default. It can be turned off by setting the
  variable to 0. This may save a small amount of resource since a separate
  thread is created when the name server is on.

  The provider detects OpenMPI and MPICH runs and changes the default setting
  to off.

*FI_PSM2_TAGGED_RMA*
: The RMA functions are implemented on top of the PSM Active Message functions.
  The Active Message functions has limit on the size of data can be transferred
  in a single message. Large transfers can be divided into small chunks and
  be pipe-lined. However, the bandwidth is sub-optimal by doing this way.

  The *psm2* provider use PSM tag-matching message queue functions to achieve
  higher bandwidth for large size RMA. It takes avdantage of the extra tag bits
  available in PSM2 to separate the RMA traffic from the regular tagged message
  queue.
   
  The option is on by default. To turn it off set the variable to 0.

*FI_PSM2_DELAY*
: Time (seconds) to sleep before closing PSM endpoints. This is a workaround
  for a bug in some versions of PSM library.

  The default setting is 1.

*FI_PSM2_TIMEOUT*
: Timeout (seconds) for gracefully closing PSM endpoints. A forced closing
  will be issued if timeout expires.

  The default setting is 5.

*FI_PSM2_PROG_INTERVAL*
: When auto progress is enabled (asked via the hints to *fi_getinfo*),
  a progress thread is created to make progress calls from time to time.
  This option set the tnterval (microseconds) between progress calls.

  The default setting is 1 if affininty is set, or 1000 if not. See
  *FI_PSM2_PROG_AFFINITY*.

*FI_PSM2_PROG_AFFINITY*
: When set, specify the set of CPU cores to set the progress thread
  affinity to. The format is
  `<start>[:<end>[:<stride>]][,<start>[:<end>[:<stride>]]]*`, 
  where each triplet `<start>:<end>:<stride>` defines a block of
  core_ids. Both `<start>` and `<end>` can be either the `core_id`
  (when >=0) or `core_id - num_cores` (when <0). 

  By default affinity is not set.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_psm`(7)](fi_psm.7.html),
