Libfabric release notes
=======================

This file contains the main features as well as overviews of specific
bug fixes (and other actions) for each version of Libfabric since
version 1.0.

v1.6.2, Fri Sep 28, 2018
========================

## Core

- Cleanup of debug messages

## GNI
- Fix problems with Scalable Endpoint creation
- Fix interoperability problem with HPC toolkit
- Improve configuration check for kdreg

## PSM

- Enforce FI_RMA_EVENT checking when updating counters
- Fix race condition in fi_cq_readerr()
- Always try to make progress when fi_cntr_read is called

## PSM2

- Revert "Avoid long delay in psm2_ep_close"
- Fix memory corruption related to sendv
- Performance tweak for bi-directional send/recv on KNL
- Fix CPU detection
- Enforce FI_RMA_EVENT checking when updating counters
- Remove stale info from address vector when disconnecting
- Fix race condition in fi_cq_readerr()
- Adjust reported context numbers for special cases
- Always try to make progress when fi_cntr_read is called
- Support control functions related to MR mode
- Unblock fi_cntr_wait on errors
- Properly update error counters
- Fix irregular performance drop for aggregated RMA operations
- Reset Tx/Rx context counter when fabric is initialized
- Fix incorrect completion event for iov send

## RXM

- Fix incorrect increments of error counters for small messages
- Increment write completion counter for small transfers
- Use FI_UNIVERSE_SIZE when defining MSG provider CQ size
- Make TX, RX queue sizes independent of MSG provider
- Make deferred requests opt-in
- Fill missing rxm_conn in rx_buf when shared context is not used
- Fix an issue where MSG endpoint recv queue got empty resulting
  in a hang
- Set FI_ORDER_NONE for tx and rx completion ordering
- Serialize access to repost_ready_list
- Reprocess unexpected messages on av update
- Fix a bug in matching directed receives
- Fix desc field when postponing RMA ops
- Fix incorrect reporting of mem_tag format
- Don't include FI_DIRECTED_RECV, FI_SOURCE caps if they're not needed
- Fix matching for RMA I/O vectors

## Sockets

- Increase maximum messages size as MPICH bug work-around

## Verbs

- Detect string format of wildcard address in node argument
- Don't report unusable fi_info (no source IP address)
- Don't assert when a verbs device exposes unsupported MTU types
- Report correct rma_iov_limit
- Add new variable - FI_VERBS_MR_CACHE_MERGE_REGIONS
- eq->err.err must return a positive error code

v1.6.1, Wed May 8, 2018
===========================

## Core

- Fix compile issues with older compilers
- Check that all debug compiler flags are supported by compiler

## PSM2

- Fix occasional assertion failure in psm2_ep_close
- Avoid long delay in psm2_ep_close
- Fix potential duplication of iov send completion
- Replace some parameter checking with assertions
- Check iov limit in sendmsg
- Avoid adding FI_TRIGGER caps automatically
- Avoid unnecessary calls to psmx2_am_progress()

## RXM

- Fix reading pointer after freeing it.
- Avoid reading invalid AV entry
- Handle deleting the same address multiple times
- Fix crash in fi_av_remove if FI_SOURCE wasn't enabled

## Sockets

- Fix use after free error handling triggered ops.

v1.6.0, Wed Mar 14, 2018
========================

## Core

- Introduces support for performing RMA operations to persistent memory
  See FI_RMA_PMEM capability in fi_getinfo.3
- Define additional errno values
- General code cleanups and restructuring
- Force provider ordering when using dynamically loaded providers
- Add const to fi_getinfo() hints parameter
- Improve use of epoll for better scalability
- Fixes to generic name service
- Implemented support of MR caching without merging of regions

## GNI
- Fix a problem with the GNI waitset implementation
- Enable use of XPMEM for intra node data transfers
- Fix a problem with usage of Crays UDREG registration cache
- Fix a problem with an assert statement
- Fix several memory leaks

## PSM

- Move environment variable reading out from fi_getinfo()
- Shortcut obviously unsuccessful fi_getinfo() calls
- Remove excessive name sever implementation
- Enable ordering of RMA operations

## PSM2

- Requires psm2 library version 10.2.235 or later
- Skip inactive units in round-robin context allocation
- Allow contexts be shared by Tx-only and Rx-only endpoints
- Use utility functions to check provider attributes
- Turn on FI_THREAD_SAFE support
- Make address vector operations thread-safe
- Move environment variable reading out from fi_getinfo()
- Reduce noise when optimizing tagged message functions
- Shortcut obviously unsuccessful fi_getinfo() calls
- Improve how Tx/Rx context limits are handled
- Support auto selection from two different tag layout schemes
- Add provider build options to debug output
- Support remote CQ data for tagged messages, add specialization.
- Support opening multiple domains
- Put trigger implementation into a separate file
- Update makefile and configure script
- Replace allocated context with reserved space in psm2_mq_req
- Limit exported symbols for DSO provider
- Reduce HW context usage for certain TX only endpoints
- Remove unnecessary dependencies from the configure script
- Refactor the handling of op context type
- Optimize the conversion between 96-bit and 64-bit tags
- Code refactoring for completion generation
- Remove obsolete feature checking code
- Report correct source address for scalable endpoints
- Allow binding any number of endpoints to a CQ/counter
- Add shared Tx context support
- Add alternative implementation for completion polling
- Change the default value of FI_PSM2_DELAY to 0
- Add an environment variable for automatic connection cleanup
- Abstract the completion polling mechanism
- Use the new psm2_am_register_handlers_2 function when available
- Allow specialization when FI_COMPLETION op_flag is set.
- Put Tx/Rx context related functions into a separate file
- Enable PSM2 multi-ep feature by default
- Add option to build with PSM2 source included
- Simplify the code for checking endpoint capabilities
- Simplify the handling of self-targeted RMA operations
- Allow all free contexts be used for scalable endpoints
- Enable ordering of RMA operations
- Enable multiple endpoints over PSM2 multi-ep support
- Support multiple Tx/Rx contexts in address vector
- Remove the virtual lane mechanism
- Less code duplication in tagged, add more specialization.
- Allow PSM2 epid be reused within the same session
- Turn on user adjustable inject size for all operations
- Use pre-allocated memory pool for RMA requests
- Add support for lazy connection
- Various bug fixes

## RXM

- Add support for completion counters
- Fix MR mode handling
- Add support for FI_MULTI_RECV
- Considerable performance optimizations
- Report correct MR key size based on core provider's size
- Fixes to endpoint address reporting to avoid wildcard addresses
- Ensure progress after core provider returns EAGAIN on transfers
- Fix crash when running of sockets provider
- Bug fixes handling large message transfers
- Report data ordering and limits based on core provider's
- Set mode bits and capabilities correctly
- General code restructuring and cleanups
- Various additional bug fixes
- Handle different API versions correctly
- Expand support for tagged message transfers
- Add support for auto progress on data transfers

## SHM

- Initial release of shared memory provider
- See the fi_shm.7 man page for details on available features and limitations

## Sockets

- Scalability enhancements
- Fix issue associating a connection with an AV entry that could result in
  application hangs
- Add support for new persistent memory capabilities
- Fix fi_cq_signal to unblock threads waiting on cq sread calls
- Fix epoll_wait loop handling to avoid out of memory errors
- Add support for TCP keepalives, controllable via environment variables
- Reduce the number of threads allocated for handling connections
- Several code cleanups in response to static code analysis reports
- Fix reporting multiple completion events for the same request in error cases

## usNIC

- Minor adjustments to match new core MR mode bits functionality
- Several code cleanups in response to static code analysis reports

## Verbs

- Code cleanups and simplifications
- General code optimizations to improve performance
- Fix handling of wildcard addresses
- Check for fatal errors during connection establishment
- Support larger inject sizes
- Fix double locking issue
- Add support for memory registration caching (disabled by default)
- Enable setting thread affinity for CM threads
- Fix hangs in MPI closing RDM endpoints
- Add support for different CQ formats
- Fix RMA read operations over iWarp devices
- Optimize CM progress handling
- Several bug fixes

v1.5.3, Wed Dec 20, 2017
========================

## Core

- Handle malloc failures
- Ensure global lock is initialized on Windows
- Fix spelling and formatting errors in man pages

## GNI

- Fix segfault when using FI_MULTI_RECV
- Fix rcache issue handling overlapping memory regions

## NetDir

- Fix fi_getname
- Remove FI_LOCAL_MR mode bit, being reported erronously
- Avoid crashing in fi_join

## PSM

- Fix print format mismatches
- Remove 15 second startup delay when no hardware is installed
- Preserve FI_MR_SCALABLE mode bit for backwards compatability

## PSM2

- Fix print format mismatches
- Allow all to all communication between scalable endpoints
- Preserve FI_MR_SCALABLE mode bit for backwards compatability
- Fix reference counting issue with opened domains
- Fix segfault for RMA/atomic operations to local scalable endpoints
- Fix resource counting related issues for Tx/Rx contexts
- Allow completion suppression when fi_context is non-NULL
- Use correct queue for triggered operations with scalable endpoints

## RXM

- Fix out of bounds access to receive IOVs
- Serialize access to connection map
- Fix CQ error handling
- Fix issue being unable to associate an fi_addr with a connection
- Fix bug matching unexpected tagged messages
- Indicate that FI_RMA is supported
- Return correct r/w ordering size limits

## Sockets

- Fix check for invalid connection handle
- Fix crash in fi_av_remove

## Util

- Fix number of bits used for connection index

## Verbs

- Fix incorrect CQ entry data for MSG endpoints
- Properly check for errors from getifaddrs
- Retry getifaddr on failure because of busy netlink sockets
- Ack CM events on error paths

v1.5.2, Wed Nov 8, 2017
=======================

## Core

- Fix Power PC 32-bit build

## RXM

- Remove dependency on shared receive contexts
- Switch to automatic data progress
- Fix removing addresses from AV

## Sockets

- Fix incorrect reporting of counter attributes

## Verbs

- Fix reporting attributes based on device limits
- Fix incorrect CQ size reported for iWarp NICs
- Update man page with known issues for specific NICs
- Fix FI_RX_CQ_DATA mode check
- Disable on-demand paging by default (can cause data corruption)
- Disable loopback (localhost) addressing (causing failures in MPI)

v1.5.1, Wed Oct 4, 2017
=======================

## Core

- Fix initialization used by DL providers to avoid crash
- Add checks for null hints and improperly terminated strings
- Check for invalid core names passed to fabric open
- Provide consistent provider ordering when using DL providers
- Fix OFI_LIKELY definitions when GNUC is not present

## GNI

- Add ability to detect local PE rank
- Fix compiler/config problems
- Fix CQ read error corruption
- Remove tests of deprecated interfaces

## PSM

- Fix CQ corruption reporting errors
- Always generate a completion on error

## PSM2

- Fix CQ corruption reporting errors
- Always generate a completion on error
- Add checks to handle out of memory errors
- Add NULL check for iov in atomic readv/writev calls
- Fix FI_PEEK src address matching
- Fix bug in scalable endpoint address resolution
- Fix segfault bug in RMA completion generation

## Sockets

- Fix missing FI_CLAIM src address data on completion
- Fix CQ corruption reporting errors
- Fix serialization issue wrt out of order CPU writes to Tx ring buffer

## Verbs

- Allow modifying rnr retry timout to improve performance
- Add checks to handle out of memory errors
- Fix crash using atomic operations for MSG EPs

v1.5.0, Wed Aug 9, 2017
============================

The 1.5 release includes updates to the libfabric API and ABI.  As a
result, the ABI bumps from 1.0 to 1.1.  All changes are backwards
compatible with previous versions of the interface.  The following
features were added to the libfabric API.  (Note that individual
providers may not support all new features).  For full details
see the man pages.

- Authorization keys
  Authorization keys, commonly referred to as job keys, are used to
  isolate processes from communicating with other processes for security
  purposes.
- Multicast support
  Datagram endpoints can now support multicast communication.
- (Experimental) socket-like endpoint types
  New FI_SOCK_STREAM and FI_SOCK_DGRAM endpoint types are introduced.
  These endpoint types target support of cloud and enterprise based
  middleware and applications.
- Tagged atomic support
  Atomic operations can now target tagged receive buffers, in
  addition to RMA buffers.
- (Experimental) deferred work queues
  Deferred work queues are enhanced triggerred operations.  They
  target support for collective-based operations.
- New mode bits: FI_RESTRICTED_COMP and FI_NOTIFY_FLAGS_ONLY
  These mode bits support optimized completion processing to
  minimize software overhead.
- Multi-threaded error reporting
  Reading CQ and EQ errors now allow the application to provide the
  error buffer, eliminating the need for the application to synchronize
  between multiple threads when handling errors.
- FI_SOURCE_ERR capability
  This feature allows the provider to validate and report the source
  address for any received messages.
- FI_ADDR_STR string based addressing
  Applications can now request and use addresses provided using a
  standardized string format.  This makes it easier to pass full
  addressing data through a command line, or handle address exchange
  through text files.
- Communication scope capabilities: FI_LOCAL_COMM and FI_REMOTE_COMM
  Used to indicate if an application requires communication with
  peers on the same node and/or remote nodes.
- New memory registration modes
  The FI_BASIC_MR and FI_SCALABLE_MR memory registration modes have
  been replaced by more refined registration mode bits.  This allows
  applications to make better use of provider hardware capabilities
  when dealing with registered memory regions.
- New mode bit: FI_CONTEXT2
  Some providers need more than the size provided by the FI_CONTEXT
  mode bit setting.  To accomodate such providers, an FI_CONTEXT2
  mode bit was added.  This mode bit doubles the amount of context
  space that an application allocates on behalf of the provider.

## BGQ provider notes

- The OFI 1.5 BGQ provider officially supports the Open Fabrics Interfaces
  utilized by the MPICH implementation of the MPI-3 standard.  In addition
  to the MPICH test suite it has been tested by several scientific applications
  running MPICH on BGQ at scale, and several bugs in the provider and MPICH
  have been identified and fixed.  At least one INCITE project is attempting
  to use it for production science.  Support of this provider is a high-
  priority for ALCF, and MPICH users on BGQ are encouraged to utilize it to
  compare function and performance against the PAMI-based default toolchain.
  Any discovered bugs will be quickly addressed with high priority.  Results
  so far have shown significant point-to-point and RMA latency improvements
  over PAMI as well as RMA functional improvements at scale.  The only
  potential drawback is collective performance degradation against the PAMI
  optimizations, but at certain message and partition sizes, performance has
  been observed to be comparable or even better.

## MLX provider notes

- New provider to replace the deprecated mxm provider.
- Targets Mellanox InfiniBand fabrics, through the UCX library.
- Supports RDM endpoints with the tagged interfaces.
- Requires FI_CONTEXT mode support.
- See fi_mlx.7 man page for more details.

## NetDir provider nodes

- New provider for Windows that runs over the NetworkDirect API.
- Supports FI_EP_MSG endpoints, with FI_MSG and FI_RMA interfaces.
- Supports shared receive contexts
- Supports FI_SOCKADDR, FI_SOCKADDR_IN, and FI_SOCKADDR_IN6 addressing
- Asynchronous operations make forward progress automatically

## PSM provider notes

- Improve the name server functionality and move to the utility code
- Handle updated mr_mode definitions
- Add support of 32 and 64 bit atomic values

## PSM2 provider notes

- Add option to adjust the locking level
- Improve the name server functionality and move to the utility code
- Add support for string address format
- Add an environment vaiable for message inject size
- Handle FI_DISCARD in tagged receive functions
- Handle updated mr_mode definitions
- Add support for scalable endpoint
- Add support of 32 and 64 bit atomic values
- Add FI_SOURCE_ERR to the supported caps
- Improve the method of checking device existence

## Sockets provider notes

- Updated and enhanced atomic operation support.
- Add support for experimental deferred work queue operations.
- Fixed counter signaling when used with wait sets.
- Improved support on Windows.
- Cleaned up event reporting for destroyed endpoints.
- Fixed several possible crash scenarios.
- Fixed handling socket disconnect events which could hang the provider.

## RxM provider notes

- Add OFI RxM provider. It is an utility provider that supports RDM
  endpoints emulated over a base provider that supports only MSG end-
  points.
- The provider was earlier experimental. It's functional from this
  release onwards.
- Please refer to the man page of the provider for more info.

## UDP provider notes

- Add support for multicast data transfers

## usNIC provider notes

- Only requires libibverbs when necessary
- Updated to handle 1.5 interface changes.

## Verbs provider notes

- Fix an issue where if the user requests higher values for tx, rx
  context sizes than default it wasn't honored.
- Introduce env variables for setting default tx, rx context sizes
  and iov limits.
- Report correct completion ordering supported by MSG endpoints.


v1.4.2, Fri May 12, 2017
========================

## Core

- Fix for OS X clock_gettime() portability issue.

## PSM provider notes

- Updated default counter wait object for improved performance
- Fix multi-threaded RMA progress stalls

## PSM2 provider notes

- Updated default counter wait object for improved performance
- Fix multi-threaded RMA progress stalls

## Sockets provider notes

- Fix error in fi_cq_sreadfrom aborting before timeout expires
- Set atomic iov count correct correctly inside fi_atomicv

## Verbs provider notes

- Fix handling of apps that call fork.  Move ibv_fork_init() before
  calling any other verbs call.
- Fix crash in fi_write when connection is not yet established and
  write data size is below inline threshold.
- Fix issues not handling multiple ipoib interfaces
- Reduce lock contention on buffer pools in send/completion handling
  code.
- To see verbs provider in fi_info output, configure the corresponding
  IPoIB interface with an IP address. This is a change in behavior from
  previous versions. Please refer fi_verbs man page for more info.

v1.4.1, Fri Feb  3, 2017
========================

## PSM provider notes

- Defer initialization of the PSM library to allow runtime selection from
  different versions of the same provider before fi_getinfo is called.

## PSM2 provider notes

- Defer initialization of the PSM2 library to allow runtime selection from
  different versions of the same provider before fi_getinfo is called.
- General bug fixes.

## UDP provider notes

- Fix setting address format in fi_getinfo call.

## usNIC provider notes

- Fixed compilation issues with newer versions of libibverbs.

v1.4.0, Fri Oct 28, 2016
========================

- Add new options, `-f` and `-d`, to fi_info that can be used to
  specify hints about the fabric and domain name. Change port to `-P`
  and provider to `-p` to be more in line with fi_pingpong.

## GNI provider notes

- General bug fixes, plugged memory leaks, performance improvements,
  improved error handling and warning messages, etc.
- Additional API support:
  - FI_THREAD_COMPLETION
  - FI_RMA_EVENT
  - iov length up to 8 for messaging data transfers
- Provider-specific API support:
  - Aries native AXOR atomic operation
  - Memory registation cache flush operation
- Memory registration cache improvements:
  - IOMMU notifier support
  - Alternatives to the internal cache
  - Additional tuning knobs
- On-node optimization for rendezvous message communication via XPMEM
- Internal fixes to support accelerators and KNL processors
- Better support for running in CCM mode (in support for fabtests)

## MXM provider

- The mxm provider has been deprecated and will be replaced in a
  future release.

## PSM provider notes

- General bug fixes
- Use utility provider for EQ, wait object, and poll set
- Allow multi-recv to post buffer larger than message size limit

## PSM2 provider notes

- General bug fixes
- Add support for multi-iov RMA read and aromic operations
- Allow multi-recv to post buffer larger than message size limit

## Sockets provider notes

- General code cleanup and bug fixes
- Set tx/rx op_flags correctly to be consistent with manpage
- Restructure struct sock_ep to support alias ep
- Refactor CQ/Cntr bindings, CQ completion generation, and counter
  increments
- Copy compare data to internal buffer when FI_INJECT is set in
  fi_compare_atomic
- Correctly handle triggered operation when FI_INJECT is set or
  triggered op is enqueued or counter is incremented. Initialize
  counter threshold to INT_MAX
- Refactor and cleanup connection management code, add locks to avoid
  race between main thread and progress thread, add logic to correctly
  handle FI_SHUTDOWN and FI_REJECT
- Set fabric name as network address in the format of a.b.c.d/e and
  domain name as network interface name
- Remove sock_compare_addr and add two utility functions ofi_equals_ipaddr
  and ofi_equals_sockaddr in fi.h
- Refactor fi_getinfo to handle corner cases and add logic if a given
  src_addr matches to any local interface addr
- Restructure acquiring/releasing the list_lock in progress thread so
  that it is only acquired once per iteration
- Refactor connection management of MSG ep so that it uses TCP instead
  of UDP for connection management msg and new port for every MSG
  endpoint
- Add sock_cq_sanitize_flags function to make sure only flags returned in
  CQ events are the ones that are listed on the manpage
- Update fi_poll semantics for counters so that it returns success if the
  counter value is different from the last-read-value
- Allow multiple threads to wait on one counter
- Update code to use ofi_util_mr - the new MR structure added to util code
- Fix fi_av_insert not to report error when the number of inserted addr
  exceeds the count attribute in fi_av_attr
- Add garbage collection of AV indices after fi_av_remove, add ep list
  in AV and cleanup conn map during fi_av_remove
- Use correct fi_tx_attr/fi_rx_attr for scalable ep

## UDP provider notes

- Enhance parameter checks for several function calls.
- Fix memory leak freeing CQ structure.
- Bind to a source address when enabling endpoint.
- Reduce reported resource limits (domain attributes).

## usNIC provider notes

- Fix handling of EP_MSG peers on different IP subnets [PR #1988]
- Fix handling of CM data. Fixes a bug where data received would
  overwrite parts of the connection management structure [PR #1991]
- Fix bug in CM connect/accept handling that would cause a seg fault
  if data was sent as part of a connection request [PR #1991]
- Fix invalid completion lengths in the MSG and RDM endpoint
  implementations of fi_recvv and fi_recvmsg [PR #2026]
- Implement the FI_CM_DATA_SIZE option for fi_getopt on passive
  endpoints [PR #2033]
- Add fi_reject implementation that supports data exchange [PR #2038]
- Fix fi_av_straddr bug that reported port in network order [PR #2244]
- Report -FI_EOPBADSTATE if the size left functions are used on an
  endpoint which has not been enabled [PR #2266]
- Change the domain/fabric naming. The fabric is now represented as
  the network address in the form of a.b.c.d/e and the domain name is
  the usNIC device name. For more information see fi_usnic(7) [PR
  #2287]
- Fix the domain name matching in fi_getinfo/fi_domain [PR #2298]
- Fix issue with AV where it is fully closed before pending
  asynchronous inserts can finish leading to invalid data accesses [PR
  #2397]
- Free all data associated with AV when fi_av_close is called [PR
  #2397]
- Fail with -FI_EINVAL if a value of FI_ADDR_NOTAVAIL is given to
  fi_av_lookup.  [PR #2397]
- Verify AV attributes and return an error if anything that is
  unsupported is requested (FI_AV_TABLE, named AVs, FI_READ, etc.) [PR
  #2397]

## Verbs provider notes

- Add fork support. It is enabled by default and can be turned off by
  setting the FI_FORK_UNSAFE variable to "yes". This can improve
  performance of memory registrations but also makes fork unsafe. The
  following are the limitations of fork support:
  - Fabric resources like endpoint, CQ, EQ, etc. should not be used in
    the forked process.
  - The memory registered using fi_mr_reg has to be page aligned since
    ibv_reg_mr marks the entire page that a memory region belongs to
    as not to be re-mapped when the process is forked (MADV_DONTFORK).
- Fix a bug where source address info was not being returned in
  fi_info when destination node is specified.

- verbs/MSG
  - Add fi_getopt for passive endpoints.
  - Add support for shared RX contexts.
- verbs/RDM
  - General bug fixes
  - Add FI_MSG capability
  - Add FI_PEEK and FI_CLAIM flags support
  - Add completion flags support
  - Add selective completion support
  - Add fi_cq_readerr support
  - Add possibility to set IPoIB network interface via FI_VERBS_IFACE
    environment variable
  - Add large data transfer support (> 1 GB)
  - Add FI_AV_TABLE support
  - Add fi_cntr support
  - Add environment variables for the provider tuning:
    FI_VERBS_RDM_BUFFER_NUM, FI_VERBS_RDM_BUFFER_SIZE,
    FI_VERBS_RDM_RNDV_SEG_SIZE, FI_VERBS_RDM_CQREAD_BUNCH_SIZE,
    FI_VERBS_RDM_THREAD_TIMEOUT, FI_VERBS_RDM_EAGER_SEND_OPCODE
  - Add iWarp support

v1.3.0, Mon Apr 11, 2016
========================

## General notes

* [See a list of provider features for this
  release](https://github.com/ofiwg/libfabric/wiki/Provider-Feature-Matrix-v1.3.0)

## GNI provider notes

- CLE 5.2UP04 required for building GNI provider
- General bug fixes, plugged memory leaks, etc.
- Improved error handling, warning messages, etc.
- Added support for the following APIs:
  - fi_endpoint: fi_getopt, fi_setopt, fi_rx_size_left, fi_tx_size_left, fi_stx_context
  - fi_cq: fi_sread, fi_sreadfrom
  - fi_msg: FI_MULTI_RECV (flag)
  - fi_domain: FI_PROGRESS_AUTO (flag)
  - fi_direct: FI_DIRECT
- Added support for FI_EP_DGRAM (datagram endpoint):
  - fi_msg, fi_tagged, fi_rma
- Memory registration improvements:
  - Improved performance
  - Additional domain open_ops
- Initial support for Cray Cluster Compatibility Mode (CCM)
- Implemented strict API checking
- Added hash list implementation for tag matching (available by domain open_ops)

Note: The current version of fabtests does not work with the GNI
provider due to the job launch mechanism on Cray XC systems.  Please
see the [GNI provider
wiki](https://github.com/ofi-cray/libfabric-cray/wiki) for
alternatives to validating your installation.

## MXM provider notes

- Initial release

## PSM provider notes

- Remove PSM2 related code.

## PSM2 provider notes

- Add support for multi-iov send, tagged send, and RMA write.
- Use utility provider for EQ, wait object, and poll set.

## Sockets provider notes

- General code cleanup
- Enable FABRIC_DIRECT
- Enable sockets-provider to run on FreeBSD
- Add support for fi_trywait
- Add support for map_addr in shared-av creation
- Add shared-av support on OSX
- Allow FI_AV_UNSPEC type during av_open
- Use loop-back address as source address if gethostname fails
- Disable control-msg ack for inject operations that do not expect completions
- Increase max_atomic_msg_size to 4096 bytes
- Remove check for cq_size availability while calculating tx/rx_size_left
- Use util-buffer pool for overflow entries in progress engine.
- Synchronize accesses to memory-registration operations
- Fix an issue that caused out-of-order arrival of messages
- Fix a bug in processing RMA access error
- Fix a bug that caused starvation in processing receive operations
- Add reference counting for pollset
- Fix a bug in connection port assignment

## UDP provider notes

- Initial release

## usNIC provider notes

- Implement fi_recvv and fi_recvmsg for FI_EP_RDM. [PR #1594]
- Add support for FI_INJECT flag in the FI_EP_RDM implementation of fi_sendv.
  [PR #1594]
- Fix crashes that occur in the FI_EP_RDM and the FI_EP_MSG implementations
  when messages are posted with the maximum IOV count.  [PR #1784]
- Fix crashes that occur in the FI_EP_RDM and the FI_EP_MSG implementations
  when posting messages with IOVs of varying lengths.  [PR #1784]
- Handle FI_PEEK flag in fi_eq_sread. [PR #1758]
- Return -FI_ENOSYS if a named AV is requested. [PR #1749]
- The ethernet header does not count against the MTU. Update reported
  max_msg_size when using FI_EP_DGRAM to reflect this. [PR #1738]
- Set the DF (do not fragment) bit in the IP header. [PR #1665]
- Fix crashes that may occur from improper handling of receive state tracking
  [PR #1809]
- Fortify the receive side of libnl communication [PR #1655]
- Fix handling of fi_info with passive endpoints. Connections opened on a
  passive endpoint now inherit the properties of the fi_info struct used to
  open the passive endpoint. [PR #1806]
- Implement pollsets. [PR #1835]
- Add version 2 of the usnic getinfo extension [PR #1866]
- Implement waitsets [PR #1893]
- Implement fi_trywait [PR #1893]
- Fix progress thread deadlock [PR #1893]
- Implement FD based CQ sread [PR #1893]

## Verbs provider notes

- Add support for fi_trywait
- Support building on OSes which have older versions of librdmacm (v1.0.16 or
  lesser). The functionality of the provider when the user passes AF_IB
  addresses is not guaranteed though.
- Added a workaround to support posting more than 'verbs send work queue length'
  number of fi_inject calls at a time.
- Make CQ reads thread safe.
- Support the case where the user creates only a send or recv queue for the
  endpoint.
- Fix an issue where RMA reads were not working on iWARP cards.
- verbs/RDM
  - Add support for RMA operations.
  - Add support for fi_cq_sread and fi_cq_sreadfrom
  - Rework connection management to make it work with fabtests and also allow
    connection to self.
  - Other bug fixes and performance improvements.

v1.2.0, Thu Jan 7, 2016
=======================

## General notes

- Added GNI provider
- Added PSM2 provider

## GNI provider notes
- Initial release

## PSM provider notes
- General bug fixes
- Support auto progress mode
- Support more threading modes
- Only set FI_CONTEXT mode if FI_TAGGED or FI_MSG is used
- Support Intel Omni-Path Fabric via the psm2-compat library

## PSM2 provider notes
- Initial addition

## Sockets provider notes

- General bug fixes and code cleanup
- Update memory registration to support 32-bit builds and fix build warnings
- Initiate conn-msg on the same tx_ctx as the tx operation for scalable ep
- Fix av mask calculation for scalable ep
- Mask out context-id during connection lookup for scalable ep
- Increase buffered receive limit
- Ignore FI_INJECT flag for atomic read operation
- Return -FI_EINVAL instead of -FI_ENODATA for fi_endpoint for invalid attributes
- Set default tag format to FI_TAG_GENERIC
- Set src/dest iov len correctly for readv operations
- Fix random crashes while closing shared contexts
- Fix an out of bound access when large multi-recv limit is specified by user
- Reset tag field in CQ entry for send completion
- Do not set prov_name in fabric_attr
- Validate flags in CQ/Cntr bind operations
- Scalability enhancements
- Increase mr_key size to 64 bit
- Use red-black tree for mr_key lookup

## usNIC provider notes
- The usNIC provider does not yet support asynchronous memory registration.
  Return -FI_EOPNOTSUPP if an event queue is bound to a domain with FI_REG_MR.
- Set fi_usnic_info::ui_version correctly in calls to
  fi_usnic_ops_fabric::getinfo().
- Improve fi_cq_sread performance.
- Return -FI_EINVAL from av_open when given invalid paramters.
- Fix bug in fi_av_remove that could lead to a seg fault.
- Implement fi_av_insertsvc.
- Report FI_PROTO_RUDP as protocol for EP_RDM.

## Verbs provider notes

- Add support for RDM EPs. Currently only FI_TAGGED capability is supported.
  RDM and MSG EPs would be reported in seperate domains since they don't share
  CQs. The RDM enpoint feature is currently experimental and no guarantees are
  given with regard to its functionality.
- Refactor the code into several files to enable adding RDM support.
- Consolidate send code paths to improve maintainability.
- Fix a bug in fi_getinfo where wild card address was not used when service
  argument is given.
- Fix fi_getinfo to always return -FI_ENODATA in case of failure.
- Add support for fi_eq_write.
- Other misc bug fixes.

v1.1.1, Fri Oct 2, 2015
=======================

## General notes

## PSM provider notes

- General bug fixes
- Proper termination of the name server thread
- Add UUID and PSM epid to debug output
- Add environment variable to control psm_ep_close timeout
- Code refactoring of AM-based messaging
- Check more fields of the hints passed to fi_getinfo
- Generate error CQ entries for empty result of recv with FI_SEEK flag
- Correctly handle overlapped local buffers in atomics
- Handle duplicated addresses in fi_av_insert
- Fix the return value of fi_cq_readerr
- Call AM progress function only when AM is used
- Detect MPI runs and turns off name server thread automatically

## Sockets provider notes

- General clean-up and restructuring
- Add fallback mechanism for getting source address
- Fix fi_getinfo to use user provided capabilities from hints
- Fix hostname and port number and added checks in sock_av_insertsym
- Add retry for connection timeout
- Release av resources in the error path
- Remove separate read/write CQ to be consistent with the man page
- Increase default connection map size and added environment variable to specify
  AV, CQ, EQ and connection map size to run large scale tests
- Fix FI_PEEK operation to be consistent with the man page
- Fix remote write event not to generate CQ event
- Fix CSWAP operation to return initial value
- Use size_t for min_multi_recv and buffered_len
- Set address size correctly in fi_getname/fi_getpeer

## usNIC provider notes

- Fix EP_RDM reassembly issue for large messages
- Return correct number of read completions on error
- Fix EP_RDM and EP_MSG data corruption issue when packets are actually
  corrupted on the wire
- Fix EP_RDM and EP_MSG fi_tx_size_left/fi_rx_size_left functions

## Verbs provider notes

- Add more logging for errors
- Bug fixes

v1.1.0, Wed Aug 5, 2015
=======================

## General notes

- Added fi_info utility tool
- Added unified global environment variable support
- Fixed configure issues with the clang/llvm compiler suite

## PSM provider notes

- General bug fixes
- Move processing of triggered ops outside of AM handlers
- Generate CQ entries for cancelled operations
- Remove environment variable FI_PSM_VERSION_CHECK
- Fix multi-recv completion generation
- Environment variable tweaks

## Sockets provider notes

- General bug fixes and code cleanup
- Add triggered operation suppport
- Generate error completion event for successful fi_cancel
- Support fi_cancel for tx operations
- Enable option for setting affinity to progress thread
- Improve error handling during connection management
- Avoid reverse lookup for every received message
- Avoid polling all connections while checking for incoming message
- Use fast_lock for progress engine's list_lock
- Handle disconnected sockets
- Add rx entry pool
- Mark tx entry as completed only if data is sent out to wire
- Add rx control context for every tx context for progressing control messages
- Set source address when addressing information is not passed by the application
- Reset return value after polling CQ ring buffer
- Reset FI_TRIGGER flag while triggering triggered operations
- Ensure progress of control context

## usNIC provider notes

- General bug fixes
- Add support for fi_getname/fi_setname, fi_cq_sread
- Change FI_PREFIX behavior per fi_getinfo(3)
- Fix to report correct lengths in all completions
- Support fi_inject() with FI_PREFIX
- Properly support iov_limit
- Support FI_MORE
- Fixed fi_tx_size_left() and fi_rx_size_left() usage
- Fixed obscure error when posting cq_size operations without reading
  a completion

## Verbs provider notes

- AF_IB addreses can now be passed as node argument to fi_getinfo
- Added support for fi_setname and migrating passive EP to active EP
- Detect and report multiple verbs devices if present
- Bug fixes

v1.0.0, Sun May 3, 2015
=======================

Initial public release, including the following providers:

- PSM
- Sockets
- usNIC
- Verbs
