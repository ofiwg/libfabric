Libfabric release notes
=======================

This file contains the main features as well as overviews of specific
bug fixes (and other actions) for each version of Libfabric since
version 1.0.

v1.4.0, TBD
===========

- Add new options, `-f` and `-d`, to fi_info that can be used to specify hints
  about the fabric and domain name. Change port to `-P` and provider to `-p` to
  be more in line with fi_pingpong.

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
