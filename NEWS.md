Libfabric release notes
=======================

This file contains the main features as well as overviews of specific
bug fixes (and other actions) for each version of Libfabric since
version 1.0.

v1.1.1, TBD
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

## usNIC provider notes

- Fix EP_RDM reassembly issue for large messages
- Return correct number of read completions on error
- Fix EP_RDM and EP_MSG data corruption issue when packets are actually
  corrupted on the wire
- Fix EP_RDM and EP_MSG fi_tx_size_left/fi_rx_size_left functions

## Verbs provider notes


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


v1.0.0, Sun May 3, 2015
=======================

Initial public release, including the following providers:

- PSM
- Sockets
- usNIC
- Verbs
