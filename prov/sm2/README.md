SM2 is an experimental provider with limited capabilities.  Its purpose is to see if a NEMESIS style (lockless FIFO Queue) is required to solve SHM's performance issues.

SM2 does not currently support:
1. HMEM messages over 4k (claims no support for HMEM)
2. SAR protocol (CMA is required to send messages over INJECT_SIZE)
2. FI_ATOMIC
3. FI_RMA
5. AV_USER_ID - We always write the peer ID (int) instead of the peers addr (string)
6. FI_ADDR_NOTAVAIL - The provider will return the peers actual ID even if the receiver didn't call fi_av_insert() on the peer which sent the message