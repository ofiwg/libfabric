/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
 * All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef EFA_RDM_ERROR_H
#define EFA_RDM_ERROR_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "efa_errno.h"
#include "efa_rdm_peer.h"

#define HOST_ID_STR_LENGTH 19

/**
 * @brief Write the error message and return its byte length
 * @param[in]    ep          RXR endpoint
 * @param[in]    addr        Remote peer fi_addr_t
 * @param[in]    err         FI_* error code(must be positive)
 * @param[in]    prov_errno  EFA provider * error code(must be positive)
 * @param[out]   buf         Pointer to the address of error data written by this function
 * @param[out]   buflen      Pointer to the returned error data size
 * @return       A status code. 0 if the error data was written successfully, otherwise a negative FI error code.
 */
static inline int efa_rdm_error_write_msg(struct rxr_ep *ep, fi_addr_t addr, int err, int prov_errno, void **buf, size_t *buflen)
{
    char ep_addr_str[OFI_ADDRSTRLEN] = {0}, peer_addr_str[OFI_ADDRSTRLEN] = {0};
    char local_host_id_str[HOST_ID_STR_LENGTH + 1] = {0}, peer_host_id_str[HOST_ID_STR_LENGTH + 1] = {0};
    const char *base_msg = efa_strerror(prov_errno, NULL);
    size_t len = 0;
    struct efa_rdm_peer *peer = rxr_ep_get_peer(ep, addr);

    *buf = NULL;
    *buflen = 0;

    len = sizeof(ep_addr_str);
    rxr_ep_raw_addr_str(ep, ep_addr_str, &len);
    len = sizeof(peer_addr_str);
    rxr_ep_get_peer_raw_addr_str(ep, addr, peer_addr_str, &len);

    if (!ep->host_id || HOST_ID_STR_LENGTH != snprintf(local_host_id_str, HOST_ID_STR_LENGTH + 1, "i-%017lx", ep->host_id)) {
        strcpy(local_host_id_str, "N/A");
    }

    if (!peer->host_id || HOST_ID_STR_LENGTH != snprintf(peer_host_id_str, HOST_ID_STR_LENGTH + 1, "i-%017lx", peer->host_id)) {
        strcpy(peer_host_id_str, "N/A");
    }

    int ret = snprintf(ep->err_msg, RXR_ERROR_MSG_BUFFER_LENGTH, "%s My EFA addr: %s My host id: %s Peer EFA addr: %s Peer host id: %s",
                       base_msg, ep_addr_str, local_host_id_str, peer_addr_str, peer_host_id_str);

    if (ret < 0 || ret > RXR_ERROR_MSG_BUFFER_LENGTH - 1) {
        return -FI_EINVAL;
    }

    if (strlen(ep->err_msg) >= RXR_ERROR_MSG_BUFFER_LENGTH) {
        return -FI_ENOBUFS;
    }

    *buf = ep->err_msg;
    *buflen = RXR_ERROR_MSG_BUFFER_LENGTH;

    return 0;
}

#endif /* EFA_RDM_ERROR_H */
