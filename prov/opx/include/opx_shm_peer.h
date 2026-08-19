/*
 * Copyright (C) 2026 Cornelis Networks.
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

#ifndef _FI_PROV_OPX_SHM_PEER_H_
#define _FI_PROV_OPX_SHM_PEER_H_

/*
 * opx_shm_peer - build the fi_info used to open the shm provider as an OPX
 * FI_PEER peer for intra-node communication.
 *
 * OPX is the FI_PEER "owner": it drives the shm provider (the "peer") through a
 * shared receive queue so node-local traffic is handled by shm while the OPX
 * fabric path handles remote peers.  This mirrors the pattern the efa provider
 * uses to import shm/sm2 (see prov/efa/src/efa_shm.c).
 */

#include <rdma/fabric.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Build a shm-provider fi_info derived from the application's info.  On success
 * *shm_info points to an allocated fi_info the caller must fi_freeinfo(); on any
 * failure (or when shm is not applicable) *shm_info is set to NULL, which the
 * caller treats as "run without an intra-node shm peer".
 */
void opx_shm_info_create(const struct fi_info *app_info, struct fi_info **shm_info);

#ifdef __cplusplus
}
#endif

#endif /* _FI_PROV_OPX_SHM_PEER_H_ */
