/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
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

#ifndef _FI_SHM_H_
#define _FI_SHM_H_

#include "config.h"

#include <stdint.h>
#include <stddef.h>

#include <fi_atom.h>
#include <fi_proto.h>
#include <fi_mem.h>
#include <fi_rbuf.h>


#ifdef __cplusplus
extern "C" {
#endif


#define SMR_VERSION	1

#ifdef HAVE_ATOMICS
#define SMR_FLAG_ATOMIC	(1 << 0)
#else
#define SMR_FLAG_ATOMIC	(0 << 0)
#endif

#define SMR_FLAG_DEBUG	(1 << 1)

enum {
	SMR_INJECT_SIZE = 4096
};

struct shm_region {
	uint8_t		version;
	uint8_t		resv;
	uint16_t	flags;
	int		pid;
	atomic_t	lock;
	int		peer_size;

	size_t		total_size;
	size_t		rx_cmd_offset;
	size_t		rx_ctrl_offset;
	size_t		inject_buf_offset;

	struct shm_region *peer[];
};

struct smr_req {
	void		*context;
	void		*buffer;
	uint64_t	flags;
};

struct smr_resp {
	uint32_t	req_id;
	uint32_t	status;
};

struct smr_inject_buf {
	uint8_t		data[SMX_INJECT_SIZE];
};

DECLARE_CIRQUE(struct smr_cmd, smr_rx_cmdq);
DECLARE_CIRQUE(struct smr_resp, smr_rx_ctrlq);
DECLARE_FREESTACK(struct smr_req, smr_tx_ctx);
DECLARE_FREESTACK(struct smr_inject_buf, smr_buf_pool);

struct smr_attr {
	char		*name;
	size_t		cmd_size;
	size_t		cmd_count;
	size_t		ctrl_size;
	size_t		ctrl_count;
	size_t		tx_size;
	size_t		tx_count;
	size_t		inject_size;
	size_t		inject_count;
	size_t		peer_count;
};

int smr_create(const struct smr_attr *attr, struct shm_region **smr);
int smr_connect(struct shm_region *smr, const char *name, int *id);
void smr_free(struct shm_region *smr);


#ifdef __cplusplus
}
#endif

#endif /* _FI_SHM_H_ */
