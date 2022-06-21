/*
 * (C) Copyright 2022 Oak Ridge National Lab
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

#ifndef OFI_XPMEM_H
#define OFI_XPMEM_H

#if HAVE_XPMEM
#include <xpmem.h>

struct xpmem_client {
	uint8_t cap;
	xpmem_apid_t apid;
	uintptr_t addr_max;
};

struct xpmem_pinfo {
	/* XPMEM segment id for this process */
	xpmem_segid_t seg_id;
	/* maximum attachment address for this process. attempts to attach
	 * past this value may fail. */
	uintptr_t address_max;
};

struct xpmem {
	struct xpmem_pinfo pinfo;
	/* maximum size that will be used with a single memcpy call.
	 * On some systems we see better peformance if we chunk the
	 * copy into multiple memcpy calls. */
	uint64_t memcpy_chunk_size;
};

extern struct xpmem *xpmem;

#endif /* HAVE_XPMEM */

#endif /* OFI_XPMEM_H */
