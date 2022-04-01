/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "ofi_hmem.h"
#include "ofi.h"

int synapseai_init(void)
{
	return -FI_ENOSYS;
}

int synapseai_cleanup(void)
{
	return -FI_ENOSYS;
}

int synapseai_copy_to_hmem(uint64_t device, void *dest, const void *src,
                           size_t size)
{
	return -FI_ENOSYS;
}

int synapseai_copy_from_hmem(uint64_t device, void *dest, const void *src,
                             size_t size)
{
	return -FI_ENOSYS;
}

bool synapseai_is_addr_valid(const void *addr, uint64_t *device,
                             uint64_t *flags)
{
	return false;
}

int synapseai_get_handle(void *dev_buf, void **handle)
{
	return -FI_ENOSYS;
}

int synapseai_open_handle(void **handle, uint64_t device, void **ipc_ptr)
{
	return -FI_ENOSYS;
}

int synapseai_close_handle(void *ipc_ptr)
{
	return -FI_ENOSYS;
}

int synapseai_host_register(void *ptr, size_t size)
{
	return -FI_ENOSYS;
}

int synapseai_host_unregister(void *ptr)
{
	return -FI_ENOSYS;
}

int synapseai_get_base_addr(const void *ptr, void **base, size_t *size)
{
	return -FI_ENOSYS;
}

bool synapseai_is_ipc_enabled(void)
{
	return false;
}
