/*
 * Copyright (c) 2026 MThreads.
 *
 * This software is available to you under the BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or without
 *     modification, are permitted provided that the following conditions are
 *     met:
 *
 *      - Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <assert.h>

#include "hmem.h"

int main(void)
{
	struct fi_mr_attr attr = {0};

	assert(FI_HMEM_MUSA == FI_HMEM_SYNAPSEAI + 1);

	attr.iface = FI_HMEM_MUSA;
	attr.device.musa = 3;
	assert(attr.device.reserved == 3);

#if !HAVE_MUSA
	/* Disabled backends must link without libfabric's internal HMEM API. */
	if (ft_musa_init() != -FI_ENOSYS ||
	    ft_musa_cleanup() != -FI_ENOSYS ||
	    ft_musa_alloc(0, NULL, 0) != -FI_ENOSYS ||
	    ft_musa_alloc_host(NULL, 0) != -FI_ENOSYS ||
	    ft_musa_free(NULL) != -FI_ENOSYS ||
	    ft_musa_free_host(NULL) != -FI_ENOSYS ||
	    ft_musa_memset(0, NULL, 0, 0) != -FI_ENOSYS ||
	    ft_musa_copy_to_hmem(0, NULL, NULL, 0) != -FI_ENOSYS ||
	    ft_musa_copy_from_hmem(0, NULL, NULL, 0) != -FI_ENOSYS ||
	    ft_musa_get_dmabuf_fd(NULL, 0, NULL, NULL) != -FI_ENOSYS ||
	    ft_musa_put_dmabuf_fd(-1) != -FI_ENOSYS)
		return 1;
#endif

	return 0;
}
