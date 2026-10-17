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

#include "ofi_hmem.h"

int ft_default_alloc_host(void **buf, size_t size);
int ft_default_free_host(void *buf);

int ft_musa_init(void)
{
	ofi_hmem_init();
	return ofi_hmem_is_initialized(FI_HMEM_MUSA) ? FI_SUCCESS : -FI_ENOSYS;
}

int ft_musa_cleanup(void)
{
	return FI_SUCCESS;
}

int ft_musa_alloc(uint64_t device, void **buf, size_t size)
{
	return ofi_hmem_dev_alloc(FI_HMEM_MUSA, device, size, 0, 0, buf,
				   NULL, NULL);
}

int ft_musa_alloc_host(void **buf, size_t size)
{
	return ft_default_alloc_host(buf, size);
}

int ft_musa_free(void *buf)
{
	ofi_hmem_dev_free(FI_HMEM_MUSA, 0, buf);
	return FI_SUCCESS;
}

int ft_musa_free_host(void *buf)
{
	return ft_default_free_host(buf);
}

int ft_musa_memset(uint64_t device, void *buf, int value, size_t size)
{
	void *host_buf;
	int ret;

	if (!size)
		return FI_SUCCESS;

	host_buf = malloc(size);
	if (!host_buf)
		return -FI_ENOMEM;
	memset(host_buf, value, size);
	ret = ofi_copy_to_hmem(FI_HMEM_MUSA, device, buf, host_buf, size);
	free(host_buf);
	return ret;
}

int ft_musa_copy_to_hmem(uint64_t device, void *dst, const void *src,
			 size_t size)
{
	return ofi_copy_to_hmem(FI_HMEM_MUSA, device, dst, src, size);
}

int ft_musa_copy_from_hmem(uint64_t device, void *dst, const void *src,
			   size_t size)
{
	return ofi_copy_from_hmem(FI_HMEM_MUSA, device, dst, src, size);
}

int ft_musa_get_dmabuf_fd(void *buf, size_t len, int *fd, uint64_t *offset)
{
	(void) buf;
	(void) len;
	(void) fd;
	(void) offset;
	return -FI_ENOSYS;
}

int ft_musa_put_dmabuf_fd(int fd)
{
	(void) fd;
	return -FI_ENOSYS;
}
