/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <ofi_util.h>

#ifdef HAVE_LINUX_USERFAULTFD_H
#include <config.h>
#include <ofi_uffd.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/userfaultfd.h>

#define MB(x)((size_t)(x) << 20)
#define GB(x)((size_t)(x) << 30)
#define ARRAY_SIZE(A) (sizeof(A)/sizeof(*A))

/* See TODO in function ofi_uffd_register */
static long page_sizes[] = {4096, MB(2), GB(1)};

static inline void set_uffdio_range(struct uffdio_range *range,
				    void *start, size_t len,
				    long page_size)
{
	range->start = ((uint64_t)start / page_size) * page_size;
	range->len = ofi_div_ceil(len, page_size) * page_size;
}

/*
 * TODO: ioctl register needs to be passed page aligned boundaries.
 * Need to find a more sane way of detecting a region's page size.
 */
int ofi_uffd_register(int uffd, void *start, size_t len)
{
	struct uffdio_register uffdio_register;
	long page_size;
	int i;

	/*
	 * Since the ioctl functions require page aligned boundaries,
	 * we repeatedly call ioctl using different page sizes until
	 * it succeeds or we run out of page sizes to try.
	 */
	for (i = 0; i < ARRAY_SIZE(page_sizes); i++) {
		page_size = page_sizes[i];
		set_uffdio_range(&uffdio_register.range, start, len, page_size);
		uffdio_register.mode = UFFDIO_REGISTER_MODE_MISSING;

		if (ioctl(uffd, UFFDIO_REGISTER, &uffdio_register) != -1)
			return 0;

		if (errno != EINVAL) {
			FI_WARN(&core_prov, FI_LOG_MR, "ioctl/uffdio_register: %s\n",
				strerror(errno));
			break;
		}
	}

	return -errno;
}

int ofi_uffd_unregister(int uffd, void *start, size_t len)
{
	struct uffdio_range uffdio_range;
	long page_size;
	int i;

	for (i = 0; i < ARRAY_SIZE(page_sizes); i++) {
		page_size = page_sizes[i];
		set_uffdio_range(&uffdio_range, start, len, page_size);
		if (ioctl(uffd, UFFDIO_UNREGISTER, &uffdio_range) != -1)
			return 0;

		if (errno != EINVAL) {
			FI_WARN(&core_prov, FI_LOG_MR,
				"ioctl/uffdio_unregister: %s\n",
				strerror(errno));
			break;
		}
	}

	return -errno;
}

void ofi_uffd_close(int uffd)
{
	close(uffd);
}

int ofi_uffd_init(int flags, uint64_t features)
{
	struct uffdio_api uffdio_api;
	int uffd;

	page_sizes[0] = get_page_size();
	if (page_sizes[0] < 0)
		return page_sizes[0];

	uffd = (int)syscall(__NR_userfaultfd, flags);
	if (OFI_UNLIKELY(uffd == -1)) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"syscall/userfaultfd: %s\n", strerror(errno));
		return -errno;
	}

	uffdio_api.api = UFFD_API;
	uffdio_api.features = features;
	if (ioctl(uffd, UFFDIO_API, &uffdio_api) == -1) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"ioctl/uffdio_api: %s\n", strerror(errno));
		ofi_uffd_close(uffd);
		return -errno;
	}

	if (uffdio_api.api != UFFD_API) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"uffdio requested features not supported.\n");
		ofi_uffd_close(uffd);
		return -FI_ENOSYS;
	}

	return uffd;
}

#else /* !HAVE_LINUX_USERFAULTFD_H */

int ofi_uffd_register(int uffd, void *start, size_t len)
{
	OFI_UNUSED(uffd);
	OFI_UNUSED(start);
	OFI_UNUSED(len);

	return -FI_ENOSYS;
}

int ofi_uffd_unregister(int uffd, void *start, size_t len)
{
	OFI_UNUSED(uffd);
	OFI_UNUSED(start);
	OFI_UNUSED(len);

	return -FI_ENOSYS;
}

void ofi_uffd_close(int uffd)
{
	OFI_UNUSED(uffd);
}

int ofi_uffd_init(int flags, uint64_t features)
{
	OFI_UNUSED(flags);
	OFI_UNUSED(features);

	return -FI_ENOSYS;
}

#endif /* HAVE_LINUX_USERFAULTFD_H */
