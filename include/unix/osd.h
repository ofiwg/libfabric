/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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

#ifndef _FI_UNIX_OSD_H_
#define _FI_UNIX_OSD_H_

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define FI_DESTRUCTOR(func) static __attribute__((destructor)) void func

struct util_shm
{
	int		shared_fd;
	void		*ptr;
	const char	*name;
	size_t		size;
};

static inline int ofi_memalign(void **memptr, size_t alignment, size_t size)
{
	return posix_memalign(memptr, alignment, size);
}

static inline void ofi_freealign(void *memptr)
{
	free(memptr);
}

static inline ssize_t ofi_read_socket(int fd, void *buf, size_t count)
{
	return read(fd, buf, count);
}

static inline ssize_t ofi_write_socket(int fd, const void *buf, size_t count)
{
	return write(fd, buf, count);
}

static inline int ofi_close_socket(int socket)
{
	return close(socket);
}

static inline int ofi_sockerr(void)
{
	return errno;
}

static inline void ofi_osd_init(void)
{
}

static inline void ofi_osd_fini(void)
{
}

#endif /* _FI_UNIX_OSD_H_ */

