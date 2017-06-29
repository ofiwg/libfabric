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

#include "config.h"

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <complex.h>
#include <sys/socket.h>
#include <netinet/in.h>

/* MSG_NOSIGNAL doesn't exist on OS X */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#ifndef SOCKET
#define SOCKET int
#endif

#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif


#define FI_DESTRUCTOR(func) static __attribute__((destructor)) void func

#ifndef UNREFERENCED_PARAMETER
#define OFI_UNUSED(var) (void)var
#else
#define OFI_UNUSED UNREFERENCED_PARAMETER
#endif

#define OFI_SOCK_TRY_RCV_AGAIN(err)			\
	((err) == EAGAIN || (err) == EWOULDBLOCK)

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

static inline void ofi_osd_init(void)
{
}

static inline void ofi_osd_fini(void)
{
}

static inline SOCKET ofi_socket(int domain, int type, int protocol)
{
	return socket(domain, type, protocol);
}

static inline ssize_t ofi_read_socket(SOCKET fd, void *buf, size_t count)
{
	return read(fd, buf, count);
}

static inline ssize_t ofi_write_socket(SOCKET fd, const void *buf, size_t count)
{
	return write(fd, buf, count);
}

static inline ssize_t ofi_send_socket(SOCKET fd, const void *buf, size_t count,
				      int flags)
{
	return send(fd, buf, count, flags);
}

static inline int ofi_close_socket(SOCKET socket)
{
	return close(socket);
}

static inline int ofi_sockerr(void)
{
	return errno;
}

static inline int ofi_sysconf(int name)
{
	return sysconf(name);
}

/* OSX has no such definition. So, add it manually */
#ifndef s6_addr32
#define s6_addr32 __u6_addr.__u6_addr32
#endif /* s6_addr32 */

static inline int ofi_is_loopback_addr(struct sockaddr *addr) {
	return (addr->sa_family == AF_INET &&
		((struct sockaddr_in *)addr)->sin_addr.s_addr == ntohl(INADDR_LOOPBACK)) ||
		(addr->sa_family == AF_INET6 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.s6_addr32[0] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.s6_addr32[1] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.s6_addr32[2] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.s6_addr32[3] == ntohl(1));
}


/* complex operations implementation */

typedef float complex ofi_complex_float;
typedef double complex ofi_complex_double;
typedef long double complex ofi_complex_long_double;

#define OFI_DEF_COMPLEX_OPS(type)				\
static inline int ofi_complex_eq_## type			\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return a == b;						\
}								\
static inline ofi_complex_## type ofi_complex_sum_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return a + b;						\
}								\
static inline ofi_complex_## type ofi_complex_prod_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return a * b;						\
}								\
static inline ofi_complex_## type ofi_complex_land_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return a && b;      					\
}								\
static inline ofi_complex_## type ofi_complex_lor_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return a || b;						\
}								\
static inline int ofi_complex_lxor_## type			\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return (a && !b) || (!a && b);				\
}								\

OFI_DEF_COMPLEX_OPS(float)
OFI_DEF_COMPLEX_OPS(double)
OFI_DEF_COMPLEX_OPS(long_double)


/* atomics primitives */
#ifdef HAVE_BUILTIN_ATOMICS
#define ofi_atomic_add_and_fetch(radix, ptr, val) __sync_add_and_fetch((ptr), (val))
#define ofi_atomic_sub_and_fetch(radix, ptr, val) __sync_sub_and_fetch((ptr), (val))
#endif /* HAVE_BUILTIN_ATOMICS */

#endif /* _FI_UNIX_OSD_H_ */
