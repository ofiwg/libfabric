/*
 * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc.  All rights reserved.
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

#ifndef _FI_WIN_OSD_H_
#define _FI_WIN_OSD_H_

#include "config.h"

#include <WinSock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>
#include <errno.h>
#include <complex.h>
#include "pthread.h"

#include <sys/uio.h>

#include <rdma/fi_errno.h>
#include <rdma/fabric.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MSG_NOSIGNAL doesn't exist on Windows */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL	0
#endif

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE	0
#endif

#define FI_DESTRUCTOR(func) void func

#define LITTLE_ENDIAN 5678
#define BIG_ENDIAN 8765
#define BYTE_ORDER LITTLE_ENDIAN

#define OFI_SOCK_TRY_RCV_AGAIN(err)				\
	((err) == WSAETIMEDOUT || (err) == WSAEWOULDBLOCK)

struct util_shm
{ /* this is dummy structure to provide compilation on Windows platform. */
  /* will be updated on real Windows implementation */
	HANDLE		shared_fd;
	void		*ptr;
	const char	*name;
	size_t		size;
};

#define FI_FFSL(val)	 			\
do						\
{						\
	int i = 0;				\
	while(val)				\
	{					\
		if((val) & 1)			\
		{				\
			return i + 1; 		\
		}				\
		else				\
		{				\
			i++;			\
			(val) = (val) >> 1;	\
		}				\
	}					\
} while(0)

#define strdup _strdup
#define strcasecmp _stricmp
#define snprintf _snprintf
#define inet_ntop InetNtopA

#define getpid (int)GetCurrentProcessId
#define sleep(x) Sleep(x * 1000)

#define __PRI64_PREFIX "ll"

#define HOST_NAME_MAX 256

#define MIN min
#define MAX max
#define OFI_UNUSED UNREFERENCED_PARAMETER

#define htonll _byteswap_uint64
#define ntohll _byteswap_uint64
#define strncasecmp _strnicmp

//#define INET_ADDRSTRLEN  (16)
//#define INET6_ADDRSTRLEN (48)

int fd_set_nonblock(int fd);

int socketpair(int af, int type, int protocol, int socks[2]);
void sock_get_ip_addr_table(struct slist *addr_list);

static inline int ffsl(long val)
{
	unsigned long v = (unsigned long)val;
	FI_FFSL(v);
	return 0;
}

static inline int ffsll(long long val)
{
	unsigned long long v = (unsigned long long)val;
	FI_FFSL(v);
	return 0;
}

static inline int asprintf(char **ptr, const char *format, ...)
{
	va_list args;
	va_start(args, format);

	int len = vsnprintf(0, 0, format, args);
	*ptr = (char*)malloc(len + 1);
	vsnprintf(*ptr, len + 1, format, args);
	(*ptr)[len] = 0; /* to be sure that string is enclosed */
	va_end(args);

	return len;
}

static inline char* strsep(char **stringp, const char *delim)
{
	char* ptr = *stringp;
	char* p;

	p = ptr ? strpbrk(ptr, delim) : NULL;

	if(!p)
		*stringp = NULL;
	else
	{
		*p = 0;
		*stringp = p + 1;
	}

	return ptr;
}

#define __attribute__(x)

static inline int ofi_memalign(void **memptr, size_t alignment, size_t size)
{
	*memptr = _aligned_malloc(size, alignment);
	return *memptr == 0;
}

static inline void ofi_freealign(void *memptr)
{
	_aligned_free(memptr);
}

static inline void ofi_osd_init(void)
{
	WORD wsa_version;
	WSADATA data;
	int err;

	wsa_version = MAKEWORD(2, 2);

	err = WSAStartup(wsa_version, &data);
}

static inline void ofi_osd_fini(void)
{
	WSACleanup();
}

static inline SOCKET ofi_socket(int domain, int type, int protocol)
{
	return socket(domain, type, protocol);
}

static inline ssize_t ofi_read_socket(SOCKET fd, void *buf, size_t count)
{
	return recv(fd, (char*)buf, (int)count, 0);
}

static inline ssize_t ofi_write_socket(SOCKET fd, const void *buf, size_t count)
{
	return send(fd, (const char*)buf, (int)count, 0);
}

static inline ssize_t ofi_send_socket(SOCKET fd, const void *buf, size_t count,
        int flags)
{
	return send(fd, (const char*)buf, (int)count, flags);
}

static inline int ofi_close_socket(SOCKET socket)
{
	return closesocket(socket);
}

static inline int ofi_sockerr(void)
{
	int wsaerror = WSAGetLastError();
	switch (wsaerror) {
	case WSAEINPROGRESS:
	case WSAEWOULDBLOCK:
	      return EINPROGRESS;
	default:
	      return wsaerror;
	}
}

static inline int fi_wait_cond(pthread_cond_t *cond, pthread_mutex_t *mut, int timeout_ms)
{
	return !SleepConditionVariableCS(cond, mut, (DWORD)timeout_ms);
}

int ofi_shm_map(struct util_shm *shm, const char *name, size_t size,
				int readonly, void **mapped);

static inline int ofi_shm_remap(struct util_shm *shm, size_t newsize, void **mapped)
{
	OFI_UNUSED(shm);
	OFI_UNUSED(newsize);
	OFI_UNUSED(mapped);

	return -FI_ENOENT;
}

static inline char * strndup(char const *src, size_t n)
{
	size_t len = strnlen(src, n);
	char *dst = (char *)malloc(len + 1);

	if (dst) {
		memcpy(dst, src, len);
		dst[len] = 0;
	}
	return dst;
}

static inline int ofi_sysconf(int name)
{
	SYSTEM_INFO si;

	GetSystemInfo(&si);

	switch (name) {
	case _SC_PAGESIZE:
		return si.dwPageSize;
	default:
		errno = EINVAL;
		return -1;
	}
}

int ofi_shm_unmap(struct util_shm *shm);

static inline int ofi_is_loopback_addr(struct sockaddr *addr) {
	return (addr->sa_family == AF_INET &&
		((struct sockaddr_in *)addr)->sin_addr.s_addr == ntohl(INADDR_LOOPBACK)) ||
		(addr->sa_family == AF_INET6 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[0] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[1] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[2] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[3] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[4] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[5] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[6] == 0 &&
		((struct sockaddr_in6 *)addr)->sin6_addr.u.Word[7] == ntohs(1));
}

/* complex operations implementation */

#define OFI_DEF_COMPLEX(type)					\
typedef struct {						\
	type real;						\
	type imag;						\
} ofi_complex_## type;						\
static inline int ofi_complex_eq_## type			\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	return a.real == b.real && a.imag == b.imag;		\
}								\
static inline ofi_complex_## type ofi_complex_sum_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	ofi_complex_## type res;				\
	res.real = a.real + b.real;				\
	res.imag = a.imag + b.imag;				\
	return res;						\
}								\
static inline ofi_complex_## type ofi_complex_prod_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	ofi_complex_## type res;				\
	res.real = a.real * b.real - a.imag * b.imag;		\
	res.imag = a.real * b.imag + a.imag * b.real;		\
	return res;						\
}								\
static inline ofi_complex_## type ofi_complex_land_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	ofi_complex_## type res;				\
	res.real = (a.real || a.imag) && (b.real || b.imag);	\
	res.imag = 0;						\
	return res;						\
}								\
static inline ofi_complex_## type ofi_complex_lor_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	ofi_complex_## type res;				\
	res.real = (a.real || a.imag) || (b.real || b.imag);	\
	res.imag = 0;						\
	return res;						\
}								\
static inline ofi_complex_## type ofi_complex_lxor_## type	\
	(ofi_complex_## type a, ofi_complex_## type b)		\
{								\
	ofi_complex_## type res;				\
	res.real = ((a.real || a.imag) && !(b.real || b.imag)) || \
		   (!(a.real || a.imag) && (b.real || b.imag));	\
	res.imag = 0;						\
	return res;						\
}

OFI_DEF_COMPLEX(float)
OFI_DEF_COMPLEX(double)
OFI_DEF_COMPLEX(long_double)


/* atomics primitives */
#ifdef HAVE_BUILTIN_ATOMICS
#define InterlockedAdd32 InterlockedAdd
typedef LONG ofi_atomic_int_32_t;
typedef LONGLONG ofi_atomic_int_64_t;

#define ofi_atomic_add_and_fetch(radix, ptr, val) InterlockedAdd##radix((ofi_atomic_int_##radix##_t *)(ptr), (ofi_atomic_int_##radix##_t)(val))
#define ofi_atomic_sub_and_fetch(radix, ptr, val) InterlockedAdd##radix((ofi_atomic_int_##radix##_t *)(ptr), -(ofi_atomic_int_##radix##_t)(val))
#endif /* HAVE_BUILTIN_ATOMICS */

#ifdef __cplusplus
}
#endif

#endif /* _FI_WIN_OSD_H_ */
