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
#define MSG_NOSIGNAL 0
#endif

#define FI_DESTRUCTOR(func) void func

#define LITTLE_ENDIAN 5678
#define BIG_ENDIAN 8765
#define BYTE_ORDER LITTLE_ENDIAN

struct util_shm
{ /* this is dummy structure to provide compilation on Windows platform. */
  /* will be updated on real Windows implementation */
	HANDLE		shared_fd;
	void		*ptr;
	const char	*name;
	size_t		size;
};

static inline void ofi_osd_init()
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

static inline ssize_t ofi_read_socket(int fd, void *buf, size_t count)
{
	return recv(fd, (char*)buf, (int)count, 0);
}

static inline ssize_t ofi_write_socket(int fd, const void *buf, size_t count)
{
	return send(fd, (const char*)buf, (int)count, 0);
}

static inline ssize_t ofi_send_socket(int fd, const void *buf, size_t count,
        int flags)
{
	return send(fd, (const char*)buf, count, flags);
}

static inline int ofi_close_socket(int socket)
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

int ofi_shm_unmap(struct util_shm *shm);

/* complex operations implementation */
#define OFI_COMPLEX(name) ofi_##name##_complex
#define OFI_COMPLEX_BASE(name) OFI_COMPLEX(name)##_base
#define OFI_COMPLEX_OP(name, op) ofi_complex_##name##_##op
#define OFI_COMPLEX_TYPE_DECL(name, type)	\
typedef type OFI_COMPLEX_BASE(name);		\
typedef struct {				\
	OFI_COMPLEX_BASE(name) re;		\
	OFI_COMPLEX_BASE(name) im;		\
} OFI_COMPLEX(name);

OFI_COMPLEX_TYPE_DECL(float, float)
OFI_COMPLEX_TYPE_DECL(double, double)
OFI_COMPLEX_TYPE_DECL(long_double, long double)

#define OFI_COMPLEX_OPS(name)								\
static inline OFI_COMPLEX_BASE(name) OFI_COMPLEX_OP(name, real)(OFI_COMPLEX(name) v)	\
{											\
	return v.re;									\
} 											\
static inline OFI_COMPLEX_BASE(name) OFI_COMPLEX_OP(name, imag)(OFI_COMPLEX(name) v)	\
{											\
	return v.im;									\
}											\
static inline OFI_COMPLEX(name) OFI_COMPLEX_OP(name, sum)(OFI_COMPLEX(name) v1, OFI_COMPLEX(name) v2) \
{											\
	OFI_COMPLEX(name) ret = {.re = v1.re + v2.re, .im = v1.im + v2.im};		\
	return ret;									\
}											\
static inline OFI_COMPLEX(name) OFI_COMPLEX_OP(name, mul)(OFI_COMPLEX(name) v1, OFI_COMPLEX(name) v2) \
{											\
	OFI_COMPLEX(name) ret = {.re = (v1.re * v2.re) - (v1.im * v2.im),		\
			      .im = (v1.re * v2.im) + (v1.im * v2.re)};			\
	return ret;									\
}											\
static inline int OFI_COMPLEX_OP(name, equ)(OFI_COMPLEX(name) v1, OFI_COMPLEX(name) v2)	\
{											\
	return v1.re == v2.re && v1.im == v2.im;					\
}											\
static inline OFI_COMPLEX(name) OFI_COMPLEX_OP(name, land)(OFI_COMPLEX(name) v1, OFI_COMPLEX(name) v2) \
{											\
	OFI_COMPLEX(name) zero = {.re = 0, .im = 0};					\
	int equ = !OFI_COMPLEX_OP(name, equ)(v1, zero) && !OFI_COMPLEX_OP(name, equ)(v2, zero); \
	OFI_COMPLEX(name) ret = {.re = equ ? 1.f : 0, .im = 0};				\
	return ret;									\
}											\
static inline OFI_COMPLEX(name) OFI_COMPLEX_OP(name, lor)(OFI_COMPLEX(name) v1, OFI_COMPLEX(name) v2) \
{											\
	OFI_COMPLEX(name) zero = {.re = 0, .im = 0};					\
	int equ = !OFI_COMPLEX_OP(name, equ)(v1, zero) || !OFI_COMPLEX_OP(name, equ)(v2, zero); \
	OFI_COMPLEX(name) ret = {.re = equ ? 1.f : 0, .im = 0};				\
	return ret;									\
}

OFI_COMPLEX_OPS(float)
OFI_COMPLEX_OPS(double)
OFI_COMPLEX_OPS(long_double)

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
