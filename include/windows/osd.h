/*
 * Copyright (c) 2016 Intel Corporation.  All rights reserved.
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

#include <WinSock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>
#include <errno.h>
#include "pthread.h"

#include <rdma/fi_errno.h>
#include <rdma/fabric.h>

#ifdef __cplusplus
extern "C" {
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

#define PRIx8         "hhx"
#define PRIx16        "hx"
#define PRIx32        "lx"
#define PRIx64        "llx"
#define __PRI64_PREFIX "ll"
# define PRIu64 __PRI64_PREFIX "u"
#define HOST_NAME_MAX 256

#define MIN min
#define MAX max
#define OFI_UNUSED(p) (void)(p)

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

struct iovec
{
	void *iov_base; /* Pointer to data.  */
	size_t iov_len; /* Length of data.  */
};

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
	return recv(fd, (char*)buf, count, 0);
}

static inline ssize_t ofi_write_socket(int fd, const void *buf, size_t count)
{
	return send(fd, (const char*)buf, count, 0);
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

static inline int fi_wait_cond(pthread_cond_t *cond, pthread_mutex_t *mut, int timeout)
{
	return !SleepConditionVariableCS(cond, mut, (DWORD)timeout);
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

int ofi_shm_unmap(struct util_shm *shm);

#ifdef __cplusplus
}
#endif

#endif /* _FI_WIN_OSD_H_ */

