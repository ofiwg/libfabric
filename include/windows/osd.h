/*
* Copyright (c) 2017 Intel Corporation.  All rights reserved.
*
* This software is available to you under the BSD license below:
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
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
* BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
* ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef _WINDOWS_OSD_H_
#define _WINDOWS_OSD_H_

#include <winsock2.h>
#include <ws2def.h>
#include <windows.h>
#include <assert.h>
#include <inttypes.h>

#include <time.h>

struct iovec
{
	void *iov_base; /* Pointer to data.  */
	size_t iov_len; /* Length of data.  */
};

#define strncasecmp _strnicmp
#define SHUT_RDWR SD_BOTH
#define CLOCK_MONOTONIC	1

#ifndef EAI_SYSTEM
# define EAI_SYSTEM	-11
#endif

typedef int pid_t;

static int clock_gettime(int which_clock, struct timespec* tp)
{
	LARGE_INTEGER freq;
	LARGE_INTEGER count;

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&count);

	tp->tv_sec = (time_t)((double)count.QuadPart / freq.QuadPart);
	tp->tv_nsec = (long)((uint64_t)
		((double)count.QuadPart * 1000000 / freq.QuadPart) % 1000000);

	return 0;
}

static inline int ft_close_fd(int fd)
{
	return closesocket(fd);
}

static inline int poll(struct pollfd *fds, int nfds, int timeout)
{
	return WSAPoll(fds, nfds, timeout);
}

static inline char* strndup(const char* str, size_t n)
{
	char* res = strdup(str);
	if (strlen(res) > n)
		res[n] = '\0';
	return res;
}

#define _SC_PAGESIZE	30

static long int sysconf(int name)
{
	switch (name) {
	case _SC_PAGESIZE:
		SYSTEM_INFO info;
		GetNativeSystemInfo(&info);
		return (long int)info.dwPageSize;
	default:
		assert(0);
	}
	errno = EINVAL;
	return -1;
}

#define AF_LOCAL AF_UNIX

int socketpair(int af, int type, int protocol, int socks[2]);

/* Bits in the fourth argument to `waitid'.  */
#define WSTOPPED	2	/* Report stopped child (same as WUNTRACED). */
#define WEXITED		4	/* Report dead child. */
#define WCONTINUED	8	/* Report continued child. */
#define WNOWAIT		0x01000000	/* Don't reap, just poll status. */

static pid_t waitpid(pid_t pid, int *status, int options)
{
	assert(0);
	return 0;
}

static const char* gai_strerror(int code)
{
	return "Unknown error";
}

static pid_t fork(void)
{
	assert(0);
	return -1;
}

static int posix_memalign(void **memptr, size_t alignment, size_t size)
{
	*memptr = _aligned_malloc(size, alignment);
	return (*memptr) ? 0 : ENOMEM;
}

static inline int ft_startup(void)
{
	int ret = 0;
	WSADATA data;

	ret = WSAStartup(MAKEWORD(2, 2), &data);
	if (ret)
		return HRESULT_FROM_WIN32(ret);
	return ret;
}

#endif /* _WINDOWS_OSD_H_ */
