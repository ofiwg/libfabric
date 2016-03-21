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
#include "pthread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define inline __inline

#ifdef _WIN32
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
#ifdef  _WIN64
typedef __int64 ssize_t;
#else
typedef int ssize_t;
#endif /* _WIN64           */
#endif /* _SSIZE_T_DEFINED */
#endif /* _WIN32           */

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

#define pthread_cond_signal WakeConditionVariable
#define pthread_mutex_init(mutex, attr) InitializeCriticalSection(mutex)
#define pthread_mutex_destroy DeleteCriticalSection
#define pthread_cond_init(cond, attr) (InitializeConditionVariable(cond), 0)
#define pthread_cond_destroy(x)	/* nothing to do */
#define getpid (int)GetCurrentProcessId
#define sleep(x) Sleep(x * 1000)

#define PRIx8         "hhx"
#define PRIx16        "hx"
#define PRIx32        "lx"
#define PRIx64        "llx"
#define __PRI64_PREFIX "ll"
# define PRIu64 __PRI64_PREFIX "u"
#define HOST_NAME_MAX 256

//#define INET_ADDRSTRLEN  (16)
//#define INET6_ADDRSTRLEN (48)

static inline int fi_close_fd(int fd);
int fd_set_nonblock(int fd);

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

static inline int asprintf(char** ptr, const char* format, ...)
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

static inline int gettimeofday(struct timeval* time, struct timezone* zone)
{
	const uint64_t shift = 116444736000000000ULL;
	zone;

	SYSTEMTIME  stime;
	FILETIME    ftime;
	uint64_t    utime;

	GetSystemTime(&stime);
	SystemTimeToFileTime(&stime, &ftime);
	utime = (((uint64_t)ftime.dwHighDateTime) << 32) + ((uint64_t)ftime.dwLowDateTime);

	time->tv_sec = (long)((utime - shift) / 10000000L);
	time->tv_usec = (long)(stime.wMilliseconds * 1000);
	return 0;
}

static inline char* strsep(char** stringp, const char* delim)
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

static inline int pthread_join(pthread_t thread, void** exit_code)
{
	if (WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0)
	{
		if (exit_code)
		{
			DWORD ex = 0;
			GetExitCodeThread(thread, &ex);
			*exit_code = (void*)(uint64_t)ex;
		}
		CloseHandle(thread);
		return 0;
	}

	return -1;
}

static int socketpair(int af, int type, int protocol, int socks[2])
{
	protocol; /* suppress warning */
	struct sockaddr_in in_addr;
	int lsock;
	int len = sizeof(in_addr);

	if(!socks)
	{
		WSASetLastError(WSAEINVAL);
		return SOCKET_ERROR;
	}

	socks[0] = socks[1] = (int)INVALID_SOCKET;
	if ((lsock = socket(af == AF_UNIX ? AF_INET : af, type, 0)) == INVALID_SOCKET)
	{
		return SOCKET_ERROR;
	}

	memset(&in_addr, 0, sizeof(in_addr));
	in_addr.sin_family = AF_INET;
	in_addr.sin_addr.s_addr = htonl(0x7f000001);

	if(bind(lsock, (struct sockaddr*)&in_addr, sizeof(in_addr)))
	{
		int err = WSAGetLastError();
		fi_close_fd(lsock);
		WSASetLastError(err);
		return SOCKET_ERROR;
	}
	if(getsockname(lsock, (struct sockaddr*) &in_addr, &len))
	{
		int err = WSAGetLastError();
		fi_close_fd(lsock);
		WSASetLastError(err);
		return SOCKET_ERROR;
	}

	if (listen(lsock, 1))
		goto err;
	if ((socks[0] = WSASocket(af == AF_UNIX ? AF_INET : af, type, 0, NULL, 0, 0)) == INVALID_SOCKET)
		goto err;
	if(connect(socks[0], (const struct sockaddr*) &in_addr, sizeof(in_addr)))
		goto err;
	if ((socks[1] = accept(lsock, NULL, NULL)) == INVALID_SOCKET)
		goto err;

	fi_close_fd(lsock);
	return 0;

	int err;
err:
	err = WSAGetLastError();
	fi_close_fd(lsock);
	fi_close_fd(socks[0]);
	fi_close_fd(socks[1]);
	WSASetLastError(err);
	return SOCKET_ERROR;
}

typedef struct fi_thread_arg
{
	void* (*routine)(void*);
	void* arg;
} fi_thread_arg;

static DWORD WINAPI fi_thread_starter(void* arg)
{
	fi_thread_arg data = *(fi_thread_arg*)arg;
	free(arg);
	return (DWORD)(uint64_t)data.routine(data.arg);
}

static int pthread_create(pthread_t* thread, void* attr, void *(*routine)(void*), void* arg)
{
	attr; /* suppress warning */
	fi_thread_arg* data = (fi_thread_arg*)malloc(sizeof(*data));
	data->routine = routine;
	data->arg = arg;
	DWORD threadid;
	*thread = CreateThread(0, 0, fi_thread_starter, data, 0, &threadid);
	return *thread == 0;
}

struct iovec
{
	void *iov_base; /* Pointer to data.  */
	size_t iov_len; /* Length of data.  */
};

#define __attribute__(x)

static inline int poll(struct pollfd *fds, int nfds, int timeout)
{
	return WSAPoll(fds, nfds, timeout);
}

#ifdef __cplusplus
}
#endif

#endif /* _FI_WIN_OSD_H_ */

