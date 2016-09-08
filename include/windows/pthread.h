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

#pragma once

#include <WinSock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <stdint.h>

#define PTHREAD_MUTEX_INITIALIZER {0}

#define pthread_cond_signal WakeConditionVariable
#define pthread_cond_broadcast WakeAllConditionVariable
#define pthread_mutex_init(mutex, attr) InitializeCriticalSection(mutex)
#define pthread_mutex_destroy DeleteCriticalSection
#define pthread_cond_init(cond, attr) (InitializeConditionVariable(cond), 0)
#define pthread_cond_destroy(x)	/* nothing to do */

typedef CRITICAL_SECTION	pthread_mutex_t;
typedef CONDITION_VARIABLE	pthread_cond_t;
typedef HANDLE			pthread_t;

static inline int pthread_mutex_lock(pthread_mutex_t* mutex)
{
	EnterCriticalSection(mutex);
	return 0;
}

static inline int pthread_mutex_trylock(pthread_mutex_t* mutex)
{
	return !TryEnterCriticalSection(mutex);
}

static inline int pthread_mutex_unlock(pthread_mutex_t* mutex)
{
	LeaveCriticalSection(mutex);
	return 0;
}

static inline int pthread_join(pthread_t thread, void** exit_code)
{
	if(WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0)
	{
		if(exit_code)
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

typedef struct fi_thread_arg
{
	void* (*routine)(void*);
	void* arg;
} fi_thread_arg;

static DWORD WINAPI ofi_thread_starter(void* arg)
{
	fi_thread_arg data = *(fi_thread_arg*)arg;
	free(arg);
	return (DWORD)(uint64_t)data.routine(data.arg);
}

static inline int pthread_create(pthread_t* thread, void* attr, void *(*routine)(void*), void* arg)
{
	attr; /* suppress warning */
	fi_thread_arg* data = (fi_thread_arg*)malloc(sizeof(*data));
	data->routine = routine;
	data->arg = arg;
	DWORD threadid;
	*thread = CreateThread(0, 0, ofi_thread_starter, data, 0, &threadid);
	return *thread == 0;
}


