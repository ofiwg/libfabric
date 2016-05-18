
#pragma once

#include <WinSock2.h>
#include <ws2tcpip.h>
#include <windows.h>

typedef CRITICAL_SECTION   pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE		   pthread_t;

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


