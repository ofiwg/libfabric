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

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include "fi_osd.h"
#include "fi_file.h"

extern pthread_mutex_t ini_lock;
static INIT_ONCE ofi_init_once = INIT_ONCE_STATIC_INIT;

int socketpair(int af, int type, int protocol, int socks[2])
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
		closesocket(lsock);
		WSASetLastError(err);
		return SOCKET_ERROR;
	}
	if(getsockname(lsock, (struct sockaddr*) &in_addr, &len))
	{
		int err = WSAGetLastError();
		closesocket(lsock);
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

	closesocket(lsock);
	return 0;

	int err;
err:
	err = WSAGetLastError();
	closesocket(lsock);
	closesocket(socks[0]);
	closesocket(socks[1]);
	WSASetLastError(err);
	return SOCKET_ERROR;
}

int fi_read_file(const char *dir, const char *file, char *buf, size_t size)
{
	char *path = 0;
	int len, lendir, lenfile, pathlen;

	HANDLE fd = INVALID_HANDLE_VALUE;
	DWORD read;

	len = -1;

	lendir = lstrlenA(dir);
	lenfile = lstrlenA(file);

	pathlen = lendir + lenfile + 2; /* dir + '\' + file + '0' */

	path = malloc(pathlen);
	if (!path)
		goto fn_nomem;

	lstrcpyA(path, dir);
	if (lenfile) {
		lstrcatA(path, "\\");
		lstrcatA(path, file);
	}

	fd = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, 0, OPEN_EXISTING, 0, 0);
	if (fd == INVALID_HANDLE_VALUE)
		goto fn_nofile;

	if (!ReadFile(fd, buf, (DWORD)size, &read, 0))
		goto fn_faread;

	len = (int)read;

	if (len > 0 && buf[len - 1] == '\n')
		buf[--len] = '\0';

fn_faread:
	CloseHandle(fd);
fn_nofile:
	free(path);
fn_nomem:
	return len;
}

static BOOL CALLBACK ofi_init_once_cb(PINIT_ONCE once, void* data, void** ctx)
{
	OFI_UNUSED(once);
	OFI_UNUSED(ctx);
	InitializeCriticalSection((CRITICAL_SECTION*)data);
	return TRUE;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
	OFI_UNUSED(instance);
	OFI_UNUSED(reserved);

	switch(reason)
	{
	case DLL_PROCESS_ATTACH:
		InitOnceExecuteOnce(&ofi_init_once, ofi_init_once_cb, &ini_lock, 0);
		break;
	case DLL_THREAD_ATTACH:
	case DLL_PROCESS_DETACH:
	case DLL_THREAD_DETACH:
	default:
		break;
	}

	return TRUE;
}

