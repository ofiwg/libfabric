/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

static void psmx2_string_to_uuid(const char *s, psm2_uuid_t uuid)
{
	int n;

	if (!s) {
		memset(uuid, 0, sizeof(psm2_uuid_t));
		return;
	}

	n = sscanf(s,
		"%2hhx%2hhx%2hhx%2hhx-"
		"%2hhx%2hhx-%2hhx%2hhx-%2hhx%2hhx-"
		"%2hhx%2hhx%2hhx%2hhx%2hhx%2hhx",
		&uuid[0], &uuid[1], &uuid[2], &uuid[3],
		&uuid[4], &uuid[5], &uuid[6], &uuid[7], &uuid[8], &uuid[9],
		&uuid[10], &uuid[11], &uuid[12], &uuid[13], &uuid[14], &uuid[15]);

	if (n != 16) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"wrong uuid format: %s\n", s);
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"correct uuid format is: "
			"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\n");
	}
}

void psmx2_get_uuid(psm2_uuid_t uuid)
{
	psmx2_string_to_uuid(psmx2_env.uuid, uuid);
}

int psmx2_uuid_to_port(psm2_uuid_t uuid)
{
	uint16_t port;
	uint16_t *u = (uint16_t *)uuid;

	port = u[0] + u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7];
	if (port < 4096)
		port += 4096;

	return (int)port;
}

char *psmx2_uuid_to_string(psm2_uuid_t uuid)
{
	static char s[40];

	sprintf(s,
		"%02hhX%02hhX%02hhX%02hhX-"
		"%02hhX%02hhX-%02hhX%02hhX-%02hhX%02hhX-"
		"%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX",
		uuid[0], uuid[1], uuid[2], uuid[3],
		uuid[4], uuid[5], uuid[6], uuid[7], uuid[8], uuid[9],
		uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);

	return s;
}

/*************************************************************
 * A simple name resolution mechanism for client-server style
 * applications. The server side has to run first. The client
 * side then passes the server name as the "node" parameter
 * of fi_getinfo call and the resulting provider info should
 * have the transport address of the server in the "dest_addr"
 * field. Both sides have to use the same UUID.
 *************************************************************/
static void psmx2_name_server_cleanup(void *arg)
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");
	close((uintptr_t)arg);
}

void *psmx2_name_server(void *args)
{
	struct psmx2_fid_fabric *fabric;
	struct addrinfo hints = {
		.ai_flags = AI_PASSIVE,
		.ai_family = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	struct addrinfo *res, *p;
	char *service;
	int listenfd = -1, connfd;
	int port;
	int n;
	int ret;

	fabric = args;
	port = psmx2_uuid_to_port(fabric->uuid);

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "port: %d\n", port);

	if (asprintf(&service, "%d", port) < 0)
		return NULL;

	n = getaddrinfo(NULL, service, &hints, &res);
	if (n < 0) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"port %d: %s\n", port, gai_strerror(n));
		free(service);
		return NULL;
	}

	for (p=res; p; p=p->ai_next) {
		listenfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
		if (listenfd >= 0) {
			n = 1;
			if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof(n)) == -1)
				FI_WARN(&psmx2_prov, FI_LOG_CORE,
					"setsockopt: %s\n", strerror(errno));
			if (!bind(listenfd, p->ai_addr, p->ai_addrlen))
				break;
			close(listenfd);
			listenfd = -1;
		}
	}

	freeaddrinfo(res);
	free(service);

	if (listenfd < 0) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"couldn't listen to port %d. try set FI_PSM2_UUID to a different value?\n",
			port);
		return NULL;
	}

	listen(listenfd, 256);

	pthread_cleanup_push(psmx2_name_server_cleanup, (void *)(uintptr_t)listenfd);
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "Start working ...\n");

	while (1) {
		connfd = accept(listenfd, NULL, 0);
		if (connfd >= 0) {
			if (fabric->active_domain) {
				ret = write(connfd, &fabric->active_domain->psm2_epid,
					    sizeof(psm2_epid_t));
				if (ret != sizeof(psm2_epid_t))
					FI_WARN(&psmx2_prov, FI_LOG_CORE,
						"error sending address info to the client\n");
			}
			close(connfd);
		}
	}

	pthread_cleanup_pop(1);

	return NULL;
}

void *psmx2_resolve_name(const char *servername, int port)
{
	struct addrinfo hints = {
		.ai_family   = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	struct addrinfo *res, *p;
	psm2_uuid_t uuid;
	char *service;
	struct psmx2_ep_name *dest_addr;
	int sockfd = -1;
	int n;

	if (!port) {
		psmx2_get_uuid(uuid);
		port = psmx2_uuid_to_port(uuid);
	}

	if (asprintf(&service, "%d", port) < 0)
		return NULL;

	n = getaddrinfo(servername, service, &hints, &res);
	if (n < 0) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"(%s:%d):%s\n", servername, port, gai_strerror(n));
		free(service);
		return NULL;
	}

	for (p = res; p; p = p->ai_next) {
		sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
		if (sockfd >= 0) {
			if (!connect(sockfd, p->ai_addr, p->ai_addrlen))
				break;
			close(sockfd);
			sockfd = -1;
		}
	}

	freeaddrinfo(res);
	free(service);

	if (sockfd < 0) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"couldn't connect to %s:%d\n", servername, port);
		return NULL;
	}

	dest_addr = calloc(1,sizeof(struct psmx2_ep_name));
	if (!dest_addr) {
		close(sockfd);
		return NULL;
	}

	if (read(sockfd, &dest_addr->epid, sizeof(psm2_epid_t)) != sizeof(psm2_epid_t)) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"error reading response from %s:%d\n", servername, port);
		free(dest_addr);
		close(sockfd);
		return NULL;
	}

	close(sockfd);

	return dest_addr;
}

static int psmx2_errno_table[PSM2_ERROR_LAST] = {
	0,		/* PSM2_OK = 0 */
	0,		/* PSM2_OK_NO_PROGRESS = 1 */
	-FI_EOTHER,
	-FI_EINVAL,	/* PSM2_PARAM_ERR = 3 */
	-FI_ENOMEM, 	/* PSM2_NO_MEMORY = 4 */
	-FI_EBADF,	/* PSM2_INIT_NOT_INIT = 5 */
	-FI_EINVAL,	/* PSM2_INIT_BAD_API_VERSION = 6 */
	-FI_ENOSYS,	/* PSM2_NO_AFFINITY = 7 */
	-FI_EIO,	/* PSM2_INTERNAL_ERR = 8 */
	-FI_EINVAL,	/* PSM2_SHMEM_SEGMENT_ERR = 9 */
	-FI_EACCES,	/* PSM2_OPT_READONLY = 10 */
	-FI_ETIMEDOUT,	/* PSM2_TIMEOUT = 11 */
	-FI_EMFILE,	/* PSM2_TOO_MANY_ENDPOINTS = 12 */
	-FI_ESHUTDOWN,	/* PSM2_IS_FINALIZED = 13 */
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER,
	-FI_ESHUTDOWN,	/* PSM2_EP_WAS_CLOSED = 20 */
	-FI_ENODEV,	/* PSM2_EP_NO_DEVICE = 21 */
	-FI_ENOENT,	/* PSM2_EP_UNIT_NOT_FOUND = 22 */
	-FI_EIO,	/* PSM2_EP_DEVICE_FAILURE = 23 */
	-FI_ETIMEDOUT, 	/* PSM2_EP_CLOSE_TIMEOUT = 24 */
	-FI_ENOENT,	/* PSM2_EP_NO_PORTS_AVAIL = 25 */
	-FI_ENETDOWN,	/* PSM2_EP_NO_NETWORK = 26 */
	-FI_EINVAL,	/* PSM2_EP_INVALID_UUID_KEY = 27 */
	-FI_ENOSPC,	/* PSM2_EP_NO_RESOURCES = 28 */
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER,
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER,
	-FI_EBADF,	/* PSM2_EPID_UNKNOWN = 40 */
	-FI_ENETUNREACH,/* PSM2_EPID_UNREACHABLE = 41 */
	-FI_EOTHER,
	-FI_EINVAL,	/* PSM2_EPID_INVALID_NODE = 43 */
	-FI_EINVAL,	/* PSM2_EPID_INVALID_MTU =  44 */
	-FI_EINVAL,	/* PSM2_EPID_INVALID_UUID_KEY = 45 */
	-FI_EINVAL,	/* PSM2_EPID_INVALID_VERSION = 46 */
	-FI_EINVAL,	/* PSM2_EPID_INVALID_CONNECT = 47 */
	-FI_EISCONN,	/* PSM2_EPID_ALREADY_CONNECTED = 48 */
	-FI_EIO,	/* PSM2_EPID_NETWORK_ERROR = 49 */
	-FI_EINVAL,	/* PSM2_EPID_INVALID_PKEY = 50 */
	-FI_ENETUNREACH,/* PSM2_EPID_PATH_RESOLUTION = 51 */
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER,
	-FI_EOTHER, -FI_EOTHER,
	-FI_EAGAIN,	/* PSM2_MQ_NO_COMPLETIONS = 60 */
	-FI_ETRUNC,	/* PSM2_MQ_TRUNCATION = 61 */
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER,
	-FI_EOTHER, -FI_EOTHER,
	-FI_EINVAL,	/* PSM2_AM_INVALID_REPLY = 70 */
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER, -FI_EOTHER,
	-FI_EOTHER, -FI_EOTHER, -FI_EOTHER
			/* PSM2_ERROR_LAST = 80 */
};

int psmx2_errno(int err)
{
	if (err >= 0 && err < PSM2_ERROR_LAST)
		return psmx2_errno_table[err];
	else
		return -FI_EOTHER;
}

/*
 * PSM context sharing requires some information from the MPI process manager.
 * Try to get the needed information from the environment.
 */
void psmx2_query_mpi(void)
{
	char *s;
	char env[32];
	int local_size = -1;
	int local_rank = -1;

	/* Check Open MPI */
	if ((s = getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))) {
		local_size = atoi(s);
		if ((s = getenv("OMPI_COMM_WORLD_LOCAL_RANK")))
			local_rank = atoi(s);
		snprintf(env, sizeof(env), "%d", local_size);
		setenv("MPI_LOCALNRANKS", env, 0);
		snprintf(env, sizeof(env), "%d", local_rank);
		setenv("MPI_LOCALRANKID", env, 0);
		return;
	}

	/* TODO: check other MPI */
}

