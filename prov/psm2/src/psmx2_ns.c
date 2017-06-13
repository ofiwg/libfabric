/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

/*
 * For each job (identified by the UUID), a name server is started on each
 * node as a thread within one of the processes on that node. It maintains
 * a database that maps "services" to "endpoint names". Other processes
 * on the same node talk to this name server to update mapping information.
 *
 * To resolve a "node:service" pair into an endpoint name that can be used
 * as the input of fi_av_insert, a process needs to make a query to the
 * name server residing on "node".
 */

enum {
	PSMX2_NS_ADD,
	PSMX2_NS_DEL,
	PSMX2_NS_QUERY,
	PSMX2_NS_ACK,
};

struct psmx2_ns_cmd {
	int op;
	int service;
	int status;
	struct psmx2_ep_name addr;
};

/*
 * NS map: service --> ep address mapping
 * locking is unnecessary because it is only accessed by the name server thread.
 */

static RbtHandle psmx2_ns_map;

static int psmx2_service_compare(void *svc1, void *svc2)
{
	if ((uintptr_t)svc1 == PSMX2_ANY_SERVICE ||
	    (uintptr_t)svc2 == PSMX2_ANY_SERVICE)
		return 0;

	return (svc1 < svc2) ?  -1 : (svc1 > svc2);
}

static int psmx2_ns_map_init(void)
{
	psmx2_ns_map = rbtNew(psmx2_service_compare);

	return psmx2_ns_map ? 0 : -FI_ENOMEM;
}

static void psmx2_ns_map_fini(void)
{
	rbtDelete(psmx2_ns_map);
}

static int psmx2_ns_map_add(int service, struct psmx2_ep_name *name_in)
{
	struct psmx2_ep_name *name;

	if (rbtFind(psmx2_ns_map, (void *)(uintptr_t)service)) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"failed to add address: service %u already in use.\n", service);
		return -FI_EBUSY;
	}

	name = malloc(sizeof(*name));
	if (!name) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"failed to add address for service %u: out of memory.\n", service);
		return -FI_ENOMEM;
	}

	*name = *name_in;

	if (rbtInsert(psmx2_ns_map, (void *)(uintptr_t)service, name)) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"failed to add address for service %u: out of memory.\n", service);
		return -FI_ENOMEM;
	}

	return 0;
}

static void psmx2_ns_map_del(int service, struct psmx2_ep_name *name_in)
{
	RbtIterator it;
	void *key;
	struct psmx2_ep_name *name;

        it = rbtFind(psmx2_ns_map, (void *)(uintptr_t)service);
        if (it) {
		rbtKeyValue(psmx2_ns_map, it, &key, (void **)&name);
		if (name->epid != name_in->epid ||
		    name->vlane != name_in->vlane) {
			FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"failed to delete address for service %u: "
				"expecting <%lx,%d>, got <%lx,%d>.\n",
				service, name_in->epid, name_in->vlane,
				name->epid, name->vlane);
			return;
		}
		free(name);
                rbtErase(psmx2_ns_map, it);
	}
}

static int psmx2_ns_map_lookup(int *service, struct psmx2_ep_name *name_out)
{
	RbtIterator it;
	void *key;
	struct psmx2_ep_name *name;

        it = rbtFind(psmx2_ns_map, (void *)(uintptr_t)(*service));
	if (!it)
		return -FI_ENOENT;

	rbtKeyValue(psmx2_ns_map, it, &key, (void **)&name);
	*name_out = *name;
	if (*service == PSMX2_ANY_SERVICE)
		*service = (uintptr_t)key;

	return 0;
}

static void psmx2_name_server_cleanup(void *arg)
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");
	close((uintptr_t)arg);
	psmx2_ns_map_fini();
}

static void *psmx2_name_server_func(void *args)
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
	struct psmx2_ns_cmd cmd;

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
			"couldn't bind to port %d. It is fine if another process from the "
			"same job has started the name server.\nHowever, if that's not the "
			"case, try set FI_PSM2_UUID to a different value.\n",
			port);
		return NULL;
	}

	if (psmx2_ns_map_init()) {
		close(listenfd);
		return NULL;
	}

	listen(listenfd, 256);

	pthread_cleanup_push(psmx2_name_server_cleanup, (void *)(uintptr_t)listenfd);
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "Start working ...\n");

	while (1) {
		connfd = accept(listenfd, NULL, 0);
		if (connfd >= 0) {
			ret = read(connfd, &cmd, sizeof(cmd));
			if (ret != sizeof(cmd)) {
				FI_WARN(&psmx2_prov, FI_LOG_CORE,
					"error reading command from client\n");
			} else {
				switch(cmd.op) {
				case PSMX2_NS_ADD:
					psmx2_ns_map_add(cmd.service, &cmd.addr);
					break;

				case PSMX2_NS_DEL:
					psmx2_ns_map_del(cmd.service, &cmd.addr);
					break;

				case PSMX2_NS_QUERY:
					cmd.op = PSMX2_NS_ACK;
					cmd.status = psmx2_ns_map_lookup(&cmd.service, &cmd.addr);
					ret = write(connfd, &cmd, sizeof(cmd));
					if (ret != sizeof(cmd))
						FI_WARN(&psmx2_prov, FI_LOG_CORE,
							"error sending address info to the client\n");
					break;

				default:
					FI_WARN(&psmx2_prov, FI_LOG_CORE,
						"invalid command from client: %d\n", cmd.op);
					break;
				}
			}
			close(connfd);
		}
	}

	pthread_cleanup_pop(1);

	return NULL;
}

/*
 * Name server API: server side
 */

static int psmx2_ns_connect_server(const char *server);

void psmx2_ns_start_server(struct psmx2_fid_fabric *fabric)
{
	int ret;
	int sockfd;
	int sleep_usec = 1000;

	ret = pthread_create(&fabric->name_server_thread, NULL,
			     psmx2_name_server_func, (void *)fabric);
	if (ret) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "pthread_create returns %d\n", ret);
		/* use the main thread's ID as invalid value for the new thread */
		fabric->name_server_thread = pthread_self();
	}

	/*
	 * Wait for the local name server to come up. It could be the thread
	 * created above, or the thread created by another process on the same
	 * node. The total wait time is about (1+2+4+...+8192)ms = 16 seconds.
	 */
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "connecting to local name server\n");
	while (sleep_usec < 10000) {
		sockfd = psmx2_ns_connect_server("localhost");
		if (sockfd >= 0) {
			close(sockfd);
			return;
		}
		usleep(sleep_usec);
		sleep_usec *= 2;
	}
	FI_WARN(&psmx2_prov, FI_LOG_CORE, "can't connect to local name server.\n", ret);
}

void psmx2_ns_stop_server(struct psmx2_fid_fabric *fabric)
{
	int ret;
	void *exit_code;

	if (pthread_equal(fabric->name_server_thread, pthread_self()))
		return;

	ret = pthread_cancel(fabric->name_server_thread);
	if (ret) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"pthread_cancel returns %d\n", ret);
	}
	ret = pthread_join(fabric->name_server_thread, &exit_code);
	if (ret) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"pthread_join returns %d\n", ret);
	} else {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"name server thread exited with code %ld (%s)\n",
			(uintptr_t)exit_code,
			(exit_code == PTHREAD_CANCELED) ? "PTHREAD_CANCELED" : "?");
	}
}

/*
 * Name server API: client side
 */

static int psmx2_ns_connect_server(const char *server)
{
	struct addrinfo hints = {
		.ai_family   = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	struct addrinfo *res, *p;
	psm2_uuid_t uuid;
	int port;
	char *service;
	int sockfd = -1;
	int n;

	psmx2_get_uuid(uuid);
	port = psmx2_uuid_to_port(uuid);

	if (asprintf(&service, "%d", port) < 0)
		return -1;

	n = getaddrinfo(server, service, &hints, &res);
	if (n < 0) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"(%s:%d):%s\n", server, port, gai_strerror(n));
		free(service);
		return -1;
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

	if (sockfd < 0) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"couldn't connect to %s:%d\n", server, port);
	}

	freeaddrinfo(res);
	free(service);

	return sockfd;
}

void psmx2_ns_add_local_name(int service, struct psmx2_ep_name *name)
{
	int sockfd;
	char *server = "localhost";
	struct psmx2_ns_cmd cmd;

	sockfd = psmx2_ns_connect_server(server);
	if (sockfd < 0)
		return;

	cmd.op = PSMX2_NS_ADD;
	cmd.service = service;
	cmd.addr = *name;
	if (write(sockfd, &cmd, sizeof(cmd)) != sizeof(cmd)) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"error write command to %s\n", server);
	}

	close(sockfd);
}

void psmx2_ns_del_local_name(int service, struct psmx2_ep_name *name)
{
	int sockfd;
	char *server = "localhost";
	struct psmx2_ns_cmd cmd;

	sockfd = psmx2_ns_connect_server(server);
	if (sockfd < 0)
		return;

	cmd.op = PSMX2_NS_DEL;
	cmd.service = service;
	cmd.addr = *name;
	if (write(sockfd, &cmd, sizeof(cmd)) != sizeof(cmd)) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"error write command to %s:%d\n", server);
	}

	close(sockfd);
}

void *psmx2_ns_resolve_name(const char *server, int *service)
{
	struct psmx2_ns_cmd cmd;
	struct psmx2_ep_name *dest_addr;
	int sockfd;

	sockfd = psmx2_ns_connect_server(server);
	if (sockfd < 0)
		return NULL;

	cmd.op = PSMX2_NS_QUERY;
	cmd.service = *service;
	if (write(sockfd, &cmd, sizeof(cmd)) != sizeof(cmd)) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"error write command to %s:%d\n", server);
		close(sockfd);
		return NULL;
	}

	if (read(sockfd, &cmd, sizeof(cmd)) != sizeof(cmd) || cmd.status) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"error reading response from %s\n", server);
		close(sockfd);
		return NULL;
	}

	close(sockfd);

	dest_addr = calloc(1,sizeof(struct psmx2_ep_name));
	if (!dest_addr)
		return NULL;

	*dest_addr = cmd.addr;
	*service = cmd.service;
	return dest_addr;
}
