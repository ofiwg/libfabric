/*
 * Copyright (c) 2026 DataDirect Networks, Inc. All rights reserved.
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

/*
 * PEP close/EQ read regression test: xnet must not use an unowned connection
 * handle after its PEP has been closed. Build and run with AddressSanitizer
 * when validating the provider lifetime fix.
 *
 * NOTE: This test is most reliable on loopback (127.0.0.1) where TCP connections
 * complete quickly enough for the bug to trigger consistently.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fabric.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "shared.h"

#define DEFAULT_PORT "29384"

struct test_resources {
	struct fi_info *info;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fid_pep *pep;
	struct fid_eq *eq;
	struct fid_cq *cq;
};

static int report(const char *call, int ret)
{
	fprintf(stderr, "%s failed: %s (%d)\n", call,
		ret < 0 ? fi_strerror(-ret) : "unknown error", ret);
	return ret;
}

static void close_resources(struct test_resources *r)
{
	FT_CLOSE_FID(r->ep);
	FT_CLOSE_FID(r->cq);
	FT_CLOSE_FID(r->pep);
	FT_CLOSE_FID(r->domain);
	FT_CLOSE_FID(r->eq);
	FT_CLOSE_FID(r->fabric);
	if (r->info)
		fi_freeinfo(r->info);
	memset(r, 0, sizeof(*r));
}

static int make_hints(struct fi_info **hints_out, const char *provider)
{
	struct fi_info *hints = fi_allocinfo();

	if (!hints)
		return -FI_ENOMEM;

	hints->fabric_attr->prov_name = provider ? strdup(provider) : NULL;
	hints->ep_attr->type = FI_EP_MSG;
	hints->caps = FI_MSG;
	hints->addr_format = FI_SOCKADDR_IN;

	*hints_out = hints;
	return 0;
}

static int setup_endpoint(struct test_resources *r, const char *node,
			  const char *service, const char *provider,
			  int is_server)
{
	struct fi_info *hints;
	struct fi_eq_attr my_eq_attr = {
		.wait_obj = FI_WAIT_UNSPEC,
	};
	uint64_t flags;
	int ret;

	flags = is_server ? FI_SOURCE : 0;

	ret = make_hints(&hints, provider);
	if (ret)
		return ret;

	ret = fi_getinfo(FI_VERSION(1, 9), node, service, flags,
			 hints, &r->info);
	fi_freeinfo(hints);
	if (ret)
		return report("fi_getinfo", ret);

	ret = fi_fabric(r->info->fabric_attr, &r->fabric, NULL);
	if (ret)
		return report("fi_fabric", ret);

	ret = fi_eq_open(r->fabric, &my_eq_attr, &r->eq, NULL);
	if (ret)
		return report("fi_eq_open", ret);

	if (is_server) {
		/* Server: create passive endpoint */

		ret = fi_passive_ep(r->fabric, r->info, &r->pep, NULL);
		if (ret)
			return report("fi_passive_ep", ret);

		ret = fi_pep_bind(r->pep, &r->eq->fid, 0);
		if (ret)
			return report("fi_pep_bind", ret);

		ret = fi_listen(r->pep);
		if (ret)
			return report("fi_listen", ret);
	} else {
		/* Client: create active endpoint */
		ret = fi_domain(r->fabric, r->info, &r->domain, NULL);
		if (ret)
			return report("fi_domain", ret);

		{
			struct fi_cq_attr cq_attr = {
				.size = 1,
				.format = FI_CQ_FORMAT_MSG,
			};
			ret = fi_cq_open(r->domain, &cq_attr, &r->cq, NULL);
		}
		if (ret)
			return report("fi_cq_open", ret);

		ret = fi_endpoint(r->domain, r->info, &r->ep, NULL);
		if (ret)
			return report("fi_endpoint", ret);

		ret = fi_ep_bind(r->ep, &r->cq->fid, FI_SEND | FI_RECV);
		if (ret)
			return report("fi_ep_bind(cq)", ret);

		ret = fi_ep_bind(r->ep, &r->eq->fid, 0);
		if (ret)
			return report("fi_ep_bind(eq)", ret);

		ret = fi_enable(r->ep);
		if (ret)
			return report("fi_enable", ret);

		ret = fi_connect(r->ep, r->info->dest_addr, NULL, 0);
		if (ret)
			return report("fi_connect", ret);
	}

	return 0;
}

static int drive_client_progress(struct fid_eq *eq)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	ssize_t ret;

	/* Complete the nonblocking connect and send the CM request.
	 * Do not wait for FI_CONNECTED event: the server closes the PEP
	 * before accepting the connection. */

	while (1) {
		ret = fi_eq_read(eq, &event, &entry, sizeof(entry), 0);
		if (ret == (ssize_t) sizeof(entry)) {
			if (entry.info)
				fi_freeinfo(entry.info);
			continue;
		}
		if (ret == -FI_EAGAIN) {
			/* No more events available */
			return 0;
		}
		/* Error occurred */
		FT_PROCESS_EQ_ERR(ret, eq, "fi_eq_read", "client");
		return (int) ret;
	}
}

static int run_client(const char *node, const char *service,
		      const char *provider)
{
	struct test_resources r = {0};
	int ret;

	printf("Running client, connecting to %s:%s\n",
	       node ? node : "default", service);

	/* Wait for server to be ready */
	ft_sock_sync(sock, 0);

	ret = setup_endpoint(&r, node, service, provider, 0);
	if (ret)
		goto out;

	/* Signal server that client has connected */
	ft_sock_sync(sock, 1);

	/* Wait for server to accept */
	ft_sock_sync(sock, 2);

	ret = drive_client_progress(r.eq);
	if (ret)
		goto out;

	/* Signal server that CM is ready */
	ft_sock_sync(sock, 3);

	/* Wait for server to complete */
	ft_sock_sync(sock, 4);

	printf("Test PASSED\n");

out:
	close_resources(&r);
	return ret;
}

static int run_server(const char *node, const char *service,
		      const char *provider)
{
	struct test_resources r = {0};
	struct fi_eq_cm_entry entry;
	uint32_t event;
	ssize_t ret;

	printf("Running server on %s:%s\n",
	       node ? node : "default", service);

	ret = setup_endpoint(&r, node, service, provider, 1);
	if (ret)
		goto out;

	/* Signal client that server is ready */
	ft_sock_sync(sock, 0);

	/* Wait for client to connect */
	ft_sock_sync(sock, 1);

	/* Call fi_eq_read to trigger the provider's progress function.
	 * This accepts the TCP socket and creates the xnet_conn_handle.
	 * On loopback, the connection completes quickly and the handle is created.
	 * On real network interfaces, timing may prevent the handle from being
	 * created, so the test may not trigger the bug reliably. */
	ret = fi_eq_read(r.eq, &event, &entry, sizeof(entry), 0);
	if (ret != -FI_EAGAIN) {
		fprintf(stderr, "initial fi_eq_read returned %zd; expected -FI_EAGAIN\n",
			ret);
		if (ret == (ssize_t) sizeof(entry) && entry.info)
			fi_freeinfo(entry.info);
		goto out;
	}

	/* Signal client that connection was accepted */
	ft_sock_sync(sock, 2);

	/* Wait for client CM to be ready */
	ft_sock_sync(sock, 3);

	/* Close the PEP while the unpublished handle still references it. */
	ret = fi_close(&r.pep->fid);
	r.pep = NULL;
	if (ret) {
		report("fi_close(pep)", (int)ret);
		goto out;
	}

	/* The next progress call must not touch the discarded connection handle.
	 * Call fi_eq_read multiple times to ensure we trigger the provider's
	 * progress path that accesses the freed handle. */
	for (int i = 0; i < 10; i++) {
		ret = fi_eq_read(r.eq, &event, &entry, sizeof(entry), 0);
		if (ret == (ssize_t) sizeof(entry) && entry.info)
			fi_freeinfo(entry.info);
		if (ret != -FI_EAGAIN && ret != (ssize_t) sizeof(entry))
			break;
		usleep(1000);  /* Small delay between calls */
	}

	if (ret == -FI_EAGAIN || ret == (ssize_t) sizeof(entry))
		ret = 0;

	if (ret == 0)
		printf("Test PASSED\n");

	/* Signal client that server is done */
	ft_sock_sync(sock, 4);

out:
	close_resources(&r);
	return ret < 0 ? (int) ret : 0;
}

static void usage(const char *prog_name)
{
	fprintf(stderr, "Usage: %s [OPTIONS] [server_address]\n", prog_name);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -p <provider>  Provider name (default: tcp)\n");
	fprintf(stderr, "  -s <address>   Source/server address (default: 127.0.0.1)\n");
	fprintf(stderr, "  -P <port>      Port/service (default: %s)\n", DEFAULT_PORT);
	fprintf(stderr, "  -h             Display this help\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Server mode (no server_address): Listen for connections\n");
	fprintf(stderr, "Client mode (with server_address): Connect to server\n");
}

int main(int argc, char **argv)
{
	const char *provider = "tcp";
	const char *node = "127.0.0.1";
	const char *service = DEFAULT_PORT;
	const char *oob_port = "47592";  /* OOB port for client/server sync */
	int op, ret;

	while ((op = getopt(argc, argv, "p:s:P:h")) != -1) {
		switch (op) {
		case 'p':
			provider = optarg;
			break;
		case 's':
			node = optarg;
			break;
		case 'P':
			service = optarg;
			break;
		case 'h':
		default:
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	/* If a positional argument is provided, it's the server address (client mode) */
	if (optind < argc) {
		/* Client mode */
		node = argv[optind];
		ret = ft_sock_connect((char *)node, (char *)oob_port);
		if (ret) {
			fprintf(stderr, "Failed to connect to OOB server %s:%s\n",
				node, oob_port);
			return EXIT_FAILURE;
		}
		ret = run_client(node, service, provider);
		ft_sock_shutdown(sock);
	} else {
		/* Server mode */
		ret = ft_sock_listen((char *)node, (char *)oob_port);
		if (ret) {
			fprintf(stderr, "Failed to start OOB server on %s:%s\n",
				node, oob_port);
			return EXIT_FAILURE;
		}

		/* Accept client connection on OOB socket */
		sock = accept(listen_sock, NULL, 0);
		if (sock < 0) {
			perror("accept");
			return EXIT_FAILURE;
		}

		ret = run_server(node, service, provider);
		ft_sock_shutdown(sock);
	}

	return ret ? EXIT_FAILURE : EXIT_SUCCESS;
}
