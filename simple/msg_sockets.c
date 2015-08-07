/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <netdb.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"

static struct cs_opts opts;
static void *buf;
static size_t buffer_size = 1024;
static int rx_depth = 512;

union sockaddr_any {
	struct sockaddr		sa;
	struct sockaddr_in	sin;
	struct sockaddr_in6	sin6;
	struct sockaddr_storage	ss;
};

static union sockaddr_any bound_addr;
static size_t bound_addr_len = sizeof bound_addr;


/* Wrapper for memcmp for sockaddr.  Note that the sockaddr structure may
 * contain holes, so sockaddr's are expected to have been initialized to all
 * zeroes prior to being filled with an address. */
static int
sockaddrcmp(const union sockaddr_any *actual, socklen_t actual_len,
	    const union sockaddr_any *expected, socklen_t expected_len)
{
	if (actual->sa.sa_family != expected->sa.sa_family) {
		return actual->sa.sa_family - expected->sa.sa_family;
	} else if (actual_len != expected_len) {
		return actual_len - expected_len;
	}

	/* Handle binds to wildcard addresses, for address types we know
	 * about */
	switch (expected->sa.sa_family) {
	case AF_INET:
		if (expected->sin.sin_addr.s_addr == INADDR_ANY) {
			return 0;
		}
		break;
	case AF_INET6:
		if (!memcmp(&expected->sin6.sin6_addr,
			    &in6addr_any, sizeof(struct in6_addr))) {
			return 0;
		}
		break;
	}
	return memcmp(actual, expected, actual_len);
}

/* Returns a string for the given sockaddr using getnameinfo().  This returns a
 * static buffer so it is not reentrant or thread-safe.  Returns the string on
 * success and NULL on failure. */
static const char *
sockaddrstr(const union sockaddr_any *addr, socklen_t len, char *buf, size_t buflen)
{
	static char namebuf[BUFSIZ];
	static char servbuf[BUFSIZ];
	int errcode;

	if ((errcode = getnameinfo(&addr->sa, len, namebuf, BUFSIZ,
				servbuf, BUFSIZ,
				NI_NUMERICHOST | NI_NUMERICSERV))) {
		if (errcode != EAI_SYSTEM) {
			fprintf(stderr, "getnameinfo: %s\n", gai_strerror(errcode));
		} else {
			fprintf(stderr, "getnameinfo: %s\n", strerror(errcode));
		}
		errno = errcode;
		return NULL;
	}

	snprintf(buf, buflen, "[%s]:%s", namebuf, servbuf);
	return buf;
}

static int check_address(struct fid *fid, const char *message)
{
	char buf1[BUFSIZ], buf2[BUFSIZ];
	union sockaddr_any tmp;
	size_t tmplen;
	int ret;

	memset(&tmp, 0, sizeof tmp);
	tmplen = sizeof tmp;
	ret = fi_getname(fid, &tmp, &tmplen);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
	}

	if (sockaddrcmp(&tmp, tmplen, &bound_addr, bound_addr_len)) {
		FT_ERR("address changed after %s: got %s expected %s\n",
			message,
			sockaddrstr(&tmp, tmplen, buf1, BUFSIZ),
			sockaddrstr(&bound_addr, bound_addr_len, buf2, BUFSIZ));
		return -FI_EINVAL;
	}

	return 0;
}

static int alloc_cm_res(void)
{
	struct fi_eq_attr cm_attr = { 0 };
	int ret;

	cm_attr.wait_obj = FI_WAIT_FD;

	/* Open EQ to receive CM events */
	ret = fi_eq_open(fabric, &cm_attr, &eq, NULL);
	if (ret)
		FT_PRINTERR("fi_eq_open", ret);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&ep->fid);
	fi_close(&mr->fid);
	fi_close(&rxcq->fid);
	fi_close(&txcq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr = { 0 };
	int ret;

	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = rx_depth;

	/* Open completion queue for send completions */
	ret = fi_cq_open(domain, &cq_attr, &txcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	/* Open completion queue for recv completions */
	ret = fi_cq_open(domain, &cq_attr, &rxcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	/* Register memory */
	ret = fi_mr_reg(domain, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err4;
	}

	return 0;

err4:
	fi_close(&mr->fid);
err3:
	fi_close(&rxcq->fid);
err2:
	fi_close(&txcq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	/* Bind EQ with endpoint */
	ret = fi_ep_bind(ep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	/* Bind Send CQ with endpoint to collect send completions */
	ret = fi_ep_bind(ep, &txcq->fid, FI_SEND);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	/* Bind Recv CQ with endpoint to collect recv completions */
	ret = fi_ep_bind(ep, &rxcq->fid, FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	return ret;
}

static int server_listen(void)
{
	int ret;

	/* Allocate connection management resources */
	ret = alloc_cm_res();
	if (ret)
		goto err2;

	/* Bind EQ to passive endpoint */
	ret = fi_pep_bind(pep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_pep_bind", ret);
		goto err3;
	}

	/* Listen for incoming connections */
	ret = fi_listen(pep);
	if (ret) {
		FT_PRINTERR("fi_listen", ret);
		goto err3;
	}

	return 0;
err3:
	fi_close(&eq->fid);
err2:
	fi_close(&pep->fid);
	fi_close(&fabric->fid);
	return ret;
}

static int server_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *info = NULL;
	ssize_t rd;
	int ret;

	/* Wait for connection request from client */
	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PRINTERR("fi_eq_sread", rd);
		return (int) rd;
	}

	info = entry.info;
	if (event != FI_CONNREQ) {
		FT_ERR("Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err1;
	}

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = alloc_ep_res(info);
	if (ret)
		 goto err1;

	ret = bind_ep_res();
	if (ret)
		goto err3;

	/* Accept the incoming connection. Also transitions endpoint to active state */
	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_accept", ret);
		goto err3;
	}

	/* Wait for the connection to be established */
	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PRINTERR("fi_eq_sread", rd);
		goto err3;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		FT_ERR("Unexpected CM event %d fid %p (ep %p)\n", event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err3;
	}

	ret = check_address(&ep->fid, "accept");
	if (ret) {
		goto err3;
	}

	fi_freeinfo(info);
	return 0;

err3:
	free_ep_res();
err1:
	fi_reject(pep, info->handle, NULL, 0);
	fi_freeinfo(info);
	return ret;
}

static int client_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *fi;
	ssize_t rd;
	int ret;

	/* Get fabric info */
	ret = fi_getinfo(FT_FIVERSION, opts.dst_addr, opts.dst_port, 0, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err0;
	}

	/* Open domain */
	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = alloc_cm_res();
	if (ret)
		goto err4;

	ret = check_address(&pep->fid, "fi_endpoint (pep)");
	if (ret)
		goto err5;

	assert(fi->handle == &pep->fid);
	ret = alloc_ep_res(fi);
	if (ret)
		goto err5;

	/* Close the passive endpoint that we "stole" the source address
	 * from */
	ret = fi_close(&pep->fid);
	if (ret)
		goto err5;

	ret = check_address(&ep->fid, "fi_endpoint (ep)");
	if (ret)
		goto err5;

	ret = bind_ep_res();
	if (ret)
		goto err6;

	/* Connect to server */
	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_connect", ret);
		goto err6;
	}

	/* Wait for the connection to be established */
	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PRINTERR("fi_eq_sread", rd);
		return (int) rd;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		FT_ERR("Unexpected CM event %d fid %p (ep %p)\n", event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err6;
	}

	ret = check_address(&ep->fid, "connect");
	if (ret) {
		goto err6;
	}

	fi_freeinfo(fi);
	return 0;

err6:
	free_ep_res();
err5:
	fi_close(&eq->fid);
err4:
	fi_close(&domain->fid);
err2:
	fi_freeinfo(fi);
err0:
	return ret;
}

static int send_recv()
{
	struct fi_cq_entry comp;
	int ret;

	if (opts.dst_addr) {
		/* Client */
		fprintf(stdout, "Posting a send...\n");
		sprintf(buf, "Hello World!");
		ret = fi_send(ep, buf, sizeof("Hello World!"), fi_mr_desc(mr), 0, buf);
		if (ret) {
			FT_PRINTERR("fi_send", ret);
			return ret;
		}

		/* Read send queue */
		do {
			ret = fi_cq_read(txcq, &comp, 1);
			if (ret < 0 && ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_read", ret);
				return ret;
			}
		} while (ret == -FI_EAGAIN);

		fprintf(stdout, "Send completion received\n");
	} else {
		/* Server */
		fprintf(stdout, "Posting a recv...\n");
		ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}

		/* Read recv queue */
		fprintf(stdout, "Waiting for client...\n");
		do {
			ret = fi_cq_read(rxcq, &comp, 1);
			if (ret < 0 && ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_read", ret);
				return ret;
			}
		} while (ret == -FI_EAGAIN);

		fprintf(stdout, "Received data from client: %s\n", (char *)buf);
	}

	return 0;
}

static int setup_handle(void)
{
	static char buf[BUFSIZ];
	struct addrinfo *ai, aihints;
	struct fi_info *fi;
	int ret;

	memset(&aihints, 0, sizeof aihints);
	aihints.ai_flags = AI_PASSIVE;
	ret = getaddrinfo(opts.src_addr, opts.src_port, &aihints, &ai);
	if (ret == EAI_SYSTEM) {
		FT_PRINTERR("getaddrinfo", -ret);
		return -ret;
	} else if (ret) {
		FT_ERR("getaddrinfo: %s\n", gai_strerror(ret));
		return -FI_ENODATA;
	}

	switch (ai->ai_family) {
	case AF_INET:
		hints->addr_format = FI_SOCKADDR_IN;
		break;
	case AF_INET6:
		hints->addr_format = FI_SOCKADDR_IN6;
		break;
	}

	/* Get fabric info */
	ret = fi_getinfo(FT_FIVERSION, opts.src_addr, NULL, FI_SOURCE, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto free_ai;
	}
	free(fi->src_addr);
	fi->src_addr = NULL;
	fi->src_addrlen = 0;

	/* Open the fabric */
	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto free_fi;
	}

	/* Open a passive endpoint */
	ret = fi_passive_ep(fabric, fi, &pep, NULL);
	if (ret) {
		FT_PRINTERR("fi_passive_ep", ret);
		goto free_fabric;
	}

	ret = fi_setname(&pep->fid, ai->ai_addr, ai->ai_addrlen);
	if (ret) {
		FT_PRINTERR("fi_setname", ret);
		goto free_pep;
	}

	ret = fi_getname(&pep->fid, &bound_addr, &bound_addr_len);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
		goto free_pep;
	}

	/* Verify port number */
	switch (ai->ai_family) {
	case AF_INET:
		if (bound_addr.sin.sin_port == 0) {
			FT_ERR("port number is 0 after fi_setname()\n");
			ret = -FI_EINVAL;
			goto free_pep;
		}
		break;
	case AF_INET6:
		if (bound_addr.sin6.sin6_port == 0) {
			FT_ERR("port number is 0 after fi_setname()\n");
			ret = -FI_EINVAL;
			goto free_pep;
		}
		break;
	}

	printf("bound_addr: \"%s\"\n",
		sockaddrstr(&bound_addr, bound_addr_len, buf, BUFSIZ));

	hints->handle = &pep->fid;
	goto free_fi;

free_pep:
	fi_close(&pep->fid);
free_fabric:
	fi_close(&fabric->fid);

free_fi:
	fi_freeinfo(fi);
free_ai:
	freeaddrinfo(ai);
	return ret;
}


int main(int argc, char **argv)
{
	int op, ret;
	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "A simple MSG client-sever example.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type	= FI_EP_MSG;
	hints->caps		= FI_MSG;
	hints->mode		= FI_LOCAL_MR;
	hints->addr_format	= FI_SOCKADDR;

	/* Fabric and connection setup */
	if (!opts.src_addr || !opts.src_port) {
		fprintf(stderr, "Source address (-s) is required for this test\n");
		return EXIT_FAILURE;
	}

	if (opts.dst_addr && (opts.src_port == opts.dst_port))
		opts.src_port = "9229";

	ret = setup_handle();
	if (ret)
		return -ret;

	if (!opts.dst_addr) {
		ret = server_listen();
		if (ret)
			return -ret;
	}

	ret = opts.dst_addr ? client_connect() : server_connect();
	if (ret) {
		return -ret;
	}

	/* Exchange data */
	ret = send_recv();

	fi_shutdown(ep, 0);
	free_ep_res();
	fi_close(&eq->fid);
	fi_close(&domain->fid);
	fi_close(&fabric->fid);

	return ret;
}
