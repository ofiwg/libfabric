/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2014 Intel Corp.  All rights reserved.
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
 * This is a port of libiverbs/examples/rc_pingong.c to libfabric.
 * It's a simple pingong test with connected endpoints.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <getopt.h>
#include <time.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>

#define FI_CLOSE(DESC, STR) 										\
	do {															\
		if (DESC) {													\
			if (fi_close(&DESC->fid)) {								\
				fprintf(stderr, STR);								\
				return 1;											\
			}														\
		}															\
	} while (0)

#define FI_ERR_LOG(STR, RC) fprintf(stderr, "%s: %s (%d)", STR, fi_strerror(-RC), -RC)

static int page_size;

enum {
	PINGPONG_RECV_WCID = 1,
	PINGPONG_SEND_WCID = 2,
};

struct pingpong_context {
	struct fi_info		*prov;
	struct fid_fabric	*fabric;
	struct fid_domain	*dom;
	struct fid_mr		*mr;
	struct fid_pep		*lep;
	struct fid_ep		*ep;
	struct fid_eq		*eq;
	struct fid_cq		*cq;
	void			*buf;
	int			 size;
	int			 send_flags;
	int			 rx_depth;
	int			 pending;
	int			use_event;
};

int pp_close_ctx(struct pingpong_context *ctx);

static int pp_eq_create(struct pingpong_context *ctx)
{
	struct fi_eq_attr cm_attr;
	int rc;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.wait_obj 	= FI_WAIT_FD;				

	rc = fi_eq_open(ctx->fabric, &cm_attr, &ctx->eq, NULL);
	if (rc)
		FI_ERR_LOG("fi_eq_open cm", rc);

	return rc;
}

static int pp_cq_create(struct pingpong_context *ctx)
{
	struct fi_cq_attr cq_attr;
	int rc = 0;

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format 		= FI_CQ_FORMAT_CONTEXT;
	if (ctx->use_event)
		cq_attr.wait_obj = FI_WAIT_FD;				
	else
		cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size 		= ctx->rx_depth + 1;

	rc = fi_cq_open(ctx->dom, &cq_attr, &ctx->cq, NULL);
	if (rc) {
		FI_ERR_LOG("fi_eq_open", rc);
		return 1;
	}

	return 0;
}

static int pp_listen_ctx(struct pingpong_context *ctx)
{
	int rc = 0;

	rc = fi_passive_ep(ctx->fabric, ctx->prov, &ctx->lep, NULL);
	if (rc) {
		fprintf(stderr, "Unable to open listener endpoint\n");
		return 1;
	}

	/* Create listener EQ */
	rc = pp_eq_create(ctx);
	if (rc) {
		fprintf(stderr, "Unable to allocate listener resources\n");
		goto err;
	}

	rc = fi_bind(&ctx->lep->fid, &ctx->eq->fid, 0);
	if (rc) {
		FI_ERR_LOG("Unable to bind listener resources", -rc);
		goto err;
	}

	rc = fi_listen(ctx->lep);
	if (rc) {
		FI_ERR_LOG("Unable to listen for connections", -rc);
		goto err;
	}

	printf("Listening for incoming connections...\n");
	return 0;

err:
	pp_close_ctx(ctx);
	return 1;
}

static int pp_accept_ctx(struct pingpong_context *ctx)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	int rc = 0;
	int rd = 0;

	rd = fi_eq_sread(ctx->eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_ERR_LOG("fi_eq_sread %s", -rd);
		goto err;
	}

	if (event != FI_CONNREQ) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		goto err;
	}

	rc = fi_domain(ctx->fabric, entry.info, &ctx->dom, NULL);
	if (rc) {
		FI_ERR_LOG("fi_fdomain", rc);
		goto err;
	}

	/* Check if we require memory registration: No provider support for now
	 * so we register memory without checking */
	//if (ctx->prov->domain_cap & FI_LOCAL_MR ) {
		rc = fi_mr_reg(ctx->dom, ctx->buf, ctx->size, FI_SEND | FI_RECV, 0, 0, 0, &ctx->mr, NULL);
		if (rc) {
			FI_ERR_LOG("fi_mr_reg", -rc);
			goto err;
		}	
	//}

	rc = fi_endpoint(ctx->dom, entry.info, &ctx->ep, NULL);
	if (rc) {
		FI_ERR_LOG("fi_endpoint for req", rc);
		goto err;
	}

	/* Create event queue */
	if (pp_cq_create(ctx)) {
		fprintf(stderr, "Unable to create event queue\n");
		goto err;
	}

	rc = fi_bind(&ctx->ep->fid, &ctx->cq->fid, FI_SEND | FI_RECV);
	if (rc) {
		FI_ERR_LOG("fi_bind", rc);
		goto err;
	}

	rc = fi_bind(&ctx->ep->fid, &ctx->eq->fid, 0);
	if (rc) {
		FI_ERR_LOG("fi_bind", rc);
		goto err;
	}

	rc = fi_accept(ctx->ep, NULL, 0);
	if (rc) {
		FI_ERR_LOG("fi_accept", rc);
		goto err;
	}

	rd = fi_eq_sread(ctx->eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_ERR_LOG("fi_eq_sread %s", -rd);
		goto err;
	}

	if (event != FI_CONNECTED) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		goto err;
	}
	printf("Connection accepted\n");

	fi_freeinfo(entry.info);
	return 0;

err:
	pp_close_ctx(ctx);
	return 1;
}

static int pp_connect_ctx(struct pingpong_context *ctx)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	int rc = 0;

	/* Open domain */
	rc = fi_domain(ctx->fabric, ctx->prov, &ctx->dom, NULL);
	if (rc) {
		FI_ERR_LOG("fi_fdomain", -rc);
		goto err;
	}

	if (pp_eq_create(ctx)) {
		fprintf(stderr, "Unable to create event queue\n");
		goto err;
	}
	
	/* Check if we require memory registration: No provider support for now
	 * so we register memory without checking */
	//if (ctx->prov->domain_cap & FI_LOCAL_MR ) {
		rc = fi_mr_reg(ctx->dom, ctx->buf, ctx->size, FI_SEND | FI_RECV, 0, 0, 0, &ctx->mr, NULL);
		if (rc) {
			FI_ERR_LOG("fi_mr_reg", -rc);
			goto err;
		}	
	//}

	/* Open endpoint */
	rc = fi_endpoint(ctx->dom, ctx->prov, &ctx->ep, NULL);
	if (rc) {
		FI_ERR_LOG("Unable to create EP", rc);
		goto err;
	}
	
	/* Create event queue */
	if (pp_cq_create(ctx)) {
		fprintf(stderr, "Unable to create event queue\n");
		goto err;
	}
	
	/* Bind eq to ep */
	rc = fi_bind(&ctx->ep->fid, &ctx->cq->fid, FI_SEND | FI_RECV);
	if (rc) {
		FI_ERR_LOG("fi_bind", rc);
		goto err;
	}	

	rc = fi_bind(&ctx->ep->fid, &ctx->eq->fid, 0);
	if (rc) {
		FI_ERR_LOG("fi_bind", rc);
		goto err;
	}

	printf("Connecting to server\n");
	rc = fi_connect(ctx->ep, ctx->prov->dest_addr, NULL, 0);
	if (rc) {
		FI_ERR_LOG("Unable to connect to destination", rc);
		goto err;
	}

	rc = fi_eq_sread(ctx->eq, &event, &entry, sizeof entry, -1, 0);
	if (rc != sizeof entry) {
		FI_ERR_LOG("fi_eq_sread %s", -rc);
		goto err;
	}

	if (event != FI_CONNECTED) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		goto err;
	}

	printf("Connection successful\n");
	return 0;

err:
	pp_close_ctx(ctx);
	return 1;
}

static struct pingpong_context *pp_init_ctx(struct fi_info *prov, int size,
					    int rx_depth, int use_event)
{
	struct pingpong_context *ctx;
	int rc = 0;

	ctx = calloc(1, sizeof *ctx);
	if (!ctx)
		return NULL;

	ctx->prov 			= prov;
	ctx->size       	= size;
	ctx->rx_depth   	= rx_depth;
	ctx->use_event   	= use_event;

	ctx->buf = memalign(page_size, size);
	if (!ctx->buf) {
		fprintf(stderr, "Couldn't allocate work buf.\n");
		goto clean_ctx;
	}

	/* FIXME memset(ctx->buf, 0, size); */
	memset(ctx->buf, 0x7b, size);

	/* Open the fabric */
	rc = fi_fabric(prov->fabric_attr, &ctx->fabric, NULL);
	if (rc) {
		FI_ERR_LOG("Couldn't open fabric", rc);
		return NULL;
	}

	return ctx;

clean_ctx:
	free(ctx);

	return NULL;
}

int pp_close_ctx(struct pingpong_context *ctx)
{
	FI_CLOSE(ctx->lep, "Couldn't destroy listener EP\n");
	FI_CLOSE(ctx->ep, "Couldn't destroy EP\n");
	FI_CLOSE(ctx->eq, "Couldn't destroy EQ\n");
	FI_CLOSE(ctx->cq, "Couldn't destroy CQ\n");
	FI_CLOSE(ctx->mr, "Couldn't destroy MR\n");
	FI_CLOSE(ctx->dom, "Couldn't deallocate Domain\n");
	FI_CLOSE(ctx->fabric, "Couldn't close fabric\n");

	if (ctx->buf)
		free(ctx->buf);
	if (ctx)
		free(ctx);

	return 0;
}

static int pp_post_recv(struct pingpong_context *ctx, int n)
{
	int rc = 0;
	int i;


	for (i = 0; i < n; ++i) {
		rc = fi_recv(ctx->ep, ctx->buf, ctx->size, fi_mr_desc(ctx->mr),
			     0, (void *)(uintptr_t)PINGPONG_RECV_WCID);
		if (rc) {
			FI_ERR_LOG("fi_recv", -rc);
			break;
		}
	}

	return i;
}

static int pp_post_send(struct pingpong_context *ctx)
{
	int rc = 0;

	rc = fi_send(ctx->ep, ctx->buf, ctx->size, fi_mr_desc(ctx->mr), 
		     0, (void *)(uintptr_t)PINGPONG_SEND_WCID);
	if (rc) {
		FI_ERR_LOG("fi_send", rc);
		return 1;
	}

	return 0;
}

static void usage(const char *argv0)
{
	printf("Usage:\n");
	printf("  %s            start a server and wait for connection\n", argv0);
	printf("  %s <host>     connect to server at <host>\n", argv0);
	printf("\n");
	printf("Options:\n");
	printf("  -p, --port=<port>      listen on/connect to port <port> (default 18515)\n");
	printf("  -d, --ib-dev=<dev>     use IB device <dev> (default first device found)\n");
	printf("  -i, --ib-port=<port>   use port <port> of IB device (default 1)\n");
	printf("  -s, --size=<size>      size of message to exchange (default 4096)\n");
	// No provider support yet
	// printf("  -m, --mtu=<size>       path MTU (default 1024)\n");
	printf("  -r, --rx-depth=<dep>   number of receives to post at a time (default 500)\n");
	printf("  -n, --iters=<iters>    number of exchanges (default 1000)\n");
	printf("  -e, --events           sleep on CQ events (default poll)\n");
}

int main(int argc, char *argv[])
{
	struct 		fi_info				*prov_list;
	struct 		fi_info				*prov;
	struct 		fi_info 			hints;
	uint64_t 	flags 				= 0;
	char 		*service 			= NULL;
	char 		*node 				= NULL;
	struct pingpong_context *ctx;
	struct timeval           start, end;
	char                    *prov_name = NULL;
	char                    *servername = NULL;
	int                      port = 18515;
	int                      ib_port = 1;
	int                      size = 4096;
	// No provider support yet
	//enum ibv_mtu		 mtu = IBV_MTU_1024;
	//size_t					 mtu = 1024;
	int                      rx_depth = 500;
	int                      iters = 1000;
	int                      use_event = 0;
	int                      routs;
	int                      rcnt, scnt;
	int						 rc = 0;

	srand48(getpid() * time(NULL));

	while (1) {
		int c;

		static struct option long_options[] = {
			{ .name = "port",     	.has_arg = 1, .val = 'p' },
			{ .name = "prov-name",	.has_arg = 1, .val = 'd' },
			{ .name = "ib-port",  	.has_arg = 1, .val = 'i' },
			{ .name = "size",     	.has_arg = 1, .val = 's' },
			// No provider support yet
			//{ .name = "mtu",      	.has_arg = 1, .val = 'm' },
			{ .name = "rx-depth", 	.has_arg = 1, .val = 'r' },
			{ .name = "iters",    	.has_arg = 1, .val = 'n' },
			{ .name = "events",   	.has_arg = 0, .val = 'e' },
			{ 0 }
		};

		c = getopt_long(argc, argv, "p:d:i:s:m:r:n:e:",
							long_options, NULL);
		if (c == -1)
			break;

		switch (c) {
		case 'p':
			port = strtol(optarg, NULL, 0);
			if (port < 0 || port > 65535) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 'd':
			prov_name = strdupa(optarg);
			break;

		case 'i':
			ib_port = strtol(optarg, NULL, 0);
			if (ib_port < 0) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 's':
			size = strtol(optarg, NULL, 0);
			break;

		// No provider support yet
		/*case 'm':
			mtu = strtol(optarg, NULL, 0);
			mtu = pp_mtu_to_enum(strtol(optarg, NULL, 0));
			if (mtu < 0) {
				usage(argv[0]);
				return 1;
			}
			break;
			*/

		case 'r':
			rx_depth = strtol(optarg, NULL, 0);
			break;

		case 'n':
			iters = strtol(optarg, NULL, 0);
			break;

		case 'e':
			++use_event;
			break;

		default:
			usage(argv[0]);
			return 1;
		}
	}

	if (optind == argc - 1)
		servername = strdupa(argv[optind]);
	else if (optind < argc) {
		usage(argv[0]);
		return 1;
	}

	page_size = sysconf(_SC_PAGESIZE);

	memset(&hints, 0, sizeof(hints));
	
	/* Infiniband provider */
	hints.ep_type = FI_EP_MSG;
	hints.caps = FI_MSG;
	hints.mode = FI_LOCAL_MR | FI_PROV_MR_ATTR;
	hints.addr_format = FI_SOCKADDR;

	asprintf(&service, "%d", port);
	if (!servername) {
		flags |= FI_SOURCE;
	} else {
		node = servername;
	}
	
	rc = fi_getinfo(FI_VERSION(1, 0), node, service, flags, &hints, &prov_list);
	if (rc) {
		FI_ERR_LOG("fi_getinfo", rc);
		return 1;
	}


	if (!prov_list) {
		perror("Failed to get providers list");
		return 1;
	}

	if (!prov_name) {
		prov = prov_list;
		if (!prov) {
			fprintf(stderr, "No providers found\n");
			return 1;
		}
	} else {
		for (prov = prov_list; prov; prov = prov->next)
			if (!strcmp(prov->fabric_attr->prov_name, prov_name))
				break;
		if (!prov) {
			fprintf(stderr, "Provider %s not found\n", prov_name);
			return 1;
		}
	}

	ctx = pp_init_ctx(prov, size, rx_depth, use_event);
	if (!ctx)
		return 1;

	if (servername) {
		/* client connect */
		if (pp_connect_ctx(ctx))
			return 1;
	} else {
		/* server listen and accept */
		pp_listen_ctx(ctx);
		pp_accept_ctx(ctx);
	}

	routs = pp_post_recv(ctx, ctx->rx_depth);
	if (routs < ctx->rx_depth) {
		fprintf(stderr, "Couldn't post receive (%d)\n", routs);
		return 1;
	}

	ctx->pending = PINGPONG_RECV_WCID;

	if (servername) {
		if (pp_post_send(ctx)) {
			fprintf(stderr, "Couldn't post send\n");
			return 1;
		}
		ctx->pending |= PINGPONG_SEND_WCID;
	}

	if (gettimeofday(&start, NULL)) {
		perror("gettimeofday");
		return 1;
	}

	rcnt = scnt = 0;
	while (rcnt < iters || scnt < iters) {
		struct fi_cq_entry wc;
		struct fi_cq_err_entry cq_err;
		int rd;

		if (use_event) {
			/* Blocking read */
			rd = fi_cq_sread(ctx->cq, &wc, sizeof wc, NULL, -1);
		} else {
			do {
				rd = fi_cq_read(ctx->cq, &wc, 1);
			} while (rd == 0);
		}

		if (rd < 0) {
			fi_cq_readerr(ctx->cq, &cq_err, 0);
			fprintf(stderr, "cq fi_eq_readerr() %s (%d)\n", 
				fi_cq_strerror(ctx->cq, cq_err.err, cq_err.err_data, NULL, 0),
				cq_err.err);
			return 1;
		}

		switch ((int) (uintptr_t) wc.op_context) {
		case PINGPONG_SEND_WCID:
			++scnt;
			break;

		case PINGPONG_RECV_WCID:
			if (--routs <= 1) {
				routs += pp_post_recv(ctx, ctx->rx_depth - routs);
				if (routs < ctx->rx_depth) {
					fprintf(stderr,
						"Couldn't post receive (%d)\n",
						routs);
					return 1;
				}
			}

			++rcnt;
			break;

		default:
			fprintf(stderr, "Completion for unknown wc_id %d\n",
				(int) (uintptr_t) wc.op_context);
			return 1;
		}

		ctx->pending &= ~(int) (uintptr_t) wc.op_context;
		if (scnt < iters && !ctx->pending) {
			if (pp_post_send(ctx)) {
				fprintf(stderr, "Couldn't post send\n");
				return 1;
			}
			ctx->pending = PINGPONG_RECV_WCID | PINGPONG_SEND_WCID;
		}
	}

	if (gettimeofday(&end, NULL)) {
		perror("gettimeofday");
		return 1;
	}

	{
		float usec = (end.tv_sec - start.tv_sec) * 1000000 +
			(end.tv_usec - start.tv_usec);
		long long bytes = (long long) size * iters * 2;

		printf("%lld bytes in %.2f seconds = %.2f Mbit/sec\n",
		       bytes, usec / 1000000., bytes * 8. / usec);
		printf("%d iters in %.2f seconds = %.2f usec/iter\n",
		       iters, usec / 1000000., usec / iters);
	}

	/* Close the connection */
	fi_shutdown(ctx->ep, 0);
	if (pp_close_ctx(ctx))
		return 1;

	fi_freeinfo(prov_list);
	free(service);
	free(prov_name);

	return 0;
}
