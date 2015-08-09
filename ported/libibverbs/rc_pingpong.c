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
#include <getopt.h>
#include <time.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <shared.h>

#define FT_CLOSE(DESC, STR) 				\
	do {						\
		if (DESC) {				\
			if (fi_close(&DESC->fid)) {	\
				fprintf(stderr, STR);	\
				return 1;		\
			}				\
		}					\
	} while (0)

static int page_size;

enum {
	PINGPONG_RECV_WCID = 1,
	PINGPONG_SEND_WCID = 2,
};

struct pingpong_context {
	struct fi_info		*info;
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
		FT_PRINTERR("fi_eq_open", rc);

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
		FT_PRINTERR("fi_cq_open", rc);
		return 1;
	}

	return 0;
}

static int pp_listen_ctx(struct pingpong_context *ctx)
{
	int rc = 0;

	rc = fi_passive_ep(ctx->fabric, ctx->info, &ctx->lep, NULL);
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

	rc = fi_pep_bind(ctx->lep, &ctx->eq->fid, 0);
	if (rc) {
		FT_PRINTERR("fi_pep_bind", rc);
		goto err;
	}

	rc = fi_listen(ctx->lep);
	if (rc) {
		FT_PRINTERR("fi_listen", rc);
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
	ssize_t rd;

	rd = fi_eq_sread(ctx->eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, ctx->eq, "fi_eq_sread", "listen");
		goto err;
	}

	if (event != FI_CONNREQ) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		goto err;
	}

	rc = fi_domain(ctx->fabric, entry.info, &ctx->dom, NULL);
	if (rc) {
		FT_PRINTERR("fi_fdomain", rc);
		goto err;
	}


	rc = fi_mr_reg(ctx->dom, ctx->buf, ctx->size, FI_SEND | FI_RECV, 0, 0, 0, &ctx->mr, NULL);
	if (rc) {
		FT_PRINTERR("fi_mr_reg", rc);
		goto err;
	}

	rc = fi_endpoint(ctx->dom, entry.info, &ctx->ep, NULL);
	if (rc) {
		FT_PRINTERR("fi_endpoint", rc);
		goto err;
	}

	/* Create event queue */
	if (pp_cq_create(ctx)) {
		fprintf(stderr, "Unable to create event queue\n");
		goto err;
	}

	rc = fi_ep_bind(ctx->ep, &ctx->cq->fid, FI_SEND | FI_RECV);
	if (rc) {
		FT_PRINTERR("fi_ep_bind", rc);
		goto err;
	}

	rc = fi_ep_bind(ctx->ep, &ctx->eq->fid, 0);
	if (rc) {
		FT_PRINTERR("fi_ep_bind", rc);
		goto err;
	}

	rc = fi_accept(ctx->ep, NULL, 0);
	if (rc) {
		FT_PRINTERR("fi_accept", rc);
		goto err;
	}

	rd = fi_eq_sread(ctx->eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, ctx->eq, "fi_eq_sread", "accept");
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
	ssize_t rd;

	/* Open domain */
	rc = fi_domain(ctx->fabric, ctx->info, &ctx->dom, NULL);
	if (rc) {
		FT_PRINTERR("fi_fdomain", rc);
		goto err;
	}

	if (pp_eq_create(ctx)) {
		fprintf(stderr, "Unable to create event queue\n");
		goto err;
	}
	
	rc = fi_mr_reg(ctx->dom, ctx->buf, ctx->size, FI_SEND | FI_RECV, 0, 0, 0, &ctx->mr, NULL);
	if (rc) {
		FT_PRINTERR("fi_mr_reg", rc);
		goto err;
	}

	/* Open endpoint */
	rc = fi_endpoint(ctx->dom, ctx->info, &ctx->ep, NULL);
	if (rc) {
		FT_PRINTERR("fi_endpoint", rc);
		goto err;
	}
	
	/* Create event queue */
	if (pp_cq_create(ctx)) {
		fprintf(stderr, "Unable to create event queue\n");
		goto err;
	}
	
	/* Bind eq to ep */
	rc = fi_ep_bind(ctx->ep, &ctx->cq->fid, FI_SEND | FI_RECV);
	if (rc) {
		FT_PRINTERR("fi_ep_bind", rc);
		goto err;
	}	

	rc = fi_ep_bind(ctx->ep, &ctx->eq->fid, 0);
	if (rc) {
		FT_PRINTERR("fi_ep_bind", rc);
		goto err;
	}

	printf("Connecting to server\n");
	rc = fi_connect(ctx->ep, ctx->info->dest_addr, NULL, 0);
	if (rc) {
		FT_PRINTERR("fi_connect", rc);
		goto err;
	}

	rd = fi_eq_sread(ctx->eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, ctx->eq, "fi_eq_sread", "connect");
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

static struct pingpong_context *pp_init_ctx(struct fi_info *info, int size,
					    int rx_depth, int use_event)
{
	struct pingpong_context *ctx;
	int rc = 0;

	ctx = calloc(1, sizeof *ctx);
	if (!ctx)
		return NULL;

	ctx->info 		= info;
	ctx->size       	= size;
	ctx->rx_depth   	= rx_depth;
	ctx->use_event   	= use_event;

	if (posix_memalign(&(ctx->buf), page_size, size)) {
		fprintf(stderr, "Couldn't allocate work buf.\n");
		goto clean_ctx;
	}

	/* FIXME memset(ctx->buf, 0, size); */
	memset(ctx->buf, 0x7b, size);

	/* Open the fabric */
	rc = fi_fabric(info->fabric_attr, &ctx->fabric, NULL);
	if (rc) {
		FT_PRINTERR("fi_fabric", rc);
		return NULL;
	}

	return ctx;

clean_ctx:
	free(ctx);

	return NULL;
}

int pp_close_ctx(struct pingpong_context *ctx)
{
	FT_CLOSE(ctx->lep, "Couldn't destroy listener EP\n");
	FT_CLOSE(ctx->ep, "Couldn't destroy EP\n");
	FT_CLOSE(ctx->eq, "Couldn't destroy EQ\n");
	FT_CLOSE(ctx->cq, "Couldn't destroy CQ\n");
	FT_CLOSE(ctx->mr, "Couldn't destroy MR\n");
	FT_CLOSE(ctx->dom, "Couldn't deallocate Domain\n");
	FT_CLOSE(ctx->fabric, "Couldn't close fabric\n");

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
			FT_PRINTERR("fi_recv", rc);
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
		FT_PRINTERR("fi_send", rc);
		return 1;
	}

	return 0;
}

static void usage(char *argv0)
{
	ft_usage(argv0, "Reliable Connection Pingpong test");
	FT_PRINT_OPTS_USAGE("-S <size>", "size of message to exchange (default 4096)");
	// No provider support yet
	// printf("  -m, --mtu=<size>       path MTU (default 1024)\n");
	FT_PRINT_OPTS_USAGE("-r <rx-depth>", "number of receives to post at a time (default 500)");
	FT_PRINT_OPTS_USAGE("-n <iters>",  "number of exchanges (default 1000)");
	FT_PRINT_OPTS_USAGE("-e",         "sleep on CQ events (default poll)");
}

int main(int argc, char *argv[])
{
	uint64_t 	flags 				= 0;
	char 		*service 			= NULL;
	char 		*node 				= NULL;
	struct pingpong_context *ctx;
	struct timeval           start, end;
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

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return 1;

	while (1) {
		int c;

		c = getopt(argc, argv, "S:m:r:n:e:h" ADDR_OPTS INFO_OPTS);
		if (c == -1)
			break;

		switch (c) {
		case 'S':
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
			ft_parse_addr_opts(c, optarg, &opts);
			ft_parseinfo(c, optarg, hints);
			break;
		case '?':
		case 'h':
			usage(argv[0]);
			return 1;
		}
	}

	if (optind == argc - 1)
		opts.dst_addr = argv[optind];
	else if (optind < argc) {
		usage(argv[0]);
		return 1;
	}

	page_size = sysconf(_SC_PAGESIZE);

	hints->ep_attr->type = FI_EP_MSG;
	hints->caps = FI_MSG;
	hints->mode = FI_LOCAL_MR;
	hints->addr_format = FI_SOCKADDR;

	rc = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (rc)
		return 1;
	
	rc = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (rc) {
		FT_PRINTERR("fi_getinfo", rc);
		return -rc;
	}
	fi_freeinfo(hints);

	ctx = pp_init_ctx(fi, size, rx_depth, use_event);
	if (!ctx)
		return 1;

	if (opts.dst_addr) {
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

	if (opts.dst_addr) {
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
			} while (rd == -FI_EAGAIN);
		}

		if (rd < 0) {
			fi_cq_readerr(ctx->cq, &cq_err, 0);
			fprintf(stderr, "cq fi_cq_readerr() %s (%d)\n", 
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

	return 0;
}
