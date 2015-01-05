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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <shared.h>

// MULTI_BUF_SIZE_FACTOR defines how large the multi recv buffer will be.
// The minimum value of the factor is 2 which will set the multi recv buffer 
// size to be twice the size of the send buffer. In order to use FI_MULTI_RECV
// feature efficiently, we need to have a large recv buffer so that we don't
// to repost the buffer often to get the remaining data when the buffer is full
#define MULTI_BUF_SIZE_FACTOR 4
#define DEFAULT_MULTI_BUF_SIZE 1024*1024
#define SYNC_DATA_SIZE 16

static int custom;
static int iterations = 1000;
static int transfer_size = 1024;
static int max_credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *send_buf, *multi_recv_buf;
static size_t max_send_buf_size, multi_buf_size;

static struct fi_info hints;
static struct fi_domain_attr domain_hints;
static struct fi_ep_attr ep_hints;
static char *dst_addr, *src_addr;
static char *port = "9228";

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cq *rcq, *scq;
static struct fid_av *av;
static struct fid_mr *mr, *mr_multi_recv;;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_ctx_multi_recv *ctx_send;
struct fi_context fi_ctx_av;
struct fi_ctx_multi_recv *ctx_multi_recv;

struct fi_ctx_multi_recv {
	struct fi_context context;
	int index;
};

enum data_type {
	DATA = 1,
	CONTROL
} ;

int wait_for_send_completion(int num_completions)
{
	int ret;
	struct fi_cq_data_entry comp = {0};
	
	while (num_completions > 0) {
	 	memset(&comp, 0, sizeof(comp));
		ret = fi_cq_read(scq, &comp, 1);
		if (ret > 0) {
			num_completions--;
		} else if (ret < 0) {
			FI_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	}
	return 0;
}

int wait_for_recv_completion(void **recv_data, enum data_type type, 
		int num_completions)
{
	int ret;
	struct fi_cq_data_entry comp = {0};
	struct fi_ctx_multi_recv *header;
	int buf_index;

	while (num_completions > 0) {
	 	memset(&comp, 0, sizeof(comp));
		ret = fi_cq_read(rcq, &comp, 1);
		if (ret > 0) {
 			header = (struct fi_ctx_multi_recv *)comp.op_context;
			buf_index = header->index;
			// buffer is used up, post it again
			if(comp.flags & FI_MULTI_RECV) {
				ret = fi_recv(ep,
					   	multi_recv_buf + (multi_buf_size/2) * buf_index, 
						multi_buf_size/2, fi_mr_desc(mr_multi_recv), 
						remote_fi_addr, &ctx_multi_recv[buf_index].context);
				if (ret) { 
					FI_PRINTERR("fi_recv", ret);
					return ret;
				} 
			} else {			
				num_completions--;
				// check the completion event is for parameter exchanging
				// otherwise, do nothing for sync data and multi_recv transfer
				if(type == DATA) {
					*recv_data = comp.buf;
				}
			}
		} else if (ret < 0) {
			FI_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	}
	return 0;
}

static int send_msg(int size)
{
	int ret;
	ctx_send = (struct fi_ctx_multi_recv *) 
		malloc(sizeof(struct fi_ctx_multi_recv));
	ret = fi_send(ep, send_buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr, 
			&ctx_send->context);
	if (ret) {
		FI_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_send_completion(1);

	return ret;
}

static int sync_test(void)
{
	int ret;
	ret = dst_addr ? send_msg(SYNC_DATA_SIZE) :
	   	wait_for_recv_completion(NULL, CONTROL, 1);
	if (ret)
		return ret;

	return dst_addr ? wait_for_recv_completion(NULL, CONTROL, 1) : 
		send_msg(SYNC_DATA_SIZE);
}

static int post_multi_recv_buffer()
{
	int ret = 0;
	int i;
			
	ctx_multi_recv = (struct fi_ctx_multi_recv *) 
		malloc(sizeof(struct fi_ctx_multi_recv) * 2);

	// post buffers into halves so that we can 
	// repost one half when the other half is full
	for(i = 0; i < 2; i++) {		
		ctx_multi_recv[i].index = i;
		// post buffers for active messages 
		ret = fi_recv(ep, multi_recv_buf + (multi_buf_size/2) * i,
			   	multi_buf_size/2, fi_mr_desc(mr_multi_recv), remote_fi_addr, 
				&ctx_multi_recv[i].context);
		if (ret) { 
			FI_PRINTERR("fi_recv", ret);
			return ret;
		}
	}

	return ret;
}

static int send_multi_recv_msg()
{
	int ret, i;
	ret = 0;
	// send multi_recv data based on the transfer size
	for(i = 0; i < iterations; i++) {
		ctx_send = (struct fi_ctx_multi_recv *) 
			malloc(sizeof(struct fi_ctx_multi_recv));
		ret = fi_send(ep, send_buf, (size_t) transfer_size, fi_mr_desc(mr), 
				remote_fi_addr, &ctx_send->context);
		if (ret) {
			FI_PRINTERR("fi_send", ret);
			return ret;
		}
		ret = wait_for_send_completion(1);
	}
	return ret;
}

static int run_test(void)
{
	int ret;
	
	ret = sync_test();
	if (ret) {
		fprintf(stderr, "sync_test failed!\n");
		goto out;
	}
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	if(dst_addr) {
		ret = send_multi_recv_msg();
		if(ret){
		  fprintf(stderr, "send_multi_recv_msg failed!\n");
			goto out;
		}
	} else {
		// wait for all the receive completion events for multi_recv transfer
		ret = wait_for_recv_completion(NULL, CONTROL, iterations);
		if(ret){
		  fprintf(stderr, "wait_for_recv_completion failed\n");
			goto out;			
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &end);

	show_perf(test_name, transfer_size, iterations, &start, &end, 2);
	ret = 0;

out:
	return ret;
}

static void free_ep_res(void)
{
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&mr_multi_recv->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(ctx_send);
	free(ctx_multi_recv);
	free(send_buf);
	free(multi_recv_buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	int ret;
	
	// size of the local address that needs to be exchanged later
	int data_size = sizeof(size_t);

	// maximum size of the buffer that needs to allocated
	max_send_buf_size = MAX(MAX(data_size, SYNC_DATA_SIZE), transfer_size);
	
	if(max_send_buf_size > fi->ep_attr->max_msg_size) {
		fprintf(stderr, "transfer size is larger than the maximum size of the" 
				"data transfer supported by the provider\n");
		return -1;		
	}

	send_buf = malloc(max_send_buf_size);
	if (!send_buf) {
		fprintf(stderr, "Cannot allocate send_buf\n");
		return -1;
	}
	
	ret = fi_mr_reg(dom, send_buf, max_send_buf_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg for send_buf", ret);
		goto err1;
	}
	
	// set the multi buffer size to be allocated
	multi_buf_size = MAX(max_send_buf_size, DEFAULT_MULTI_BUF_SIZE) * 
		MULTI_BUF_SIZE_FACTOR;
	multi_recv_buf = malloc(multi_buf_size);
	if(!multi_recv_buf) {
		fprintf(stderr, "Cannot allocate multi_recv_buf\n");
		ret = -1;
		goto err1;
	}
	
	ret = fi_mr_reg(dom, multi_recv_buf, multi_buf_size, 0, 0, 1, 0, 
			&mr_multi_recv, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg for multi_recv_buf", ret);
		goto err2;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = max_credits << 1;
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: scq", ret);
		goto err3;
	}
	
	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: rcq", ret);
		goto err4;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = "addr to fi_addr map";

	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FI_PRINTERR("fi_av_open", ret);
		goto err5;
	}

	return 0;
err5:
	fi_close(&rcq->fid);
err4:
	fi_close(&scq->fid);
err3:
	fi_close(&mr_multi_recv->fid);	
err2:
	free(multi_recv_buf);
	fi_close(&mr->fid);
err1:
	free(send_buf);
	
	return ret;
}

static int set_min_multi_recv()
{
	int ret;
	
	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV, 
			(void *)&max_send_buf_size, sizeof(max_send_buf_size));
	if(ret)
		return ret;

	return 0;
}

static int bind_ep_res(void)
{
	int ret;

	ret = fi_bind(&ep->fid, &scq->fid, FI_SEND);
	if (ret) {
		FI_PRINTERR("fi_bind: scq", ret);
		return ret;
	}

	ret = fi_bind(&ep->fid, &rcq->fid, FI_RECV);
	if (ret) {
		FI_PRINTERR("fi_bind: rcq", ret);
		return ret;
	}
	
	ret = fi_bind(&ep->fid, &av->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_bind: av", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		FI_PRINTERR("fi_enable", ret);
		return ret;
	}

	return ret;
}

static int init_fabric(void)
{
	struct fi_info *fi;
	char *node;
	int ret;
	if (src_addr) {
		ret = getaddr(src_addr, NULL, (struct sockaddr **) &hints.src_addr,
			      (socklen_t *) &hints.src_addrlen);
		if (ret) {
			fprintf(stderr, "source address error %s\n", gai_strerror(ret));
			return ret;
		}
	}

	node = dst_addr ? dst_addr : src_addr;

	ret = fi_getinfo(FI_VERSION(1, 0), node, port, 0, &hints, &fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		return ret;
	}
	
	/* Get remote address */
	if (dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}
	
	// set FI_MULTI_RECV flag for all recv operations
	fi->rx_attr->op_flags = FI_MULTI_RECV;
	
	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
		goto err2;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err3;
	
	ret = bind_ep_res();
	if (ret)
		goto err4;
	
	// set the value of FI_OPT_MIN_MULTI_RECV which defines the minimum receive 
	// buffer space available when the receive buffer is automatically freed
	set_min_multi_recv();	

	// post the initial receive buffers to get MULTI_RECV data
	ret = post_multi_recv_buffer();

	if(ret)
		goto err4;
	return 0;

err4:
	free_ep_res();
err3:
	fi_close(&ep->fid);
err2:
	fi_close(&dom->fid);
err1:
	fi_close(&fab->fid);
err0:
	fi_freeinfo(fi);

	return ret;
}

static int init_av(void)
{
	int ret;
	void *recv_buf = NULL;
	
	if (dst_addr) {
		/* Get local address blob. Find the addrlen first. We set addrlen 
		 * as 0 and fi_getname will return the actual addrlen. */
		addrlen = 0;
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret != -FI_ETOOSMALL) {
			FI_PRINTERR("fi_getname", ret);
			return ret;
		}

		local_addr = malloc(addrlen);
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret) {
			FI_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, &fi_ctx_av);
		if (ret) {
			FI_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send local addr size */
		memcpy(send_buf, &addrlen, sizeof(size_t));
		ret = send_msg(sizeof(size_t));
		if (ret)
			return ret;
		
		/* Send local addr */
		memcpy(send_buf, local_addr, addrlen);
		ret = send_msg(addrlen);
		if (ret)
			return ret;

	} else {
		
		/* Get the size of remote address */
		ret = wait_for_recv_completion(&recv_buf, DATA, 1);
		if (ret)
			return ret;
		memcpy(&addrlen, recv_buf, sizeof(size_t));

		/* Get remote address */
		remote_addr = malloc(addrlen);
		ret = wait_for_recv_completion(&recv_buf, DATA, 1);
		if (ret)
			return ret;
		memcpy(remote_addr, recv_buf, addrlen);

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, &fi_ctx_av);
		if (ret) {
			FI_PRINTERR("fi_av_insert", ret);
			return ret;
		}
	}

	return ret;
}

static int run(void)
{
	int ret = 0;

	ret = init_fabric();
	if (ret)
		goto out;

	ret = init_av();
	if (ret)
		goto out;

	ret = run_test();	
	
	//synchronize before exiting
	ret = sync_test();
	if (ret) {
		fprintf(stderr, "sync_test failed!\n");
		goto out;
	}
out:
	fi_close(&ep->fid);
	free_ep_res();
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	while ((op = getopt(argc, argv, "d:n:p:s:I:S:")) != -1) {
		switch (op) {
		case 'd':
			dst_addr = optarg;
			break;
		case 'n':
			domain_hints.name = optarg;
			break;
		case 'p':
			port = optarg;
			break;
		case 's':
			src_addr = optarg;
			break;
		case 'I':
			custom = 1;
			iterations = atoi(optarg);
			break;
		case 'S':
			custom = 1;
			transfer_size = atoi(optarg);
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-d destination_address]\n");
			printf("\t[-n domain_name]\n");
			printf("\t[-p port_number]\n");
			printf("\t[-s source_address]\n");
			printf("\t[-I iterations]\n");
			printf("\t[-S transfer_size(in bytes)]\n");
			exit(1);
		}
	}

	hints.domain_attr = &domain_hints;
	hints.ep_attr = &ep_hints;
	hints.ep_type = FI_EP_RDM;
	hints.caps = FI_MSG | FI_MULTI_RECV | FI_BUFFERED_RECV;
	hints.mode = FI_CONTEXT;
	hints.addr_format = FI_FORMAT_UNSPEC;

	ret = run();
	return ret;
}
