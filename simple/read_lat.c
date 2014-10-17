/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the OpenIB.org BSD license
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
#include <strings.h>
#include <errno.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <netdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_atomic.h>
#include <shared.h>

static int custom;
static int size_option;
static int iterations = 1000;
static int transfer_size = 1000;
static int max_credits = 128;
static int warmup_iters = 128;
static char test_name[10] = "custom";
static struct timeval start, end;
static void *buf;
static void *rem_buf;
static uint64_t rem_key;
static size_t buffer_size;

static struct fi_info hints;
static struct fi_domain_attr domain_hints;
static struct fi_ep_attr ep_hints;
static char *dst_addr, *src_addr;
static char *port = "9228";

static struct fid_fabric *fab;
static struct fid_pep *pep;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_eq *cmeq;
static struct fid_cq *rcq, *scq;
static struct fid_mr *mr;

static void show_perf(void)
{
	char str[32];
	float usec;
	long long bytes;

	usec = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	bytes = (long long) iterations * transfer_size * 2;

	/* name size transfers iterations bytes seconds Gb/sec usec/xfer */
	fprintf(stderr, "%-10s", test_name);
	size_str(str, sizeof str, transfer_size);
	fprintf(stderr, "%-8s", str);
	cnt_str(str, sizeof str, 1);
	fprintf(stderr, "%-8s", str);
	cnt_str(str, sizeof str, iterations);
	fprintf(stderr, "%-8s", str);
	size_str(str, sizeof str, bytes);
	fprintf(stderr, "%-8s", str);
	fprintf(stderr, "%8.2fs%10.2f%11.2f\n",
		usec / 1000000., (bytes * 8) / (1000. * usec),
		(usec / iterations) );
}

static void init_test(int size)
{
	char sstr[5];

	size_str(sstr, sizeof sstr, size);
	snprintf(test_name, sizeof test_name, "%s_lat", sstr);
	transfer_size = size;
	iterations = size_to_count(transfer_size);
}

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep, buf, (size_t) size, fi_mr_desc(mr), NULL);
	if (ret) {
		fprintf(stderr, "fi_send %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	return wait_for_completion(scq, 1);
}

static int post_recv()
{
	int ret;

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), NULL);
	if (ret) {
		fprintf(stderr, "fi_recv %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	return ret;
}

static int read_data(size_t size)
{
	int ret;

	ret = fi_read(ep,  buf, size, fi_mr_desc(mr), 
		       (uint64_t)rem_buf, rem_key, NULL);
	if (ret) {
		fprintf(stderr, "fi_read %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	return 0;
}

static int warmup(int iters)
{
	int ret, i;
	for (i = 0; i < iters; i++) {
		if ((ret = read_data(16)) < 0)
			return ret;
		if ((ret = wait_for_completion(scq, 1)) < 0)
			return ret;
	}
	return 0;
}

static int run_test(void)
{
	int ret, i;

	ret = warmup(warmup_iters);
	if (ret)
		goto out;

	gettimeofday(&start, NULL);
	for (i = 0; i < iterations; i++) {
		ret = read_data(transfer_size);
		if (ret)
			goto out;
		ret = wait_for_completion(scq, 1);
		if (ret)
			goto out;
	}
	gettimeofday(&end, NULL);
	show_perf();
	ret = 0;

out:
	return ret;
}

static void free_lres(void)
{
	fi_close(&cmeq->fid);
}

static int alloc_cm_res(void)
{
	struct fi_eq_attr cm_attr;
	int ret;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.wait_obj = FI_WAIT_FD;
	ret = fi_eq_open(fab, &cm_attr, &cmeq, NULL);
	if (ret)
		fprintf(stderr, "fi_eq_open cm %s\n", fi_strerror(-ret));

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	int ret;

	buffer_size = !custom ? test_size[TEST_CNT - 1].size : transfer_size;
	buf = malloc(MAX(buffer_size, sizeof(uint64_t)));
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = max_credits << 1;
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		fprintf(stderr, "fi_eq_open send comp %s\n", fi_strerror(-ret));
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		fprintf(stderr, "fi_eq_open recv comp %s\n", fi_strerror(-ret));
		goto err2;
	}
	
	ret = fi_mr_reg(dom, buf, MAX(buffer_size, sizeof(uint64_t)), 
			FI_REMOTE_READ, 0, 0, 0, &mr, NULL);
	if (ret) {
		fprintf(stderr, "fi_mr_reg %s\n", fi_strerror(-ret));
		goto err3;
	}

	if (!cmeq) {
		ret = alloc_cm_res();
		if (ret)
			goto err4;
	}

	return 0;

err4:
	fi_close(&mr->fid);
err3:
	fi_close(&rcq->fid);
err2:
	fi_close(&scq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	ret = bind_fid(&ep->fid, &cmeq->fid, 0);
	if (ret)
		return ret;

	ret = bind_fid(&ep->fid, &scq->fid, FI_SEND|FI_READ);
	if (ret)
		return ret;

	ret = bind_fid(&ep->fid, &rcq->fid, FI_RECV);
	if (ret)
		return ret;

	ret = fi_enable(ep);
	if (ret)
		return ret;

	return ret;
}

static int server_listen(void)
{
	struct fi_info *fi;
	int ret;

	ret = fi_getinfo(FI_VERSION(1, 0), src_addr, port, FI_SOURCE, &hints, &fi);
	if (ret) {
		fprintf(stderr, "fi_getinfo %s\n", strerror(-ret));
		return ret;
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		fprintf(stderr, "fi_fabric %s\n", fi_strerror(-ret));
		goto err0;
	}

	ret = fi_pendpoint(fab, fi, &pep, NULL);
	if (ret) {
		fprintf(stderr, "fi_endpoint %s\n", fi_strerror(-ret));
		goto err1;
	}

	ret = alloc_cm_res();
	if (ret)
		goto err2;

	ret = bind_fid(&pep->fid, &cmeq->fid, 0);
	if (ret)
		goto err3;

	ret = fi_listen(pep);
	if (ret) {
		fprintf(stderr, "fi_listen %s\n", fi_strerror(-ret));
		goto err3;
	}

	fi_freeinfo(fi);
	return 0;
err3:
	free_lres();
err2:
	fi_close(&pep->fid);
err1:
	fi_close(&fab->fid);
err0:
	fi_freeinfo(fi);
	return ret;
}

static int server_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *info = NULL;
	ssize_t rd;
	int ret;

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		fprintf(stderr, "fi_eq_sread %zd %s\n", rd, fi_strerror((int) -rd));
		return (int) rd;
	}

	if (event != FI_CONNREQ) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err1;
	}

	info = entry.info;
	ret = fi_domain(fab, info, &dom, NULL);
	if (ret) {
		fprintf(stderr, "fi_fdomain %s\n", fi_strerror(-ret));
		goto err1;
	}


	ret = fi_endpoint(dom, info, &ep, NULL);
	if (ret) {
		fprintf(stderr, "fi_endpoint for req %s\n", fi_strerror(-ret));
		goto err1;
	}

	ret = alloc_ep_res(info);
	if (ret)
		 goto err2;

	ret = bind_ep_res();
	if (ret)
		goto err3;

	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		fprintf(stderr, "fi_accept %s\n", fi_strerror(-ret));
		goto err3;
	}

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
 	if (rd != sizeof entry) {
		fprintf(stderr, "fi_eq_sread %zd %s\n", rd, fi_strerror((int) -rd));
		goto err3;
 	}

	if (event != FI_COMPLETE || entry.fid != &ep->fid) {
 		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
 		ret = -FI_EOTHER;
 		goto err3;
 	}
 
 	fi_freeinfo(info);
 	return 0;

err3:
	free_ep_res();
err2:
	fi_close(&ep->fid);
err1:

 	fi_reject(pep, info->connreq, NULL, 0);
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

	if (src_addr) {
		ret = getaddr(src_addr, NULL, (struct sockaddr **) &hints.src_addr,
			      (socklen_t *) &hints.src_addrlen);
		if (ret)
			fprintf(stderr, "source address error %s\n", gai_strerror(ret));
	}

	ret = fi_getinfo(FI_VERSION(1, 0), dst_addr, port, 0, &hints, &fi);
	if (ret) {
		fprintf(stderr, "fi_getinfo %s\n", strerror(-ret));
		goto err0;
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		fprintf(stderr, "fi_fabric %s\n", fi_strerror(-ret));
		goto err1;
	}

 	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		fprintf(stderr, "fi_fdomain %s %s\n", fi_strerror(-ret),
			fi->domain_attr->name);
		goto err2;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		fprintf(stderr, "fi_endpoint %s\n", fi_strerror(-ret));
		goto err3;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err4;

	ret = bind_ep_res();
	if (ret)
		goto err5;

	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		fprintf(stderr, "fi_connect %s\n", fi_strerror(-ret));
		goto err5;
	}

 	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		fprintf(stderr, "fi_eq_condread %zd %s\n", rd, fi_strerror((int) -rd));
		return (int) rd;
	}

 	if (event != FI_COMPLETE || entry.fid != &ep->fid) {
 		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
 			event, entry.fid, ep);
 		ret = -FI_EOTHER;
 		goto err1;
 	}

	if (hints.src_addr)
		free(hints.src_addr);
	fi_freeinfo(fi);
	return 0;

err5:
	free_ep_res();
err4:
	fi_close(&ep->fid);
err3:
	fi_close(&dom->fid);
err2:
	fi_close(&fab->fid);
err1:
	fi_freeinfo(fi);
err0:
	if (hints.src_addr)
		free(hints.src_addr);
	return ret;
}

static void exchange_params(void)
{
	*((uint64_t *) buf) = (uint64_t)buf;
	*(((uint64_t *) buf + 1)) = fi_mr_key(mr);
	post_recv();
	send_msg(sizeof(uint64_t *) * 2);
	wait_for_completion(rcq, 1);

	rem_buf = (void *) (*((uint64_t *) buf));
	rem_key = (uint64_t) (*(((uint64_t *) buf + 1)));
}

static void synchronize(void)
{
	if (dst_addr) {
		post_recv();
		wait_for_completion(rcq, 1);
	} else {
		send_msg(sizeof(uint64_t *));
	}
}

static int run(void)
{
	int i, ret = 0;

	if (!dst_addr) {
		ret = server_listen();
		if (ret)
			return ret;
	}

	fprintf(stderr, "%-10s%-8s%-8s%-8s%-8s%8s %10s%13s\n",
	       "name", "bytes", "xfers", "iters", "total", "time", "Gb/sec", "usec/xfer");

	ret = dst_addr ? client_connect() : server_connect();
	if (ret)
		return ret;

	exchange_params();

	if (!custom) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > size_option)
				continue;
			init_test(test_size[i].size);
			ret = run_test();
			if(ret)
				goto out;
		}
	} else {
		ret = run_test();
	}
	synchronize();

out:
	fi_shutdown(ep, 0);
	fi_close(&ep->fid);
	free_ep_res();
	if (!dst_addr)
		free_lres();
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	while ((op = getopt(argc, argv, "d:n:p:s:C:I:w:S:")) != -1) {
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
		case 'w':
			warmup_iters = atoi(optarg);
			break;
		case 'S':
			if (!strncasecmp("all", optarg, 3)) {
				size_option = 1;
			} else {
				custom = 1;
				transfer_size = atoi(optarg);
			}
			break;
		default:
			fprintf(stderr, "usage: %s\n", argv[0]);
			fprintf(stderr, "\t[-d destination_address]\n");
			fprintf(stderr, "\t[-n domain_name]\n");
			fprintf(stderr, "\t[-p port_number]\n");
			fprintf(stderr, "\t[-s source_address]\n");
			fprintf(stderr, "\t[-I iterations]\n");
			fprintf(stderr, "\t[-w warmup iterations]\n");
			fprintf(stderr, "\t[-S transfer_size or 'all']\n");
			exit(1);
		}
	}

	hints.domain_attr = &domain_hints;
	hints.ep_attr = &ep_hints;
	hints.ep_type = FI_EP_MSG;
	hints.caps = FI_MSG | FI_RMA;
	hints.mode = FI_LOCAL_MR | FI_PROV_MR_KEY;
	hints.addr_format = FI_SOCKADDR;

	ret = run();
	return ret;
}
