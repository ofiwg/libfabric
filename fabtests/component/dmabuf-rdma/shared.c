
/*
 * Copyright (c) 2021-2026 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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
  * This shared file is for fi- tests and is not yet used by the ib only tests.
  */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <arpa/inet.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_errno.h>
#include <level_zero/ze_api.h>
#include "shared.h"
#include "util.h"

struct options_t options = {
	.bidirectional	= false,
	.proxy_block	= MAX_SIZE,
	.mapping_str	= NULL,
	.ep_type	= FI_EP_RDM,
	.max_size 	= MAX_SIZE,
	.loc1		= MALLOC,
	.loc2		= MALLOC,
	.iters		= 1000,
	.prov_name	= NULL,
	.use_proxy	= false,
	.max_ranks	= 1,
	.rank		= 0,
	.port		= 12345,
	.msg_size	= 0,
	.test_type	= READ,
	.verbose	= false,
	.verify		= 0,
	.server_name	= NULL,
	.client		= false,
	.sockfd		= -1,
	.use_sync_ofi	= false,
	.prepost	= false,
	.buf_location	= MALLOC,
	.use_raw_key	= 0,
	.algo		= POINT_TO_POINT,
};

struct nic		nics[MAX_NICS];
struct business_card 	me;
struct business_card	*peers;

struct context_pool	*context_pool;

int			TX_DEPTH = 128;
int			RX_DEPTH = 256;

void remove_characters(char *str, char *removes)
{
	int i, j;
	for (i = 0; i < strlen(str); i++) {
		for (j = 0; j < strlen(removes); j++) {
			if (str[i] == removes[j]) {
				memmove(&str[i], &str[i + 1], strlen(&str[i]));
				i--;
				break;
			}
		}
	}
}

void buf_location_str(int loc, char *str, int len)
{
	switch (loc) {
		case MALLOC:
			snprintf(str, len, "malloc");
			break;
		case HOST:
			snprintf(str, len, "host");
			break;
		case DEVICE:
			snprintf(str, len, "device");
			break;
		case SHARED:
			snprintf(str, len, "shared");
			break;
		default:
			snprintf(str, len, "unknown");
			break;
	}
}

void parse_buf_location(char *string, int *loc1, int *loc2, int default_loc)
{
	char *s;
	char *saveptr;

	s = strtok_r(string, ":", &saveptr);
	if (s) {
		*loc1 = string_to_location(s, default_loc);
		s = strtok_r(NULL, ":", &saveptr);
		if (s)
			*loc2 = string_to_location(s, default_loc);
		else
			*loc2 = *loc1;
	} else {
		*loc1 = *loc2 = default_loc;
	}
}

size_t parse_size(char *string)
{
	size_t size = MAX_SIZE;
	char unit = '\0';

	sscanf(string, "%zd%c", &size, &unit);

	if (unit == 'k' || unit == 'K')
		size *= 1024;
	else if (unit == 'm' || unit == 'M')
		size *= 1024 * 1024;
	else if (unit == 'g' || unit == 'G')
		size *= 1024 * 1024 * 1024;

	return size;
}

void print_nic_info(void)
{
	int i;

	for (i = 0; i < options.num_mappings; i++) {
		printf("NIC %d:\n"
		       "\tdomain %s\n"
		       "\tGPU %d<.%d>\n"
		       "\tfabric = %p\n"
		       "\tep = %p\n"
		       "\tav = %p\n"
		       "\tcq = %p\n",
		       i,
		       nics[i].mapping.domain_name,
		       nics[i].mapping.gpu.dev_num,
		       nics[i].mapping.gpu.subdev_num,
		       nics[i].fabric,
		       nics[i].ep,
		       nics[i].av,
		       nics[i].cq);
	}
}

int wait_conn_req(struct fid_eq *eq, struct fi_info **fi)
{
	struct fi_eq_cm_entry entry;
	struct fi_eq_err_entry err_entry;
	uint32_t event;
	ssize_t ret;

	ret = fi_eq_sread(eq, &event, &entry, sizeof(entry), -1, 0);
	if (ret != sizeof entry) {
		printf("%s: fi_eq_sread returned %ld, expecting %ld\n",
			__func__, ret, sizeof(entry));
		if (ret == -FI_EAVAIL) {
			fi_eq_readerr(eq, &err_entry, 0);
			printf("%s: error %d prov_errno %d\n", __func__,
				err_entry.err, err_entry.prov_errno);
		}
		return (int) ret;
	}

	*fi = entry.info;
	if (event != FI_CONNREQ) {
		printf("%s: unexpected CM event %d\n", __func__, event);
		return -FI_EOTHER;
	}

	return 0;
}

int wait_connected(struct fid_ep *ep, struct fid_eq *eq)
{
	struct fi_eq_cm_entry entry;
	struct fi_eq_err_entry err_entry;
	uint32_t event;
	ssize_t ret;

	ret = fi_eq_sread(eq, &event, &entry, sizeof(entry), -1, 0);
	if (ret != sizeof(entry)) {
		printf("%s: fi_eq_sread returns %ld, expecting %ld\n",
			__func__, ret, sizeof(entry));
		if (ret == -FI_EAVAIL) {
			fi_eq_readerr(eq, &err_entry, 0);
			printf("%s: error %d prov_errno %d\n", __func__,
				err_entry.err, err_entry.prov_errno);
		}
		return (int)ret;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		printf("%s: unexpected CM event %d fid %p (ep %p)\n",
			__func__, event, entry.fid, ep);
		return -FI_EOTHER;
	}

	return 0;
}

void init_buf(int nic, size_t buf_size, char c)
{
	int page_size = sysconf(_SC_PAGESIZE);
	int j;

	for (j = 0; j < options.max_ranks; j++) {
		if (!xe_alloc_buf(page_size, buf_size, options.buf_location,
				  &nics[nic].mapping.gpu,
				  &nics[nic].bufs.xe_buf[j])) {
			fprintf(stderr, "Couldn't allocate work buf.\n");
			exit(-1);
		}

		xe_set_buf(nics[nic].bufs.xe_buf[j].buf, c,
			   nics[nic].bufs.xe_buf[j].size,
			   options.buf_location, &nics[nic].mapping.gpu);

		if (options.buf_location == DEVICE &&
			options.use_proxy) {
			if (options.verbose)
				printf("Allocating proxy buffer of "
					"size %d on host\n",
					options.proxy_block);
			if (!xe_alloc_buf(page_size, options.proxy_block,
					  HOST, &nics[nic].mapping.gpu,
					  &nics[nic].proxy_bufs.xe_buf[j])) {
				fprintf(stderr, "Couldn't allocate "
					"proxy buf.\n");
				exit(-1);
			}
		}
	}

	if (!xe_alloc_buf(page_size, page_size, MALLOC,
			  &nics[nic].mapping.gpu,
			  &nics[nic].sync_buf.xe_buf[0])) {
		fprintf(stderr, "Couldn't allocate sync buf.\n");
		exit(-1);
	}
}

void check_buf(int nic, size_t size, char c, int rank)
{
	unsigned long mismatches = 0;
	int i;
	char *bounce_buf;

	bounce_buf = malloc(size);
	if (!bounce_buf) {
		perror("malloc bounce buffer");
		return;
	}

	xe_copy_buf(bounce_buf, nics[nic].bufs.xe_buf[rank].buf, size,
		    &nics[nic].mapping.gpu);

	for (i = 0; i < size; i++)
		if (bounce_buf[i] != c) {
			mismatches++;
			if (mismatches < 10)
				printf("value at [%d] is '%c'(0x%02x), "
				       "expecting '%c'(0x%02x)\n", i,
				       bounce_buf[i], bounce_buf[i], c, c);
		}

	free(bounce_buf);

	if (mismatches)
		printf("%lu mismatches found\n", mismatches);
	else
		if (options.verbose)
			printf("all %lu bytes are correct.\n", size);
}

void free_buf(void)
{
	int i, j;

	for (i = 0; i < options.num_mappings; i++) {
		for (j = 0; j < options.max_ranks; j++) {
			if (nics[i].bufs.xe_buf[j].buf)
				xe_free_buf(&nics[i].bufs.xe_buf[j]);

			if (options.use_proxy &&
			    nics[i].proxy_bufs.xe_buf[j].buf)
				xe_free_buf(&nics[i].proxy_bufs.xe_buf[j]);
		}
		xe_free_buf(&nics[i].sync_buf.xe_buf[0]);
	}
}

static int get_info(int nic, struct fi_info **fi)
{
	struct fi_info *hints;
	int version;
	char port_name[16];
	EXIT_ON_NULL((hints = fi_allocinfo()));

	hints->ep_attr->type = options.ep_type;
	hints->ep_attr->tx_ctx_cnt = 1;
	hints->ep_attr->rx_ctx_cnt = 1;
	hints->tx_attr->size = TX_DEPTH;
	hints->rx_attr->size = RX_DEPTH;
	if (options.prov_name)
		hints->fabric_attr->prov_name = strdup(options.prov_name);
	hints->caps = FI_MSG | FI_RMA | FI_TAGGED;
	if (options.buf_location != MALLOC)
		hints->caps |= FI_HMEM;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
	hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_PROV_KEY |
				      FI_MR_VIRT_ADDR | FI_MR_LOCAL |
				      FI_MR_HMEM | FI_MR_ENDPOINT | FI_MR_RAW;
	if (nics[nic].mapping.domain_name)
		hints->domain_attr->name = strdup(
						 nics[nic].mapping.domain_name);

	sprintf(port_name, "%d", options.port);
	version = FI_VERSION(2, 0);
	if (options.ep_type == FI_EP_MSG) {
		if (options.client)
			CHECK_ERROR(fi_getinfo(version, options.server_name,
					       port_name, 0, hints, fi));
		else
			CHECK_ERROR(fi_getinfo(version, options.server_name,
					       port_name, FI_SOURCE, hints,
					       fi));
	} else {
		CHECK_ERROR(fi_getinfo(version, NULL, NULL, 0, hints, fi));
	}

	fi_freeinfo(hints);
	return 0;

err_out:
	fi_freeinfo(hints);
	fprintf(stderr, "\033[0;31mFrom Developer: You should probably check "
		"nic:gpu mapping and/or provider spelling\n\033[0m");
	return errno;
}

static int open_fabric_resources(int nic, struct fi_info *fi)
{
	struct fi_info *fi_pep = NULL;
	struct fid_fabric *fabric = NULL;
	struct fid_eq *eq = NULL;
	struct fid_domain *domain = NULL;
	struct fid_ep *ep = NULL;
	struct fid_av *av = NULL;
	struct fid_cq *cq = NULL;
	struct fid_pep *pep = NULL;
	struct fi_cq_attr cq_attr = { .format = FI_CQ_FORMAT_CONTEXT };
	struct fi_av_attr av_attr = {};
	struct fi_eq_attr eq_attr = { .wait_obj = FI_WAIT_UNSPEC };

	if (options.ep_type == FI_EP_RDM || options.client)
		printf("Using OFI device: %s (%s)\n",
			fi->fabric_attr->prov_name, fi->domain_attr->name);

	CHECK_ERROR(fi_fabric(fi->fabric_attr, &fabric, NULL));
	if (options.ep_type == FI_EP_MSG) {
		CHECK_ERROR(fi_eq_open(fabric, &eq_attr, &eq, NULL));
		if (!options.client) {
			fi_pep = fi;
			CHECK_ERROR(fi_passive_ep(fabric, fi_pep, &pep,
				      NULL));
			CHECK_ERROR(fi_pep_bind(pep, (fid_t)eq, 0));
			CHECK_ERROR(fi_listen(pep));
			CHECK_ERROR(wait_conn_req(eq, &fi));
			printf("Using OFI device: %s (%s)\n",
				fi_pep->fabric_attr->prov_name,
				fi->domain_attr->name);
		}
	}
	CHECK_ERROR(fi_domain(fabric, fi, &domain, NULL));
	CHECK_ERROR(fi_endpoint(domain, fi, &ep, NULL));
	CHECK_ERROR(fi_cq_open(domain, &cq_attr, &cq, NULL));
	if (options.ep_type == FI_EP_RDM) {
		CHECK_ERROR(fi_av_open(domain, &av_attr, &av, NULL));
		CHECK_ERROR(fi_ep_bind(ep, (fid_t)av, 0));
	} else {
		CHECK_ERROR(fi_ep_bind(ep, (fid_t)eq, 0));
	}
	CHECK_ERROR(fi_ep_bind(ep, (fid_t)cq,
		    (FI_TRANSMIT | FI_RECV)));
	CHECK_ERROR(fi_enable(ep));

	if (options.ep_type == FI_EP_MSG) {
		if (options.client)
			CHECK_ERROR(fi_connect(ep, fi->dest_addr, NULL, 0));
		else
			CHECK_ERROR(fi_accept(ep, NULL, 0));
		CHECK_ERROR(wait_connected(ep, eq));
	}

	nics[nic].fi = fi;
	nics[nic].fi_pep = fi_pep;
	nics[nic].fabric = fabric;
	nics[nic].eq = eq;
	nics[nic].domain = domain;
	nics[nic].pep = pep;
	nics[nic].ep = ep;
	nics[nic].av = av;
	nics[nic].cq = cq;
	return 0;
err_out:
	return -1;
}

static int register_bufs(int nic, struct fi_info *fi)
{

	int j;
	static int req_key = 1;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_attr = {};
	struct iovec iov;

	for (j = 0; j < options.max_ranks; j++) {
		mr = NULL;
		iov.iov_base = nics[nic].bufs.xe_buf[j].buf;
		iov.iov_len = nics[nic].bufs.xe_buf[j].size;
		mr_attr.mr_iov = &iov;
		mr_attr.iov_count = 1;
		mr_attr.access = FI_REMOTE_READ | FI_REMOTE_WRITE |
				 FI_READ | FI_WRITE | FI_SEND | FI_RECV;
		mr_attr.requested_key = req_key++;
		mr_attr.iface = nics[nic].bufs.xe_buf[j].location ==
				MALLOC ? FI_HMEM_SYSTEM : FI_HMEM_ZE;
		mr_attr.device.ze = nics[nic].mapping.gpu.dev_num;
		CHECK_ERROR(fi_mr_regattr(nics[nic].domain, &mr_attr, 0, &mr));

		if (fi->domain_attr->mr_mode & FI_MR_ENDPOINT) {
			CHECK_ERROR(fi_mr_bind(mr, (fid_t)nics[nic].ep, 0));
			CHECK_ERROR(fi_mr_enable(mr));
		}

		nics[nic].bufs.mr[j] = mr;

		if (options.buf_location == DEVICE &&
			options.use_proxy) {
			mr = NULL;
			iov.iov_base = nics[nic].proxy_bufs.xe_buf[j].buf;
			iov.iov_len = nics[nic].proxy_bufs.xe_buf[j].size;
			mr_attr.mr_iov = &iov;
			mr_attr.iov_count = 1;
			mr_attr.access = FI_REMOTE_READ | FI_REMOTE_WRITE |
					 FI_READ | FI_WRITE | FI_SEND |
					 FI_RECV;
			mr_attr.requested_key = req_key++;
			mr_attr.iface = FI_HMEM_ZE;
			mr_attr.device.ze = nics[nic].mapping.gpu.dev_num;
			CHECK_ERROR(fi_mr_regattr(nics[nic].domain, &mr_attr,
				    0, &mr));

			if (fi->domain_attr->mr_mode & FI_MR_ENDPOINT) {
				CHECK_ERROR(fi_mr_bind(mr, (fid_t)nics[nic].ep,
						       0));
				CHECK_ERROR(fi_mr_enable(mr));
			}

			nics[nic].proxy_bufs.mr[j] = mr;
		}
	}

	if (!nic) {
		mr = NULL;
		iov.iov_base = nics[nic].sync_buf.xe_buf[0].buf;
		iov.iov_len = nics[nic].sync_buf.xe_buf[0].size;
		mr_attr.mr_iov = &iov;
		mr_attr.iov_count = 1;
		mr_attr.access = FI_SEND | FI_RECV;
		mr_attr.requested_key = req_key++;
		mr_attr.iface = FI_HMEM_SYSTEM;
		mr_attr.device.ze = nics[nic].mapping.gpu.dev_num;
		CHECK_ERROR(fi_mr_regattr(nics[nic].domain, &mr_attr, 0, &mr));

		if (fi->domain_attr->mr_mode & FI_MR_ENDPOINT) {
			CHECK_ERROR(fi_mr_bind(mr, (fid_t)nics[nic].ep, 0));
			CHECK_ERROR(fi_mr_enable(mr));
		}

		nics[nic].sync_buf.mr[0] = mr;
	}

	return 0;
err_out:
	return -1;
}

int init_nic(int nic)
{
	struct fi_info *fi;

	CHECK_ERROR(get_info(nic, &fi));
	CHECK_ERROR(open_fabric_resources(nic, fi));

	if (fi->tx_attr->size < TX_DEPTH) {
		if (options.verbose)
			printf("Reducing tx size from %d to %zu to avoid "
			       "overrunning CQ\n", TX_DEPTH, fi->tx_attr->size);
		TX_DEPTH = fi->tx_attr->size;
	}
	if (fi->rx_attr->size < RX_DEPTH) {
		if (options.verbose)
			printf("Reducing rx size from %d to %zu to avoid "
			       "overrunning CQ\n", RX_DEPTH, fi->rx_attr->size);
		RX_DEPTH = fi->rx_attr->size;
	}

	init_buf(nic, options.max_size, 'A');

	if (options.test_type == SEND &&
	    !(fi->domain_attr->mr_mode & (FI_MR_HMEM | FI_MR_LOCAL))) {
		printf("Local MR registration skipped.\n");
		return 0;
	}

	if (fi->domain_attr->mr_mode & FI_MR_RAW)
		options.use_raw_key = 1;

	CHECK_ERROR(register_bufs(nic, fi));
	return 0;

err_out:
	return -1;
}

void show_business_card(struct business_card *bc, char *name)
{
	int i, j;

	printf("%s:\n\tnum_nics %d\n\tuse_raw_key %d\n", name,
	       bc->num_nics, bc->use_raw_key);
	for (i = 0; i < bc->num_nics; i++) {
		printf("\t[NIC %d] %lx:%lx:%lx:%lx\n", i,
		       bc->nic[i].ep_name.one, bc->nic[i].ep_name.two,
		       bc->nic[i].ep_name.three, bc->nic[i].ep_name.four);
		for (j = 0; j < options.max_ranks; j++) {
			printf("\t\t[GPU %d] [Buf %d]\taddr %lx rkeys (",
				bc->nic[i].dev_num, j, bc->nic[i].bufs[j].addr);
			printf("%lx", bc->nic[i].bufs[j].rkey);
			printf(")\n");
			if (bc->use_raw_key) {
				printf("\t\t\t\traw_key (");
				for (int b = 0;
				     b < bc->nic[i].bufs[j].raw_key.size;
				     b++)
					printf("%02x", bc->nic[i].bufs[j].
					       raw_key.key[b]);
				printf(")\n");
			}
		}
	}
	printf("\t[SYNC BUF]\taddr %lx rkeys (%lx)\n", bc->sync_buf.addr,
	       bc->sync_buf.rkey);
	printf("\n");
}

int fill_in_my_business_card(void)
{
	int i, k;
	size_t len;

	me.algo = options.algo;
	me.num_nics = options.num_mappings;
	for (k = 0; k < options.num_mappings; k++) {
		for (i = 0; i < options.max_ranks; i++) {
			me.nic[k].dev_num = nics[k].mapping.gpu.dev_num;
			me.nic[k].bufs[i].addr = (uint64_t)nics[k].bufs.
							   xe_buf[i].buf;
			len = sizeof(me.nic[k].ep_name);
			CHECK_ERROR(fi_getname((fid_t)nics[k].ep,
						 &me.nic[k].ep_name, &len));
			me.nic[k].bufs[i].rkey = nics[k].bufs.mr[i] ?
						 fi_mr_key(nics[k].bufs.mr[i]) :
						 0;
			if (options.use_raw_key && nics[k].bufs.mr[i]) {
				me.nic[k].bufs[i].raw_key.size =
							       MAX_RAW_KEY_SIZE;
				fi_mr_raw_attr(nics[k].bufs.mr[i],
				      (void *)(uintptr_t)me.nic[k].bufs[i].addr,
				      me.nic[k].bufs[i].raw_key.key,
				      &me.nic[k].bufs[i].raw_key.size, 0);
			}
		}
	}
	me.use_raw_key = options.use_raw_key;
	me.sync_buf.addr = (uint64_t)nics[0].sync_buf.xe_buf[0].buf;
	me.sync_buf.rkey = nics[0].sync_buf.mr[0] ?
			   fi_mr_key(nics[0].sync_buf.mr[0]) : 0;
	if (options.use_raw_key && nics[0].sync_buf.mr[0]) {
		me.sync_buf.raw_key.size = MAX_RAW_KEY_SIZE;
		fi_mr_raw_attr(nics[0].sync_buf.mr[0],
			       (void *)(uintptr_t)me.sync_buf.addr,
			       me.sync_buf.raw_key.key,
			       &me.sync_buf.raw_key.size, 0);
	}

	if (!options.client) {
		peers[0] = me; /* Server is always rank 0 */
		me.rank = 0;
		options.rank = 0;
	}

	return 0;
err_out:
	return -1;
}

int exchange_business_cards(void)
{
	int j, k, assigned_rank;
	char peer_str[9];

	if (!options.client) {
		if (options.verbose)
			printf("Server: Collecting business cards from %d "
			       "clients...\n", options.max_ranks - 1);

		for (k = 1; k < options.max_ranks; k++) {
			assigned_rank = k;
			if (options.verbose)
				printf("Server: Waiting for business card "
				       "from client %d...\n", k);

			CHECK_ERROR(send_to_socket(client_sockets[k],
				    sizeof(int), &assigned_rank));

			CHECK_ERROR(recv_from_socket(client_sockets[k],
						sizeof(struct business_card),
						&peers[k])
			);
			snprintf(peer_str, sizeof(peer_str), "Client %d", k);
			if (options.verbose)
				show_business_card(&peers[k], peer_str);
		}

		if (options.verbose)
			printf("Server: All clients connected. Broadcasting "
			       "complete peer list...\n");

		for (k = 1; k < options.max_ranks; k++) {
			if (options.verbose)
				printf("Server: Sending peer list to client "
				       "%d...\n", k);

			for (j = 0; j < options.max_ranks; j++) {
				if (k == j)
					continue;

				CHECK_ERROR(send_to_socket(client_sockets[k],
					    sizeof(struct business_card),
					    &peers[j]));
			}
		}

		if (options.verbose)
			printf("Server: Peer exchange complete. All %d peers "
			       "synchronized.\n", options.max_ranks);

	} else {
		CHECK_ERROR(recv_from_socket(options.sockfd, sizeof(int),
			    &assigned_rank));

		options.rank = assigned_rank;
		me.rank = assigned_rank;
		peers[options.rank] = me;

		if (options.verbose)
			printf("Client: Assigned rank %d by server\n",
			       options.rank);

		if (options.verbose)
			printf("Client %d: Sending business card to server..."
			       "\n", options.rank);

		CHECK_ERROR(send_to_socket(options.sockfd,
					   sizeof(struct business_card), &me));

		if (options.verbose)
			printf("Client %d: Waiting for complete peer list from "
			       "server...\n", options.rank);

		for (k = 0; k < options.max_ranks; k++) {
			if (k == options.rank)
				continue;

			CHECK_ERROR(recv_from_socket(options.sockfd,
				    sizeof(struct business_card), &peers[k]));
			snprintf(peer_str, 9, "Peer %d", k);
			if (options.verbose)
				show_business_card(&peers[k], peer_str);
		}

		if (options.verbose)
			printf("Client %d: Received complete peer list "
			       "(%d peers total).\n", options.rank,
			       options.max_ranks);
	}

	return 0;
err_out:
	errno = EIO;
	return EIO;
}

int init_ofi(void)
{
	int i, j, k, h, num_nics;

	EXIT_ON_NULL((context_pool = init_context_pool(TX_DEPTH + 1)));

	for (num_nics = 0; num_nics < options.num_mappings; num_nics++)
		CHECK_ERROR(init_nic(num_nics));

	CHECK_ERROR(fill_in_my_business_card());
	show_business_card(&me, options.client ? "Client" : "Server");
	CHECK_ERROR(exchange_business_cards());

	for (k = 0; k < options.max_ranks; k++) {
		if (k == options.rank)
			continue;

		if (me.use_raw_key != peers[k].use_raw_key) {
			printf("Error: Use of raw key doesn't match with "
			       "peer %d (mine: %d, peer: %d). Exiting\n",
			       k, me.use_raw_key, peers[k].use_raw_key);
			errno = EINVAL;
			goto err_out;
		}

		if (me.algo != peers[k].algo) {
			printf("Error: Algorithm doesn't match with peer %d. "
			       "Exiting\n", k);
			errno = EINVAL;
			goto err_out;
		}
	}

	for (k = 0; k < me.num_nics; k++) {
		if (nics[k].fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
			continue;

		for (i = 0; i < options.max_ranks; i++) {
			if (i == options.rank)
				continue;

			for (j = 0; j < peers[i].num_nics; j++)
				for (h = 0; h < options.max_ranks; h++)
					peers[i].nic[j].bufs[h].addr = 0;
		}
	}

	if (!options.use_raw_key)
		goto after_map_raw_key;

	for (i = 0; i < me.num_nics; i++) {
		for (j = 0; j < options.max_ranks; j++) {
			if (j == options.rank)
				continue;

			for (k = 0; k < peers[k].num_nics; k++) {
				for (h = 0; h < options.max_ranks; h++) {
					if (!peers[j].nic[k].bufs[h].rkey)
							continue;

					CHECK_ERROR(fi_mr_map_raw(
					   nics[i].domain,
					   peers[j].nic[k].bufs[h].addr,
					   peers[j].nic[k].bufs[h].raw_key.key,
					   peers[j].nic[k].bufs[h].raw_key.size,
					   &peers[j].nic[k].bufs[h].rkey,
					   0)
					);
				}
			}
		}
	}

after_map_raw_key:
	if (options.ep_type == FI_EP_MSG)
		goto print_done;

	for (i = 0; i < me.num_nics; i++) {
		for (j = 0; j < options.max_ranks; j++) {
			if (j == options.rank)
				continue;

			for (k = 0; k < peers[j].num_nics; k++) {
				CHECK_NEG_ERROR(fi_av_insert(nics[i].av,
						&peers[j].nic[k].ep_name, 1,
						&nics[i].peer_addr[j][k], 0,
						NULL));
			}
		}
	}

print_done:
	if (options.verbose)
		printf("OFI initialization complete for %s (rank %d of %d)\n",
		       options.client ? "client" : "server", options.rank,
		       options.max_ranks - 1);
	return 0;
err_out:
	return -1;
}

void finalize_ofi(void)
{
	int h, i, j, k, err;

	if (!options.use_raw_key)
		goto skip_raw_key_cleanup;

	for (i = 0; i < me.num_nics; i++) {
		for (j = 0; j < options.max_ranks; j++) {
			if (j == options.rank)
				continue;

			for (k = 0; k < peers[j].num_nics; k++) {
				for (h = 0; h < options.max_ranks; h++) {
					if (!peers[j].nic[k].bufs[h].rkey)
						continue;

					err = fi_mr_unmap_key(nics[i].domain,
						peers[j].nic[k].bufs[h].rkey);
					if (err)
						fprintf(stderr,
							"fi_mr_unmap_key "
							"failed: %s\n",
							fi_strerror(-err));
				}
			}
		}
	}

skip_raw_key_cleanup:
	for (i = 0; i < me.num_nics; i++) {
		if (nics[i].sync_buf.mr[0])
			fi_close((fid_t)nics[i].sync_buf.mr[0]);

		for (j = 0; j < options.max_ranks; j++) {
			if (nics[i].proxy_bufs.mr[j])
				fi_close((fid_t)nics[i].proxy_bufs.mr[j]);

			if (nics[i].bufs.mr[j])
				fi_close((fid_t)nics[i].bufs.mr[j]);
		}

		if (nics[i].ep)
			fi_close((fid_t)nics[i].ep);

		if (nics[i].av)
			fi_close((fid_t)nics[i].av);

		if (nics[i].cq)
			fi_close((fid_t)nics[i].cq);

		if (nics[i].domain)
			fi_close((fid_t)nics[i].domain);

		if (nics[i].pep)
			fi_close((fid_t)nics[i].pep);

		if(nics[i].eq)
			fi_close((fid_t)nics[i].eq);

		if (nics[i].fabric)
			fi_close((fid_t)nics[i].fabric);

		if (nics[i].fi)
			fi_freeinfo(nics[i].fi);

		if (nics[i].fi_pep)
			fi_freeinfo(nics[i].fi_pep);
	}

	if (options.verbose)
		printf("OFI cleanup complete for %s (rank %d)\n",
		       options.client ? "client" : "server", options.rank);

	return;
}

int process_completions(int nic, int *pending, int *completed)
{
	int n0 = 0;
	int n, k, i;
	struct fi_cq_entry wc[16];
	struct fi_cq_err_entry error;

	do {
		n0 = 0;
		n = fi_cq_read(nics[nic].cq, wc, 16);
		if (n == -FI_EAGAIN) {
			for (i = 0; i < options.num_mappings; i++) {
				if (i == nic)
					continue;

				(void) fi_cq_read(nics[i].cq, NULL, 0);
			}
			continue;
		} else if (n < 0) {
			fi_cq_readerr(nics[nic].cq, &error, 0);
			fprintf(stderr,
				"Completion with error: %s (err %d "
				"prov_errno %d).\n",
				fi_strerror(error.err),
				error.err, error.prov_errno);
			return -1;
		} else {
			for (k = 0; k < n; k++)
				put_context(context_pool, wc[k].op_context);
			*pending -= n;
			*completed += n;
			n0 += n;
		}
	} while (n0 > 0);

	return 0;
}

int post_rdma(int nic, int rnic, int test_type, size_t size, int signaled,
	      int peer_rank)
{
	struct iovec iov;
	void *desc = nics[nic].bufs.mr[options.rank] ?
		     fi_mr_desc(nics[nic].bufs.mr[options.rank]) : NULL;
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	int err;

	iov.iov_base = (char *)nics[nic].bufs.xe_buf[options.rank].buf;
	iov.iov_len = size;
	rma_iov.addr = peers[peer_rank].nic[rnic].bufs[options.rank].addr;
	rma_iov.len = size;
	rma_iov.key = peers[peer_rank].nic[rnic].bufs[options.rank].rkey;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = get_context(context_pool);
	msg.data = 0;

try_again:
	if (test_type == READ)
		err = fi_readmsg(nics[nic].ep, &msg,
				 signaled ? FI_COMPLETION : 0);
	else
		err = fi_writemsg(nics[nic].ep, &msg,
				  signaled ? FI_COMPLETION : 0);

	if (err == -FI_EAGAIN) {
		fi_cq_read(nics[nic].cq, NULL, 0);
		goto try_again;
	}

	return err;
}

int post_proxy_write(int nic, int rnic, size_t size, int signaled,
		     int peer_rank)
{
	struct iovec iov;
	void *desc = fi_mr_desc(nics[nic].proxy_bufs.mr[peer_rank]);
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;
	size_t sent, block_size = options.proxy_block;
	int flags = 0;
	int ret;

	iov.iov_base = (char *)nics[nic].proxy_bufs.xe_buf[peer_rank].buf;
	iov.iov_len =  block_size;
	rma_iov.addr = peers[peer_rank].nic[rnic].bufs[options.rank].addr;
	rma_iov.len =  block_size;
	rma_iov.key = peers[peer_rank].nic[rnic].bufs[options.rank].rkey;
	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = get_context(context_pool);
	msg.data = 0;

	for (sent = 0; sent < size;) {
		if (block_size >= size - sent) {
			block_size = size - sent;
			iov.iov_len = block_size;
			rma_iov.len = block_size;
			flags = signaled ? FI_COMPLETION : 0;
		}

		xe_copy_buf(
		      (char *)nics[nic].proxy_bufs.xe_buf[peer_rank].buf + sent,
		      (char *)nics[nic].bufs.xe_buf[options.rank].buf + sent,
		      block_size, &nics[nic].mapping.gpu);

try_again:
		ret = fi_writemsg(nics[nic].ep, &msg, flags);
		if (ret == -FI_EAGAIN) {
			fi_cq_read(nics[nic].cq, NULL, 0);
			goto try_again;
		} else if (ret) {
			break;
		}

		sent += block_size;
		iov.iov_base = (char *)iov.iov_base + block_size;
		rma_iov.addr += block_size;
	}

	return ret;
}

int post_proxy_send(int nic, int rnic, size_t size, int signaled, int peer_rank)
{
	struct iovec iov;
	void *desc = nics[nic].proxy_bufs.mr[peer_rank] ?
		     fi_mr_desc(nics[nic].proxy_bufs.mr[peer_rank]) : NULL;
	struct fi_msg msg;
	size_t sent, block_size = options.proxy_block;
	int flags = 0;
	int ret;

	iov.iov_base = (char *)nics[nic].proxy_bufs.xe_buf[peer_rank].buf;
	iov.iov_len = options.proxy_block;
	msg.msg_iov = &iov;
	msg.desc = nics[nic].proxy_bufs.mr[peer_rank] ? &desc : NULL;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.context = get_context(context_pool);
	msg.data = 0;

	for (sent = 0; sent < size;) {
		if (block_size >= size - sent) {
			block_size = size - sent;
			iov.iov_len = block_size;
			flags = signaled ? FI_COMPLETION : 0;
		}

		xe_copy_buf(
		      (char *)nics[nic].proxy_bufs.xe_buf[peer_rank].buf + sent,
		      (char *)nics[nic].bufs.xe_buf[options.rank].buf + sent,
		      block_size, &nics[nic].mapping.gpu);

try_again:
		ret = fi_sendmsg(nics[nic].ep, &msg, flags);
		if (ret == -FI_EAGAIN) {
			fi_cq_read(nics[nic].cq, NULL, 0);
			goto try_again;
		} else if (ret) {
			break;
		}

		sent += block_size;
		iov.iov_base = (char *)iov.iov_base + block_size;
	}

	return ret;
}

int post_send(int nic, int rnic, size_t size, int signaled, int peer_rank)
{
	struct iovec iov;
	void *desc = nics[nic].bufs.mr[options.rank] ?
		     fi_mr_desc(nics[nic].bufs.mr[options.rank]) : NULL;
	struct fi_msg msg;
	int ret;

	iov.iov_base = (char *)nics[nic].bufs.xe_buf[options.rank].buf;
	iov.iov_len = size;
	msg.msg_iov = &iov;
	msg.desc = nics[nic].bufs.mr[options.rank] ? &desc : NULL;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.context = get_context(context_pool);
	msg.data = 0;

try_again:
	ret = fi_sendmsg(nics[nic].ep, &msg, signaled ? FI_COMPLETION : 0);
	if (ret == -FI_EAGAIN) {
		fi_cq_read(nics[nic].cq, NULL, 0);
		goto try_again;
	}
	return ret;
}

int post_recv(int nic, int rnic, size_t size, int peer_rank)
{
	struct iovec iov;
	void *desc = nics[nic].bufs.mr[peer_rank] ?
		     fi_mr_desc(nics[nic].bufs.mr[peer_rank]) : NULL;
	struct fi_msg msg;
	int ret;

	iov.iov_base = (char *)nics[nic].bufs.xe_buf[peer_rank].buf;
	iov.iov_len = size;
	msg.msg_iov = &iov;
	msg.desc = nics[nic].bufs.mr[peer_rank] ? &desc : NULL;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.context = get_context(context_pool);
	msg.data = 0;

try_again:
	ret = fi_recvmsg(nics[nic].ep, &msg, FI_COMPLETION);
	if (ret == -FI_EAGAIN) {
		fi_cq_read(nics[nic].cq, NULL, 0);
		goto try_again;
	}
	return ret;
}

int post_sync_send(int nic, int rnic, size_t size, int peer_rank)
{
	struct iovec iov;
	void *desc = nics[nic].sync_buf.mr[0] ?
		     fi_mr_desc(nics[nic].sync_buf.mr[0]) : NULL;
	struct fi_msg_tagged msg;
	int ret;

	iov.iov_base = (char *)nics[nic].sync_buf.xe_buf[0].buf +
		       (size * options.rank);
	iov.iov_len = size;
	msg.msg_iov = &iov;
	msg.desc = nics[nic].sync_buf.mr[0] ? &desc : NULL;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.context = get_context(context_pool);
	msg.data = 0;
	msg.tag = 0xdeadbeef;

try_again:
	ret = fi_tsendmsg(nics[nic].ep, &msg,
			  FI_COMPLETION | FI_DELIVERY_COMPLETE);
	if (ret == -FI_EAGAIN) {
		fi_cq_read(nics[nic].cq, NULL, 0);
		goto try_again;
	}
	return ret;
}

int post_sync_recv(int nic, int rnic, size_t size, int peer_rank)
{
	struct iovec iov;
	void *desc = nics[nic].sync_buf.mr[0] ?
		     fi_mr_desc(nics[nic].sync_buf.mr[0]) : NULL;
	struct fi_msg_tagged msg;
	int ret;

	iov.iov_base = (char *)nics[nic].sync_buf.xe_buf[0].buf +
		       (size * peer_rank);
	iov.iov_len = size;
	msg.msg_iov = &iov;
	msg.desc = nics[nic].sync_buf.mr[0] ? &desc : NULL;
	msg.iov_count = 1;
	msg.addr = nics[nic].peer_addr[peer_rank][rnic];
	msg.context = get_context(context_pool);
	msg.data = 0;
	msg.tag = 0xdeadbeef;

try_again:
	ret = fi_trecvmsg(nics[nic].ep, &msg, FI_COMPLETION);
	if (ret == -FI_EAGAIN) {
		fi_cq_read(nics[nic].cq, NULL, 0);
		goto try_again;
	}
	return ret;
}

void sync_ofi(size_t size, int ranks, int my_rank)
{
	int j, num_completions, sync_nic;

	sync_nic = num_completions = 0;
	for (j = 0; j < ranks; j++) {
		if (j == my_rank)
			continue;

		EXIT_ON_ERROR(post_sync_recv(sync_nic, sync_nic, size, j));
		num_completions++;
	}
	for (j = 0; j < ranks; j++) {
		if (j == my_rank)
			continue;

		EXIT_ON_ERROR(post_sync_send(sync_nic, sync_nic, size, j));
		num_completions++;
	}

	wait_completion(sync_nic, num_completions);
	return;
}

static int transmit(int nic, int rnic, int peer_rank, size_t size, int signaled)
{
	switch (options.test_type) {
	case SEND:
		if (options.buf_location == DEVICE && options.use_proxy)
			CHECK_ERROR(post_proxy_send(nic, rnic, size, signaled,
						    peer_rank));
		else
			CHECK_ERROR(post_send(nic, rnic, size, signaled,
					      peer_rank));
		break;
	case RECV:
		CHECK_ERROR(post_recv(nic, rnic, size, peer_rank));
		break;
	case READ:
		CHECK_ERROR(post_rdma(nic, rnic, READ, size, signaled,
				      peer_rank));
		break;
	case WRITE:
		if (options.buf_location == DEVICE && options.use_proxy)
			CHECK_ERROR(post_proxy_write(nic, rnic, size, signaled,
						     peer_rank));
		else
			CHECK_ERROR(post_rdma(nic, rnic, WRITE, size, signaled,
					      peer_rank));
		break;
	default:
		break;
	}

	return 0;
err_out:
	printf("Error in transmit to rank %d\n", peer_rank);
	return -1;
}

static int calculate_warmup_iters(void)
{
	int i, warmup_iters, steps;

	for (i = warmup_iters = 0; i < options.max_ranks; i++) {
		if (i == options.rank && warmup_iters < me.num_nics) {
			warmup_iters = me.num_nics;
			continue;
		}

		if (warmup_iters < peers[i].num_nics)
			warmup_iters = peers[i].num_nics;
	}

	steps = options.iters / 100;
	warmup_iters = steps ? warmup_iters * steps : warmup_iters;

	return warmup_iters;
}

int all_to_all(size_t size)
{
	int i, j, k, completed, pending, max_pending, pending_per_iter;
	int expected_completions, warmup_iters;
	double start, stop;
	int mapping = 0;
	int rmapping = 0;

	i = pending = 0;
	warmup_iters = calculate_warmup_iters();
	expected_completions = (warmup_iters * (options.max_ranks - 1)) +
			       (options.iters * (options.max_ranks - 1));
	max_pending = TX_DEPTH;
	pending_per_iter = options.max_ranks - 1;
	if (options.test_type == SEND) {
		pending_per_iter *= 2;
		expected_completions *= 2;
		max_pending += TX_DEPTH >= RX_DEPTH ? RX_DEPTH : TX_DEPTH;
	}

	while (i < options.iters + warmup_iters ||
	       completed < expected_completions) {
		while (i < options.iters + warmup_iters &&
		       pending + pending_per_iter < max_pending) {
			mapping = i % options.num_mappings;
			if (options.test_type == SEND) {
				for (k = 0; k < options.max_ranks; k++) {
					if (k == options.rank)
						continue;

					rmapping = i % peers[k].num_nics;
					CHECK_ERROR(post_recv(mapping, rmapping,
							      size, k));
					pending++;
				}
			}

			for (j = 0; j < options.max_ranks; j++) {
				if (j == options.rank)
					continue;

				rmapping = i % peers[j].num_nics;
				CHECK_ERROR(transmit(mapping, rmapping, j, size,
						     1));
				pending++;
			}
			if (++i == warmup_iters)
				start = when();
		}
		for (j = 0; j < options.num_mappings; j++)
			EXIT_ON_ERROR(process_completions(j, &pending,
							  &completed));
	}

	sync_ofi(4, options.max_ranks, options.rank);
	stop = when();

	if (options.verify) {
		for (i = 0; i < options.max_ranks; i++) {
			if (i == options.rank)
				continue;

			check_buf(0, size, 'A', i);
		}
	}

	if (options.test_type == RECV)
		return 0;

	printf("%10zd (x %4d) %10.2lf us/xfer %12.2lf MB/s\n", size,
		options.iters,
		(stop - start) / (options.iters * options.max_ranks),
		((long)size * options.iters * options.max_ranks) /
		(stop - start));

	return 0;

err_out:
	return -1;
}

int point_to_point(size_t size)
{
	int i, j, completed, pending, max_pending, pending_per_iter;
	int expected_completions, warmup_iters, peer_rank;
	double start, stop;
	int mapping = 0;
	int rmapping = 0;

	i = pending = 0;
	if (options.client) {
		peer_rank = 0;
		max_pending = RX_DEPTH;
	} else {
		peer_rank = 1;
		max_pending = TX_DEPTH;
	}
	warmup_iters = calculate_warmup_iters();
	expected_completions = warmup_iters + options.iters;
	pending_per_iter = 1;

	while (i < options.iters + warmup_iters ||
	       completed < expected_completions) {
		while (i < options.iters + warmup_iters &&
		       pending + pending_per_iter < max_pending) {
			mapping = i % options.num_mappings;
			rmapping = i % peers[peer_rank].num_nics;
			if (options.test_type == RECV) {
				CHECK_ERROR(post_recv(mapping, rmapping, size,
						      peer_rank));
				pending++;
			} else {
				CHECK_ERROR(transmit(mapping, rmapping,
						     peer_rank, size, 1));
				pending++;
			}
			if (++i == warmup_iters)
				start = when();
		}
		for (j = 0; j < options.num_mappings; j++)
			EXIT_ON_ERROR(process_completions(j, &pending,
							  &completed));
	}

	sync_ofi(4, options.max_ranks, options.rank);
	stop = when();

	if (options.verify) {
		for (i = 0; i < options.max_ranks; i++) {
			if (i == options.rank)
				continue;

			check_buf(0, size, 'A', i);
		}
	}

	if (!options.client)
		return 0;

	printf("%10zd (x %4d) %10.2lf us/xfer %12.2lf MB/s\n", size,
		options.iters,
		(stop - start) / options.iters,
		((long)size * options.iters) / (stop - start));

	return 0;

err_out:
	return -1;
}

struct algo_funcs algorithms[MAX_ALGORITHM] = {
	{
		.algo = ALL_TO_ALL,
		.name = "all-to-all",
		.run = all_to_all,
	},
	{
		.algo = POINT_TO_POINT,
		.name = "point-to-point",
		.run = point_to_point,
	},
};

int run_test(void)
{
	int i;
	size_t size;

	for (i = 0; i < MAX_ALGORITHM; i++) {
		if (options.algo == MAX_ALGORITHM ||
		    options.algo == algorithms[i].algo) {
			if (!i && options.verbose)
				printf("Running %s algorithm\n",
					algorithms[i].name);
			for (size = 1; size <= options.max_size; size <<= 1) {
				if (options.msg_size)
					size = options.msg_size;

				sync_ofi(4, options.max_ranks, options.rank);
				CHECK_ERROR(algorithms[i].run(size));
				if (options.msg_size)
					break;
			}
		}
	}

	sync_ofi(4, options.max_ranks, options.rank);
	return 0;
err_out:
	return -1;
}
