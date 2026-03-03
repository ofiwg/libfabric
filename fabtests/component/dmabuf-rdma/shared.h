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

#ifndef _DMABUF_RDMA_TESTS_SHARED_H_
#define _DMABUF_RDMA_TESTS_SHARED_H_

#include <unistd.h>
#include <stdint.h>
#include "xe.h"
#include "util.h"
#include "ofi_ctx_pool.h"

#define OPTS "A:bB:d:e:M:m:n:p:PRr:S:t:Vvh?"

#define MAX_NICS	8
#define MAX_SIZE	(4*1024*1024)
#define MIN_PROXY_BLOCK	(131072)
#define MAX_RAW_KEY_SIZE (256)

enum test_type {
	READ,
	WRITE,
	SEND,
	RECV, /* internal use only */
};

enum algorithm {
	ALL_TO_ALL,
	POINT_TO_POINT,
	MAX_ALGORITHM
};

struct algo_funcs {
	enum algorithm	algo;
	const char	*name;
	int		(*run)(size_t size);
};

struct raw_key {
	uint64_t	size;
	uint8_t		key[MAX_RAW_KEY_SIZE];
};

struct business_card {
	struct {
		int dev_num;
		struct {
			uint64_t	addr;
			uint64_t	rkey;
			struct raw_key 	raw_key;
		} bufs[MAX_CLIENTS];
		union {
			struct {
				uint64_t	one;
				uint64_t	two;
				uint64_t	three;
				uint64_t	four;
			};
			uint8_t bytes[1024];
		} ep_name;
	} nic[MAX_NICS];
	struct {
		uint64_t	addr;
		uint64_t	rkey;
		struct raw_key 	raw_key;
	} sync_buf;
	int dev_num;
	int num_nics;
	int use_raw_key;
	int rank;
	enum algorithm algo;
};

struct domain_gpu_mapping {
	char *domain_name;
	struct gpu gpu;
};

struct buf {
	struct xe_buf		xe_buf[MAX_CLIENTS];
	struct fid_mr		*mr[MAX_CLIENTS];
};

struct nic {
	struct fi_info		*fi, *fi_pep;
	struct fid_fabric	*fabric;
	struct fid_eq		*eq;
	struct fid_domain	*domain;
	struct fid_pep		*pep;
	struct fid_ep		*ep;
	struct fid_av		*av;
	struct fid_cq		*cq;
	fi_addr_t		peer_addr[MAX_CLIENTS][MAX_NICS];
	struct domain_gpu_mapping mapping;
	struct buf		bufs;
	struct buf		proxy_bufs;
	struct buf		sync_buf;
};

struct options_t {
	bool	bidirectional;
	int	proxy_block;
	char 	*mapping_str;
	int	num_mappings;
	int	ep_type;
	ssize_t	max_size;
	int	loc1;
	int	loc2;
	int	iters;
	char 	*prov_name;
	bool	use_proxy;
	int 	max_ranks;
	int	rank;
	unsigned int port;
	size_t	msg_size;
	int	test_type;
	bool	verbose;
	int	verify;
	char 	*server_name;
	bool	client;
	int	sockfd;
	bool	use_sync_ofi;
	bool	prepost;
	int	buf_location;
	int	use_raw_key;
	enum algorithm	algo;
};

extern struct nic		nics[MAX_NICS];
extern struct business_card 	me;
extern struct business_card	*peers;
extern struct context_pool	*context_pool;
extern struct options_t		options;
extern int			TX_DEPTH;
extern int			RX_DEPTH;

void remove_characters(char *str, char *removes);
void buf_location_str(int loc, char *str, int len);
void parse_buf_location(char *string, int *loc1, int *loc2, int default_loc);
size_t parse_size(char *string);
void print_nic_info(void);
int wait_conn_req(struct fid_eq *eq, struct fi_info **fi);
int wait_connected(struct fid_ep *ep, struct fid_eq *eq);
int init_nic(int nic);
void show_business_card(struct business_card *bc, char *name);
int fill_in_my_business_card(void);
int exchange_business_cards(void);
int init_ofi(void);
void finalize_ofi(void);
void init_buf(int nic, size_t buf_size, char c);
void check_buf(int nic, size_t size, char c, int rank);
void free_buf(void);
int exchange_business_cards(void);
int init_ofi(void);
int process_completions(int nic, int *pending, int *completed);
int post_rdma(int nic, int rnic, int test_type, size_t size, int signaled,
	      int peer_rank);
int post_proxy_write(int nic, int rnic, size_t size, int signaled,
		     int peer_rank);
int post_proxy_send(int nic, int rnic, size_t size, int signaled,
		    int peer_rank);
int post_send(int nic, int rnic, size_t size, int signaled, int rank);
int post_recv(int nic, int rnic, size_t size, int rank);
int post_sync_send(int nic, int rnic, size_t size, int rank);
int post_sync_recv(int nic, int rnic, size_t size, int rank);
void sync_ofi(size_t size, int ranks, int my_rank);
int all_to_all(size_t size);
int point_to_point(size_t size);
int run_test(void);

static inline void wait_completion(int nic, int n)
{
	struct fi_cq_entry wc;
	struct fi_cq_err_entry error;
	int ret, i, completed = 0;

	while (completed < n) {
		ret = fi_cq_read(nics[nic].cq, &wc, 1);
		if (ret == -FI_EAGAIN) {
			for (i = 0; i < options.num_mappings; i++) {
				if (i == nic)
					continue;

				(void) fi_cq_read(nics[i].cq, NULL, 0);
			}
			continue;
		}
		if (ret < 0) {
			fi_cq_readerr(nics[nic].cq, &error, 0);
			fprintf(stderr,
				"Completion with error: %s (err %d "
				"prov_errno %d).\n",
				fi_strerror(error.err), error.err,
				error.prov_errno);
			return;
		}
		put_context(context_pool, wc.op_context);
		completed++;
	}
}

static inline int string_to_location(char *s, int default_loc)
{
	int loc;

	if (strcasecmp(s, "malloc") == 0)
		loc = MALLOC;
	else if (strcasecmp(s, "host") == 0)
		loc = HOST;
	else if (strcasecmp(s, "device") == 0)
		loc = DEVICE;
	else if (strcasecmp(s, "shared") == 0)
		loc = SHARED;
	else
		loc = default_loc;

	return loc;
}

#endif /* _DMABUF_RDMA_TESTS_SHARED_H_ */
