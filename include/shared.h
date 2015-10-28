/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _SHARED_H_
#define _SHARED_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/socket.h>
#include <sys/types.h>
#include <inttypes.h>

#include <rdma/fabric.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* all tests should work with 1.0 API or newer */
#ifndef FT_FIVERSION
#define FT_FIVERSION FI_VERSION(1,1)
#endif

#ifdef __APPLE__
#include "osx/osd.h"
#endif

struct test_size_param {
	int size;
	int option;
};

enum precision {
	NANO = 1,
	MICRO = 1000,
	MILLI = 1000000,
};

enum {
	FT_OPT_ACTIVE		= 1 << 0,
	FT_OPT_ITER		= 1 << 1,
	FT_OPT_SIZE		= 1 << 2,
	FT_OPT_RX_CQ		= 1 << 3,
	FT_OPT_TX_CQ		= 1 << 4,
	FT_OPT_RX_CNTR		= 1 << 5,
	FT_OPT_TX_CNTR		= 1 << 6,
	FT_OPT_VERIFY_DATA	= 1 << 7,
};

struct ft_opts {
	int iterations;
	int transfer_size;
	char *src_port;
	char *dst_port;
	char *src_addr;
	char *dst_addr;
	int size_option;
	int options;
	int machr;
	int argc;
	char **argv;
};

extern struct fi_info *fi, *hints;
extern struct fid_fabric *fabric;
extern struct fid_wait *waitset;
extern struct fid_domain *domain;
extern struct fid_poll *pollset;
extern struct fid_pep *pep;
extern struct fid_ep *ep;
extern struct fid_cq *txcq, *rxcq;
extern struct fid_cntr *txcntr, *rxcntr;
extern struct fid_mr *mr;
extern struct fid_av *av;
extern struct fid_eq *eq;

extern fi_addr_t remote_fi_addr;
extern void *buf, *tx_buf, *rx_buf;
extern size_t buf_size, tx_size, rx_size;
extern int timeout;

extern struct fi_context tx_ctx, rx_ctx;

extern uint64_t tx_seq, rx_seq, tx_cq_cntr, rx_cq_cntr;
extern struct fi_av_attr av_attr;
extern struct fi_eq_attr eq_attr;
extern struct fi_cq_attr cq_attr;
extern struct fi_cntr_attr cntr_attr;

extern char test_name[10];
extern struct timespec start, end;
extern struct ft_opts opts;


void ft_parseinfo(int op, char *optarg, struct fi_info *hints);
void ft_parse_addr_opts(int op, char *optarg, struct ft_opts *opts);
void ft_parsecsopts(int op, char *optarg, struct ft_opts *opts);
void ft_usage(char *name, char *desc);
void ft_csusage(char *name, char *desc);
void ft_fill_buf(void *buf, int size);
int ft_check_buf(void *buf, int size);
#define ADDR_OPTS "b:p:s:"
#define INFO_OPTS "n:f:"
#define CS_OPTS ADDR_OPTS "I:S:m"

extern char default_port[8];

#define INIT_OPTS (struct ft_opts) \
	{	.options = FT_OPT_RX_CQ | FT_OPT_TX_CQ, \
		.iterations = 1000, \
		.transfer_size = 1024, \
		.argc = argc, .argv = argv \
	}

extern struct test_size_param test_size[];
const unsigned int test_cnt;
#define TEST_CNT test_cnt
#define FT_STR_LEN 32
#define FT_MAX_CTRL_MSG 64
#define FT_MR_KEY 0xC0DE

int ft_getsrcaddr(char *node, char *service, struct fi_info *hints);
int ft_read_addr_opts(char **node, char **service, struct fi_info *hints,
		uint64_t *flags, struct ft_opts *opts);
char *size_str(char str[FT_STR_LEN], long long size);
char *cnt_str(char str[FT_STR_LEN], long long cnt);
int size_to_count(int size);


#define FT_CLOSE_FID(fd)			\
	do {					\
		if ((fd)) {			\
			fi_close(&(fd)->fid);	\
			fd = NULL;		\
		}				\
	} while (0)

#define FT_CLOSEV_FID(fd, cnt)			\
	do {					\
		int i;				\
		if (!(fd))			\
			break;			\
		for (i = 0; i < (cnt); i++) {	\
			FT_CLOSE_FID((fd)[i]);	\
		}				\
	} while (0)

int ft_alloc_bufs();
int ft_open_fabric_res();
int ft_start_server();
int ft_alloc_active_res(struct fi_info *fi);
int ft_init_ep();
int ft_av_insert(struct fid_av *av, void *addr, size_t count, fi_addr_t *fi_addr,
		uint64_t flags, void *context);
int ft_init_av();
int ft_exchange_keys(struct fi_rma_iov *peer_iov);
void ft_free_res();
void init_test(struct ft_opts *opts, char *test_name, size_t test_name_len);

static inline void ft_start(void)
{
	opts.options |= FT_OPT_ACTIVE;
	clock_gettime(CLOCK_MONOTONIC, &start);
}
static inline void ft_stop(void)
{
	clock_gettime(CLOCK_MONOTONIC, &end);
	opts.options &= ~FT_OPT_ACTIVE;
}
int ft_sync();
int ft_finalize();

size_t ft_rx_prefix_size();
size_t ft_tx_prefix_size();
ssize_t ft_post_rx(size_t size);
ssize_t ft_post_tx(size_t size);
ssize_t ft_rx(size_t size);
ssize_t ft_tx(size_t size);

int ft_cq_readerr(struct fid_cq *cq);
int ft_get_rx_comp(uint64_t total);
int ft_get_tx_comp(uint64_t total);

void eq_readerr(struct fid_eq *eq, const char *eq_str);


int64_t get_elapsed(const struct timespec *b, const struct timespec *a,
		enum precision p);
void show_perf(char *name, int tsize, int iters, struct timespec *start,
		struct timespec *end, int xfers_per_iter);
void show_perf_mr(int tsize, int iters, struct timespec *start,
		struct timespec *end, int xfers_per_iter, int argc, char *argv[]);

#define FT_PRINTERR(call, retv) \
	do { fprintf(stderr, call "(): %s:%d, ret=%d (%s)\n", __FILE__, __LINE__, (int) retv, fi_strerror((int) -retv)); } while (0)

#define FT_ERR(fmt, ...) \
	do { fprintf(stderr, "%s:%d: " fmt, __FILE__, __LINE__, ##__VA_ARGS__); } while (0)

#define FT_PROCESS_QUEUE_ERR(readerr, rd, queue, fn, str)	\
	do {							\
		if (rd == -FI_EAVAIL) {				\
			readerr(queue, fn " " str);		\
		} else {					\
			FT_PRINTERR(fn, rd);			\
		}						\
	} while (0)

#define FT_PROCESS_EQ_ERR(rd, eq, fn, str) \
	FT_PROCESS_QUEUE_ERR(eq_readerr, rd, eq, fn, str)

#define FT_PRINT_OPTS_USAGE(opt, desc) fprintf(stderr, " %-20s %s\n", opt, desc)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define ARRAY_SIZE(A) (sizeof(A)/sizeof(*A))

#define TEST_ENUM_SET_N_RETURN(str, enum_val, type, data)	\
	TEST_SET_N_RETURN(str, #enum_val, enum_val, type, data)

#define TEST_SET_N_RETURN(str, val_str, val, type, data)	\
	do {							\
		if (!strncmp(str, val_str, strlen(val_str))) {	\
			*(type *)(data) = val;			\
			return 0;				\
		}						\
	} while (0)

/* for RMA tests --- we want to be able to select fi_writedata, but there is no
 * constant in libfabric for this */
enum ft_rma_opcodes {
	FT_RMA_READ = 1,
	FT_RMA_WRITE,
	FT_RMA_WRITEDATA,
};

#ifdef __cplusplus
}
#endif

#endif /* _SHARED_H_ */
