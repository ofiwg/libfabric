/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2016 Cray Inc.  All rights reserved.
 * Copyright (c) 2014-2016, Cisco Systems, Inc. All rights reserved.
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

#include <assert.h>
#include <netdb.h>
#include <poll.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

#include <shared.h>

struct fi_info *fi_pep, *fi, *hints;
struct fid_fabric *fabric;
struct fid_wait *waitset;
struct fid_domain *domain;
struct fid_poll *pollset;
struct fid_pep *pep;
struct fid_ep *ep;
struct fid_cq *txcq, *rxcq;
struct fid_cntr *txcntr, *rxcntr;
struct fid_mr *mr;
struct fid_av *av;
struct fid_eq *eq;

struct fid_mr no_mr;
struct fi_context tx_ctx, rx_ctx;

uint64_t tx_seq, rx_seq, tx_cq_cntr, rx_cq_cntr;
int ft_skip_mr = 0;
int ft_parent_proc = 0;
pid_t ft_child_pid = 0;
int ft_socket_pair[2];

fi_addr_t remote_fi_addr = FI_ADDR_UNSPEC;
void *buf, *tx_buf, *rx_buf;
size_t buf_size, tx_size, rx_size;
int rx_fd = -1, tx_fd = -1;
char default_port[8] = "9228";

char test_name[50] = "custom";
int timeout = -1;
struct timespec start, end;

struct fi_av_attr av_attr = {
	.type = FI_AV_MAP,
	.count = 1
};
struct fi_eq_attr eq_attr = {
	.wait_obj = FI_WAIT_UNSPEC
};
struct fi_cq_attr cq_attr = {
	.wait_obj = FI_WAIT_NONE
};
struct fi_cntr_attr cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.wait_obj = FI_WAIT_NONE
};

struct ft_opts opts;

struct test_size_param test_size[] = {
	{ 1 <<  1, 0 }, { (1 <<  1) + (1 <<  0), 0 },
	{ 1 <<  2, 0 }, { (1 <<  2) + (1 <<  1), 0 },
	{ 1 <<  3, 0 }, { (1 <<  3) + (1 <<  2), 0 },
	{ 1 <<  4, 0 }, { (1 <<  4) + (1 <<  3), 0 },
	{ 1 <<  5, 0 }, { (1 <<  5) + (1 <<  4), 0 },
	{ 1 <<  6, FT_DEFAULT_SIZE }, { (1 <<  6) + (1 <<  5), 0 },
	{ 1 <<  7, 0 }, { (1 <<  7) + (1 <<  6), 0 },
	{ 1 <<  8, FT_DEFAULT_SIZE }, { (1 <<  8) + (1 <<  7), 0 },
	{ 1 <<  9, 0 }, { (1 <<  9) + (1 <<  8), 0 },
	{ 1 << 10, FT_DEFAULT_SIZE }, { (1 << 10) + (1 <<  9), 0 },
	{ 1 << 11, 0 }, { (1 << 11) + (1 << 10), 0 },
	{ 1 << 12, FT_DEFAULT_SIZE }, { (1 << 12) + (1 << 11), 0 },
	{ 1 << 13, 0 }, { (1 << 13) + (1 << 12), 0 },
	{ 1 << 14, 0 }, { (1 << 14) + (1 << 13), 0 },
	{ 1 << 15, 0 }, { (1 << 15) + (1 << 14), 0 },
	{ 1 << 16, FT_DEFAULT_SIZE }, { (1 << 16) + (1 << 15), 0 },
	{ 1 << 17, 0 }, { (1 << 17) + (1 << 16), 0 },
	{ 1 << 18, 0 }, { (1 << 18) + (1 << 17), 0 },
	{ 1 << 19, 0 }, { (1 << 19) + (1 << 18), 0 },
	{ 1 << 20, FT_DEFAULT_SIZE }, { (1 << 20) + (1 << 19), 0 },
	{ 1 << 21, 0 }, { (1 << 21) + (1 << 20), 0 },
	{ 1 << 22, 0 }, { (1 << 22) + (1 << 21), 0 },
	{ 1 << 23, 0 },
};

const unsigned int test_cnt = (sizeof test_size / sizeof test_size[0]);

#define INTEG_SEED 7
static const char integ_alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const int integ_alphabet_length = (sizeof(integ_alphabet)/sizeof(*integ_alphabet)) - 1;


static int ft_poll_fd(int fd, int timeout)
{
	struct pollfd fds;
	int ret;

	fds.fd = fd;
	fds.events = POLLIN;
	ret = poll(&fds, 1, timeout);
	if (ret == -1) {
		FT_PRINTERR("poll", -errno);
		ret = -errno;
	} else if (!ret) {
		ret = -FI_EAGAIN;
	} else {
		ret = 0;
	}
	return ret;
}

size_t ft_tx_prefix_size()
{
	return (fi->tx_attr->mode & FI_MSG_PREFIX) ?
		fi->ep_attr->msg_prefix_size : 0;
}

size_t ft_rx_prefix_size()
{
	return (fi->rx_attr->mode & FI_MSG_PREFIX) ?
		fi->ep_attr->msg_prefix_size : 0;
}

static int ft_check_opts(uint64_t flags)
{
	return (opts.options & flags) == flags;
}

static void ft_cq_set_wait_attr(void)
{
	switch (opts.comp_method) {
	case FT_COMP_SREAD:
		cq_attr.wait_obj = FI_WAIT_UNSPEC;
		cq_attr.wait_cond = FI_CQ_COND_NONE;
		break;
	case FT_COMP_WAITSET:
		assert(waitset);
		cq_attr.wait_obj = FI_WAIT_SET;
		cq_attr.wait_cond = FI_CQ_COND_NONE;
		cq_attr.wait_set = waitset;
		break;
	case FT_COMP_WAIT_FD:
		cq_attr.wait_obj = FI_WAIT_FD;
		cq_attr.wait_cond = FI_CQ_COND_NONE;
		break;
	default:
		cq_attr.wait_obj = FI_WAIT_NONE;
		break;
	}
}

static void ft_cntr_set_wait_attr(void)
{
	switch (opts.comp_method) {
	case FT_COMP_SREAD:
		cntr_attr.wait_obj = FI_WAIT_UNSPEC;
		break;
	case FT_COMP_WAITSET:
		assert(waitset);
		cntr_attr.wait_obj = FI_WAIT_SET;
		break;
	case FT_COMP_WAIT_FD:
		cntr_attr.wait_obj = FI_WAIT_FD;
		break;
	default:
		cntr_attr.wait_obj = FI_WAIT_NONE;
		break;
	}
}

static uint64_t ft_caps_to_mr_access(uint64_t caps)
{
	uint64_t mr_access = 0;

	if (caps & (FI_MSG | FI_TAGGED)) {
		if (caps & FT_MSG_MR_ACCESS)
			mr_access |= caps & FT_MSG_MR_ACCESS;
		else
			mr_access |= FT_MSG_MR_ACCESS;
	}

	if (caps & (FI_RMA | FI_ATOMIC)) {
		if (caps & FT_RMA_MR_ACCESS)
			mr_access |= caps & FT_RMA_MR_ACCESS;
		else
			mr_access |= FT_RMA_MR_ACCESS;
	}

	return mr_access;
}

/*
 * Include FI_MSG_PREFIX space in the allocated buffer, and ensure that the
 * buffer is large enough for a control message used to exchange addressing
 * data.
 */
int ft_alloc_msgs(void)
{
	int ret;
	long alignment = 1;

	/* TODO: support multi-recv tests */
	if (fi->rx_attr->op_flags == FI_MULTI_RECV)
		return 0;

	tx_size = opts.options & FT_OPT_SIZE ?
		  opts.transfer_size : test_size[TEST_CNT - 1].size;
	if (tx_size > fi->ep_attr->max_msg_size)
		tx_size = fi->ep_attr->max_msg_size;
	rx_size = tx_size + ft_rx_prefix_size();
	tx_size += ft_tx_prefix_size();
	buf_size = MAX(tx_size, FT_MAX_CTRL_MSG) + MAX(rx_size, FT_MAX_CTRL_MSG);

	if (opts.options & FT_OPT_ALIGN) {
		alignment = sysconf(_SC_PAGESIZE);
		if (alignment < 0)
			return -errno;
		buf_size += alignment;

		ret = posix_memalign(&buf, (size_t) alignment, buf_size);
		if (ret) {
			FT_PRINTERR("posix_memalign", ret);
			return ret;
		}
	} else {
		buf = malloc(buf_size);
		if (!buf) {
			perror("malloc");
			return -FI_ENOMEM;
		}
	}
	memset(buf, 0, buf_size);
	rx_buf = buf;
	tx_buf = (char *) buf + MAX(rx_size, FT_MAX_CTRL_MSG);
	tx_buf = (void *) (((uintptr_t) tx_buf + alignment - 1) &
			   ~(alignment - 1));

	if (!ft_skip_mr && ((fi->mode & FI_LOCAL_MR) ||
				(fi->caps & (FI_RMA | FI_ATOMIC)))) {
		ret = fi_mr_reg(domain, buf, buf_size, ft_caps_to_mr_access(fi->caps),
				0, FT_MR_KEY, 0, &mr, NULL);
		if (ret) {
			FT_PRINTERR("fi_mr_reg", ret);
			return ret;
		}
	} else {
		mr = &no_mr;
	}

	return 0;
}

int ft_open_fabric_res(void)
{
	int ret;

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		return ret;
	}

	ret = fi_eq_open(fabric, &eq_attr, &eq, NULL);
	if (ret) {
		FT_PRINTERR("fi_eq_open", ret);
		return ret;
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		return ret;
	}

	return 0;
}

int ft_alloc_active_res(struct fi_info *fi)
{
	int ret;

	ret = ft_alloc_msgs();
	if (ret)
		return ret;

	if (cq_attr.format == FI_CQ_FORMAT_UNSPEC) {
		if (fi->caps & FI_TAGGED)
			cq_attr.format = FI_CQ_FORMAT_TAGGED;
		else
			cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	}

	if (opts.options & FT_OPT_TX_CQ) {
		ft_cq_set_wait_attr();
		cq_attr.size = fi->tx_attr->size;
		ret = fi_cq_open(domain, &cq_attr, &txcq, &txcq);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	if (opts.options & FT_OPT_TX_CNTR) {
		ft_cntr_set_wait_attr();
		ret = fi_cntr_open(domain, &cntr_attr, &txcntr, &txcntr);
		if (ret) {
			FT_PRINTERR("fi_cntr_open", ret);
			return ret;
		}
	}

	if (opts.options & FT_OPT_RX_CQ) {
		ft_cq_set_wait_attr();
		cq_attr.size = fi->rx_attr->size;
		ret = fi_cq_open(domain, &cq_attr, &rxcq, &rxcq);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	if (opts.options & FT_OPT_RX_CNTR) {
		ft_cntr_set_wait_attr();
		ret = fi_cntr_open(domain, &cntr_attr, &rxcntr, &rxcntr);
		if (ret) {
			FT_PRINTERR("fi_cntr_open", ret);
			return ret;
		}
	}

	if (fi->ep_attr->type == FI_EP_RDM || fi->ep_attr->type == FI_EP_DGRAM) {
		if (fi->domain_attr->av_type != FI_AV_UNSPEC)
			av_attr.type = fi->domain_attr->av_type;

		if (opts.av_name) {
			av_attr.name = opts.av_name;
		}
		ret = fi_av_open(domain, &av_attr, &av, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			return ret;
		}
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	return 0;
}

int ft_start_server(void)
{
	int ret;

	ret = fi_getinfo(FT_FIVERSION, opts.src_addr, opts.src_port, FI_SOURCE,
			 hints, &fi_pep);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = fi_fabric(fi_pep->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		return ret;
	}

	ret = fi_eq_open(fabric, &eq_attr, &eq, NULL);
	if (ret) {
		FT_PRINTERR("fi_eq_open", ret);
		return ret;
	}

	ret = fi_passive_ep(fabric, fi_pep, &pep, NULL);
	if (ret) {
		FT_PRINTERR("fi_passive_ep", ret);
		return ret;
	}

	ret = fi_pep_bind(pep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_pep_bind", ret);
		return ret;
	}

	ret = fi_listen(pep);
	if (ret) {
		FT_PRINTERR("fi_listen", ret);
		return ret;
	}

	return 0;
}

#define FT_EP_BIND(ep, fd, flags)					\
	do {								\
		int ret;						\
		if ((fd)) {						\
			ret = fi_ep_bind((ep), &(fd)->fid, (flags));	\
			if (ret) {					\
				FT_PRINTERR("fi_ep_bind", ret);		\
				return ret;				\
			}						\
		}							\
	} while (0)

int ft_get_cq_fd(struct fid_cq *cq, int *fd)
{
	int ret = FI_SUCCESS;

	if (cq && opts.comp_method == FT_COMP_WAIT_FD) {
		ret = fi_control(&cq->fid, FI_GETWAIT, fd);
		if (ret)
			FT_PRINTERR("fi_control(FI_GETWAIT)", ret);
	}

	return ret;
}

int ft_init_ep(void)
{
	int flags, ret;

	if (fi->ep_attr->type == FI_EP_MSG)
		FT_EP_BIND(ep, eq, 0);
	FT_EP_BIND(ep, av, 0);
	FT_EP_BIND(ep, txcq, FI_TRANSMIT);
	FT_EP_BIND(ep, rxcq, FI_RECV);

	ret = ft_get_cq_fd(txcq, &tx_fd);
	if (ret)
		return ret;

	ret = ft_get_cq_fd(rxcq, &rx_fd);
	if (ret)
		return ret;

	/* TODO: use control structure to select counter bindings explicitly */
	flags = !txcq ? FI_SEND : 0;
	if (hints->caps & (FI_WRITE | FI_READ))
		flags |= hints->caps & (FI_WRITE | FI_READ);
	else if (hints->caps & FI_RMA)
		flags |= FI_WRITE | FI_READ;
	FT_EP_BIND(ep, txcntr, flags);
	flags = !rxcq ? FI_RECV : 0;
	if (hints->caps & (FI_REMOTE_WRITE | FI_REMOTE_READ))
		flags |= hints->caps & (FI_REMOTE_WRITE | FI_REMOTE_READ);
	else if (hints->caps & FI_RMA)
		flags |= FI_REMOTE_WRITE | FI_REMOTE_READ;
	FT_EP_BIND(ep, rxcntr, flags);

	ret = fi_enable(ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		return ret;
	}

	if (fi->rx_attr->op_flags != FI_MULTI_RECV) {
		/* Initial receive will get remote address for unconnected EPs */
		ret = ft_post_rx(MAX(rx_size, FT_MAX_CTRL_MSG));
		if (ret)
			return ret;
	}

	return 0;
}

int ft_av_insert(struct fid_av *av, void *addr, size_t count, fi_addr_t *fi_addr,
		uint64_t flags, void *context)
{
	int ret;

	ret = fi_av_insert(av, addr, count, fi_addr, flags, context);
	if (ret < 0) {
		FT_PRINTERR("fi_av_insert", ret);
		return ret;
	} else if (ret != count) {
		FT_ERR("fi_av_insert: number of addresses inserted = %d;"
			       " number of addresses given = %zd\n", ret, count);
		return -EXIT_FAILURE;
	}

	return 0;
}

/* TODO: retry send for unreliable endpoints */
int ft_init_av(void)
{
	size_t addrlen;
	int ret;

	if (opts.dst_addr) {
		ret = ft_av_insert(av, fi->dest_addr, 1, &remote_fi_addr, 0, NULL);
		if (ret)
			return ret;

		addrlen = FT_MAX_CTRL_MSG;
		ret = fi_getname(&ep->fid, (char *) tx_buf + ft_tx_prefix_size(),
				 &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = (int) ft_tx(addrlen);
		if (ret)
			return ret;

		ret = ft_rx(1);
	} else {
		ret = (int) ft_rx(FT_MAX_CTRL_MSG);
		if (ret)
			return ret;

		ret = ft_av_insert(av, (char *) rx_buf + ft_rx_prefix_size(),
				   1, &remote_fi_addr, 0, NULL);
		if (ret)
			return ret;

		ret = (int) ft_tx(1);
	}

	return ret;
}

int ft_exchange_keys(struct fi_rma_iov *peer_iov)
{
	struct fi_rma_iov *rma_iov;
	int ret;

	if (opts.dst_addr) {
		rma_iov = tx_buf + ft_tx_prefix_size();
		rma_iov->addr = fi->domain_attr->mr_mode == FI_MR_SCALABLE ?
				0 : (uintptr_t) rx_buf + ft_rx_prefix_size();
		rma_iov->key = fi_mr_key(mr);
		ret = ft_tx(sizeof *rma_iov);
		if (ret)
			return ret;

		ret = ft_get_rx_comp(rx_seq);
		if (ret)
			return ret;

		rma_iov = rx_buf + ft_rx_prefix_size();
		*peer_iov = *rma_iov;
		ret = ft_post_rx(rx_size);
	} else {
		ret = ft_get_rx_comp(rx_seq);
		if (ret)
			return ret;

		rma_iov = rx_buf + ft_rx_prefix_size();
		*peer_iov = *rma_iov;
		ret = ft_post_rx(rx_size);
		if (ret)
			return ret;

		rma_iov = tx_buf + ft_tx_prefix_size();
		rma_iov->addr = fi->domain_attr->mr_mode == FI_MR_SCALABLE ?
				0 : (uintptr_t) rx_buf + ft_rx_prefix_size();
		rma_iov->key = fi_mr_key(mr);
		ret = ft_tx(sizeof *rma_iov);
	}

	return ret;
}

static void ft_close_fids(void)
{
	if (mr != &no_mr)
		FT_CLOSE_FID(mr);
	FT_CLOSE_FID(ep);
	FT_CLOSE_FID(pep);
	FT_CLOSE_FID(pollset);
	FT_CLOSE_FID(rxcq);
	FT_CLOSE_FID(txcq);
	FT_CLOSE_FID(rxcntr);
	FT_CLOSE_FID(txcntr);
	FT_CLOSE_FID(av);
	FT_CLOSE_FID(eq);
	FT_CLOSE_FID(domain);
	FT_CLOSE_FID(waitset);
	FT_CLOSE_FID(fabric);
}

void ft_free_res(void)
{
	ft_close_fids();
	if (buf) {
		free(buf);
		buf = rx_buf = tx_buf = NULL;
		buf_size = rx_size = tx_size = 0;
	}
	if (fi_pep) {
		fi_freeinfo(fi_pep);
		fi_pep = NULL;
	}
	if (fi) {
		fi_freeinfo(fi);
		fi = NULL;
	}
	if (hints) {
		fi_freeinfo(hints);
		hints = NULL;
	}
}

static int dupaddr(void **dst_addr, size_t *dst_addrlen,
		void *src_addr, size_t src_addrlen)
{
	*dst_addr = malloc(src_addrlen);
	if (!*dst_addr) {
		FT_ERR("address allocation failed");
		return EAI_MEMORY;
	}
	*dst_addrlen = src_addrlen;
	memcpy(*dst_addr, src_addr, src_addrlen);
	return 0;
}

static int getaddr(char *node, char *service,
			struct fi_info *hints, uint64_t flags)
{
	int ret;
	struct fi_info *fi;

	if (!node && !service) {
		if (flags & FI_SOURCE) {
			hints->src_addr = NULL;
			hints->src_addrlen = 0;
		} else {
			hints->dest_addr = NULL;
			hints->dest_addrlen = 0;
		}
		return 0;
	}

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}
	hints->addr_format = fi->addr_format;

	if (flags & FI_SOURCE) {
		ret = dupaddr(&hints->src_addr, &hints->src_addrlen,
				fi->src_addr, fi->src_addrlen);
	} else {
		ret = dupaddr(&hints->dest_addr, &hints->dest_addrlen,
				fi->dest_addr, fi->dest_addrlen);
	}

	fi_freeinfo(fi);
	return ret;
}

int ft_getsrcaddr(char *node, char *service, struct fi_info *hints)
{
	return getaddr(node, service, hints, FI_SOURCE);
}

int ft_read_addr_opts(char **node, char **service, struct fi_info *hints,
		uint64_t *flags, struct ft_opts *opts)
{
	int ret;

	if (opts->dst_addr) {
		if (!opts->dst_port)
			opts->dst_port = default_port;

		ret = ft_getsrcaddr(opts->src_addr, opts->src_port, hints);
		if (ret)
			return ret;
		*node = opts->dst_addr;
		*service = opts->dst_port;
	} else {
		if (!opts->src_port)
			opts->src_port = default_port;

		*node = opts->src_addr;
		*service = opts->src_port;
		*flags = FI_SOURCE;
	}

	return 0;
}

char *size_str(char str[FT_STR_LEN], long long size)
{
	long long base, fraction = 0;
	char mag;

	memset(str, '\0', FT_STR_LEN);

	if (size >= (1 << 30)) {
		base = 1 << 30;
		mag = 'g';
	} else if (size >= (1 << 20)) {
		base = 1 << 20;
		mag = 'm';
	} else if (size >= (1 << 10)) {
		base = 1 << 10;
		mag = 'k';
	} else {
		base = 1;
		mag = '\0';
	}

	if (size / base < 10)
		fraction = (size % base) * 10 / base;

	if (fraction)
		snprintf(str, FT_STR_LEN, "%lld.%lld%c", size / base, fraction, mag);
	else
		snprintf(str, FT_STR_LEN, "%lld%c", size / base, mag);

	return str;
}

char *cnt_str(char str[FT_STR_LEN], long long cnt)
{
	if (cnt >= 1000000000)
		snprintf(str, FT_STR_LEN, "%lldb", cnt / 1000000000);
	else if (cnt >= 1000000)
		snprintf(str, FT_STR_LEN, "%lldm", cnt / 1000000);
	else if (cnt >= 1000)
		snprintf(str, FT_STR_LEN, "%lldk", cnt / 1000);
	else
		snprintf(str, FT_STR_LEN, "%lld", cnt);

	return str;
}

int size_to_count(int size)
{
	if (size >= (1 << 20))
		return 100;
	else if (size >= (1 << 16))
		return 1000;
	else
		return 10000;
}

void init_test(struct ft_opts *opts, char *test_name, size_t test_name_len)
{
	char sstr[FT_STR_LEN];

	size_str(sstr, opts->transfer_size);
	if (!strcmp(test_name, "custom"))
		snprintf(test_name, test_name_len, "%s_lat", sstr);
	if (!(opts->options & FT_OPT_ITER))
		opts->iterations = size_to_count(opts->transfer_size);
}

ssize_t ft_post_tx(size_t size)
{
	ssize_t ret;

	if (hints->caps & FI_TAGGED) {
		ret = fi_tsend(ep, tx_buf, size + ft_tx_prefix_size(),
				fi_mr_desc(mr), remote_fi_addr, tx_seq, &tx_ctx);
	} else {
		ret = fi_send(ep, tx_buf, size + ft_tx_prefix_size(),
				fi_mr_desc(mr), remote_fi_addr, &tx_ctx);
	}
	if (ret) {
		FT_PRINTERR("transmit", ret);
		return ret;
	}

	tx_seq++;
	return 0;
}

ssize_t ft_tx(size_t size)
{
	ssize_t ret;

	if (ft_check_opts(FT_OPT_VERIFY_DATA | FT_OPT_ACTIVE))
		ft_fill_buf((char *) tx_buf + ft_tx_prefix_size(), size);

	ret = ft_post_tx(size);
	if (ret)
		return ret;

	ret = ft_get_tx_comp(tx_seq);
	return ret;
}

ssize_t ft_post_inject(size_t size)
{
	ssize_t ret;

	if (hints->caps & FI_TAGGED) {
		ret = fi_tinject(ep, tx_buf, size + ft_tx_prefix_size(),
				remote_fi_addr, tx_seq);
	} else {
		ret = fi_inject(ep, tx_buf, size + ft_tx_prefix_size(),
				remote_fi_addr);
	}
	if (ret) {
		FT_PRINTERR("transmit", ret);
		return ret;
	}

	tx_seq++;
	tx_cq_cntr++;
	return 0;
}

ssize_t ft_inject(size_t size)
{
	ssize_t ret;

	if (ft_check_opts(FT_OPT_VERIFY_DATA | FT_OPT_ACTIVE))
		ft_fill_buf((char *) tx_buf + ft_tx_prefix_size(), size);

	ret = ft_post_inject(size);
	if (ret)
		return ret;

	return ret;
}

ssize_t ft_post_rx(size_t size)
{
	ssize_t ret;

	if (hints->caps & FI_TAGGED) {
		ret = fi_trecv(ep, rx_buf, size + ft_rx_prefix_size(), fi_mr_desc(mr),
				0, rx_seq, 0, &rx_ctx);
	} else {
		ret = fi_recv(ep, rx_buf, size + ft_rx_prefix_size(), fi_mr_desc(mr),
				0, &rx_ctx);
	}
	if (ret) {
		FT_PRINTERR("receive", ret);
		return ret;
	}

	rx_seq++;
	return 0;
}

ssize_t ft_rx(size_t size)
{
	ssize_t ret;

	ret = ft_get_rx_comp(rx_seq);
	if (ret)
		return ret;

	if (ft_check_opts(FT_OPT_VERIFY_DATA | FT_OPT_ACTIVE)) {
		ret = ft_check_buf((char *) rx_buf + ft_rx_prefix_size(), size);
		if (ret)
			return ret;
	}
	/* TODO: verify CQ data, if available */

	ret = ft_post_rx(rx_size);
	return ret;
}

/*
 * fi_cq_err_entry can be cast to any CQ entry format.
 */
static int ft_spin_for_comp(struct fid_cq *cq, uint64_t *cur,
			    uint64_t total, int timeout)
{
	struct fi_cq_err_entry comp;
	struct timespec a, b;
	int ret;

	if (timeout >= 0)
		clock_gettime(CLOCK_MONOTONIC, &a);

	while (total - *cur > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			if (timeout >= 0)
				clock_gettime(CLOCK_MONOTONIC, &a);

			(*cur)++;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			return ret;
		} else if (timeout >= 0) {
			clock_gettime(CLOCK_MONOTONIC, &b);
			if ((b.tv_sec - a.tv_sec) > timeout) {
				fprintf(stderr, "%ds timeout expired\n", timeout);
				return -FI_ENODATA;
			}
		}
	}

	return 0;
}

/*
 * fi_cq_err_entry can be cast to any CQ entry format.
 */
static int ft_wait_for_comp(struct fid_cq *cq, uint64_t *cur,
			    uint64_t total, int timeout)
{
	struct fi_cq_err_entry comp;
	int ret;

	while (total - *cur > 0) {
		ret = fi_cq_sread(cq, &comp, 1, NULL, timeout);
		if (ret > 0)
			(*cur)++;
		else if (ret < 0 && ret != -FI_EAGAIN)
			return ret;
	}

	return 0;
}

/*
 * fi_cq_err_entry can be cast to any CQ entry format.
 */
static int ft_fdwait_for_comp(struct fid_cq *cq, uint64_t *cur,
			    uint64_t total, int timeout)
{
	struct fi_cq_err_entry comp;
	struct fid *fids[1];
	int fd, ret;

	fd = cq == txcq ? tx_fd : rx_fd;
	fids[0] = &cq->fid;

	while (total - *cur > 0) {
		ret = fi_trywait(fabric, fids, 1);
		if (ret == FI_SUCCESS) {
			ret = ft_poll_fd(fd, timeout);
			if (ret && ret != -FI_EAGAIN)
				return ret;
		}

		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			(*cur)++;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			return ret;
		}
	}

	return 0;
}

static int ft_get_cq_comp(struct fid_cq *cq, uint64_t *cur,
			  uint64_t total, int timeout)
{
	int ret;

	switch (opts.comp_method) {
	case FT_COMP_SREAD:
		ret = ft_wait_for_comp(cq, cur, total, timeout);
		break;
	case FT_COMP_WAIT_FD:
		ret = ft_fdwait_for_comp(cq, cur, total, timeout);
		break;
	default:
		ret = ft_spin_for_comp(cq, cur, total, timeout);
		break;
	}

	if (ret) {
		if (ret == -FI_EAVAIL) {
			ret = ft_cq_readerr(cq);
			(*cur)++;
		} else {
			FT_PRINTERR("ft_get_cq_comp", ret);
		}
	}
	return ret;
}

int ft_get_rx_comp(uint64_t total)
{
	int ret = FI_SUCCESS;

	if (rxcq) {
		ret = ft_get_cq_comp(rxcq, &rx_cq_cntr, total, timeout);
	} else if (rxcntr) {
		while (fi_cntr_read(rxcntr) < total) {
			ret = fi_cntr_wait(rxcntr, total, timeout);
			if (ret)
				FT_PRINTERR("fi_cntr_wait", ret);
			else
				break;
		}
	} else {
		FT_ERR("Trying to get a RX completion when no RX CQ or counter were opened");
		ret = -FI_EOTHER;
	}
	return ret;
}

int ft_get_tx_comp(uint64_t total)
{
	int ret;

	if (txcq) {
		ret = ft_get_cq_comp(txcq, &tx_cq_cntr, total, -1);
	} else if (txcntr) {
		ret = fi_cntr_wait(txcntr, total, -1);
		if (ret)
			FT_PRINTERR("fi_cntr_wait", ret);
	} else {
		FT_ERR("Trying to get a TX completion when no TX CQ or counter were opened");
		ret = -FI_EOTHER;
	}
	return ret;
}

int ft_cq_readerr(struct fid_cq *cq)
{
	struct fi_cq_err_entry cq_err;
	int ret;

	ret = fi_cq_readerr(cq, &cq_err, 0);
	if (ret < 0) {
		FT_PRINTERR("fi_cq_readerr", ret);
	} else {
		FT_CQ_ERR(cq, cq_err, NULL, 0);
		ret = -cq_err.err;
	}
	return ret;
}

void eq_readerr(struct fid_eq *eq, const char *eq_str)
{
	struct fi_eq_err_entry eq_err;
	int rd;

	rd = fi_eq_readerr(eq, &eq_err, 0);
	if (rd != sizeof(eq_err)) {
		FT_PRINTERR("fi_eq_readerr", rd);
	} else {
		FT_EQ_ERR(eq, eq_err, NULL, 0);
	}
}

int ft_sync()
{
	int ret;

	if (opts.dst_addr) {
		ret = ft_tx(1);
		if (ret)
			return ret;

		ret = ft_rx(1);
	} else {
		ret = ft_rx(1);
		if (ret)
			return ret;

		ret = ft_tx(1);
	}

	return ret;
}

int ft_sync_pair(int status)
{
	int ret;
	int pair_status;

	if (ft_parent_proc) {
		ret = write(ft_socket_pair[1], &status, sizeof(int));
		if (ret < 0) {
			FT_PRINTERR("write", errno);
			return ret;
		}
		ret = read(ft_socket_pair[1], &pair_status, sizeof(int));
		if (ret < 0) {
			FT_PRINTERR("read", errno);
			return ret;
		}
	} else {
		ret = read(ft_socket_pair[0], &pair_status, sizeof(int));
		if (ret < 0) {
			FT_PRINTERR("read", errno);
			return ret;
		}
		ret = write(ft_socket_pair[0], &status, sizeof(int));
		if (ret < 0) {
			FT_PRINTERR("write", errno);
			return ret;
		}
	}

	/* check status reported the other guy */
	if (pair_status != FI_SUCCESS)
		return pair_status;

	return 0;
}

int ft_fork_and_pair()
{
	int ret;

	ret = socketpair(AF_LOCAL, SOCK_STREAM, 0, ft_socket_pair);
	if (ret) {
		FT_PRINTERR("socketpair", errno);
		return -errno;
	}

	ft_child_pid = fork();
	if (ft_child_pid < 0) {
		FT_PRINTERR("fork", ft_child_pid);
		return -errno;
	}
	if (ft_child_pid)
		ft_parent_proc = 1;

	return 0;
}

int ft_wait_child()
{
	int ret;

	ret = close(ft_socket_pair[0]);
	if (ret) {
		FT_PRINTERR("close", errno);
		return ret;
	}
	ret = close(ft_socket_pair[1]);
	if (ret) {
		FT_PRINTERR("close", errno);
		return ret;
	}
	if (ft_parent_proc) {
		ret = waitpid(ft_child_pid, NULL, WCONTINUED);
		if (ret < 0) {
			FT_PRINTERR("waitpid", errno);
			return ret;
		}
	}

	return 0;
}

int ft_finalize(void)
{
	struct iovec iov;
	int ret;

	strcpy(tx_buf + ft_tx_prefix_size(), "fin");
	iov.iov_base = tx_buf;
	iov.iov_len = 4 + ft_tx_prefix_size();

	if (hints->caps & FI_TAGGED) {
		struct fi_msg_tagged tmsg;

		memset(&tmsg, 0, sizeof tmsg);
		tmsg.msg_iov = &iov;
		tmsg.iov_count = 1;
		tmsg.addr = remote_fi_addr;
		tmsg.tag = tx_seq;
		tmsg.ignore = 0;

		ret = fi_tsendmsg(ep, &tmsg, FI_INJECT | FI_TRANSMIT_COMPLETE);
	} else {
		struct fi_msg msg;

		memset(&msg, 0, sizeof msg);
		msg.msg_iov = &iov;
		msg.iov_count = 1;
		msg.addr = remote_fi_addr;

		ret = fi_sendmsg(ep, &msg, FI_INJECT | FI_TRANSMIT_COMPLETE);
	}
	if (ret) {
		FT_PRINTERR("transmit", ret);
		return ret;
	}


	ret = ft_get_tx_comp(++tx_seq);
	if (ret)
		return ret;

	ret = ft_get_rx_comp(rx_seq);
	if (ret)
		return ret;

	return 0;
}

int64_t get_elapsed(const struct timespec *b, const struct timespec *a,
		    enum precision p)
{
	int64_t elapsed;

	elapsed = difftime(a->tv_sec, b->tv_sec) * 1000 * 1000 * 1000;
	elapsed += a->tv_nsec - b->tv_nsec;
	return elapsed / p;
}

void show_perf(char *name, int tsize, int iters, struct timespec *start,
		struct timespec *end, int xfers_per_iter)
{
	static int header = 1;
	char str[FT_STR_LEN];
	int64_t elapsed = get_elapsed(start, end, MICRO);
	long long bytes = (long long) iters * tsize * xfers_per_iter;
	float usec_per_xfer;

	if (name) {
		if (header) {
			printf("%-50s%-8s%-8s%-8s%8s %10s%13s%13s\n",
					"name", "bytes", "iters",
					"total", "time", "MB/sec",
					"usec/xfer", "Mxfers/sec");
			header = 0;
		}

		printf("%-50s", name);
	} else {
		if (header) {
			printf("%-8s%-8s%-8s%8s %10s%13s%13s\n",
					"bytes", "iters", "total",
					"time", "MB/sec", "usec/xfer",
					"Mxfers/sec");
			header = 0;
		}
	}

	printf("%-8s", size_str(str, tsize));

	printf("%-8s", cnt_str(str, iters));

	printf("%-8s", size_str(str, bytes));

	usec_per_xfer = ((float)elapsed / iters / xfers_per_iter);
	printf("%8.2fs%10.2f%11.2f%11.2f\n",
		elapsed / 1000000.0, bytes / (1.0 * elapsed),
		usec_per_xfer, 1.0/usec_per_xfer);
}

void show_perf_mr(int tsize, int iters, struct timespec *start,
		  struct timespec *end, int xfers_per_iter, int argc, char *argv[])
{
	static int header = 1;
	int64_t elapsed = get_elapsed(start, end, MICRO);
	long long total = (long long) iters * tsize * xfers_per_iter;
	int i;
	float usec_per_xfer;

	if (header) {
		printf("---\n");

		for (i = 0; i < argc; ++i)
			printf("%s ", argv[i]);

		printf(":\n");
		header = 0;
	}

	usec_per_xfer = ((float)elapsed / iters / xfers_per_iter);

	printf("- { ");
	printf("xfer_size: %d, ", tsize);
	printf("iterations: %d, ", iters);
	printf("total: %lld, ", total);
	printf("time: %f, ", elapsed / 1000000.0);
	printf("MB/sec: %f, ", (total) / (1.0 * elapsed));
	printf("usec/xfer: %f, ", usec_per_xfer);
	printf("Mxfers/sec: %f", 1.0/usec_per_xfer);
	printf(" }\n");
}

void ft_usage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);

	if (desc)
		fprintf(stderr, "\n%s\n", desc);

	fprintf(stderr, "\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-n <domain>", "domain name");
	FT_PRINT_OPTS_USAGE("-b <src_port>", "non default source port number");
	FT_PRINT_OPTS_USAGE("-p <dst_port>", "non default destination port number");
	FT_PRINT_OPTS_USAGE("-f <provider>", "specific provider name eg sockets, verbs");
	FT_PRINT_OPTS_USAGE("-s <address>", "source address");
	FT_PRINT_OPTS_USAGE("-a <address vector name>", "name of address vector");
	FT_PRINT_OPTS_USAGE("-h", "display this help output");

	return;
}

void ft_csusage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);

	if (desc)
		fprintf(stderr, "\n%s\n", desc);

	fprintf(stderr, "\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-n <domain>", "domain name");
	FT_PRINT_OPTS_USAGE("-b <src_port>", "non default source port number");
	FT_PRINT_OPTS_USAGE("-p <dst_port>", "non default destination port number");
	FT_PRINT_OPTS_USAGE("-f <provider>", "specific provider name eg sockets, verbs");
	FT_PRINT_OPTS_USAGE("-s <address>", "source address");
	FT_PRINT_OPTS_USAGE("-I <number>", "number of iterations");
	FT_PRINT_OPTS_USAGE("-w <number>", "number of warmup iterations");
	FT_PRINT_OPTS_USAGE("-S <size>", "specific transfer size or 'all'");
	FT_PRINT_OPTS_USAGE("-l", "align transmit and receive buffers to page size");
	FT_PRINT_OPTS_USAGE("-m", "machine readable output");
	FT_PRINT_OPTS_USAGE("-t <type>", "completion type [queue, counter]");
	FT_PRINT_OPTS_USAGE("-c <method>", "completion method [spin, sread, fd]");
	FT_PRINT_OPTS_USAGE("-h", "display this help output");

	return;
}

void ft_parseinfo(int op, char *optarg, struct fi_info *hints)
{
	switch (op) {
	case 'n':
		if (!hints->domain_attr) {
			hints->domain_attr = malloc(sizeof *(hints->domain_attr));
			if (!hints->domain_attr) {
				perror("malloc");
				exit(EXIT_FAILURE);
			}
		}
		hints->domain_attr->name = strdup(optarg);
		break;
	case 'f':
		if (!hints->fabric_attr) {
			hints->fabric_attr = malloc(sizeof *(hints->fabric_attr));
			if (!hints->fabric_attr) {
				perror("malloc");
				exit(EXIT_FAILURE);
			}
		}
		hints->fabric_attr->prov_name = strdup(optarg);
		break;
	default:
		/* let getopt handle unknown opts*/
		break;

	}
}

void ft_parse_addr_opts(int op, char *optarg, struct ft_opts *opts)
{
	switch (op) {
	case 's':
		opts->src_addr = optarg;
		break;
	case 'b':
		opts->src_port = optarg;
		break;
	case 'p':
		opts->dst_port = optarg;
		break;
	default:
		/* let getopt handle unknown opts*/
		break;
	}
}

void ft_parsecsopts(int op, char *optarg, struct ft_opts *opts)
{
	ft_parse_addr_opts(op, optarg, opts);

	switch (op) {
	case 'I':
		opts->options |= FT_OPT_ITER;
		opts->iterations = atoi(optarg);
		break;
	case 'S':
		if (!strncasecmp("all", optarg, 3)) {
			opts->sizes_enabled = FT_ENABLE_ALL;
		} else {
			opts->options |= FT_OPT_SIZE;
			opts->transfer_size = atoi(optarg);
		}
		break;
	case 'm':
		opts->machr = 1;
		break;
	case 'c':
		if (!strncasecmp("sread", optarg, 5))
			opts->comp_method = FT_COMP_SREAD;
		else if (!strncasecmp("fd", optarg, 2))
			opts->comp_method = FT_COMP_WAIT_FD;
		break;
	case 't':
		if (!strncasecmp("counter", optarg, 7)) {
			opts->options |= FT_OPT_RX_CNTR | FT_OPT_TX_CNTR;
			opts->options &= ~(FT_OPT_RX_CQ | FT_OPT_TX_CQ);
		}
		break;
	case 'a':
		opts->av_name = optarg;
		break;
	case 'w':
		opts->warmup_iterations = atoi(optarg);
		break;
	case 'l':
		opts->options |= FT_OPT_ALIGN;
		break;
	default:
		/* let getopt handle unknown opts*/
		break;
	}
}

void ft_fill_buf(void *buf, int size)
{
	char *msg_buf;
	int msg_index;
	static unsigned int iter = 0;
	int i;

	msg_index = ((iter++)*INTEG_SEED) % integ_alphabet_length;
	msg_buf = (char *)buf;
	for (i = 0; i < size; i++) {
		msg_buf[i] = integ_alphabet[msg_index++];
		if (msg_index >= integ_alphabet_length)
			msg_index = 0;
	}
}

int ft_check_buf(void *buf, int size)
{
	char *recv_data;
	char c;
	static unsigned int iter = 0;
	int msg_index;
	int i;

	msg_index = ((iter++)*INTEG_SEED) % integ_alphabet_length;
	recv_data = (char *)buf;

	for (i = 0; i < size; i++) {
		c = integ_alphabet[msg_index++];
		if (msg_index >= integ_alphabet_length)
			msg_index = 0;
		if (c != recv_data[i])
			break;
	}
	if (i != size) {
		printf("Error at iteration=%d size=%d byte=%d\n",
			iter, size, i);
		return 1;
	}

	return 0;
}

uint64_t ft_init_cq_data(struct fi_info *info)
{
	if (info->domain_attr->cq_data_size >= sizeof(uint64_t)) {
		return 0x0123456789abcdefULL;
	} else {
		return 0x0123456789abcdef &
			((0x1ULL << (info->domain_attr->cq_data_size * 8)) - 1);
	}
}

int check_recv_msg(const char *message)
{
	size_t recv_len;
	size_t message_len = strlen(message) + 1;
	/* Account for null terminated byte. */
	recv_len = strlen(rx_buf) + 1;

	if (recv_len != message_len) {
		fprintf(stderr, "Received length does not match expected length.\n");
		return -1;
	}

	if (strncmp(rx_buf, message, message_len)) {
		fprintf(stderr, "Received message does not match expected message.\n");
		return -1;
	}
	fprintf(stdout, "Data check OK\n");
	return 0;
}

int send_recv_greeting(void)
{
	int ret;
	const char *message = "Hello from Client!";
	size_t message_len = strlen(message) + 1;

	if (opts.dst_addr) {
		fprintf(stdout, "Sending message...\n");
		if (snprintf(tx_buf, tx_size, "%s", message) >= tx_size) {
			fprintf(stderr, "Transmit buffer too small.\n");
			return -FI_ETOOSMALL;
		}

		ret = ft_tx(message_len);
		if (ret)
			return ret;

		fprintf(stdout, "Send completion received\n");
	} else {
		fprintf(stdout, "Waiting for message from client...\n");
		ret = ft_get_rx_comp(rx_seq);
		if (ret)
			return ret;

		ret = check_recv_msg(message);
		if (ret)
			return ret;

		fprintf(stdout, "Received data from client: %s\n",
				(char *) rx_buf);
	}

	return 0;
}
