/*
 * Copyright (c) Intel Corporation.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include <shared.h>
#include <hmem.h>

#define SENTINEL	0xa5
#define ITER_STRIDE	0x632be5abu
#define KNUTH_HASH	2654435761u
#define CANCEL_WAIT	5
#define NAME_LEN	256

struct segment {
	void			*buf;
	size_t			len;
	enum fi_hmem_iface	iface;
	uint64_t		device;
	struct fid_mr		*mr;
	void			*desc;
};

struct test_case {
	const char	*source;
	const char	*layout;
};

static const struct test_case default_cases[] = {
	{ "h", "h" },
	{ "h", "h,h" },
	{ "h", "h,d0" },
	{ "h", "d0,h,d0" },
	{ "h", "h,d0,h" },
	{ "d0", "h" },
	{ "d0", "h,h" },
	{ "d0", "d0,d0" },
	{ "d0", "h,d0" },
	{ "d0", "h,d0,h" },
	{ "d0", "d0,h,d0" },
};

enum source_kind {
	SOURCE_CLIENT,
	SOURCE_HOST,
	SOURCE_DEVICE,
};

struct case_info {
	uint64_t	nsegs;
	uint64_t	seg_len;
	uint64_t	source_kind;
	uint64_t	source_device;
	char		names[NAME_LEN];
};

static struct segment *segs;
static size_t nsegs;
static const char *layout;
static size_t seg_len;
static bool device_set;
static enum fi_hmem_iface src_iface;
static uint64_t src_device;
static char *stage;
static bool recv_posted;
static bool send_posted;
static char direction[2 * NAME_LEN + 3];
static size_t failed_size;

/* The byte expected at a position in the message, for a given iteration. It
 * depends on both, so a byte in the wrong place, or left over from another
 * message, does not match.
 *
 * pos % N repeats every N bytes, and data landing a multiple of N bytes away
 * from where it belongs would still match. So the position is multiplied by
 * 2654435761, Knuth's multiplicative hash. It is the prime nearest 2^32 / phi,
 * phi being the golden ratio, which spreads neighboring positions about as
 * far apart as any multiplier can. It is odd, so the multiplication modulo
 * 2^32 is a bijection and no two positions collapse into one. The top byte is
 * used because every bit of the position reaches it; the low bits of a
 * product only depend on the low bits of the position.
 *
 * The iteration is added to the position in a large odd stride, so the data
 * of one iteration is never a shifted copy of another's.
 *
 * The result is 1 to 160: never 0, which is what fresh memory holds, and below
 * the sentinel (0xa5 = 165), so unwritten memory never looks like data.
 */
static unsigned char pattern(size_t pos, int iter)
{
	uint32_t x = (uint32_t) pos + (uint32_t) iter * ITER_STRIDE;

	return (x * KNUTH_HASH >> 24) % 160 + 1;
}

static int parse_memory(const char *tok, enum fi_hmem_iface *iface,
			uint64_t *device)
{
	char *end;

	if (!strcmp(tok, "h")) {
		*iface = FI_HMEM_SYSTEM;
		*device = 0;
		return 0;
	}

	if (tok[0] != 'd' || !tok[1]) {
		FT_ERR("invalid layout entry '%s'", tok);
		return -FI_EINVAL;
	}

	*device = strtoul(tok + 1, &end, 0);
	if (*end) {
		FT_ERR("invalid layout entry '%s'", tok);
		return -FI_EINVAL;
	}

	if (opts.iface == FI_HMEM_SYSTEM) {
		FT_ERR("layout entry '%s' needs -D <iface>", tok);
		return -FI_EINVAL;
	}

	*iface = opts.iface;
	return 0;
}

static int parse_layout(const char *lay)
{
	size_t max_segs = fi->rx_attr->iov_limit, count = 1;
	char *copy, *tok, *next;
	const char *c;
	int ret = 0;

	if (!max_segs) {
		FT_ERR("provider does not receive into iovs");
		return -FI_EINVAL;
	}

	for (c = lay; *c; c++)
		count += *c == ',';

	if (count > max_segs) {
		FT_ERR("layout '%s' has %zu entries, more than the %zu "
		       "iovs the provider receives", lay, count, max_segs);
		return -FI_EINVAL;
	}

	segs = calloc(count, sizeof(*segs));
	copy = strdup(lay);
	if (!segs || !copy) {
		free(copy);
		return -FI_ENOMEM;
	}

	for (tok = copy; tok && !ret; tok = next) {
		next = strchr(tok, ',');
		if (next)
			*next++ = '\0';

		ret = parse_memory(tok, &segs[nsegs].iface,
				   &segs[nsegs].device);
		nsegs++;
	}

	free(copy);
	return ret;
}

static void set_direction(const char *src, const char *dst)
{
	snprintf(direction, sizeof(direction), "%s->%s", src, dst);
}

static const char *iface_name(enum fi_hmem_iface iface)
{
	switch (iface) {
	case FI_HMEM_CUDA:
		return "CUDA";
	case FI_HMEM_ROCR:
		return "ROCR";
	case FI_HMEM_ZE:
		return "ZE";
	case FI_HMEM_NEURON:
		return "NEURON";
	case FI_HMEM_SYNAPSEAI:
		return "SYNAPSEAI";
	default:
		return "?";
	}
}

static int memory_name(char *name, size_t size, enum fi_hmem_iface iface,
		       uint64_t device)
{
	int len;

	if (iface == FI_HMEM_SYSTEM)
		len = snprintf(name, size, "H");
	else
		len = snprintf(name, size, "%s:D%" PRIu64, iface_name(iface),
			       device);

	return len < (int) size ? 0 : -FI_EINVAL;
}

static int layout_name(char *name)
{
	size_t i, len = 0;
	int ret;

	for (i = 0; i < nsegs; i++) {
		if (i)
			name[len++] = ',';

		ret = memory_name(name + len, NAME_LEN - len, segs[i].iface,
				  segs[i].device);
		if (ret) {
			FT_ERR("the layout is longer than %d characters",
			       NAME_LEN - 1);
			return ret;
		}
		len += strlen(name + len);
	}

	return 0;
}

static int exchange_server(const struct test_case *tc)
{
	struct case_info info = { .seg_len = seg_len };
	char src[NAME_LEN];
	size_t i;
	int ret;

	ret = parse_layout(tc ? tc->layout : layout);
	if (ret)
		return ret;

	for (i = 0; tc && device_set && i < nsegs; i++) {
		if (segs[i].iface != FI_HMEM_SYSTEM)
			segs[i].device = opts.device;
	}

	info.nsegs = nsegs;
	info.source_kind = SOURCE_CLIENT;
	if (tc && !strcmp(tc->source, "h")) {
		info.source_kind = SOURCE_HOST;
	} else if (tc) {
		info.source_kind = SOURCE_DEVICE;
		info.source_device = strtoul(tc->source + 1, NULL, 0);
	}

	ret = layout_name(info.names);
	if (ret)
		return ret;

	ret = ft_sock_send(oob_sock, &info, sizeof(info));
	if (ret)
		return ret;

	ret = ft_sock_recv(oob_sock, src, sizeof(src));
	if (ret)
		return ret;

	src[NAME_LEN - 1] = '\0';
	set_direction(src, info.names);
	return 0;
}

static int pick_source(const struct case_info *info)
{
	if (info->source_kind == SOURCE_CLIENT) {
		if (opts.iface == FI_HMEM_SYSTEM)
			FT_DEBUG("no -D given, assuming -H and sending from "
				 "host memory");

		src_iface = opts.iface;
		src_device = opts.device;
		return 0;
	}

	if (info->source_kind == SOURCE_HOST) {
		src_iface = FI_HMEM_SYSTEM;
		src_device = 0;
		return 0;
	}

	if (opts.iface == FI_HMEM_SYSTEM) {
		FT_ERR("the server sends from device %" PRIu64 ", the client "
		       "needs -D <iface>", info->source_device);
		return -FI_EINVAL;
	}

	src_iface = opts.iface;
	src_device = device_set ? opts.device : info->source_device;
	return 0;
}

static int exchange_client(void)
{
	struct case_info info;
	char src[NAME_LEN];
	int ret;

	ret = ft_sock_recv(oob_sock, &info, sizeof(info));
	if (ret) {
		FT_PRINTERR("receiving the test from the server", ret);
		return ret;
	}

	nsegs = info.nsegs;
	seg_len = info.seg_len;
	if (!nsegs) {
		FT_ERR("the server sent an empty layout");
		return -FI_EINVAL;
	}

	ret = pick_source(&info);
	if (ret)
		return ret;

	ret = memory_name(src, sizeof(src), src_iface, src_device);
	if (ret)
		return ret;

	ret = ft_sock_send(oob_sock, src, sizeof(src));
	if (ret)
		return ret;

	info.names[NAME_LEN - 1] = '\0';
	set_direction(src, info.names);
	return 0;
}

static size_t posted_len(size_t size)
{
	return seg_len ? seg_len : (size + nsegs - 1) / nsegs;
}

static bool fits(size_t size)
{
	return nsegs * posted_len(size) >= size;
}

static int fill_sentinel(struct segment *seg)
{
	memset(stage, SENTINEL, seg->len);
	return ft_hmem_copy_to(seg->iface, seg->device, seg->buf, stage,
			       seg->len);
}

static int alloc_segments(size_t seg_cap)
{
	size_t i;
	int ret;

	for (i = 0; i < nsegs; i++) {
		ret = ft_hmem_alloc(segs[i].iface, segs[i].device,
				    &segs[i].buf, seg_cap);
		if (ret)
			return ret;

		ret = ft_reg_mr(fi, segs[i].buf, seg_cap, FI_RECV,
				FT_MR_KEY + i, segs[i].iface, segs[i].device,
				&segs[i].mr, &segs[i].desc);
		if (ret)
			return ret;
	}

	return 0;
}

static void free_segments(void)
{
	size_t i;

	if (!segs)
		return;

	for (i = 0; i < nsegs; i++) {
		FT_CLOSE_FID(segs[i].mr);
		if (segs[i].buf)
			(void) ft_hmem_free(segs[i].iface, segs[i].buf);
		segs[i].buf = NULL;
	}

	free(segs);
	segs = NULL;
	nsegs = 0;
}

static int post_recvv(void)
{
	struct iovec iov[nsegs];
	void *desc[nsegs];
	size_t i;
	int ret;

	for (i = 0; i < nsegs; i++) {
		ret = fill_sentinel(&segs[i]);
		if (ret)
			return ret;

		iov[i].iov_base = segs[i].buf;
		iov[i].iov_len = segs[i].len;
		desc[i] = segs[i].desc;
	}

	do {
		ret = (int) fi_recvv(ep, iov, desc, nsegs,
				     FI_ADDR_UNSPEC, &rx_ctx);
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(rxcq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret)
		FT_PRINTERR("fi_recvv", ret);
	else
		recv_posted = true;
	return ret;
}

static int wait_recvv(struct fi_cq_data_entry *comp)
{
	time_t end = time(NULL) + timeout;
	ssize_t ret;

	do {
		ret = fi_cq_read(rxcq, comp, 1);
		if (ret == -FI_EAGAIN && timeout >= 0 && time(NULL) > end) {
			FT_ERR("%ds timeout expired waiting for the receive",
			       timeout);
			return -FI_ETIMEDOUT;
		}
	} while (ret == -FI_EAGAIN);

	if (ret < 0) {
		if (ret == -FI_EAVAIL) {
			ret = ft_cq_readerr(rxcq);
			recv_posted = false;
		} else {
			FT_PRINTERR("fi_cq_read", ret);
		}
		return (int) ret;
	}

	if (comp->op_context != &rx_ctx) {
		FT_ERR("completion for context %p, expected %p",
		       comp->op_context, (void *) &rx_ctx);
		return -FI_EIO;
	}

	if (!(comp->flags & FI_RECV)) {
		FT_ERR("completion flags 0x%llx do not include FI_RECV",
		       (unsigned long long) comp->flags);
		return -FI_EIO;
	}

	recv_posted = false;
	return 0;
}

static void drain_cq(struct fid_cq *cq)
{
	struct fi_cq_data_entry comp;
	struct fi_cq_err_entry err;
	time_t end = time(NULL) + CANCEL_WAIT;
	ssize_t ret;

	while (time(NULL) < end) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret == 1)
			return;

		if (ret == -FI_EAVAIL) {
			(void) fi_cq_readerr(cq, &err, 0);
			return;
		}

		if (ret != -FI_EAGAIN)
			return;
	}

	FT_WARN("timed out waiting for a cancelled operation");
}

static void cancel_pending(void)
{
	ssize_t ret;

	if (recv_posted) {
		ret = fi_cancel(&ep->fid, &rx_ctx);
		if (ret)
			FT_WARN("fi_cancel of the receive: %s",
				fi_strerror((int) -ret));
		else
			drain_cq(rxcq);
		recv_posted = false;
	}

	if (send_posted) {
		ret = fi_cancel(&ep->fid, &tx_ctx);
		if (ret)
			FT_WARN("fi_cancel of the send: %s",
				fi_strerror((int) -ret));
		else
			drain_cq(txcq);
		send_posted = false;
	}
}

/* The completion must report the message length, the message must fill the
 * posted buffers in order, and the bytes after its end must still hold the
 * sentinel.
 */
static int check_recvv(size_t msg_len, struct fi_cq_data_entry *comp,
		       int iter)
{
	size_t i, j, pos = 0, covered;
	int ret;

	if (comp->len != msg_len) {
		FT_ERR("completion length %zu, expected %zu", comp->len,
		       msg_len);
		return -FI_EIO;
	}

	for (i = 0; i < nsegs; i++) {
		ret = ft_hmem_copy_from(segs[i].iface, segs[i].device, stage,
					segs[i].buf, segs[i].len);
		if (ret)
			return ret;

		covered = msg_len > pos ? MIN(segs[i].len, msg_len - pos) : 0;
		for (j = 0; j < covered; j++) {
			if ((unsigned char) stage[j] !=
			    pattern(pos + j, iter)) {
				FT_ERR("buffer %zu byte %zu: got 0x%02x, "
				       "expected 0x%02x", i, j,
				       (unsigned char) stage[j],
				       pattern(pos + j, iter));
				return -FI_EIO;
			}
		}
		for (; j < segs[i].len; j++) {
			if ((unsigned char) stage[j] != SENTINEL) {
				FT_ERR("buffer %zu byte %zu was written past "
				       "the end of the message", i, j);
				return -FI_EIO;
			}
		}
		pos += covered;
	}

	return 0;
}

static int run_server(void)
{
	struct fi_cq_data_entry comp;
	size_t i;
	int j, ret;

	for (i = 0; i < nsegs; i++)
		segs[i].len = posted_len(opts.transfer_size);

	for (j = 0; j < opts.iterations; j++) {
		ret = post_recvv();
		if (ret)
			return ret;

		ret = ft_sync();
		if (ret)
			return ret;

		ret = wait_recvv(&comp);
		if (ret)
			return ret;

		ret = check_recvv(opts.transfer_size, &comp, j);
		if (ret)
			return ret;
	}

	return 0;
}

static int run_client(void)
{
	size_t i;
	int j, ret;

	for (j = 0; j < opts.iterations; j++) {
		for (i = 0; i < opts.transfer_size; i++)
			stage[i] = pattern(i, j);

		ret = ft_hmem_copy_to(src_iface, src_device, tx_buf, stage,
				      opts.transfer_size);
		if (ret)
			return ret;

		ret = ft_sync();
		if (ret)
			return ret;

		ret = (int) ft_post_tx(ep, remote_fi_addr, opts.transfer_size,
				       NO_CQ_DATA, &tx_ctx);
		if (ret)
			return ret;
		send_posted = true;

		ret = ft_get_tx_comp(tx_seq);
		if (ret)
			return ret;
		send_posted = false;
	}

	return 0;
}

struct run_info {
	uint64_t	ncases;
	uint64_t	server_device;
	uint64_t	client_device;
};

static bool server_device;
static bool client_device;

static bool case_enabled(const struct test_case *tc)
{
	return (server_device || !strchr(tc->layout, 'd')) &&
	       (client_device || !strchr(tc->source, 'd'));
}

static const struct test_case *next_case(size_t *idx)
{
	while (*idx < ARRAY_SIZE(default_cases)) {
		if (case_enabled(&default_cases[*idx]))
			return &default_cases[(*idx)++];

		(*idx)++;
	}

	return NULL;
}

static uint64_t count_cases(void)
{
	size_t idx = 0;
	uint64_t count = 0;

	while (next_case(&idx))
		count++;

	return count;
}

static size_t list_name_len(const char *list, size_t devlen)
{
	size_t len = 0;

	for (; *list; list++) {
		if (*list == ',' || *list == 'h')
			len++;
		else if (*list == 'd')
			len += devlen;
	}

	return len;
}

static size_t default_width(void)
{
	char dev[NAME_LEN], src[NAME_LEN];
	size_t i, len, max = 0;

	(void) memory_name(dev, sizeof(dev), opts.iface,
			   opts.dst_addr ? 0 : opts.device);
	(void) memory_name(src, sizeof(src), opts.iface,
			   opts.dst_addr ? opts.device : 0);
	for (i = 0; i < ARRAY_SIZE(default_cases); i++) {
		if (!case_enabled(&default_cases[i]))
			continue;

		len = list_name_len(default_cases[i].source, strlen(src)) + 2 +
		      list_name_len(default_cases[i].layout, strlen(dev));
		max = MAX(max, len);
	}

	return max;
}

static void show_test(void)
{
	size_t i, len = strlen(direction);
	size_t width = MAX(default_width(), len) + 2;

	printf("%s ", direction);
	for (i = len + 1; i < width; i++)
		putchar('-');

	printf("> ");
	fflush(stdout);
}

static void show_result(int ret)
{
	printf("%s", ret ? "FAIL" : "PASS");
	if (ret && failed_size)
		printf(" (%zu bytes)", failed_size);

	printf("\n");
	fflush(stdout);
}

static int run_size(void)
{
	int ret;

	ret = opts.dst_addr ? run_client() : run_server();
	if (ret)
		failed_size = opts.transfer_size;
	return ret;
}

static size_t max_size(void)
{
	size_t max = 0;
	int i;

	if (opts.options & FT_OPT_SIZE)
		return opts.transfer_size;

	for (i = 0; i < TEST_CNT; i++) {
		if (ft_use_size(i, opts.sizes_enabled))
			max = MAX(max, test_size[i].size);
	}
	return max;
}

static int alloc_case(void)
{
	size_t max = max_size(), cap;
	int ret;

	if (!max) {
		FT_ERR("no message sizes are enabled");
		return -FI_EINVAL;
	}

	if (opts.dst_addr) {
		stage = malloc(max);
		if (!stage)
			return -FI_ENOMEM;

		ret = ft_hmem_alloc(src_iface, src_device, (void **) &tx_buf,
				    max);
		if (ret)
			return ret;

		return ft_reg_mr(fi, tx_buf, max, FI_SEND, FT_MR_KEY,
				 src_iface, src_device, &mr, &mr_desc);
	}

	cap = posted_len(max);
	stage = malloc(cap);
	if (!stage)
		return -FI_ENOMEM;

	return alloc_segments(cap);
}

static void free_case(void)
{
	free_segments();
	free(stage);
	stage = NULL;
	FT_CLOSE_FID(mr);
	if (tx_buf) {
		(void) ft_hmem_free(src_iface, tx_buf);
		tx_buf = NULL;
	}
}

static int run_sizes(void)
{
	size_t ran = 0;
	int i, ret = 0;

	if (opts.options & FT_OPT_SIZE) {
		if (fits(opts.transfer_size))
			return run_size();

		FT_ERR("%zu buffers of %zu bytes cannot hold a %zu byte "
		       "message", nsegs, posted_len(opts.transfer_size),
		       opts.transfer_size);
		return -FI_EINVAL;
	}

	for (i = 0; i < TEST_CNT && !ret; i++) {
		if (!ft_use_size(i, opts.sizes_enabled))
			continue;

		opts.transfer_size = test_size[i].size;
		if (!fits(opts.transfer_size)) {
			FT_DEBUG("skipping %zu bytes, %zu buffers of %zu "
				 "bytes cannot hold it", opts.transfer_size,
				 nsegs, seg_len);
			continue;
		}

		ret = run_size();
		ran++;
	}

	if (!ret && !ran) {
		FT_ERR("no message size fits in %zu buffers of %zu bytes",
		       nsegs, seg_len);
		ret = -FI_EINVAL;
	}

	return ret;
}

static int run_case(const struct test_case *tc)
{
	int ret;

	ret = opts.dst_addr ? exchange_client() : exchange_server(tc);
	if (ret)
		return ret;

	failed_size = 0;
	show_test();
	ret = alloc_case();
	if (!ret)
		ret = run_sizes();

	free_case();
	show_result(ret);
	return ret;
}

static int check_hmem(void)
{
	struct fi_info *probe, *info;
	int ret;

	probe = fi_dupinfo(hints);
	if (!probe)
		return -FI_ENOMEM;

	if (!probe->ep_attr->type)
		probe->ep_attr->type = FI_EP_RDM;

	probe->caps |= FI_HMEM;
	probe->domain_attr->mr_mode |= FI_MR_HMEM;
	ret = fi_getinfo(ft_fiversion, NULL, NULL, 0, probe, &info);
	fi_freeinfo(probe);
	if (ret) {
		FT_ERR("the provider does not support FI_HMEM: %s",
		       fi_strerror(-ret));
		return ret;
	}

	fi_freeinfo(info);
	return 0;
}

static int exchange_run_info(uint64_t *ncases)
{
	struct run_info info = { 0 };
	uint64_t has_device;
	int ret;

	if (opts.dst_addr) {
		has_device = opts.iface != FI_HMEM_SYSTEM;
		ret = ft_sock_send(oob_sock, &has_device, sizeof(has_device));
		if (ret)
			return ret;

		ret = ft_sock_recv(oob_sock, &info, sizeof(info));
	} else {
		ret = ft_sock_recv(oob_sock, &has_device, sizeof(has_device));
		if (ret)
			return ret;

		server_device = opts.iface != FI_HMEM_SYSTEM;
		client_device = has_device;
		if (!layout && !server_device)
			FT_DEBUG("the server has no -D, skipping the tests that "
				 "receive into device memory");
		if (!layout && !client_device)
			FT_DEBUG("the client has no -D, skipping the tests "
				 "that send from device memory");

		info.ncases = layout ? 1 : count_cases();
		info.server_device = server_device;
		info.client_device = client_device;
		ret = ft_sock_send(oob_sock, &info, sizeof(info));
	}

	if (ret)
		return ret;

	server_device = info.server_device;
	client_device = info.client_device;
	*ncases = info.ncases;
	return 0;
}

static int run(void)
{
	const struct test_case *tc;
	uint64_t k, ncases = 0;
	size_t idx = 0;
	int ret;

	ret = check_hmem();
	if (ret)
		return ret;

	ret = hints->ep_attr->type == FI_EP_MSG ?
	      ft_init_fabric_cm() : ft_init_fabric();
	if (ret)
		return ret;

	ret = exchange_run_info(&ncases);
	for (k = 0; k < ncases && !ret; k++) {
		tc = (opts.dst_addr || layout) ? NULL : next_case(&idx);
		ret = run_case(tc);
	}

	if (ret)
		return ret;

	return ft_sync();
}

static struct option test_long_opts[] = {
	{"timeout", required_argument, NULL, LONG_OPT_TIMEOUT},
	{NULL, 0, NULL, 0},
};

static void usage(char *name)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);
	fprintf(stderr, "\nReceive messages into a list of host and device "
		"buffers with fi_recvv.\nRuns these tests, as source->receive "
		"buffers (D0 is device 0 of the\n-D interface), unless -y "
		"is given:\n  H->H  H->H,H  H->H,D0  H->D0,H,D0  H->H,D0,H\n  "
		"D0->H  D0->H,H  D0->D0,D0  D0->H,D0  D0->H,D0,H  "
		"D0->D0,H,D0\nA side without -D skips the tests that use "
		"device memory on it.\n\nOptions:\n");
	FT_PRINT_OPTS_USAGE("-b[=<oob_port>]", "Out-of-band address exchange and synchronization over the");
	FT_PRINT_OPTS_USAGE("", "optional port");
	FT_PRINT_OPTS_USAGE("-D <iface>", "Device interface, e.g. ze, cuda");
	FT_PRINT_OPTS_USAGE("", "The default tests that use device memory need it. With -y, server: needed if the");
	FT_PRINT_OPTS_USAGE("", "layout has a device entry. client: send from device memory,");
	FT_PRINT_OPTS_USAGE("", "not host (the device is chosen with -i)");
	FT_PRINT_OPTS_USAGE("-e <ep_type>", "Endpoint type: msg|rdm (default: rdm)");
	FT_PRINT_OPTS_USAGE("-E[=<oob_port>]", "Out-of-band address exchange over the optional port");
	FT_PRINT_OPTS_USAGE("-h", "Display this help output");
	FT_PRINT_OPTS_USAGE("-H", "Accepted and ignored, FI_HMEM is always requested");
	FT_PRINT_OPTS_USAGE("-i <dev_num>", "Device to use in the default tests instead of device 0 (default: 0).");
	FT_PRINT_OPTS_USAGE("", "Needs -D. It applies to the side it is given to: on the client");
	FT_PRINT_OPTS_USAGE("", "the device it sends from, on the server the device it receives");
	FT_PRINT_OPTS_USAGE("", "into. With -y the client sends from it, and the server ignores");
	FT_PRINT_OPTS_USAGE("", "it as the layout names the devices");
	FT_PRINT_OPTS_USAGE("-I <number>", "Number of iterations (default: 100)");
	FT_PRINT_OPTS_USAGE("-O <address>", "Address to use for the out-of-band connection");
	FT_PRINT_OPTS_USAGE("-p <provider>", "Provider name, e.g. shm, verbs");
	FT_PRINT_OPTS_USAGE("-s <address>", "Source address");
	FT_PRINT_OPTS_USAGE("-S <size>", "Message size, a range (syntax r:start,inc,end),");
	FT_PRINT_OPTS_USAGE("", "a list (syntax l:1,2,3,...) or 'all'");
	FT_PRINT_OPTS_USAGE("", "(default: the usual sweep of sizes)");
	FT_PRINT_OPTS_USAGE("-y <layout>", "Run only this test instead of the defaults: the");
	FT_PRINT_OPTS_USAGE("", "server's receive buffers, comma separated: h (host) or");
	FT_PRINT_OPTS_USAGE("", "d<N> (device N of the -D interface), for example h,d0,h.");
	FT_PRINT_OPTS_USAGE("", "Server only");
	FT_PRINT_OPTS_USAGE("-z <bytes>", "Size of each receive buffer, server only");
	FT_PRINT_OPTS_USAGE("", "(default: the message size divided across the buffers)");
	FT_PRINT_OPTS_USAGE("--timeout <seconds>", "Seconds to wait for the peer or a completion before failing (default: no limit)");
}

int main(int argc, char **argv)
{
	char *end;
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.iterations = 100;
	opts.options |= FT_OPT_ENABLE_HMEM | FT_OPT_SKIP_MSG_ALLOC |
			FT_OPT_OOB_SYNC | FT_OPT_OOB_ADDR_EXCH;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "hHi:y:z:D:p:e:s:I:S:b::E::O:",
				 test_long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case 'H':
			/* FI_HMEM is always requested, accepted so command
			 * lines shared with other tests work
			 */
			break;
		case 'i':
			opts.device = strtoul(optarg, &end, 0);
			if (!*optarg || *end) {
				FT_ERR("invalid device number '%s'", optarg);
				ret = EXIT_FAILURE;
				goto out;
			}
			device_set = true;
			break;
		case 'y':
			layout = optarg;
			break;
		case 'z':
			seg_len = strtoul(optarg, NULL, 0);
			break;
		case '?':
		case 'h':
			usage(argv[0]);
			ret = EXIT_FAILURE;
			goto out;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (opts.iterations < 1) {
		FT_ERR("-I must be at least 1");
		ret = EXIT_FAILURE;
		goto out;
	}

	if (opts.dst_addr && (layout || seg_len)) {
		FT_ERR("-y and -z are only for the server, which receives");
		ret = EXIT_FAILURE;
		goto out;
	}

	if (device_set && opts.iface == FI_HMEM_SYSTEM) {
		FT_ERR("-i needs -D <iface>");
		ret = EXIT_FAILURE;
		goto out;
	}

	if (!opts.dst_addr && device_set && layout) {
		FT_DEBUG("ignoring -i on the server, the -y layout selects the "
			 "devices");
		device_set = false;
		opts.device = 0;
	}


	hints->caps = FI_MSG;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	cq_attr.format = FI_CQ_FORMAT_DATA;

	ret = run();

out:
	cancel_pending();
	free_case();
	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
