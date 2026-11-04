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
#include <string.h>
#include <time.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>

#include <shared.h>
#include <hmem.h>

#define SMALL_SIZE	64
#define CTRL_SIZE	8
#define CTRL_AREA	256
#define SECOND_RECV	128
#define EXTRA_SIZE	512
#define SENTINEL	0xee
#define NUM_CTX		4
#define RAR_READS	50
#define LABEL_WIDTH	24

typedef unsigned char (*pattern_fn)(size_t pos);

enum target_state {
	TARGET_SENTINEL,
	TARGET_OLD,
};

struct order_case {
	const char	*name;
	uint64_t	order;
	bool		best_effort;
	bool		per_size;
	bool		reads;
	int		target_state;
	int		recvs;
	int		(*prepare)(size_t n);
	int		(*initiate)(size_t n);
	int		(*react)(size_t n);
	int		(*verify_target)(size_t n);
	int		(*verify_initiator)(size_t n);
};

struct run_msg {
	uint64_t	cases;
	uint64_t	iterations;
	uint64_t	nsizes;
	uint64_t	size_sum;
	uint64_t	size_max;
};

static struct fi_context2 ctx[NUM_CTX];
static struct run_msg size_info;
static bool single_size;
static size_t single_value;
static uint64_t selected = ~0ULL;
static char *stage;
static bool check_failed;
static unsigned int rar_violations;
static size_t failed_size;

/* Each operation writes its own pattern, and the byte values of the three
 * patterns never overlap and never include 0 or SENTINEL. A mismatch then
 * shows which write landed: the large one, the small one, the old contents
 * of the target, or untouched memory. The patterns repeat every 100 bytes
 * (20 for the old one) so they do not line up with the common buffer and
 * page sizes, and a copy that lands at the wrong offset is detected.
 * pattern_big is 1-100, pattern_small is 101-200, pattern_old is 211-230.
 */
static unsigned char pattern_big(size_t pos)
{
	return pos % 100 + 1;
}

static unsigned char pattern_small(size_t pos)
{
	return pos % 100 + 101;
}

static unsigned char pattern_old(size_t pos)
{
	return pos % 20 + 211;
}

static char *region(void)
{
	return rx_buf + CTRL_AREA;
}

static int put_pattern(char *buf, size_t len, pattern_fn pat)
{
	size_t i;

	for (i = 0; i < len; i++)
		stage[i] = pat(i);

	return ft_hmem_copy_to(opts.iface, opts.device, buf, stage, len);
}

static int put_fill(char *buf, size_t len, int byte)
{
	memset(stage, byte, len);
	return ft_hmem_copy_to(opts.iface, opts.device, buf, stage, len);
}

static int check_bytes(const char *what, char *buf, size_t len,
		       size_t boundary, pattern_fn first, pattern_fn second)
{
	unsigned char want;
	size_t i;
	int ret;

	ret = ft_hmem_copy_from(opts.iface, opts.device, stage, buf, len);
	if (ret)
		return ret;

	for (i = 0; i < len; i++) {
		want = i < boundary ? first(i) : second(i);
		if ((unsigned char) stage[i] == want)
			continue;

		FT_ERR("%s: byte %zu: got 0x%02x, expected 0x%02x", what, i,
		       (unsigned char) stage[i], want);
		check_failed = true;
		break;
	}

	return 0;
}

static int check_either(const char *what, char *buf, size_t len,
			pattern_fn a, pattern_fn b)
{
	unsigned char got;
	size_t i;
	int ret;

	ret = ft_hmem_copy_from(opts.iface, opts.device, stage, buf, len);
	if (ret)
		return ret;

	for (i = 0; i < len; i++) {
		got = stage[i];
		if (got == a(i) || got == b(i))
			continue;

		FT_ERR("%s: byte %zu: got 0x%02x, expected 0x%02x or 0x%02x",
		       what, i, got, a(i), b(i));
		check_failed = true;
		break;
	}

	return 0;
}

static int post_write(const void *src, size_t len, struct fi_context2 *c)
{
	FT_POST(fi_write, ft_progress, txcq, tx_seq, &tx_cq_cntr, "fi_write",
		ep, src, len, mr_desc, remote_fi_addr,
		remote.addr + CTRL_AREA, remote.key, c);
	return 0;
}

static int post_read(void *dst, size_t len, struct fi_context2 *c)
{
	FT_POST(fi_read, ft_progress, txcq, tx_seq, &tx_cq_cntr, "fi_read",
		ep, dst, len, mr_desc, remote_fi_addr,
		remote.addr + CTRL_AREA, remote.key, c);
	return 0;
}

static int post_send(const void *src, struct fi_context2 *c)
{
	FT_POST(fi_send, ft_progress, txcq, tx_seq, &tx_cq_cntr, "fi_send",
		ep, src, CTRL_SIZE, mr_desc, remote_fi_addr, c);
	return 0;
}

static int post_recv(void *dst, size_t len, struct fi_context2 *c)
{
	FT_POST(fi_recv, ft_progress, rxcq, rx_seq, &rx_cq_cntr, "fi_recv",
		ep, dst, len, mr_desc, FI_ADDR_UNSPEC, c);
	return 0;
}

static char *ctrl_a(size_t n)
{
	return tx_buf + n + SMALL_SIZE;
}

static char *ctrl_b(size_t n)
{
	return tx_buf + n + 2 * SMALL_SIZE;
}

/* Layout of tx_buf for a large operation of n bytes. The offsets are the
 * ones used by the initiate functions and by ctrl_a() and ctrl_b():
 *
 * offset 0                 n bytes          pattern_big
 * offset n                 MIN(n,64) bytes  pattern_small
 *                                           (rest of the 64-byte slot: unused)
 * offset n + 64            8 bytes          'A'
 * offset n + 64 + 8        56 bytes         gap, never set by init_source
 * offset n + 128           8 bytes          'B'
 * total copied: n + 128 + 8
 */
static int init_source(size_t n)
{
	size_t i, small = MIN(n, SMALL_SIZE);

	for (i = 0; i < n; i++)
		stage[i] = pattern_big(i);
	for (i = 0; i < small; i++)
		stage[n + i] = pattern_small(i);
	memset(stage + n + SMALL_SIZE, 'A', CTRL_SIZE);
	memset(stage + n + 2 * SMALL_SIZE, 'B', CTRL_SIZE);

	return ft_hmem_copy_to(opts.iface, opts.device, tx_buf, stage,
			       n + 2 * SMALL_SIZE + CTRL_SIZE);
}

static int wait_message(void)
{
	return ft_get_rx_comp(rx_seq);
}

static int init_waw(size_t n)
{
	int ret;

	ret = post_write(tx_buf, n, &ctx[0]);
	if (ret)
		return ret;

	/* the second write is ordered after the first, so it must win */
	ret = post_write(tx_buf + n, MIN(n, SMALL_SIZE), &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int verify_waw_target(size_t n)
{
	return check_bytes("WAW target", region(), n, MIN(n, SMALL_SIZE),
			   pattern_small, pattern_big);
}

static int init_raw(size_t n)
{
	int ret;

	ret = post_write(tx_buf, n, &ctx[0]);
	if (ret)
		return ret;

	ret = post_read(region(), n, &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int verify_raw_initiator(size_t n)
{
	return check_bytes("RAW read", region(), n, 0, NULL, pattern_big);
}

static int init_war(size_t n)
{
	int ret;

	ret = post_read(region(), n, &ctx[0]);
	if (ret)
		return ret;

	/* the read is ordered before the write, so it must see the old data */
	ret = post_write(tx_buf, n, &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int verify_war_initiator(size_t n)
{
	return check_bytes("WAR read", region(), n, 0, NULL, pattern_old);
}

static int verify_big_target(size_t n)
{
	return check_bytes("target", region(), n, 0, NULL, pattern_big);
}

static int init_saw(size_t n)
{
	int ret;

	ret = post_write(tx_buf, n, &ctx[0]);
	if (ret)
		return ret;

	ret = post_send(ctrl_a(n), &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int react_saw(size_t n)
{
	int ret;

	ret = wait_message();
	if (ret)
		return ret;

	/* seeing the send means the write before it has landed */
	return check_bytes("SAW target", region(), n, 0, NULL, pattern_big);
}

static int init_sar(size_t n)
{
	int ret;

	ret = post_read(region(), n, &ctx[0]);
	if (ret)
		return ret;

	ret = post_send(ctrl_a(n), &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int react_sar(size_t n)
{
	int ret;

	ret = wait_message();
	if (ret)
		return ret;

	/* the read before the send is done, so changing the data must not
	 * change what it returned
	 */
	return put_pattern(region(), n, pattern_big);
}

static int prepare_sas(size_t n)
{
	int ret;

	(void) n;

	ret = put_fill(rx_buf, CTRL_SIZE, SENTINEL);
	if (ret)
		return ret;

	ret = put_fill(rx_buf + SECOND_RECV, CTRL_SIZE, SENTINEL);
	if (ret)
		return ret;

	return post_recv(rx_buf + SECOND_RECV, CTRL_SIZE, &ctx[2]);
}

static int init_sas(size_t n)
{
	int ret;

	ret = post_send(ctrl_a(n), &ctx[0]);
	if (ret)
		return ret;

	ret = post_send(ctrl_b(n), &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int react_sas(size_t n)
{
	(void) n;
	return wait_message();
}

static int verify_sas_target(size_t n)
{
	int ret;

	(void) n;

	ret = ft_hmem_copy_from(opts.iface, opts.device, stage, rx_buf,
				CTRL_SIZE);
	if (ret)
		return ret;

	ret = ft_hmem_copy_from(opts.iface, opts.device, stage + CTRL_SIZE,
				rx_buf + SECOND_RECV, CTRL_SIZE);
	if (ret)
		return ret;

	if (stage[0] != 'A' || stage[CTRL_SIZE] != 'B') {
		FT_ERR("SAS target: the receives got 0x%02x then 0x%02x, "
		       "expected 'A' then 'B'", (unsigned char) stage[0],
		       (unsigned char) stage[CTRL_SIZE]);
		check_failed = true;
	}

	return 0;
}

/* BEST EFFORT. Two reads of a range that the target keeps changing must not
 * return the changes out of order. The target counts up while the reads are
 * in flight, so a second read that returns a smaller count than the first
 * proves a misordering. The reads can still be ordered wrongly without the
 * count changing between them, which this cannot see.
 */
static int prepare_rar(size_t n)
{
	(void) n;
	return put_fill(region(), sizeof(uint64_t), 0);
}

static int init_rar(size_t n)
{
	uint64_t first, second;
	int i, ret;

	rar_violations = 0;
	for (i = 0; i < RAR_READS; i++) {
		ret = post_read(region(), sizeof(uint64_t), &ctx[0]);
		if (ret)
			return ret;

		ret = post_read(region() + sizeof(uint64_t),
				sizeof(uint64_t), &ctx[1]);
		if (ret)
			return ret;

		ret = ft_get_tx_comp(tx_seq);
		if (ret)
			return ret;

		ret = ft_hmem_copy_from(opts.iface, opts.device, stage,
					region(), 2 * sizeof(uint64_t));
		if (ret)
			return ret;

		memcpy(&first, stage, sizeof(first));
		memcpy(&second, stage + sizeof(first), sizeof(second));
		if (second < first)
			rar_violations++;
	}

	ret = post_send(ctrl_a(n), &ctx[0]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int react_rar(size_t n)
{
	uint64_t counter = 0;
	int ret;

	(void) n;

	do {
		counter++;
		memcpy(stage, &counter, sizeof(counter));
		ret = ft_hmem_copy_to(opts.iface, opts.device, region(), stage,
				      sizeof(counter));
		if (ret)
			return ret;

		ret = ft_progress(rxcq, rx_seq, &rx_cq_cntr);
		if (ret)
			return ret;
	} while (rx_cq_cntr < rx_seq);

	return 0;
}

static int verify_rar_initiator(size_t n)
{
	(void) n;
	if (rar_violations) {
		FT_ERR("RAR: %u of %d read pairs returned the second read "
		       "older than the first", rar_violations, RAR_READS);
		check_failed = true;
	}

	return 0;
}

/* BEST EFFORT. The target reacts to the send, then the read that follows it
 * may run before or after that reaction, so the data it returns is either the
 * old or the new data. Only a byte that is neither, or a read that never
 * completes, shows a problem. An ordering bug that picks the wrong one of the
 * two valid answers cannot be seen.
 */
static int init_ras(size_t n)
{
	int ret;

	ret = post_send(ctrl_a(n), &ctx[0]);
	if (ret)
		return ret;

	ret = post_read(region(), n, &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int react_ras(size_t n)
{
	int ret;

	ret = wait_message();
	if (ret)
		return ret;

	return put_pattern(region(), n, pattern_big);
}

static int verify_ras_initiator(size_t n)
{
	return check_either("RAS read (best effort)", region(), n, pattern_old,
			    pattern_big);
}

/* BEST EFFORT. The write is ordered after the send, but the target can look
 * at the data before or after it lands, so it may see the old or the new
 * data. The target only checks that every byte is one of the two, and that
 * the write has landed once both operations are done. A write that lands
 * early cannot be told from one that lands on time.
 */
static int init_was(size_t n)
{
	int ret;

	ret = post_send(ctrl_a(n), &ctx[0]);
	if (ret)
		return ret;

	ret = post_write(tx_buf, n, &ctx[1]);
	if (ret)
		return ret;

	return ft_get_tx_comp(tx_seq);
}

static int react_was(size_t n)
{
	int ret;

	ret = wait_message();
	if (ret)
		return ret;

	return check_either("WAS target (best effort)", region(), n,
			    pattern_old, pattern_big);
}

static const struct order_case cases[] = {
	{
		.name = "WAW",
		.order = FI_ORDER_WAW,
		.best_effort = false,
		.per_size = true,
		.reads = false,
		.target_state = TARGET_SENTINEL,
		.recvs = 0,
		.prepare = NULL,
		.initiate = init_waw,
		.react = NULL,
		.verify_target = verify_waw_target,
		.verify_initiator = NULL,
	},
	{
		.name = "RAW",
		.order = FI_ORDER_RAW,
		.best_effort = false,
		.per_size = true,
		.reads = true,
		.target_state = TARGET_SENTINEL,
		.recvs = 0,
		.prepare = NULL,
		.initiate = init_raw,
		.react = NULL,
		.verify_target = NULL,
		.verify_initiator = verify_raw_initiator,
	},
	{
		.name = "WAR",
		.order = FI_ORDER_WAR,
		.best_effort = false,
		.per_size = true,
		.reads = true,
		.target_state = TARGET_OLD,
		.recvs = 0,
		.prepare = NULL,
		.initiate = init_war,
		.react = NULL,
		.verify_target = verify_big_target,
		.verify_initiator = verify_war_initiator,
	},
	{
		.name = "SAW",
		.order = FI_ORDER_SAW,
		.best_effort = false,
		.per_size = true,
		.reads = false,
		.target_state = TARGET_SENTINEL,
		.recvs = 1,
		.prepare = NULL,
		.initiate = init_saw,
		.react = react_saw,
		.verify_target = NULL,
		.verify_initiator = NULL,
	},
	{
		.name = "SAR",
		.order = FI_ORDER_SAR,
		.best_effort = false,
		.per_size = true,
		.reads = true,
		.target_state = TARGET_OLD,
		.recvs = 1,
		.prepare = NULL,
		.initiate = init_sar,
		.react = react_sar,
		.verify_target = NULL,
		.verify_initiator = verify_war_initiator,
	},
	{
		.name = "SAS",
		.order = FI_ORDER_SAS,
		.best_effort = false,
		.per_size = false,
		.reads = false,
		.target_state = TARGET_SENTINEL,
		.recvs = 2,
		.prepare = prepare_sas,
		.initiate = init_sas,
		.react = react_sas,
		.verify_target = verify_sas_target,
		.verify_initiator = NULL,
	},
	{
		.name = "RAR",
		.order = FI_ORDER_RAR,
		.best_effort = true,
		.per_size = false,
		.reads = false,
		.target_state = TARGET_SENTINEL,
		.recvs = 1,
		.prepare = prepare_rar,
		.initiate = init_rar,
		.react = react_rar,
		.verify_target = NULL,
		.verify_initiator = verify_rar_initiator,
	},
	{
		.name = "RAS",
		.order = FI_ORDER_RAS,
		.best_effort = true,
		.per_size = true,
		.reads = true,
		.target_state = TARGET_OLD,
		.recvs = 1,
		.prepare = NULL,
		.initiate = init_ras,
		.react = react_ras,
		.verify_target = NULL,
		.verify_initiator = verify_ras_initiator,
	},
	{
		.name = "WAS",
		.order = FI_ORDER_WAS,
		.best_effort = true,
		.per_size = true,
		.reads = false,
		.target_state = TARGET_OLD,
		.recvs = 1,
		.prepare = NULL,
		.initiate = init_was,
		.react = react_was,
		.verify_target = verify_big_target,
		.verify_initiator = NULL,
	},
};

static int wait_peer(void)
{
	time_t end = time(NULL) + timeout;
	int ret;

	while ((ret = ft_poll_fd(oob_sock, 0)) == -FI_EAGAIN) {
		ret = ft_progress(rxcq, rx_seq, &rx_cq_cntr);
		if (ret)
			return ret;

		if (timeout >= 0 && time(NULL) > end) {
			FT_ERR("%ds timeout expired waiting for the peer",
			       timeout);
			return -FI_ETIMEDOUT;
		}
	}

	return ret;
}

static int sync_progress(void)
{
	char byte = 'a';
	int ret;

	if (opts.dst_addr) {
		ret = ft_sock_send(oob_sock, &byte, 1);
		if (ret)
			return ret;
	}

	ret = wait_peer();
	if (ret)
		return ret;

	ret = ft_sock_recv(oob_sock, &byte, 1);
	if (ret)
		return ret;

	return opts.dst_addr ? 0 : ft_sock_send(oob_sock, &byte, 1);
}

static int exchange_status(bool *failed)
{
	int32_t mine = check_failed, peer;
	int ret;

	if (opts.dst_addr) {
		ret = ft_sock_send(oob_sock, &mine, sizeof(mine));
		if (ret)
			return ret;

		ret = ft_sock_recv(oob_sock, &peer, sizeof(peer));
	} else {
		ret = ft_sock_recv(oob_sock, &peer, sizeof(peer));
		if (ret)
			return ret;

		ret = ft_sock_send(oob_sock, &mine, sizeof(mine));
	}

	*failed = mine || peer;
	check_failed = false;
	return ret;
}

static int prepare_target(const struct order_case *c, size_t n)
{
	int ret;

	if (c->target_state == TARGET_OLD)
		ret = put_pattern(region(), n, pattern_old);
	else
		ret = put_fill(region(), n, SENTINEL);
	if (ret)
		return ret;

	return c->prepare ? c->prepare(n) : 0;
}

static int run_iteration(const struct order_case *c, size_t n, bool *failed)
{
	bool target = !opts.dst_addr;
	int ret;

	if (target)
		ret = prepare_target(c, n);
	else
		ret = c->reads ? put_fill(region(), n, SENTINEL) : 0;
	if (ret)
		return ret;

	ret = ft_sync();
	if (ret)
		return ret;

	if (target)
		ret = c->react ? c->react(n) : 0;
	else
		ret = c->initiate(n);
	if (ret)
		return ret;

	if (target && c->recvs) {
		ret = post_recv(rx_buf, rx_size, &rx_ctx);
		if (ret)
			return ret;
	}

	ret = sync_progress();
	if (ret)
		return ret;

	if (target)
		ret = c->verify_target ? c->verify_target(n) : 0;
	else
		ret = c->verify_initiator ? c->verify_initiator(n) : 0;
	if (ret)
		return ret;

	return exchange_status(failed);
}

static void show_case(const struct order_case *c)
{
	char label[LABEL_WIDTH + 1];
	size_t i, len;

	snprintf(label, sizeof(label), "%s%s", c->name,
		 c->best_effort ? " (best effort)" : "");
	len = strlen(label);
	printf("%s ", label);
	for (i = len + 1; i < LABEL_WIDTH; i++)
		putchar('-');

	printf("> ");
	fflush(stdout);
}

static void show_result(bool failed)
{
	printf("%s", failed ? "FAIL" : "PASS");
	if (failed && failed_size)
		printf(" (%zu bytes)", failed_size);

	printf("\n");
	fflush(stdout);
}

static int run_size(const struct order_case *c, size_t n, bool *failed)
{
	int i, ret;

	if (opts.dst_addr) {
		ret = init_source(n);
		if (ret)
			return ret;
	}

	for (i = 0; i < opts.iterations && !*failed; i++) {
		ret = run_iteration(c, n, failed);
		if (ret)
			return ret;
	}

	if (*failed && c->per_size)
		failed_size = n;

	return 0;
}

static int run_case(const struct order_case *c)
{
	bool failed = false, ran = false;
	unsigned int i;
	int ret = 0;

	failed_size = 0;
	show_case(c);
	if (single_size) {
		ret = run_size(c, single_value, &failed);
	} else {
		for (i = 0; i < test_cnt && !failed && !ret; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;

			ret = run_size(c, test_size[i].size, &failed);
			ran = true;
			if (!c->per_size)
				break;
		}

		if (!ran && !ret) {
			FT_ERR("no message size can be tested");
			ret = -FI_EINVAL;
		}
	}

	show_result(failed || ret);
	return ret ? ret : (failed ? -FI_EIO : 0);
}

static struct fi_info *probe_hints(void)
{
	struct fi_info *probe;

	probe = fi_dupinfo(hints);
	if (!probe)
		return NULL;

	if (!probe->ep_attr->type)
		probe->ep_attr->type = FI_EP_RDM;

	if (opts.options & FT_OPT_ENABLE_HMEM) {
		probe->caps |= FI_HMEM;
		probe->domain_attr->mr_mode |= FI_MR_HMEM;
	}

	return probe;
}

static int pin_domain(void)
{
	struct fi_info *probe, *info;
	int ret;

	if (hints->domain_attr->name)
		return 0;

	probe = probe_hints();
	if (!probe)
		return -FI_ENOMEM;

	ret = fi_getinfo(ft_fiversion, NULL, NULL, 0, probe, &info);
	fi_freeinfo(probe);
	if (ret)
		return ret;

	hints->domain_attr->name = strdup(info->domain_attr->name);
	fi_freeinfo(info);
	if (!hints->domain_attr->name)
		return -FI_ENOMEM;

	FT_DEBUG("no domain was given with -d, using %s",
		 hints->domain_attr->name);
	return 0;
}

static bool single_env_provider(void)
{
	const char *env = getenv("FI_PROVIDER");

	return env && *env && *env != '^' && !strchr(env, ',');
}

static int probe_orders(uint64_t *mask, uint64_t *orders)
{
	struct fi_info *probe, *info;
	size_t i;
	int ret;

	*mask = 0;
	*orders = 0;
	if (!hints->fabric_attr->prov_name && !single_env_provider()) {
		FT_ERR("a single provider must be given with -p or "
		       "FI_PROVIDER");
		return -FI_EINVAL;
	}

	ret = pin_domain();
	if (ret)
		return ret;

	for (i = 0; i < ARRAY_SIZE(cases); i++) {
		if (!(selected & (1ULL << i)))
			continue;

		probe = probe_hints();
		if (!probe)
			return -FI_ENOMEM;

		probe->tx_attr->msg_order = cases[i].order;
		probe->rx_attr->msg_order = cases[i].order;
		ret = fi_getinfo(ft_fiversion, NULL, NULL, 0, probe, &info);
		fi_freeinfo(probe);
		if (ret) {
			FT_DEBUG("the provider does not support the %s "
				 "ordering, skipping it", cases[i].name);
			continue;
		}

		fi_freeinfo(info);
		*mask |= 1ULL << i;
		*orders |= cases[i].order;
	}

	return 0;
}

static bool size_enabled(unsigned int i)
{
	return opts.sizes_enabled == FT_ENABLE_SIZES ||
	       (opts.sizes_enabled & test_size[i].enable_flags);
}

static void size_stats(struct run_msg *info)
{
	unsigned int i;

	if (single_size) {
		info->nsizes = 1;
		info->size_sum = single_value;
		info->size_max = single_value;
		return;
	}

	for (i = 0; i < test_cnt; i++) {
		if (!size_enabled(i))
			continue;

		info->nsizes++;
		info->size_sum += test_size[i].size;
		info->size_max = MAX(info->size_max, test_size[i].size);
	}
}

static int exchange_run(uint64_t mask)
{
	struct run_msg msg = { 0 };
	int32_t status = 0;
	int ret;

	if (opts.dst_addr) {
		msg = size_info;
		msg.cases = mask;
		msg.iterations = opts.iterations;

		ret = ft_sock_send(oob_sock, &msg, sizeof(msg));
		if (ret)
			return ret;

		ret = ft_sock_recv(oob_sock, &status, sizeof(status));
		if (ret)
			return ret;

		if (status) {
			FT_ERR("the server rejected the options of this "
			       "client");
			return -FI_EINVAL;
		}
	} else {
		ret = ft_sock_recv(oob_sock, &msg, sizeof(msg));
		if (ret)
			return ret;

		if (msg.cases != mask) {
			FT_ERR("the server and the client would run different "
			       "cases: -y differs, or the provider supports "
			       "different orderings on each side");
			status = -FI_EINVAL;
		}

		if (msg.iterations != (uint64_t) opts.iterations) {
			FT_ERR("-I differs: %d on the server, %" PRIu64 " on "
			       "the client", opts.iterations, msg.iterations);
			status = -FI_EINVAL;
		}

		if (msg.nsizes != size_info.nsizes ||
		    msg.size_sum != size_info.size_sum ||
		    msg.size_max != size_info.size_max) {
			FT_ERR("-S differs between the server and the client");
			status = -FI_EINVAL;
		}

		ret = ft_sock_send(oob_sock, &status, sizeof(status));
		if (ret || status)
			return ret ? ret : status;
	}

	return 0;
}

static int run(void)
{
	uint64_t mask, orders;
	size_t i;
	int ret;

	ret = probe_orders(&mask, &orders);
	if (ret)
		return ret;

	if (!mask) {
		FT_ERR("no provider supports any of the orderings to test with "
		       "these options");
		return -FI_ENODATA;
	}

	hints->tx_attr->msg_order = orders;
	hints->rx_attr->msg_order = orders;
	ret = hints->ep_attr->type == FI_EP_MSG ?
	      ft_init_fabric_cm() : ft_init_fabric();
	if (ret)
		return ret;

	ret = ft_exchange_keys(&remote);
	if (ret)
		return ret;

	ret = exchange_run(mask);
	if (ret)
		return ret;

	for (i = 0; i < ARRAY_SIZE(cases); i++) {
		if (!(mask & (1ULL << i)))
			continue;

		ret = run_case(&cases[i]);
		if (ret)
			return ret;
	}

	return ft_sync();
}

static int parse_cases(char *list)
{
	char *name, *save = NULL;
	size_t i;

	selected = 0;
	for (name = strtok_r(list, ",", &save); name;
	     name = strtok_r(NULL, ",", &save)) {
		for (i = 0; i < ARRAY_SIZE(cases); i++) {
			if (!strcasecmp(name, cases[i].name))
				break;
		}

		if (i == ARRAY_SIZE(cases)) {
			FT_ERR("unknown case '%s'", name);
			return -FI_EINVAL;
		}

		selected |= 1ULL << i;
	}

	return selected ? 0 : -FI_EINVAL;
}

static int size_buffers(void)
{
	single_size = opts.options & FT_OPT_SIZE;
	single_value = opts.transfer_size;
	if (single_size && !single_value) {
		FT_ERR("a size of 0 cannot be tested");
		return -FI_EINVAL;
	}

	size_stats(&size_info);
	if (!size_info.size_max) {
		FT_ERR("no message sizes are enabled");
		return -FI_EINVAL;
	}

	opts.transfer_size = size_info.size_max + EXTRA_SIZE;
	/* make shared allocate our new buffer size */
	opts.options |= FT_OPT_SIZE;
	return 0;
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.iterations = 100;
	opts.options |= FT_OPT_OOB_SYNC | FT_OPT_OOB_ADDR_EXCH;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "hy:" CS_OPTS INFO_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case 'y':
			if (parse_cases(optarg)) {
				ret = EXIT_FAILURE;
				goto out;
			}
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Check the RMA and message "
				   "orderings the provider supports, one at "
				   "a time. The initiator and the target can "
				   "each use host or device memory. RAR, RAS "
				   "and WAS cannot be seen from the data "
				   "alone, so they are best effort.");
			FT_PRINT_OPTS_USAGE("-y <cases>", "Comma separated list of the cases to run: waw, raw, war, saw, sar, sas, rar, ras, was (default: all the provider supports). Give the same list to both sides");
			FT_PRINT_OPTS_USAGE("-S <size>", "Size of the large operation, a range (syntax r:start,inc,end), a list (syntax l:1,2,3,...) or 'all' (default: the usual sweep of sizes)");
			ft_longopts_usage();
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

	if (size_buffers()) {
		ret = EXIT_FAILURE;
		goto out;
	}

	stage = malloc(opts.transfer_size);
	if (!stage) {
		ret = EXIT_FAILURE;
		goto out;
	}

	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->addr_format = opts.address_format;
	hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;

	ret = run();

out:
	free(stage);
	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
