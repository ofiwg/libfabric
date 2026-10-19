/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * fi_efa_av_insert_race: race fi_av_insert against the TX and CQ read paths.
 *
 * Both processes run --threads worker threads. Every worker owns one endpoint
 * and one CQ; all endpoints of a process share one FI_AV_TABLE AV. Worker i
 * talks to the remote workers N(i) = { (i + d) mod T : d in D }, with
 * D = {0, +1, -1, +2, -2, ...} truncated to --peers elements, and receives from
 * every remote worker j with i in N(j).
 *
 * No address is inserted up front. A worker inserts a remote endpoint either
 * when the first message from it arrives -- so the insert promotes the
 * endpoint's implicit AV entry while other workers' endpoints are still
 * receiving from it -- or when a random per-peer delay expires -- so some
 * endpoints are inserted before they have sent anything. It starts sending to
 * a remote endpoint as soon as its own insert returns, so lazy peer creation on
 * the TX path races with promotions done by other threads. Together this drives
 * the CQ read path through both the implicit and the explicit AV, the TX path,
 * and implicit-to-explicit promotion, all concurrently on one AV.
 *
 * Checks:
 *  - every worker receives exactly --msgs messages from each remote worker that
 *    targets it, each sequence number exactly once, and in order unless
 *    --no-order-check is given (FI_ORDER_SAS orders matching, so it only
 *    implies completion order for single-packet messages);
 *  - every send completes, and no CQ or EQ error is reported;
 *  - all workers inserting the same remote address get the same fi_addr, and
 *    the T remote addresses end up as fi_addrs 0..T-1 (FI_AV_TABLE);
 *  - fi_av_lookup of each fi_addr returns the address that was inserted;
 *  - a completion's source address, when it is not FI_ADDR_NOTAVAIL, is the
 *    fi_addr of the sending endpoint.
 */

#include <getopt.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include "shared.h"

#define MAX_ADDR_LEN	64
#define MSG_MAGIC	0xa71a5aceu
#define CQ_BATCH	16

struct msg_hdr {
	uint32_t magic;
	uint16_t side;
	uint16_t worker;
	uint32_t seq;
	uint32_t round;
};

struct peer_state {
	int remote;			/* remote worker index */
	bool tx;			/* we send to it */
	bool rx;			/* it sends to us */
	fi_addr_t fi_addr;		/* FI_ADDR_NOTAVAIL until this worker inserted it */
	uint64_t insert_at_ns;		/* insert by this time at the latest */
	uint32_t next_send_seq;
	uint32_t next_recv_seq;		/* number received so far */
	uint64_t *recv_seen;		/* bitmap of received seqs (--no-order-check) */
};

struct worker {
	int id;
	pthread_t thread;
	struct fid_ep *ep;
	struct fid_cq *cq;
	struct fid_eq *eq;
	struct fid_mr *mr;
	void *desc;
	char *buf;			/* rx_depth receive buffers, then tx_depth send buffers */
	struct fi_context2 *rx_ctx;
	struct fi_context2 *tx_ctx;
	int *tx_free;
	int tx_nfree;
	struct peer_state *peers;
	int npeers;
	int *peer_of_remote;		/* remote worker -> index in peers, or -1 */
	int tx_rr;			/* round-robin cursor over peers */
	uint64_t total_to_send, total_to_recv;
	uint64_t sent, send_completed, received;
	uint64_t src_checked, src_unknown, src_notavail;
	uint64_t inserts_on_recv, inserts_on_timer;
	unsigned int seed;
	int ret;
};

static int nthreads = 64;
static int npeers = 5;
static int nmsgs = 200;
static int rx_depth = 64;
static int tx_depth = 32;
static int max_delay_us = 20000;
static int nrounds = 1;
static int cur_round;
static bool check_order = true;
static unsigned int random_seed;

static int my_side;			/* 0 = server, 1 = client */
static size_t msg_size;
static struct fid_av *shared_av;
static struct worker *workers;
static char (*local_addrs)[MAX_ADDR_LEN];
static char (*remote_addrs)[MAX_ADDR_LEN];
/* first fi_addr returned by fi_av_insert for each remote worker's address */
static fi_addr_t *remote_fiaddr_tbl;

enum {
	OPT_THREADS = 256,
	OPT_PEERS,
	OPT_MSGS,
	OPT_RX_DEPTH,
	OPT_TX_DEPTH,
	OPT_MAX_DELAY_US,
	OPT_NO_ORDER_CHECK,
	OPT_SEED,
	OPT_ROUNDS,
};

static struct option test_long_opts[] = {
	{"threads", required_argument, NULL, OPT_THREADS},
	{"peers", required_argument, NULL, OPT_PEERS},
	{"msgs", required_argument, NULL, OPT_MSGS},
	{"rx-depth", required_argument, NULL, OPT_RX_DEPTH},
	{"tx-depth", required_argument, NULL, OPT_TX_DEPTH},
	{"max-delay-us", required_argument, NULL, OPT_MAX_DELAY_US},
	{"no-order-check", no_argument, NULL, OPT_NO_ORDER_CHECK},
	{"seed", required_argument, NULL, OPT_SEED},
	{"rounds", required_argument, NULL, OPT_ROUNDS},
	{"threading", required_argument, NULL, LONG_OPT_THREADING},
	{"timeout", required_argument, NULL, LONG_OPT_TIMEOUT},
	{0, 0, 0, 0}
};

static uint64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* D = {0, +1, -1, +2, -2, ...}: the k-th offset */
static int peer_offset(int k)
{
	return (k & 1) ? (k + 1) / 2 : -(k / 2);
}

/* Fill targets with N(i); returns the number of distinct remote workers. */
static int targets_of(int i, int *targets)
{
	int k, n = 0, j, m;
	bool dup;

	for (k = 0; n < npeers && k < 2 * nthreads; k++) {
		j = ((i + peer_offset(k)) % nthreads + nthreads) % nthreads;
		dup = false;
		for (m = 0; m < n; m++)
			dup |= targets[m] == j;
		if (!dup)
			targets[n++] = j;
	}
	return n;
}

static struct peer_state *peer_of(struct worker *w, int remote)
{
	int idx = w->peer_of_remote[remote];

	return idx < 0 ? NULL : &w->peers[idx];
}

static struct peer_state *add_peer(struct worker *w, int remote)
{
	struct peer_state *p = peer_of(w, remote);

	if (p)
		return p;
	p = &w->peers[w->npeers];
	memset(p, 0, sizeof(*p));
	p->remote = remote;
	p->fi_addr = FI_ADDR_NOTAVAIL;
	w->peer_of_remote[remote] = w->npeers++;
	return p;
}

static int init_peers(struct worker *w)
{
	int *targets;
	int n, i, j;
	struct peer_state *p;

	targets = calloc(nthreads, sizeof(*targets));
	w->peers = calloc(2 * nthreads, sizeof(*w->peers));
	w->peer_of_remote = malloc(nthreads * sizeof(*w->peer_of_remote));
	if (!targets || !w->peers || !w->peer_of_remote) {
		free(targets);
		return -FI_ENOMEM;
	}
	for (j = 0; j < nthreads; j++)
		w->peer_of_remote[j] = -1;

	n = targets_of(w->id, targets);
	for (i = 0; i < n; i++) {
		p = add_peer(w, targets[i]);
		p->tx = true;
		w->total_to_send += nmsgs;
	}

	/* The remote side uses the same function, so remote worker j sends to
	 * us iff we are in N(j). */
	for (j = 0; j < nthreads; j++) {
		n = targets_of(j, targets);
		for (i = 0; i < n; i++) {
			if (targets[i] != w->id)
				continue;
			p = add_peer(w, j);
			p->rx = true;
			w->total_to_recv += nmsgs;
		}
	}
	free(targets);

	for (i = 0; i < w->npeers; i++) {
		w->peers[i].recv_seen = calloc((nmsgs + 63) / 64, sizeof(uint64_t));
		if (!w->peers[i].recv_seen)
			return -FI_ENOMEM;
	}
	return 0;
}

/* Forget everything about the previous round; the AV is empty again. */
static void reset_worker(struct worker *w)
{
	struct peer_state *p;
	int i;

	for (i = 0; i < w->npeers; i++) {
		p = &w->peers[i];
		p->fi_addr = FI_ADDR_NOTAVAIL;
		p->next_send_seq = 0;
		p->next_recv_seq = 0;
		memset(p->recv_seen, 0, ((nmsgs + 63) / 64) * sizeof(uint64_t));
	}
	w->sent = w->send_completed = w->received = 0;
	w->src_checked = w->src_unknown = w->src_notavail = 0;
	w->inserts_on_recv = w->inserts_on_timer = 0;
	w->tx_rr = 0;
	w->ret = 0;
}

static int insert_remote(struct worker *w, struct peer_state *p)
{
	fi_addr_t fi_addr = FI_ADDR_NOTAVAIL, expected = FI_ADDR_NOTAVAIL;
	int ret;

	ret = fi_av_insert(shared_av, remote_addrs[p->remote], 1, &fi_addr, 0, NULL);
	if (ret != 1) {
		fprintf(stderr, "worker %d: fi_av_insert of remote %d returned %d\n",
			w->id, p->remote, ret);
		return ret < 0 ? ret : -FI_EOTHER;
	}

	/* Every insert of the same address into the shared AV must return the
	 * same fi_addr, whichever thread does it and whatever the AV had for it
	 * before (nothing, or an implicit entry). */
	if (!__atomic_compare_exchange_n(&remote_fiaddr_tbl[p->remote], &expected,
					 fi_addr, false, __ATOMIC_ACQ_REL,
					 __ATOMIC_ACQUIRE) &&
	    expected != fi_addr) {
		fprintf(stderr, "worker %d: fi_av_insert of remote %d returned "
			"fi_addr %" PRIu64 ", another worker got %" PRIu64 "\n",
			w->id, p->remote, fi_addr, expected);
		return -FI_EOTHER;
	}
	p->fi_addr = fi_addr;
	return 0;
}

static int post_recv(struct worker *w, int idx)
{
	int ret;

	do {
		ret = fi_recv(w->ep, w->buf + (size_t) idx * msg_size, msg_size,
			      w->desc, FI_ADDR_UNSPEC, &w->rx_ctx[idx]);
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(w->cq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret)
		fprintf(stderr, "worker %d: fi_recv failed: %s\n", w->id,
			fi_strerror(-ret));
	return ret;
}

static int handle_recv(struct worker *w, struct fi_cq_data_entry *comp,
		       fi_addr_t src)
{
	int idx = (struct fi_context2 *) comp->op_context - w->rx_ctx;
	struct msg_hdr *hdr;
	struct peer_state *p;
	fi_addr_t known;
	int ret;

	if (idx < 0 || idx >= rx_depth) {
		fprintf(stderr, "worker %d: receive completion with a bad context\n",
			w->id);
		return -FI_EOTHER;
	}
	hdr = (struct msg_hdr *) (w->buf + (size_t) idx * msg_size);

	if (comp->len != msg_size || hdr->magic != MSG_MAGIC ||
	    hdr->side != 1 - my_side || hdr->worker >= nthreads ||
	    hdr->round != (uint32_t) cur_round) {
		fprintf(stderr, "worker %d: corrupt message: len %zu magic %#x "
			"side %u worker %u round %u (current %d)\n", w->id,
			comp->len, hdr->magic, hdr->side, hdr->worker,
			hdr->round, cur_round);
		return -FI_EOTHER;
	}

	p = peer_of(w, hdr->worker);
	if (!p || !p->rx) {
		fprintf(stderr, "worker %d: unexpected message from remote %u\n",
			w->id, hdr->worker);
		return -FI_EOTHER;
	}

	if (hdr->seq >= (uint32_t) nmsgs ||
	    (check_order && hdr->seq != p->next_recv_seq)) {
		fprintf(stderr, "worker %d: message %u from remote %d, expected "
			"%u\n", w->id, hdr->seq, p->remote, p->next_recv_seq);
		return -FI_EOTHER;
	}
	if (!check_order) {
		/* Multi-packet messages can complete out of order; still
		 * require each one exactly once. */
		uint64_t bit = 1ull << (hdr->seq % 64);

		if (p->recv_seen[hdr->seq / 64] & bit) {
			fprintf(stderr, "worker %d: duplicate message %u from "
				"remote %d\n", w->id, hdr->seq, p->remote);
			return -FI_EOTHER;
		}
		p->recv_seen[hdr->seq / 64] |= bit;
	}
	p->next_recv_seq++;
	w->received++;

	if (src == FI_ADDR_NOTAVAIL) {
		w->src_notavail++;
	} else {
		known = __atomic_load_n(&remote_fiaddr_tbl[hdr->worker],
					__ATOMIC_ACQUIRE);
		if (known == FI_ADDR_NOTAVAIL) {
			w->src_unknown++;
		} else if (known != src) {
			fprintf(stderr, "worker %d: message from remote %u "
				"reported source fi_addr %" PRIu64 ", the remote "
				"was inserted as %" PRIu64 "\n", w->id,
				hdr->worker, src, known);
			return -FI_EOTHER;
		} else {
			w->src_checked++;
		}
	}

	/* First contact: insert the sender, promoting its implicit AV entry. */
	if (p->fi_addr == FI_ADDR_NOTAVAIL) {
		ret = insert_remote(w, p);
		if (ret)
			return ret;
		w->inserts_on_recv++;
	}

	return post_recv(w, idx);
}

static int handle_send(struct worker *w, struct fi_cq_data_entry *comp)
{
	int idx = (struct fi_context2 *) comp->op_context - w->tx_ctx;

	if (idx < 0 || idx >= tx_depth) {
		fprintf(stderr, "worker %d: send completion with a bad context\n",
			w->id);
		return -FI_EOTHER;
	}
	w->tx_free[w->tx_nfree++] = idx;
	w->send_completed++;
	return 0;
}

/* Returns the number of completions handled, or a negative error. */
static int progress(struct worker *w)
{
	struct fi_cq_data_entry comp[CQ_BATCH];
	fi_addr_t src[CQ_BATCH];
	struct fi_cq_err_entry err = {0};
	ssize_t n;
	int i, ret;

	n = fi_cq_readfrom(w->cq, comp, CQ_BATCH, src);
	if (n == -FI_EAGAIN)
		return 0;
	if (n == -FI_EAVAIL) {
		fi_cq_readerr(w->cq, &err, 0);
		fprintf(stderr, "worker %d: CQ error: %s (%s)\n", w->id,
			fi_strerror(err.err),
			fi_cq_strerror(w->cq, err.prov_errno, err.err_data,
				       NULL, 0));
		return -err.err ? -err.err : -FI_EOTHER;
	}
	if (n < 0) {
		fprintf(stderr, "worker %d: fi_cq_readfrom: %s\n", w->id,
			fi_strerror((int) -n));
		return (int) n;
	}

	for (i = 0; i < n; i++) {
		if (comp[i].flags & FI_RECV)
			ret = handle_recv(w, &comp[i], src[i]);
		else if (comp[i].flags & FI_SEND)
			ret = handle_send(w, &comp[i]);
		else
			ret = -FI_EOTHER;
		if (ret)
			return ret;
	}
	return (int) n;
}

static int check_eq(struct worker *w)
{
	struct fi_eq_err_entry err = {0};
	uint32_t event;
	int ret;

	ret = fi_eq_read(w->eq, &event, NULL, 0, 0);
	if (ret != -FI_EAVAIL)
		return 0;
	ret = fi_eq_readerr(w->eq, &err, 0);
	if (ret < 0)
		return 0;
	fprintf(stderr, "worker %d: EQ error: %s (prov_errno %d)\n", w->id,
		fi_strerror(err.err), err.prov_errno);
	return -err.err ? -err.err : -FI_EOTHER;
}

/* Post at most one send; -FI_EAGAIN if nothing could be posted. */
static int send_one(struct worker *w)
{
	struct peer_state *p;
	struct msg_hdr *hdr;
	int i, slot, ret;

	if (!w->tx_nfree)
		return -FI_EAGAIN;

	for (i = 0; i < w->npeers; i++) {
		p = &w->peers[(w->tx_rr + i) % w->npeers];
		if (!p->tx || p->fi_addr == FI_ADDR_NOTAVAIL ||
		    p->next_send_seq >= (uint32_t) nmsgs)
			continue;

		slot = w->tx_free[w->tx_nfree - 1];
		hdr = (struct msg_hdr *) (w->buf +
			(size_t) (rx_depth + slot) * msg_size);
		hdr->magic = MSG_MAGIC;
		hdr->side = my_side;
		hdr->worker = w->id;
		hdr->seq = p->next_send_seq;
		hdr->round = cur_round;

		ret = fi_send(w->ep, hdr, msg_size, w->desc, p->fi_addr,
			      &w->tx_ctx[slot]);
		if (ret == -FI_EAGAIN)
			return ret;
		if (ret) {
			fprintf(stderr, "worker %d: fi_send to remote %d failed: "
				"%s\n", w->id, p->remote, fi_strerror(-ret));
			return ret;
		}
		w->tx_nfree--;
		p->next_send_seq++;
		w->sent++;
		w->tx_rr = (w->tx_rr + i + 1) % w->npeers;
		return 0;
	}
	return -FI_EAGAIN;
}

static void dump_state(struct worker *w)
{
	int i;

	fprintf(stderr, "worker %d: sent %" PRIu64 "/%" PRIu64 " completed %"
		PRIu64 " received %" PRIu64 "/%" PRIu64 "\n", w->id, w->sent,
		w->total_to_send, w->send_completed, w->received,
		w->total_to_recv);
	for (i = 0; i < w->npeers; i++)
		fprintf(stderr, "  remote %d tx %d rx %d fi_addr %" PRIu64
			" sent %u received %u\n", w->peers[i].remote,
			w->peers[i].tx, w->peers[i].rx, w->peers[i].fi_addr,
			w->peers[i].next_send_seq, w->peers[i].next_recv_seq);
}

static void *run_worker(void *arg)
{
	struct worker *w = arg;
	uint64_t now, start, last_progress, iter = 0;
	struct peer_state *p;
	int i, n, ret = 0;
	bool progressed;

	start = last_progress = now_ns();
	for (i = 0; i < w->npeers; i++) {
		w->peers[i].insert_at_ns = start + (max_delay_us ?
			(uint64_t) (rand_r(&w->seed) % max_delay_us) * 1000 : 0);
	}

	while (w->received < w->total_to_recv ||
	       w->send_completed < w->total_to_send) {
		progressed = false;

		n = progress(w);
		if (n < 0) {
			ret = n;
			break;
		}
		progressed |= n > 0;

		now = now_ns();
		for (i = 0; i < w->npeers; i++) {
			p = &w->peers[i];
			if (!p->tx || p->fi_addr != FI_ADDR_NOTAVAIL ||
			    now < p->insert_at_ns)
				continue;
			ret = insert_remote(w, p);
			if (ret)
				goto out;
			w->inserts_on_timer++;
			progressed = true;
		}

		while ((ret = send_one(w)) == 0)
			progressed = true;
		if (ret != -FI_EAGAIN)
			break;
		ret = 0;

		if (!(++iter & 0xff)) {
			ret = check_eq(w);
			if (ret)
				break;
		}

		if (progressed) {
			last_progress = now;
		} else if (now - last_progress > (uint64_t) timeout * 1000000000ull) {
			fprintf(stderr, "worker %d: no progress for %d seconds\n",
				w->id, timeout);
			ret = -FI_ETIMEDOUT;
			break;
		}
	}
out:
	if (ret)
		dump_state(w);
	w->ret = ret;
	return NULL;
}

static int setup_worker(struct worker *w)
{
	struct fi_cq_attr cq_attr = {
		.format = FI_CQ_FORMAT_DATA,
		.wait_obj = FI_WAIT_NONE,
		.size = 4 * (rx_depth + tx_depth),
	};
	size_t addrlen = MAX_ADDR_LEN;
	size_t buf_len;
	int i, ret;

	ret = init_peers(w);
	if (ret)
		return ret;

	ret = fi_endpoint(domain, fi, &w->ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}
	ret = fi_cq_open(domain, &cq_attr, &w->cq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		return ret;
	}
	ret = fi_eq_open(fabric, &eq_attr, &w->eq, NULL);
	if (ret) {
		FT_PRINTERR("fi_eq_open", ret);
		return ret;
	}
	ret = fi_ep_bind(w->ep, &w->cq->fid, FI_SEND | FI_RECV);
	if (!ret)
		ret = fi_ep_bind(w->ep, &shared_av->fid, 0);
	if (!ret)
		ret = fi_ep_bind(w->ep, &w->eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}
	ret = fi_enable(w->ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		return ret;
	}

	memset(local_addrs[w->id], 0, MAX_ADDR_LEN);
	ret = fi_getname(&w->ep->fid, local_addrs[w->id], &addrlen);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
		return ret;
	}

	buf_len = (size_t) (rx_depth + tx_depth) * msg_size;
	w->buf = calloc(1, buf_len);
	w->rx_ctx = calloc(rx_depth, sizeof(*w->rx_ctx));
	w->tx_ctx = calloc(tx_depth, sizeof(*w->tx_ctx));
	w->tx_free = calloc(tx_depth, sizeof(*w->tx_free));
	if (!w->buf || !w->rx_ctx || !w->tx_ctx || !w->tx_free)
		return -FI_ENOMEM;
	for (i = 0; i < tx_depth; i++)
		w->tx_free[i] = i;
	w->tx_nfree = tx_depth;

	if (fi->domain_attr->mr_mode & FI_MR_LOCAL) {
		ret = fi_mr_reg(domain, w->buf, buf_len, FI_SEND | FI_RECV, 0,
				0, 0, &w->mr, NULL);
		if (ret) {
			FT_PRINTERR("fi_mr_reg", ret);
			return ret;
		}
		w->desc = fi_mr_desc(w->mr);
	}

	for (i = 0; i < rx_depth; i++) {
		ret = post_recv(w, i);
		if (ret)
			return ret;
	}
	return 0;
}

static void cleanup_worker(struct worker *w)
{
	if (w->ep)
		fi_close(&w->ep->fid);
	if (w->mr)
		fi_close(&w->mr->fid);
	if (w->eq)
		fi_close(&w->eq->fid);
	if (w->cq)
		fi_close(&w->cq->fid);
	free(w->buf);
	free(w->rx_ctx);
	free(w->tx_ctx);
	free(w->tx_free);
	for (int i = 0; w->peers && i < w->npeers; i++)
		free(w->peers[i].recv_seen);
	free(w->peers);
	free(w->peer_of_remote);
}

/* The T remote addresses must have been numbered 0..T-1 (FI_AV_TABLE), and
 * fi_av_lookup must map each number back to the address inserted. */
static int check_av(void)
{
	char addr[MAX_ADDR_LEN];
	size_t addrlen;
	bool *seen;
	int j, ret = 0;

	seen = calloc(nthreads, sizeof(*seen));
	if (!seen)
		return -FI_ENOMEM;

	for (j = 0; j < nthreads && !ret; j++) {
		fi_addr_t fa = remote_fiaddr_tbl[j];

		if (fa == FI_ADDR_NOTAVAIL)
			continue;
		if (fa >= (fi_addr_t) nthreads || seen[fa]) {
			fprintf(stderr, "remote %d has fi_addr %" PRIu64 ": "
				"FI_AV_TABLE numbering broken\n", j, fa);
			ret = -FI_EOTHER;
			break;
		}
		seen[fa] = true;

		addrlen = sizeof(addr);
		memset(addr, 0, sizeof(addr));
		ret = fi_av_lookup(shared_av, fa, addr, &addrlen);
		if (ret) {
			FT_PRINTERR("fi_av_lookup", ret);
			break;
		}
		if (addrlen > MAX_ADDR_LEN ||
		    memcmp(addr, remote_addrs[j], addrlen)) {
			fprintf(stderr, "fi_av_lookup(%" PRIu64 ") does not "
				"return the address of remote %d\n", fa, j);
			ret = -FI_EOTHER;
		}
	}
	free(seen);
	return ret;
}

static uint64_t tot_on_recv, tot_on_timer, tot_checked, tot_notavail;

/* Remove every remote address, so the next round starts with an empty AV. */
static int remove_all(void)
{
	int j, ret;

	for (j = 0; j < nthreads; j++) {
		if (remote_fiaddr_tbl[j] == FI_ADDR_NOTAVAIL)
			continue;
		ret = fi_av_remove(shared_av, &remote_fiaddr_tbl[j], 1, 0);
		if (ret) {
			FT_PRINTERR("fi_av_remove", ret);
			return ret;
		}
		remote_fiaddr_tbl[j] = FI_ADDR_NOTAVAIL;
	}
	return 0;
}

static int run_round(void)
{
	uint64_t sent = 0, received = 0, checked = 0, unknown = 0, notavail = 0;
	uint64_t on_recv = 0, on_timer = 0, start, elapsed;
	int i, j, ret;

	for (i = 0; i < nthreads; i++)
		reset_worker(&workers[i]);

	/* Both sides have quiesced and emptied their AV. */
	ret = ft_sync_oob();
	if (ret)
		return ret;

	start = now_ns();
	for (i = 0; i < nthreads; i++) {
		ret = pthread_create(&workers[i].thread, NULL, run_worker,
				     &workers[i]);
		if (ret) {
			fprintf(stderr, "pthread_create: %s\n", strerror(ret));
			for (j = 0; j < i; j++)
				pthread_join(workers[j].thread, NULL);
			return -ret;
		}
	}
	for (i = 0; i < nthreads; i++)
		pthread_join(workers[i].thread, NULL);
	elapsed = now_ns() - start;

	for (i = 0; i < nthreads; i++) {
		struct worker *w = &workers[i];

		if (w->ret && !ret)
			ret = w->ret;
		sent += w->sent;
		received += w->received;
		checked += w->src_checked;
		unknown += w->src_unknown;
		notavail += w->src_notavail;
		on_recv += w->inserts_on_recv;
		on_timer += w->inserts_on_timer;
	}
	tot_on_recv += on_recv;
	tot_on_timer += on_timer;
	tot_checked += checked;
	tot_notavail += notavail;

	printf("round %d: sent %" PRIu64 " received %" PRIu64 " in %.3f s; "
	       "inserts %" PRIu64 " on first receive, %" PRIu64 " on timer; "
	       "source address %" PRIu64 " verified, %" PRIu64 " not yet known, %"
	       PRIu64 " FI_ADDR_NOTAVAIL (implicit)\n", cur_round, sent,
	       received, elapsed / 1e9, on_recv, on_timer, checked, unknown,
	       notavail);

	if (!ret)
		ret = check_av();

	/* Do not remove addresses, or close endpoints, the remote side may
	 * still be sending to. */
	i = ft_sync_oob();
	if (!ret)
		ret = i;
	if (!ret && cur_round + 1 < nrounds)
		ret = remove_all();
	return ret;
}

static int run(void)
{
	struct fi_av_attr av_attr = {
		.type = FI_AV_TABLE,
		.count = nthreads,
	};
	int remote_nthreads, i, j, ret;

	/* Both sides must run the same number of workers. */
	ret = ft_sock_send(oob_sock, &nthreads, sizeof(nthreads));
	if (!ret)
		ret = ft_sock_recv(oob_sock, &remote_nthreads, sizeof(remote_nthreads));
	if (ret)
		return ret;
	if (remote_nthreads != nthreads) {
		fprintf(stderr, "local --threads %d, remote --threads %d\n",
			nthreads, remote_nthreads);
		return -FI_EINVAL;
	}

	ret = fi_av_open(domain, &av_attr, &shared_av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		return ret;
	}

	workers = calloc(nthreads, sizeof(*workers));
	local_addrs = calloc(nthreads, MAX_ADDR_LEN);
	remote_addrs = calloc(nthreads, MAX_ADDR_LEN);
	remote_fiaddr_tbl = malloc(nthreads * sizeof(*remote_fiaddr_tbl));
	if (!workers || !local_addrs || !remote_addrs || !remote_fiaddr_tbl) {
		ret = -FI_ENOMEM;
		goto out;
	}
	for (j = 0; j < nthreads; j++)
		remote_fiaddr_tbl[j] = FI_ADDR_NOTAVAIL;

	for (i = 0; i < nthreads; i++) {
		workers[i].id = i;
		workers[i].seed = random_seed * 1000003u + my_side * 7919u + i;
		ret = setup_worker(&workers[i]);
		if (ret)
			goto out;
	}

	ret = ft_sock_send(oob_sock, local_addrs, (size_t) nthreads * MAX_ADDR_LEN);
	if (!ret)
		ret = ft_sock_recv(oob_sock, remote_addrs,
				   (size_t) nthreads * MAX_ADDR_LEN);
	if (ret)
		goto out;

	printf("%d workers, %d peers each, %d messages of %zu bytes per peer, "
	       "insert delay up to %d us, %d rounds, seed %u\n", nthreads,
	       npeers, nmsgs, msg_size, max_delay_us, nrounds, random_seed);

	for (cur_round = 0; cur_round < nrounds; cur_round++) {
		ret = run_round();
		if (ret)
			goto out;
	}
	printf("totals over %d rounds: inserts %" PRIu64 " on first receive, %"
	       PRIu64 " on timer; source address %" PRIu64 " verified, %"
	       PRIu64 " FI_ADDR_NOTAVAIL (implicit)\n", nrounds, tot_on_recv,
	       tot_on_timer, tot_checked, tot_notavail);

out:
	if (workers) {
		for (i = 0; i < nthreads; i++)
			cleanup_worker(&workers[i]);
	}
	if (shared_av)
		fi_close(&shared_av->fid);
	free(workers);
	free(local_addrs);
	free(remote_addrs);
	free(remote_fiaddr_tbl);
	return ret;
}

static void print_test_usage(void)
{
	FT_PRINT_OPTS_USAGE("--threads <N>",
			    "worker threads (endpoints) per process (default: 64)");
	FT_PRINT_OPTS_USAGE("--peers <N>",
			    "remote endpoints each worker sends to (default: 5)");
	FT_PRINT_OPTS_USAGE("--msgs <N>",
			    "messages per (worker, remote endpoint) pair (default: 200)");
	FT_PRINT_OPTS_USAGE("--rx-depth <N>", "posted receives per endpoint (default: 64)");
	FT_PRINT_OPTS_USAGE("--tx-depth <N>", "outstanding sends per endpoint (default: 32)");
	FT_PRINT_OPTS_USAGE("--max-delay-us <N>",
			    "insert a target after at most this long even if it "
			    "has not sent anything yet (default: 20000)");
	FT_PRINT_OPTS_USAGE("--no-order-check",
			    "only check that every message arrives exactly once");
	FT_PRINT_OPTS_USAGE("--rounds <N>",
			    "repeat, removing every remote address from the AV "
			    "between rounds so the next round starts from "
			    "implicit entries again (default: 1)");
	FT_PRINT_OPTS_USAGE("--seed <N>", "random seed (default: time)");
	FT_PRINT_OPTS_USAGE("--threading <model>", "safe|completion (default: safe)");
	FT_PRINT_OPTS_USAGE("--timeout <s>", "fail after this long without progress (default: 60)");
}

static int parse_opts(int argc, char **argv)
{
	int op;

	while ((op = getopt_long(argc, argv, "h" ADDR_OPTS INFO_OPTS CS_OPTS,
				 test_long_opts, NULL)) != -1) {
		switch (op) {
		case OPT_THREADS:
			nthreads = atoi(optarg);
			break;
		case OPT_PEERS:
			npeers = atoi(optarg);
			break;
		case OPT_MSGS:
			nmsgs = atoi(optarg);
			break;
		case OPT_RX_DEPTH:
			rx_depth = atoi(optarg);
			break;
		case OPT_TX_DEPTH:
			tx_depth = atoi(optarg);
			break;
		case OPT_MAX_DELAY_US:
			max_delay_us = atoi(optarg);
			break;
		case OPT_NO_ORDER_CHECK:
			check_order = false;
			break;
		case OPT_SEED:
			random_seed = (unsigned int) strtoul(optarg, NULL, 0);
			break;
		case OPT_ROUNDS:
			nrounds = atoi(optarg);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "Race fi_av_insert against the TX and CQ read paths");
			print_test_usage();
			return -2;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		}
	}

	if (nthreads < 1 || npeers < 1 || nmsgs < 1 || rx_depth < 1 ||
	    tx_depth < 1 || max_delay_us < 0 || nthreads > UINT16_MAX ||
	    nrounds < 1) {
		fprintf(stderr, "invalid arguments\n");
		return -1;
	}
	if (npeers > nthreads)
		npeers = nthreads;
	return 0;
}

int main(int argc, char **argv)
{
	int ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;
	opts.threading = FI_THREAD_SAFE;
	opts.transfer_size = sizeof(struct msg_hdr);
	timeout = 60;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	ret = parse_opts(argc, argv);
	if (ret)
		goto out;

	if (optind < argc)
		opts.dst_addr = argv[optind];
	my_side = opts.dst_addr ? 1 : 0;
	if (!random_seed)
		random_seed = (unsigned int) time(NULL);
	msg_size = MAX(opts.transfer_size, sizeof(struct msg_hdr));

	hints->caps = FI_MSG | FI_SOURCE;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_LOCAL |
				      FI_MR_VIRT_ADDR | FI_MR_PROV_KEY;
	if (check_order) {
		hints->tx_attr->msg_order = FI_ORDER_SAS;
		hints->rx_attr->msg_order = FI_ORDER_SAS;
	}

	ret = ft_init_oob();
	if (ret) {
		FT_PRINTERR("ft_init_oob", ret);
		goto out;
	}
	ret = ft_sync_oob();
	if (ret)
		goto out;

	ret = ft_getinfo(hints, &fi);
	if (ret)
		goto out;
	ret = ft_open_fabric_res();
	if (ret)
		goto out;

	ret = run();
out:
	ft_close_oob();
	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
