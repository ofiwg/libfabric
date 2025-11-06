/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/* Ethernet UDP packet generator for Cassini */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stddef.h>
#include <arpa/inet.h>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <locale.h>

#include "libcxi.h"

#define EQ_BUF_SIZE (4U * 1024 * 1024)
#define PAGE_SIZE 4096U
#define ETH_HEADER_SIZE (sizeof(struct ether_header))
#define IP_HEADER_SIZE (sizeof(struct ip))
#define UDP_HEADER_SIZE (sizeof(struct udphdr))

enum checksum_type {
	CSUM_NORMAL, /* normal UDP checksum */
	CSUM_NONE, /* no UDP checksum, ie. 0 */
	CSUM_BAD, /* bad UDP checksum */
	CSUM_HW, /* Cassini will compute the checksum */
};
struct skb {
	struct ether_header ethhdr;
	struct ip iphdr;
	struct udphdr udphdr;
	uint8_t payload[];
} __attribute__((packed));

/* Shared resources across all threads. */
static struct cxil_dev *dev;
static struct cxil_lni *lni;
static struct cxi_cp *cp;

/* Per thread resources. */
static struct cxi_cq **txqs;
static void **eq_bufs;
static struct cxi_md **mds;
static struct cxi_eq **eqs;
static struct skb **skbs;
static struct cxi_md **skb_mds;
static pthread_t *txq_threads;
static pthread_t *eq_threads;
static atomic_uint *tx_credits;
static atomic_ulong bytes_sent;

/* User defined variables. */
#define IP_SIZE 17
#define MAC_SIZE 19
static char src_ip[IP_SIZE];
static char dst_ip[IP_SIZE];
static char src_mac[MAC_SIZE];
static char dst_mac[MAC_SIZE];
static size_t packet_size = 64;
static size_t queue_depth = 4096;
static size_t num_threads = 1;
static size_t batch_submit = 1;
static size_t warmup_seconds = 5;
static size_t runtime_seconds = 5;
static int csum_type = CSUM_NORMAL;

/* Thread control variable to sync starting. */
static volatile bool threads_start;

static void set_ethernet_header(struct skb *skb)
{
	struct ether_header *eth = &skb->ethhdr;
	struct ether_addr *addr;

	addr = ether_aton(src_mac);
	assert(addr);
	memcpy(eth->ether_shost, addr, ETH_ALEN);

	addr = ether_aton(dst_mac);
	assert(addr);
	memcpy(eth->ether_dhost, addr, ETH_ALEN);

	eth->ether_type = htons(ETHERTYPE_IP);
}

/* Compute checksum on regular payload */
static uint16_t checksum_buffer(const void *data, size_t payload_len,
				uint32_t sum)
{
	const unsigned short *payload = data;

	/* Add the payload datagram */
	while (payload_len > 1) {
		sum += *payload;
		payload++;
		payload_len -= 2;
	}

	/* If odd length, pad */
	if (payload_len)
		sum += *payload & htons(0xff00);

	/* Take care of carries */
	while (sum >> 16)
		sum = (sum & 0xffff) + (sum >> 16);

	return sum;
}

static void set_ipv4_header(struct skb *skb, unsigned int payload_size)
{
	struct ip *ip = &skb->iphdr;
	uint16_t checksum = 0;
	int rc;

	memset(ip, 0, sizeof(*ip));
	ip->ip_hl = 5;
	ip->ip_v = 4;
	ip->ip_len = htons(payload_size);
	ip->ip_p = IPPROTO_UDP;

	rc = inet_pton(AF_INET, src_ip, &ip->ip_src.s_addr);
	assert(rc == 1);

	rc = inet_pton(AF_INET, dst_ip, &ip->ip_dst.s_addr);
	assert(rc == 1);

	checksum = checksum_buffer(ip, IP_HEADER_SIZE, 0);

	ip->ip_sum = ~checksum & 0xFFFF;
}

/* Compute checksum on pseudo IPV4 header */
static uint16_t header_checksum(const struct skb *skb, size_t ip_payload_len)
{
	const struct ip *iphdr = &skb->iphdr;
	uint32_t sum;

	/* Add the pseudo header */
	sum = (iphdr->ip_src.s_addr >> 16) & 0xffff;
	sum += (iphdr->ip_src.s_addr) & 0xffff;
	sum += (iphdr->ip_dst.s_addr >> 16) & 0xffff;
	sum += (iphdr->ip_dst.s_addr) & 0xffff;
	sum += ntohs(iphdr->ip_p); // swap 8 bits value to put in right place
	sum += ntohs(ip_payload_len);

	/* Take care of carries */
	while (sum >> 16)
		sum = (sum & 0xffff) + (sum >> 16);

	return sum;
}

static void set_udp_header(struct skb *skb, unsigned int src_port,
			   size_t udp_payload_len)
{
	struct udphdr *udp = &skb->udphdr;
	uint32_t sum;

	udp->uh_sport = htons(src_port);
	udp->uh_dport = htons(55555);
	udp->uh_ulen = htons(udp_payload_len);
	udp->uh_sum = 0;

	sum = header_checksum(skb, udp_payload_len);

	switch (csum_type) {
	case CSUM_BAD:
		sum = ~checksum_buffer((unsigned short *)&skb->udphdr,
				       udp_payload_len, sum);
		sum = (sum + 1) & 0xffff;
		if (sum == 0)
			sum = 1;
		break;

	case CSUM_NONE:
		sum = 0;
		break;

	case CSUM_HW:
		/* Just the header checksum. HW will do the rest for
		 * every packet.
		 */
		break;

	case CSUM_NORMAL:
		sum = ~checksum_buffer((unsigned short *)&skb->udphdr,
				       udp_payload_len, sum);
		sum = sum ?: 0xffff;
	}

	udp->uh_sum = sum;
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

static void *txq_thread(void *context)
{
	unsigned long id = (unsigned long)context;
	struct cxi_md *skb_md = skb_mds[id];
	void *skb = skbs[id];
	struct cxi_cq *txq = txqs[id];
	struct cxi_eq *eq = eqs[id];
	struct c_dma_eth_cmd eth_cmd = {
		.read_lac = skb_md->lac,
		.fmt = C_PKT_FORMAT_STD,
		.flow_hash = id,
		.checksum_ctrl = C_CHECKSUM_CTRL_NONE,
		.eq = eq->eqn,
		.total_len = packet_size,
		.user_ptr = packet_size,
	};
	struct c_idc_eth_cmd idc_eth_cmd = {
		.fmt = C_PKT_FORMAT_STD,
		.flow_hash = id,
	};
	int i;
	int rc;
	unsigned int submitted = 0;

	if (csum_type == CSUM_HW) {
		eth_cmd.checksum_ctrl = C_CHECKSUM_CTRL_UDP;
		eth_cmd.checksum_start = (ETH_HEADER_SIZE + IP_HEADER_SIZE) / 2;
		eth_cmd.checksum_offset = (offsetof(struct udphdr, check)) / 2;
	}

	for (i = 0; (i * PAGE_SIZE) < packet_size; i++) {
		eth_cmd.addr[i] = CXI_VA_TO_IOVA(skb_md, skb) + (i * PAGE_SIZE);
		eth_cmd.len[i] = MIN(PAGE_SIZE, packet_size - (i * PAGE_SIZE));
	}

	eth_cmd.num_segments = i;

	printf("TXQ %ld thread waiting.....\n", id);
	printf("TXQ %ld TX credit count: %u\n", id,
	       atomic_load(&tx_credits[id]));

	/* Spin until told to go. */
	while (!threads_start)
		sched_yield();

	printf("TXQ %ld thread started!\n", id);

	/* Run forever until the app is killed. */
	while (1) {
		if (atomic_load(&tx_credits[id]) > 0) {
			if (packet_size > C_MAX_IDC_PAYLOAD_RES)
				rc = cxi_cq_emit_dma_eth(txq, &eth_cmd);
			else
				rc = cxi_cq_emit_idc_eth(txq, &idc_eth_cmd, skb,
							 packet_size);

			if (!rc) {
				atomic_fetch_sub(&tx_credits[id], 1);
				submitted++;
				if (submitted >= batch_submit) {
					cxi_cq_ring(txq);
					submitted = 0;
				}
			}
		}
	}

	return NULL;
}

static void *eq_thread(void *context)
{
	unsigned long id = (unsigned long)context;
	struct cxi_eq *eq = eqs[id];
	const union c_event *event;
	int event_count = 0;

	printf("EQ %ld thread waiting.....\n", id);

	/* Spin until told to go. */
	while (!threads_start)
		sched_yield();

	printf("EQ %ld thread started!\n", id);

	/* Run forever until the app is killed. */
	while (1) {
		while ((event = cxi_eq_get_event(eq))) {
			assert(event->hdr.event_size == C_EVENT_SIZE_16_BYTE);
			assert(event->hdr.event_type == C_EVENT_SEND);
			assert(event->init_short.return_code == C_RC_OK);

			/* Transfer size is stored in user pointer. */
			atomic_fetch_add(&bytes_sent, packet_size);

			/* Return credits to the matching TXQ thread. */
			assert(atomic_fetch_add(&tx_credits[id], 1) <=
			       queue_depth);

			/* Ack the EQ every 16 events. */
			event_count++;
			if (event_count > 16)
				break;
		}

		event_count = 0;
		cxi_eq_ack_events(eq);
	}
}

static void test_setup(unsigned int num_threads)
{
	long i;
	int j;
	int rc;
	struct cxi_cq_alloc_opts cq_opts = {
		.count = CXI_MAX_CQ_COUNT,
		.policy = CXI_CQ_UPDATE_LOW_FREQ_EMPTY,
		.flags = CXI_CQ_IS_TX | CXI_CQ_TX_ETHERNET,
	};
	struct cxi_eq_attr eq_attrs = {
		.queue_len = EQ_BUF_SIZE,
	};
	struct c_cstate_cmd c_state = {
		.restricted = 1,
		.event_success_disable = 1,
	};

	rc = cxil_open_device(0, &dev);
	assert(rc == 0);

	rc = cxil_alloc_lni(dev, &lni, CXI_DEFAULT_SVC_ID);
	assert(rc == 0);

	rc = cxil_alloc_cp(lni, 0, CXI_TC_ETH, CXI_TC_TYPE_DEFAULT, &cp);
	assert(rc == 0);

	eq_bufs = calloc(num_threads, sizeof(*eq_bufs));
	assert(eq_bufs);

	mds = calloc(num_threads, sizeof(*mds));
	assert(mds);

	eqs = calloc(num_threads, sizeof(*eqs));
	assert(eqs);

	txqs = calloc(num_threads, sizeof(*txqs));
	assert(txqs);

	skbs = calloc(num_threads, sizeof(*skbs));
	assert(skbs);

	skb_mds = calloc(num_threads, sizeof(*skb_mds));
	assert(skb_mds);

	txq_threads = calloc(num_threads, sizeof(*txq_threads));
	assert(txq_threads);

	eq_threads = calloc(num_threads, sizeof(*eq_threads));
	assert(eq_threads);

	tx_credits = calloc(num_threads, sizeof(*tx_credits));
	assert(tx_credits);

	cq_opts.lcid = cp->lcid;
	for (i = 0; i < num_threads; i++) {
		eq_bufs[i] = aligned_alloc(sysconf(_SC_PAGE_SIZE), EQ_BUF_SIZE);
		assert(eq_bufs[i]);

		rc = cxil_map(lni, eq_bufs[i], EQ_BUF_SIZE,
			      CXI_MAP_PIN | CXI_MAP_WRITE, NULL, &mds[i]);
		assert(rc == 0);

		eq_attrs.queue = eq_bufs[i];
		rc = cxil_alloc_evtq(lni, mds[i], &eq_attrs, NULL, NULL,
				     &eqs[i]);
		assert(rc == 0);

		rc = cxil_alloc_cmdq(lni, eqs[i], &cq_opts, &txqs[i]);
		assert(rc == 0);

		skbs[i] = aligned_alloc(packet_size, PAGE_SIZE);
		assert(skbs[i]);

		rc = cxil_map(lni, skbs[i], packet_size,
			      CXI_MAP_PIN | CXI_MAP_READ, NULL, &skb_mds[i]);
		assert(rc == 0);

		set_ethernet_header(skbs[i]);
		set_ipv4_header(skbs[i], packet_size - ETH_HEADER_SIZE);
		set_udp_header(skbs[i], 60000 + i,
			       packet_size - ETH_HEADER_SIZE - IP_HEADER_SIZE);

		atomic_store(&tx_credits[i], queue_depth);

		c_state.eq = eqs[i]->eqn;
		/* Issue 8 commands to align CQ on 256 byte boundary. */
		for (j = 0; j < 8; j++) {
			rc = cxi_cq_emit_c_state(txqs[i], &c_state);
			assert(rc == 0);
		}
		cxi_cq_ring(txqs[i]);

		rc = pthread_create(&txq_threads[i], NULL, &txq_thread,
				    (void *)i);
		assert(rc == 0);

		rc = pthread_create(&eq_threads[i], NULL, &eq_thread,
				    (void *)i);
		assert(rc == 0);
	}

	atomic_store(&bytes_sent, 0);
}

static double timespec_diff(struct timespec *end, struct timespec *start)
{
	return (end->tv_sec - start->tv_sec) +
	       1.0e-9 * (end->tv_nsec - start->tv_nsec);
}

static void print_help(void)
{
	printf("cxi_udp_gen [OPTION]...\n\n");
	printf("Options:\n");
	printf("\t-s <src_ipv4>: Source IPv4 address used in the packet\n");
	printf("\t-d <dest_ipv4>: Destination IPv4 address used in the packet\n");
	printf("\t-n <src_mac>: Source MAC address used in the packet\n");
	printf("\t-m <dest_mac>: Destination MAC address used in the packet\n");
	printf("\t-c <checksum type>: 0=none, 1=regular, 2=bad, 3=hardware\n");
	printf("\t-b <packet_size>: Packet size in bytes. Must be >= 64.\n");
	printf("\t-t <num_of_threas>: Number of threads to issue and process commands\n");
	printf("\t-q <queue_depth>: TX queue depth per thread\n");
	printf("\t-z <batch_submit>: Number of commands to be batch together per CQ ring\n");
	printf("\t-w <warmup_secs>: Test warmup period\n");
	printf("\t-r <runtime_secs>: Runtime in seconds\n");
}

static void parse_args(int argc, char *argv[])
{
	int c;

	while ((c = getopt(argc, argv, "c:s:d:n:m:b:t:q:z:w:r:")) != -1) {
		switch (c) {
		/* Source IP address. */
		case 's':
			snprintf(src_ip, IP_SIZE - 1, "%s", optarg);
			break;

		/* Destination IP address. */
		case 'd':
			snprintf(dst_ip, IP_SIZE - 1, "%s", optarg);
			break;

		/* Source MAC address. */
		case 'n':
			snprintf(src_mac, MAC_SIZE - 1, "%s", optarg);
			break;

		/* Destination MAC address. */
		case 'm':
			snprintf(dst_mac, MAC_SIZE - 1, "%s", optarg);
			break;

		case 'c':
			csum_type = atoi(optarg);
			break;

		/* UDP payload size in bytes. */
		case 'b':
			packet_size = (size_t)atol(optarg);
			break;

		/* Number of threads. */
		case 't':
			num_threads = (size_t)atol(optarg);
			break;

		/* Queue depth */
		case 'q':
			queue_depth = (size_t)atol(optarg);
			break;

		case 'z':
			batch_submit = (size_t)atol(optarg);
			break;

		case 'w':
			warmup_seconds = (size_t)atol(optarg);
			break;

		case 'r':
			runtime_seconds = (size_t)atol(optarg);
			break;

		default:
			print_help();
			exit(1);
		}
	}

	assert(num_threads > 0);
	assert(packet_size >= 60);
	assert(csum_type >= 0 && csum_type <= 3);
}

#define __ALIGN_KERNEL_MASK(x, mask) (((x) + (mask)) & ~(mask))
#define __ALIGN_KERNEL(x, a) __ALIGN_KERNEL_MASK(x, (typeof(x))(a)-1)
#define ALIGN(x, a) __ALIGN_KERNEL((x), (a))

int main(int argc, char *argv[])
{
	int rc;
	struct timespec cur_time;
	struct timespec measure_time;
	double time_diff;
	unsigned long start_bytes_sent;
	unsigned long end_bytes_sent;

	parse_args(argc, argv);
	queue_depth = ALIGN(queue_depth, batch_submit);

	printf("Running Ethernet test with following arguments\n");
	printf("Source IP address: %s\n", src_ip);
	printf("Destination IP address: %s\n", dst_ip);
	printf("Source MAC address: %s\n", src_mac);
	printf("Destination MAC address: %s\n", dst_mac);
	printf("Checksum type: %d\n", csum_type);
	printf("Number of threads: %lu\n", num_threads);
	printf("TX queue depth per thread: %lu\n", queue_depth);
	printf("Packet size: %lu\n", packet_size);
	printf("Batch submit: %lu\n", batch_submit);
	printf("Warmup seconds: %lu\n", warmup_seconds);
	printf("Test seconds: %lu\n", runtime_seconds);

	test_setup(num_threads);

	/* Kick off all the threads. */
	threads_start = true;
	__sync_synchronize();

	/* Warmup period. */
	rc = clock_gettime(CLOCK_MONOTONIC, &measure_time);
	assert(rc == 0);
	do {
		rc = clock_gettime(CLOCK_MONOTONIC, &cur_time);
		assert(rc == 0);

		time_diff = timespec_diff(&cur_time, &measure_time);
	} while (time_diff < warmup_seconds);

	/* Run test. */
	rc = clock_gettime(CLOCK_MONOTONIC, &measure_time);
	assert(rc == 0);
	start_bytes_sent = atomic_load(&bytes_sent);
	do {
		rc = clock_gettime(CLOCK_MONOTONIC, &cur_time);
		assert(rc == 0);
		time_diff = timespec_diff(&cur_time, &measure_time);
	} while (time_diff < runtime_seconds);
	end_bytes_sent = atomic_load(&bytes_sent);

	printf("Bytes sent including warmup: %lu\n", end_bytes_sent);
	printf("Packets sent including warmup: %lu\n",
	       end_bytes_sent / packet_size);

	/* For the thousand separator */
	setlocale(LC_NUMERIC, "");

	end_bytes_sent -= start_bytes_sent;
	printf("Overall Bandwidth: %'lu bytes/s\n",
	       (uint64_t)(end_bytes_sent / time_diff));

	return 0;
}
