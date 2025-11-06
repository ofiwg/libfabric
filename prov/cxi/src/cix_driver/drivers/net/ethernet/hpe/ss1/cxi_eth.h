/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Cassini ethernet driver
 * Copyright 2018-2020,2022 Hewlett Packard Enterprise Development LP
 */

#ifndef __CXI_ETH_H__
#define __CXI_ETH_H__
#include <linux/hpe/cxi/cxi.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 16, 0)
#define HAVE_KERNEL_RINGPARAM 1
#elif defined(RHEL_MAJOR) && ((RHEL_MAJOR == 8 && RHEL_MINOR >= 7) || (RHEL_MAJOR == 9 && RHEL_MINOR >= 3) )
#define HAVE_KERNEL_RINGPARAM 1
#elif defined(CONFIG_SUSE_KERNEL) && CONFIG_SUSE_VERSION == 15 && \
		CONFIG_SUSE_PATCHLEVEL >= 5
#define HAVE_KERNEL_RINGPARAM 1
#endif

#ifdef HAVE_KERNEL_RINGPARAM
#include <linux/ethtool.h>
#endif

#include "cxi_ethtool.h"
#include "cassini_user_defs.h"
#include "cxi_core.h"

/* netif_napi_add defined in RHEL 8.8 */
#if defined(netif_napi_add) || (LINUX_VERSION_CODE >= KERNEL_VERSION(5, 19, 0)) || defined(RHEL9_3_PLUS)
#define NETIF_NAPI_ADD netif_napi_add
#define NETIF_NAPI_ADD_TX netif_napi_add_tx
#else
#define NETIF_NAPI_ADD(ndev, napi, poll) netif_napi_add(ndev, napi, poll, NAPI_POLL_WEIGHT)
#define NETIF_NAPI_ADD_TX(ndev, napi, poll) netif_tx_napi_add(ndev, napi, poll, NAPI_POLL_WEIGHT)
#endif /* netif_napi_add */

#define CXI_ETH_1_MB (1024 * 1024)
#define CXI_ETH_1_GB (CXI_ETH_1_MB * 1024)

#define SMALL_PKTS_BUF_COUNT_MIN 1U
#define SMALL_PKTS_BUF_COUNT_MAX (16 * 1024U)

#define LARGE_PKTS_BUF_COUNT_MIN 1U
#define LARGE_PKTS_BUF_COUNT_MAX 10000U

#define TX_QUEUE_LEN_DEFAULT 1000

#define MAX_LE_LIMIT 16384

/* Align max TX queues to max RX queues. */
#define CXI_ETH_MAX_TX_QUEUES CXI_ETH_MAX_RSS_QUEUES

/* 4.12 kernels do not define these.
 * TODO: remove eventually
 */
#define SPEED_400000       400000
#define SPEED_200000       200000
#define ETHTOOL_LINK_MODE_50000baseKR_Full_BIT   52
#define	ETHTOOL_LINK_MODE_100000baseCR2_Full_BIT 59
#define	ETHTOOL_LINK_MODE_200000baseCR4_Full_BIT 66

/* Cassini 1 & 2 support up to 5 segments */
#define C_MAX_ETH_FRAGS 5
#define MAX_CB_LEN 3

/* Private DMA registration data stored in the SKB cb */
struct cb_data {
	u8 num_frags;
	u16 dma_len[MAX_CB_LEN]; /* alas, not enough room for C_MAX_ETH_FRAGS */
	dma_addr_t dma_addr[C_MAX_ETH_FRAGS];
};

static inline void cxi_force_icrc_check(struct sk_buff *skb)
{
#ifdef SKBTX_ZEROCOPY_FRAG
	skb_shinfo(skb)->__unused |= BIT(1);
#else
	skb_shinfo(skb)->flags |= BIT(6);
#endif
}

static inline void cxi_skb_icrc_gone(struct sk_buff *skb)
{
#ifdef SKBTX_ZEROCOPY_FRAG
	skb_shinfo(skb)->__unused |= BIT(0);
#else
	skb_shinfo(skb)->flags |= BIT(7);
#endif
}

/* An RX buffer. It can end-up on the request or priority list. */
struct rx_buffer {
	struct list_head buf_list;

	unsigned int id;

	size_t data_size;

	enum c_ptl_list ptl_list; /* PRIORITY or REQUEST */

	union {
		struct {
			/* Valid only if list is PRIORITY */
			struct page *page;
			dma_addr_t mapping;
			size_t cur_offset;
			unsigned int posted_count;
		};
		struct {
			/* Valid only if list is REQUEST */
			struct cxi_md *md;
			u8 *data;
		};
	};
};

struct tx_queue {
	struct cxi_eth *dev;
	struct cxi_eq_attr eq_attr;
	struct cxi_md *eq_md;
	struct cxi_eq *eq;
	struct cxi_cq *eth1_cq;
	struct cxi_cq *eth2_cq; /* Initialized conditionally */
	struct cxi_cq *shared_cq;
	struct cxi_cq *untagged_cq; /* Points to one of the above CQs */
	struct napi_struct napi;

	/* It is possible for start_xmit to get a lot of small packets
	 * that fill up the TX queue, in which case
	 * netif_tx_stop_queue() would be called. However since IDC
	 * packets do not generate an interrupt, the send queue may
	 * never be awaken again. Protect against that by forcing a
	 * DMA instead of IDC every 'force_dma_interval' contiguous
	 * IDC requests.
	 */
	int force_dma;
	int force_dma_interval;

	/* Associated netdev TX queue. */
	unsigned int id;

	/* NUMA node and CPU TX queue should be bound to. */
	unsigned int node;
	unsigned int cpu;

	/* Each DMA op can use up to an EQ slot (16 or 32 bytes event,
	 * plus padding). If too many DMA operations are in flight,
	 * the EQ might get overrun. Pause the EQ in that case.
	 */
	atomic_t dma_eq_cdt;

	struct {
		u64 polling;
		u64 stopped;	/* netif queue stopped by driver, CQ almost full */
		u64 stopped_eq;	/* netif queue stopped by driver, EQ almost full */
		u64 restarted;	/* netif queue restarted by driver */
		u64 tx_busy;	/* returned NETDEV_TX_BUSY */
		u64 idc;
		u64 idc_bytes;
		u64 dma;
		u64 dma_bytes;
		u64 dma_forced;
	} stats;
};

#define NB_BUCKETS 23
struct bucket {
	u64 min_ts;
	u64 max_ts;
	u64 b[NB_BUCKETS];
};

struct rx_queue {
	struct cxi_eth *dev;
	void *eq_buf;
	size_t eq_buf_size;
	struct cxi_md *eq_md;
	struct cxi_eq *eq;
	struct cxi_pte *pt;
	struct napi_struct napi;

	struct cxi_cq *cq_tgt_prio;

	/* Pool of receive buffers. Buffers start on the rx_ready
	 * list, where they wait to be appended to a portal. When a
	 * buffer is appended, it moves to the rx_in_use
	 * list. Eventually the buffer is unlinked by Cassini (usually
	 * after it becomes full) and reused again by the driver once
	 * all its data has been consumed.
	 *
	 * The buffers on the request list are for "small" packets,
	 * while the buffers on the priority list are for large
	 * packets.
	 */
	struct list_head rx_unallocated;
	struct list_head rx_ready; /* allocated but not appended */
	struct list_head rx_in_use; /* appended or unlinked */

	unsigned int rx_bufs_count;
	struct rx_buffer *rx_bufs;

	/* Segmented packet reassembly state */
	enum { FRAG_NONE, FRAG_MORE, FRAG_DROP } frag_state;
	u64 pkt_cnt;
	u8 next_seg_cnt;

	unsigned int last_bad_frag_drop_rc;

	struct {
		u64 append_prio;
		u64 append_req;
		u64 append_failed;
		u64 unlinked_prio;
		u64 unlinked_req;
	} stats;

	/* Associated netdev RX queue. */
	unsigned int id;

	/* NUMA node and CPU RX queue should be bound to. */
	unsigned int node;
	unsigned int cpu;

	/* Debug code. Should be removed eventually. Measure the time
	 * between a napi schedule call, and the call to rx_eq_cb() to
	 * process the queue.
	 */
	ktime_t time_napi_schedule;
	struct bucket eth_napi_schedule;

	/* Number of times an page should be posted to the priority queue. */
	unsigned int page_chunks;
};

/* A queue only for PTP packets. Cassini ERRATA-3258. */
#define PTP_RX_Q max_rss_queues

struct cxi_eth {
	struct list_head dev_list;
	struct cxi_dev *cxi_dev;
	bool is_c2;

	struct net_device *ndev;

	struct dentry *debug;

	bool is_active;

	int svc_id;
	u32 min_free;
	u64 mac_addr;

	/* Cassini ERRATA-3258 for PTP. Cassini 1/1.1 doesn't set the timestamp
	 * bit. Instead we have a separate portal entry that will only
	 * get the PTP packets with the special MAC address. If
	 * ptp_ts_enabled is true then these packets will have a
	 * timestamp, except for a very short window.
	 */
	bool ptp_ts_enabled;
	u64 ptp_mac_addr;
	struct sk_buff *tstamp_skb;

	unsigned int phys_lac;

	bool eth2_active;
	int eth1_pcp;
	int eth2_pcp;

	struct cxi_lni *lni;
	struct cxi_cp *eth1_cp;
	struct cxi_cp *eth2_cp;
	struct cxi_cp *shared_cp;

	/* Shared CQ with every RX channel for their target request
	 * list. As posted buffers are big, operations on this CQ will
	 * be few and far between.
	 */
	spinlock_t cq_tgt_req_lock;
	struct cxi_cq *cq_tgt_req;

	/* Event queue used for CQ errors. */
	struct cxi_eq *err_eq;
	struct cxi_eq_attr err_eq_attr;

	/* TX queues/rings. TXQ zero remains allocated for the duration of the
	 * Ethernet interface remaining open. TXQs greater than zero are
	 * allocated/deallocated base on number of configured TX queues.
	 */
	struct tx_queue *txqs;
	unsigned int cur_txqs;

	/* RX queues/rings. RXQ zero is the default/catch-all RX queue and
	 * remains allocated for the duration of the Ethernet interface
	 * remaining open. RXQs greater than zero and allocated/deallocated
	 * based on number of configured RX queues.
	 */
	struct rx_queue *rxqs;

	/* Number of RX buffers */
	unsigned int num_prio_buffers;
	unsigned int num_req_buffers;

	struct ethtool_ringparam ringparam;
#ifdef HAVE_KERNEL_RINGPARAM
	struct kernel_ethtool_ringparam kernel_ring;
#endif

	struct cxi_eth_res res;
	struct cxi_eth_info eth_info;

	/* ethtool private flags (CXI_ETH_PF_...) */
	u32 priv_flags;

	/* Timestamp config copy */
	struct hwtstamp_config tstamp_config;
};

extern unsigned int rss_indir_size;
extern unsigned int max_rss_queues;
extern unsigned int max_tx_queues;
extern unsigned int small_pkts_buf_size;
extern unsigned int idc_dma_threshold;
extern unsigned int lpe_cdt_thresh_id;
extern unsigned long rx_repost_retry_jiffies;
extern unsigned int small_pkts_buf_count;
extern unsigned int large_pkts_buf_count;
extern unsigned int buffer_threshold;

#define CXI_ETH_MAX_MTU 9000
#define CXI_ETH_TX_TIMEOUT (5 * HZ)

/* Compute an approximative number of packets that can be contained in
 * the small buffers. Since each segment cannot be larger that the
 * buffer_threshold, use that as a base.
 */
static inline u32 num_small_packets(u32 buf_count)
{
	return buf_count * (small_pkts_buf_size / buffer_threshold);
}

size_t get_rxq_eq_buf_size(struct cxi_eth *dev);
void hw_cleanup(struct cxi_eth *dev);
int hw_setup(struct cxi_eth *dev);

/* Netdev methods */
int cxi_eth_open(struct net_device *ndev);
int cxi_eth_close(struct net_device *ndev);
netdev_tx_t cxi_eth_start_xmit(struct sk_buff *skb, struct net_device *dev);
int cxi_eth_mac_addr(struct net_device *dev, void *p);
void cxi_eth_set_rx_mode(struct net_device *dev);
int cxi_change_mtu(struct net_device *netdev, int new_mtu);
void disable_rx_queue(struct rx_queue *rx);
void free_rx_queue(struct rx_queue *rx);
int alloc_rx_queue(struct cxi_eth *dev, unsigned int id);
void enable_rx_queue(struct rx_queue *rx);
int post_rx_buffers(struct rx_queue *rx, gfp_t gfp);
int alloc_tx_queue(struct cxi_eth *dev, unsigned int id);
void free_tx_queue(struct tx_queue *tx);
void enable_tx_queue(struct tx_queue *tx);
void disable_tx_queue(struct tx_queue *tx);
int cxi_set_rx_channels(struct cxi_eth *dev, unsigned int num_rx_channels);
int cxi_set_tx_channels(struct cxi_eth *dev, unsigned int num_tx_channels);
int cxi_do_ioctl(struct net_device *ndev, struct ifreq *ifr, int cmd);

extern const struct ethtool_ops cxi_eth_ethtool_ops;
#endif
