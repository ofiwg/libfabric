// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* cxi-eth debug fs*/
#include <linux/netdevice.h>
#include "cxi_eth.h"
#include "cxi_eth_debugfs.h"

static u64 bucket_times[NB_BUCKETS];

static void dump_rx_queue(struct seq_file *s, const struct rx_queue *rx)
{
	const struct bucket *bucket;
	int i;

	seq_printf(s, "  EQ=%u, PtlTE=%u TGT PRIO CQ=%u\n",
		   rx->eq->eqn, rx->pt->id, cxi_cq_get_cqn(rx->cq_tgt_prio));

	seq_printf(s, "  Preferred NUMA Node: %u\n", rx->node);
	seq_printf(s, "  Preferred CPU: %u\n", rx->cpu);

	seq_printf(s, "  list rx_unallocated=%s\n",
		   list_empty(&rx->rx_unallocated) ? "empty" : "not empty");
	seq_printf(s, "  list rx_ready=%s\n",
		   list_empty(&rx->rx_ready) ? "empty" : "not empty");
	seq_printf(s, "  list rx_in_use=%s\n",
		   list_empty(&rx->rx_in_use) ? "empty" : "not empty");

	seq_printf(s, "  rx_bufs_count=%u\n", rx->rx_bufs_count);
	seq_printf(s, "  append_prio=%llu\n", rx->stats.append_prio);
	seq_printf(s, "  append_req=%llu\n", rx->stats.append_req);
	seq_printf(s, "  append_failed=%llu\n", rx->stats.append_failed);
	seq_printf(s, "  unlinked_prio=%llu\n", rx->stats.unlinked_prio);
	seq_printf(s, "  unlinked_req=%llu\n", rx->stats.unlinked_req);
	seq_printf(s, "  Last bad RC for dropped fragment: %u\n",
		   rx->last_bad_frag_drop_rc);

	/* Each bucket */
	seq_puts(s, "  bucket for napi_schedule\n");
	bucket = &rx->eth_napi_schedule;
	if (bucket->min_ts == UINT_MAX)
		seq_printf(s, "    min_ts=%llu\n", 0ULL);
	else
		seq_printf(s, "    min_ts=%llu\n", bucket->min_ts);
	seq_printf(s, "    max_ts=%llu\n", bucket->max_ts);
	for (i = 0; i < NB_BUCKETS; i++) {
		if (i < (NB_BUCKETS - 1))
			seq_printf(s, "      < %llu ns       %llu\n",
				   bucket_times[i],
				   bucket->b[i]);
		else
			seq_printf(s, "      higher          %llu\n",
				   bucket->b[i]);
	}
	seq_puts(s, "\n");
}

static void dump_cq_info(struct seq_file *s, struct cxi_cq *cq)
{
	seq_printf(s, "    id: %u\n", cxi_cq_get_cqn(cq));
	seq_printf(s, "    ptrs: size32=%u, rp32=%llu, hw_rp32=%u, wp32=%llu, hw_wp32=%llu\n",
		   cq->size32, cq->rp32, cq->status->rd_ptr * 2,
		   cq->wp32, cq->hw_wp32);
	seq_printf(s, "    free CQ slots: %d\n", __cxi_cq_free_slots(cq));
}

static void dump_tx_queue(struct seq_file *s, const struct tx_queue *txq,
			  struct cxi_eth *dev)
{
	struct netdev_queue *dev_queue =
		netdev_get_tx_queue(txq->dev->ndev, txq->id);

	seq_printf(s, "  EQ=%u\n", txq->eq->eqn);
	seq_puts(s,   "  eth1 CQ:\n");
	dump_cq_info(s, txq->eth1_cq);
	if (dev->eth2_active) {
		seq_puts(s,   "  eth2 CQ:\n");
		dump_cq_info(s, txq->eth2_cq);
	}
	seq_puts(s,   "  shared CQ:\n");
	dump_cq_info(s, txq->shared_cq);

	seq_printf(s, "  Preferred NUMA Node: %u\n", txq->node);
	seq_printf(s, "  Preferred CPU: %u\n", txq->cpu);
	seq_printf(s, "  polling: %llu\n", txq->stats.polling);
	seq_printf(s, "  TXQ EQ credits: %d\n", atomic_read(&txq->dma_eq_cdt));
	seq_printf(s, "  TXQ stopped count for full CQ: %llu\n", txq->stats.stopped);
	seq_printf(s, "  TXQ stopped for no EQ credits: %llu\n", txq->stats.stopped_eq);
	seq_printf(s, "  TXQ restarted count: %llu\n", txq->stats.restarted);
	seq_printf(s, "  TXQ got busy: %llu\n", txq->stats.tx_busy);
	seq_printf(s, "  idc: %llu\n", txq->stats.idc);
	seq_printf(s, "  idc_bytes: %llu\n", txq->stats.idc_bytes);
	seq_printf(s, "  dma: %llu\n", txq->stats.dma);
	seq_printf(s, "  dma_bytes: %llu\n", txq->stats.dma_bytes);
	seq_printf(s, "  dma_forced: %llu\n", txq->stats.dma_forced);
	seq_printf(s, "  force_dma: %u\n", txq->force_dma);
	seq_printf(s, "  force_dma_interval: %u\n", txq->force_dma_interval);
	seq_printf(s, "  netdev queue state: 0x%lx\n", dev_queue->state);
}

static int dump_dev(struct seq_file *s, void *unused)
{
	struct cxi_eth *dev = s->private;
	u64 min_ts = UINT_MAX;
	u64 max_ts = 0;
	int i;
	int j;

	seq_printf(s, "rss_indir_size=%u\n", rss_indir_size);
	seq_printf(s, "idc_dma_threshold=%u\n", idc_dma_threshold);
	seq_printf(s, "small_pkts_buf_size=%u\n", small_pkts_buf_size);
	seq_printf(s, "small_pkts_buf_count=%u\n", small_pkts_buf_count);
	seq_printf(s, "large_pkts_buf_count=%u\n", large_pkts_buf_count);
	seq_printf(s, "num rx_pending=%u\n", dev->ringparam.rx_pending);
	seq_printf(s, "num rx_mini_pending=%u\n", dev->ringparam.rx_mini_pending);
	seq_printf(s, "num tx_pending=%u\n", dev->ringparam.tx_pending);
	seq_printf(s, "num_prio_buffers=%u\n", dev->num_prio_buffers);
	seq_printf(s, "num_req_buffers=%u\n", dev->num_req_buffers);

	if (!dev->is_active)
		return 0;

	seq_printf(s, "shared TGT REQ CQ=%u\n",
		   cxi_cq_get_cqn(dev->cq_tgt_req));

	for (i = 0; i < dev->res.rss_queues; i++) {
		seq_printf(s, "RX queue %u\n", i);
		dump_rx_queue(s, &dev->rxqs[i]);
	}

	if (dev->res.rss_queues) {
		seq_printf(s, "PTP RX queue %u\n", PTP_RX_Q);
		dump_rx_queue(s, &dev->rxqs[PTP_RX_Q]);
	}

	for (i = 0; i < dev->cur_txqs; i++) {
		seq_printf(s, "TX queue %u\n", i);
		dump_tx_queue(s, &dev->txqs[i], dev);
	}

	/* Sum all buckets */
	seq_puts(s, "\n");
	seq_puts(s, "ALL buckets for RX napi_schedule\n");

	for (i = 0; i < dev->res.rss_queues; i++) {
		const struct bucket *bucket = &dev->rxqs[i].eth_napi_schedule;

		if (min_ts > bucket->min_ts)
			min_ts = bucket->min_ts;
		if (max_ts < bucket->max_ts)
			max_ts = bucket->max_ts;
	}

	if (min_ts == UINT_MAX)
		min_ts = 0;

	seq_printf(s, "    min_ts=%llu\n", min_ts);
	seq_printf(s, "    max_ts=%llu\n", max_ts);

	for (i = 0; i < NB_BUCKETS; i++) {
		u64 nb = 0;

		for (j = 0; j < dev->res.rss_queues; j++)
			nb += dev->rxqs[j].eth_napi_schedule.b[i];

		if (i < (NB_BUCKETS - 1))
			seq_printf(s, "      < %llu ns       %llu\n",
				   bucket_times[i],
				   nb);
		else
			seq_printf(s, "      higher          %llu\n",
				   nb);
	}
	seq_puts(s, "\n");

	return 0;
}

static int dump_dev_debug_dev_open(struct inode *inode, struct file *file)
{
	return single_open(file, dump_dev, inode->i_private);
}

static const struct file_operations cxi_eth_fops = {
	.owner = THIS_MODULE,
	.open = dump_dev_debug_dev_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

struct dentry *device_eth_debugfs_init(void)
{
	int i;

	for (i = 0; i < NB_BUCKETS; i++)
		bucket_times[i] = 1 << i;
	return debugfs_create_dir("cxi_eth", NULL);
}

void device_debugfs_create(char *name, struct cxi_eth *dev,
			struct dentry *cxieth_debug_dir)
{
	dev->debug = debugfs_create_file(name, 0444, cxieth_debug_dir,
					 dev, &cxi_eth_fops);
}
