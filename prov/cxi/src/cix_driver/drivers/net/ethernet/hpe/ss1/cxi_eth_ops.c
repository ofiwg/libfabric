// SPDX-License-Identifier: GPL-2.0
/*
 * Cray Cassini ethernet driver
 * © Copyright 2018-2020 Cray Inc
 * © Copyright 2020 Hewlett Packard Enterprise Development LP
 */

#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/etherdevice.h>
#include <linux/if_vlan.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/netdevice.h>
#include <linux/pci.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/vmalloc.h>
#include <rdma/ib_verbs.h>	/* for RoCE port number */

#include "cxi_eth.h"
#include "cxi_core.h"

static unsigned int tx_eq_count;
module_param(tx_eq_count, uint, 0444);
MODULE_PARM_DESC(tx_eq_count, "Override the number of entries in transmit event queue");

static int cxi_rx_eth_poll(struct napi_struct *napi, int budget);
static void rx_eq_cb(void *context);

static void err_eq_cb(void *context)
{
	struct cxi_eth *dev = context;
	const union c_event *event;

	while ((event = cxi_eq_get_event(dev->err_eq))) {
		switch (event->hdr.event_type) {
		case C_EVENT_COMMAND_FAILURE:
			netdev_warn(dev->ndev, "%s CQ %u disabled\n",
				    event->cmd_fail.is_target ?
				    "target" : "transmit",
				    event->cmd_fail.cq_id);
			break;
		default:
			netdev_warn(dev->ndev, "Unhandled event %u, rc %d\n",
				    event->hdr.event_type,
				    event->hdr.return_code);
			print_hex_dump(KERN_INFO,
				       "cxi eth unexpected tx event: ",
				       DUMP_PREFIX_ADDRESS, 16, 1, event,
				       (8 << event->hdr.event_size), false);
		}

		cxi_eq_ack_events(dev->err_eq);
	}

	cxi_eq_int_enable(dev->err_eq);
}

static void finalize_tx_tstamp(struct cxi_eth *dev, struct sk_buff *skb)
{
	struct skb_shared_hwtstamps tstamps;
	int rc;

	if (skb != dev->tstamp_skb) {
		netdev_warn_once(dev->ndev, "skb != dev->tstamp_skb : %p != %p",
				 skb, dev->tstamp_skb);
		return;
	}

	rc = cxi_eth_get_tx_timestamp(dev->cxi_dev, &tstamps);
	if (!rc)
		skb_tstamp_tx(skb, &tstamps);

	/* Release timestamp slot */
	cmpxchg(&dev->tstamp_skb, skb, NULL);
}

static void unmap_frags(struct device *dma_dev, struct sk_buff *skb)
{
	struct cb_data *cb = (struct cb_data *)skb->cb;
	int i;

	for (i = 0; i < cb->num_frags; i++) {
		int len;

		if (i < MAX_CB_LEN) {
			len = cb->dma_len[i];
		} else {
			const skb_frag_t *frag = &skb_shinfo(skb)->frags[i - 1];

			len = skb_frag_size(frag);
		}

		dma_unmap_single(dma_dev, cb->dma_addr[i], len, DMA_TO_DEVICE);
	}
}

/* Prepare the segments in the command, mapping them for DMA */
static int prepare_frags(struct device *dma_dev, struct sk_buff *skb,
			 struct c_dma_eth_cmd *dma_eth)
{
	struct cb_data *cb = (struct cb_data *)skb->cb;
	int i;

	BUILD_BUG_ON(sizeof(struct cb_data) > sizeof(skb->cb));

	dma_eth->num_segments = 1 + skb_shinfo(skb)->nr_frags;

	if (skb_shinfo(skb)->nr_frags) {
		unsigned int segnum = 0;

		cb->dma_len[segnum] = dma_eth->len[segnum] = skb_headlen(skb);
		segnum++;

		for (i = 0; i < skb_shinfo(skb)->nr_frags; i++) {
			const skb_frag_t *frag = &skb_shinfo(skb)->frags[i];

			dma_eth->len[segnum] = skb_frag_size(frag);

			if (segnum < MAX_CB_LEN)
				cb->dma_len[segnum] = dma_eth->len[segnum];

			segnum++;
		}
	} else {
		cb->dma_len[0] = skb->len;
		dma_eth->len[0] = skb->len;
	}

	cb->num_frags = 0;

	for (i = 0; i < dma_eth->num_segments; i++) {
		dma_addr_t dma_addr;

		if (i == 0) {
			dma_addr = dma_map_single(dma_dev, skb->data,
						  dma_eth->len[i], DMA_TO_DEVICE);
		} else {
			const skb_frag_t *frag = &skb_shinfo(skb)->frags[i - 1];

			dma_addr = dma_map_single(dma_dev, skb_frag_address(frag),
						  dma_eth->len[i], DMA_TO_DEVICE);
		}

		if (dma_mapping_error(dma_dev, dma_addr)) {
			unmap_frags(dma_dev, skb);
			return -ENOMEM;
		}

		cb->dma_addr[i] = dma_eth->addr[i] = dma_addr;
		cb->num_frags++;
	}

	return 0;
}

/* A packet was sent with a DMA command. Returns the number of EQ
 * credits to return to the TX queue.
 */
static unsigned int eth_complete(struct cxi_eth *dev, struct tx_queue *txq,
				 const union c_event *event, int budget,
				 unsigned int *pkts, unsigned int *bytes)
{
	struct net_device *ndev = dev->ndev;
	struct sk_buff *skb;
	enum c_return_code return_code;

	if (event->hdr.event_size == C_EVENT_SIZE_16_BYTE) {
		return_code = event->init_short.return_code;
		skb = (void *)event->init_short.user_ptr;
	} else {
		return_code = event->init_long.return_code;
		skb = (void *)event->init_long.user_ptr;
	}

	if (skb == NULL) {
		/* Bad IDC commands will generate a SEND event on error. */
		if (return_code == C_RC_OK)
			netdev_warn_once(dev->ndev,
					 "Successful command with NULL skb\n");

		ndev->stats.tx_dropped++;

		return 0;
	}

	if (return_code == C_RC_OK) {
		ndev->stats.tx_packets++;
		ndev->stats.tx_bytes += skb->len;
		*pkts += 1;
		*bytes += skb->len;
	} else {
		netdev_err_once(dev->ndev,
				"DMA command failed to complete: %d\n",
				return_code);
		ndev->stats.tx_dropped++;
	}

	if (skb_shinfo(skb)->tx_flags & SKBTX_IN_PROGRESS)
		finalize_tx_tstamp(dev, skb);

	unmap_frags(&dev->cxi_dev->pdev->dev, skb);

	napi_consume_skb(skb, budget);

	return 1;
}

static int cxi_tx_eth_poll(struct napi_struct *napi, int budget)
{
	struct tx_queue *txq = container_of(napi, struct tx_queue, napi);
	struct cxi_eth *dev = txq->dev;
	struct netdev_queue *dev_queue =
		netdev_get_tx_queue(dev->ndev, txq->id);
	const union c_event *event;
	int work_done = 0;
	unsigned int event_count = 0;
	unsigned int pkts = 0;
	unsigned int bytes = 0;
	unsigned int cdts = 0;

	txq->stats.polling++;

	while (work_done < budget && (event = cxi_eq_get_event(txq->eq))) {
		switch (event->hdr.event_type) {
		case C_EVENT_SEND:
			cdts += eth_complete(dev, txq, event, budget, &pkts, &bytes);
			work_done++;
			break;
		default:
			netdev_warn(dev->ndev, "Unhandled event %u, rc %d\n",
				    event->hdr.event_type,
				    event->hdr.return_code);
			print_hex_dump(KERN_INFO,
				       "cxi eth unexpected tx event: ",
				       DUMP_PREFIX_ADDRESS, 16, 1, event,
				       (8 << event->hdr.event_size), false);
		}

		event_count++;
		if (event_count == 16) {
			cxi_eq_int_disable(txq->eq);
			event_count = 0;

			atomic_add(cdts, &txq->dma_eq_cdt);
			cdts = 0;
		}
	}

	BUG_ON(work_done > budget);

	/* Check to see if the TX queue needs to be re-enabled. */
	if (netif_tx_queue_stopped(dev_queue)) {
		if (__cxi_cq_free_slots(txq->shared_cq) >= 16 &&
		    __cxi_cq_free_slots(txq->eth1_cq) >= 16 &&
		    (!dev->eth2_active || __cxi_cq_free_slots(txq->eth2_cq) >= 16) &&
		    atomic_read(&txq->dma_eq_cdt) > 0) {
			txq->stats.restarted++;
			netif_tx_wake_queue(dev_queue);
		}
	}

	if (pkts)
		netdev_tx_completed_queue(dev_queue, pkts, bytes);

	if (work_done < budget && napi_complete_done(napi, work_done)) {
		/* Ack events and re-enable the EQ interrupt. */
		cxi_eq_int_enable(txq->eq);
	} else if (event_count) {
		cxi_eq_int_disable(txq->eq);
	}

	atomic_add(cdts, &txq->dma_eq_cdt);

	return work_done;
}

static void tx_eq_cb(void *context)
{
	struct tx_queue *tx = context;

	napi_schedule(&tx->napi);
}

void enable_tx_queue(struct tx_queue *tx)
{
	napi_enable(&tx->napi);
	cxi_eq_int_enable(tx->eq);
}

void disable_tx_queue(struct tx_queue *tx)
{
	cxi_eq_int_disable(tx->eq);
	napi_disable(&tx->napi);
}

void free_tx_queue(struct tx_queue *tx)
{
	netif_napi_del(&tx->napi);

	tx->untagged_cq = NULL;

	cxi_cq_free(tx->shared_cq);
	tx->shared_cq = NULL;

	cxi_cq_free(tx->eth1_cq);
	tx->eth1_cq = NULL;

	if (tx->eth2_cq) {
		cxi_cq_free(tx->eth2_cq);
		tx->eth2_cq = NULL;
	}

	cxi_eq_free(tx->eq);
	tx->eq = NULL;

	if (is_vmalloc_addr(tx->eq_attr.queue))
		cxi_unmap(tx->eq_md);
	kvfree(tx->eq_attr.queue);
	tx->eq_attr.queue = NULL;

	tx->dev = NULL;
}

static void cxi_eth_get_node_cpu(struct cxi_eth *dev, unsigned int queue_id,
				 unsigned int *queue_node,
				 unsigned int *queue_cpu)
{
	int node;
	int cpu;
	int cpu_offset;
	const struct cpumask *cpus;
	int cpu_count;
	int num_cpu_nodes = 0;
	int i;

	/* Figure out how many numa nodes are online and contain
	 * CPUs. Apparently linux will reorder the nodes so that the
	 * empty ones are at the end of the list. If it's not the
	 * case, then further changes will be needed.
	 */
	for_each_online_node(node) {
		cpus = cpumask_of_node(node);
		if (!cpumask_empty(cpus))
			num_cpu_nodes++;
	}

	/* Round robin queues across all the nodes in the system instead of
	 * overloading 1 node with multiple queues. Favor the local node to
	 * start with.
	 */
	node = (dev_to_node(&dev->cxi_dev->pdev->dev) + queue_id) % num_cpu_nodes;
	if (!node_online(node)) {
		/* For sanity, but that shouldn't be possible. */
		netdev_err(dev->ndev, "Node %u not online. Defaulting to %u\n",
			   node, dev_to_node(&dev->cxi_dev->pdev->dev));
		node = dev_to_node(&dev->cxi_dev->pdev->dev);
	}

	/* If the queue ID exceeds the number of nodes, multiple queues will be
	 * mapped to the same node. For this case, round robin queues within the
	 * node.
	 */
	cpus = cpumask_of_node(node);
	if (cpumask_empty(cpus)) {
		/* For sanity, but that shouldn't be possible. */
		netdev_err(dev->ndev, "Node %u is empty. Defaulting to all cpus.\n",
			   node);
		cpus = cpu_all_mask;
	}

	cpu_count = 0;
	for_each_cpu(cpu, cpus) {
		if (cpu_is_offline(cpu))
			continue;
		/* We only care about CPUs which belong to the NUMA node. */
		if (cpumask_test_cpu(cpu, cpus))
			cpu_count++;
	}

	cpu_offset = (queue_id / num_cpu_nodes) % cpu_count;
	i = 0;
	for_each_cpu(cpu, cpus) {
		if (cpu_is_offline(cpu))
			continue;
		if (cpumask_test_cpu(cpu, cpus)) {
			if (i == cpu_offset)
				break;
			i++;
		}
	}

	*queue_node = cpu_to_node(cpu);
	*queue_cpu = cpu;
}

int alloc_tx_queue(struct cxi_eth *dev, unsigned int id)
{
	struct tx_queue *tx = &dev->txqs[id];
	struct cxi_cq_alloc_opts cq_alloc_opts = {};
	struct c_cstate_cmd c_state = {};
	unsigned int eq_count;
	int rc;

	memset(tx, 0, sizeof(*tx));

	cxi_eth_get_node_cpu(dev, id, &tx->node, &tx->cpu);

	tx->force_dma_interval = dev->ndev->tx_queue_len / 8;
	tx->force_dma = tx->force_dma_interval;

	cq_alloc_opts.count = min_t(unsigned int, CXI_MAX_CQ_COUNT,
				    2 * dev->ndev->tx_queue_len);

	eq_count = tx_eq_count ? tx_eq_count : 4 * cq_alloc_opts.count;
	tx->eq_attr.queue_len = PAGE_ALIGN(C_EE_CFG_ECB_SIZE * eq_count);

	tx->eq_attr.queue = kvzalloc_node(tx->eq_attr.queue_len, GFP_KERNEL,
					  tx->node);
	if (!tx->eq_attr.queue) {
		rc = -ENOMEM;
		goto err;
	}

	/* Keep a 5% margin for IDC send errors if they ever happen. */
	atomic_set(&tx->dma_eq_cdt, (eq_count * 95) / 100);

	if (is_vmalloc_addr(tx->eq_attr.queue)) {
		tx->eq_md = cxi_map(dev->lni, (uintptr_t)tx->eq_attr.queue,
				    tx->eq_attr.queue_len, CXI_MAP_WRITE, NULL);
		if (IS_ERR(tx->eq_md)) {
			rc = PTR_ERR(tx->eq_md);
			goto err_free_eq_buf;
		}
	} else {
		tx->eq_attr.flags = CXI_EQ_PASSTHROUGH;
	}

	tx->eq_attr.cpu_affinity = tx->cpu;
	tx->eq = cxi_eq_alloc(dev->lni, tx->eq_md, &tx->eq_attr, tx_eq_cb, tx,
			      NULL, NULL);
	if (IS_ERR(tx->eq)) {
		rc = PTR_ERR(tx->eq);
		goto err_unmap_eq_buf;
	}

	cq_alloc_opts.flags = CXI_CQ_IS_TX | CXI_CQ_TX_ETHERNET;
	cq_alloc_opts.lcid = dev->eth1_cp->lcid;
	tx->eth1_cq = cxi_cq_alloc(dev->lni, dev->err_eq, &cq_alloc_opts,
				      tx->node);
	if (IS_ERR(tx->eth1_cq)) {
		rc = PTR_ERR(tx->eth1_cq);
		goto err_free_eq;
	}

	cq_alloc_opts.lcid = dev->shared_cp->lcid;
	tx->shared_cq = cxi_cq_alloc(dev->lni, dev->err_eq, &cq_alloc_opts,
				     tx->node);
	if (IS_ERR(tx->shared_cq)) {
		rc = PTR_ERR(tx->shared_cq);
		goto err_free_eth1_cq;
	}

	if (dev->eth2_active) {
		cq_alloc_opts.lcid = dev->eth2_cp->lcid;
		tx->eth2_cq = cxi_cq_alloc(dev->lni, dev->err_eq,
					   &cq_alloc_opts, tx->node);

		if (IS_ERR(tx->eth2_cq)) {
			rc = PTR_ERR(tx->eth2_cq);
			goto err_free_shared_cq;
		}
	}

	/* Add C_STATE command. */
	c_state.eq = tx->eq->eqn;
	c_state.restricted = 1;
	c_state.event_success_disable = 1;
	c_state.event_send_disable = 1;

	rc = cxi_cq_emit_c_state(tx->eth1_cq, &c_state);
	if (rc) {
		netdev_err(dev->ndev, "Command emit failed: %d\n", rc);
		goto err_free_eth2_cq;
	}

	cxi_cq_ring(tx->eth1_cq);

	rc = cxi_cq_emit_c_state(tx->shared_cq, &c_state);
	if (rc) {
		netdev_err(dev->ndev, "Command emit failed: %d\n", rc);
		goto err_free_eth2_cq;
	}

	cxi_cq_ring(tx->shared_cq);

	if (dev->eth2_active) {
		rc = cxi_cq_emit_c_state(tx->eth2_cq, &c_state);
		if (rc) {
			netdev_err(dev->ndev, "Command emit failed: %d\n", rc);
			goto err_free_eth2_cq;
		}

		cxi_cq_ring(tx->eth2_cq);
	}

	/* Determine which tx_queue CQ should be used for untagged traffic */
	if (dev->cxi_dev->untagged_eth_pcp == dev->eth1_pcp)
		tx->untagged_cq = tx->eth1_cq;
	else if (dev->eth2_active &&
		 dev->cxi_dev->untagged_eth_pcp == dev->eth2_pcp)
		tx->untagged_cq = tx->eth2_cq;
	else
		tx->untagged_cq = tx->shared_cq;

	tx->dev = dev;
	tx->id = id;

	NETIF_NAPI_ADD_TX(dev->ndev, &tx->napi, cxi_tx_eth_poll);

	return 0;

err_free_eth2_cq:
	if (dev->eth2_active) {
		cxi_cq_free(tx->eth2_cq);
		tx->eth2_cq = NULL;
	}
err_free_shared_cq:
	cxi_cq_free(tx->shared_cq);
	tx->shared_cq = NULL;
err_free_eth1_cq:
	cxi_cq_free(tx->eth1_cq);
	tx->eth1_cq = NULL;
err_free_eq:
	cxi_eq_free(tx->eq);
	tx->eq = NULL;
err_unmap_eq_buf:
	if (is_vmalloc_addr(tx->eq_attr.queue))
		cxi_unmap(tx->eq_md);
err_free_eq_buf:
	kvfree(tx->eq_attr.queue);
	tx->eq_attr.queue = NULL;
err:
	return rc;
}

/* Free the receive buffers on the ready lists.
 * TODO: when closing the device, make sure all buffers are back on
 * the ready lists.
 */
static void free_rx_buffers(struct rx_queue *rx)
{
	struct cxi_eth *dev = rx->dev;
	struct rx_buffer *buf;

	/* Move all buffers to the ready list and free them. */
	list_splice_init(&rx->rx_in_use, &rx->rx_ready);

	while ((buf = list_first_entry_or_null(&rx->rx_ready,
					       struct rx_buffer, buf_list))) {
		list_del(&buf->buf_list);
		list_add_tail(&buf->buf_list, &rx->rx_unallocated);

		if (buf->ptl_list == C_PTL_LIST_PRIORITY) {
			if (buf->page) {
				dma_unmap_page(&dev->cxi_dev->pdev->dev,
					dma_unmap_addr(buf, mapping),
				PAGE_SIZE, DMA_FROM_DEVICE);

				for (; buf->posted_count > 0; buf->posted_count--)
					put_page(buf->page);

				__free_page(buf->page);
				buf->page = NULL;
			}
		} else {
			cxi_unmap(buf->md);
			kvfree(buf->data);
		}
		buf->data = NULL;
	}
}

/* Allocate and map the receive buffers. */
static int alloc_rx_buffers(struct rx_queue *rx, unsigned int count,
			    enum c_ptl_list ptl_list)
{
	int rc;
	unsigned int i;
	struct rx_buffer *buf;
	unsigned int data_size;
	struct cxi_eth *dev = rx->dev;
	int numa_node = dev_to_node(&dev->cxi_dev->pdev->dev);

	if (ptl_list == C_PTL_LIST_PRIORITY)
		data_size = dev->eth_info.max_segment_size;
	else
		data_size = small_pkts_buf_size;

	for (i = 0; i < count; i++) {
		buf = list_first_entry_or_null(&rx->rx_unallocated,
					       struct rx_buffer, buf_list);
		if (!buf)
			break;

		list_del(&buf->buf_list);

		buf->data_size = data_size;

		if (ptl_list == C_PTL_LIST_REQUEST) {
			buf->data = kvzalloc_node(buf->data_size, GFP_KERNEL,
						  numa_node);
			if (!buf->data)
				goto out_requeue;

			buf->md = cxi_map(dev->lni, (uintptr_t)buf->data,
					  buf->data_size, CXI_MAP_WRITE, NULL);
			if (IS_ERR(buf->md)) {
				rc = PTR_ERR(buf->md);
				netdev_err(dev->ndev,
					   "cxi_map failed: %d\n", rc);
				kvfree(buf->data);
				buf->data = NULL;
				goto out_requeue;
			}
		}

		/* Allocated. Enqueue on the corresponding ready list. */
		buf->ptl_list = ptl_list;

		list_add_tail(&buf->buf_list, &rx->rx_ready);

		if (ptl_list == C_PTL_LIST_PRIORITY)
			dev->num_prio_buffers++;
		else
			dev->num_req_buffers++;
	}

	return i;

out_requeue:
	list_add_tail(&buf->buf_list, &rx->rx_unallocated);

	return i;
}

/* Re-post an RX buffer. The buffer is on the rx_ready list. */
static int append_rx_buffer(struct rx_queue *rx, struct rx_buffer *buf,
			    gfp_t gfp)
{
	int rc;
	dma_addr_t mapping;
	struct page *page;
	struct cxi_eth *dev = rx->dev;
	struct c_target_cmd cmd = {
		.command.opcode = C_CMD_TGT_APPEND,
		.ptl_list = buf->ptl_list,
		.ptlte_index = rx->pt->id,
		.op_put = 1,
		.no_truncate = 1,
		.event_link_disable = 1,
		.length = buf->data_size,
		.buffer_id = buf->id,
		.unrestricted_end_ro = 1,
		.unrestricted_body_ro = 1,
		.event_unlink_disable = 1,
	};
	int i;
	size_t posted_offset;

	if (buf->ptl_list == C_PTL_LIST_PRIORITY) {
		cmd.use_once = 1;

		if (buf->page == NULL) {
			page = alloc_page(gfp);
			if (!page) {
				rc = -ENOMEM;
				goto fail;
			}

			mapping = dma_map_page(&dev->cxi_dev->pdev->dev, page,
					       0, PAGE_SIZE,
					       DMA_FROM_DEVICE);
			if (dma_mapping_error(&dev->cxi_dev->pdev->dev,
					      mapping)) {
				__free_page(page);
				rc = -EIO;
				goto fail;
			}
			buf->page = page;
			buf->mapping = mapping;
		}

		cmd.start = buf->mapping;
		cmd.lac = dev->phys_lac;
		buf->cur_offset = 0;
		buf->posted_count = 0;
		posted_offset = 0;

		for (i = 0; i < rx->page_chunks; i++) {
			cmd.start = buf->mapping + posted_offset;

			rc = cxi_cq_emit_target(rx->cq_tgt_prio, &cmd);
			if (rc)
				goto fail;

			get_page(buf->page);
			buf->posted_count++;
			posted_offset += dev->eth_info.max_segment_size;
			rx->stats.append_prio++;
		}
	} else {
		cmd.manage_local = 1;
		cmd.min_free = dev->min_free;
		cmd.start = buf->md->iova;
		cmd.lac = buf->md->lac;

		spin_lock_bh(&dev->cq_tgt_req_lock);
		rc = cxi_cq_emit_target(dev->cq_tgt_req, &cmd);
		spin_unlock_bh(&dev->cq_tgt_req_lock);

		rx->stats.append_req++;
	}

fail:
	if (rc) {
		netdev_err_once(dev->ndev, "Append failed on list %d: %d\n",
				buf->ptl_list, rc);
		rx->stats.append_failed++;
	} else {
		list_del(&buf->buf_list);
		list_add_tail(&buf->buf_list, &rx->rx_in_use);
	}

	return rc;
}

/* Append all ready RX buffers to the portal.
 * Return the number of buffers appended.
 */
int post_rx_buffers(struct rx_queue *rx, gfp_t gfp)
{
	struct cxi_eth *dev = rx->dev;
	unsigned int posted = 0;
	unsigned int count = 0;
	struct rx_buffer *buf;
	struct rx_buffer *tmp;
	int rc = 0;

	list_for_each_entry_safe(buf, tmp, &rx->rx_ready, buf_list) {
		rc = append_rx_buffer(rx, buf, gfp);
		if (rc)
			break;

		/* Ring the CQ regularly */
		if (buf->ptl_list == C_PTL_LIST_PRIORITY) {
			posted++;
			if (posted == 16) {
				cxi_cq_ring(rx->cq_tgt_prio);
				posted = 0;
			}
		} else {
			spin_lock_bh(&dev->cq_tgt_req_lock);
			cxi_cq_ring(dev->cq_tgt_req);
			spin_unlock_bh(&dev->cq_tgt_req_lock);
		}

		count++;
	}

	if (posted)
		cxi_cq_ring(rx->cq_tgt_prio);

	return rc ? rc : count;
}

#define EXPECTED_TARGET_EVENT_SIZE (sizeof(struct c_event_target_enet))
#define MIN_PACKET_SIZE 64U
size_t get_rxq_eq_buf_size(struct cxi_eth *dev)
{
	size_t num_events;

	/* Calculate the maximum number of events which could occur based on
	 * RX buffer configuration.
	 */
	num_events = dev->ringparam.rx_pending +
		(small_pkts_buf_count * (small_pkts_buf_size / MIN_PACKET_SIZE));

	return ALIGN((num_events * EXPECTED_TARGET_EVENT_SIZE) +
		     C_EE_CFG_ECB_SIZE, PAGE_SIZE);
}

/* CXI_CQ_UPDATE_HIGH_FREQ_EMPTY updates the read pointer every 16 slots.
 * Since a read pointer status update could be missed on CQ wrap, 2 * 16 slots
 * should be included as overhead.
 *
 * In addition, 4 event slots are reserved for status updates. Thus, an
 * additional 4 slots should be included as overhead.
 */
#define TGT_CQ_OVERHEAD_COUNT ((2U * 16U) + 4U)

int alloc_rx_queue(struct cxi_eth *dev, unsigned int id)
{
	int i;
	int rc;
	static const struct cxi_pt_alloc_opts pt_alloc_opts = {
		.en_flowctrl = 1, /* TODO: needed for ethernet? */
		.ethernet = 1,
		.do_space_check = 1,
		.lossless = 1,
	};
	struct cxi_eq_attr eq_attr = {};
	struct cxi_cq_alloc_opts cq_alloc_opts = {};
	struct rx_queue *rx = &dev->rxqs[id];
	unsigned int prio_bufs;

	memset(rx, 0, sizeof(*rx));

	cxi_eth_get_node_cpu(dev, id, &rx->node, &rx->cpu);

	rx->dev = dev;
	rx->id = id;
	rx->last_bad_frag_drop_rc = C_RC_OK;
	rx->eth_napi_schedule.min_ts = UINT_MAX;

	INIT_LIST_HEAD(&rx->rx_unallocated);
	INIT_LIST_HEAD(&rx->rx_ready);
	INIT_LIST_HEAD(&rx->rx_in_use);

	/* Packets landing on priority buffers will be chopped into
	 * max_segment_size pieces. Since pages are priority buffers, to avoid
	 * wasting page memory, pages should be chunked in max_segment_size
	 * pieces and posted.
	 */
	rx->page_chunks = PAGE_SIZE / dev->eth_info.max_segment_size;

	/* The number of priority buffers, not pages, is the size of the RX
	 * ring divided by number of page chunks. If the
	 * PAGE_SIZE ==  max_segment_size, the number of prio_bufs will equal
	 * rx_pending.
	 */
	prio_bufs = DIV_ROUND_UP(dev->ringparam.rx_pending, rx->page_chunks);
	rx->rx_bufs_count = prio_bufs + small_pkts_buf_count;
	rx->rx_bufs = vzalloc(rx->rx_bufs_count * sizeof(*rx->rx_bufs));
	if (rx->rx_bufs == NULL) {
		rc = -ENOMEM;
		goto err;
	}

	for (i = 0; i < rx->rx_bufs_count; i++) {
		struct rx_buffer *buf = &rx->rx_bufs[i];

		buf->id = i;
		list_add_tail(&buf->buf_list, &rx->rx_unallocated);
	}

	/* Allocate RX buffers */
	rc = alloc_rx_buffers(rx, prio_bufs, C_PTL_LIST_PRIORITY);
	if (rc == 0) {
		netdev_info(dev->ndev, "Cannot allocate RX priority buffers\n");
		rc = -ENOMEM;
		goto err_free_rx_bufs;
	} else if (rc != prio_bufs) {
		netdev_info(dev->ndev,
			    "Only allocated %d RX priority buffers out of %u requested\n",
			    rc, prio_bufs);
	}

	rc = alloc_rx_buffers(rx, small_pkts_buf_count, C_PTL_LIST_REQUEST);
	if (rc == 0) {
		netdev_info(dev->ndev, "Cannot allocate RX request buffers\n");
		rc = -ENOMEM;
		goto err_free_buffers;
	} else if (rc != small_pkts_buf_count) {
		netdev_info(dev->ndev,
			    "Only allocated %d RX request buffers out of %d requested\n",
			    rc, small_pkts_buf_count);
	}

	cq_alloc_opts.policy = CXI_CQ_UPDATE_HIGH_FREQ_EMPTY;
	cq_alloc_opts.count = dev->ringparam.rx_pending + TGT_CQ_OVERHEAD_COUNT;
	cq_alloc_opts.lpe_cdt_thresh_id = lpe_cdt_thresh_id;
	rx->cq_tgt_prio = cxi_cq_alloc(dev->lni, dev->err_eq, &cq_alloc_opts,
				       rx->node);
	if (IS_ERR(rx->cq_tgt_prio)) {
		rc = PTR_ERR(rx->cq_tgt_prio);
		goto err_free_buffers;
	}

	rx->eq_buf_size = get_rxq_eq_buf_size(dev);
	rx->eq_buf = kvzalloc_node(rx->eq_buf_size, GFP_KERNEL, rx->node);
	if (!rx->eq_buf) {
		rc = -ENOMEM;
		goto err_free_cq;
	}

	if (is_vmalloc_addr(rx->eq_buf)) {
		rx->eq_md = cxi_map(dev->lni, (uintptr_t)rx->eq_buf,
				    rx->eq_buf_size, CXI_MAP_WRITE, NULL);
		if (IS_ERR(rx->eq_md)) {
			rc = PTR_ERR(rx->eq_md);
			goto err_free_eq_buf;
		}
	} else {
		eq_attr.flags = CXI_EQ_PASSTHROUGH;
	}

	eq_attr.queue = rx->eq_buf;
	eq_attr.queue_len = rx->eq_buf_size;
	eq_attr.cpu_affinity = rx->cpu;
	rx->eq = cxi_eq_alloc(dev->lni, rx->eq_md, &eq_attr, rx_eq_cb, rx, NULL,
			      NULL);
	if (IS_ERR(rx->eq)) {
		rc = PTR_ERR(rx->eq);
		goto err_unmap_eq_buf;
	}

	cxi_eq_int_disable(rx->eq);

	/* Create an Ethernet portal */
	rx->pt = cxi_pte_alloc(dev->lni, rx->eq, &pt_alloc_opts);
	if (IS_ERR(rx->pt)) {
		rc = PTR_ERR(rx->pt);
		goto err_free_eq;
	}

	NETIF_NAPI_ADD(dev->ndev, &rx->napi, cxi_rx_eth_poll);

	return 0;

err_free_eq:
	cxi_eq_free(rx->eq);
err_unmap_eq_buf:
	if (is_vmalloc_addr(rx->eq_buf))
		cxi_unmap(rx->eq_md);
err_free_eq_buf:
	kvfree(rx->eq_buf);
err_free_cq:
	cxi_cq_free(rx->cq_tgt_prio);
err_free_buffers:
	free_rx_buffers(rx);
err_free_rx_bufs:
	vfree(rx->rx_bufs);
err:
	rx->dev = NULL;
	return rc;
}

void free_rx_queue(struct rx_queue *rx)
{
	if (!rx->dev)
		return;

	netif_napi_del(&rx->napi);

	cxi_pte_free(rx->pt);
	rx->pt = NULL;
	cxi_eq_free(rx->eq);
	rx->eq = NULL;

	if (is_vmalloc_addr(rx->eq_buf))
		cxi_unmap(rx->eq_md);
	kvfree(rx->eq_buf);
	rx->eq_md = NULL;
	rx->eq_buf = NULL;

	cxi_cq_free(rx->cq_tgt_prio);
	rx->cq_tgt_prio = NULL;
	free_rx_buffers(rx);
	vfree(rx->rx_bufs);
	rx->rx_bufs = NULL;
	rx->dev = NULL;
}

void enable_rx_queue(struct rx_queue *rx)
{
	napi_enable(&rx->napi);
	cxi_eq_int_enable(rx->eq);
}

void disable_rx_queue(struct rx_queue *rx)
{
	if (!rx->dev)
		return;

	cxi_eq_int_disable(rx->eq);
	napi_disable(&rx->napi);
}

#define DEFAULT_RESERVED_LES 1024U

static unsigned int get_reserved_les(void)
{
	int reserved_les;
	int max_rx_queues = max_rss_queues + 1; // +1 for PTP queue
	int max_les_per_queue;

	reserved_les = cxi_get_lpe_append_credits(lpe_cdt_thresh_id);
	if (reserved_les < 0) {
		pr_err("Failed to get LPE append credits. Used default %u\n",
		       DEFAULT_RESERVED_LES);
		return DEFAULT_RESERVED_LES;
	}

	max_les_per_queue = reserved_les + small_pkts_buf_count;

	/* Each RX queue needs max_les_per_queue. Since the CXI service reserves
	 * LEs against each LPE pool (4 pools total), the total number of LEs
	 * needed needs to be divided by 4. cxi_pte_alloc() will then round
	 * robind across the LPE pools.
	 */
	reserved_les = max_les_per_queue * (max_rx_queues / C_PE_COUNT);

	/* Need to account if the number of queues is not a factor of 4. */
	if (max_rx_queues % C_PE_COUNT)
		reserved_les += max_les_per_queue;

	return reserved_les;
}

/* Allocate the hardware resources and some receive buffers */
int hw_setup(struct cxi_eth *dev)
{
	int rc;
	int lac;
	struct net_device *ndev = dev->ndev;
	const struct cxi_rsrc_limits limits = {
		.acs = {
			.max = 1,
			.res = 1,
		},
		.eqs = {
			.max = 1 + max_rss_queues + (max_tx_queues * 2),
			.res = 1,
		},
		.cts = {
			.max = 0,
			.res = 0,
		},
		.ptes = {
			.max = 1 + max_rss_queues,
			.res = 8,
		},
		.txqs = {
			.max = max_tx_queues * 2,
			.res = 1,
		},
		.tgqs = {
			.max = 1 + 1 + max_rss_queues,
			.res = 1,
		},
		.tles = {
			.max = 0,
			.res = 0,
		},
		/* Will actually have 4 * res value for les */
		.les = {
			.max = MAX_LE_LIMIT,
			.res = get_reserved_les(),
		},
	};
	struct cxi_svc_desc svc_desc = {
		.resource_limits = true,
		.limits = limits,
		.is_system_svc = true,
		.restricted_members = true,
		.restricted_vnis = true,
		.num_vld_vnis = 0,
		.enable = true,
		.members[0] = {
			.type = CXI_SVC_MEMBER_UID,
			.svc_member.uid = (current_euid()).val,
		},
	};
	struct cxi_cq_alloc_opts cq_alloc_opts = {};
	u8 shared_cp_pcp;

	/* Allocate a Service */
	rc = cxi_svc_alloc(dev->cxi_dev, &svc_desc, NULL, "ethernet-svc");
	if (rc < 0) {
		netdev_info(ndev, "Can't reserve resources: %d\n", rc);
		goto err;
	}
	dev->svc_id = rc;

	dev->lni = cxi_lni_alloc(dev->cxi_dev, dev->svc_id);
	if (IS_ERR(dev->lni)) {
		rc = PTR_ERR(dev->lni);
		netdev_info(ndev, "Can't get an LNI: %d\n", rc);
		goto err_free_svc;
	}

	lac = cxi_phys_lac_alloc(dev->lni);
	if (lac < 0) {
		rc = -EINVAL;
		netdev_info(ndev, "failed to get a physical LAC: %d\n", rc);
		goto err_free_ni;
	}
	dev->phys_lac = lac;

	/* Store PCPs for all the Eth Classes */
	dev->eth1_pcp = cxi_get_tc_req_pcp(dev->cxi_dev, CXI_ETH_TC1);
	if (dev->eth1_pcp < 0) {
		rc = -EINVAL;
		goto err_free_ni;
	}

	/* Determine if Eth2 is active. Eth1 and Shared will always be active */
	dev->eth2_pcp = cxi_get_tc_req_pcp(dev->cxi_dev, CXI_ETH_TC2);
	dev->eth2_active = (dev->eth2_pcp >= 0 && dev->eth2_pcp < 8);

	/* The Shared Ethernet Class is configured with all PCPs apart from
	 * the ones used by Eth1 and Eth2. Use any of them for configuring
	 * the CP
	 */
	for (shared_cp_pcp = 0; shared_cp_pcp <= 7; shared_cp_pcp++) {
		if (shared_cp_pcp != dev->eth1_pcp &&
		    (!dev->eth2_active || shared_cp_pcp != dev->eth2_pcp))
			break;
	}

	/* Allocate CPs for each Eth TC
	 * TODO: Support unique communication profiles + CQ + MCUs for each
	 * PCP. This may not be feasible with the resource needed for each.
	 */
	dev->eth1_cp = cxi_cp_alloc(dev->lni, dev->eth1_pcp,
				    CXI_ETH_TC1, CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(dev->eth1_cp)) {
		rc = PTR_ERR(dev->eth1_cp);
		netdev_info(ndev, "Can't allocate Eth1 CP: %d\n", rc);
		goto err_free_lac;
	}

	dev->shared_cp = cxi_cp_alloc(dev->lni, shared_cp_pcp, CXI_ETH_SHARED,
				      CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(dev->shared_cp)) {
		rc = PTR_ERR(dev->shared_cp);
		netdev_info(ndev, "Can't allocate Eth shared CP: %d\n", rc);
		goto err_free_eth1_cp;
	}

	if (dev->eth2_active) {
		dev->eth2_cp = cxi_cp_alloc(dev->lni, dev->eth2_pcp,
					    CXI_ETH_TC2, CXI_TC_TYPE_DEFAULT);
		if (IS_ERR(dev->eth2_cp)) {
			rc = PTR_ERR(dev->eth2_cp);
			netdev_info(ndev, "Can't allocate Eth2 CP: %d\n", rc);
			goto err_free_shared_cp;
		}
	}


	dev->err_eq_attr.queue_len = PAGE_SIZE;
	dev->err_eq_attr.queue = (void *)
		__get_free_pages(GFP_KERNEL | __GFP_ZERO,
				 get_order(dev->err_eq_attr.queue_len));
	if (!dev->err_eq_attr.queue) {
		rc = -ENOMEM;
		netdev_info(ndev, "Can't allocate the EQ buffer: %d\n", rc);
		goto err_free_eth2_cp;
	}

	dev->err_eq_attr.flags = CXI_EQ_PASSTHROUGH;
	dev->err_eq = cxi_eq_alloc(dev->lni, NULL, &dev->err_eq_attr, err_eq_cb,
				   dev, NULL, NULL);
	if (IS_ERR(dev->err_eq)) {
		rc = PTR_ERR(dev->err_eq);
		netdev_info(ndev, "Can't allocate the EQ: %d\n", rc);
		goto err_free_err_eq_buf;
	}

	cq_alloc_opts.policy = CXI_CQ_UPDATE_HIGH_FREQ_EMPTY;
	cq_alloc_opts.count =
		(small_pkts_buf_count * (max_rss_queues + 1)) +	TGT_CQ_OVERHEAD_COUNT;
	dev->cq_tgt_req = cxi_cq_alloc(dev->lni, dev->err_eq, &cq_alloc_opts,
				       dev_to_node(&dev->cxi_dev->pdev->dev));
	if (IS_ERR(dev->cq_tgt_req)) {
		rc = PTR_ERR(dev->cq_tgt_req);
		netdev_info(ndev, "Can't allocate the target CQ: %d\n", rc);
		goto err_free_err_eq;
	}

	/* Configure RXQ zero used for default/catch-all traffic. */
	rc = alloc_rx_queue(dev, 0);
	if (rc) {
		netdev_info(ndev, "Can't allocate the receive queue: %d\n", rc);
		goto err_free_cq_tgt_req;
	}
	dev->res.ptn_def = dev->rxqs[0].pt->id;

	rc = post_rx_buffers(&dev->rxqs[0], GFP_KERNEL);
	if (rc < 0) {
		netdev_info(ndev, "Cannot post any RX buffers: %d\n", rc);
		goto err_free_rx_queue;
	}

	/* Configure PTP RX queue */
	rc = alloc_rx_queue(dev, PTP_RX_Q);
	if (rc) {
		netdev_info(ndev, "Can't allocate the PTP receive queue: %d\n",
			    rc);
		goto err_free_rx_queue;
	}
	dev->res.ptn_ptp = dev->rxqs[PTP_RX_Q].pt->id;
	rc = post_rx_buffers(&dev->rxqs[PTP_RX_Q], GFP_KERNEL);
	if (rc < 0) {
		netdev_info(ndev, "Cannot post RX buffers: %d\n", rc);
		goto err_free_ptp_queue;
	}

	rc = alloc_tx_queue(dev, 0);
	if (rc) {
		netdev_info(ndev, "Can't allocate the transmit queue: %d\n",
			    rc);
		goto err_free_ptp_queue;
	}

	dev->mac_addr = ether_addr_to_u64(ndev->dev_addr);
	dev->ptp_mac_addr = PTP_L2_MAC;

	cxi_eth_add_mac(dev->cxi_dev, &dev->res, dev->mac_addr,	false);

	enable_rx_queue(&dev->rxqs[0]);
	enable_rx_queue(&dev->rxqs[PTP_RX_Q]);
	enable_tx_queue(&dev->txqs[0]);

	dev->res.rss_queues = 1;
	rc = cxi_set_rx_channels(dev, dev->ndev->real_num_rx_queues);
	if (rc) {
		netdev_info(ndev,
			    "Can't set the number of RX channels to %d: %d\n",
			    dev->ndev->real_num_rx_queues, rc);
		goto err_disable_queues;
	}

	dev->cur_txqs = 1;
	rc = cxi_set_tx_channels(dev, dev->ndev->real_num_tx_queues);
	if (rc) {
		netdev_info(ndev,
			    "Can't set the number of TX channels to %d: %d\n",
			    dev->ndev->real_num_tx_queues, rc);
		goto err_set_rx_queues;
	}

	cxi_eth_set_rx_mode(ndev);
	netif_tx_start_all_queues(ndev);
	dev->is_active = true;

	return 0;

err_set_rx_queues:
	cxi_set_rx_channels(dev, 1);
err_disable_queues:
	disable_tx_queue(&dev->txqs[0]);
	disable_rx_queue(&dev->rxqs[PTP_RX_Q]);
	disable_rx_queue(&dev->rxqs[0]);
	cxi_eth_set_list_invalidate_all(dev->cxi_dev, &dev->res);
	free_tx_queue(&dev->txqs[0]);
err_free_ptp_queue:
	free_rx_queue(&dev->rxqs[PTP_RX_Q]);
err_free_rx_queue:
	free_rx_queue(&dev->rxqs[0]);
err_free_cq_tgt_req:
	cxi_cq_free(dev->cq_tgt_req);
err_free_err_eq:
	cxi_eq_free(dev->err_eq);
err_free_err_eq_buf:
	free_pages((unsigned long)dev->err_eq_attr.queue,
		   get_order(dev->err_eq_attr.queue_len));
err_free_eth2_cp:
	if (dev->eth2_active)
		cxi_cp_free(dev->eth2_cp);
err_free_shared_cp:
	cxi_cp_free(dev->shared_cp);
err_free_eth1_cp:
	cxi_cp_free(dev->eth1_cp);
err_free_lac:
	cxi_phys_lac_free(dev->lni, dev->phys_lac);
err_free_ni:
	cxi_lni_free(dev->lni);
err_free_svc:
	cxi_svc_destroy(dev->cxi_dev, dev->svc_id);
err:
	return rc;
}

/* Release hardware resources belonging to a device. */
void hw_cleanup(struct cxi_eth *dev)
{
	int i;

	if (!dev->is_active)
		return;

	netif_tx_stop_all_queues(dev->ndev);

	dev->is_active = false;

	for (i = 0; i < dev->cur_txqs; i++)
		disable_tx_queue(&dev->txqs[i]);

	/* Disable RSS support and free RX queues. */
	cxi_eth_clear_indir_table(dev->cxi_dev, &dev->res);
	for (i = 0; i < dev->res.rss_queues; i++)
		disable_rx_queue(&dev->rxqs[i]);
	disable_rx_queue(&dev->rxqs[PTP_RX_Q]);

	cxi_eth_set_list_invalidate_all(dev->cxi_dev, &dev->res);

	for (i = 0; i < dev->cur_txqs; i++)
		free_tx_queue(&dev->txqs[i]);

	for (i = 0; i < dev->res.rss_queues; i++)
		free_rx_queue(&dev->rxqs[i]);
	free_rx_queue(&dev->rxqs[PTP_RX_Q]);

	cxi_cq_free(dev->cq_tgt_req);
	dev->cq_tgt_req = NULL;
	cxi_eq_free(dev->err_eq);
	free_pages((unsigned long)dev->err_eq_attr.queue,
		   get_order(dev->err_eq_attr.queue_len));
	dev->err_eq = NULL;
	if (dev->eth2_active) {
		cxi_cp_free(dev->eth2_cp);
		dev->eth2_cp = NULL;
	}
	cxi_cp_free(dev->shared_cp);
	dev->shared_cp = NULL;
	cxi_cp_free(dev->eth1_cp);
	dev->eth1_cp = NULL;
	cxi_phys_lac_free(dev->lni, dev->phys_lac);
	dev->phys_lac = 0;
	cxi_lni_free(dev->lni);
	dev->lni = NULL;
	cxi_svc_destroy(dev->cxi_dev, dev->svc_id);
}

static struct sk_buff *eth_rx_copy(struct rx_queue *rx,
				   struct rx_buffer *buf,
				   const struct c_event_target_enet *event)
{
	struct cxi_eth *dev = rx->dev;
	struct net_device *ndev = dev->ndev;
	struct sk_buff *skb;
	struct page *page;
	unsigned char *start;

	start = (void *)CXI_IOVA_TO_VA(buf->md, event->start);

	if (likely(rx->frag_state == FRAG_NONE && !event->more_frags)) {
		/* single fragment */
		skb = napi_alloc_skb(&rx->napi, event->length);
		if (unlikely(!skb))
			return NULL;

		__skb_put_data(skb, start, event->length);

		skb->protocol = eth_type_trans(skb, ndev);
	} else {
		skb = napi_get_frags(&rx->napi);
		if (unlikely(!skb))
			return NULL;

		/* The SKB may already be full, due to the adapter
		 * being configured to fragment too much.
		 */
		if (unlikely(skb_shinfo(skb)->nr_frags == MAX_SKB_FRAGS))
			return NULL;

		if (rx->frag_state == FRAG_NONE &&
		    event->length <= skb_availroom(skb)) {
			__skb_put_data(skb, start, event->length);
		} else {
			page = alloc_page(GFP_ATOMIC);
			if (unlikely(!page)) {
				napi_free_frags(&rx->napi);
				return NULL;
			}

			memcpy(page_address(page), start, event->length);

			skb_add_rx_frag(skb, skb_shinfo(skb)->nr_frags,
					page, 0, event->length, PAGE_SIZE);
		}
	}

	return skb;
}

static struct sk_buff *eth_rx_frags(struct rx_queue *rx,
				    struct rx_buffer *buf,
				    const struct c_event_target_enet *event)
{
	struct cxi_eth *eth_dev = rx->dev;
	struct device *dev = &eth_dev->cxi_dev->pdev->dev;
	struct sk_buff *skb;
	struct page *page;
	int frag_len;

	skb = napi_get_frags(&rx->napi);
	if (unlikely(!skb))
		return NULL;

	/* The SKB may already be full, due to the adapter
	 * being configured to fragment too much.
	 */
	if (unlikely(skb_shinfo(skb)->nr_frags == MAX_SKB_FRAGS))
		return NULL;

	page = buf->page;
	frag_len = event->length;

	if (rx->frag_state == FRAG_NONE)
		prefetch(page_address(page));

	BUG_ON(frag_len > PAGE_SIZE);

	skb_add_rx_frag(skb, skb_shinfo(skb)->nr_frags,
			page, buf->cur_offset, frag_len,
			eth_dev->eth_info.max_segment_size);
	dma_sync_single_for_cpu(dev, buf->mapping + buf->cur_offset, frag_len,
				DMA_FROM_DEVICE);

	buf->cur_offset += eth_dev->eth_info.max_segment_size;
	buf->posted_count--;

	if (buf->posted_count == 0) {
		dma_unmap_page(dev, dma_unmap_addr(buf, mapping), PAGE_SIZE,
			DMA_FROM_DEVICE);
		put_page(page);
		buf->page = NULL;
	}

	return skb;
}

/* Check whether the RoCE packet in the SKB has been segmented by
 * hardware, which also removed its iCRC. Fix the headers if that
 * happened.
 */
static void check_segmented_roce(struct sk_buff *skb)
{
	int frag_len;
	u16 protocol;
	struct ethhdr *ethhdr;
	size_t vhdr_size = 0;

	/* heuristic to figure out whether Cassini stripped the iCRC
	 * from a RoCE packet, following its segmentation.
	 */
	if (skb_shinfo(skb)->nr_frags != 1)
		return;

	ethhdr = (void *)skb->data;
	protocol = be16_to_cpu(ethhdr->h_proto);

	/* Check for vlan header - we will support at most one vlan header,
	 * no nested headers, ipv6 extensions, etc.
	 */
	if (eth_type_vlan(ethhdr->h_proto)) {
		struct vlan_ethhdr *veth;

		veth = (struct vlan_ethhdr *)skb->data;
		protocol = be16_to_cpu(veth->h_vlan_encapsulated_proto);
		vhdr_size = sizeof(struct vlan_hdr);
	}

	/* Check length. ETH+IPv4/6+UDP+BTH headers. */
	if (skb_headlen(skb) < (54 + vhdr_size) || skb_headlen(skb) > (90 + vhdr_size))
		return;

	frag_len = skb_frag_size(&skb_shinfo(skb)->frags[0]);
	if (frag_len != 1024 && frag_len != 2048 && frag_len != 4096)
		return;

	/* Adjust the IP packet by removing the 4 bytes and
	 * recomputing the IP header checksum.
	 */
	switch (protocol) {
	case ETH_P_IP: {
		struct udphdr *udph;
		struct iphdr *iph;

		iph = (void *)skb->data + sizeof(struct ethhdr) + vhdr_size;
		iph->tot_len = cpu_to_be16(be16_to_cpu(iph->tot_len) - 4);
		iph->check = 0;
		iph->check = ip_fast_csum((const void *)iph, iph->ihl);

		udph = (void *)(iph + 1);
		udph->len = cpu_to_be16(be16_to_cpu(udph->len) - 4);
	}
		break;

	case ETH_P_IPV6: {
		struct udphdr *udph;
		struct ipv6hdr *ip6h;

		ip6h = (void *)skb->data + sizeof(struct ethhdr) + vhdr_size;
		ip6h->payload_len =
			cpu_to_be16(be16_to_cpu(ip6h->payload_len) - 4);

		if (ip6h->nexthdr != NEXTHDR_UDP) {
			netdev_warn_once(skb->dev,
					 "Unexpected ipv6 packet format (nexthdr = %d)\n",
					 ip6h->nexthdr);
			return;
		}

		udph = (void *)(ip6h + 1);
		udph->len = cpu_to_be16(be16_to_cpu(udph->len) - 4);
	}
		break;

	default:
		netdev_warn_once(skb->dev,
				 "unexpected protocol in RoCE packet: %x\n",
				 protocol);
		return;
	}

	/* TODO. For now, use a bit in the SKB to tell RXE that
	 * the iCRC is gone, and that the packet is otherwise correct.
	 */
	cxi_skb_icrc_gone(skb);
}

/* Convert from hash type (i.e. C_RSS_HASH_IPV4_UDP) to linux hash
 * type (i.e. PKT_HASH_TYPE_L4). Only valid if the hash type is not
 * C_RSS_HASH_NONE.
 */
static const u8 rss_hash_type_to_level[C_RSS_HASH_IPV6_FLOW_LABEL + 1] = {
	[C_RSS_HASH_IPV4] = PKT_HASH_TYPE_L3,
	[C_RSS_HASH_IPV4_PROTOCOL] = PKT_HASH_TYPE_L3,
	[C_RSS_HASH_IPV4_FLOW_LABEL] = PKT_HASH_TYPE_L3,
	[C_RSS_HASH_IPV6] = PKT_HASH_TYPE_L3,
	[C_RSS_HASH_IPV6_PROTOCOL] = PKT_HASH_TYPE_L3,
	[C_RSS_HASH_IPV6_FLOW_LABEL] = PKT_HASH_TYPE_L3,

	[C_RSS_HASH_IPV4_TCP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV4_UDP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV4_PROTOCOL_TCP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV4_PROTOCOL_UDP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV4_PROTOCOL_UDP_ROCE] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV6_TCP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV6_UDP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV6_PROTOCOL_TCP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV6_PROTOCOL_UDP] = PKT_HASH_TYPE_L4,
	[C_RSS_HASH_IPV6_PROTOCOL_UDP_ROCE] = PKT_HASH_TYPE_L4,
};

static void set_rss_hash_value(const struct c_event_target_enet *event,
			       struct sk_buff *skb)
{
	if (event->rss_hash_type == C_RSS_HASH_NONE ||
	    event->rss_hash_type > C_RSS_HASH_IPV6_FLOW_LABEL)
		return;

	skb_set_hash(skb, event->rss_hash_value,
		     rss_hash_type_to_level[event->rss_hash_type]);
}

/* Process a received packet */
static bool eth_receive(struct rx_queue *rx,
			const struct c_event_target_enet *event)
{
	struct cxi_eth *dev = rx->dev;
	struct net_device *ndev = dev->ndev;
	struct sk_buff *skb = NULL;
	struct rx_buffer *buf;
	bool packet_complete = false;

	/* Find the target buffer */
	if (event->ptlte_index != rx->pt->id) {
		netdev_err(ndev, "Unknown portal target in RX event (%d instead of %d)\n",
			   event->ptlte_index, rx->pt->id);
		return packet_complete;
	}

	buf = &rx->rx_bufs[event->buffer_id];

	if (unlikely(event->return_code != C_RC_OK ||
		     (event->is_roce && !event->more_frags &&
		      !event->roce_icrc_ok) ||
		     event->parser_err ||
		     rx->frag_state == FRAG_DROP)) {
		goto drop;
	}

	if (event->ptl_list == C_PTL_LIST_PRIORITY)
		skb = eth_rx_frags(rx, buf, event);
	else
		skb = eth_rx_copy(rx, buf, event);

	if (unlikely(!skb))
		goto drop;

	if (dev->is_c2) {
		if (event->seg_cnt == 0) {
			/* It's a new packet */
			rx->pkt_cnt = event->pkt_cnt;
			rx->next_seg_cnt = 1;

			/* Free existing incomplete segments if any */
			if (rx->frag_state != FRAG_NONE) {
				ndev->stats.rx_dropped++;
				rx->frag_state = FRAG_NONE;
				napi_free_frags(&rx->napi);
			}
		} else if (event->pkt_cnt != rx->pkt_cnt ||
			   event->seg_cnt != rx->next_seg_cnt) {
			goto drop;
		} else {
			rx->next_seg_cnt++;
		}
	}

	if (event->more_frags == 0) {
		__be16 csum;

		ndev->stats.rx_packets++;
		ndev->stats.rx_bytes += event->length;

		csum = (__force __be16)event->checksum;
		skb->csum = (__force __wsum)be16_to_cpu(csum);
		skb->ip_summed = CHECKSUM_COMPLETE;

		/* Cassini ERRATA-3258. This should test event->timestamp, but
		 * Cassini 1/1.1 doesn't set it.
		 */
		if (dev->ptp_ts_enabled &&
		    event->ptlte_index == dev->res.ptn_ptp) {
			struct c_ts ts;
			int rc;

			rc = skb_copy_bits(skb, skb->len - sizeof(ts),
					   &ts, sizeof(ts));
			if (rc)
				goto drop;

			/* Cassini ERRATA-3276. Ignore the timestamp on Cassini 1
			 * if ns is close to 1s, as RX timestamp can
			 * be off by one second in that case. No issue
			 * on Cassini 2.
			 */
			if (dev->is_c2 || ts.ns < 999999700)
				skb_hwtstamps(skb)->hwtstamp =
					ktime_set(ts.sec, ts.ns);

			pskb_trim(skb, skb->len - sizeof(ts));
		}

		if (event->is_roce) {
			if (skb->len <= 64 && !dev->is_c2 &&
			    (skb_shinfo(skb)->nr_frags == 0))
				/* Due to a rarely occurring hardware
				 * bug, iCRC on packets smaller than
				 * 65 bytes may be passed although
				 * they are corrupted.  Tell the rxe
				 * driver to check it again.
				 */
				cxi_force_icrc_check(skb);
			else if (dev->priv_flags & CXI_ETH_PF_ROCE_OPT)
				check_segmented_roce(skb);
		}

		skb_record_rx_queue(skb, rx->id);

		if (ndev->features & NETIF_F_RXHASH)
			set_rss_hash_value(event, skb);

		if (skb_is_nonlinear(skb)) {
			napi_gro_frags(&rx->napi);

			rx->frag_state = FRAG_NONE;
		} else {
			napi_gro_receive(&rx->napi, skb);
		}

		packet_complete = true;
	} else {
		rx->frag_state = FRAG_MORE;
	}

requeue:
	if (event->ptl_list == C_PTL_LIST_PRIORITY) {
		rx->stats.unlinked_prio++;

		if (buf->page == NULL) {
			list_del(&buf->buf_list);
			list_add_tail(&buf->buf_list, &rx->rx_ready);
		}
	} else if (event->auto_unlinked) {
		rx->stats.unlinked_req++;
		list_del(&buf->buf_list);
		list_add_tail(&buf->buf_list, &rx->rx_ready);
	}

	return packet_complete;

drop:
	/* Error on receive, NIC dropped this frag, or driver dropped a previous
	 * frag, or skb allocation failure
	 */
	netdev_dbg(dev->ndev,
		   "received return_code=%d more=%d state=%d ptl_list=%d roce=%d\n",
		   event->return_code, event->more_frags, rx->frag_state,
		   event->ptl_list, event->is_roce);

	if (event->return_code != C_RC_OK)
		rx->last_bad_frag_drop_rc = event->return_code;

	/* Not always a whole packet, so in theory a single packet
	 * could be counted several times, one for each segment.
	 */
	ndev->stats.rx_dropped++;

	if (rx->frag_state != FRAG_DROP)
		napi_free_frags(&rx->napi);

	if (event->more_frags)
		rx->frag_state = FRAG_DROP;
	else
		rx->frag_state = FRAG_NONE;

	goto requeue;
}

/* RX queue interrupt handler. Disable the interrupts for that EQ and
 * trigger NAPI.
 */
static void rx_eq_cb(void *context)
{
	struct rx_queue *rx = context;

	rx->time_napi_schedule = ktime_get_raw();
	napi_schedule(&rx->napi);
}

static void add_ktime_to_bucket(struct bucket *bucket, u64 ns)
{
	int i;

	if (ns > bucket->max_ts)
		bucket->max_ts = ns;
	if (ns < bucket->min_ts)
		bucket->min_ts = ns;

	i = fls(ns + 1);
	if (i >= NB_BUCKETS)
		i = NB_BUCKETS - 1;

	bucket->b[i]++;
}

#define RX_ETH_EQ_UNACKED_MAX 16U

/* NAPI poll function.
 * Only the receive event will be counted towards the budget.
 */
static int cxi_rx_eth_poll(struct napi_struct *napi, int budget)
{
	struct rx_queue *rx = container_of(napi, struct rx_queue, napi);
	struct cxi_eth *dev = rx->dev;
	const union c_event *event;
	int work_done = 0;
	int event_count = 0;
	bool reschedule = false;
	bool packet_processed = false;
	int rc;

	if (rx->time_napi_schedule) {
		ktime_t enter_ts;
		u64 time_exec;

		enter_ts = ktime_get_raw();
		time_exec =
			ktime_to_ns(ktime_get_raw()) - ktime_to_ns(enter_ts);
		add_ktime_to_bucket(&rx->eth_napi_schedule, time_exec);

		/* Prevent adding time when the function is re-entered */
		rx->time_napi_schedule = 0;
	}

	while (work_done < budget &&
	       (event = cxi_eq_get_event(rx->eq))) {
		switch (event->hdr.event_type) {
		case C_EVENT_LINK:
			packet_processed = false;
			netdev_warn(dev->ndev, "LE append failure: %d\n",
				    event->tgt_long.return_code);
			break;
		case C_EVENT_ETHERNET:
			packet_processed = eth_receive(rx, &event->enet);
			if (packet_processed)
				work_done++;
			break;
		case C_EVENT_STATE_CHANGE:
			packet_processed = false;
			netdev_warn(dev->ndev, "State change (%d, %d, %d)\n",
			       event->tgt_long.ptlte_index,
			       event->tgt_long.initiator.state_change.ptlte_state,
			       event->tgt_long.return_code);
			break;
		default:
			packet_processed = false;
			netdev_warn(dev->ndev, "Unhandled event %d, rc %d\n",
				    event->hdr.event_type,
				    event->hdr.return_code);
			print_hex_dump(KERN_INFO,
				       "cxi eth unexpected target event: ",
				       DUMP_PREFIX_ADDRESS, 16, 1, event,
				       (8 << event->hdr.event_size), false);
		}

		event_count++;
		if ((event_count >= RX_ETH_EQ_UNACKED_MAX) &&
		    packet_processed) {
			cxi_eq_int_disable(rx->eq);
			rc = post_rx_buffers(rx, GFP_ATOMIC);
			if (rc < 0)
				reschedule = true;

			event_count = 0;
		}
	}

	BUG_ON(work_done > budget);

	/* ACK all EQ space before appending new buffers. */
	if (event_count)
		cxi_eq_int_disable(rx->eq);

	if (!list_empty(&rx->rx_ready)) {
		rc = post_rx_buffers(rx, GFP_ATOMIC);
		if (rc < 0)
			reschedule = true;
	}

	if (work_done < budget && napi_complete_done(napi, work_done)) {
		/* Only re-enable interrupts if NAPI is not going to be
		 * rescheduled.
		 */
		if (reschedule) {
			netdev_warn_once(dev->ndev,
					 "Forcing NAPI reschedule\n");
#if (defined(RHEL_MAJOR) && ((RHEL_MAJOR == 8 && RHEL_MINOR >= 10) || \
			     (RHEL_MAJOR == 9 && RHEL_MINOR >= 4))) || \
	LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
			napi_schedule(&rx->napi);
#else
			napi_reschedule(&rx->napi);
#endif
		} else {
			cxi_eq_int_enable(rx->eq);
		}
	}

	return work_done;
}

/* Hardware defined RSS hash bit is one less than the corresponding
 * c_rss_hash_type enum value. This macro should only be used if the
 * c_rss_hash_type enum value is not C_RSS_HASH_NONE.
 */
#define RSS_HASH_BIT(hash_type) ((hash_type) - 1)

int cxi_eth_open(struct net_device *ndev)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct hwtstamp_config ts_cfg = {
		.flags = 0,
		.tx_type = HWTSTAMP_TX_OFF,
		.rx_filter = HWTSTAMP_FILTER_NONE,
	};

	cxi_set_eth_name(dev->cxi_dev, netdev_name(ndev));

	/* The indirection base is always 0 for now, as there is only
	 * one Ethernet device supported. The driver will need an
	 * allocation scheme when VFs support Ethernet.
	 */
	dev->res.portal_index_indir_base = 0;

	/* Enable some default hash type.
	 * TODO: remove when ethtool supports them.
	 */
	dev->res.hash_types_enabled =
		BIT(RSS_HASH_BIT(C_RSS_HASH_IPV4_TCP)) |
		BIT(RSS_HASH_BIT(C_RSS_HASH_IPV4_UDP)) |
		BIT(RSS_HASH_BIT(C_RSS_HASH_IPV4_PROTOCOL_UDP_ROCE)) |
		BIT(RSS_HASH_BIT(C_RSS_HASH_IPV6_TCP)) |
		BIT(RSS_HASH_BIT(C_RSS_HASH_IPV6_UDP)) |
		BIT(RSS_HASH_BIT(C_RSS_HASH_IPV6_PROTOCOL_UDP_ROCE));

	cxi_eth_cfg_timestamp(dev->cxi_dev, &ts_cfg);

	return hw_setup(dev);
}

int cxi_eth_close(struct net_device *ndev)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	hw_cleanup(dev);

	return 0;
}

/* Get a flow hash for the commands */
static u8 get_flow_hash(struct sk_buff *skb)
{
	u32 hash = skb_get_hash(skb);

	/* XOR the 4 bytes into one */
	hash ^= hash >> 16;
	hash ^= hash >> 8;

	return hash & 0xff;
}

static void cxi_eth_get_skb_checksum_settings(struct sk_buff *skb,
					      struct cxi_eth *dev,
					      enum c_checksum_ctrl *ctrl,
					      unsigned int *start,
					      unsigned int *offset)
{
	__be16 proto;

	if (skb->ip_summed != CHECKSUM_PARTIAL)
		goto no_crc;

	*start = skb_checksum_start_offset(skb) / 2;
	*offset = skb->csum_offset / 2;

	if (eth_type_vlan(skb->protocol))
		proto = vlan_get_protocol(skb);
	else
		proto = skb->protocol;

	if (proto == htons(ETH_P_IP)) {
		struct iphdr *iph = ip_hdr(skb);

		if (iph->protocol == IPPROTO_UDP) {
			if ((dev->priv_flags & CXI_ETH_PF_ROCE_OPT) &&
			    ntohs(udp_hdr(skb)->dest) == ROCE_V2_UDP_DPORT)
				*ctrl = C_CHECKSUM_CTRL_ROCE;
			else
				*ctrl = C_CHECKSUM_CTRL_UDP;
		} else if (iph->protocol == IPPROTO_TCP) {
			*ctrl = C_CHECKSUM_CTRL_TCP;
		}
	} else if (proto == htons(ETH_P_IPV6)) {
		struct ipv6hdr *iph = ipv6_hdr(skb);

		if (iph->nexthdr == IPPROTO_UDP) {
			if ((dev->priv_flags & CXI_ETH_PF_ROCE_OPT) &&
			    ntohs(udp_hdr(skb)->dest) == ROCE_V2_UDP_DPORT)
				*ctrl = C_CHECKSUM_CTRL_ROCE;
			else
				*ctrl = C_CHECKSUM_CTRL_UDP;
		} else if (iph->nexthdr == IPPROTO_TCP) {
			*ctrl = C_CHECKSUM_CTRL_TCP;
		}
	} else {
		netdev_warn_once(dev->ndev,
				 "Checksum type not supported. Protocol=%u\n",
				 skb->protocol);
		goto no_crc;
	}

	return;

no_crc:
	*ctrl = C_CHECKSUM_CTRL_NONE;
	*start = 0;
	*offset = 0;
}

/* Emit a packet using an IDC command */
static netdev_tx_t emit_idc(struct sk_buff *skb, struct tx_queue *txq,
			    struct cxi_cq *cq)
{
	struct c_idc_eth_cmd idc_eth = {
		.fmt = C_PKT_FORMAT_STD,
	};
	struct cxi_eth *dev = txq->dev;
	struct net_device *ndev = dev->ndev;
	enum c_checksum_ctrl checksum_ctrl;
	unsigned int checksum_start;
	unsigned int checksum_offset;
	int rc;

	cxi_eth_get_skb_checksum_settings(skb, dev, &checksum_ctrl,
					  &checksum_start, &checksum_offset);

	idc_eth.checksum_ctrl = checksum_ctrl;
	idc_eth.checksum_start = checksum_start;
	idc_eth.checksum_offset = checksum_offset;
	idc_eth.flow_hash = get_flow_hash(skb);

	skb_tx_timestamp(skb);

	rc = cxi_cq_emit_idc_eth(cq, &idc_eth, skb->data, skb->len);
	if (rc) {
		/* Hard error which should never happen. */
		netdev_err(dev->ndev, "TX queue %u full when queue awake\n",
			   txq->id);
		txq->stats.tx_busy++;

		return NETDEV_TX_BUSY;
	}

	txq->stats.idc++;
	txq->stats.idc_bytes += skb->len;

	ndev->stats.tx_packets++;
	ndev->stats.tx_bytes += skb->len;

	dev_kfree_skb(skb);

	return NETDEV_TX_OK;
}

/* Emit a packet using a DMA command */
static netdev_tx_t emit_dma(struct sk_buff *skb, struct tx_queue *txq,
			    struct cxi_cq *cq, unsigned int *bytes)
{
	struct cxi_eth *dev = txq->dev;
	struct device *dma_dev = &dev->cxi_dev->pdev->dev;
	struct net_device *ndev = dev->ndev;
	struct c_dma_eth_cmd dma_eth = {
		.read_lac = txq->dev->phys_lac,
		.fmt = C_PKT_FORMAT_STD,
		.checksum_ctrl = C_CHECKSUM_CTRL_NONE,
		.eq = txq->eq->eqn,
		.user_ptr = (uintptr_t)skb,
		.total_len = skb->len,
	};
	enum c_checksum_ctrl checksum_ctrl;
	unsigned int checksum_start;
	unsigned int checksum_offset;
	int rc;

	if (skb_shinfo(skb)->nr_frags >= C_MAX_ETH_FRAGS) {
		if (skb_linearize(skb) ||
		    skb_shinfo(skb)->nr_frags >= C_MAX_ETH_FRAGS) {
			netdev_warn(ndev, "skb could not be linearized\n");
			dev_kfree_skb(skb);
			ndev->stats.tx_dropped++;
			return NETDEV_TX_OK;
		}
	}

	if (prepare_frags(dma_dev, skb, &dma_eth)) {
		netdev_warn(ndev, "skb could not be mapped for DMA\n");
		dev_kfree_skb(skb);
		ndev->stats.tx_dropped++;
		return NETDEV_TX_OK;
	}

	cxi_eth_get_skb_checksum_settings(skb, dev, &checksum_ctrl,
					  &checksum_start, &checksum_offset);

	dma_eth.checksum_ctrl = checksum_ctrl;
	dma_eth.checksum_start = checksum_start;
	dma_eth.checksum_offset = checksum_offset;
	dma_eth.flow_hash = get_flow_hash(skb);

	skb_tx_timestamp(skb);

	rc = cxi_cq_emit_dma_eth(cq, &dma_eth);
	if (rc) {
		/* Hard error which should never happen. */
		netdev_err(dev->ndev, "TX queue %u full when queue awake\n",
			   txq->id);
		txq->stats.tx_busy++;

		unmap_frags(dma_dev, skb);

		return NETDEV_TX_BUSY;
	}

	atomic_dec(&txq->dma_eq_cdt);
	txq->stats.dma++;
	txq->stats.dma_bytes += skb->len;

	*bytes += skb->len;
	return NETDEV_TX_OK;
}

/* Check whether an outgoing packet can be timestamped, and reserve
 * the unique slot if it can.
 */
static bool tstamp_xmit(struct cxi_eth *dev, struct sk_buff *skb)
{
	struct ethhdr *ethhdr;

	if (likely(!(skb_shinfo(skb)->tx_flags & SKBTX_HW_TSTAMP)))
		return false;

	/* Only Ethernet PTP packets are supported. That's what was
	 * programmed in cxi_eth_cfg_timestamp().
	 */
	ethhdr = (struct ethhdr *)skb->data;
	if (ethhdr->h_proto != htons(PTP_L2_ETHERTYPE))
		return false;

	if (cmpxchg(&dev->tstamp_skb, NULL, skb))
		return false;

	skb_shinfo(skb)->tx_flags |= SKBTX_IN_PROGRESS;

	return true;
}

static void ring_txq(struct tx_queue *txq)
{
	cxi_cq_ring(txq->eth1_cq);
	if (txq->eth2_cq)
		cxi_cq_ring(txq->eth2_cq);
	cxi_cq_ring(txq->shared_cq);
}

netdev_tx_t cxi_eth_start_xmit(struct sk_buff *skb, struct net_device *ndev)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	unsigned int qid = skb_get_queue_mapping(skb);
	struct tx_queue *txq = &dev->txqs[qid];
	struct netdev_queue *dev_queue =
		netdev_get_tx_queue(dev->ndev, txq->id);
	unsigned int bytes = 0;
	netdev_tx_t tx_status;
	bool kick_cq;
#if KERNEL_VERSION(4, 18, 0) > LINUX_VERSION_CODE
	bool more = skb->xmit_more;
#else
	bool more = netdev_xmit_more();
#endif
	struct cxi_cq *cq;

	if (eth_type_vlan(skb->protocol)) {
		/* Tagged traffic. Check PCPs to steer appropriately */
		struct vlan_ethhdr *veth;
		unsigned int pcp;

		veth = (struct vlan_ethhdr *)vlan_eth_hdr(skb);
		pcp = (be16_to_cpu(veth->h_vlan_TCI) & VLAN_PRIO_MASK) >> VLAN_PRIO_SHIFT;

		if (pcp == dev->cxi_dev->untagged_eth_pcp)
			cq = txq->untagged_cq;
		else if (pcp == dev->eth1_pcp)
			cq = txq->eth1_cq;
		else if (dev->eth2_active && pcp == dev->eth2_pcp)
			cq = txq->eth2_cq;
		else
			cq = txq->shared_cq;
	} else {
		/* Untagged traffic. */
		cq = txq->untagged_cq;
	}

	if (tstamp_xmit(dev, skb) || skb_shinfo(skb)->nr_frags ||
	    skb->len > idc_dma_threshold) {
		tx_status = emit_dma(skb, txq, cq, &bytes);
		txq->force_dma = txq->force_dma_interval;
	} else if (--txq->force_dma == 0) {
		/* Too many consecutive IDC. Use DMA instead */
		tx_status = emit_dma(skb, txq, cq, &bytes);
		txq->force_dma = txq->force_dma_interval;
		txq->stats.dma_forced++;
	} else {
		tx_status = emit_idc(skb, txq, cq);
	}

	/* Kick CQ if netdev TX queue is ready. */
	kick_cq = __netdev_tx_sent_queue(dev_queue, bytes, more);
	if (kick_cq)
		ring_txq(txq);

	/* Stop TX queue if not enough free slots to emit another SKB
	 * or NETDEV_TX_BUSY is set. 8 slots, aligned on an 8-slot
	 * boundary, are required to emit the largest IDC or DMA
	 * commands. TX queue will be re-enabled as space frees up. If
	 * the TX queue is to be stopped, the TX queue is always
	 * kicked in an attempt to free up space.
	 */
	if (__cxi_cq_free_slots(cq) < 16 || tx_status == NETDEV_TX_BUSY) {
		if (!kick_cq)
			ring_txq(txq);
		netif_tx_stop_queue(dev_queue);
		txq->stats.stopped++;
	} else if (atomic_read(&txq->dma_eq_cdt) <= 0) {
		/* The EQ is getting full */
		if (!kick_cq)
			ring_txq(txq);
		netif_tx_stop_queue(dev_queue);
		txq->stats.stopped_eq++;
	}

	return tx_status;
}

int cxi_eth_mac_addr(struct net_device *ndev, void *p)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	int rc;

	rc = eth_mac_addr(ndev, p);
	if (rc)
		return rc;

	dev->mac_addr = ether_addr_to_u64(ndev->dev_addr);
	ether_addr_copy(dev->cxi_dev->mac_addr, ndev->dev_addr);

	cxi_eth_set_rx_mode(ndev);

	/* We expect the NID to be configured before the MAC address is set.
	 * To be backwards compatible, we will attempt to set the NID from the MAC address.
	 * If the NID was not set, it will be derived from the MAC. Otherwise this will be ignored
	 */
	cxi_set_nid_from_mac(dev->cxi_dev, ndev->dev_addr);

	return 0;
}

/* Program MAC filters.
 *
 * Since the ethernet driver is the only thing using the MAC filter
 * (set_list), we can reset the whole list when entering this
 * function, and reprogram everything.
 */
void cxi_eth_set_rx_mode(struct net_device *ndev)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct netdev_hw_addr *ha;
	int rc;

	/* Invalidate all existing filters. */
	cxi_eth_set_list_invalidate_all(dev->cxi_dev, &dev->res);

	/* Cassini ERRATA-3258. Workaround timestamp bug. Program the PTP MAC
	 * address first so all the PTP packets will land on the PTP
	 * RX queue, even if promiscuous mode is enabled.
	 */
	cxi_eth_add_mac(dev->cxi_dev, &dev->res, dev->ptp_mac_addr, true);

	if (ndev->flags & IFF_PROMISC) {
		/* Listen on everything. */
		cxi_eth_set_promiscuous(dev->cxi_dev, &dev->res);
		return;
	}

	rc = cxi_eth_add_mac(dev->cxi_dev, &dev->res, dev->mac_addr, false);
	if (rc) {
		netdev_err(ndev, "Cannot program MAC address\n");
		return;
	}

	if (ndev->flags & IFF_BROADCAST) {
		rc = cxi_eth_add_mac(dev->cxi_dev, &dev->res,
				     0xffffffffffffULL, false);
		if (rc) {
			netdev_err(ndev, "Cannot program broadcast address\n");
			return;
		}
	}

	netdev_for_each_uc_addr(ha, ndev) {
		netdev_dbg(ndev, "Adding UC addr %pM\n", ha->addr);

		rc = cxi_eth_add_mac(dev->cxi_dev, &dev->res,
				     ether_addr_to_u64(ha->addr), false);
		if (rc) {
			netdev_err(ndev, "Cannot program unicast address\n");
			return;
		}
	}

	if (ndev->flags & IFF_ALLMULTI) {
		/* Accept all multicasts. */
		rc = cxi_eth_set_all_multi(dev->cxi_dev, &dev->res);
		if (rc) {
			netdev_err(ndev,
				   "Cannot program all multicast addresses\n");
			return;
		}
	} else if (ndev->flags & IFF_MULTICAST) {
		netdev_for_each_mc_addr(ha, ndev) {
			netdev_dbg(ndev, "Adding MC addr %pM\n", ha->addr);

			rc = cxi_eth_add_mac(dev->cxi_dev, &dev->res,
					     ether_addr_to_u64(ha->addr),
					     false);
			if (rc) {
				netdev_err(ndev,
					   "Cannot program multicast address\n");
				return;
			}
		}
	}

	cxi_eth_set_indir_table(dev->cxi_dev, &dev->res);
}

int cxi_change_mtu(struct net_device *ndev, int new_mtu)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	bool was_running;
	int rc;

	was_running = netif_running(ndev);
	if (was_running)
		hw_cleanup(dev);

	rc = cxi_set_max_eth_rxsize(dev->cxi_dev, new_mtu + VLAN_ETH_HLEN);
	if (rc)
		netdev_err(ndev, "Invalid MTU size: %u\n", new_mtu);
	else
		ndev->mtu = new_mtu;

	if (was_running) {
		int rc2;

		rc2 = hw_setup(dev);
		if (rc2) {
			netdev_err(ndev,
				   "Failed to setup resources after an MTU change\n");
			rc = rc2;
		}
	}

	return rc;
}
