#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_pke_rta.h"
#include "rdm/efa_rdm_pke_rtw.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_pke_cmd.h"


/**
 * @brief Test that efa_rdm_pke_handle_send_completion does not crash
 * when peer has been removed (use-after-free fix).
 *
 * Scenario: A pkt_entry is submitted, then the peer is removed via
 * fi_av_remove (which calls peer_destruct, setting pkt_entry->peer = NULL
 * and releasing the txe). When the send completion arrives later,
 * the handler must not dereference the freed ope.
 */
void test_efa_rdm_pke_handle_send_completion_peer_removed(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr = 0;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Use test utility to allocate txe (also inserts a peer at addr 0) */
	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);

	/* Allocate and init a TX pkt_entry */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);

	err = efa_rdm_pke_init_eager_msgrtm(pkt_entry, txe);
	assert_int_equal(err, 0);

	/* Simulate device submission */
	efa_rdm_ep_record_tx_op_submitted(efa_rdm_ep, pkt_entry);

	/* Remove peer from AV, which triggers peer_destruct:
	 * - sets pkt_entry->peer = NULL
	 * - releases the txe (ope is now freed/reused)
	 */
	err = fi_av_remove(resource->av, &peer_addr, 1, 0);
	assert_int_equal(err, 0);
	assert_null(pkt_entry->peer);
	assert_null(pkt_entry->ope);
	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);

	/* Simulate the send completion arriving after peer removal.
	 * The handler must not dereference pkt_entry->ope since
	 * peer_destruct already released the txe.
	 */
	efa_rdm_pke_handle_send_completion(pkt_entry);
}

/**
 * @brief Test that efa_rdm_pke_handle_tx_error does not crash
 * when peer has been removed (use-after-free fix).
 *
 * Same scenario as above but for the error path.
 */
void test_efa_rdm_pke_handle_tx_error_peer_removed(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *txe;
	fi_addr_t peer_addr = 0;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Use test utility to allocate txe (also inserts a peer at addr 0) */
	txe = efa_unit_test_alloc_txe(resource, ofi_op_msg);
	assert_non_null(txe);

	/* Allocate and init a TX pkt_entry */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);

	err = efa_rdm_pke_init_eager_msgrtm(pkt_entry, txe);
	assert_int_equal(err, 0);

	/* Simulate device submission */
	efa_rdm_ep_record_tx_op_submitted(efa_rdm_ep, pkt_entry);

	/* Remove peer from AV */
	err = fi_av_remove(resource->av, &peer_addr, 1, 0);
	assert_int_equal(err, 0);
	assert_null(pkt_entry->peer);
	assert_null(pkt_entry->ope);
	assert_int_equal(efa_unit_test_get_ope_list_length(efa_rdm_ep, EFA_RDM_TXE), 0);

	/* Simulate a TX error arriving after peer removal.
	 * The handler must not dereference pkt_entry->ope since
	 * peer_destruct already released the txe.
	 */
	efa_rdm_pke_handle_tx_error(pkt_entry, EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR);
}

/**
 * @brief When handling a long cts rtm as read nack fallback,
 * efa_rdm_pke_handle_longcts_rtm_send_completion shouldn't touch
 * txe and write send completion.
 */
void test_efa_rdm_pke_handle_longcts_rtm_send_completion(void **state)
{
    struct efa_resource *resource = *state;
    struct efa_rdm_pke *pkt_entry;
    struct efa_rdm_ep *efa_rdm_ep;
    struct efa_rdm_peer *peer;
    struct fi_msg msg = {0};
    char buf[16];
	struct iovec iov = {
        .iov_base = buf,
        .iov_len = sizeof buf
    };
    struct efa_ep_addr raw_addr = {0};
    size_t raw_addr_len = sizeof(struct efa_ep_addr);
    fi_addr_t peer_addr;
    int err, numaddr;
    struct efa_rdm_ope *txe;

    efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

    efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

    /* create a fake peer */
    err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
    assert_int_equal(err, 0);
    raw_addr.qpn = 1;
    raw_addr.qkey = 0x1234;
    numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
    assert_int_equal(numaddr, 1);
    peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
    assert_non_null(peer);

    /* Construct a txe with read nack flag added */
    msg.addr = peer_addr;
    msg.iov_count = 1;
    msg.msg_iov = &iov;
    msg.desc = NULL;
    txe = ofi_buf_alloc(efa_rdm_ep->base_ep.ope_pool);
    assert_non_null(txe);
    efa_rdm_txe_construct(txe, efa_rdm_ep, peer, &msg, ofi_op_msg, 0, 0);
    txe->internal_flags |= EFA_RDM_OPE_READ_NACK;

    /* construct a fallback long cts rtm pkt */
    pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
    assert_non_null(pkt_entry);

    err = efa_rdm_pke_init_longcts_msgrtm(pkt_entry, txe);
    assert_int_equal(err, 0);

    assert_int_equal(pkt_entry->payload_size, 0);

    /* Mimic the case when CTSDATA pkts have completed all data and released the txe */
    txe->bytes_acked = txe->total_len;
    txe->bytes_sent = txe->total_len;
    efa_rdm_txe_release(txe);

    efa_rdm_pke_handle_longcts_rtm_send_completion(pkt_entry);

    /* CQ should be empty as send completion shouldn't be written */
    assert_int_equal(fi_cq_read(resource->cq, NULL, 1), -FI_EAGAIN);

    efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Test the efa_rdm_pke_release_rx_list function
 *
 * @param state
 */
void test_efa_rdm_pke_release_rx_list(void **state)
{
    struct efa_resource *resource = *state;
    struct efa_rdm_ep *efa_rdm_ep;
    struct efa_rdm_pke *pke, *curr;
    int i;

    efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

    efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

    /* Fake a rx pkt entry */
    pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
    assert_non_null(pke);
    efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

    /* link multiple pkes to this pke */
    for (i = 1; i < 10; i++) {
        curr = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
        assert_non_null(curr);
        efa_rdm_pke_append(pke, curr);
    }

    /* Release all entries in the linked list */
    efa_rdm_pke_release_rx_list(pke);

    /**
     * Now the rx pkt buffer pool should be empty so we can destroy it
     * Otherwise there will be an assertion error on the use cnt is
     * is non-zero
     */
    ofi_bufpool_destroy(efa_rdm_ep->efa_rx_pkt_pool);
    efa_rdm_ep->efa_rx_pkt_pool = NULL;
}

void test_efa_rdm_pke_alloc_rta_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *rxe;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Fake a rx pkt entry */
	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke);
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	/* Create and register a fake peer */
	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, 0);
	pke->peer = peer;

	rxe = efa_rdm_pke_alloc_rta_rxe(pke, ofi_op_atomic);
	assert_true(rxe->internal_flags & EFA_RDM_OPE_INTERNAL);

	efa_rdm_rxe_release(rxe);
	efa_rdm_pke_release_rx(pke);
}

void test_efa_rdm_pke_alloc_rtw_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *rxe;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_base_hdr *base_hdr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Fake a rx pkt entry */
	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke);
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	/* Create and register a fake peer */
	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	base_hdr = efa_rdm_pke_get_base_hdr(pke);
	/* Clean the flags to avoid having garbage value */
	base_hdr->flags = 0;

	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, 0);
	pke->peer = peer;

	rxe = efa_rdm_pke_alloc_rtw_rxe(pke);

	assert_true(rxe->internal_flags & EFA_RDM_OPE_INTERNAL);
	assert_int_equal(rxe->bytes_received, 0);
	assert_int_equal(rxe->bytes_copied, 0);

	efa_rdm_rxe_release(rxe);
	efa_rdm_pke_release_rx(pke);
}

void test_efa_rdm_pke_alloc_rtr_rxe(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pke;
	struct efa_rdm_ope *rxe;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = 0;
	struct efa_rdm_base_hdr *base_hdr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep,
				  base_ep.util_ep.ep_fid);

	/* Fake a rx pkt entry */
	pke = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pke);
	efa_rdm_ep->efa_rx_pkts_posted =
		efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	/* Create and register a fake peer */
	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	base_hdr = efa_rdm_pke_get_base_hdr(pke);
	/* Clean the flags to avoid having garbage value */
	base_hdr->flags = 0;

	struct efa_rdm_peer *peer = efa_rdm_ep_get_peer(efa_rdm_ep, 0);
	pke->peer = peer;

	rxe = efa_rdm_pke_alloc_rtw_rxe(pke);

	assert_true(rxe->internal_flags & EFA_RDM_OPE_INTERNAL);
	assert_int_equal(rxe->bytes_received, 0);
	assert_int_equal(rxe->bytes_copied, 0);

	efa_rdm_rxe_release(rxe);
	efa_rdm_pke_release_rx(pke);
}

void test_efa_rdm_pke_get_unexp(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_pke *pkt_entry, *unexp_pkt_entry;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&efa_rdm_ep->base_ep);

	unexp_pkt_entry = efa_rdm_pke_get_unexp(&pkt_entry);
	assert_non_null(unexp_pkt_entry);

#if ENABLE_DEBUG
	/* The unexp pkt entry should be inserted to the rx_pkt_list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rx_pkt_list), 1);
#endif
}

/**
 * @brief Test packet entry flag tracking for double linked list insertion/removal
 *
 * This test verifies that packet entries correctly track their insertion status
 * in outstanding_tx_pkts and queued_pkts lists using internal flags, and that
 * the release function properly removes them from the appropriate lists.
 *
 * @param state
 */
void test_efa_rdm_pke_flag_tracking(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *txe;
	struct fi_msg msg = {0};
	char buf[16];
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = sizeof(buf)
	};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t peer_addr;
	int err, numaddr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	/* Create a fake peer */
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	numaddr = fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL);
	assert_int_equal(numaddr, 1);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, peer_addr);
	assert_non_null(peer);

	/* Create a txe */
	msg.addr = peer_addr;
	msg.iov_count = 1;
	msg.msg_iov = &iov;
	msg.desc = NULL;
	txe = ofi_buf_alloc(efa_rdm_ep->base_ep.ope_pool);
	assert_non_null(txe);
	efa_rdm_txe_construct(txe, efa_rdm_ep, peer, &msg, ofi_op_msg, 0, 0);

	/* Allocate a packet entry */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = peer;

	/* Initially, packet should not be in any list */
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_IN_PEER_OUTSTANDING_TX_PKTS, 0);
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_IN_OPE_QUEUED_PKTS, 0);
	assert_true(dlist_empty(&peer->outstanding_tx_pkts));
	assert_true(dlist_empty(&txe->queued_pkts));

	/* Test insertion into outstanding_tx_pkts */
	efa_rdm_ep_record_tx_op_submitted(efa_rdm_ep, pkt_entry);
	assert_int_not_equal(pkt_entry->flags & EFA_RDM_PKE_IN_PEER_OUTSTANDING_TX_PKTS, 0);
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_IN_OPE_QUEUED_PKTS, 0);
	assert_false(dlist_empty(&peer->outstanding_tx_pkts));
	assert_true(dlist_empty(&txe->queued_pkts));

	/* Test removal from outstanding_tx_pkts */
	efa_rdm_ep_record_tx_op_completed(efa_rdm_ep, pkt_entry);
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_IN_PEER_OUTSTANDING_TX_PKTS, 0);
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_IN_OPE_QUEUED_PKTS, 0);
	assert_true(dlist_empty(&peer->outstanding_tx_pkts));
	assert_true(dlist_empty(&txe->queued_pkts));

	/* Test insertion into queued_pkts */
	efa_rdm_ep_queue_rnr_pkt(efa_rdm_ep, pkt_entry);
	assert_int_equal(pkt_entry->flags & EFA_RDM_PKE_IN_PEER_OUTSTANDING_TX_PKTS, 0);
	assert_int_not_equal(pkt_entry->flags & EFA_RDM_PKE_IN_OPE_QUEUED_PKTS, 0);
	assert_true(dlist_empty(&peer->outstanding_tx_pkts));
	assert_false(dlist_empty(&txe->queued_pkts));

	/* Test that release_tx removes from queued_pkts */
	efa_rdm_pke_release_tx(pkt_entry);
	assert_true(dlist_empty(&peer->outstanding_tx_pkts));
	assert_true(dlist_empty(&txe->queued_pkts));

	/* Clean up */
	efa_rdm_txe_release(txe);
}


/**
 * @brief Test efa_rdm_pke_proc_matched_eager_rtm releases pkt_entry on error.
 *
 * On error, efa_rdm_pke_proc_matched_eager_rtm (via the copy layer) releases
 * pkt_entry; the caller releases only rxe. Verify no leak and no double free.
 *
 * @param state
 */
void test_efa_rdm_pke_proc_matched_eager_rtm_error(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_base_hdr *base_hdr;
	char buf[16];
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	assert_non_null(pkt_entry);
	pkt_entry->payload_size = 1024;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	base_hdr->type = EFA_RDM_EAGER_MSGRTM_PKT;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->internal_flags = 0;
	rxe->iov_count = 1;
	rxe->iov[0].iov_base = buf;
	rxe->iov[0].iov_len = sizeof buf;
	rxe->cq_entry.len = 1024;
	rxe->total_len = 1024;
	efa_rdm_pke_set_ope(pkt_entry, rxe);

	g_efa_unit_test_mocks.efa_rdm_pke_copy_payload_to_ope = &efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock;
	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock, -FI_EINVAL);

	err = efa_rdm_pke_proc_matched_eager_rtm(pkt_entry);
	assert_int_not_equal(err, 0);

	/* The copy layer (mock) released pkt_entry on error; the caller releases
	 * only rxe. Releasing pkt_entry here would be a double free. */
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Helper function to create a medium RTM packet
 */
static struct efa_rdm_pke *create_medium_rtm_pkt(struct efa_rdm_ep *ep, uint32_t msg_id,
                                                 uint64_t msg_length, uint64_t seg_offset,
                                                 size_t payload_size)
{
	struct efa_rdm_pke *pkt;
	struct efa_rdm_base_hdr *base_hdr;
	struct efa_rdm_medium_rtm_base_hdr *medium_hdr;
	static char payload[1024];

	pkt = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!pkt)
		return NULL;

	pkt->payload = payload;
	pkt->payload_size = payload_size;
	pkt->next = NULL;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt);
	base_hdr->type = EFA_RDM_MEDIUM_MSGRTM_PKT;

	medium_hdr = efa_rdm_pke_get_medium_rtm_base_hdr(pkt);
	medium_hdr->hdr.flags = EFA_RDM_REQ_MSG;
	medium_hdr->hdr.msg_id = msg_id;
	medium_hdr->msg_length = msg_length;
	medium_hdr->seg_offset = seg_offset;

	return pkt;
}

/**
 * @brief Test efa_rdm_pke_proc_matched_mulreq_rtm releases the (first/only) pkt
 * on error. The copy layer releases the packet on error; the caller releases
 * rxe only. Verify no leak and no double free.
 *
 * @param state
 */
void test_efa_rdm_pke_proc_matched_mulreq_rtm_first_packet_error(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_ope *rxe;
	char buf[16];
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	pkt_entry = create_medium_rtm_pkt(efa_rdm_ep, 1, 1024, 0, 512);
	assert_non_null(pkt_entry);

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->internal_flags = 0;
	rxe->iov_count = 1;
	rxe->iov[0].iov_base = buf;
	rxe->iov[0].iov_len = sizeof buf;
	rxe->cq_entry.len = 1024;
	rxe->total_len = 1024;
	rxe->bytes_received = 0;
	rxe->bytes_received_via_mulreq = 0;
	efa_rdm_pke_set_ope(pkt_entry, rxe);

	g_efa_unit_test_mocks.efa_rdm_pke_copy_payload_to_ope = &efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock;
	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock, -FI_EINVAL);

	err = efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
	assert_int_not_equal(err, 0);

	/* The copy layer (mock) released the (first/only) pkt on error; the
	 * caller releases only rxe. Releasing pkt_entry here would be a double
	 * free. */
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Test efa_rdm_pke_proc_matched_mulreq_rtm correctly handles selective
 * packet failures using CMocka will_return. First packet succeeds, second fails.
 *
 * @param state
 */
void test_efa_rdm_pke_proc_matched_mulreq_rtm_second_packet_error(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry, *second_pkt;
	struct efa_rdm_ope *rxe;
	char buf[1024];
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	pkt_entry = create_medium_rtm_pkt(efa_rdm_ep, 1, 1024, 0, 512);
	assert_non_null(pkt_entry);

	second_pkt = create_medium_rtm_pkt(efa_rdm_ep, 1, 1024, 512, 512);
	assert_non_null(second_pkt);
	pkt_entry->next = second_pkt;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->internal_flags = 0;
	rxe->iov_count = 1;
	rxe->iov[0].iov_base = buf;
	rxe->iov[0].iov_len = sizeof buf;
	rxe->cq_entry.len = 2048;
	rxe->total_len = 2048;
	rxe->bytes_received = 0;
	rxe->bytes_received_via_mulreq = 0;
	efa_rdm_pke_set_ope(pkt_entry, rxe);

	g_efa_unit_test_mocks.efa_rdm_pke_copy_payload_to_ope = &efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock;

	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock, 0);          /* first pkt: success, NOT freed by mock */
	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_release_and_return_mock, -FI_EINVAL); /* second pkt: error, freed by mock */

	err = efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
	assert_int_not_equal(err, 0);

	/* First pkt: mock simulated success without releasing, so the test still
	 * owns it. Second pkt: released by the copy layer (mock) on error. */
	efa_rdm_pke_release_rx(pkt_entry);   /* first pkt only */
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Helper function to create a RUNTREAD MSGRTM packet from the rx pool
 */
static struct efa_rdm_pke *create_runtread_msgrtm_pkt(struct efa_rdm_ep *ep, uint32_t msg_id,
						      uint64_t msg_length, uint64_t runt_length, size_t payload_size)
{
	struct efa_rdm_pke *pkt;
	struct efa_rdm_base_hdr *base_hdr;
	struct efa_rdm_runtread_rtm_base_hdr *runtread_hdr;
	static char payload[1024];

	pkt = efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
	if (!pkt)
		return NULL;

	pkt->payload = payload;
	pkt->payload_size = payload_size;
	pkt->next = NULL;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt);
	base_hdr->type = EFA_RDM_RUNTREAD_MSGRTM_PKT;

	runtread_hdr = efa_rdm_pke_get_runtread_rtm_base_hdr(pkt);
	runtread_hdr->hdr.flags = EFA_RDM_REQ_MSG;
	runtread_hdr->hdr.msg_id = msg_id;
	runtread_hdr->msg_length = msg_length;
	runtread_hdr->send_id = 0;
	runtread_hdr->read_iov_count = 0;
	runtread_hdr->seg_offset = 0;
	runtread_hdr->runt_length = runt_length;

	return pkt;
}

/**
 * @brief The function must release the WHOLE chain via efa_rdm_pke_release_rx_list().
 * Verify: the call returns the truncation error, and BOTH packets are returned to the rx pool.
 * @param state
 */
void test_efa_rdm_pke_proc_matched_mulreq_rtm_runtread_trunc_chain(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry, *pkt_entry2;
	struct efa_rdm_ope *rxe;
	char buf[16];
	size_t to_post_before;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	pkt_entry  = create_runtread_msgrtm_pkt(efa_rdm_ep, 1, 1024, 256, 512);
	assert_non_null(pkt_entry);
	pkt_entry2 = create_runtread_msgrtm_pkt(efa_rdm_ep, 1, 1024, 256, 512);
	assert_non_null(pkt_entry2);
	pkt_entry->next = pkt_entry2;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_msg);
	assert_non_null(rxe);
	rxe->state = EFA_RDM_RXE_MATCHED;
	rxe->internal_flags = 0;
	/* Undersized receive: iov total (16) < total_len (1024) -> ofi_truncate_iov
	 * returns -FI_ETRUNC inside efa_rdm_ope_prepare_to_post_read(). */
	rxe->iov_count = 1;
	rxe->iov[0].iov_base = buf;
	rxe->iov[0].iov_len = sizeof buf;
	rxe->desc[0] = NULL;
	rxe->cq_entry.len = sizeof buf;
	rxe->total_len = 1024;
	rxe->bytes_read_total_len = 0;
	rxe->bytes_received = 0;
	rxe->bytes_received_via_mulreq = 0;
	efa_rdm_pke_set_ope(pkt_entry, rxe);
	to_post_before = efa_rdm_ep->efa_rx_pkts_to_post;
	err = efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
	assert_int_equal(err, -FI_ETRUNC);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post, to_post_before + 2);
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Test that flush_queued_blocking_copy_to_hmem releases all pkt entries
 * when a copy-size mismatch is detected, avoiding a memory leak.
 * @param state
 */
void test_efa_rdm_pke_flush_queued_blocking_copy_to_hmem_copy_size_mismatch(void **state){
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_pke *pkt_entry[2];
	struct efa_rdm_ope *rxe;
	struct efa_rdm_mr mock_mr = {0};
	char buf[1024];
	size_t pkts_to_post_before;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	mock_mr.efa_mr.iface = FI_HMEM_CUDA;
	mock_mr.device = 0;
	mock_mr.flags = 0;

	rxe = efa_rdm_ep_alloc_rxe(efa_rdm_ep, NULL, ofi_op_msg);
	assert_non_null(rxe);
	rxe->iov_count = 1;
	rxe->iov[0].iov_base = buf;
	rxe->iov[0].iov_len = sizeof(buf);
	rxe->cq_entry.len = sizeof(buf);
	rxe->desc[0] = &mock_mr;
	rxe->total_len = sizeof(buf);

	for (int i = 0; i < 2; i++) {
		pkt_entry[i] = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_rx_pkt_pool, EFA_RDM_PKE_FROM_EFA_RX_POOL);
		assert_non_null(pkt_entry[i]);
		pkt_entry[i]->payload = pkt_entry[i]->wiredata;
		pkt_entry[i]->payload_size = 64;
		pkt_entry[i]->ope = rxe;
		pkt_entry[i]->ope_gen = rxe->gen;

		/* Replicate efa_rdm_pke_queued_copy_payload_to_hmem() */
		efa_rdm_ep->queued_copy_vec[i].pkt_entry = pkt_entry[i];
		efa_rdm_ep->queued_copy_vec[i].data = pkt_entry[i]->payload;
		efa_rdm_ep->queued_copy_vec[i].data_size = pkt_entry[i]->payload_size;
		efa_rdm_ep->queued_copy_vec[i].data_offset = 0;
		efa_rdm_ep->queued_copy_num++;
		rxe->bytes_queued_blocking_copy += pkt_entry[i]->payload_size;
		efa_rdm_pke_mark_held(pkt_entry[i]);
	}

	/* Mock ofi_copy_to_hmem_iov to return 0 to trigger the copy-size mismatch error path */
	g_efa_unit_test_mocks.ofi_copy_to_hmem_iov = &efa_mock_ofi_copy_to_hmem_iov_return_mock;
	will_return_always(efa_mock_ofi_copy_to_hmem_iov_return_mock, 0);

	/*
	 * efa_rdm_pke_release_rx() increments efa_rx_pkts_to_post for entries allocated from EFA_RX_POOL.
	 * Snapshot before the call to verify: The 2 queued pkt entries were released back to the pool (i.e. no leak).
	 */
	pkts_to_post_before = efa_rdm_ep->efa_rx_pkts_to_post;

	ret = efa_rdm_ep_flush_queued_blocking_copy_to_hmem(efa_rdm_ep);
	assert_int_equal(ret, -FI_EIO);
	assert_int_equal(efa_rdm_ep->queued_copy_num, 0);
	assert_int_equal(efa_rdm_ep->efa_rx_pkts_to_post - pkts_to_post_before, 2);
	assert_int_equal(rxe->bytes_queued_blocking_copy, 0);

	efa_rdm_rxe_release(rxe);
}
