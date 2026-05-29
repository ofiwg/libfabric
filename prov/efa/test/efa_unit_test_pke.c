#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_pke_rta.h"
#include "rdm/efa_rdm_pke_rtw.h"
#include "rdm/efa_rdm_pke_utils.h"
#include "rdm/efa_rdm_pke_nonreq.h"


/**
 * @brief When handling a long cts rtm as read nack fallback,
 * efa_rdm_pke_handle_longcts_rtm_send_completion shouldn't touch
 * txe and write send completion.
 */
void test_efa_rdm_pke_handle_longcts_rtm_send_completion(struct efa_resource **state)
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
    txe = ofi_buf_alloc(efa_rdm_ep->ope_pool);
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
void test_efa_rdm_pke_release_rx_list(struct efa_resource **state)
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

void test_efa_rdm_pke_alloc_rta_rxe(struct efa_resource **state)
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

void test_efa_rdm_pke_alloc_rtw_rxe(struct efa_resource **state)
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

void test_efa_rdm_pke_alloc_rtr_rxe(struct efa_resource **state)
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

void test_efa_rdm_pke_get_unexp(struct efa_resource **state)
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
void test_efa_rdm_pke_flag_tracking(struct efa_resource **state)
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
	txe = ofi_buf_alloc(efa_rdm_ep->ope_pool);
	assert_non_null(txe);
	efa_rdm_txe_construct(txe, efa_rdm_ep, peer, &msg, ofi_op_msg, 0, 0);

	/* Allocate a packet entry */
	pkt_entry = efa_rdm_pke_alloc(efa_rdm_ep, efa_rdm_ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	pkt_entry->ope = txe;
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
 * @brief Test efa_rdm_pke_proc_matched_eager_rtm doesn't free pkt_entry on error
 * because it is handled by the caller.
 *
 *
 * @param state
 */
void test_efa_rdm_pke_proc_matched_eager_rtm_error(struct efa_resource **state)
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
	pkt_entry->ope = rxe;

	g_efa_unit_test_mocks.efa_rdm_pke_copy_payload_to_ope = &efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock;
	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock, -FI_EINVAL);

	err = efa_rdm_pke_proc_matched_eager_rtm(pkt_entry);
	assert_int_not_equal(err, 0);

	/* Verify there is no double free */
	efa_rdm_pke_release_rx(pkt_entry);
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
 * @brief Test efa_rdm_pke_proc_matched_mulreq_rtm doesn't double free the first
 * pkt_entry on error. The first packet should be released by the caller, not
 * by the function itself.
 *
 * @param state
 */
void test_efa_rdm_pke_proc_matched_mulreq_rtm_first_packet_error(struct efa_resource **state)
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
	pkt_entry->ope = rxe;

	g_efa_unit_test_mocks.efa_rdm_pke_copy_payload_to_ope = &efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock;
	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock, -FI_EINVAL);

	err = efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
	assert_int_not_equal(err, 0);

	/* Verify there is no double free by releasing the first packet entry */
	efa_rdm_pke_release_rx(pkt_entry);
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Test efa_rdm_pke_proc_matched_mulreq_rtm correctly handles selective
 * packet failures using CMocka will_return. First packet succeeds, second fails.
 *
 * @param state
 */
void test_efa_rdm_pke_proc_matched_mulreq_rtm_second_packet_error(struct efa_resource **state)
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
	pkt_entry->ope = rxe;

	g_efa_unit_test_mocks.efa_rdm_pke_copy_payload_to_ope = &efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock;

	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock, 0);
	will_return_int(efa_mock_efa_rdm_pke_copy_payload_to_ope_return_mock, -FI_EINVAL);

	err = efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
	assert_int_not_equal(err, 0);

	/* The function should have:
	 * 1. NOT released the first packet - caller's responsibility
	 * 2. Released the second packet - function's responsibility
	 *
	 * We only release the first packet here. The second packet should have
	 * been released by the function when it failed.
	 */
	efa_rdm_pke_release_rx(pkt_entry);
	efa_rdm_rxe_release(rxe);
}

/**
 * @brief Test that flush_queued_blocking_copy_to_hmem releases all pkt entries
 * when a copy-size mismatch is detected, avoiding a memory leak.
 * @param state
 */
void test_efa_rdm_pke_flush_queued_blocking_copy_to_hmem_copy_size_mismatch(struct efa_resource **state){
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

/**
 * @brief Verify efa_rdm_prov_errno_is_peer_abort() classifies the
 *        in-scope and out-of-scope provider errnos correctly.
 *
 * Peer-abort statuses are EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS
 * and EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT. All other statuses
 * (LOCAL_ERROR_*, BAD_LENGTH, RNR, UNRESP_REMOTE, UNKNOWN_PEER, ...)
 * must classify as non-peer-abort so the existing user-visible error
 * paths remain untouched.
 */
void test_efa_rdm_prov_errno_is_peer_abort(struct efa_resource **state)
{
	(void) state;

	struct {
		int prov_errno;
		bool expected;
	} cases[] = {
		/* In scope: peer-clean-abort statuses. */
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS, true },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT, true },

		/* Out of scope: genuine local/network/protocol errors that
		 * the user must continue to see. */
		{ EFA_IO_COMP_STATUS_OK, false },
		{ EFA_IO_COMP_STATUS_FLUSHED, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH, false },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN, false },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR, false },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH, false },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE, false },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER, false },
		{ EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE, false },
		{ EFA_IO_COMP_STATUS_REMOTE_ERROR_FEATURE_MISMATCH, false },
		{ FI_EFA_ERR_OOM, false },
		{ FI_EFA_ERR_OTHER, false },
	};

	for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
		bool got = efa_rdm_prov_errno_is_peer_abort(cases[i].prov_errno);
		assert_int_equal(got, cases[i].expected);
	}
}

/**
 * @brief Verify efa_rdm_pkt_is_rxe_remote_read() classifies packet
 *        types correctly.
 */
void test_efa_rdm_pkt_is_rxe_remote_read(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_base_hdr *base_hdr;
	struct efa_rdm_rma_context_pkt *ctx_pkt;
	struct efa_rdm_ope rxe_ope = { .type = EFA_RDM_RXE };
	struct efa_rdm_ope txe_ope = { .type = EFA_RDM_TXE };

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	pkt_entry->ope = &rxe_ope;

	base_hdr = (struct efa_rdm_base_hdr *) pkt_entry->wiredata;

	/* Receiver-initiated RDMA READ context packet — in scope */
	ctx_pkt = (struct efa_rdm_rma_context_pkt *) pkt_entry->wiredata;
	ctx_pkt->type = EFA_RDM_RMA_CONTEXT_PKT;
	ctx_pkt->context_type = EFA_RDM_RDMA_READ_CONTEXT;
	assert_true(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	/* Same RDMA READ context, but owned by a txe (one-sided
	 * fi_read) — out of scope: the ope->type guard must reject it. */
	pkt_entry->ope = &txe_ope;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));
	pkt_entry->ope = &rxe_ope;

	/* Sender-initiated RDMA WRITE context packet — out of scope */
	ctx_pkt->context_type = EFA_RDM_RDMA_WRITE_CONTEXT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	/* Receiver-side control SENDs that ride on an rxe — out of
	 * scope for this helper (they are not routed through the
	 * peer-abort handler today; if that changes, extend or split
	 * this helper). */
	memset(pkt_entry->wiredata, 0,
	       sizeof(struct efa_rdm_rma_context_pkt) + 64);
	base_hdr->type = EFA_RDM_CTS_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_EOR_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_RECEIPT_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	/* Sender-side / non-rxe-protocol packets — out of scope */
	base_hdr->type = EFA_RDM_HANDSHAKE_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_CTSDATA_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_READRSP_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_READ_NACK_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_EAGER_MSGRTM_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_LONGCTS_TAGRTM_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	base_hdr->type = EFA_RDM_LONGREAD_MSGRTM_PKT;
	assert_false(efa_rdm_pkt_is_rxe_remote_read(pkt_entry));

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Verify efa_rdm_pke_init_peer_error_for_ope() derives the wire
 *        op_id/ref_kind/prov_errno for the OPE_INDEX directions.
 *
 * The packet's op_id always names an ope owned by the RECEIVER of the
 * packet, so the sender populates it from the peer's id captured on the
 * wire:
 *   - rxe (LONGREAD direction, receiver -> sender): op_id = rxe->tx_id.
 *   - txe (LONGCTS direction, sender -> receiver): op_id = txe->rx_id.
 * Both use ref_kind = EFA_RDM_PEER_ERROR_REF_OPE_INDEX.
 */
void test_efa_rdm_pke_init_peer_error_for_ope_ope_index(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *ep;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_peer_error_hdr *hdr;
	struct efa_rdm_ope rxe = {0};
	struct efa_rdm_ope txe = {0};

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ep = container_of(resource->ep, struct efa_rdm_ep,
			  base_ep.util_ep.ep_fid);

	/* rxe (LONGREAD direction): op_id = tx_id. */
	rxe.type = EFA_RDM_RXE;
	rxe.ep = ep;
	rxe.tx_id = 0x1234;
	rxe.peer_error_prov_errno = EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS;
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	assert_int_equal(efa_rdm_pke_init_peer_error_for_ope(pkt_entry, &rxe), 0);
	hdr = efa_rdm_pke_get_peer_error_hdr(pkt_entry);
	assert_int_equal(hdr->type, EFA_RDM_PEER_ERROR_PKT);
	assert_int_equal(hdr->ref_kind, EFA_RDM_PEER_ERROR_REF_OPE_INDEX);
	assert_int_equal(hdr->op_id, rxe.tx_id);
	assert_int_equal(hdr->prov_errno,
			 EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS);
	efa_rdm_pke_release_tx(pkt_entry);

	/* txe (LONGCTS direction): op_id = rx_id. */
	txe.type = EFA_RDM_TXE;
	txe.ep = ep;
	txe.rx_id = 0x5678;
	txe.peer_error_prov_errno = EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY;
	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	assert_non_null(pkt_entry);
	assert_int_equal(efa_rdm_pke_init_peer_error_for_ope(pkt_entry, &txe), 0);
	hdr = efa_rdm_pke_get_peer_error_hdr(pkt_entry);
	assert_int_equal(hdr->ref_kind, EFA_RDM_PEER_ERROR_REF_OPE_INDEX);
	assert_int_equal(hdr->op_id, txe.rx_id);
	assert_int_equal(hdr->prov_errno,
			 EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY);
	efa_rdm_pke_release_tx(pkt_entry);
}
