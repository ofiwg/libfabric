#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_pke_rta.h"
#include "rdm/efa_rdm_pke_rtw.h"
#include "rdm/efa_rdm_pke_utils.h"


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
    txe = efa_rdm_ep_alloc_txe(efa_rdm_ep, peer, &msg, ofi_op_msg, 0, 0);
    assert_non_null(txe);
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
    efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

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
		efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

	/* Create and register a fake peer */
	assert_int_equal(
		fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	assert_int_equal(
		fi_av_insert(resource->av, &raw_addr, 1, &peer_addr, 0, NULL),
		1);

	pke->addr = peer_addr;

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
		efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

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
	pke->addr = peer_addr;

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
		efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

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
	pke->addr = peer_addr;

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
	efa_rdm_ep->efa_rx_pkts_posted = efa_rdm_ep_get_rx_pool_size(efa_rdm_ep);

	unexp_pkt_entry = efa_rdm_pke_get_unexp(&pkt_entry);
	assert_non_null(unexp_pkt_entry);

#if ENABLE_DEBUG
	/* The unexp pkt entry should be inserted to the rx_pkt_list */
	assert_int_equal(efa_unit_test_get_dlist_length(&efa_rdm_ep->rx_pkt_list), 1);
#endif
}
