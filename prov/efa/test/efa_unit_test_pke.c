#include "efa_unit_tests.h"
#include "rdm/efa_rdm_pke_rtm.h"

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
