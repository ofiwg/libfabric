#ifndef EFA_UNIT_TEST_RDMA_CORE_MOCKS_H
#define EFA_UNIT_TEST_RDMA_CORE_MOCKS_H

extern int g_ibv_create_ah_call_counter;

struct ibv_ah *__real_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr);

struct ibv_ah *efa_mock_ibv_create_ah_increase_call_counter(struct ibv_pd *pd, struct ibv_ah_attr *attr);

int __real_efadv_query_device(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
			      uint32_t inlen);

int efa_mock_efadv_query_device_return_mock(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
					    uint32_t inlen);

void __real_rxr_pkt_handle_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void efa_mock_rxr_pkt_handle_send_completion_check_args_only(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void __real_rxr_pkt_handle_recv_completion(struct rxr_ep *ep,
					   struct rxr_pkt_entry *pkt_entry,
					   enum rxr_lower_ep_type lower_ep_type);

void efa_mock_rxr_pkt_handle_recv_completion_check_args_only(struct rxr_ep *ep,
							     struct rxr_pkt_entry *pkt_entry,
							     enum rxr_lower_ep_type lower_ep_type);

void __real_rxr_pkt_handle_send_error(struct rxr_ep *ep,
				      struct rxr_pkt_entry *pkt_entry,
				      int err, int prov_errno);

void efa_mock_rxr_pkt_handle_send_error_check_args_only(struct rxr_ep *ep,
							struct rxr_pkt_entry *pkt_entry,
							int err, int prov_errno);

void __real_rxr_pkt_handle_recv_error(struct rxr_ep *ep,
				      struct rxr_pkt_entry *pkt_entry,
				      int err, int prov_errno);

void efa_mock_rxr_pkt_handle_recv_error_check_args_only(struct rxr_ep *ep,
							struct rxr_pkt_entry *pkt_entry,
							int err, int prov_errno);

struct efa_unit_test_mocks
{
	struct ibv_ah *(*ibv_create_ah)(struct ibv_pd *pd, struct ibv_ah_attr *attr);

	int (*efadv_query_device)(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
				  uint32_t inlen);

	void (*rxr_pkt_handle_send_completion)(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

	void (*rxr_pkt_handle_recv_completion)(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
					       enum rxr_lower_ep_type lower_ep_type);

	void (*rxr_pkt_handle_send_error)(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
					  int err, int prov_errno);

	void (*rxr_pkt_handle_recv_error)(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
					  int err, int prov_errno);
};

extern struct efa_unit_test_mocks g_efa_unit_test_mocks;

#endif
