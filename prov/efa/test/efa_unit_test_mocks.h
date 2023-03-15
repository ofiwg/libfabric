#ifndef EFA_UNIT_TEST_RDMA_CORE_MOCKS_H
#define EFA_UNIT_TEST_RDMA_CORE_MOCKS_H

extern struct efa_unit_test_mocks g_efa_unit_test_mocks;

struct efa_mock_ibv_send_wr_list
{
	struct ibv_send_wr *head;
	struct ibv_send_wr *tail;
};

void efa_mock_ibv_send_wr_list_destruct(struct efa_mock_ibv_send_wr_list *wr_list);

struct ibv_ah *__real_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr);

struct ibv_ah *efa_mock_ibv_create_ah_check_mock(struct ibv_pd *pd, struct ibv_ah_attr *attr);

int __real_efadv_query_device(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
			      uint32_t inlen);

int efa_mock_efadv_query_device_return_mock(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
					    uint32_t inlen);

extern struct efa_mock_ibv_send_wr_list g_ibv_send_wr_list;
int efa_mock_ibv_post_send_save_send_wr(struct ibv_qp *qp, struct ibv_send_wr *wr,
					struct ibv_send_wr **bad_wr);
int efa_mock_ibv_post_send_verify_handshake_pkt_local_host_id_and_save_wr(struct ibv_qp *qp, struct ibv_send_wr *wr,
					struct ibv_send_wr **bad_wr);

int efa_mock_ibv_start_poll_return_mock(struct ibv_cq_ex *ibvcqx,
					struct ibv_poll_cq_attr *attr);

int efa_mock_ibv_start_poll_use_saved_send_wr_with_mock_status(struct ibv_cq_ex *ibvcqx,
							       struct ibv_poll_cq_attr *attr);

int efa_mock_ibv_next_poll_return_mock(struct ibv_cq_ex *ibvcqx);

int efa_mock_ibv_next_poll_use_saved_send_wr_with_mock_status(struct ibv_cq_ex *ibvcqx);

void efa_mock_ibv_end_poll_check_mock(struct ibv_cq_ex *ibvcqx);

uint32_t efa_mock_ibv_read_opcode_return_mock(struct ibv_cq_ex *current);

uint32_t efa_mock_ibv_read_vendor_err_return_mock(struct ibv_cq_ex *current);

ssize_t __real_ofi_copy_from_hmem_iov(void *dest, size_t size,
				      enum fi_hmem_iface hmem_iface, uint64_t device,
				      const struct iovec *hmem_iov,
				      size_t hmem_iov_count, uint64_t hmem_iov_offset);

extern int g_ofi_copy_from_hmem_iov_call_counter;
ssize_t efa_mock_ofi_copy_from_hmem_iov_inc_counter(void *dest, size_t size,
						    enum fi_hmem_iface hmem_iface, uint64_t device,
						    const struct iovec *hmem_iov,
						    size_t hmem_iov_count, uint64_t hmem_iov_offset);

struct efa_unit_test_mocks
{
	uint64_t local_host_id;
	uint64_t peer_host_id;
	struct ibv_ah *(*ibv_create_ah)(struct ibv_pd *pd, struct ibv_ah_attr *attr);

	int (*efadv_query_device)(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
							  uint32_t inlen);
#if HAVE_EFADV_CQ_EX

	struct ibv_cq_ex *(*efadv_create_cq)(struct ibv_context *ibvctx,
										 struct ibv_cq_init_attr_ex *attr_ex,
										 struct efadv_cq_init_attr *efa_attr,
										 uint32_t inlen);
#endif

#if HAVE_NEURON
	void *(*neuron_alloc)(void **handle, size_t size);
#endif

	ssize_t (*ofi_copy_from_hmem_iov)(void *dest, size_t size,
					  enum fi_hmem_iface hmem_iface, uint64_t device,
					  const struct iovec *hmem_iov,
					  size_t hmem_iov_count, uint64_t hmem_iov_offset);
};

struct ibv_cq_ex *efa_mock_create_cq_ex_return_null(struct ibv_context *context, struct ibv_cq_init_attr_ex *init_attr);

#if HAVE_EFADV_CQ_EX
struct ibv_cq_ex *__real_efadv_create_cq(struct ibv_context *ibvctx,
											struct ibv_cq_init_attr_ex *attr_ex,
											struct efadv_cq_init_attr *efa_attr,
											uint32_t inlen);
uint32_t efa_mock_ibv_read_src_qp_return_mock(struct ibv_cq_ex *current);
uint32_t efa_mock_ibv_read_byte_len_return_mock(struct ibv_cq_ex *current);
uint32_t efa_mock_ibv_read_slid_return_mock(struct ibv_cq_ex *current);
int efa_mock_efadv_wc_read_sgid_return_mock(struct efadv_cq *efadv_cq, union ibv_gid *sgid);
int efa_mock_efadv_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid(struct efadv_cq *efadv_cq, union ibv_gid *sgid);
int efa_mock_ibv_start_poll_expect_efadv_wc_read_ah_and_return_mock(struct ibv_cq_ex *ibvcqx,
																	struct ibv_poll_cq_attr *attr);
int efa_mock_ibv_next_poll_check_function_called_and_return_mock(struct ibv_cq_ex *ibvcqx);
struct ibv_cq_ex *efa_mock_efadv_create_cq_with_ibv_create_cq_ex(struct ibv_context *ibvctx,
																 struct ibv_cq_init_attr_ex *attr_ex,
																 struct efadv_cq_init_attr *efa_attr,
																 uint32_t inlen);
struct ibv_cq_ex *efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null(struct ibv_context *ibvctx,
																		  struct ibv_cq_init_attr_ex *attr_ex,
																		  struct efadv_cq_init_attr *efa_attr,
																		  uint32_t inlen);
#endif

#if HAVE_NEURON
void *__real_neuron_alloc(void **handle, size_t size);
void *efa_mock_neuron_alloc_return_null(void **handle, size_t size);
#endif

#endif
