#define _GNU_SOURCE
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "efa.h"
#include "efa_unit_test_mocks.h"

/* mock of rdma-core functions */

/**
 * @brief call real ibv_create_ah and mock()
 *
 * When combined with will_return_count(), this mock of ibv_create_ah() can be used to verify
 * number of times ibv_create_ah() is called.
 */
struct ibv_ah *efa_mock_ibv_create_ah_check_mock(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	mock();
	return  __real_ibv_create_ah(pd, attr);
}

int efa_mock_efadv_query_device_return_mock(struct ibv_context *ibv_ctx,
					    struct efadv_device_attr *attr,
					    uint32_t inlen)
{
	return mock();
}


/**
 * @brief a list of ibv_send_wr that was called by ibv_post_send
 */
struct efa_mock_ibv_send_wr_list g_ibv_send_wr_list =
{
	.head = NULL,
	.tail = NULL,
};

/**
 * @brief Destruct efa_mock_ibv_send_wr_list and free up memory
 *
 * @param[in] wr_list Pointer to efa_mock_ibv_send_wr_list
 */
void efa_mock_ibv_send_wr_list_destruct(struct efa_mock_ibv_send_wr_list *wr_list) {
	struct ibv_send_wr *wr;

	while (wr_list->head) {
		wr = wr_list->head;
		wr_list->head = wr_list->head->next;
		free(wr);
	}

	wr_list->tail = NULL;
}

/**
 * @brief save send_wr in global variable g_ibv_send_wr_list
 *
 * This mock of ibv_post_send will NOT actually send data over wire,
 * but save the send work request (wr) in a list named g_ibv_send_wr_list.
 *
 * The saved work request is then be used by efa_mock_ibv_start_poll_use_send_wr()
 * to make ibv_cq_ex to look like it indeed got a completion from device.
 */
int efa_mock_ibv_post_send_save_send_wr(struct ibv_qp *qp, struct ibv_send_wr *wr,
					struct ibv_send_wr **bad_wr)
{
	struct ibv_send_wr *head, *tail, *entry;

	head = tail = NULL;

	while(wr) {
		entry = calloc(sizeof(struct ibv_send_wr), 1);
		if (!entry) {
			*bad_wr = wr;
			return ENOMEM;
		}

		memcpy(entry, wr, sizeof(struct ibv_send_wr));
		if (!head)
			head = entry;

		if (tail)
			tail->next = entry;

		tail = entry;
		tail->next = NULL;
		wr = wr->next;
	}

	if (!g_ibv_send_wr_list.head) {
		g_ibv_send_wr_list.head = head;
	}

	if (g_ibv_send_wr_list.tail) {
		g_ibv_send_wr_list.tail->next = head;
	}

	g_ibv_send_wr_list.tail = tail;
	return 0;
}

int efa_mock_ibv_post_send_verify_handshake_pkt_local_host_id_and_save_wr(struct ibv_qp *qp, struct ibv_send_wr *wr,
																	  struct ibv_send_wr **bad_wr)
{
	struct ibv_sge sge;
	struct rxr_base_hdr *rxr_base_hdr;
	struct rxr_handshake_opt_host_id_hdr *host_id_hdr;

	assert_true(wr->num_sge > 0);

	sge = wr->sg_list[0];
	assert_true(sge.length > 0);

	rxr_base_hdr = rxr_get_base_hdr((void *)sge.addr);

	assert_int_equal(rxr_base_hdr->type, RXR_HANDSHAKE_PKT);

	if (g_efa_unit_test_mocks.local_host_id) {
		assert_true(rxr_base_hdr->flags & RXR_HANDSHAKE_HOST_ID_HDR);
		host_id_hdr = rxr_get_handshake_opt_host_id_hdr(rxr_base_hdr);
		assert_true(host_id_hdr->host_id == g_efa_unit_test_mocks.local_host_id);
	} else {
		assert_false(rxr_base_hdr->flags & RXR_HANDSHAKE_HOST_ID_HDR);
	}

	function_called();

	return efa_mock_ibv_post_send_save_send_wr(qp, wr, bad_wr);
}

int efa_mock_ibv_start_poll_return_mock(struct ibv_cq_ex *ibvcqx,
					struct ibv_poll_cq_attr *attr)
{
	return mock();
}

static inline
int efa_mock_use_saved_send_wr(struct ibv_cq_ex *ibv_cqx, int status)
{
	struct ibv_send_wr *used;

	if (!g_ibv_send_wr_list.head) {
		assert(!g_ibv_send_wr_list.tail);
		return ENOENT;
	}

	ibv_cqx->wr_id = g_ibv_send_wr_list.head->wr_id;
	ibv_cqx->status = status;

	used = g_ibv_send_wr_list.head;
	g_ibv_send_wr_list.head = g_ibv_send_wr_list.head->next;
	free(used);

	if (!g_ibv_send_wr_list.head)
		g_ibv_send_wr_list.tail = NULL;

	return 0;
}

int efa_mock_ibv_start_poll_use_saved_send_wr_with_mock_status(struct ibv_cq_ex *ibv_cqx,
							       struct ibv_poll_cq_attr *attr)
{
	return efa_mock_use_saved_send_wr(ibv_cqx, mock());
}

int efa_mock_ibv_next_poll_return_mock(struct ibv_cq_ex *ibvcqx)
{
	return mock();
}

int efa_mock_ibv_next_poll_use_saved_send_wr_with_mock_status(struct ibv_cq_ex *ibv_cqx)
{
	return efa_mock_use_saved_send_wr(ibv_cqx, mock());
}

void efa_mock_ibv_end_poll_check_mock(struct ibv_cq_ex *ibvcqx)
{
	mock();
}

uint32_t efa_mock_ibv_read_opcode_return_mock(struct ibv_cq_ex *current)
{
	return mock();
}

uint32_t efa_mock_ibv_read_vendor_err_return_mock(struct ibv_cq_ex *current)
{
	return mock();
}

int g_ofi_copy_from_hmem_iov_call_counter;
ssize_t efa_mock_ofi_copy_from_hmem_iov_inc_counter(void *dest, size_t size,
						    enum fi_hmem_iface hmem_iface, uint64_t device,
						    const struct iovec *hmem_iov,
						    size_t hmem_iov_count, uint64_t hmem_iov_offset)
{
	g_ofi_copy_from_hmem_iov_call_counter += 1;
	return __real_ofi_copy_from_hmem_iov(dest, size, hmem_iface, device, hmem_iov, hmem_iov_count, hmem_iov_offset);
}

struct efa_unit_test_mocks g_efa_unit_test_mocks = {
	.local_host_id = 0,
	.peer_host_id = 0,
	.ibv_create_ah = __real_ibv_create_ah,
	.efadv_query_device = __real_efadv_query_device,
#if HAVE_EFADV_CQ_EX
	.efadv_create_cq = __real_efadv_create_cq,
#endif
#if HAVE_NEURON
	.neuron_alloc = __real_neuron_alloc,
#endif
	.ofi_copy_from_hmem_iov = __real_ofi_copy_from_hmem_iov,
};

struct ibv_ah *__wrap_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	return g_efa_unit_test_mocks.ibv_create_ah(pd, attr);
}

int __wrap_efadv_query_device(struct ibv_context *ibv_ctx, struct efadv_device_attr *attr,
			      uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_query_device(ibv_ctx, attr, inlen);
}

struct ibv_cq_ex *efa_mock_create_cq_ex_return_null(struct ibv_context *context, struct ibv_cq_init_attr_ex *init_attr)
{
	function_called();
	return NULL;
};

#if HAVE_EFADV_CQ_EX
struct ibv_cq_ex *__wrap_efadv_create_cq(struct ibv_context *ibvctx,
										 struct ibv_cq_init_attr_ex *attr_ex,
										 struct efadv_cq_init_attr *efa_attr,
										 uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_create_cq(ibvctx, attr_ex, efa_attr, inlen);
}

uint32_t efa_mock_ibv_read_src_qp_return_mock(struct ibv_cq_ex *current)
{
	return mock();
}

uint32_t efa_mock_ibv_read_byte_len_return_mock(struct ibv_cq_ex *current)
{
	return mock();
};

uint32_t efa_mock_ibv_read_slid_return_mock(struct ibv_cq_ex *current)
{
	return mock();
}

int efa_mock_efadv_wc_read_sgid_return_mock(struct efadv_cq *efadv_cq, union ibv_gid *sgid)
{
	return mock();
}

int efa_mock_efadv_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid(struct efadv_cq *efadv_cq, union ibv_gid *sgid)
{
	/* Make sure this mock is always called before ibv_next_poll */
	expect_function_call(efa_mock_ibv_next_poll_check_function_called_and_return_mock);
	memcpy(sgid->raw, (uint8_t *)mock(), sizeof(sgid->raw));
	/* Must return 0 for unknown AH */
	return 0;
};

int efa_mock_ibv_next_poll_check_function_called_and_return_mock(struct ibv_cq_ex *ibvcqx)
{
	function_called();
	return mock();
};

struct ibv_cq_ex *efa_mock_efadv_create_cq_with_ibv_create_cq_ex(struct ibv_context *ibvctx,
																 struct ibv_cq_init_attr_ex *attr_ex,
																 struct efadv_cq_init_attr *efa_attr,
																 uint32_t inlen)
{
	function_called();
	return ibv_create_cq_ex(ibvctx, attr_ex);
}

struct ibv_cq_ex *efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null(struct ibv_context *ibvctx,
																		  struct ibv_cq_init_attr_ex *attr_ex,
																		  struct efadv_cq_init_attr *efa_attr,
																		  uint32_t inlen)
{
	function_called();
	errno = EOPNOTSUPP;
	return NULL;
}
#endif

#if HAVE_NEURON
void *__wrap_neuron_alloc(void **handle, size_t size)
{
	return g_efa_unit_test_mocks.neuron_alloc(handle, size);
}

void *efa_mock_neuron_alloc_return_null(void **handle, size_t size)
{
	return NULL;
}
#endif

ssize_t __wrap_ofi_copy_from_hmem_iov(void *dest, size_t size,
				      enum fi_hmem_iface hmem_iface, uint64_t device,
				      const struct iovec *hmem_iov,
				      size_t hmem_iov_count, uint64_t hmem_iov_offset)
{
	return g_efa_unit_test_mocks.ofi_copy_from_hmem_iov(dest, size, hmem_iface, device, hmem_iov, hmem_iov_count, hmem_iov_offset);
}