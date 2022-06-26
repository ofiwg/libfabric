#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct ibv_ah *__wrap_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr) {
    check_expected(pd);
    check_expected(attr);
    return (struct ibv_ah*) mock();
}

int __real_ibv_destroy_ah(struct ibv_ah *ibv_ah);

int __wrap_ibv_destroy_ah(struct ibv_ah *ibv_ah)
{
	int val = mock();
	if (val == 4242) {
		return __real_ibv_destroy_ah(ibv_ah);
	}
	return val;
}

int __wrap_efadv_query_ah(struct ibv_ah *ibvah, struct efadv_ah_attr *attr, uint32_t inlen) {
    check_expected(ibvah);
    check_expected(attr);
    check_expected(inlen);
    return (int) mock();
}

int __real_efadv_query_device(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
			     uint32_t inlen);

int __wrap_efadv_query_device(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
			     uint32_t inlen)
{
	int retval;

	retval = mock();
	/* Expected return value being 0 means we want this function to work as expected
	 * hence call the real function in this case
	 */
	return (retval == 0) ? __real_efadv_query_device(ibvctx, attr, inlen) : retval;
}

static int efa_unit_test_mock_ibv_start_poll(struct ibv_cq_ex *ibvcqx,
					     struct ibv_poll_cq_attr *attr)
{
	return mock();
}

static int efa_unit_test_mock_ibv_next_poll(struct ibv_cq_ex *ibvcqx)
{
	return mock();
}

static void efa_unit_test_mock_ibv_end_poll(struct ibv_cq_ex *ibvcqx)
{
	mock();
}

static uint32_t efa_unit_test_mock_ibv_read_vendor_err(struct ibv_cq_ex *current)
{
	return 1;
}

static unsigned int efa_unit_test_mock_ibv_read_wc_flags(struct ibv_cq_ex *current)
{
	return 0;
}

static uint32_t efa_unit_test_mock_ibv_read_qp_num(struct ibv_cq_ex *current)
{
	return 0;
}

static uint32_t efa_unit_test_mock_ibv_read_byte_len(struct ibv_cq_ex *current)
{
	return 1;
}

static uint32_t efa_unit_test_mock_ibv_read_src_qp(struct ibv_cq_ex *current)
{
	return 0;
}

static uint8_t efa_unit_test_mock_ibv_read_sl(struct ibv_cq_ex *current)
{
	return 0;
}

static uint32_t efa_unit_test_mock_ibv_read_slid(struct ibv_cq_ex *current)
{
	return 0;
}

static __be32 efa_unit_test_mock_ibv_read_imm_data(struct ibv_cq_ex *current)
{
	return 0;
}

static enum ibv_wc_opcode efa_unit_test_mock_ibv_read_opcode(struct ibv_cq_ex *current)
{
	return mock();
}

void efa_unit_test_ibv_cq_ex_use_mock(struct ibv_cq_ex *ibv_cq_ex)
{
	ibv_cq_ex->start_poll = &efa_unit_test_mock_ibv_start_poll;
	ibv_cq_ex->next_poll = &efa_unit_test_mock_ibv_next_poll;
	ibv_cq_ex->end_poll = &efa_unit_test_mock_ibv_end_poll;
	ibv_cq_ex->read_vendor_err = &efa_unit_test_mock_ibv_read_vendor_err;
	ibv_cq_ex->read_wc_flags = &efa_unit_test_mock_ibv_read_wc_flags;
	ibv_cq_ex->read_qp_num = &efa_unit_test_mock_ibv_read_qp_num;
	ibv_cq_ex->read_byte_len = &efa_unit_test_mock_ibv_read_byte_len;
	ibv_cq_ex->read_src_qp = &efa_unit_test_mock_ibv_read_src_qp;
	ibv_cq_ex->read_sl = &efa_unit_test_mock_ibv_read_sl;
	ibv_cq_ex->read_slid = &efa_unit_test_mock_ibv_read_slid;
	ibv_cq_ex->read_imm_data = &efa_unit_test_mock_ibv_read_imm_data;
	ibv_cq_ex->read_opcode = &efa_unit_test_mock_ibv_read_opcode;
}

