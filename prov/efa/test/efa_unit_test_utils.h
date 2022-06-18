#include <cmocka.h>
#include <infiniband/verbs.h>
#include "efa.h"

/* Mock functions to replace function pointers */
static int _start_poll(struct ibv_cq_ex *ibvcqx,
                       struct ibv_poll_cq_attr *attr) { return mock(); }
static int _next_poll(struct ibv_cq_ex *ibvcqx) { return mock(); }
static void _end_poll(struct ibv_cq_ex *ibvcqx) { mock(); }
static uint32_t _read_vendor_err(struct ibv_cq_ex *current) { return 1; }
static unsigned int _read_wc_flags(struct ibv_cq_ex *current) { return 0; }
static uint32_t _read_qp_num(struct ibv_cq_ex *current) { return 0; }
static uint32_t _read_byte_len(struct ibv_cq_ex *current) { return 1; }
static uint32_t _read_src_qp(struct ibv_cq_ex *current) { return 0; }
static uint8_t _read_sl(struct ibv_cq_ex *current) { return 0; }
static uint32_t _read_slid(struct ibv_cq_ex *current) { return 0; }
static __be32 _read_imm_data(struct ibv_cq_ex *current) { return 0; }
static enum ibv_wc_opcode _read_send_code(struct ibv_cq_ex *current) { return IBV_WC_SEND; }
static enum ibv_wc_opcode _read_opcode(struct ibv_cq_ex *current) { return mock(); }
static void _read_entry(struct efa_wc *wc, int index, void *buf) {}
static ssize_t _eq_write_successful(struct fid_eq *eq, uint32_t event,
                                    const void *buf, size_t len, uint64_t flags)
{
    check_expected(eq);
    return len;
};

static void efa_unit_test_ibv_cq_ex_update_func_ptr(struct ibv_cq_ex *ibv_cq_ex)
{
    ibv_cq_ex->start_poll = &_start_poll;
    ibv_cq_ex->next_poll = &_next_poll;
    ibv_cq_ex->end_poll = &_end_poll;
    ibv_cq_ex->read_vendor_err = &_read_vendor_err;
    ibv_cq_ex->read_wc_flags = &_read_wc_flags;
    ibv_cq_ex->read_qp_num = &_read_qp_num;
    ibv_cq_ex->read_byte_len = &_read_byte_len;
    ibv_cq_ex->read_src_qp = &_read_src_qp;
    ibv_cq_ex->read_sl = &_read_sl;
    ibv_cq_ex->read_slid = &_read_slid;
    ibv_cq_ex->read_imm_data = &_read_imm_data;
    ibv_cq_ex->read_opcode = &_read_opcode;
}