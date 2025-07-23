#ifndef EFAGDA_H
#define EFAGDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_sub_cq {
	uint8_t *buf;
	uint32_t cqe_size;
	uint32_t queue_mask;
	uint32_t ref_cnt;
	uint32_t consumed_cnt;
	int phase;
};

struct efa_cq {
	size_t size;
	uint16_t cq_idx;
	uint8_t *buf;
	size_t buf_size;
	struct efa_sub_cq *sub_cq_arr;
	uint16_t num_sub_cqs;
	uint32_t *db;
	uint32_t next_poll_idx;
	uint32_t cc;
	uint8_t cmd_sn;
};

struct efa_wq {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t *db;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	/* Producer counter */
	uint32_t pc;
	int phase;
	uint16_t sub_cq_idx;
};

struct efa_rq {
	struct efa_wq wq;
	uint8_t *buf;
	size_t buf_size;
};

struct efa_sq {
	struct efa_wq wq;
	uint8_t *buf;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
	uint32_t max_batch_wr;
};

struct efa_qp {
	size_t rq_size;
	struct efa_sq sq;
	struct efa_rq rq;
};

struct ibv_wc {
	uint64_t wr_id;
	int status;
	int opcode;
	uint32_t vendor_err;
	uint32_t byte_len;
	/* When (wc_flags & IBV_WC_WITH_IMM): Immediate data in network byte
	 * order. When (wc_flags & IBV_WC_WITH_INV): Stores the invalidated
	 * rkey.
	 */
	union {
		uint32_t imm_data;
		uint32_t invalidated_rkey;
	};
	uint32_t qp_num;
	uint32_t src_qp;
	unsigned int wc_flags;
	uint16_t pkey_index;
	uint16_t slid;
	uint8_t sl;
	uint8_t dlid_path_bits;
};

// Opaque handle for CUDA stream
typedef struct CudaStreamHandle *cuda_stream_t;

// Pure C interface
cuda_stream_t cuda_create_stream(void);
void cuda_destroy_stream(cuda_stream_t stream);

struct efa_cq *efagda_create_cuda_cq(void *device_buffer, uint16_t num_sub_cqs,
				     uint32_t sub_cq_size, uint32_t cqe_size);
void efagda_destroy_cuda_cq(struct efa_cq *d_cq);
struct efa_qp *efagda_create_cuda_qp(uint8_t *sq_buffer, uint32_t sq_num_wqes,
				     uint32_t *sq_db, uint32_t sq_max_batch,
				     uint8_t *rq_buffer, uint32_t rq_num_wqes,
				     uint32_t *rq_db);
void efagda_destroy_cuda_qp(struct efa_qp *d_qp);
int efagda_poll_cq(struct efa_cq *d_cq, int nwc, struct ibv_wc *h_wc,
		   cuda_stream_t stream);
int efagda_post_send(struct efa_qp *d_qp, uint16_t ah, uint16_t remote_qpn,
		     uint32_t remote_qkey, uint64_t addr, uint32_t length,
		     uint32_t lkey, cuda_stream_t stream);
int efagda_post_recv(struct efa_qp *d_qp, uint64_t addr, uint32_t length,
		     uint32_t lkey, cuda_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
