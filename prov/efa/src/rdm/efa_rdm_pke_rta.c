/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "ofi_iov.h"
#include "ofi_proto.h"
#include "ofi_atomic.h"
#include "efa.h"
#include "efa_hmem.h"
#include "efa_errno.h"
#include "efa_base_ep.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_rma.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pke_rta.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_utils.h"

/**
 * @brief initialize the common elements of WRITE_RTA, FETCH_RTA and COMPARE_RTA
 *
 * @param[in,out]	pkt_entry	packet entry
 * @param[in]		pkt_type	packet type. possible values are:
 * 					EFA_RDM_WRITE_RTA_PKT, EFA_RDM_FETCH_RTA_PKT and
 *					EFA_RDM_COMPARE_RTA_PKT
 * @param[in]		txe		TX entry that has information of the
 * 					atomic operation
 * @retunrns
 *
 * 0 on success.
 * negative libfabric error code on error. Possible error include:
 * 	-FI_ETRUNC	user buffer is larger than maxium atomic message size
 */
ssize_t efa_rdm_pke_init_rta_common(struct efa_rdm_pke *pkt_entry,
				    int pkt_type,
				    struct efa_rdm_ope *txe)
{
	struct efa_rma_iov *rma_iov;
	struct efa_rdm_rta_hdr *rta_hdr;
	char *data;
	size_t hdr_size, data_size;
	ssize_t ret;
	int i;

	rta_hdr = (struct efa_rdm_rta_hdr *)pkt_entry->wiredata;
	rta_hdr->msg_id = txe->msg_id;
	rta_hdr->rma_iov_count = txe->rma_iov_count;
	rta_hdr->atomic_datatype = txe->atomic_hdr.datatype;
	rta_hdr->atomic_op = txe->atomic_hdr.atomic_op;
	efa_rdm_pke_init_req_hdr_common(pkt_entry, pkt_type, txe);
	rta_hdr->flags |= EFA_RDM_REQ_ATOMIC;
	rma_iov = rta_hdr->rma_iov;
	for (i=0; i < txe->rma_iov_count; ++i) {
		rma_iov[i].addr = txe->rma_iov[i].addr;
		rma_iov[i].len = txe->rma_iov[i].len;
		rma_iov[i].key = txe->rma_iov[i].key;
	}

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);

	data = pkt_entry->wiredata + hdr_size;
	/**
	 * @todo use efa_rdm_pke_init_payload_from_ope() to set up
	 * payload, which can avoid copy under certain condition.
	 */
	if (txe->atomic_hdr.atomic_op == FI_ATOMIC_READ) {
		memset(data, 0, txe->total_len);
		data_size = txe->total_len;
	} else {
		ret = efa_copy_from_hmem_iov(txe->desc, data, txe->ep->mtu_size - hdr_size,
									 txe->iov, txe->iov_count);
		if (OFI_UNLIKELY(ret < 0))
			return ret;
		data_size = ret;
	}

	pkt_entry->pkt_size = hdr_size + data_size;
	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = txe->peer;
	return 0;
}

/**
 * @brief allocate a RX entry to process an incoming RTA packet
 *
 * @param[in]	pkt_entry	received RTA packet
 * @param[in]	op		libfabric operation type. Possible values are:
 * 				ofi_op_atomic, ofi_op_atomic_fetch, ofi_op_atomic_compare
 * @return
 * pointer to efa_rdm_ope on success.
 * NULL when rx entry pool is exhausted.
 */
struct efa_rdm_ope *efa_rdm_pke_alloc_rta_rxe(struct efa_rdm_pke *pkt_entry, int op)
{
	struct efa_rdm_ope *rxe;
	struct efa_rdm_rta_hdr *rta_hdr;

	rxe = efa_rdm_ep_alloc_rxe(pkt_entry->ep, pkt_entry->peer, op);
	if (OFI_UNLIKELY(!rxe)) {
		EFA_WARN(FI_LOG_CQ,
			"RX entries exhausted.\n");
		return NULL;
	}

	rxe->internal_flags |= EFA_RDM_OPE_INTERNAL;

	if (op == ofi_op_atomic)
		return rxe;

	rta_hdr = (struct efa_rdm_rta_hdr *)pkt_entry->wiredata;
	rxe->atomic_hdr.atomic_op = rta_hdr->atomic_op;
	rxe->atomic_hdr.datatype = rta_hdr->atomic_datatype;

	rxe->iov_count = rta_hdr->rma_iov_count;
	return rxe;
}

static int efa_rdm_pke_prepare_atomic_response_rxe(
	struct efa_rdm_pke *pkt_entry, struct efa_rdm_ope *rxe, uint32_t access){
	struct efa_rdm_rta_hdr *rta_hdr;
	int err;

	rta_hdr = efa_rdm_pke_get_rta_hdr(pkt_entry);
	err = efa_rdm_rma_verified_copy_iov(pkt_entry->ep, rta_hdr->rma_iov,
					    rxe->iov_count, access, rxe->iov, rxe->desc);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ, "RMA address verification failed.\n");
		return err;
	}

	rxe->total_len = ofi_total_iov_len(rxe->iov, rxe->iov_count);
	/*
	 * prepare a buffer to hold response data.
	 * Atomic_op operates on 3 data buffers:
	 *          local_data (input/output),
	 *          request_data (input),
	 *          response_data (output)
	 * The fact local data will be changed by atomic_op means
	 * response_data is not reproducible.
	 * Because sending response packet can fail due to
	 * -FI_EAGAIN, we need a buffer to hold response_data.
	 * The internal RXE owns the buffer through response retry and cleanup.
	 */
	rxe->atomrsp_data = ofi_buf_alloc(pkt_entry->ep->rx_atomrsp_pool);
	if (!rxe->atomrsp_data) {
		EFA_WARN(FI_LOG_CQ,
			"atomic response buffer pool exhausted.\n");
		return -FI_ENOBUFS;
	}

	return 0;
}

static int efa_rdm_pke_rta_handle_error(struct efa_rdm_pke *pkt_entry,
					struct efa_rdm_ope *rxe, int err, int prov_errno) {
	efa_rdm_rxe_handle_error(rxe, -err, prov_errno);
	efa_rdm_pke_release_rx(pkt_entry);
	assert(rxe->efa_outstanding_tx_ops == 0);
	efa_rdm_rxe_release(rxe);
	return err;
}

static int efa_rdm_atomic_validate_op_datatype(
	uint32_t op, uint32_t datatype, uint64_t flags){
	if (op >= OFI_SWAP_OP_LAST || datatype >= OFI_DATATYPE_CNT)
		return -FI_EINVAL;

	return ofi_atomic_valid(&efa_prov, datatype, op, flags)
		? -FI_EINVAL : FI_SUCCESS;
}

static int efa_rdm_pke_validate_fetch_compare_atomic(
	struct efa_rdm_pke *pkt_entry, struct efa_rdm_ope *rxe, uint64_t flags){
	uint32_t op = rxe->atomic_hdr.atomic_op;
	uint32_t datatype = rxe->atomic_hdr.datatype;
	int err;

	err = efa_rdm_atomic_validate_op_datatype(op, datatype, flags);
	if (OFI_UNLIKELY(err))
		goto invalid;

	return 0;

invalid:
	EFA_WARN(FI_LOG_CQ,
		 "Invalid atomic operation or datatype: op=%u datatype=%u "
		 "flags=%" PRIx64 "\n", op, datatype, flags);
	return efa_rdm_pke_rta_handle_error(pkt_entry, rxe, -FI_EINVAL,
					    FI_EFA_ERR_INVALID_DATATYPE);
}

/**
 * @brief initialize a WRITE_RTA packet
 *
 * @param[in,out]	pkt_entry	packet entry
 * @param[in]		txe		TX entry that has information of the
 * 					atomic operation
 * @returns
 *
 * 0 on success.
 * negative libfabric error code on error. Possible error include:
 * 	-FI_ETRUNC	user buffer is larger than maxium atomic message size
 */
ssize_t efa_rdm_pke_init_write_rta(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe)
{
	return efa_rdm_pke_init_rta_common(pkt_entry, EFA_RDM_WRITE_RTA_PKT, txe);
}

static int efa_rdm_pke_validate_write_atomic(struct efa_rdm_pke *pkt_entry){
	struct efa_rdm_rta_hdr *rta_hdr;
	uint32_t op, datatype;
	int err;

	rta_hdr = efa_rdm_pke_get_rta_hdr(pkt_entry);
	op = rta_hdr->atomic_op;
	datatype = rta_hdr->atomic_datatype;

	err = efa_rdm_atomic_validate_op_datatype(op, datatype, 0);
	if (OFI_UNLIKELY(err))
		goto invalid;

	return 0;

invalid:
	EFA_WARN(FI_LOG_CQ,
		 "Invalid write atomic operation or datatype: op=%u "
		 "datatype=%u\n",
		 op, datatype);
	efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep, FI_EINVAL,
				   FI_EFA_ERR_INVALID_DATATYPE);
	efa_rdm_pke_release_rx(pkt_entry);
	return -FI_EINVAL;
}

static
int efa_rdm_write_atomic_hmem(struct efa_mr *efa_mr, struct iovec *dst, char *data,
                                 size_t dtsize, int op, int dt)
{
	char *host_data = (char *) malloc(dst->iov_len);
	int err;

	/* Step 1: Copy data from device to temporary host buffer */
	err = efa_copy_from_hmem(efa_mr, host_data, dst->iov_base, dst->iov_len);
	if (OFI_UNLIKELY(err)) {
		free(host_data);
		return err;
	}

	/* Step 2: Perform atomic operation on host buffer */
	ofi_atomic_write_handlers[op][dt](host_data,
	                                  data,
	                                  dst->iov_len / dtsize);

	/* Step 3: Copy temporary host buffer to device */
	err = efa_copy_to_hmem(efa_mr, dst->iov_base, host_data, dst->iov_len);
	free(host_data);
	return err;
}

/**
 * @brief process a received WRITE RTA packet
 *
 * @param[in]	pkt_entry	received WRITE RTA packet
 */
int efa_rdm_pke_proc_write_rta(struct efa_rdm_pke *pkt_entry)
{
	struct iovec iov[EFA_RDM_IOV_LIMIT];
	struct efa_mr *efa_mr;
	struct efa_rdm_rta_hdr *rta_hdr;
	void *desc[EFA_RDM_IOV_LIMIT];
	char *data;
	int iov_count, op, dt, i, err;
	size_t dtsize, offset, hdr_size;
	enum fi_hmem_iface hmem_iface;

	rta_hdr = (struct efa_rdm_rta_hdr *)pkt_entry->wiredata;
	op = rta_hdr->atomic_op;
	dt = rta_hdr->atomic_datatype;
	err = efa_rdm_pke_validate_write_atomic(pkt_entry);
	if (OFI_UNLIKELY(err))
		return err;

	dtsize = ofi_datatype_size(dt);
	if (OFI_UNLIKELY(!dtsize)) {
		return -errno;
	}

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	data = pkt_entry->wiredata + hdr_size;
	iov_count = rta_hdr->rma_iov_count;
	err = efa_rdm_rma_verified_copy_iov(pkt_entry->ep, rta_hdr->rma_iov, iov_count, FI_REMOTE_WRITE, iov, desc);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ, "RMA address verification failed.\n");
		efa_base_ep_write_eq_error(
			&pkt_entry->ep->base_ep, -err, FI_EFA_ERR_RMA_ADDR);
		efa_rdm_pke_release_rx(pkt_entry);
		return err;
	}

	offset = 0;
	for (i = 0; i < iov_count; ++i) {
		/* Get hmem_iface from MR */
		efa_mr = (struct efa_mr*) ofi_mr_map_get(&pkt_entry->ep->base_ep.util_ep.domain->mr_map, (rta_hdr->rma_iov + i)->key);
		hmem_iface = efa_mr->iface;

		if (hmem_iface == FI_HMEM_SYSTEM) {
			ofi_atomic_write_handlers[op][dt](iov[i].iov_base,
			                                  data + offset,
			                                  iov[i].iov_len / dtsize);
		} else {
			err = efa_rdm_write_atomic_hmem(efa_mr, &iov[i], data + offset, dtsize, op, dt);
			if (OFI_UNLIKELY(err)) {
				return err;
			}
		}

		offset += iov[i].iov_len;
	}

	efa_rdm_pke_release_rx(pkt_entry);
	return 0;
}

/**
 * @brief initialize a DC_WRITE_RTA packet
 *
 * DC means delivery complete.
 * @param[in,out]	pkt_entry	packet entry
 * @param[in]		txe		TX entry that has information of the
 * 					atomic operation
 * @returns
 *
 * 0 on success.
 * negative libfabric error code on error. Possible error include:
 * 	-FI_ETRUNC	user buffer is larger than maxium atomic message size
 */
ssize_t efa_rdm_pke_init_dc_write_rta(struct efa_rdm_pke *pkt_entry,
				      struct efa_rdm_ope *txe)

{
	struct efa_rdm_rta_hdr *rta_hdr;
	ssize_t ret;

	txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	ret = efa_rdm_pke_init_rta_common(pkt_entry, EFA_RDM_DC_WRITE_RTA_PKT, txe);
	if (ret)
		return ret;

	rta_hdr = efa_rdm_pke_get_rta_hdr(pkt_entry);
	rta_hdr->send_id = txe->tx_id;
	return 0;
}

/**
 * @brief process a received DC WRITE RTA packet
 *
 * @param[in]	pkt_entry	received DC WRITE RTA packet
 */
int efa_rdm_pke_proc_dc_write_rta(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe;
	struct efa_rdm_rta_hdr *rta_hdr;
	ssize_t err;
	int ret;

	rxe = efa_rdm_pke_alloc_rta_rxe(pkt_entry, ofi_op_atomic);
	if (OFI_UNLIKELY(!rxe)) {
		efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(pkt_entry);
		return -FI_ENOBUFS;
	}

	rta_hdr = (struct efa_rdm_rta_hdr *)pkt_entry->wiredata;
	rxe->tx_id = rta_hdr->send_id;
	rxe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	ret = efa_rdm_pke_proc_write_rta(pkt_entry);
	if (OFI_UNLIKELY(ret)) {
		EFA_WARN(FI_LOG_CQ, "Error while processing the write rta packet\n");
		efa_rdm_rxe_release(rxe);
		return ret;
	}

	err = efa_rdm_ope_post_send_or_queue(rxe, EFA_RDM_RECEIPT_PKT);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ,
			"Posting of receipt packet failed! err=%s\n",
			fi_strerror(err));
		efa_rdm_rxe_handle_error(rxe, -err, FI_EFA_ERR_PKT_POST);
		return err;
	}

	return ret;
}

/**
 * @brief initialize a FETCH_RTA packet
 *
 * DC means delivery complete.
 * @param[in,out]	pkt_entry	packet entry
 * @param[in]		txe		TX entry that has information of the
 * 					atomic operation
 * @returns
 *
 * 0 on success.
 * negative libfabric error code on error. Possible error include:
 * 	-FI_ETRUNC	user buffer is larger than maxium atomic message size
 */
ssize_t efa_rdm_pke_init_fetch_rta(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe)

{
	struct efa_rdm_rta_hdr *rta_hdr;
	ssize_t ret;

	ret = efa_rdm_pke_init_rta_common(pkt_entry, EFA_RDM_FETCH_RTA_PKT, txe);
	if (ret)
		return ret;
	rta_hdr = efa_rdm_pke_get_rta_hdr(pkt_entry);
	rta_hdr->recv_id = txe->tx_id;
	return 0;
}

static
int efa_rdm_fetch_atomic_hmem(struct efa_mr *efa_mr, struct iovec *dst, char *data,
			      void* result, size_t dtsize, int op, int dt)
{
	char *host_data = (char *) malloc(dst->iov_len);
	int err;

	if (OFI_UNLIKELY(!host_data))
		return -FI_ENOMEM;

	/* Step 1: Copy data from device to temporary host buffer */
	err = efa_copy_from_hmem(efa_mr, host_data, dst->iov_base, dst->iov_len);
	if (OFI_UNLIKELY(err)) {
		free(host_data);
		return err;
	}

	/* Step 2: Perform atomic operation on temporary host buffer */
	ofi_atomic_readwrite_handlers[op][dt](host_data,
	                                      data,
	                                      result,
	                                      dst->iov_len / dtsize);

	if (op == FI_ATOMIC_READ) {
		free(host_data);
		return FI_SUCCESS;
	}

	/* Step 3: Copy data from host buffer to device */
	err = efa_copy_to_hmem(efa_mr, dst->iov_base, host_data, dst->iov_len);
	free(host_data);
	return err;
}

/**
 * @brief process a received FETCH RTA packet
 *
 * @param[in]	pkt_entry	received FETCH RTA packet
 */
int efa_rdm_pke_proc_fetch_rta(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct efa_mr *efa_mr;
	char *data;
	int op, dt, i;
	uint32_t access;
	size_t offset, dtsize;
	ssize_t err;
	enum fi_hmem_iface hmem_iface;

	ep = pkt_entry->ep;

	rxe = efa_rdm_pke_alloc_rta_rxe(pkt_entry, ofi_op_atomic_fetch);
	if(OFI_UNLIKELY(!rxe)) {
		efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep, FI_ENOBUFS,
					   FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(pkt_entry);
		return -FI_ENOBUFS;
	}

	rxe->tx_id = efa_rdm_pke_get_rta_hdr(pkt_entry)->recv_id;
	op = rxe->atomic_hdr.atomic_op;
	dt = rxe->atomic_hdr.datatype;
	err = efa_rdm_pke_validate_fetch_compare_atomic(pkt_entry, rxe, FI_FETCH_ATOMIC);
	if (OFI_UNLIKELY(err))
		return err;

	dtsize = ofi_datatype_size(rxe->atomic_hdr.datatype);
	if (OFI_UNLIKELY(!dtsize))
		return efa_rdm_pke_rta_handle_error(
			pkt_entry, rxe, -FI_EINVAL,
			FI_EFA_ERR_INVALID_DATATYPE);

	access = FI_REMOTE_READ;
	if (op != FI_ATOMIC_READ)
		access |= FI_REMOTE_WRITE;

	err = efa_rdm_pke_prepare_atomic_response_rxe(pkt_entry, rxe, access);
	if (OFI_UNLIKELY(err))
		return efa_rdm_pke_rta_handle_error(
			pkt_entry, rxe, err,
			err == -FI_ENOBUFS ? FI_EFA_ERR_RXE_POOL_EXHAUSTED : FI_EFA_ERR_RMA_ADDR);

	data = pkt_entry->wiredata + efa_rdm_pke_get_req_hdr_size(pkt_entry);

	offset = 0;
	for (i = 0; i < rxe->iov_count; ++i) {
		efa_mr = (struct efa_mr*) ofi_mr_map_get(&ep->base_ep.util_ep.domain->mr_map, (efa_rdm_pke_get_rta_hdr(pkt_entry)->rma_iov + i)->key);
		hmem_iface = efa_mr->iface;
		if (hmem_iface == FI_HMEM_SYSTEM) {
			ofi_atomic_readwrite_handlers[op][dt](rxe->iov[i].iov_base,
			                                      data + offset,
			                                      rxe->atomrsp_data + offset,
			                                      rxe->iov[i].iov_len / dtsize);
		} else {
			err = efa_rdm_fetch_atomic_hmem(efa_mr, &rxe->iov[i], data + offset,
							rxe->atomrsp_data + offset, dtsize, op, dt);
			if (OFI_UNLIKELY(err))
				return efa_rdm_pke_rta_handle_error(
					pkt_entry, rxe, err, FI_EFA_ERR_RXE_COPY);
		}

		offset += rxe->iov[i].iov_len;
	}

	err = efa_rdm_ope_post_send_or_queue(rxe, EFA_RDM_ATOMRSP_PKT);
	if (OFI_UNLIKELY(err))
		return efa_rdm_pke_rta_handle_error(
			pkt_entry, rxe, err, FI_EFA_ERR_PKT_POST);

	efa_rdm_pke_release_rx(pkt_entry);
	return 0;
}

/**
 * @brief initialize a COMPARE_RTA packet
 *
 * DC means delivery complete.
 * @param[in,out]	pkt_entry	packet entry
 * @param[in]		txe		TX entry that has information of the
 * 					atomic operation
 * @returns
 *
 * 0 on success.
 * negative libfabric error code on error. Possible error include:
 * 	-FI_ETRUNC	user buffer is larger than maxium atomic message size
 */
ssize_t efa_rdm_pke_init_compare_rta(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe)

{
	char *data;
	size_t data_size;
	ssize_t ret;
	struct efa_rdm_rta_hdr *rta_hdr;

	ret = efa_rdm_pke_init_rta_common(pkt_entry, EFA_RDM_COMPARE_RTA_PKT, txe);
	if (ret)
		return ret;
	rta_hdr = efa_rdm_pke_get_rta_hdr(pkt_entry);
	rta_hdr->recv_id = txe->tx_id;
	/* efa_rdm_pke_init_rta() will copy data from txe->iov to pkt entry
	 * the following append the data to be compared
	 */

	data = pkt_entry->wiredata + pkt_entry->pkt_size;
	ret = efa_copy_from_hmem_iov(txe->atomic_ex.compare_desc, data, txe->ep->mtu_size - pkt_entry->pkt_size,
	                             txe->atomic_ex.comp_iov, txe->atomic_ex.comp_iov_count);

	if (OFI_UNLIKELY(ret < 0))
		return ret;
	data_size = ret;

	if (OFI_UNLIKELY(data_size != txe->total_len))
		return -FI_EIO;
	pkt_entry->pkt_size += data_size;
	return 0;
}

static
int efa_rdm_compare_atomic_hmem(struct efa_mr *efa_mr, struct iovec *dst, char *src, void* res,
				void* cmp, size_t dtsize, int op, int dt)
{
	char *host_data = (char *) malloc(dst->iov_len);
	int err;

	if (OFI_UNLIKELY(!host_data))
		return -FI_ENOMEM;

	/* Step 1: Copy From HMEM into temp_host_buffer */
	err = efa_copy_from_hmem(efa_mr, host_data, dst->iov_base, dst->iov_len);
	if (OFI_UNLIKELY(err)) {
		free(host_data);
		return err;
	}

	/* Step 2: Perform the atomic operation on host buffer */
	ofi_atomic_swap_handler(op, dt, host_data, src, cmp, res, dst->iov_len / dtsize);

	/* Step 3: Copy host buffer back to device*/
	err = efa_copy_to_hmem(efa_mr, dst->iov_base, host_data, dst->iov_len);
	free(host_data);
	return err;
}

/**
 * @brief process a received COMPARE RTA packet
 *
 * @param[in]		pkt_entry	packet entry
 */
int efa_rdm_pke_proc_compare_rta(struct efa_rdm_pke *pkt_entry)
{
	struct efa_mr *efa_mr;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	char *src_data, *cmp_data;
	int op, dt, i;
	enum fi_hmem_iface hmem_iface;
	size_t offset, dtsize;
	ssize_t err;

	ep = pkt_entry->ep;

	rxe = efa_rdm_pke_alloc_rta_rxe(pkt_entry, ofi_op_atomic_compare);
	if(OFI_UNLIKELY(!rxe)) {
		efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(pkt_entry);
		return -FI_ENOBUFS;
	}

	rxe->tx_id = efa_rdm_pke_get_rta_hdr(pkt_entry)->recv_id;
	op = rxe->atomic_hdr.atomic_op;
	dt = rxe->atomic_hdr.datatype;
	err = efa_rdm_pke_validate_fetch_compare_atomic(pkt_entry, rxe, FI_COMPARE_ATOMIC);
	if (OFI_UNLIKELY(err))
		return err;

	dtsize = ofi_datatype_size(rxe->atomic_hdr.datatype);
	if (OFI_UNLIKELY(!dtsize))
		return efa_rdm_pke_rta_handle_error(
			pkt_entry, rxe, -FI_EINVAL, FI_EFA_ERR_INVALID_DATATYPE);

	err = efa_rdm_pke_prepare_atomic_response_rxe(
		pkt_entry, rxe, FI_REMOTE_READ | FI_REMOTE_WRITE);
	if (OFI_UNLIKELY(err))
		return efa_rdm_pke_rta_handle_error(
			pkt_entry, rxe, err, err == -FI_ENOBUFS ? FI_EFA_ERR_RXE_POOL_EXHAUSTED : FI_EFA_ERR_RMA_ADDR);

	src_data = pkt_entry->wiredata + efa_rdm_pke_get_req_hdr_size(pkt_entry);
	cmp_data = src_data + rxe->total_len;
	offset = 0;
	for (i = 0; i < rxe->iov_count; ++i) {
		efa_mr = (struct efa_mr*) ofi_mr_map_get(&ep->base_ep.util_ep.domain->mr_map, (efa_rdm_pke_get_rta_hdr(pkt_entry)->rma_iov + i)->key);
		hmem_iface = efa_mr->iface;

		if (hmem_iface == FI_HMEM_SYSTEM) {
			ofi_atomic_swap_handler(op, dt, rxe->iov[i].iov_base,
			                        src_data + offset,
			                        cmp_data + offset,
			                        rxe->atomrsp_data + offset,
			                        rxe->iov[i].iov_len / dtsize);
		} else {
			err = efa_rdm_compare_atomic_hmem(efa_mr, &rxe->iov[i], src_data + offset,
							  rxe->atomrsp_data + offset, cmp_data + offset,
							  dtsize, op, dt);
			if (OFI_UNLIKELY(err))
				return efa_rdm_pke_rta_handle_error(
					pkt_entry, rxe, err, FI_EFA_ERR_RXE_COPY);
		}
		offset += rxe->iov[i].iov_len;
	}

	err = efa_rdm_ope_post_send_or_queue(rxe, EFA_RDM_ATOMRSP_PKT);
	if (OFI_UNLIKELY(err))
		return efa_rdm_pke_rta_handle_error(
			pkt_entry, rxe, err,FI_EFA_ERR_PKT_POST);

	efa_rdm_pke_release_rx(pkt_entry);
	return 0;
}
