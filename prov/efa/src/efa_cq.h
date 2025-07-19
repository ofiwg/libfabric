/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef _EFA_CQ_H
#define _EFA_CQ_H

#include "efa.h"

#include "efa_cqdirect_structs.h"
enum ibv_cq_ex_type {
	IBV_CQ,
	EFADV_CQ
};

struct efa_ibv_cq {
	struct ibv_cq_ex *ibv_cq_ex;
	enum ibv_cq_ex_type ibv_cq_ex_type;
	struct efa_ibv_cq_ops *ops;
	bool cqdirect_enabled;
#if HAVE_EFADV_QUERY_CQ
	struct efa_cqdirect_cq cqdirect;
#endif
	int (*start_poll)(struct efa_ibv_cq *ibv_cq,
			  struct ibv_poll_cq_attr *attr);
	int (*next_poll)(struct efa_ibv_cq *ibv_cq);
	enum ibv_wc_opcode (*read_opcode)(struct efa_ibv_cq *ibv_cq);
	void (*end_poll)(struct efa_ibv_cq *ibv_cq);
	uint32_t (*read_qp_num)(struct efa_ibv_cq *ibv_cq);
	uint32_t (*read_vendor_err)(struct efa_ibv_cq *ibv_cq);
	uint32_t (*read_src_qp)(struct efa_ibv_cq *ibv_cq);
	uint32_t (*read_slid)(struct efa_ibv_cq *ibv_cq);
	uint32_t (*read_byte_len)(struct efa_ibv_cq *ibv_cq);
	unsigned int (*read_wc_flags)(struct efa_ibv_cq *ibv_cq);
	__be32 (*read_imm_data)(struct efa_ibv_cq *ibv_cq);
	bool (*wc_is_unsolicited)(struct efa_ibv_cq *ibv_cq);
	int (*read_sgid)(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid);
};

struct efa_ibv_cq_poll_list_entry {
	struct dlist_entry	entry;
	struct efa_ibv_cq	*cq;
};

struct efa_cq {
	struct util_cq		util_cq;
	struct efa_ibv_cq	ibv_cq;
	int	(*poll_ibv_cq)(ssize_t cqe_to_progress, struct efa_ibv_cq *ibv_cq);
};

extern struct fi_ops_cq efa_cq_ops;

extern struct fi_ops efa_cq_fi_ops;

/* Default ibv cq ops that use rdma-core */
static inline int efa_ibv_start_poll(struct efa_ibv_cq *ibv_cq, struct ibv_poll_cq_attr *attr)
{
    return ibv_start_poll(ibv_cq->ibv_cq_ex, attr);
}

static inline int efa_ibv_next_poll(struct efa_ibv_cq *ibv_cq)
{
    return ibv_next_poll(ibv_cq->ibv_cq_ex);
}

static inline enum ibv_wc_opcode efa_ibv_wc_read_opcode(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_opcode(ibv_cq->ibv_cq_ex);
}

static inline void efa_ibv_end_poll(struct efa_ibv_cq *ibv_cq)
{
    ibv_end_poll(ibv_cq->ibv_cq_ex);
}

static inline uint32_t efa_ibv_wc_read_qp_num(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_qp_num(ibv_cq->ibv_cq_ex);
}

static inline uint32_t efa_ibv_wc_read_vendor_err(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_vendor_err(ibv_cq->ibv_cq_ex);
}

static inline uint32_t efa_ibv_wc_read_slid(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_slid(ibv_cq->ibv_cq_ex);
}

static inline uint32_t efa_ibv_wc_read_src_qp(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_src_qp(ibv_cq->ibv_cq_ex);
}

static inline uint32_t efa_ibv_wc_read_byte_len(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_byte_len(ibv_cq->ibv_cq_ex);
}

static inline unsigned int efa_ibv_wc_read_wc_flags(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_wc_flags(ibv_cq->ibv_cq_ex);
}

static inline __be32 efa_ibv_wc_read_imm_data(struct efa_ibv_cq *ibv_cq)
{
    return ibv_wc_read_imm_data(ibv_cq->ibv_cq_ex);
}

#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
static inline bool efa_ibv_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
    return efadv_wc_is_unsolicited(efadv_cq_from_ibv_cq_ex(ibv_cq->ibv_cq_ex));
}
#else
static inline bool efa_ibv_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
	return false;
}
#endif /* HAVE_CAPS_UNSOLICITED_WRITE_RECV */

/*
 * Control header with completion data. CQ data length is static.
 */
#define EFA_CQ_DATA_SIZE (4)

static inline
int efa_ibv_cq_poll_list_match(struct dlist_entry *entry, const void *cq)
{
	struct efa_ibv_cq_poll_list_entry *item;
	item = container_of(entry, struct efa_ibv_cq_poll_list_entry, entry);
	return (item->cq == cq);
}


static inline
int efa_ibv_cq_poll_list_insert(struct dlist_entry *poll_list, struct ofi_genlock *lock, struct efa_ibv_cq *cq)
{
	int ret = 0;
	struct dlist_entry *entry;
	struct efa_ibv_cq_poll_list_entry *item;

	ofi_genlock_lock(lock);
	entry = dlist_find_first_match(poll_list, efa_ibv_cq_poll_list_match, cq);
	if (entry) {
		ret = -FI_EALREADY;
		goto out;
	}

	item = calloc(1, sizeof(*item));
	if (!item) {
		ret = -FI_ENOMEM;
		goto out;
	}

	item->cq = cq;
	dlist_insert_tail(&item->entry, poll_list);

out:
	ofi_genlock_unlock(lock);
	return (!ret || (ret == -FI_EALREADY)) ? 0 : ret;
}

static inline
void efa_ibv_cq_poll_list_remove(struct dlist_entry *poll_list, struct ofi_genlock *lock,
		      struct efa_ibv_cq *cq)
{
	struct efa_ibv_cq_poll_list_entry *item;
	struct dlist_entry *entry;

	ofi_genlock_lock(lock);
	entry = dlist_remove_first_match(poll_list, efa_ibv_cq_poll_list_match, cq);
	ofi_genlock_unlock(lock);

	if (entry) {
		item = container_of(entry, struct efa_ibv_cq_poll_list_entry, entry);
		free(item);
	}
}

/**
 * @brief Create ibv_cq_ex by calling ibv_create_cq_ex
 *
 * @param[in] ibv_cq_init_attr_ex Pointer to ibv_cq_init_attr_ex
 * @param[in] ibv_ctx Pointer to ibv_context
 * @param[in,out] ibv_cq_ex Pointer to newly created ibv_cq_ex
 * @param[in,out] ibv_cq_ex_type enum indicating if efadv_create_cq or ibv_create_cq_ex was used
 * @return Return 0 on success, error code otherwise
 */
static inline int efa_cq_open_ibv_cq_with_ibv_create_cq_ex(
	struct ibv_cq_init_attr_ex *ibv_cq_init_attr_ex,
	struct ibv_context *ibv_ctx, struct ibv_cq_ex **ibv_cq_ex,
	enum ibv_cq_ex_type *ibv_cq_ex_type)
{
	*ibv_cq_ex = ibv_create_cq_ex(ibv_ctx, ibv_cq_init_attr_ex);

	if (!*ibv_cq_ex) {
		EFA_WARN(FI_LOG_CQ, "Unable to create extended CQ: %s\n", strerror(errno));
		return -FI_EINVAL;
	}

	*ibv_cq_ex_type = IBV_CQ;
	return 0;
}

int efa_cq_open(struct fid_domain *domain_fid, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);

void efa_cq_progress(struct util_cq *cq);

int efa_cq_close(fid_t fid);

const char *efa_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
			    const void *err_data, char *buf, size_t len);

#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
/**
 * @brief Check whether a completion consumes recv buffer
 *
 * @param ibv_cq_ex extended ibv cq
 * @return true the wc consumes a recv buffer
 * @return false the wc doesn't consume a recv buffer
 */
static inline
bool efa_cq_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
	return efa_use_unsolicited_write_recv() && ibv_cq->wc_is_unsolicited(ibv_cq);
}

#else

static inline
bool efa_cq_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
	return false;
}

#endif

#if HAVE_EFADV_CQ_EX

static inline int efa_ibv_read_sgid(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid)
{
	return efadv_wc_read_sgid(efadv_cq_from_ibv_cq_ex(ibv_cq->ibv_cq_ex), sgid);
}

#else

static inline int efa_ibv_read_sgid(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid)
{
	return -FI_ENOSYS;
}

#endif

/**
 * @brief Write the error message and return its byte length
 * @param[in]    ep          EFA base endpoint
 * @param[in]    addr        Remote peer fi_addr_t
 * @param[in]    prov_errno  EFA provider * error code(must be positive)
 * @param[out]   err_msg     Pointer to the address of error message written by
 * this function
 * @param[out]   buflen      Pointer to the returned error data size
 * @return       A status code. 0 if the error data was written successfully,
 * otherwise a negative FI error code.
 */
static inline int efa_write_error_msg(struct efa_base_ep *ep, fi_addr_t addr,
				      int prov_errno, char *err_msg,
				      size_t *buflen)
{
	char ep_addr_str[OFI_ADDRSTRLEN] = {0}, peer_addr_str[OFI_ADDRSTRLEN] = {0};
	char peer_host_id_str[EFA_HOST_ID_STRING_LENGTH + 1] = {0};
	char local_host_id_str[EFA_HOST_ID_STRING_LENGTH + 1] = {0};
	const char *base_msg = efa_strerror(prov_errno);
	size_t len = 0;
	uint64_t local_host_id;

	*buflen = 0;

	len = sizeof(ep_addr_str);
	efa_base_ep_raw_addr_str(ep, ep_addr_str, &len);
	len = sizeof(peer_addr_str);
	efa_base_ep_get_peer_raw_addr_str(ep, addr, peer_addr_str, &len);

	local_host_id = efa_get_host_id(efa_env.host_id_file);
	if (!local_host_id ||
	    EFA_HOST_ID_STRING_LENGTH != snprintf(local_host_id_str,
						  EFA_HOST_ID_STRING_LENGTH + 1,
						  "i-%017lx", local_host_id)) {
		strcpy(local_host_id_str, "N/A");
	}

	/* efa-raw cannot get peer host id without a handshake */
	strcpy(peer_host_id_str, "N/A");

	int ret = snprintf(err_msg, EFA_ERROR_MSG_BUFFER_LENGTH,
			   "%s My EFA addr: %s My host id: %s Peer EFA addr: "
			   "%s Peer host id: %s",
			   base_msg, ep_addr_str, local_host_id_str,
			   peer_addr_str, peer_host_id_str);

	if (ret < 0 || ret > EFA_ERROR_MSG_BUFFER_LENGTH - 1) {
		return -FI_EINVAL;
	}

	if (strlen(err_msg) >= EFA_ERROR_MSG_BUFFER_LENGTH) {
		return -FI_ENOBUFS;
	}

	*buflen = EFA_ERROR_MSG_BUFFER_LENGTH;

	return 0;
}

int efa_cq_poll_ibv_cq(ssize_t cqe_to_process, struct efa_ibv_cq *ibv_cq);

int efa_cq_open_ibv_cq(struct fi_cq_attr *attr,
					struct ibv_context *ibv_ctx,
					struct efa_ibv_cq *ibv_cq,
					struct fi_efa_cq_init_attr *efa_cq_init_attr);

#endif /* end of _EFA_CQ_H*/