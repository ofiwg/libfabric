#include "efa_unit_tests.h"
#include "rdm/rxr_pkt_type_base.h"
#include "rdm/rxr_pkt_type_misc.h"

struct fi_info *efa_unit_test_alloc_hints(enum fi_ep_type ep_type)
{
	struct fi_info *hints;

	hints = calloc(sizeof(struct fi_info), 1);
	if (!hints)
		return NULL;

	hints->domain_attr = calloc(sizeof(struct fi_domain_attr), 1);
	if (!hints->domain_attr) {
		fi_freeinfo(hints);
		return NULL;
	}

	hints->fabric_attr = calloc(sizeof(struct fi_fabric_attr), 1);
	if (!hints->fabric_attr) {
		fi_freeinfo(hints);
		return NULL;
	}

	hints->ep_attr = calloc(sizeof(struct fi_ep_attr), 1);
	if (!hints->ep_attr) {
		fi_freeinfo(hints);
		return NULL;
	}

	hints->fabric_attr->prov_name = "efa";
	hints->ep_attr->type = ep_type;

	hints->domain_attr->mr_mode |= FI_MR_LOCAL | FI_MR_ALLOCATED;
	if (ep_type == FI_EP_DGRAM) {
		hints->mode |= FI_MSG_PREFIX;
	}

	return hints;
}

void efa_unit_test_resource_construct(struct efa_resource *resource, enum fi_ep_type ep_type)
{
	int ret = 0;
	struct fi_av_attr av_attr = {0};
	struct fi_cq_attr cq_attr = {0};
	struct fi_eq_attr eq_attr = {0};

	resource->hints = efa_unit_test_alloc_hints(ep_type);
	if (!resource->hints)
		goto err;

	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info);
	if (ret)
		goto err;

	ret = fi_fabric(resource->info->fabric_attr, &resource->fabric, NULL);
	if (ret)
		goto err;

	ret = fi_domain(resource->fabric, resource->info, &resource->domain, NULL);
	if (ret)
		goto err;

	ret = fi_endpoint(resource->domain, resource->info, &resource->ep, NULL);
	if (ret)
		goto err;

	ret = fi_eq_open(resource->fabric, &eq_attr, &resource->eq, NULL);
	if (ret)
		goto err;

	fi_ep_bind(resource->ep, &resource->eq->fid, 0);

	ret = fi_av_open(resource->domain, &av_attr, &resource->av, NULL);
	if (ret)
		goto err;

	fi_ep_bind(resource->ep, &resource->av->fid, 0);

	ret = fi_cq_open(resource->domain, &cq_attr, &resource->cq, NULL);
	if (ret)
		goto err;

	fi_ep_bind(resource->ep, &resource->cq->fid, FI_SEND | FI_RECV);

	ret = fi_enable(resource->ep);
	if (ret)
		goto err;

	return;

err:
	efa_unit_test_resource_destruct(resource);

	/* Fail test early if the resource struct fails to initialize */
	assert_int_equal(ret, 0);
}

/**
 * @brief Clean up test resources.
 * Note: Resources should be destroyed in order.
 * @param[in] resource	struct efa_resource to clean up.
 */
void efa_unit_test_resource_destruct(struct efa_resource *resource)
{
	if (resource->ep) {
		assert_int_equal(fi_close(&resource->ep->fid), 0);
	}

	if (resource->eq) {
		assert_int_equal(fi_close(&resource->eq->fid), 0);
	}

	if (resource->cq) {
		assert_int_equal(fi_close(&resource->cq->fid), 0);
	}

	if (resource->av) {
		assert_int_equal(fi_close(&resource->av->fid), 0);
	}

	if (resource->domain) {
		assert_int_equal(fi_close(&resource->domain->fid), 0);
	}

	if (resource->fabric) {
		assert_int_equal(fi_close(&resource->fabric->fid), 0);
	}

	if (resource->info) {
		fi_freeinfo(resource->info);
	}
}

void efa_unit_test_buff_construct(struct efa_unit_test_buff *buff, struct efa_resource *resource, size_t buff_size)
{
	int err;

	buff->buff = calloc(buff_size, sizeof(uint8_t));
	assert_non_null(buff->buff);

	buff->size = buff_size;
	err = fi_mr_reg(resource->domain, buff->buff, buff_size, FI_SEND | FI_RECV,
			0 /*offset*/, 0 /*requested_key*/, 0 /*flags*/, &buff->mr, NULL);
	assert_int_equal(err, 0);
}

void efa_unit_test_buff_destruct(struct efa_unit_test_buff *buff)
{
	int err;

	assert_non_null(buff->mr);
	err = fi_close(&buff->mr->fid);
	assert_int_equal(err, 0);

	free(buff->buff);
}

/**
 * @brief Construct RXR_EAGER_MSGRTM_PKT
 *
 * @param[in] pkt_entry Packet entry. Must be non-NULL.
 * @param[in] attr Packet attributes.
 */
void efa_unit_test_eager_msgrtm_pkt_construct(struct rxr_pkt_entry *pkt_entry, struct efa_unit_test_eager_rtm_pkt_attr *attr)
{
	struct rxr_eager_msgrtm_hdr base_hdr = {0};
	struct rxr_req_opt_connid_hdr opt_connid_hdr = {0};
	uint32_t *connid = NULL;

	base_hdr.hdr.type = RXR_EAGER_MSGRTM_PKT;
	base_hdr.hdr.flags |= RXR_PKT_CONNID_HDR | RXR_REQ_MSG;
	base_hdr.hdr.msg_id = attr->msg_id;
	memcpy(pkt_entry->wiredata, &base_hdr, sizeof(struct rxr_eager_msgrtm_hdr));
	assert_int_equal(rxr_get_base_hdr(pkt_entry->wiredata)->type, RXR_EAGER_MSGRTM_PKT);
	assert_int_equal(rxr_pkt_req_base_hdr_size(pkt_entry), sizeof(struct rxr_eager_msgrtm_hdr));
	opt_connid_hdr.connid = attr->connid;
	memcpy(pkt_entry->wiredata + sizeof(struct rxr_eager_msgrtm_hdr), &opt_connid_hdr, sizeof(struct rxr_req_opt_connid_hdr));
	connid = rxr_pkt_connid_ptr(pkt_entry);
	assert_int_equal(*connid, attr->connid);
	pkt_entry->pkt_size = sizeof(base_hdr) + sizeof(opt_connid_hdr);
}

/**
 * @brief Construct RXR_HANDSHAKE_PKT
 *	The function will include the optional connid/host id headers if and only if
 *	attr->connid/host id are non-zero.
 *
 * @param[in,out] pkt_entry Packet entry. Must be non-NULL.
 * @param[in] attr Packet attributes.
 */
void efa_unit_test_handshake_pkt_construct(struct rxr_pkt_entry *pkt_entry, struct efa_unit_test_handshake_pkt_attr *attr)
{

	int nex = (RXR_NUM_EXTRA_FEATURE_OR_REQUEST - 1) / 64 + 1;
	struct rxr_handshake_hdr *handshake_hdr = (struct rxr_handshake_hdr *)pkt_entry->wiredata;

	handshake_hdr->type = RXR_HANDSHAKE_PKT;
	handshake_hdr->version = RXR_PROTOCOL_VERSION;
	handshake_hdr->nextra_p3 = nex + 3;
	handshake_hdr->flags = 0;

	calloc((uintptr_t)handshake_hdr->extra_info, nex * sizeof(uint64_t));
	pkt_entry->pkt_size = sizeof(struct rxr_handshake_hdr) + nex * sizeof(uint64_t);
	memcpy(pkt_entry->wiredata, handshake_hdr, sizeof(struct rxr_handshake_hdr));
	assert_int_equal(rxr_get_base_hdr(pkt_entry->wiredata)->type, RXR_HANDSHAKE_PKT);

	if (attr->connid) {
		struct rxr_handshake_opt_connid_hdr opt_connid_hdr = {0};
		opt_connid_hdr.connid = attr->connid;
		handshake_hdr->flags |= RXR_PKT_CONNID_HDR;
		memcpy(pkt_entry->wiredata + pkt_entry->pkt_size, &opt_connid_hdr, sizeof(struct rxr_handshake_opt_connid_hdr));
		assert_int_equal(*rxr_pkt_connid_ptr(pkt_entry), attr->connid);
		pkt_entry->pkt_size += sizeof(opt_connid_hdr);
	}

	if (attr->host_id) {
		struct rxr_handshake_opt_host_id_hdr opt_host_id_hdr = {0};
		opt_host_id_hdr.host_id = attr->host_id;
		handshake_hdr->flags |= RXR_HANDSHAKE_HOST_ID_HDR;
		memcpy(pkt_entry->wiredata + pkt_entry->pkt_size, &opt_host_id_hdr, sizeof(struct rxr_handshake_opt_host_id_hdr));
		assert_int_equal(*rxr_pkt_handshake_host_id_ptr(pkt_entry), attr->host_id);
		pkt_entry->pkt_size += sizeof(opt_host_id_hdr);
	}
}
