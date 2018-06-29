#include "rstream.h"

static void rstream_iwarp_settings(struct fi_info *core_info)
{
	core_info->ep_attr->max_msg_size = 2147483647;
	core_info->domain_attr->cq_data_size = 0;
	core_info->domain_attr->mr_cnt = 2289662;
	core_info->mode = FI_CONTEXT;
}

static void rstream_default_settings(struct fi_info *core_info)
{
	core_info->mode = FI_RX_CQ_DATA | FI_CONTEXT;
	core_info->rx_attr->mode = FI_RX_CQ_DATA;
}

int rstream_info_to_core(uint32_t version, const struct fi_info *irstream_info,
	struct fi_info *core_info)
{
	core_info->ep_attr->type = FI_EP_MSG;
	core_info->ep_attr->protocol = FI_PROTO_UNSPEC;
	core_info->caps = FI_RMA | FI_MSG;
	core_info->domain_attr->caps = FI_LOCAL_COMM | FI_REMOTE_COMM;
	core_info->domain_attr->mr_mode = FI_MR_LOCAL | OFI_MR_BASIC_MAP;
	core_info->tx_attr->op_flags = FI_COMPLETION;
	core_info->rx_attr->op_flags = FI_COMPLETION;
	core_info->fabric_attr->api_version =  FI_VERSION(1, 6);
	core_info->fabric_attr->prov_version = FI_VERSION(1, 0);
	(RSTREAM_USING_IWARP) ? rstream_iwarp_settings(core_info):
		rstream_default_settings(core_info);

	return 0;
}

static void update_rstream_info(const struct fi_info *core_info)
{
	rstream_info.tx_attr->iov_limit = core_info->tx_attr->iov_limit;
	rstream_info.rx_attr->iov_limit = core_info->rx_attr->iov_limit;
	rstream_info.tx_attr->size = core_info->tx_attr->size;
	rstream_info.rx_attr->size = core_info->rx_attr->size;
	rstream_info.domain_attr->max_ep_rx_ctx =
		core_info->domain_attr->max_ep_rx_ctx;
	rstream_info.domain_attr->max_ep_srx_ctx =
		core_info->domain_attr->max_ep_srx_ctx;
	rstream_info.ep_attr->max_msg_size =
		core_info->ep_attr->max_msg_size;
	rstream_info.rx_attr->iov_limit = core_info->rx_attr->iov_limit;
	rstream_info.domain_attr->cq_data_size =
		core_info->domain_attr->cq_data_size;
	rstream_info.domain_attr->cq_cnt = core_info->domain_attr->cq_cnt;
	rstream_info.domain_attr->ep_cnt = core_info->domain_attr->ep_cnt;
	rstream_info.domain_attr->max_err_data =
		core_info->domain_attr->max_err_data;
}

int rstream_info_to_rstream(uint32_t version, const struct fi_info *core_info,
	struct fi_info *info)
{
	info->caps = RSTREAM_CAPS;
	info->mode = 0;

	*info->tx_attr = *rstream_info.tx_attr;
	*info->rx_attr = *rstream_info.rx_attr;
	*info->domain_attr = *rstream_info.domain_attr;
	*info->ep_attr = *rstream_info.ep_attr;
	info->fabric_attr->api_version = FI_VERSION(1, 6);
	info->fabric_attr->prov_version = FI_VERSION(1, 6);
	update_rstream_info(core_info);

	return 0;
}

static int rstream_getinfo(uint32_t version, const char *node,
	const char *service, uint64_t flags, const struct fi_info *hints,
	struct fi_info **info)
{
	int ret;

	if (!info)
		return -FI_EINVAL;

	if (hints && hints->ep_attr->protocol == FI_PROTO_IWARP) {
		rstream_info.ep_attr->protocol = FI_PROTO_IWARP;
		rstream_info.tx_attr->iov_limit = 3;
		rstream_info.rx_attr->iov_limit = 3;
		rstream_info.domain_attr->max_ep_srx_ctx = 0;
	}

	ret = ofix_getinfo(version, node, service, flags, &rstream_util_prov,
		hints, rstream_info_to_core, rstream_info_to_rstream, info);

	return ret;
}

static void rstream_fini(void)
{
	/* yawn */
}

struct fi_provider rstream_prov = {
	.name = OFI_UTIL_PREFIX "rstream",
	.version = FI_VERSION(1 ,6),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = rstream_getinfo,
	.fabric = rstream_fabric_open,
	.cleanup = rstream_fini
};

struct util_prov rstream_util_prov = {
	.prov = &rstream_prov,
	.info = &rstream_info,
	.flags = 0,
};

RSTREAM_INI
{
	return &rstream_prov;
}
