#include "rstream.h"

struct fi_tx_attr rstream_tx_attr = {
	.caps = RSTREAM_CAPS,
	.msg_order = FI_ORDER_SAS,
	.size = RSTREAM_DEFAULT_QP_SIZE,
};

struct fi_rx_attr rstream_rx_attr = {
	.caps = RSTREAM_CAPS,
	.msg_order = FI_ORDER_SAS,
	.size = RSTREAM_DEFAULT_QP_SIZE,
};

struct fi_ep_attr rstream_ep_attr = {
	.type = FI_EP_SOCK_STREAM,
	.protocol = FI_PROTO_RSTREAM,
	.protocol_version = 1,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
};

struct fi_domain_attr rstream_domain_attr = {
	.caps = FI_LOCAL_COMM | FI_REMOTE_COMM,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	/* for the ofi mr_check  */
	.mr_mode = 0,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
	.max_ep_tx_ctx = 1,
	.mr_iov_limit = 1,
};

struct fi_fabric_attr rstream_fabric_attr = {
	.prov_version = FI_VERSION(1, 6),
};

struct fi_info rstream_info = {
	.caps = RSTREAM_CAPS,
	.addr_format = FI_SOCKADDR,
	.tx_attr = &rstream_tx_attr,
	.rx_attr = &rstream_rx_attr,
	.ep_attr = &rstream_ep_attr,
	.domain_attr = &rstream_domain_attr,
	.fabric_attr = &rstream_fabric_attr
};

/* settings post CONNREQ for users */
void rstream_set_info(struct fi_info *info)
{
	info->caps = RSTREAM_CAPS;
	info->mode = 0;
	info->ep_attr->type = FI_EP_SOCK_STREAM;
	info->ep_attr->protocol = rstream_info.ep_attr->protocol;
	info->domain_attr->mr_mode = 0;
	info->domain_attr->mr_cnt = 0;
	*info->rx_attr = rstream_rx_attr;
	*info->tx_attr = rstream_tx_attr;
}
