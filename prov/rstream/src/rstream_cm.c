#include "rstream.h"

static void rstream_format_data(struct rstream_cm_data *cm,
	const struct rstream_ep *ep)
{
	assert(cm && ep->local_mr.rx.data_start);

	cm->version = RSTREAM_RSOCKETV2;
	cm->max_rx_credits = htons(ep->qp_win.max_rx_credits);
	cm->base_addr = htonll((uintptr_t)ep->local_mr.rx.data_start);
	cm->rkey = htonll(ep->local_mr.rkey);
	cm->rmr_size = htonl(ep->local_mr.rx.size);
}

static int rstream_setname(fid_t fid, void *addr, size_t addrlen)
{
	fid_t rstream_fid;
	struct rstream_pep *rstream_pep;
	struct rstream_ep *rstream_ep;

	if (fid->fclass == FI_CLASS_PEP) {
		rstream_pep = container_of(fid, struct rstream_pep,
			util_pep.pep_fid);
		rstream_fid = &rstream_pep->pep_fd->fid;
	} else if (fid->fclass == FI_CLASS_EP) {
		rstream_ep = container_of(fid, struct rstream_ep,
			util_ep.ep_fid);
		rstream_fid = &rstream_ep->ep_fd->fid;
	} else {
		return -FI_ENOSYS;
	}

	return fi_setname(rstream_fid, addr, addrlen);
}

static int rstream_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	struct rstream_ep *rstream_ep =
		container_of(ep, struct rstream_ep, util_ep.ep_fid);

	return fi_getpeer(rstream_ep->ep_fd, addr, addrlen);
}

static int rstream_check_cm_size(struct rstream_ep *ep)
{
	int ret;
	size_t cm_max_size = 0, opt_size = sizeof(size_t);

	ret = fi_getopt(&ep->ep_fd->fid, FI_OPT_ENDPOINT, FI_OPT_CM_DATA_SIZE,
		&cm_max_size, &opt_size);
	if (ret < 0)
		return ret;
	if (cm_max_size < sizeof(struct rstream_cm_data))
		return -FI_ETOOSMALL;
	return ret;
}

static int rstream_connect(struct fid_ep *ep, const void *addr,
	const void *param, size_t paramlen)
{
	struct rstream_ep *rstream_ep =
		container_of(ep, struct rstream_ep, util_ep.ep_fid);
	struct rstream_cm_data cm_data;

	if (param || paramlen > 0 || rstream_check_cm_size(rstream_ep) != 0)
		return -FI_ENOSYS;

	rstream_format_data(&cm_data, rstream_ep);

	return fi_connect(rstream_ep->ep_fd, addr, &cm_data, sizeof(cm_data));
}

static int rstream_listen(struct fid_pep *pep)
{
	struct rstream_pep *rstream_pep = container_of(pep,
		struct rstream_pep, util_pep.pep_fid);

	return fi_listen(rstream_pep->pep_fd);
}

static int rstream_accept(struct fid_ep *ep, const void *param,
	size_t paramlen)
{
	struct rstream_cm_data cm_data;
	struct rstream_ep *rstream_ep =
		container_of(ep, struct rstream_ep, util_ep.ep_fid);

	if (param || paramlen > 0 || rstream_check_cm_size(rstream_ep) != 0)
		return -FI_ENOSYS;

	rstream_format_data(&cm_data, rstream_ep);

	return fi_accept(rstream_ep->ep_fd, &cm_data, sizeof(cm_data));
}

static int rstream_reject(struct fid_pep *pep, fid_t handle,
	const void *param, size_t paramlen)
{
	return -FI_ENOSYS;
}

static int rstream_shutdown(struct fid_ep *ep, uint64_t flags)
{
	return -FI_ENOSYS;
}

struct fi_ops_cm rstream_ops_pep_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rstream_setname,
	.getname = fi_no_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = rstream_listen,
	.accept = fi_no_accept,
	.reject = rstream_reject,
	.shutdown = rstream_shutdown,
	.join = fi_no_join,
};

struct fi_ops_cm rstream_ops_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = fi_no_getname,
	.getpeer = rstream_getpeer,
	.connect = rstream_connect,
	.listen = fi_no_listen,
	.accept = rstream_accept,
	.reject = fi_no_reject,
	.shutdown = rstream_shutdown,
	.join = fi_no_join,
};
