#ifndef _FI_BGQ_DIRECT_CM_H_
#define _FI_BGQ_DIRECT_CM_H_

#ifdef FABRIC_DIRECT
#define FABRIC_DIRECT_CM 1

static inline int fi_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	return ep->cm->getname(fid, addr, addrlen);
}

static inline int fi_listen(struct fid_pep *pep)
{
	return pep->cm->listen(pep);
}

static inline int
fi_connect(struct fid_ep *ep, const void *addr,
	   const void *param, size_t paramlen)
{
	return ep->cm->connect(ep, addr, param, paramlen);
}

static inline int
fi_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	return ep->cm->accept(ep, param, paramlen);
}

static inline int
fi_reject(struct fid_pep *pep, fid_t handle,
	  const void *param, size_t paramlen)
{
	return pep->cm->reject(pep, handle, param, paramlen);
}

static inline int fi_shutdown(struct fid_ep *ep, uint64_t flags)
{
	return ep->cm->shutdown(ep, flags);
}

#endif

#endif /* _FI_BGQ_DIRECT_CM_H_ */
