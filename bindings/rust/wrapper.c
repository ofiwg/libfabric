#include "wrapper.h"

/* Proprietary helper function declarations. */

fid_t get_fid_ptr(void *ptr)
{
	return (fid_t) ptr;
}

/* Static inline function declarations from fi_prov.h */

int wrap_fi_param_get_str(struct fi_provider *provider, const char *param_name,
			  char **value)
{
	return fi_param_get_str(provider, param_name, value);
}

int wrap_fi_param_get_int(struct fi_provider *provider, const char *param_name,
			  int *value)
{
	return fi_param_get_int(provider, param_name, value);
}

int wrap_fi_param_get_bool(struct fi_provider *provider, const char *param_name,
			   int *value)
{
	return fi_param_get_bool(provider, param_name, value);
}

int wrap_fi_param_get_size_t(struct fi_provider *provider,
			     const char *param_name, size_t *value)
{
	return fi_param_get_size_t(provider, param_name, value);
}

/* Static inline function declarations from fabric.h */

struct fi_info *wrap_fi_allocinfo(void)
{
	return fi_allocinfo();
}

int wrap_fi_close(struct fid *fid)
{
	return fi_close(fid);
}

int wrap_fi_control(struct fid *fid, int command, void *arg)
{
	return fi_control(fid, command, arg);
}

int wrap_fi_alias(struct fid *fid, struct fid **alias_fid, uint64_t flags)
{
	return fi_alias(fid, alias_fid, flags);
}

int wrap_fi_get_val(struct fid *fid, int name, void *val)
{
	return fi_get_val(fid, name, val);
}

int wrap_fi_set_val(struct fid *fid, int name, void *val)
{
	return fi_set_val(fid, name, val);
}

int wrap_fi_open_ops(struct fid *fid, const char *name, uint64_t flags,
		     void **ops, void *context)
{
	return fi_open_ops(fid, name, flags, ops, context);
}

int wrap_fi_set_ops(struct fid *fid, const char *name, uint64_t flags,
		    void *ops, void *context)
{
	return fi_set_ops(fid, name, flags, ops, context);
}

/* Static inline function declarations from fi_endpoint.h */

int wrap_fi_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		       struct fid_pep **pep, void *context)
{
	return fi_passive_ep(fabric, info, pep, context);
}

int wrap_fi_endpoint(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **ep, void *context)
{
	return fi_endpoint(domain, info, ep, context);
}

int wrap_fi_endpoint2(struct fid_domain *domain, struct fi_info *info,
		      struct fid_ep **ep, uint64_t flags, void *context)
{
	return fi_endpoint2(domain, info, ep, flags, context);
}

int wrap_fi_scalable_ep(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **sep, void *context)
{
	return fi_scalable_ep(domain, info, sep, context);
}

int wrap_fi_ep_bind(struct fid_ep *ep, struct fid *bfid, uint64_t flags)
{
	return fi_ep_bind(ep, bfid, flags);
}

int wrap_fi_pep_bind(struct fid_pep *pep, struct fid *bfid, uint64_t flags)
{
	return fi_pep_bind(pep, bfid, flags);
}

int wrap_fi_scalable_ep_bind(struct fid_ep *sep, struct fid *bfid,
			     uint64_t flags)
{
	return fi_scalable_ep_bind(sep, bfid, flags);
}

int wrap_fi_enable(struct fid_ep *ep)
{
	return fi_enable(ep);
}

ssize_t wrap_fi_cancel(fid_t fid, void *context)
{
	return fi_cancel(fid, context);
}

int wrap_fi_setopt(fid_t fid, int level, int optname, const void *optval,
		   size_t optlen)
{
	return fi_setopt(fid, level, optname, optval, optlen);
}

int wrap_fi_getopt(fid_t fid, int level, int optname, void *optval,
		   size_t *optlen)
{
	return fi_getopt(fid, level, optname, optval, optlen);
}

int wrap_fi_ep_alias(struct fid_ep *ep, struct fid_ep **alias_ep,
		     uint64_t flags)
{
	return fi_ep_alias(ep, alias_ep, flags);
}

int wrap_fi_tx_context(struct fid_ep *ep, int idx, struct fi_tx_attr *attr,
		       struct fid_ep **tx_ep, void *context)
{
	return fi_tx_context(ep, idx, attr, tx_ep, context);
}

int wrap_fi_rx_context(struct fid_ep *ep, int idx, struct fi_rx_attr *attr,
		       struct fid_ep **rx_ep, void *context)
{
	return fi_rx_context(ep, idx, attr, rx_ep, context);
}

ssize_t wrap_fi_rx_size_left(struct fid_ep *ep)
{
	return fi_rx_size_left(ep);
}

ssize_t wrap_fi_tx_size_left(struct fid_ep *ep)
{
	return fi_tx_size_left(ep);
}

int wrap_fi_stx_context(struct fid_domain *domain, struct fi_tx_attr *attr,
			struct fid_stx **stx, void *context)
{
	return fi_stx_context(domain, attr, stx, context);
}

int wrap_fi_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
			struct fid_ep **rx_ep, void *context)
{
	return fi_srx_context(domain, attr, rx_ep, context);
}

ssize_t wrap_fi_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
		     fi_addr_t src_addr, void *context)
{
	return fi_recv(ep, buf, len, desc, src_addr, context);
}

ssize_t wrap_fi_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		      size_t count, fi_addr_t src_addr, void *context)
{
	return fi_recvv(ep, iov, desc, count, src_addr, context);
}

ssize_t wrap_fi_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	return fi_recvmsg(ep, msg, flags);
}

ssize_t wrap_fi_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		     fi_addr_t dest_addr, void *context)
{
	return fi_send(ep, buf, len, desc, dest_addr, context);
}

ssize_t wrap_fi_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		      size_t count, fi_addr_t dest_addr, void *context)
{
	return fi_sendv(ep, iov, desc, count, dest_addr, context);
}

ssize_t wrap_fi_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	return fi_sendmsg(ep, msg, flags);
}

ssize_t wrap_fi_inject(struct fid_ep *ep, const void *buf, size_t len,
		       fi_addr_t dest_addr)
{
	return fi_inject(ep, buf, len, dest_addr);
}

ssize_t wrap_fi_senddata(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, uint64_t data, fi_addr_t dest_addr,
			 void *context)
{
	return fi_senddata(ep, buf, len, desc, data, dest_addr, context);
}

ssize_t wrap_fi_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			   uint64_t data, fi_addr_t dest_addr)
{
	return fi_injectdata(ep, buf, len, data, dest_addr);
}

/* Static inline function declarations from fi_atomic.h */

ssize_t wrap_fi_atomic(struct fid_ep *ep, const void *buf, size_t count,
		       void *desc, fi_addr_t dest_addr, uint64_t addr,
		       uint64_t key, enum fi_datatype datatype, enum fi_op op,
		       void *context)
{
	return fi_atomic(ep, buf, count, desc, dest_addr, addr, key, datatype,
			 op, context);
}

ssize_t wrap_fi_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
			void **desc, size_t count, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key, enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	return fi_atomicv(ep, iov, desc, count, dest_addr, addr, key, datatype,
			  op, context);
}

ssize_t wrap_fi_atomicmsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
			  uint64_t flags)
{
	return fi_atomicmsg(ep, msg, flags);
}

ssize_t wrap_fi_inject_atomic(struct fid_ep *ep, const void *buf, size_t count,
			      fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			      enum fi_datatype datatype, enum fi_op op)
{
	return fi_inject_atomic(ep, buf, count, dest_addr, addr, key, datatype,
				op);
}

ssize_t wrap_fi_fetch_atomic(struct fid_ep *ep, const void *buf, size_t count,
			     void *desc, void *result, void *result_desc,
			     fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			     enum fi_datatype datatype, enum fi_op op,
			     void *context)
{
	return fi_fetch_atomic(ep, buf, count, desc, result, result_desc,
			       dest_addr, addr, key, datatype, op, context);
}

ssize_t wrap_fi_fetch_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
			      void **desc, size_t count, struct fi_ioc *resultv,
			      void **result_desc, size_t result_count,
			      fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			      enum fi_datatype datatype, enum fi_op op,
			      void *context)
{
	return fi_fetch_atomicv(ep, iov, desc, count, resultv, result_desc,
				result_count, dest_addr, addr, key, datatype,
				op, context);
}

ssize_t wrap_fi_fetch_atomicmsg(struct fid_ep *ep,
				const struct fi_msg_atomic *msg,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count, uint64_t flags)
{
	return fi_fetch_atomicmsg(ep, msg, resultv, result_desc, result_count,
				  flags);
}

ssize_t wrap_fi_compare_atomic(struct fid_ep *ep, const void *buf, size_t count,
			       void *desc, const void *compare,
			       void *compare_desc, void *result,
			       void *result_desc, fi_addr_t dest_addr,
			       uint64_t addr, uint64_t key,
			       enum fi_datatype datatype, enum fi_op op,
			       void *context)
{
	return fi_compare_atomic(ep, buf, count, desc, compare, compare_desc,
				 result, result_desc, dest_addr, addr, key,
				 datatype, op, context);
}

ssize_t wrap_fi_compare_atomicv(
	struct fid_ep *ep, const struct fi_ioc *iov, void **desc, size_t count,
	const struct fi_ioc *comparev, void **compare_desc,
	size_t compare_count, struct fi_ioc *resultv, void **result_desc,
	size_t result_count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
	enum fi_datatype datatype, enum fi_op op, void *context)
{
	return fi_compare_atomicv(ep, iov, desc, count, comparev, compare_desc,
				  compare_count, resultv, result_desc,
				  result_count, dest_addr, addr, key, datatype,
				  op, context);
}

ssize_t wrap_fi_compare_atomicmsg(struct fid_ep *ep,
				  const struct fi_msg_atomic *msg,
				  const struct fi_ioc *comparev,
				  void **compare_desc, size_t compare_count,
				  struct fi_ioc *resultv, void **result_desc,
				  size_t result_count, uint64_t flags)
{
	return fi_compare_atomicmsg(ep, msg, comparev, compare_desc,
				    compare_count, resultv, result_desc,
				    result_count, flags);
}

int wrap_fi_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
			enum fi_op op, size_t *count)
{
	return fi_atomicvalid(ep, datatype, op, count);
}

int wrap_fi_fetch_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
			      enum fi_op op, size_t *count)
{
	return fi_fetch_atomicvalid(ep, datatype, op, count);
}

int wrap_fi_compare_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	return fi_compare_atomicvalid(ep, datatype, op, count);
}

int wrap_fi_query_atomic(struct fid_domain *domain, enum fi_datatype datatype,
			 enum fi_op op, struct fi_atomic_attr *attr,
			 uint64_t flags)
{
	return fi_query_atomic(domain, datatype, op, attr, flags);
}

/* Static inline function declarations from fi_domain.h */

int wrap_fi_domain(struct fid_fabric *fabric, struct fi_info *info,
		   struct fid_domain **domain, void *context)
{
	return fi_domain(fabric, info, domain, context);
}

int wrap_fi_domain2(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_domain **domain, uint64_t flags, void *context)
{
	return fi_domain2(fabric, info, domain, flags, context);
}

int wrap_fi_domain_bind(struct fid_domain *domain, struct fid *fid,
			uint64_t flags)
{
	return fi_domain_bind(domain, fid, flags);
}

int wrap_fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		    struct fid_cq **cq, void *context)
{
	return fi_cq_open(domain, attr, cq, context);
}

int wrap_fi_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		      struct fid_cntr **cntr, void *context)
{
	return fi_cntr_open(domain, attr, cntr, context);
}

int wrap_fi_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		      struct fid_wait **waitset)
{
	return fi_wait_open(fabric, attr, waitset);
}

int wrap_fi_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
		      struct fid_poll **pollset)
{
	return fi_poll_open(domain, attr, pollset);
}

int wrap_fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
		   uint64_t acs, uint64_t offset, uint64_t requested_key,
		   uint64_t flags, struct fid_mr **mr, void *context)
{
	return fi_mr_reg(domain, buf, len, acs, offset, requested_key, flags,
			 mr, context);
}

int wrap_fi_mr_regv(struct fid_domain *domain, const struct iovec *iov,
		    size_t count, uint64_t acs, uint64_t offset,
		    uint64_t requested_key, uint64_t flags, struct fid_mr **mr,
		    void *context)
{
	return fi_mr_regv(domain, iov, count, acs, offset, requested_key, flags,
			  mr, context);
}

int wrap_fi_mr_regattr(struct fid_domain *domain, const struct fi_mr_attr *attr,
		       uint64_t flags, struct fid_mr **mr)
{
	return fi_mr_regattr(domain, attr, flags, mr);
}

void *wrap_fi_mr_desc(struct fid_mr *mr)
{
	return fi_mr_desc(mr);
}

uint64_t wrap_fi_mr_key(struct fid_mr *mr)
{
	return fi_mr_key(mr);
}

int wrap_fi_mr_raw_attr(struct fid_mr *mr, uint64_t *base_addr,
			uint8_t *raw_key, size_t *key_size, uint64_t flags)
{
	return fi_mr_raw_attr(mr, base_addr, raw_key, key_size, flags);
}

int wrap_fi_mr_map_raw(struct fid_domain *domain, uint64_t base_addr,
		       uint8_t *raw_key, size_t key_size, uint64_t *key,
		       uint64_t flags)
{
	return fi_mr_map_raw(domain, base_addr, raw_key, key_size, key, flags);
}

int wrap_fi_mr_unmap_key(struct fid_domain *domain, uint64_t key)
{
	return fi_mr_unmap_key(domain, key);
}

int wrap_fi_mr_bind(struct fid_mr *mr, struct fid *bfid, uint64_t flags)
{
	return fi_mr_bind(mr, bfid, flags);
}

int wrap_fi_mr_refresh(struct fid_mr *mr, const struct iovec *iov, size_t count,
		       uint64_t flags)
{
	return fi_mr_refresh(mr, iov, count, flags);
}

int wrap_fi_mr_enable(struct fid_mr *mr)
{
	return fi_mr_enable(mr);
}

int wrap_fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		    struct fid_av **av, void *context)
{
	return fi_av_open(domain, attr, av, context);
}

int wrap_fi_av_bind(struct fid_av *av, struct fid *fid, uint64_t flags)
{
	return fi_av_bind(av, fid, flags);
}

int wrap_fi_av_insert(struct fid_av *av, const void *addr, size_t count,
		      fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	return fi_av_insert(av, addr, count, fi_addr, flags, context);
}

int wrap_fi_av_insertsvc(struct fid_av *av, const char *node,
			 const char *service, fi_addr_t *fi_addr,
			 uint64_t flags, void *context)
{
	return fi_av_insertsvc(av, node, service, fi_addr, flags, context);
}

int wrap_fi_av_insertsym(struct fid_av *av, const char *node, size_t nodecnt,
			 const char *service, size_t svccnt, fi_addr_t *fi_addr,
			 uint64_t flags, void *context)
{
	return fi_av_insertsym(av, node, nodecnt, service, svccnt, fi_addr,
			       flags, context);
}

int wrap_fi_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
		      uint64_t flags)
{
	return fi_av_remove(av, fi_addr, count, flags);
}

int wrap_fi_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
		      size_t *addrlen)
{
	return fi_av_lookup(av, fi_addr, addr, addrlen);
}

const char *wrap_fi_av_straddr(struct fid_av *av, const void *addr, char *buf,
			       size_t *len)
{
	return fi_av_straddr(av, addr, buf, len);
}

int wrap_fi_av_insert_auth_key(struct fid_av *av, const void *auth_key,
			       size_t auth_key_size, fi_addr_t *fi_addr,
			       uint64_t flags)
{
	return fi_av_insert_auth_key(av, auth_key, auth_key_size, fi_addr,
				     flags);
}

int wrap_fi_av_lookup_auth_key(struct fid_av *av, fi_addr_t addr,
			       void *auth_key, size_t *auth_key_size)
{
	return fi_av_lookup_auth_key(av, addr, auth_key, auth_key_size);
}

int wrap_fi_av_set_user_id(struct fid_av *av, fi_addr_t fi_addr,
			   fi_addr_t user_id, uint64_t flags)
{
	return fi_av_set_user_id(av, fi_addr, user_id, flags);
}

fi_addr_t wrap_fi_rx_addr(fi_addr_t fi_addr, int rx_index, int rx_ctx_bits)
{
	return fi_rx_addr(fi_addr, rx_index, rx_ctx_bits);
}

fi_addr_t wrap_fi_group_addr(fi_addr_t fi_addr, uint32_t group_id)
{
	return fi_group_addr(fi_addr, group_id);
}

/* Static inline function declarations from fi_cm.h */

int wrap_fi_setname(fid_t fid, void *addr, size_t addrlen)
{
	return fi_setname(fid, addr, addrlen);
}

int wrap_fi_getname(fid_t fid, void *addr, size_t *addrlen)
{
	return fi_getname(fid, addr, addrlen);
}

int wrap_fi_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	return fi_getpeer(ep, addr, addrlen);
}

int wrap_fi_listen(struct fid_pep *pep)
{
	return fi_listen(pep);
}

int wrap_fi_connect(struct fid_ep *ep, const void *addr, const void *param,
		    size_t paramlen)
{
	return fi_connect(ep, addr, param, paramlen);
}

int wrap_fi_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	return fi_accept(ep, param, paramlen);
}

int wrap_fi_reject(struct fid_pep *pep, fid_t handle, const void *param,
		   size_t paramlen)
{
	return fi_reject(pep, handle, param, paramlen);
}

int wrap_fi_shutdown(struct fid_ep *ep, uint64_t flags)
{
	return fi_shutdown(ep, flags);
}

int wrap_fi_join(struct fid_ep *ep, const void *addr, uint64_t flags,
		 struct fid_mc **mc, void *context)
{
	return fi_join(ep, addr, flags, mc, context);
}

fi_addr_t wrap_fi_mc_addr(struct fid_mc *mc)
{
	return fi_mc_addr(mc);
}

/* Static inline function declarations from fi_collective.h */

int wrap_fi_av_set(struct fid_av *av, struct fi_av_set_attr *attr,
		   struct fid_av_set **set, void *context)
{
	return fi_av_set(av, attr, set, context);
}

int wrap_fi_av_set_union(struct fid_av_set *dst, const struct fid_av_set *src)
{
	return fi_av_set_union(dst, src);
}

int wrap_fi_av_set_intersect(struct fid_av_set *dst,
			     const struct fid_av_set *src)
{
	return fi_av_set_intersect(dst, src);
}

int wrap_fi_av_set_diff(struct fid_av_set *dst, const struct fid_av_set *src)
{
	return fi_av_set_diff(dst, src);
}

int wrap_fi_av_set_insert(struct fid_av_set *set, fi_addr_t addr)
{
	return fi_av_set_insert(set, addr);
}

int wrap_fi_av_set_remove(struct fid_av_set *set, fi_addr_t addr)
{
	return fi_av_set_remove(set, addr);
}

int wrap_fi_av_set_addr(struct fid_av_set *set, fi_addr_t *coll_addr)
{
	return fi_av_set_addr(set, coll_addr);
}

int wrap_fi_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			    const struct fid_av_set *set, uint64_t flags,
			    struct fid_mc **mc, void *context)
{
	return fi_join_collective(ep, coll_addr, set, flags, mc, context);
}

ssize_t wrap_fi_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context)
{
	return fi_barrier(ep, coll_addr, context);
}

ssize_t wrap_fi_barrier2(struct fid_ep *ep, fi_addr_t coll_addr, uint64_t flags,
			 void *context)
{
	return fi_barrier2(ep, coll_addr, flags, context);
}

ssize_t wrap_fi_broadcast(struct fid_ep *ep, void *buf, size_t count,
			  void *desc, fi_addr_t coll_addr, fi_addr_t root_addr,
			  enum fi_datatype datatype, uint64_t flags,
			  void *context)
{
	return fi_broadcast(ep, buf, count, desc, coll_addr, root_addr,
			    datatype, flags, context);
}

ssize_t wrap_fi_alltoall(struct fid_ep *ep, const void *buf, size_t count,
			 void *desc, void *result, void *result_desc,
			 fi_addr_t coll_addr, enum fi_datatype datatype,
			 uint64_t flags, void *context)
{
	return fi_alltoall(ep, buf, count, desc, result, result_desc, coll_addr,
			   datatype, flags, context);
}

ssize_t wrap_fi_allreduce(struct fid_ep *ep, const void *buf, size_t count,
			  void *desc, void *result, void *result_desc,
			  fi_addr_t coll_addr, enum fi_datatype datatype,
			  enum fi_op op, uint64_t flags, void *context)
{
	return fi_allreduce(ep, buf, count, desc, result, result_desc,
			    coll_addr, datatype, op, flags, context);
}

ssize_t wrap_fi_allgather(struct fid_ep *ep, const void *buf, size_t count,
			  void *desc, void *result, void *result_desc,
			  fi_addr_t coll_addr, enum fi_datatype datatype,
			  uint64_t flags, void *context)
{
	return fi_allgather(ep, buf, count, desc, result, result_desc,
			    coll_addr, datatype, flags, context);
}

ssize_t wrap_fi_reduce_scatter(struct fid_ep *ep, const void *buf, size_t count,
			       void *desc, void *result, void *result_desc,
			       fi_addr_t coll_addr, enum fi_datatype datatype,
			       enum fi_op op, uint64_t flags, void *context)
{
	return fi_reduce_scatter(ep, buf, count, desc, result, result_desc,
				 coll_addr, datatype, op, flags, context);
}

ssize_t wrap_fi_reduce(struct fid_ep *ep, const void *buf, size_t count,
		       void *desc, void *result, void *result_desc,
		       fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, enum fi_op op, uint64_t flags,
		       void *context)
{
	return fi_reduce(ep, buf, count, desc, result, result_desc, coll_addr,
			 root_addr, datatype, op, flags, context);
}

ssize_t wrap_fi_scatter(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, void *result, void *result_desc,
			fi_addr_t coll_addr, fi_addr_t root_addr,
			enum fi_datatype datatype, uint64_t flags,
			void *context)
{
	return fi_scatter(ep, buf, count, desc, result, result_desc, coll_addr,
			  root_addr, datatype, flags, context);
}

ssize_t wrap_fi_gather(struct fid_ep *ep, const void *buf, size_t count,
		       void *desc, void *result, void *result_desc,
		       fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, uint64_t flags, void *context)
{
	return fi_gather(ep, buf, count, desc, result, result_desc, coll_addr,
			 root_addr, datatype, flags, context);
}

int wrap_fi_query_collective(struct fid_domain *domain,
			     enum fi_collective_op coll,
			     struct fi_collective_attr *attr, uint64_t flags)
{
	return fi_query_collective(domain, coll, attr, flags);
}

/* Static inline function declarations from fi_eq.h */

int wrap_fi_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	return fi_trywait(fabric, fids, count);
}

int wrap_fi_wait(struct fid_wait *waitset, int timeout)
{
	return fi_wait(waitset, timeout);
}

int wrap_fi_poll(struct fid_poll *pollset, void **context, int count)
{
	return fi_poll(pollset, context, count);
}

int wrap_fi_poll_add(struct fid_poll *pollset, struct fid *event_fid,
		     uint64_t flags)
{
	return fi_poll_add(pollset, event_fid, flags);
}

int wrap_fi_poll_del(struct fid_poll *pollset, struct fid *event_fid,
		     uint64_t flags)
{
	return fi_poll_del(pollset, event_fid, flags);
}

int wrap_fi_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		    struct fid_eq **eq, void *context)
{
	return fi_eq_open(fabric, attr, eq, context);
}

ssize_t wrap_fi_eq_read(struct fid_eq *eq, uint32_t *event, void *buf,
			size_t len, uint64_t flags)
{
	return fi_eq_read(eq, event, buf, len, flags);
}

ssize_t wrap_fi_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *buf,
			   uint64_t flags)
{
	return fi_eq_readerr(eq, buf, flags);
}

ssize_t wrap_fi_eq_write(struct fid_eq *eq, uint32_t event, const void *buf,
			 size_t len, uint64_t flags)
{
	return fi_eq_write(eq, event, buf, len, flags);
}

ssize_t wrap_fi_eq_sread(struct fid_eq *eq, uint32_t *event, void *buf,
			 size_t len, int timeout, uint64_t flags)
{
	return fi_eq_sread(eq, event, buf, len, timeout, flags);
}

const char *wrap_fi_eq_strerror(struct fid_eq *eq, int prov_errno,
				const void *err_data, char *buf, size_t len)
{
	return fi_eq_strerror(eq, prov_errno, err_data, buf, len);
}

ssize_t wrap_fi_cq_read(struct fid_cq *cq, void *buf, size_t count)
{
	return fi_cq_read(cq, buf, count);
}

ssize_t wrap_fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
			    fi_addr_t *src_addr)
{
	return fi_cq_readfrom(cq, buf, count, src_addr);
}

ssize_t wrap_fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
			   uint64_t flags)
{
	return fi_cq_readerr(cq, buf, flags);
}

ssize_t wrap_fi_cq_sread(struct fid_cq *cq, void *buf, size_t count,
			 const void *cond, int timeout)
{
	return fi_cq_sread(cq, buf, count, cond, timeout);
}

ssize_t wrap_fi_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
			     fi_addr_t *src_addr, const void *cond, int timeout)
{
	return fi_cq_sreadfrom(cq, buf, count, src_addr, cond, timeout);
}

int wrap_fi_cq_signal(struct fid_cq *cq)
{
	return fi_cq_signal(cq);
}

const char *wrap_fi_cq_strerror(struct fid_cq *cq, int prov_errno,
				const void *err_data, char *buf, size_t len)
{
	return fi_cq_strerror(cq, prov_errno, err_data, buf, len);
}

uint64_t wrap_fi_cntr_read(struct fid_cntr *cntr)
{
	return fi_cntr_read(cntr);
}

uint64_t wrap_fi_cntr_readerr(struct fid_cntr *cntr)
{
	return fi_cntr_readerr(cntr);
}

int wrap_fi_cntr_add(struct fid_cntr *cntr, uint64_t value)
{
	return fi_cntr_add(cntr, value);
}

int wrap_fi_cntr_adderr(struct fid_cntr *cntr, uint64_t value)
{
	return fi_cntr_adderr(cntr, value);
}

int wrap_fi_cntr_set(struct fid_cntr *cntr, uint64_t value)
{
	return fi_cntr_set(cntr, value);
}

int wrap_fi_cntr_seterr(struct fid_cntr *cntr, uint64_t value)
{
	return fi_cntr_seterr(cntr, value);
}

int wrap_fi_cntr_wait(struct fid_cntr *cntr, uint64_t threshold, int timeout)
{
	return fi_cntr_wait(cntr, threshold, timeout);
}

/* Static inline function declarations from fi_ext.h */

int wrap_fi_export_fid(struct fid *fid, uint64_t flags, struct fid **expfid,
		       void *context)
{
	return fi_export_fid(fid, flags, expfid, context);
}

int wrap_fi_import_fid(struct fid *fid, struct fid *expfid, uint64_t flags)
{
	return fi_import_fid(fid, expfid, flags);
}

int wrap_fi_import(uint32_t version, const char *name, void *attr,
		   size_t attr_len, uint64_t flags, struct fid *fid,
		   void *context)
{
	return fi_import(version, name, attr, attr_len, flags, fid, context);
}

int wrap_fi_import_log(uint32_t version, uint64_t flags,
		       struct fid_logging *log_fid)
{
	return fi_import_log(version, flags, log_fid);
}

/* Static inline function declarations from fi_ext.h */

void wrap_fi_profile_reset(struct fid_profile *prof_fid, uint64_t flags)
{
	return fi_profile_reset(prof_fid, flags);
}

ssize_t wrap_fi_profile_query_vars(struct fid_profile *prof_fid,
				   struct fi_profile_desc *varlist,
				   size_t *count)
{
	return fi_profile_query_vars(prof_fid, varlist, count);
}

ssize_t wrap_fi_profile_query_events(struct fid_profile *prof_fid,
				     struct fi_profile_desc *eventlist,
				     size_t *count)
{
	return fi_profile_query_events(prof_fid, eventlist, count);
}

ssize_t wrap_fi_profile_read_u64(struct fid_profile *prof_fid, uint32_t var_id,
				 uint64_t *data)
{
	return fi_profile_read_u64(prof_fid, var_id, data);
}

int wrap_fi_profile_register_callback(
	struct fid_profile *prof_fid, uint32_t event_id,
	int (*callback)(struct fid_profile *prof_fid,
			struct fi_profile_desc *event, void *param, size_t size,
			void *context),
	void *context)
{
	return fi_profile_register_callback(prof_fid, event_id, callback,
					    context);
}

void wrap_fi_profile_start_reads(struct fid_profile *prof_fid, uint64_t flags)
{
	return fi_profile_start_reads(prof_fid, flags);
}

void wrap_fi_profile_end_reads(struct fid_profile *prof_fid, uint64_t flags)
{
	return fi_profile_end_reads(prof_fid, flags);
}

int wrap_fi_profile_open(struct fid *fid, uint64_t flags,
			 struct fid_profile **prof_fid, void *context)
{
	return fi_profile_open(fid, flags, prof_fid, context);
}

int wrap_fi_profile_close(struct fid_profile *prof_fid)
{
	return fi_profile_close(prof_fid);
}

/* Static inline function declarations from fi_rma.h */

ssize_t wrap_fi_read(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		     fi_addr_t src_addr, uint64_t addr, uint64_t key,
		     void *context)
{
	return fi_read(ep, buf, len, desc, src_addr, addr, key, context);
}

ssize_t wrap_fi_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		      size_t count, fi_addr_t src_addr, uint64_t addr,
		      uint64_t key, void *context)
{
	return fi_readv(ep, iov, desc, count, src_addr, addr, key, context);
}

ssize_t wrap_fi_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	return fi_readmsg(ep, msg, flags);
}

ssize_t wrap_fi_write(struct fid_ep *ep, const void *buf, size_t len,
		      void *desc, fi_addr_t dest_addr, uint64_t addr,
		      uint64_t key, void *context)
{
	return fi_write(ep, buf, len, desc, dest_addr, addr, key, context);
}

ssize_t wrap_fi_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t dest_addr, uint64_t addr,
		       uint64_t key, void *context)
{
	return fi_writev(ep, iov, desc, count, dest_addr, addr, key, context);
}

ssize_t wrap_fi_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			 uint64_t flags)
{
	return fi_writemsg(ep, msg, flags);
}

ssize_t wrap_fi_inject_write(struct fid_ep *ep, const void *buf, size_t len,
			     fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	return fi_inject_write(ep, buf, len, dest_addr, addr, key);
}

ssize_t wrap_fi_writedata(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key, void *context)
{
	return fi_writedata(ep, buf, len, desc, data, dest_addr, addr, key,
			    context);
}

ssize_t wrap_fi_inject_writedata(struct fid_ep *ep, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr,
				 uint64_t addr, uint64_t key)
{
	return fi_inject_writedata(ep, buf, len, data, dest_addr, addr, key);
}

/* Static inline function declarations from fi_tagged.h */

ssize_t wrap_fi_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
		      fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		      void *context)
{
	return fi_trecv(ep, buf, len, desc, src_addr, tag, ignore, context);
}

ssize_t wrap_fi_trecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t src_addr, uint64_t tag,
		       uint64_t ignore, void *context)
{
	return fi_trecvv(ep, iov, desc, count, src_addr, tag, ignore, context);
}

ssize_t wrap_fi_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			 uint64_t flags)
{
	return fi_trecvmsg(ep, msg, flags);
}

ssize_t wrap_fi_tsend(struct fid_ep *ep, const void *buf, size_t len,
		      void *desc, fi_addr_t dest_addr, uint64_t tag,
		      void *context)
{
	return fi_tsend(ep, buf, len, desc, dest_addr, tag, context);
}

ssize_t wrap_fi_tsendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t dest_addr, uint64_t tag,
		       void *context)
{
	return fi_tsendv(ep, iov, desc, count, dest_addr, tag, context);
}

ssize_t wrap_fi_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			 uint64_t flags)
{
	return fi_tsendmsg(ep, msg, flags);
}

ssize_t wrap_fi_tinject(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t tag)
{
	return fi_tinject(ep, buf, len, dest_addr, tag);
}

ssize_t wrap_fi_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t tag, void *context)
{
	return fi_tsenddata(ep, buf, len, desc, data, dest_addr, tag, context);
}

ssize_t wrap_fi_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
			    uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	return fi_tinjectdata(ep, buf, len, data, dest_addr, tag);
}
