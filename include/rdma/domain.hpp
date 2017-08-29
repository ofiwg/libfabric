#pragma once

#include <rdma/fi_domain.h>
#include <rdma/fi_atomic.h> // fi_query_atomic

#include <rdma/ops.hpp>
#include <rdma/endpoint.hpp>

namespace fi {

class av : public ops<struct fid_av> {
public:
	av(struct fid_domain *domain, struct fi_av_attr *attr, void *context) {
		int ret = fi_av_open(domain, attr, &obj, context);
		init(ret);
	}
};

class cq : public ops<struct fid_cq> {
public:
	cq(struct fid_domain *domain, struct fi_cq_attr *attr, void *context) {
		int ret = fi_cq_open(domain, attr, &obj, context);
		init(ret);
	}

	// fi_ops_cq
	defmethod(cq_read, obj)
	defmethod(cq_readfrom, obj)
	defmethod(cq_readerr, obj)
	defmethod(cq_sread, obj)
	defmethod(cq_sreadfrom, obj)
	defmethod(cq_signal, obj)
	defmethod(cq_strerror, obj)
};

class domain : public ops<struct fid_domain> {
public:
	// TODO: should we also make a fi::fabric consuming ctor ?
	domain(struct fid_fabric *fabric, struct fi_info *info, void *context) {
		int ret = fi_domain(fabric, info, &obj, context);
		init(ret);
	}

	av av_open(struct fi_av_attr *attr, void *context) {
		return fi::av(obj, attr, context);
	}

	cq cq_open(struct fi_cq_attr *attr, void *context) {
		return fi::cq(obj, attr, context);
	}

	ep endpoint(struct fi_info *info, void *context) {
		return fi::ep(obj, info, context);
	}

	ep scalable_ep(struct fi_info *info, void *context) {
		return fi::sep(obj, info, context);
	}

	int query_atomic(enum fi_datatype datatype, enum fi_op op,
			struct fi_atomic_attr *attr,
			uint64_t flags) {
		return fi_query_atomic(obj, datatype, op, attr, flags);
	}

};

}
