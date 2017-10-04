#pragma once

#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_atomic.h>

#include <rdma/ops.hpp>

namespace fi {

class mc : public ops<struct fid_mc> {
public:
	// mutlicast object just exports an address
	fi_addr_t addr;

	mc(struct fid_ep *ep, const void *addr, uint64_t flags, void *context) {
		int ret = fi_join(ep, addr, flags, &obj, context);
		init(ret);
		this->addr = obj->fi_addr;
	}
};

class pep : public ops<struct fid_pep> {
public:
	pep(struct fid_fabric *fabric, struct fi_info *info, void *context) {
		int ret = fi_passive_ep(fabric, info, &obj, context);
		init(ret);
	}

	// fi_ops_ep
	defmethod(cancel, FID(obj))
	defmethod(getopt, FID(obj))
	defmethod(setopt, FID(obj))

	// fi_ops_cm
	defmethod(setname, FID(obj))
	defmethod(getname, FID(obj))
	defmethod(listen, obj)
	defmethod(reject, obj)
};

class ep : public ops<struct fid_ep> {
protected:
	// ctor only for sep
	ep() {}
public:
	ep(struct fid_domain *domain, struct fi_info *info, void *context) {
		int ret = fi_endpoint(domain, info, &obj, context);
		init(ret);
	}

	// fi_ops_ep
	defmethod(cancel, FID(obj))
	defmethod(getopt, FID(obj))
	defmethod(setopt, FID(obj))

	// fi_ops_cm
	defmethod(setname, FID(obj))
	defmethod(getname, FID(obj))
	defmethod(getpeer, obj)
	defmethod(connect, obj)
	defmethod(accept, obj)
	defmethod(shutdown, obj)
	
	mc join(const void *addr, uint64_t flags, void *context) {
		return fi::mc(obj, addr, flags, context);
	}

	// fi_ops_msg
	defmethod(recv, obj)
	defmethod(recvv, obj)
	defmethod(recvmsg, obj)
	defmethod(send, obj)
	defmethod(sendv, obj)
	defmethod(sendmsg, obj)
	defmethod(inject, obj)
	defmethod(senddata, obj)
	defmethod(injectdata, obj)

	// fi_ops_rma
	defmethod(read, obj)
	defmethod(readv, obj)
	defmethod(readmsg, obj)
	defmethod(write, obj)
	defmethod(writev, obj)
	defmethod(writemsg, obj)
	defmethod(inject_write, obj)
	defmethod(writedata, obj)
	defmethod(injectdata_writedata, obj)
	
	// fi_ops_tagged
	defmethod(trecv, obj)
	defmethod(trecvv, obj)
	defmethod(trecvmsg, obj)
	defmethod(tsend, obj)
	defmethod(tsendv, obj)
	defmethod(tsendmsg, obj)
	defmethod(tinject, obj)
	defmethod(tsenddata, obj)
	defmethod(tinjectdata, obj)
	
};

class sep : public ep {
public:
	sep(struct fid_domain *domain, struct fi_info *info, void *context) {
		int ret = fi_scalable_ep(domain, info, &obj, context);
		init(ret);
	}

	defmethod(tx_context, obj)
	defmethod(rx_context, obj)
};

} // fi::
