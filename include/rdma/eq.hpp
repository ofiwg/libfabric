#pragma once

#include <rdma/fi_eq.h>

#include <rdma/ops.hpp>

namespace fi {

class eq : public ops<struct fid_eq> {
public:
	eq(struct fid_fabric *fabric, struct fi_eq_attr *attr, void *context) {
		int ret = fi_eq_open(fabric, attr, &obj, context);
		init(ret);
	}

	// fi_ops_eq
	defmethod(eq_read, obj)
	defmethod(eq_readerr, obj)
	defmethod(eq_write, obj)
	defmethod(eq_sread, obj)
	defmethod(eq_strerror, obj)
};

class wait : public ops<struct fid_wait> {
public:
	wait(struct fid_fabric *fabric, struct fi_wait_attr *attr) {
		int ret = fi_wait_open(fabric, attr, &obj);
		init(ret);
	}

};

}

