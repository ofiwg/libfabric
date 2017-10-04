#pragma once

#include <rdma/fabric.h>

#include <rdma/ops.hpp>
#include <rdma/domain.hpp>
#include <rdma/endpoint.hpp>
#include <rdma/eq.hpp>

namespace fi {

class fabric : public ops<fid_fabric> {
public:
	fabric(struct fi_fabric_attr *attr, void *context) {
		int ret = fi_fabric(attr, &obj, context);
		init(ret);
	};

	// disambiguate class from function name
	fi::domain domain(struct fi_info *info, void *context) {
		return fi::domain(obj, info, context);
	}

	pep passive_ep(struct fi_info *info, void *context) {
		return fi::pep(obj, info, context);
	}

	eq eq_open(struct fi_eq_attr *attr, void *context) {
		return fi::eq(obj, attr, context);
	}

	wait wait_open(struct fi_wait_attr *attr) {
		return fi::wait(obj, attr);
	}

	// TODO: use optional here?
	wait trywait(int count);


};

} // fi::
