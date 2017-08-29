#pragma once

#include <cstdio>
#include <memory>

#include <rdma/errno.hpp>

#include <rdma/fabric.h>

#define FID(obj) (&(obj->fid))

// helper macro for defining methods
// method is fi_$name sans the fi_ prefix, ie fi_write -> write
#define defmethod(method, obj_ptr)	\
	template <typename ...Args>	\
	auto method(Args && ... args) -> \
		decltype(fi_##method(obj_ptr, std::forward<Args>(args)...)) {	\
		return fi_##method(obj_ptr, std::forward<Args>(args)...); \
	}

namespace fi {

template<typename P>
class ops {
protected:
	P *obj;

	void init(int ret) {
		if (ret)
			throw fi_error(ret);
	}

public:
	~ops() {
		printf("Closing: %lu\n", obj->fid.fclass);
		if (fi_close(&(obj->fid)))
			printf("Failed closing\n");
	}

	//ops() {}
	//ops(const ops&) = delete;

	// generic bind for various fi_xxx_bind() calls:
	int bind(struct fid *bfid, uint64_t flags) {
		return obj->fid.ops->bind(FID(obj), bfid, flags);
	}

	defmethod(control, FID(obj))

	// TODO: set errno to ret ? or throw an exception?
	// should name be open_ops or ops_open?
	void* ops_open(const char *name, uint64_t flags, void *context) {
		void *ops;
		int ret = fi_open_ops(FID(obj), name, flags, &ops, context);
		if (ret) { ops = NULL; }
		return ops;
	}
};

}
