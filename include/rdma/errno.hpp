#pragma once

#include <stdexcept>

#include <rdma/fi_errno.h>

namespace fi {

// TODO: perhaps over-ride system_error and defer calling fi_strerror()
// TODO: until what() is called? More efficient, but more code.
// TODO: Or just throw a plain old int and let the callee deal with it.
class fi_error : public std::runtime_error {
public:
	fi_error(int num) : std::runtime_error(fi_strerror(num)) {}
};

} // fi::
