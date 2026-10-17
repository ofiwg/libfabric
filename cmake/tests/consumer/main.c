/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_atomic.h>

#if defined(HAVE_CONFIG_H) || defined(ENABLE_DEBUG)
#error Private build definitions leaked into the public interface
#endif

int main(void)
{
	struct fi_info *info = fi_allocinfo();
	struct fi_info *copy;

	if (!info)
		return 1;
	copy = fi_dupinfo(info);
	fi_freeinfo(info);
	if (!copy)
		return 2;
	fi_freeinfo(copy);
	return fi_version() != FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION);
}
