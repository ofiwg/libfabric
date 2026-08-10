/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef FI_XPU_H
#define FI_XPU_H

#include <rdma/fabric.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * XPU Memory Operations
 */
struct fi_xpu_ops {
	size_t	size;
	int	(*alloc)(uint64_t device, uint64_t size,
			 uint64_t alignment, uint64_t flags,
			 void **addr, int *fd, uint64_t *offset);
	int	(*import)(uint64_t device, void *host_addr,
			  uint64_t size, uint64_t flags,
			  void **dev_addr);
	void	(*free)(uint64_t device, void *addr);
};

/*
 * XPU Attribute — input to fi_xpu_ctx() creation
 */
struct fi_xpu_attr {
	enum fi_hmem_iface	iface;
	uint64_t		device;
	struct fi_xpu_ops	*ops;
};

#ifdef __cplusplus
}
#endif

#endif /* FI_XPU_H */
