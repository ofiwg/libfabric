/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <shared.h>
#include <infiniband/verbs.h>

#ifndef _EFA_EXHAUST_MR_REG_COMMON_H
#define _EFA_EXHAUST_MR_REG_COMMON_H

/* The EFA NIC currently supports a maximum of 256 * 1024 = 262144 registrations */
#define EFA_MR_REG_LIMIT 262144
#define EFA_MR_REG_BUF_SIZE 128

int ft_efa_setup_ibv_pd(struct ibv_pd **pd);
void ft_efa_destroy_ibv_pd(struct ibv_pd *pd);
int ft_efa_register_mr_reg(struct ibv_pd *pd, void **buffers, size_t buf_size,
			   struct ibv_mr **mr_reg_vec, size_t count, size_t *registered);
int ft_efa_deregister_mr_reg(struct ibv_mr **mr_reg_vec, size_t count);
int ft_efa_alloc_bufs(void **buffers, size_t buf_size, size_t count);

#endif /* _EFA_EXHAUST_MR_REG_COMMON_H */
