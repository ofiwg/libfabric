/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_exhaust_mr_reg_common.h"

int ft_efa_setup_ibv_pd(struct ibv_pd **pd)
{
	int num_dev = 0;
	struct ibv_device **dev_list;
	struct ibv_context *ctx;

	dev_list = ibv_get_device_list(&num_dev);
	if (num_dev < 1) {
		FT_ERR("No ibv devices found");
		ibv_free_device_list(dev_list);
		return EXIT_FAILURE;
	} else if (num_dev > 1) {
		FT_WARN("More than 1 ibv devices found! This test will only "
			"exhaust MRs on the first device");
	}
	ctx = ibv_open_device(dev_list[0]);
	ibv_free_device_list(dev_list);
	*pd = ibv_alloc_pd(ctx);
	if (!*pd) {
		FT_ERR("alloc_pd failed with error %d\n", errno);
		return EXIT_FAILURE;
	}
	return FI_SUCCESS;
}

void ft_efa_destroy_ibv_pd(struct ibv_pd *pd)
{
	int err;

	err = ibv_dealloc_pd(pd);
	if (err) {
		FT_ERR("deallloc_pd failed with error %d\n", errno);
	}
}

int ft_efa_register_mr_reg(struct ibv_pd *pd, void **buffers, size_t buf_size,
			   struct ibv_mr **mr_reg_vec, size_t count, size_t *registered)
{
	int i, err = 0;

	for (i = 0; i < count; i++) {
		mr_reg_vec[i] = ibv_reg_mr(pd, buffers[i], buf_size,
					   IBV_ACCESS_LOCAL_WRITE);
		if (!mr_reg_vec[i]) {
			FT_ERR("reg_mr index %d failed with errno %d\n", i,
			       errno);
			err = errno;
			goto out;
		}
		if (i % 50000 == 0) {
			printf("Registered %d MRs...\n", i+1);
		}
	}
out:
	*registered = i;
	printf("Registered %d MRs\n", i+1);
	return err;
}

int ft_efa_deregister_mr_reg(struct ibv_mr **mr_reg_vec, size_t count)
{
	int i, err = 0;

	for (i = 0; i < count; i++) {
		if (mr_reg_vec[i])
			err = ibv_dereg_mr(mr_reg_vec[i]);
		if (err) {
			FT_ERR("dereg_mr index %d failed with errno %d\n", i,
			       errno);
		}
		if (i % 50000 == 0) {
			printf("Deregistered %d MRs...\n", i+1);
		}
	}
	printf("Deregistered %ld MRs\n", count);
	return err;
}

int ft_efa_alloc_bufs(void **buffers, size_t buf_size, size_t count) {
	int i;
	for (i = 0; i < count; i++) {
		buffers[i] = malloc(buf_size);
		if (!buffers[i]) {
			FT_ERR("malloc failed!\n");
			return EXIT_FAILURE;
		}
	}
	return FI_SUCCESS;
}
