/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libcxi.h>

int main(void)
{
	int ret;
	int i;

	struct cxil_device_list *dev_list;

	ret = cxil_get_device_list(&dev_list);
	if (ret) {
		fprintf(stderr, "Cannot get the list of CXI devices\n");
		return EXIT_FAILURE;
	}

	printf("Number of CXI devices found: %d\n", dev_list->count);
	for (i = 0; i < dev_list->count; i++) {
		printf("cxi%u:\n", dev_list->info[i].dev_id);
		printf("  NID: %u\n", dev_list->info[i].nid);
		printf("  pid_granule: %d\n", dev_list->info[i].pid_granule);
	}

	cxil_free_device_list(dev_list);

	return EXIT_SUCCESS;
}
