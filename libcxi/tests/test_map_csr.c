/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

#include <libcxi.h>

#include <cassini_user_defs.h>

int main(void)
{
	int ret;
	struct cxil_dev *dev;
	union c_mb_sts_rev rev = {};

	ret = cxil_open_device(0, &dev);
	if (ret != 0)
		errx(ret, "open failed\n");

	ret = cxil_map_csr(dev);
	if (ret != 0)
		errx(ret, "map failed\n");

	ret = cxil_read_csr(dev, C_MB_STS_REV, &rev,
			    sizeof(union c_mb_sts_rev));
	if (ret != 0)
		errx(ret, "can't read CSR\n");

	if (rev.vendor_id != 0x17db || rev.device_id != 0x501)
		errx(EINVAL, "bad vendor / revision\n");

	printf("Success\n");

	cxil_close_device(dev);

	return EXIT_SUCCESS;
}
