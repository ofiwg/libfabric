/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

#include <libcxi.h>

#include <cassini_user_defs.h>

/* Tests ability to byte write CSRs.
 * Does not work on Netsim
 */
int main(void)
{
	int ret;
	struct cxil_dev *dev;
	union c_pct_cfg_spt_misc_info misc_info = {};
	unsigned int misc_info_offset;

	ret = cxil_open_device(0, &dev);
	if (ret != 0)
		errx(ret, "open failed\n");

	if (dev->info.cassini_version & CASSINI_1)
		misc_info_offset = C1_PCT_CFG_SPT_MISC_INFO(0);
	else
		misc_info_offset = C2_PCT_CFG_SPT_MISC_INFO(0);

	ret = cxil_map_csr(dev);
	if (ret != 0)
		errx(ret, "map failed\n");

	ret = cxil_read_csr(dev, misc_info_offset, &misc_info,
			    sizeof(misc_info));
	if (ret != 0)
		errx(ret, "can't read csr\n");

	misc_info.to_flag = 1;
	misc_info.req_try = 1;

	/* Offset 2 corresponds with req_try field*/
	ret = cxil_write8_csr(dev, misc_info_offset, 2, &misc_info,
			      sizeof(misc_info));
	if (ret != 0)
		errx(ret, "can't write csr\n");

	/* Reread */
	ret = cxil_read_csr(dev, misc_info_offset, &misc_info,
			    sizeof(misc_info));
	if (ret != 0)
		errx(ret, "can't read csr\n");

	if (misc_info.req_try != 1)
		errx(EINVAL, "byte write failed\n");

	/* Ensure other fields weren't written to */
	if (misc_info.to_flag != 0)
		errx(EINVAL, "wrote more than single byte\n");

	/* Reset */
	misc_info.req_try = 0;
	ret = cxil_write8_csr(dev, misc_info_offset, 2, &misc_info,
			      sizeof(misc_info));
	if (ret != 0)
		errx(ret, "can't write csr\n");

	/* Reread, ensure reset worked */
	ret = cxil_read_csr(dev, misc_info_offset, &misc_info,
			    sizeof(misc_info));
	if (ret != 0)
		errx(ret, "can't read csr\n");

	if (misc_info.req_try != 0)
		errx(EINVAL, "byte write failed\n");

	printf("Success\n");

	cxil_close_device(dev);

	return EXIT_SUCCESS;
}
