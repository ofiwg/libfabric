/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2022 Hewlett Packard Enterprise Development LP
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <err.h>
#include <getopt.h>
#include <limits.h>

#include "libcxi.h"

static void usage(void)
{
	fprintf(stderr,
		"cxi_dump_csrs - Dump Cassini registers\n\n"
		"Usage: cxi_dump_csrs [option]\n"
		" -d, --device=DEV       CXI device. Default is cxi0\n");
}

int main(int argc, char *argv[])
{
	char path[1024];
	char *buf;
	int fd;
	int ret;
	struct cxil_dev *dev;
	unsigned int dev_id = 0;
	bool is_c1;
	bool is_c2;
	struct option long_options[] = {
		{ "device", required_argument, 0, 'd' },
		{ NULL, 0, 0, 0 }
	};
	long tmp;
	char *endptr;

	while (1) {
		int option_index = 0;
		int c = getopt_long(argc, argv, "hvd:s:y:V", long_options,
				    &option_index);

		if (c == -1)
			break;

		switch (c) {
		case 'h':
			usage();
			return 0;

		case 'd':
			if (strlen(optarg) < 4 || strncmp(optarg, "cxi", 3))
				errx(1, "Invalid device name: %s", optarg);
			optarg += 3;

			errno = 0;
			endptr = NULL;
			tmp = strtol(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0 ||
			    tmp < 0 || tmp > INT_MAX)
				errx(1, "Invalid device name: cxi%s", optarg);
			dev_id = tmp;
			break;

		default:
			usage();
			return 1;
		}
	}

	if (optind < argc) {
		usage();
		return 1;
	}

	ret = cxil_open_device(dev_id, &dev);
	if (ret != 0)
		errx(1, "Failed to open device");

	ret = cxil_map_csr(dev);
	if (ret != 0) {
		if (ret == -EPERM)
			fprintf(stderr,
				"root permission is needed to run this tool\n");
		errx(1, "map csrs failed: %d\n", ret);
	}

	is_c1 = dev->info.cassini_version & CASSINI_1;
	is_c2 = dev->info.cassini_version & CASSINI_2;

	sprintf(path, "%s-csrs-%lu.bin", dev->info.device_name, time(NULL));
	fd = open(path, O_CREAT | O_RDWR, 0600);
	if (fd == -1) {
		err(1, "Cannot create the output file");
		return 1;
	}

	ret = ftruncate(fd, C_MEMORG_CSR_SIZE);
	if (ret == -1) {
		err(1, "Cannot resize the output file");
		return 1;
	}

	buf = mmap(NULL, C_MEMORG_CSR_SIZE, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, 0);
	if (buf == MAP_FAILED) {
		err(1, "Cannot create the output memory mapping");
		return 1;
	}

#include "cxi_dump_csrs.h"

	ret = msync(buf, C_MEMORG_CSR_SIZE, MS_SYNC);
	if (ret) {
		err(1, "Cannot sync memory mapping");
		unlink(path);
		return 1;
	}

	munmap(buf, C_MEMORG_CSR_SIZE);
	close(fd);

	printf("Cassini CSRs dumped in %s\n", path);

	return 0;
}
