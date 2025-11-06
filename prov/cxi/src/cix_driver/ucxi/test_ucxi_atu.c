// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Cray Inc. All rights reserved */

/* User space ATU test */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>

#include "test_ucxi_common.h"

#define MB (1024UL * 1024)
#define MIN_LEN (1024UL * 4)
#define DEF_LEN (1 * MB)
#define MAX_LEN (64 * MB)

#define DPRINT(...)			\
do {					\
	if (debug)			\
		printf(__VA_ARGS__);	\
} while (0)


static int svc_id = CXI_DEFAULT_SVC_ID;
static int vni_in = 10;
char dev_name[10];
static int nbuffers = 800;
static size_t default_len = DEF_LEN;
static int debug;
static int test_uid;
static int test_gid;

struct test_data {
	void *addr;
	size_t len;
	unsigned int md_hndl;
	struct cxi_md md;
};

static void help(void)
{
	printf("test_ucxi_atu [-dfh -u uid -g gid -l length -n num -rR -s rgroup id -d device -v vni]\n");
	printf("\tD - turn on debug (default is off)\n");
	printf("\tf - fault pages (default is pin)\n");
	printf("\tl - length of buffers (default %ld)\n", default_len);
	printf("\ts - resource group or service id\n");
	printf("\td - device - cxi0-3\n");
	printf("\tv - vni to use\n");
	printf("\tu - uid to use\n");
	printf("\tg - gid to use\n");
	printf("\tn - number of buffers (default %d)\n", nbuffers);
	printf("\tr - random length buffers from %ld to %ld (default off)\n",
	       MIN_LEN, MAX_LEN);
	printf("\tR - randomize pinning (default off)\n");
	printf("\th - this message\n");
}

static int test_init(struct cass_dev **dev, int *lni, int rg_id, const char *devn)
{
	*dev = open_device(devn);
	if (*dev == NULL) {
		fprintf(stderr, "cannot open %s\n", devn);
		return 1;
	}

	/* Get an LNI */
	*lni = alloc_lni(*dev, rg_id);
	if (*lni < 0) {
		fprintf(stderr, "cannot get an LNI\n");
		return 1;
	}

	return 0;
}

static int get_len(int randomize)
{
	if (randomize)
		return (rand() % (MAX_LEN + 1 - MIN_LEN)) + MIN_LEN;

	return default_len;
}

int main(int argc, char **argv)
{
	int i;
	int rc;
	int opt;
	int lni;
	int domain;
	int retry;
	struct ucxi_cp *cp;
	struct ucxi_cp *cp2;
	struct ucxi_cq *transmit_cq;
	struct ucxi_cq *target_cq;
	struct cass_dev *dev;
	uint32_t flags;
	unsigned int md_hndl;
	struct cxi_md md;
	struct test_data *my_data;
	int niovas;
	int random_len = 0;
	long total_allocd = 0;

	strcpy(dev_name, "cxi0");

	while ((opt = getopt(argc, argv, "Dhv:s:u:g:d:l:n:r")) != -1) {
		switch (opt) {
		case 'D':
			debug++;
			break;
		case 'l':
			default_len = strtol(optarg, NULL, 0);
			break;
		case 'n':
			nbuffers = atoi(optarg);
			break;
		case 'r':
			random_len = 1;
			break;
		case 's':
			svc_id = atoi(optarg);
			printf("rgroup/service id %d\n", svc_id);
			break;
		case 'u':
			test_uid = atoi(optarg);
			printf("User ID %d\n", test_uid);
			rc = seteuid(test_uid);
			if (rc) {
				fprintf(stderr, "cannot seteuid rc:%d\n", rc);
				return 1;
			}
			break;
		case 'g':
			test_gid = atoi(optarg);
			printf("Group ID %d\n", test_gid);
			rc = setegid(test_gid);
			if (rc) {
				fprintf(stderr, "cannot setegid rc:%d\n", rc);
				return 1;
			}
			break;
		case 'v':
			vni_in = atoi(optarg);
			printf("VNI %d\n", vni_in);
			break;
		case 'd':
			strncpy(dev_name, optarg, ARRAY_SIZE(dev_name) - 1);
			dev_name[ARRAY_SIZE(dev_name) - 1] = '\0';
			break;
		case 'h':
			help();
			exit(0);
		default:
			break;
		}
	}

	if (random_len)
		printf("Randomize length\n");
	else
		printf("buffer length %ld\n", default_len);

	rc = test_init(&dev, &lni, svc_id, dev_name);
	if (rc)
		return rc;

	my_data = calloc(nbuffers, sizeof(*my_data));
	if (!my_data)
		return 1;

	flags = CXI_MAP_WRITE | CXI_MAP_READ | CXI_MAP_PIN;
	srand(1);

	for (i = 0; i < nbuffers; i++) {
		uint64_t *va;
		size_t len;

		len = get_len(random_len);

		va = aligned_alloc(sysconf(_SC_PAGESIZE), len);
		if (!va) {
			printf("Allocation failed for len %ld\n", len);
			break;
		}

		my_data[i].addr = va;
		my_data[i].len = len;
		total_allocd += len;
		DPRINT("addr:%p len:0x%08lx (%2ld.%03ld MB) tot:%ld MB\n", va,
		       len, len / MB, len / MB, total_allocd / MB);
	}

	printf("Allocated %d buffers of %d (%ld MB)\n",
		i, nbuffers, total_allocd / MB);

	niovas = i;

	for (i = 0; i < niovas; i++) {

		rc = atu_map(dev, lni, my_data[i].addr, my_data[i].len,
			     flags, &md_hndl, &md);
		if (rc < 0)
			break;

		DPRINT("map va:%p len:0x%08lx flags:0x%x iova:0x%016llx ioval:0x%08lx ac:%d\n",
		       my_data[i].addr, my_data[i].len, flags, md.iova, md.len, md.lac);
		my_data[i].md_hndl = md_hndl;
		my_data[i].md = md;

		if ((uint64_t)my_data[i].addr != md.va) {
			printf("va was adjusted va:%lx iova:%llx\n",
				(uint64_t)my_data[i].addr, md.va);
			break;
		}
	}

	printf("Mapped %d buffers of %d\n", i, niovas);

	niovas = i;

	for (i = 0; i < niovas; i++) {
		if (my_data[i].md.va) {
			rc = atu_unmap(dev, my_data[i].md_hndl);
			DPRINT("unmap va:%08llx len:0x%lx\n",
			       my_data[i].md.va, my_data[i].md.len);
		}

		free((void *)my_data[i].addr);

		if (rc)
			break;
	}

	/* Get a domain */
	domain = alloc_domain(dev, lni, vni_in, 40, 1024);
	if (domain < 0) {
		fprintf(stderr, "cannot get a domain rc:%d\n", domain);
		return 1;
	}
	DPRINT("Domain allocated: %d\n", domain);

	cp = alloc_cp(dev, lni, vni_in, CXI_TC_BEST_EFFORT);
	if (!cp) {
		fprintf(stderr, "cannot get a CP\n");
		return 1;
	}

	cp2 = alloc_cp(dev, lni, vni_in, CXI_TC_BEST_EFFORT);
	if (!cp2) {
		fprintf(stderr, "cannot get a CP2\n");
		return 1;
	}

	if (cp2->lcid != cp->lcid) {
		fprintf(stderr, "CP didn't get re-used\n");
		return 1;
	}

	/* Create a transmit and target CQ */
	transmit_cq = create_cq(dev, lni, true, cp->lcid);
	if (transmit_cq == NULL) {
		fprintf(stderr, "Transmit CQ cannot be created\n");
		return 1;
	}
	DPRINT("Transmit CQ allocated: %d\n", transmit_cq->cq);

	target_cq = create_cq(dev, lni, false, 0);
	if (target_cq == NULL) {
		fprintf(stderr, "Transmit CQ cannot be created\n");
		return 1;
	}
	DPRINT("Target CQ allocated: %d\n", target_cq->cq);

	/* Set an LCID */
	rc = cxi_cq_emit_cq_lcid(&transmit_cq->cmdq, 0);
	if (rc) {
		fprintf(stderr, "Command emit failure\n");
		return 1;
	}

	printf("Command emit success\n");
	cxi_cq_ring(&transmit_cq->cmdq);

	/* Wait for the adapter to consume the command */
	retry = 200;
	while (retry-- &&
	       transmit_cq->cmdq.status->rd_ptr != C_CQ_FIRST_WR_PTR + 2) {
		usleep(10000);
		__sync_synchronize();
	}

	DPRINT("rd_ptr %d\n", transmit_cq->cmdq.status->rd_ptr);

	if (transmit_cq->cmdq.status->rd_ptr != C_CQ_FIRST_WR_PTR + 2) {
		fprintf(stderr, "Command ptr hasn't moved\n");
		return 1;
	}

	free(my_data);

	/* Freeing resources */
	destroy_cq(dev, transmit_cq);
	destroy_cq(dev, target_cq);
	destroy_cp(dev, cp2);
	destroy_cp(dev, cp);
	destroy_domain(dev, domain);
	destroy_lni(dev, lni);

	close_device(dev);

	return 0;
}
