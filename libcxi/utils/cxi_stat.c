/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2022-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI status utility */

#define _GNU_SOURCE
#include <getopt.h>
#include <dirent.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <glob.h>
#include <err.h>

#include <libcxi.h>

#define MAX_LEN 256
#define MAX_LENX2 (MAX_LEN * 2)
#define NUM_CNTR_READS 2
#define CNTR_MAX 72057594037927936 /* 2^56, counters are 56 bits wide */
#define CAS_HZ 1000000000 /* cassini system clock rate */
#define NUM_PAUSES 8
#define DEFAULT_PAUSE 4

static const char *name = "cxi_stat";
static const char *version = "1.5.5";

enum rc_names {
	RATE_CNTR_UC_CW,
	RATE_CNTR_C_CW,
	RATE_CNTR_G_CW,
	RATE_CNTR_RX_P_STD,
	RATE_CNTR_RX_P_0,
	RATE_CNTR_RX_P_1,
	RATE_CNTR_RX_P_2,
	RATE_CNTR_RX_P_3,
	RATE_CNTR_RX_P_4,
	RATE_CNTR_RX_P_5,
	RATE_CNTR_RX_P_6,
	RATE_CNTR_RX_P_7,
	RATE_CNTR_TX_P_0,
	RATE_CNTR_TX_P_1,
	RATE_CNTR_TX_P_2,
	RATE_CNTR_TX_P_3,
	RATE_CNTR_TX_P_4,
	RATE_CNTR_TX_P_5,
	RATE_CNTR_TX_P_6,
	RATE_CNTR_TX_P_7,
	RATE_CNTR_NUM
};

struct cxi_dev {
	struct cxil_devinfo info;

	char fw_version[MAX_LEN];
	char part_number[MAX_LEN];
	char serial_number[MAX_LEN];
	char pcie_speed[MAX_LEN];
	char pcie_width[MAX_LEN];
	char pcie_slot[MAX_LEN];
	char link_layer_retry[MAX_LEN];
	char link_loopback[MAX_LEN];
	char link_media[MAX_LEN];
	char link_speed[MAX_LEN];
	char link_state[MAX_LEN];
	char network_dev[MAX_LEN];
	char mac_addr[MAX_LEN];
	char rx_pause_state[MAX_LEN];
	char tx_pause_state[MAX_LEN];
	char *aer_cor;
	char *aer_fatal;
	char *aer_nonfatal;
	float corrected_cw;
	float uncorrected_cw;
	float corrected_ber;
	float uncorrected_ber;
	float good_cw;
	float rx_pause_pct[NUM_PAUSES];
	float tx_pause_pct[NUM_PAUSES];
};

struct stat_opts {
	int dev_id;
	unsigned int pause;
	bool report_aer;
	bool report_rates;
	bool list_devs_only;
	bool list_macs_only;
};

struct rate_cntr {
	uint64_t addr;
	uint64_t count[NUM_CNTR_READS];
	struct timespec ts[NUM_CNTR_READS];
};

void usage(void)
{
	fprintf(stderr,
		"cxi_stat - CXI device status utility\n"
		"Usage: -hlm\n"
		" -h --help             Show this help\n"
		" -l --list             List all CXI devices\n"
		" -m --mac-list         List all CXI MAC addresses\n"
		" -d --device=DEV       List only specified CXI device\n"
		"                       Default lists all CXI devices\n"
		" -a --aer              Report AER statistics\n"
		" -r --rates            Report codeword rates and pause percentages\n"
		" -p --pause            Pause time used for rate calculations\n"
		"                       Units = seconds, Default = %d\n"
		" -V --version          Print the version and exit\n", DEFAULT_PAUSE);
}

int copy_data_from_file(char *filename, char *param)
{
	int rc = 0;
	FILE *fp;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Error opening %s\n", filename);
		rc = -ENOENT;
		return rc;
	}
	if (fgets(param, MAX_LEN, fp) == NULL) {
		fprintf(stderr, "Error reading from %s\n", filename);
		rc = -ENODATA;
	}
	param[strcspn(param, "\n")] = 0;
	fclose(fp);
	return rc;
}

/* Searches for search_str in buffer. If it is found, strip off newline and
 * return pointer to first buffer character after search string.
 */
const char *field_parse(char *buffer, char *search_str, char *end_str)
{
	char *tmp = NULL;
	char *idx;

	if (strncmp(buffer, search_str, strlen(search_str)) == 0) {
		buffer[strcspn(buffer, "\n")] = 0;
		if (end_str) {
			idx = strstr(buffer, end_str);
			if (idx)
				*idx = 0;
		}
		tmp = buffer + strlen(search_str);
	}
	return tmp;
}

static void get_port_status(struct cxi_dev *cd)
{
	char buf[MAX_LEN];
	char *base_path = NULL;
	char *tmp_path = NULL;
	int rc;
	bool is_c1;

	is_c1 = (cd->info.cassini_version & CASSINI_1) ? true : false;

	if (is_c1) {
		rc = asprintf(&base_path, "/sys/class/cxi/%s/device/port",
				cd->info.device_name);
	} else {
		rc = asprintf(&base_path, "/sys/class/cxi/%s/device/port/0",
				cd->info.device_name);
	}
	if (rc < 0) {
		fprintf(stderr, "Error executing %s", __func__);
		return;
	}

	if (is_c1)
		rc = asprintf(&tmp_path, "%s/media", base_path);
	else
		rc = asprintf(&tmp_path, "%s/media/type", base_path);
	if (rc > 0) {
		copy_data_from_file(tmp_path, cd->link_media);
		free(tmp_path);
	}

	if (is_c1)
		rc = asprintf(&tmp_path, "%s/speed", base_path);
	else
		rc = asprintf(&tmp_path, "%s/link/speed", base_path);
	if (rc > 0) {
		copy_data_from_file(tmp_path, cd->link_speed);
		free(tmp_path);
	}

	if (is_c1)
		rc = asprintf(&tmp_path, "%s/link", base_path);
	else
		rc = asprintf(&tmp_path, "%s/link/state", base_path);
	if (rc > 0) {
		copy_data_from_file(tmp_path, cd->link_state);
		free(tmp_path);
	}

	if (is_c1)
		rc = asprintf(&tmp_path, "%s/link_layer_retry", base_path);
	else
		rc = asprintf(&tmp_path, "%s/llr/state", base_path);
	if (rc > 0) {
		copy_data_from_file(tmp_path, cd->link_layer_retry);
		free(tmp_path);
	}

	if (is_c1)
		rc = asprintf(&tmp_path, "%s/loopback", base_path);
        else
		rc = asprintf(&tmp_path, "%s/link/config/loopback", base_path);
	if (rc > 0) {
		copy_data_from_file(tmp_path, cd->link_loopback);
		free(tmp_path);
	}

	if (is_c1)
		rc = asprintf(&tmp_path, "%s/pause", base_path);
	else
		rc = asprintf(&tmp_path, "%s/link/config/pause_map", base_path);
	if (rc > 0) {
		if (copy_data_from_file(tmp_path, buf) == 0) {
			snprintf(cd->tx_pause_state, MAX_LEN, "%s", buf);
			snprintf(cd->rx_pause_state, MAX_LEN, "%s", buf);
		}
		free(tmp_path);
	}

	free(base_path);
}

float cntr_rate(struct rate_cntr r)
{
	uint64_t c_delta;
	float t_delta;
	float rate;

	if (r.count[1] >= r.count[0])
		c_delta = r.count[1] - r.count[0];
	else
		c_delta = CNTR_MAX + r.count[1] - r.count[0];

	t_delta = (r.ts[1].tv_sec + 1e-9 * r.ts[1].tv_nsec) - (r.ts[0].tv_sec + 1e-9 * r.ts[0].tv_nsec);

	if (t_delta > 0)
		rate = c_delta / t_delta;
	else
		rate = 999999999;

	return rate;
}

float rate_to_ber(float rate, char *speed)
{
	float ber = 0;

	if (strstr(speed, "CK_400G") != NULL ||
		strstr(speed, "ck400G") != NULL)
		ber = rate / 425000000000.0;
	else if (strstr(speed, "BS_200G") != NULL ||
			 strstr(speed, "bs200G") != NULL)
		ber = rate / 212500000000.0;
	else if (strstr(speed, "BJ_100G") != NULL ||
			 strstr(speed, "bj100G") != NULL)
		ber = rate / 103125000000.0;
	else if (strstr(speed, "CD_100G") != NULL ||
			 strstr(speed, "cd100G") != NULL)
		ber = rate / 106250000000.0;
	else if (strstr(speed, "CD_50G") != NULL ||
			 strstr(speed, "cd50G") != NULL)
		ber = rate / 53125000000.0;
	else {
		if (rate > 0) {
			fprintf(stderr, "Warning: encountered unexpected Link Mode %s for non-zero error rate %.3e\n",
				   speed, rate);
			ber = rate / 212500000000.0;
		}
	}
	return ber;
}

int copy_file(char *f_aer, char **param)
{
	int rc = 0;
	FILE *fp;
	size_t fsize;

	fp = fopen(f_aer, "r");
	if (fp == NULL) {
		fprintf(stderr, "Error opening %s\n", f_aer);
		rc = -ENOENT;
	} else {
		fseek(fp, 0, SEEK_END);
		fsize = ftell(fp);
		rewind(fp);
		*param = malloc(fsize);
		rc = fread(*param, sizeof(**param), fsize, fp);
		if (rc <= 0)
			fprintf(stderr, "Error reading %s\n", f_aer);
		else
			rc = 0;

		fclose(fp);
	}
	return rc;
}

int get_aer(struct cxi_dev *cd, int num_devs)
{
	int i;
	int rc = 0;
	int rc2 = 0;
	char *f_aer = NULL;

	for (i = 0; i < num_devs; i++) {
		rc = asprintf(&f_aer,
				"/sys/class/cxi/%s/device/aer_dev_correctable",
				cd[i].info.device_name);
		if (rc < 0) {
			fprintf(stderr, "Error executing %s", __func__);
			rc2 = 1;
		} else {
			rc = copy_file(f_aer, &cd[i].aer_cor);
			free(f_aer);
			if (rc)
				rc2 = 1;
		}

		rc = asprintf(&f_aer,
				"/sys/class/cxi/%s/device/aer_dev_fatal",
				cd[i].info.device_name);
		if (rc < 0) {
			fprintf(stderr, "Error executing %s", __func__);
			rc2 = 1;
		} else {
			rc = copy_file(f_aer, &cd[i].aer_fatal);
			free(f_aer);
			if (rc)
				rc2 = 1;
		}

		rc = asprintf(&f_aer,
				"/sys/class/cxi/%s/device/aer_dev_nonfatal",
				cd[i].info.device_name);
		if (rc < 0) {
			fprintf(stderr, "Error executing %s", __func__);
			rc2 = 1;
		} else {
			rc = copy_file(f_aer, &cd[i].aer_nonfatal);
			free(f_aer);
			if (rc)
				rc2 = 1;
		}
	}

	return rc2;
}

void get_port_rates(int dev_id, struct cxi_dev *cd, unsigned int pause)
{
	int i;
	int j;
	int rc;
	struct cxil_dev *csr_dev;
	struct rate_cntr rcnts[RATE_CNTR_NUM];

	rc = cxil_open_device(dev_id, &csr_dev);
	if (rc)
		fprintf(stderr, "Failed to open CXI device: %s\n", strerror(-rc));

	rcnts[RATE_CNTR_UC_CW].addr = C1_CNTR_HNI_PCS_UNCORRECTED_CW;
	rcnts[RATE_CNTR_C_CW].addr = C1_CNTR_HNI_PCS_CORRECTED_CW;
	rcnts[RATE_CNTR_G_CW].addr = C1_CNTR_HNI_PCS_GOOD_CW;
	rcnts[RATE_CNTR_RX_P_STD].addr = C_CNTR_HNI_RX_PAUSED_STD;
	rcnts[RATE_CNTR_RX_P_0].addr = C_CNTR_HNI_RX_PAUSED_0;
	rcnts[RATE_CNTR_RX_P_1].addr = C_CNTR_HNI_RX_PAUSED_1;
	rcnts[RATE_CNTR_RX_P_2].addr = C_CNTR_HNI_RX_PAUSED_2;
	rcnts[RATE_CNTR_RX_P_3].addr = C_CNTR_HNI_RX_PAUSED_3;
	rcnts[RATE_CNTR_RX_P_4].addr = C_CNTR_HNI_RX_PAUSED_4;
	rcnts[RATE_CNTR_RX_P_5].addr = C_CNTR_HNI_RX_PAUSED_5;
	rcnts[RATE_CNTR_RX_P_6].addr = C_CNTR_HNI_RX_PAUSED_6;
	rcnts[RATE_CNTR_RX_P_7].addr = C_CNTR_HNI_RX_PAUSED_7;
	rcnts[RATE_CNTR_TX_P_0].addr = C_CNTR_HNI_TX_PAUSED_0;
	rcnts[RATE_CNTR_TX_P_1].addr = C_CNTR_HNI_TX_PAUSED_1;
	rcnts[RATE_CNTR_TX_P_2].addr = C_CNTR_HNI_TX_PAUSED_2;
	rcnts[RATE_CNTR_TX_P_3].addr = C_CNTR_HNI_TX_PAUSED_3;
	rcnts[RATE_CNTR_TX_P_4].addr = C_CNTR_HNI_TX_PAUSED_4;
	rcnts[RATE_CNTR_TX_P_5].addr = C_CNTR_HNI_TX_PAUSED_5;
	rcnts[RATE_CNTR_TX_P_6].addr = C_CNTR_HNI_TX_PAUSED_6;
	rcnts[RATE_CNTR_TX_P_7].addr = C_CNTR_HNI_TX_PAUSED_7;

	/* Read counters */
	for (i = 0; i < NUM_CNTR_READS; i++) {
		for (j = 0; j < RATE_CNTR_NUM; j++) {
			rc = cxil_read_cntr(csr_dev, rcnts[j].addr, &rcnts[j].count[i],
								&rcnts[j].ts[i]);
			if (rc)
				fprintf(stderr, "Failed to read counter at %ld: %s\n",
						rcnts[j].addr, strerror(-rc));
		}
		/* Pause + 100ms buffer for counter update */
		/* Only pause the first time */
		if (i == 0)
			usleep(pause * 1000000 + 100000);
	}

	cxil_close_device(csr_dev);

	cd->uncorrected_cw = cntr_rate(rcnts[RATE_CNTR_UC_CW]);
	cd->corrected_cw = cntr_rate(rcnts[RATE_CNTR_C_CW]);
	cd->good_cw = cntr_rate(rcnts[RATE_CNTR_G_CW]);
	cd->uncorrected_ber = rate_to_ber(cd->uncorrected_cw, cd->link_speed);
	cd->corrected_ber = rate_to_ber(cd->corrected_cw, cd->link_speed);

	if (strncmp(cd->tx_pause_state, "pfc", 3) == 0) {
		for (j = 0; j < NUM_PAUSES; j++) {
			cd->tx_pause_pct[j] =
				cntr_rate(rcnts[RATE_CNTR_TX_P_0 + j]) * 100.0 / CAS_HZ;
			cd->rx_pause_pct[j] =
				cntr_rate(rcnts[RATE_CNTR_RX_P_0 + j]) * 100.0 / CAS_HZ;
		}
	} else if (strncmp(cd->tx_pause_state, "global", 6) == 0) {
		cd->tx_pause_pct[0] =
			cntr_rate(rcnts[RATE_CNTR_TX_P_0]) * 100.0 / CAS_HZ;
		cd->rx_pause_pct[0] =
			cntr_rate(rcnts[RATE_CNTR_RX_P_STD]) * 100.0 / CAS_HZ;
	}
}

void get_pcie_slot(struct cxi_dev *cd)
{
	char buf[MAX_LEN];
	char tp[MAX_LENX2];
	const char *tmp;
	FILE *fp;

	snprintf(tp, MAX_LENX2, "/sys/class/cxi/%s/device/uevent",
		 cd->info.device_name);
	fp = fopen(tp, "r");
	if (fp) {
		while (fgets(buf, MAX_LEN, fp)) {
			tmp = field_parse(buf, "PCI_SLOT_NAME=", NULL);
			if (tmp)
				snprintf(cd->pcie_slot, MAX_LEN, "%s", tmp);
		}
		fclose(fp);
	}
}

void get_network_device(struct cxi_dev *cd, int num_devs)
{
	glob_t globbuf;
	int rc;
	char if_path[MAX_LEN];
	char addr_path[MAX_LEN];
	char tp[MAX_LENX2];
	int i;
	int j;
	int count;
	char *addr;

	/* Search for directories with <dev_name>/device/cxi/cxi<num>
	 * We don't know what <dev_name> or <num> will be, so we need to enter
	 * the directories and look around. Tried nftw, but it got lost with
	 * the symbolic links.
	 */
	rc = glob("/sys/class/net/*", 0, NULL, &globbuf);
	if (rc)
		return;

	count = globbuf.gl_pathc;
	for (i = 0; i < count; i++) {
		rc = snprintf(if_path, MAX_LEN, "%s/device",
			      globbuf.gl_pathv[i]);
		if (rc < 0)
			goto free_glob;

		rc = readlink(if_path, addr_path, MAX_LEN - 1);
		if (rc < 0) {
			/* A virtual device, like a bridge, doesn't have a
			 * device link.
			 */
			if (errno == ENOENT || errno == ENOTDIR)
				continue;

			goto free_glob;
		}
		addr_path[rc] = '\0';

		/* addr is pcie slot e.g.0000:21:00.0 */
		addr = basename(addr_path);

		for (j = 0; j < num_devs; j++) {
			if (strcmp(addr, cd[j].pcie_slot) == 0) {
				snprintf(cd[j].network_dev, MAX_LEN, "%s",
					 basename(globbuf.gl_pathv[i]));
				snprintf(tp, MAX_LENX2,
					 "/sys/class/net/%s/address",
					 basename(globbuf.gl_pathv[i]));
				copy_data_from_file(tp, cd[j].mac_addr);
			}
		}
	}

free_glob:
	globfree(&globbuf);
}

struct cxi_dev *get_cxi_dev_status(int *n_devs, struct stat_opts opts)
{
	unsigned int i;
	int n = 0;
	int ret;
	char class_path[] = "/sys/class/cxi/";
	char tp[MAX_LENX2];
	struct cxi_dev *cdev = NULL;
	struct cxil_device_list *dev_list;

	ret = cxil_get_device_list(&dev_list);
	if (ret)
		fprintf(stderr, "Cannot get the list of CXI devices\n");

	cdev = calloc(dev_list->count, sizeof(struct cxi_dev));
	if (cdev == NULL) {
		fprintf(stderr, "Cannot allocate memory for cdev\n");
		exit(-ENOMEM);
	}

	for (i = 0; i < dev_list->count; i++) {
		if (opts.dev_id >= 0 &&
				(unsigned int)opts.dev_id != dev_list->info[i].dev_id)
			continue;

		cdev[n].info = dev_list->info[i];

		/* Get FRU info */
		snprintf(tp, MAX_LENX2, "%s%s/device/fru/part_number",
			 class_path, cdev[n].info.device_name);
		copy_data_from_file(tp, cdev[n].part_number);

		snprintf(tp, MAX_LENX2, "%s%s/device/fru/serial_number",
			 class_path, cdev[n].info.device_name);
		copy_data_from_file(tp, cdev[n].serial_number);

		/* Get PCIE status */
		snprintf(tp, MAX_LENX2, "%s%s/device/properties/current_esm_link_speed",
			 class_path, cdev[n].info.device_name);
		ret = copy_data_from_file(tp, cdev[n].pcie_speed);
		if (ret != 0 || strcmp(cdev[n].pcie_speed, "Absent") == 0 ||
			strcmp(cdev[n].pcie_speed, "Disabled") == 0) {
			snprintf(tp, MAX_LENX2, "%s%s/device/current_link_speed",
				 class_path, cdev[n].info.device_name);
			copy_data_from_file(tp, cdev[n].pcie_speed);
		}

		snprintf(tp, MAX_LENX2, "%s%s/device/current_link_width",
			 class_path, cdev[n].info.device_name);
		copy_data_from_file(tp, cdev[n].pcie_width);

		get_pcie_slot(&cdev[n]);

		/* Get uC status */
		snprintf(tp, MAX_LENX2, "%s%s/device/uc/qspi_blob_version",
			 class_path, cdev[n].info.device_name);
		copy_data_from_file(tp, cdev[n].fw_version);

		/* Get port status */
		get_port_status(&cdev[n]);

		/* Get port rates */
		if (opts.report_rates)
			get_port_rates(dev_list->info[i].dev_id, &cdev[n], opts.pause);

		n++;
	}
	*n_devs = n;
	cxil_free_device_list(dev_list);
	return cdev;
}

void print_aer(char *astr)
{
	int i = 0;

	if (astr == NULL) {
		printf("No data\n");
		return;
	}

	while (*(astr + i) != '\0') {
		if (*(astr + i) != '\n')
			printf("%c", *(astr + i));
		else {
			if (*(astr + i + 1) != '\0')
				printf("%c        ", *(astr + i));
			else
				printf("%c", *(astr + i));
		}
		i++;
	}
}

void print_cxi_status(struct cxi_dev *cdev, int num_devs, struct stat_opts opts)
{
	int i;
	int j;
	float min_ber = 0.0;

	for (i = 0; i < num_devs; i++) {
		if (opts.report_rates)
			min_ber = rate_to_ber(1.0/opts.pause, cdev[i].link_speed);

		printf("Device: %s\n", cdev[i].info.device_name);
		printf("    Description: %s\n", cdev[i].info.fru_description);
		printf("    Part Number: %s\n", cdev[i].part_number);
		printf("    Serial Number: %s\n", cdev[i].serial_number);
		printf("    FW Version: %s\n", cdev[i].fw_version);
		printf("    Network device: %s\n", cdev[i].network_dev);
		printf("    MAC: %s\n", cdev[i].mac_addr);
		printf("    NID: %u (0x%05x)\n",
		       cdev[i].info.nid, cdev[i].info.nid);
		printf("    PID granule: %u\n", cdev[i].info.pid_granule);
		printf("    PCIE speed/width: %s x%s\n", cdev[i].pcie_speed,
		       cdev[i].pcie_width);
		printf("    PCIE slot: %s\n", cdev[i].pcie_slot);
		printf("        Link layer retry: %s\n", cdev[i].link_layer_retry);
		printf("        Link loopback: %s\n", cdev[i].link_loopback);
		printf("        Link media: %s\n", cdev[i].link_media);
		printf("        Link MTU: %lu\n", cdev[i].info.link_mtu);
		printf("        Link speed: %s\n", cdev[i].link_speed);
		printf("        Link state: %s\n", cdev[i].link_state);
		if (opts.report_rates) {
			printf("    Rates:\n");
			printf("        Good CW: %.2f/s\n", cdev[i].good_cw);
			printf("        Corrected CW: %.2f/s\n",
			       cdev[i].corrected_cw);
			printf("        Uncorrected CW: %.2f/s\n",
			       cdev[i].uncorrected_cw);
			printf("        Corrected BER: %s%.3e\n",
			       cdev[i].corrected_ber ? "" : "<",
			       cdev[i].corrected_ber ? cdev[i].corrected_ber : min_ber);
			printf("        Uncorrected BER: %s%.3e\n",
			       cdev[i].uncorrected_ber ? "" : "<",
			       cdev[i].uncorrected_ber ? cdev[i].uncorrected_ber : min_ber);
			printf("        TX Pause state: %s\n",
			       cdev[i].tx_pause_state);
			printf("        RX Pause state: %s\n",
			       cdev[i].rx_pause_state);
			if (strncmp(cdev[i].tx_pause_state, "pfc", 3) == 0) {
				for (j = 0; j < NUM_PAUSES; j++) {
					printf("            RX Pause PCP %d: %3.1f%%\n",
					       j, cdev[i].rx_pause_pct[j]);
					printf("            TX Pause PCP %d: %3.1f%%\n",
					       j, cdev[i].tx_pause_pct[j]);
				}
			} else if (strncmp(cdev[i].tx_pause_state, "std", 3) ==
				   0) {
				printf("            RX Pause: %3.1f%%\n",
				       cdev[i].rx_pause_pct[0]);
				printf("            TX Pause: %3.1f%%\n",
				       cdev[i].tx_pause_pct[0]);
			}
		}
		if (opts.report_aer) {
			printf("    AER CORRECTABLE:\n        ");
			print_aer(cdev[i].aer_cor);
			printf("    AER FATAL:\n        ");
			print_aer(cdev[i].aer_fatal);
			printf("    AER NONFATAL:\n        ");
			print_aer(cdev[i].aer_nonfatal);

			free(cdev[i].aer_cor);
			free(cdev[i].aer_fatal);
			free(cdev[i].aer_nonfatal);
		}
	}
}

void print_cxi_devs_or_macs(struct cxi_dev *cdev, int num_devs, bool print_macs)
{
	int i;

	for (i = 0; i < num_devs; i++) {
		if (print_macs)
			printf("%s\n", cdev[i].mac_addr);
		else
			printf("%s\n", cdev[i].info.device_name);
	}
}

int main(int argc, char *argv[])
{
	int rc = 0;
	int num_devs = 0;
	char *endptr;
	struct cxi_dev *cdev = NULL;
	struct stat_opts opts = { 0 };

	opts.dev_id = -1;
	opts.pause = DEFAULT_PAUSE;

	struct option long_options[] = { { "help", no_argument, 0, 'h' },
					 { "version", no_argument, 0, 'V' },
					 { "list", no_argument, 0, 'l' },
					 { "mac-list", no_argument, 0, 'm' },
					 { "aer", no_argument, 0, 'a' },
					 { "rates", no_argument, 0, 'r' },
					 { "pause", required_argument, 0, 'p' },
					 { "device", required_argument, 0, 'd' },
					 { NULL, 0, 0, 0 } };

	while (1) {
		int option_index = 0;
		int c = getopt_long(argc, argv, "hlmd:arVp:", long_options,
				    &option_index);

		if (c == -1)
			break;
		switch (c) {
		case 'h':
			usage();
			return 0;
		case 'V':
			printf("%s version: %s\n", name, version);
			exit(0);
		case 'l':
			opts.list_devs_only = true;
			break;
		case 'm':
			opts.list_macs_only = true;
			break;
		case 'd':
			if (strlen(optarg) < 4 || strncmp(optarg, "cxi", 3))
				errx(1, "Invalid device name: %s", optarg);
			optarg += 3;

			errno = 0;
			endptr = NULL;
			opts.dev_id = strtol(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0)
				errx(1, "Invalid device name: cxi%s", optarg);
			break;
		case 'a':
			opts.report_aer = true;
			break;
		case 'r':
			opts.report_rates = true;
			break;
		case 'p':
			opts.pause = strtoul(optarg, &endptr, 0);
			break;
		default:
			usage();
			return 1;
		}
	}
	if (optind < argc)
		errx(1, "Unexpected argument: %s", argv[optind]);

	cdev = get_cxi_dev_status(&num_devs, opts);
	get_network_device(cdev, num_devs);
	if (opts.report_aer)
		rc = get_aer(cdev, num_devs);

	if (opts.list_devs_only || opts.list_macs_only)
		print_cxi_devs_or_macs(cdev, num_devs, opts.list_macs_only);
	else
		print_cxi_status(cdev, num_devs, opts);

	free(cdev);
	return rc ? 1 : 0;
}
