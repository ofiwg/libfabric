/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2020-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI heatsink check */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <fcntl.h>
#include <semaphore.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <glob.h>
#include <stdbool.h>
#include <limits.h>
#include <errno.h>
#include <numa.h>
#include <dirent.h>
#include <sensors/sensors.h>
#include <sensors/error.h>

#include "libcxi.h"
#include "utils_common.h"
#include "cassini_cntr_desc.h"

#define WARMUP_USEC 1000000
#define EVENT_TIMEOUT_USEC 1000000
/* Pre-run buffer randomization and post-run quiescence can take some time */
#define SYNC_TIMEOUT_SEC 60
#define NO_TIMEOUT 0
#define BUF_MSG_ALIGN 64
#define NUM_NONMATCHING_LES 32
#define MAX_LEN 256
#define MAX_BUF_LEN 4194304

#define MSG_SIZE_DFLT 512
#define LIST_SIZE_DFLT 256
#define INTERVAL_DFLT 10
#define DURATION_DFLT 600
#define MAX_SIZE 262144

#define FAIL_TEMP 85
#define QSFP_VR_FAIL_TEMP 125
#define QSFP_INT_FAIL_TEMP 70
#define OSFP_INT_FAIL_TEMP 70
#define ABORT_TEMP 90
#define QSFP_VR_ABORT_TEMP 130
#define QSFP_INT_ABORT_TEMP 75
#define OSFP_INT_ABORT_TEMP 75
#define TARGET_BW 19

#define BRD_UNKNOWN 0
#define BRD_BRAZOS 1
#define BRD_SAWTOOTH 2
#define BRD_WASHINGTON 3
#define BRD_KENNEBEC 4
#define BRD_SOUHEGAN 5

#define TIME_W 7
#define RATE_W 10
#define RATE_FRAC_W 2
#define VDD_W 6
#define AVDD_W 7
#define TRVDD_W 8
#define QSFP_P_W 7
#define TEMPS_W 10
#define QSFP_VR_W 11
#define QSFP_INT_W 12
#define OSFP_INT_W 12
#define SAW_CRIT_W 44
#define BRZ_CRIT_W 50
#define RESULT_W 13
#define RESULT_FRAC_W 2

#define MAX_SEM_NAME 32
#define MAX_AFF_STR_LEN 1024
#define QSFP_INT_NA -41 /* Min reading - 1 */
#define OSFP_INT_NA -41 /* Min reading - 1 */
#define QSFP_VR_MIN -40

enum { OPSTATE_ENABLED = 0, OPSTATE_UNAVAILABLE = 2, OPSTATE_IN_TEST = 7 };

static const char *name = "cxi_heatsink_check";
static const char *version = "2.2.2";
static const char *asic_temp_0_name = "Cassini 0 Temperature";
static const char *asic_temp_1_name = "Cassini 1 Temperature";
static const char *qsfp_vr_temp_name = "3.3V QSFP VR Temperature";
static const char *qsfp_int_temp_name = "QSFP Internal Temperature";
static const char *osfp_int_temp_name = "OSFP Internal Temperature";
static const char *vdd_pwr_name = "0.85V S0 Power";
static const char *c2_vdd_pwr_name = "0.765V Power";
static const char *avdd_pwr_name = "0.9V S0 Power";
static const char *c2_avdd_pwr_name = "VDDA 0.75V Power";
static const char *c2_trvdd_pwr_name = "TRVDD 0.9V Power";
static const char *qsfp_pwr_name = "3.3V QSFP Power";
static const char *brz_pn_pref_1 = "10232510";
static const char *brz_pn_pref_2 = "P41345-";
static const char *saw_pn_pref_1 = "10225100";
static const char *saw_pn_pref_2 = "P43012-";
static const char *ken_pn_pref = "P52930-";
static const char *was_pn_pref = "P48765-";
static const char *sou_pn_pref = "P68492-";

#define MAX_HDR_LEN 100
static char results_header[MAX_HDR_LEN];
static size_t header_len;

#define DEV_NUMA_FILE "/sys/class/cxi/cxi%d/device/numa_node"
#define NUMA_CPULIST_FILE "/sys/devices/system/node/node%d/cpulist"
#define CPU_PHYS_PKG_FILE                                                      \
	"/sys/devices/system/cpu/cpu%d/topology/physical_package_id"
#define CPU_THREADS_FILE                                                       \
	"/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list"
#define NUMA_DIR "/sys/devices/system/node"
#define DEV_DIR "/sys/class/cxi"

volatile sig_atomic_t run_finished;

void sigusr1_handler(int signum)
{
	run_finished = 1;
}

struct heatsink_opts {
	uint64_t msg_size;
	uint64_t duration;
	uint64_t interval;
	uint16_t procs;
	uint16_t list_size;
	int no_hrp;
	int no_idc;
	int no_ple;
	int tx_gpu;
	int rx_gpu;
	int gpu_type;
};

/* Container for CXI resources */
struct heatsink_ctx {
	struct cxi_context cxi;
	int proc;
	sem_t *sem_parent;
	sem_t *sem_children;
	int board_type;
	sensors_chip_name const *sensor;

	/* configuration based on multiple command line opts */
	bool use_idc;
	bool use_hrp;
	bool use_tx_gpu;
	bool use_rx_gpu;

	/* avoid building identical commands many times */
	union c_cmdu dma_cmd;
	union c_cmdu idc_cmd;
	union c_cmdu ct_cmd;
	union c_cmdu target_cmd;

	/* sensor "present_reading" sysfs files */
	sensors_subfeature const *asic_temp_0_fp;
	sensors_subfeature const *asic_temp_1_fp;
	sensors_subfeature const *qsfp_vr_temp_fp;
	sensors_subfeature const *qsfp_int_temp_fp;
	sensors_subfeature const *osfp_int_temp_fp;
	sensors_subfeature const *vdd_pwr_fp;
	sensors_subfeature const *avdd_pwr_fp;
	sensors_subfeature const *trvdd_pwr_fp;
	sensors_subfeature const *qsfp_pwr_fp;
};

struct sensor_readings {
	int asic_temp_0;
	int asic_temp_1;
	int qsfp_vr_temp;
	int qsfp_int_temp;
	int osfp_int_temp;
	int vdd_pwr;
	int avdd_pwr;
	int trvdd_pwr;
	int qsfp_pwr;
};

/* Use two semaphores to synchronize the parent and children processes */
int synchronize(struct heatsink_ctx *ctx, struct heatsink_opts *opts)
{
	int rc = 0;
	int i;
	struct timespec tmo;

	clock_gettime(CLOCK_REALTIME, &tmo);
	tmo.tv_sec += SYNC_TIMEOUT_SEC;

	if (!ctx->proc) {
		for (i = 1; i < opts->procs; i++) {
			rc = sem_timedwait(ctx->sem_parent, &tmo);
			if (rc < 0) {
				rc = -errno;
				if (rc != -EINTR)
					fprintf(stderr,
						"proc%d wait failed: %s\n",
						ctx->proc, strerror(-rc));
				return rc;
			}
		}
		for (i = 1; i < opts->procs; i++) {
			rc = sem_post(ctx->sem_children);
			if (rc < 0) {
				rc = -errno;
				fprintf(stderr, "proc%d post failed: %s\n",
					ctx->proc, strerror(-rc));
				return rc;
			}
		}
	} else {
		rc = sem_post(ctx->sem_parent);
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "proc%d wait failed: %s\n", ctx->proc,
				strerror(-rc));
			return rc;
		}
		rc = sem_timedwait(ctx->sem_children, &tmo);
		if (rc < 0) {
			rc = -errno;
			if (rc != -EINTR)
				fprintf(stderr, "proc%d sync failed: %s\n",
					ctx->proc, strerror(-rc));
			return rc;
		}
	}

	return rc;
}

int get_board_type(uint32_t dev_id)
{
	int rc = 0;
	char path[MAX_LEN];
	char part_num[MAX_LEN];
	FILE *fp;

	rc = snprintf(path, MAX_LEN,
		      "/sys/class/cxi/cxi%u/device/fru/part_number", dev_id);
	if (rc < 0)
		return rc;

	fp = fopen(path, "r");
	if (!fp)
		return -errno;

	if (fgets(part_num, MAX_LEN, fp)) {
		if (!strncmp(part_num, brz_pn_pref_1, strlen(brz_pn_pref_1)) ||
		    !strncmp(part_num, brz_pn_pref_2, strlen(brz_pn_pref_2)))
			rc = BRD_BRAZOS;
		else if (!strncmp(part_num, saw_pn_pref_1,
				  strlen(saw_pn_pref_1)) ||
			 !strncmp(part_num, saw_pn_pref_2,
				  strlen(saw_pn_pref_2)))
			rc = BRD_SAWTOOTH;
		else if (!strncmp(part_num, was_pn_pref, strlen(was_pn_pref)))
			rc = BRD_WASHINGTON;
		else if (!strncmp(part_num, ken_pn_pref, strlen(ken_pn_pref)))
			rc = BRD_KENNEBEC;
		else if (!strncmp(part_num, sou_pn_pref, strlen(sou_pn_pref)))
			rc = BRD_SOUHEGAN;
		else
			rc = BRD_UNKNOWN;
	} else {
		rc = -ENODATA;
	}

	fclose(fp);
	return rc;
}

sensors_subfeature const *get_subfeature(sensors_chip_name const *chip,
					 sensors_feature const *feat)
{
	/* Match found, process getting sub feature  */
	sensors_subfeature const *subf;
	int s = 0;

	while ((subf = sensors_get_all_subfeatures(chip, feat, &s)) != 0) {
		if (strstr(subf->name, "in"))
			return subf;
	}

	return NULL;
}

int failed_get_subfeature(const char *name)
{
	fprintf(stderr, "Failed to get subfeature %s\n", name);
	return -1;
}

int open_sensor_files(struct heatsink_ctx *ctx, uint32_t dev_id)
{
	int rc = 0;
	char sensor[MAX_LEN];
	int count = 0;

	sensors_chip_name const *chip;

	/* find the sensor chip and save for use later */
	rc = snprintf(sensor, MAX_LEN, "cxi%u", dev_id);
	if (rc < 0)
		return rc;

	rc = sensors_init(NULL);
	if (rc != 0) {
		fprintf(stderr, "failed to initialize sensor data:%s\n",
			sensors_strerror(rc));
		return rc;
	}

	ctx->sensor = NULL;
	while ((chip = sensors_get_detected_chips(0, &count)) != 0) {
		if (strstr(chip->prefix, sensor) != NULL) {
			ctx->sensor = chip;
			break;
		}
	}

	if (ctx->sensor == NULL) {
		fprintf(stderr, "failed to find %s\n", sensor);
		return -1;
	}

	sensors_feature const *feat;
	int f = 0;

	while ((feat = sensors_get_features(ctx->sensor, &f)) != 0) {
		/* Compare names to features, mark which features we have */
		if (!ctx->asic_temp_0_fp &&
		    !strncmp(sensors_get_label(ctx->sensor, feat),
			     asic_temp_0_name, strlen(asic_temp_0_name))) {
			ctx->asic_temp_0_fp = get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_SAWTOOTH ||
			    ctx->board_type == BRD_WASHINGTON ||
			    ctx->board_type == BRD_SOUHEGAN) &&
			    !ctx->asic_temp_1_fp &&
			    !strncmp(sensors_get_label(ctx->sensor, feat),
				     asic_temp_1_name,
				     strlen(asic_temp_1_name))) {
			ctx->asic_temp_1_fp = get_subfeature(ctx->sensor, feat);
		} else if (ctx->board_type == BRD_BRAZOS &&
			   !ctx->qsfp_vr_temp_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    qsfp_vr_temp_name,
				    strlen(qsfp_vr_temp_name))) {
			ctx->qsfp_vr_temp_fp =
				get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_BRAZOS ||
			    ctx->board_type == BRD_KENNEBEC) &&
			   !ctx->qsfp_int_temp_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    qsfp_int_temp_name,
				    strlen(qsfp_int_temp_name))) {
			ctx->qsfp_int_temp_fp =
				get_subfeature(ctx->sensor, feat);
		} else if (ctx->board_type == BRD_SOUHEGAN &&
			   !ctx->osfp_int_temp_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    osfp_int_temp_name,
				    strlen(osfp_int_temp_name))) {
			ctx->osfp_int_temp_fp =
				get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_BRAZOS ||
			    ctx->board_type == BRD_SAWTOOTH) &&
			   !ctx->vdd_pwr_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    vdd_pwr_name, strlen(vdd_pwr_name))) {
			ctx->vdd_pwr_fp = get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_KENNEBEC ||
			    ctx->board_type == BRD_WASHINGTON ||
			    ctx->board_type == BRD_SOUHEGAN) &&
			   !ctx->vdd_pwr_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    c2_vdd_pwr_name, strlen(c2_vdd_pwr_name))) {
			ctx->vdd_pwr_fp = get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_BRAZOS ||
			    ctx->board_type == BRD_SAWTOOTH) &&
			   !ctx->avdd_pwr_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    avdd_pwr_name, strlen(avdd_pwr_name))) {
			ctx->avdd_pwr_fp = get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_KENNEBEC ||
			    ctx->board_type == BRD_WASHINGTON ||
			    ctx->board_type == BRD_SOUHEGAN) &&
			   !ctx->avdd_pwr_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    c2_avdd_pwr_name,
				    strlen(c2_avdd_pwr_name))) {
			ctx->avdd_pwr_fp = get_subfeature(ctx->sensor, feat);
		} else if (ctx->board_type == BRD_BRAZOS && !ctx->qsfp_pwr_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    qsfp_pwr_name, strlen(qsfp_pwr_name))) {
			ctx->qsfp_pwr_fp = get_subfeature(ctx->sensor, feat);
		} else if ((ctx->board_type == BRD_KENNEBEC ||
			    ctx->board_type == BRD_WASHINGTON ||
			    ctx->board_type == BRD_SOUHEGAN) &&
			   !ctx->trvdd_pwr_fp &&
			   !strncmp(sensors_get_label(ctx->sensor, feat),
				    c2_trvdd_pwr_name,
				    strlen(c2_trvdd_pwr_name))) {
			ctx->trvdd_pwr_fp = get_subfeature(ctx->sensor, feat);
		}
	}

	/* Verify all found */
	rc = 0;
	if (!ctx->asic_temp_0_fp) {
		rc = failed_get_subfeature(asic_temp_0_name);
	} else if (!ctx->asic_temp_1_fp &&
		   (ctx->board_type == BRD_SAWTOOTH ||
		    ctx->board_type == BRD_WASHINGTON ||
		    ctx->board_type == BRD_SOUHEGAN)) {
		rc = failed_get_subfeature(asic_temp_1_name);
	} else if (!ctx->vdd_pwr_fp) {
		if (ctx->board_type == BRD_WASHINGTON ||
		    ctx->board_type == BRD_KENNEBEC ||
		    ctx->board_type == BRD_SOUHEGAN)
			rc = failed_get_subfeature(c2_vdd_pwr_name);
		else
			rc = failed_get_subfeature(vdd_pwr_name);
	} else if (!ctx->avdd_pwr_fp) {
		if (ctx->board_type == BRD_WASHINGTON ||
		    ctx->board_type == BRD_KENNEBEC ||
		    ctx->board_type == BRD_SOUHEGAN)
			rc = failed_get_subfeature(c2_avdd_pwr_name);
		else
			rc = failed_get_subfeature(avdd_pwr_name);
	} else if (!ctx->qsfp_vr_temp_fp && ctx->board_type == BRD_BRAZOS) {
		rc = failed_get_subfeature(qsfp_vr_temp_name);
	} else if (!ctx->qsfp_pwr_fp && ctx->board_type == BRD_BRAZOS) {
		rc = failed_get_subfeature(qsfp_pwr_name);
	} else if (!ctx->qsfp_int_temp_fp && (ctx->board_type == BRD_KENNEBEC ||
					      ctx->board_type == BRD_BRAZOS)) {
		rc = failed_get_subfeature(qsfp_int_temp_name);
	} else if (!ctx->osfp_int_temp_fp &&
		   (ctx->board_type == BRD_SOUHEGAN)) {
		rc = failed_get_subfeature(osfp_int_temp_name);
	} else if (!ctx->trvdd_pwr_fp && (ctx->board_type == BRD_WASHINGTON ||
					  ctx->board_type == BRD_KENNEBEC ||
					  ctx->board_type == BRD_SOUHEGAN)) {
		rc = failed_get_subfeature(c2_trvdd_pwr_name);
	}

	return rc;
}

void close_sensor_files(struct heatsink_ctx *ctx)
{
	sensors_cleanup();
}

/* Allocate resources */
int heatsink_alloc(struct heatsink_ctx *ctx, struct heatsink_opts *opts)
{
	int rc;
	int i;
	struct cxi_ctx_ini_opts ini_opts = { 0 };
	struct cxi_ctx_tgt_opts tgt_opts = { 0 };
	struct cxi_context *cxi = &ctx->cxi;
	union c_cmdu c_st_cmd = { 0 };
	uint32_t flags;
	size_t buf_len;
	size_t eq_len;
	uint64_t match_bits;

	buf_len = (opts->msg_size + BUF_MSG_ALIGN - 1) / BUF_MSG_ALIGN;
	buf_len *= BUF_MSG_ALIGN;
	buf_len *= opts->list_size;
	if (buf_len > MAX_BUF_LEN)
		buf_len = MAX_BUF_LEN;

	eq_len = (opts->list_size + 1) * 64;
	eq_len = NEXT_MULTIPLE(eq_len, s_page_size);

	/* Config */
	ini_opts.alloc_hrp = ctx->use_hrp;
	ini_opts.alloc_ct = true;

	ini_opts.eq_attr.queue_len = eq_len;
	ini_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	ini_opts.buf_opts.length = buf_len;
	ini_opts.buf_opts.pattern = CTX_BUF_PAT_URAND;
	ini_opts.use_gpu_buf = ctx->use_tx_gpu;
	ini_opts.gpu_id = opts->tx_gpu;

	/* count granularity is 64B, and IDC commands can be up to 256B each
	 * with a preceding 64B CT State command. So worst case we need 4x + 1
	 */
	ini_opts.cq_opts.count = (opts->list_size * 4) + 1;
	ini_opts.cq_opts.flags = CXI_CQ_IS_TX;

	tgt_opts.eq_attr.queue_len = eq_len;
	tgt_opts.eq_attr.flags = CXI_EQ_EC_DISABLE;

	tgt_opts.buf_opts.length = buf_len;
	tgt_opts.buf_opts.pattern = CTX_BUF_PAT_ZERO;
	tgt_opts.use_gpu_buf = ctx->use_rx_gpu;
	tgt_opts.gpu_id = opts->rx_gpu;

	tgt_opts.cq_opts.count = opts->list_size * 4;

	tgt_opts.pt_opts.is_matching = !ctx->use_hrp;

	/* Allocate */
	cxi->rmt_addr.nic = cxi->loc_addr.nic;
	cxi->rmt_addr.pid = cxi->loc_addr.pid;
	rc = alloc_ini(cxi, &ini_opts);
	if (rc < 0)
		return rc;
	rc = alloc_tgt(cxi, &tgt_opts);
	if (rc < 0)
		return rc;

	if (ctx->use_hrp) {
		rc = cxi_cq_emit_cq_lcid(cxi->ini_cq, cxi->hrp_cp->lcid);
		if (rc) {
			fprintf(stderr,
				"Failed to change Tx CQ to HRP LCID: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	if (ctx->use_idc) {
		/* C State */
		c_st_cmd.c_state.command.opcode = C_CMD_CSTATE;
		c_st_cmd.c_state.index_ext = cxi->index_ext;
		c_st_cmd.c_state.event_send_disable = 1;
		c_st_cmd.c_state.event_success_disable = 1;
		c_st_cmd.c_state.event_ct_ack = 1;
		if (ctx->use_hrp)
			c_st_cmd.c_state.restricted = 1;
		c_st_cmd.c_state.eq = cxi->ini_eq->eqn;
		c_st_cmd.c_state.ct = cxi->ini_ct->ctn;

		rc = cxi_cq_emit_c_state(cxi->ini_cq, &c_st_cmd.c_state);
		if (rc) {
			fprintf(stderr, "Failed to issue C State command: %s\n",
				strerror(-rc));
			return rc;
		}
		cxi_cq_ring(cxi->ini_cq);

		/* IDC Put/Msg */
		if (ctx->use_hrp) {
			ctx->idc_cmd.idc_put.idc_header.command.opcode =
				C_CMD_NOMATCH_PUT;
			ctx->idc_cmd.idc_put.idc_header.dfa = cxi->dfa;
		} else {
			ctx->idc_cmd.idc_msg.command.opcode =
				C_CMD_SMALL_MESSAGE;
			ctx->idc_cmd.idc_msg.dfa = cxi->dfa;
			if (!opts->no_ple)
				ctx->idc_cmd.idc_msg.match_bits = 2;
		}
	} else {
		/* Nomatch HRP or Matching Put DMA */
		ctx->dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		if (ctx->use_hrp) {
			ctx->dma_cmd.nomatch_dma.command.opcode =
				C_CMD_NOMATCH_PUT;
			ctx->dma_cmd.nomatch_dma.index_ext = cxi->index_ext;
			ctx->dma_cmd.nomatch_dma.lac = cxi->ini_buf->md->lac;
			ctx->dma_cmd.nomatch_dma.event_send_disable = 1;
			ctx->dma_cmd.nomatch_dma.event_success_disable = 1;
			ctx->dma_cmd.nomatch_dma.event_ct_ack = 1;
			ctx->dma_cmd.nomatch_dma.restricted = 1;
			ctx->dma_cmd.nomatch_dma.dfa = cxi->dfa;
			ctx->dma_cmd.nomatch_dma.eq = cxi->ini_eq->eqn;
			ctx->dma_cmd.nomatch_dma.ct = cxi->ini_ct->ctn;
		} else {
			ctx->dma_cmd.full_dma.command.opcode = C_CMD_PUT;
			ctx->dma_cmd.full_dma.index_ext = cxi->index_ext;
			ctx->dma_cmd.full_dma.lac = cxi->ini_buf->md->lac;
			ctx->dma_cmd.full_dma.event_send_disable = 1;
			ctx->dma_cmd.full_dma.event_success_disable = 1;
			ctx->dma_cmd.full_dma.event_ct_ack = 1;
			ctx->dma_cmd.full_dma.dfa = cxi->dfa;
			ctx->dma_cmd.full_dma.eq = cxi->ini_eq->eqn;
			ctx->dma_cmd.full_dma.ct = cxi->ini_ct->ctn;
			if (!opts->no_ple)
				ctx->dma_cmd.full_dma.match_bits = 2;
		}
	}

	/* CT Event command setup */
	ctx->ct_cmd.ct.eq = cxi->ini_eq->eqn;
	ctx->ct_cmd.ct.trig_ct = cxi->ini_ct->ctn;

	flags = C_LE_EVENT_SUCCESS_DISABLE | C_LE_EVENT_COMM_DISABLE |
		C_LE_EVENT_UNLINK_DISABLE | C_LE_UNRESTRICTED_BODY_RO |
		C_LE_UNRESTRICTED_END_RO | C_LE_OP_PUT;

	/* Append non-matching LEs to make the matchers work harder for PUTs */
	if (!ctx->use_hrp) {
		match_bits = 1;
		for (i = 0; i < NUM_NONMATCHING_LES; i++) {
			rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf,
				       0, flags, cxi->tgt_pte->ptn, 0, 0,
				       match_bits, 0, 0);
			if (rc)
				return rc;
		}
	}

	/* Append Persistent LE */
	if (!opts->no_ple) {
		if (ctx->use_hrp) {
			rc = append_le(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf,
				       0, flags, cxi->tgt_pte->ptn, 0, 0);
			if (rc)
				return rc;
		} else {
			match_bits = 2;
			rc = append_me(cxi->tgt_cq, cxi->tgt_eq, cxi->tgt_buf,
				       0, flags, cxi->tgt_pte->ptn, 0, 0,
				       match_bits, 0, 0);
			if (rc)
				return rc;
		}
	}
	return rc;
}

/* Send list_size writes and wait for their ACKs */
int do_single_iteration(struct heatsink_ctx *ctx, struct heatsink_opts *opts,
			uint64_t buf_granularity)
{
	int rc = 0;
	struct cxi_context *cxi = &ctx->cxi;
	int i;
	uint64_t rmt_offset = 0;
	uint64_t local_addr = (uintptr_t)cxi->ini_buf->buf;
	uint64_t loc_addr_end_offset;
	uint32_t flags;
	uint64_t match_bits;
	bool last_append = false;

	/* Append Use-Once LEs */
	if (opts->no_ple) {
		flags = C_LE_EVENT_SUCCESS_DISABLE | C_LE_EVENT_COMM_DISABLE |
			C_LE_EVENT_UNLINK_DISABLE | C_LE_EVENT_LINK_DISABLE |
			C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
			C_LE_OP_PUT | C_LE_USE_ONCE;
		for (i = 0; i < opts->list_size; i++) {
			last_append = (i == opts->list_size - 1);
			if (last_append)
				flags = flags ^ C_LE_EVENT_LINK_DISABLE;
			if (ctx->use_hrp) {
				rc = append_le(cxi->tgt_cq,
					       (last_append ? cxi->tgt_eq :
							      NULL),
					       cxi->tgt_buf, rmt_offset, flags,
					       cxi->tgt_pte->ptn, 0, 0);
				if (rc)
					return rc;
			} else {
				match_bits = (i + 2);
				rc = append_me(
					cxi->tgt_cq,
					(last_append ? cxi->tgt_eq : NULL),
					cxi->tgt_buf, rmt_offset, flags,
					cxi->tgt_pte->ptn, 0, 0, match_bits, 0,
					0);
				if (rc)
					return rc;
			}
			rmt_offset += buf_granularity;
		}
	}

	rc = inc_ct(cxi->ini_trig_cq, &ctx->ct_cmd.ct, opts->list_size);
	if (rc)
		return rc;

	rmt_offset = 0;

	/* Enqueue TX command and ring doorbell */
	for (i = 0; i < opts->list_size; i++) {
		if (ctx->use_idc && ctx->use_hrp) {
			/* IDC & HRP */
			if (!opts->no_ple)
				ctx->idc_cmd.idc_put.idc_header.remote_offset =
					rmt_offset;
			rc = cxi_cq_emit_idc_put(cxi->ini_cq,
						 &ctx->idc_cmd.idc_put,
						 (void *)local_addr,
						 opts->msg_size);
			if (rc) {
				fprintf(stderr,
					"Failed to issue IDC Put command %d: %s\n",
					i, strerror(-rc));
				return rc;
			}
		} else if (ctx->use_idc) {
			/* IDC & PUT */
			if (opts->no_ple)
				ctx->idc_cmd.idc_msg.match_bits = (i + 2);
			rc = cxi_cq_emit_idc_msg(cxi->ini_cq,
						 &ctx->idc_cmd.idc_msg,
						 (void *)local_addr,
						 opts->msg_size);
			if (rc) {
				fprintf(stderr,
					"Failed to issue IDC Msg command %d: %s\n",
					i, strerror(-rc));
				return rc;
			}
		} else if (ctx->use_hrp) {
			/* DMA & HRP */
			ctx->dma_cmd.nomatch_dma.request_len = opts->msg_size;
			if (!opts->no_ple)
				ctx->dma_cmd.nomatch_dma.remote_offset =
					rmt_offset;
			ctx->dma_cmd.nomatch_dma.local_addr =
				CXI_VA_TO_IOVA(cxi->ini_buf->md, local_addr);
			rc = cxi_cq_emit_nomatch_dma(cxi->ini_cq,
						     &ctx->dma_cmd.nomatch_dma);
			if (rc) {
				fprintf(stderr,
					"Failed to issue Nomatch DMA command %d: %s\n",
					i, strerror(-rc));
				return rc;
			}
		} else {
			/* DMA & PUT */
			ctx->dma_cmd.full_dma.request_len = opts->msg_size;
			if (opts->no_ple)
				ctx->dma_cmd.full_dma.match_bits = (i + 2);
			else
				ctx->dma_cmd.full_dma.remote_offset =
					rmt_offset;
			ctx->dma_cmd.full_dma.local_addr =
				CXI_VA_TO_IOVA(cxi->ini_buf->md, local_addr);
			rc = cxi_cq_emit_dma(cxi->ini_cq,
					     &ctx->dma_cmd.full_dma);
			if (rc) {
				fprintf(stderr,
					"Failed to issue Full DMA command %d: %s\n",
					i, strerror(-rc));
				return rc;
			}
		}

		local_addr += buf_granularity;
		rmt_offset += buf_granularity;
		loc_addr_end_offset = local_addr + buf_granularity -
				      (uintptr_t)cxi->ini_buf->buf - 1;
		if ((loc_addr_end_offset >= cxi->ini_buf->md->len) ||
		    (loc_addr_end_offset >= MAX_BUF_LEN)) {
			rmt_offset = 0;
			local_addr = (uintptr_t)cxi->ini_buf->buf;
		}
	}
	cxi_cq_ring(cxi->ini_cq);

	/* Wait for ACK CT Event */
	rc = wait_for_ct(cxi->ini_eq, NO_TIMEOUT, "initiator ACK");
	if (rc)
		return rc;

	return rc;
}

/* Get response count for calculating bandwidth */
uint64_t get_pkt_count(struct heatsink_ctx *ctx, struct heatsink_opts *opts)
{
	int rc;
	uint64_t count = 0;

	if (ctx->use_hrp) {
		rc = cxil_read_cntr(ctx->cxi.dev,
				    C_CNTR_PCT_HRP_RESPONSES_RECEIVED,
				    &count, NULL);
		if (rc)
			fprintf(stderr,
				"Failed to read PCT HRP Response count: %s\n",
				strerror(-rc));
	} else {
		rc = cxil_read_cntr(ctx->cxi.dev, C_CNTR_PCT_PTLS_RESPONSE,
				    &count, NULL);
		if (rc)
			fprintf(stderr,
				"Failed to read PCT Portals Response count: %s\n",
				strerror(-rc));
	}
	return count;
}

/* Read a single sensor reading sensors data */
int get_sensor_reading(sensors_chip_name const *chip,
		       sensors_subfeature const *subf, int *rd,
		       const char *name, bool skip_err)
{
	int rc = 0;
	double val;

	/* Check sensor state and get reading */
	if (subf->flags & SENSORS_MODE_R)
		rc = sensors_get_value(chip, subf->number, &val);

	if ((skip_err && (rc == -SENSORS_ERR_ACCESS_R)) || (rc == 0))
		*rd = val;
	else
		fprintf(stderr, "Failed to get reading for %s: %s\n", name,
			sensors_strerror(rc));

	return rc;
}

/* Get the latest sensor readings */
int get_sensor_readings(struct heatsink_ctx *ctx, struct sensor_readings *rd)
{
	int rc;

	rc = get_sensor_reading(ctx->sensor, ctx->asic_temp_0_fp,
				&rd->asic_temp_0, asic_temp_0_name, false);
	if (rc)
		return rc;

	if (ctx->board_type == BRD_SAWTOOTH ||
	    ctx->board_type == BRD_WASHINGTON ||
	    ctx->board_type == BRD_SOUHEGAN) {
		rc = get_sensor_reading(ctx->sensor, ctx->asic_temp_1_fp,
					&rd->asic_temp_1, asic_temp_1_name,
					false);
		if (rc)
			return rc;
	}
	if (ctx->board_type == BRD_BRAZOS) {
		rc = get_sensor_reading(ctx->sensor, ctx->qsfp_vr_temp_fp,
					&rd->qsfp_vr_temp, qsfp_vr_temp_name,
					false);
		if (rc)
			return rc;

		rc = get_sensor_reading(ctx->sensor, ctx->qsfp_pwr_fp,
					&rd->qsfp_pwr, qsfp_pwr_name, false);
		if (rc)
			return rc;
	}
	if (ctx->board_type == BRD_BRAZOS || ctx->board_type == BRD_KENNEBEC) {
		rc = get_sensor_reading(ctx->sensor, ctx->qsfp_int_temp_fp,
					&rd->qsfp_int_temp, qsfp_int_temp_name,
					true);
		if (rc == -SENSORS_ERR_ACCESS_R)
			rd->qsfp_int_temp = QSFP_INT_NA;
		else if (rc)
			return rc;
	}
	if (ctx->board_type == BRD_SOUHEGAN) {
		rc = get_sensor_reading(ctx->sensor, ctx->osfp_int_temp_fp,
					&rd->osfp_int_temp, osfp_int_temp_name,
					true);
		if (rc == -SENSORS_ERR_ACCESS_R)
			rd->osfp_int_temp = OSFP_INT_NA;
		else if (rc)
			return rc;
	}
	if (ctx->board_type == BRD_WASHINGTON ||
	    ctx->board_type == BRD_KENNEBEC ||
	    ctx->board_type == BRD_SOUHEGAN) {
		rc = get_sensor_reading(ctx->sensor, ctx->trvdd_pwr_fp,
					&rd->trvdd_pwr, c2_trvdd_pwr_name,
					false);
		if (rc)
			return rc;

		rc = get_sensor_reading(ctx->sensor, ctx->vdd_pwr_fp,
					&rd->vdd_pwr, c2_vdd_pwr_name, false);
		if (rc)
			return rc;

		rc = get_sensor_reading(ctx->sensor, ctx->avdd_pwr_fp,
					&rd->avdd_pwr, c2_avdd_pwr_name, false);
		if (rc)
			return rc;
	} else {
		rc = get_sensor_reading(ctx->sensor, ctx->vdd_pwr_fp,
					&rd->vdd_pwr, vdd_pwr_name, false);
		if (rc)
			return rc;

		rc = get_sensor_reading(ctx->sensor, ctx->avdd_pwr_fp,
					&rd->avdd_pwr, avdd_pwr_name, false);
		if (rc)
			return rc;
	}

	return 0;
}

/* Print results and return 0 if passing or 1 if failing */
int print_results(struct sensor_readings max, long double bw_avg,
		  int board_type)
{
	int rc = 0;
	char criteria[MAX_LEN];
	char result[MAX_LEN];
	int crit_w;

	if (board_type == BRD_SAWTOOTH || board_type == BRD_WASHINGTON) {
		crit_w = SAW_CRIT_W;
	} else {
		crit_w = BRZ_CRIT_W;
	}

	print_separator(header_len);
	snprintf(criteria, MAX_LEN,
		 "%s (ASIC_0) under %d °C:", asic_temp_0_name, FAIL_TEMP);
	snprintf(result, MAX_LEN, "%d °C", max.asic_temp_0);
	printf("%-*s  %-*s", crit_w, criteria, RESULT_W, result);
	if (max.asic_temp_0 < FAIL_TEMP) {
		printf("PASS\n");
	} else {
		printf("FAIL\n");
		rc = 1;
	}
	if (board_type == BRD_SAWTOOTH ||
	    board_type == BRD_WASHINGTON ||
	    board_type == BRD_SOUHEGAN) {
		snprintf(criteria, MAX_LEN,
			 "%s (ASIC_1) under %d °C:", asic_temp_1_name,
			 FAIL_TEMP);
		snprintf(result, MAX_LEN, "%d °C", max.asic_temp_1);
		printf("%-*s  %-*s", crit_w, criteria, RESULT_W, result);
		if (max.asic_temp_1 < FAIL_TEMP) {
			printf("PASS\n");
		} else {
			printf("FAIL\n");
			rc = 1;
		}
	}
	if (board_type == BRD_BRAZOS) {
		snprintf(criteria, MAX_LEN,
			 "%s (QSFP_VR) under %d °C:", qsfp_vr_temp_name,
			 QSFP_VR_FAIL_TEMP);
		snprintf(result, MAX_LEN, "%d °C", max.qsfp_vr_temp);
		printf("%-*s  %-*s", crit_w, criteria, RESULT_W, result);
		if (max.qsfp_vr_temp < QSFP_VR_FAIL_TEMP) {
			printf("PASS\n");
		} else {
			printf("FAIL\n");
			rc = 1;
		}
	}
	if (board_type == BRD_BRAZOS || board_type == BRD_KENNEBEC) {
		snprintf(criteria, MAX_LEN,
			 "%s (QSFP_INT) under %d °C:", qsfp_int_temp_name,
			 QSFP_INT_FAIL_TEMP);
		if (max.qsfp_int_temp == QSFP_INT_NA) {
			printf("%-*s  %-*s  NA\n", crit_w, criteria,
			       (RESULT_W - 1), "-");
		} else {
			snprintf(result, MAX_LEN, "%d °C", max.qsfp_int_temp);
			printf("%-*s  %-*s", crit_w, criteria, RESULT_W,
			       result);
			if (max.qsfp_int_temp < QSFP_INT_FAIL_TEMP) {
				printf("PASS\n");
			} else {
				printf("FAIL\n");
				rc = 1;
			}
		}
	}
	if (board_type == BRD_SOUHEGAN) {
		snprintf(criteria, MAX_LEN,
			 "%s (OSFP_INT) under %d °C:", osfp_int_temp_name,
			 OSFP_INT_FAIL_TEMP);
		if (max.osfp_int_temp == OSFP_INT_NA) {
			printf("%-*s  %-*s  NA\n", crit_w, criteria,
			       (RESULT_W - 1), "-");
		} else {
			snprintf(result, MAX_LEN, "%d °C", max.osfp_int_temp);
			printf("%-*s  %-*s", crit_w, criteria, RESULT_W,
			       result);
			if (max.osfp_int_temp < OSFP_INT_FAIL_TEMP) {
				printf("PASS\n");
			} else {
				printf("FAIL\n");
				rc = 1;
			}
		}
	}
	if (board_type == BRD_WASHINGTON ||
	    board_type == BRD_KENNEBEC ||
	    board_type == BRD_SOUHEGAN) {
		snprintf(criteria, MAX_LEN,
			 "%s (VDD):", c2_vdd_pwr_name);
	} else {
		snprintf(criteria, MAX_LEN,
			 "%s (VDD):", vdd_pwr_name);
	}
	snprintf(result, MAX_LEN, "%d W", max.vdd_pwr);
	printf("%-*s  %-*s", (crit_w - 1), criteria, (RESULT_W - 1), result);
	printf("----\n");
	if (board_type == BRD_WASHINGTON ||
	    board_type == BRD_KENNEBEC ||
	    board_type == BRD_SOUHEGAN) {
		snprintf(criteria, MAX_LEN,
			 "%s (AVDD):", c2_avdd_pwr_name);
	} else {
		snprintf(criteria, MAX_LEN,
			 "%s (AVDD):", avdd_pwr_name);
	}
	snprintf(result, MAX_LEN, "%d W", max.avdd_pwr);
	printf("%-*s  %-*s", (crit_w - 1), criteria, (RESULT_W - 1), result);
	printf("----\n");
	snprintf(criteria, MAX_LEN, "Average BW over %d GB/s:", TARGET_BW);
	snprintf(result, (MAX_LEN - 5), "%.*Lf GB/s", RESULT_FRAC_W, bw_avg);
	printf("%-*s  %-*s", (crit_w - 1), criteria, (RESULT_W - 1), result);
	if (bw_avg >= TARGET_BW) {
		printf("PASS\n");
	} else {
		printf("FAIL\n");
		rc = 1;
	}

	return rc;
}

/* Monitor temperature and power sensors as well as Put bandwidth. Exit early if
 * the temperature gets too high.
 */
int monitor_sensors(struct heatsink_ctx *ctx, struct heatsink_opts *opts)
{
	int rc;
	uint64_t start_time;
	uint64_t duration_usec;
	uint64_t elapsed;
	uint64_t interval_usec;
	uint64_t interval_start_time;
	uint64_t interval_elapsed;
	uint64_t pkt_count_pre;
	uint64_t pkt_count_post;
	uint64_t pkts_per_msg;
	long double bw;
	long double bw_sum;
	int bw_count;
	struct sensor_readings latest;
	struct sensor_readings max = { .qsfp_vr_temp = QSFP_VR_MIN,
				       .qsfp_int_temp = QSFP_INT_NA };

	/* Warmup to avoid BW blips */
	elapsed = 0;
	start_time = gettimeofday_usec();
	while (elapsed < WARMUP_USEC)
		elapsed = gettimeofday_usec() - start_time;

	/* Get sensor readings and BW after every interval */
	elapsed = 0;
	interval_elapsed = 0;
	duration_usec = opts->duration * 1000000;
	interval_usec = opts->interval * 1000000;
	start_time = gettimeofday_usec();
	interval_start_time = start_time;
	pkt_count_pre = get_pkt_count(ctx, opts);
	bw_count = 0;
	bw_sum = 0;
	while (elapsed < duration_usec) {
		elapsed = gettimeofday_usec() - start_time;
		interval_elapsed = elapsed + start_time - interval_start_time;

		if (interval_elapsed >= interval_usec) {
			pkt_count_post = get_pkt_count(ctx, opts);
			pkts_per_msg = ((opts->msg_size - 1) / PORTALS_MTU) + 1;
			bw = (pkt_count_post - pkt_count_pre) * opts->msg_size;
			bw /= pkts_per_msg;
			bw /= interval_elapsed;
			bw /= 1000;
			bw_sum += bw;
			bw_count++;
			rc = get_sensor_readings(ctx, &latest);
			if (rc)
				break;

			printf("%*lu  %*.*Lf  %*d  %*d", TIME_W,
			       (elapsed / 1000000), RATE_W, RATE_FRAC_W, bw,
			       VDD_W, latest.vdd_pwr, AVDD_W, latest.avdd_pwr);
			if (ctx->board_type == BRD_KENNEBEC ||
			    ctx->board_type == BRD_WASHINGTON ||
			    ctx->board_type == BRD_SOUHEGAN)
				printf("  %*d", TRVDD_W, latest.trvdd_pwr);
			if (ctx->board_type == BRD_SAWTOOTH ||
			    ctx->board_type == BRD_WASHINGTON ||
			    ctx->board_type == BRD_SOUHEGAN) {
				printf("  %*d  %*d\n", TEMPS_W,
				       latest.asic_temp_0, TEMPS_W,
				       latest.asic_temp_1);
			}
			if (ctx->board_type == BRD_BRAZOS) {
				printf("  %*d", QSFP_P_W, latest.qsfp_pwr);
				printf("  %*d", QSFP_VR_W, latest.qsfp_vr_temp);
			}
			if (ctx->board_type == BRD_BRAZOS ||
			    ctx->board_type == BRD_KENNEBEC) {
				printf("  %*d", TEMPS_W, latest.asic_temp_0);
				if (latest.qsfp_int_temp > QSFP_INT_NA)
					printf("  %*d\n", QSFP_INT_W,
					       latest.qsfp_int_temp);
				else
					printf("  %*s\n", QSFP_INT_W, "-");
			}
			if (ctx->board_type == BRD_SOUHEGAN) {
				if (latest.osfp_int_temp > OSFP_INT_NA)
					printf("  %*d\n", OSFP_INT_W,
					       latest.osfp_int_temp);
				else
					printf("  %*s\n", OSFP_INT_W, "-");
			}

			/* Overtemp Check */
			if (latest.asic_temp_0 >= ABORT_TEMP) {
				printf("\nError! %s has exceeded safe range. Exiting early.\n",
				       asic_temp_0_name);
				break;
			}
			if ((ctx->board_type == BRD_SAWTOOTH ||
			     ctx->board_type == BRD_WASHINGTON ||
			     ctx->board_type == BRD_SOUHEGAN) &&
			    latest.asic_temp_1 >= ABORT_TEMP) {
				printf("\nError! %s has exceeded safe range. Exiting early.\n",
				       asic_temp_1_name);
				break;
			}
			if (ctx->board_type == BRD_BRAZOS &&
			    latest.qsfp_vr_temp >= QSFP_VR_ABORT_TEMP) {
				printf("\nError! %s has exceeded safe range. Exiting early.\n",
				       qsfp_vr_temp_name);
				break;
			}
			if ((ctx->board_type == BRD_BRAZOS ||
			     ctx->board_type == BRD_KENNEBEC) &&
			    latest.qsfp_int_temp >= QSFP_INT_ABORT_TEMP) {
				printf("\nError! %s has exceeded safe range. Exiting early.\n",
				       qsfp_int_temp_name);
				break;
			}
			if ((ctx->board_type == BRD_SOUHEGAN) &&
			    latest.osfp_int_temp >= OSFP_INT_ABORT_TEMP) {
				printf("\nError! %s has exceeded safe range.",
				       osfp_int_temp_name);
				printf(" Exiting early.\n");
				break;
			}

			/* High water marks */
			if (latest.asic_temp_0 > max.asic_temp_0)
				max.asic_temp_0 = latest.asic_temp_0;
			if ((ctx->board_type == BRD_SAWTOOTH ||
			     ctx->board_type == BRD_WASHINGTON ||
			     ctx->board_type == BRD_SOUHEGAN) &&
			    latest.asic_temp_1 > max.asic_temp_1)
				max.asic_temp_1 = latest.asic_temp_1;
			if (ctx->board_type == BRD_BRAZOS) {
				if (latest.qsfp_vr_temp > max.qsfp_vr_temp)
					max.qsfp_vr_temp = latest.qsfp_vr_temp;
				if (latest.qsfp_pwr > max.qsfp_pwr)
					max.qsfp_pwr = latest.qsfp_pwr;
			}
			if ((ctx->board_type == BRD_BRAZOS ||
			     ctx->board_type == BRD_KENNEBEC) &&
			    latest.qsfp_int_temp > max.qsfp_int_temp)
				max.qsfp_int_temp = latest.qsfp_int_temp;
			if ((ctx->board_type == BRD_SOUHEGAN) &&
			    latest.osfp_int_temp > max.osfp_int_temp)
				max.osfp_int_temp = latest.osfp_int_temp;
			if (latest.vdd_pwr > max.vdd_pwr)
				max.vdd_pwr = latest.vdd_pwr;
			if (latest.avdd_pwr > max.avdd_pwr)
				max.avdd_pwr = latest.avdd_pwr;
			if ((ctx->board_type == BRD_WASHINGTON ||
			     ctx->board_type == BRD_KENNEBEC ||
			     ctx->board_type == BRD_SOUHEGAN) &&
			    latest.trvdd_pwr > max.trvdd_pwr)
				max.trvdd_pwr = latest.trvdd_pwr;

			pkt_count_pre = pkt_count_post;
			interval_start_time += interval_usec;
		}
	}

	/* Stop the other processes */
	kill(0, SIGUSR1);

	if (!rc)
		rc = print_results(max, (bw_count ? (bw_sum / bw_count) : 0),
				   ctx->board_type);

	return rc;
}

/* For children processes, perform writes until signalled by the parent to stop.
 * For the parent process, monitor sensors for the given duration.
 */
int run_heatsink_check(struct heatsink_ctx *ctx, struct heatsink_opts *opts)
{
	int rc = 0;
	uint64_t buf_granularity;

	if (!ctx->proc)
		return monitor_sensors(ctx, opts);

	buf_granularity = (opts->msg_size + BUF_MSG_ALIGN - 1) / BUF_MSG_ALIGN;
	buf_granularity *= BUF_MSG_ALIGN;
	while (!run_finished) {
		rc = do_single_iteration(ctx, opts, buf_granularity);
		if (rc) {
			if (rc != -ECANCELED)
				fprintf(stderr, "proc%d Iteration failed: %s\n",
					ctx->proc, strerror(-rc));
			return rc;
		}
	}

	return rc;
}

/* Parse the user-provided processor affinities. We assume the user only
 * provided online processors.
 */
int parse_proc_affinities(char *cpu_list, int nprocs, int **affinities)
{
	long range_start;
	long num;
	bool hyphen = false;
	int *aff = NULL;
	int aff_len = 0;
	long i;

	aff = malloc(nprocs * sizeof(*aff));
	if (!aff)
		err(1, "Failed to allocate affinities list");

	do {
		/* Catch invalid patterns */
		if (cpu_list[0] == ',' || cpu_list[0] == '-')
			goto error;

		errno = 0;
		num = strtoul(cpu_list, &cpu_list, 10);
		if (errno != 0)
			goto error;

		if (cpu_list[0] == '\0' || cpu_list[0] == ',') {
			/* End of number, range, or list */
			if (!hyphen)
				range_start = num;
			if (range_start > num)
				goto error;
			if (num >= nprocs)
				goto error;
			for (i = range_start; i <= num; i++) {
				if (aff_len == nprocs)
					goto error;
				aff[aff_len++] = i;
			}
			hyphen = false;
			if (cpu_list[0] != '\0')
				cpu_list++;

		} else if (cpu_list[0] == '-') {
			if (hyphen)
				goto error;
			hyphen = true;
			range_start = num;
			cpu_list++;
		} else {
			goto error;
		}
	} while (cpu_list[0] != '\0');

	if (aff_len) {
		*affinities = realloc(aff, aff_len * sizeof(*aff));
		if (!*affinities)
			err(1, "Failed to reallocate affinities list");
	} else {
		free(aff);
	}
	return aff_len;
error:
	free(aff);
	return 0;
}

int compare_ints(const void *a, const void *b)
{
	int arg1 = *(const int *)a;
	int arg2 = *(const int *)b;

	if (arg1 < arg2)
		return -1;
	if (arg2 < arg1)
		return 1;
	return 0;
}

int maybe_read_int_from_file(const char *path_fmt, int path_arg, int *output,
			     bool allow_zero_matches)
{
	int rc = 0;
	char path[MAX_LEN];
	FILE *fp;

	snprintf(path, MAX_LEN, path_fmt, path_arg);
	errno = 0;
	fp = fopen(path, "r");
	if (!fp) {
		rc = -errno;
		fprintf(stderr, "fopen('%s'): %s\n", path, strerror(-rc));
		return rc;
	}
	if (fscanf(fp, "%d", output) != 1) {
		rc = -errno;
		if (rc) {
			fprintf(stderr, "fscanf() failed for %s: %s\n", path,
				strerror(-rc));
		} else if (!allow_zero_matches) {
			fprintf(stderr,
				"fscanf() failed to read an int from %s\n",
				path);
			rc = -EINVAL;
		}
	}
	fclose(fp);
	return rc;
}

int read_int_from_file(const char *path_fmt, int path_arg, int *output)
{
	return maybe_read_int_from_file(path_fmt, path_arg, output, false);
}

/* Attempt to intelligently pick core affinities for traffic-generating
 * processes based on CXI device IDs and their NUMA nodes.
 * We can improve performance by assigning core affinities with these
 * three goals:
 * 1. The processes all run on the socket that the NIC belongs to.
 * 2. The processes are spread across all NUMA nodes of that socket.
 * 3. The processes are offset based on device number in case multiple
 *    devices are being tested at once.
 *
 * Example: Dual-socket processor; 4 NUMA nodes to each socket; 4 cores
 * to each NUMA node; 4 NICs, 2 per socket.
 * cxi0:  0,  4,  8, 12,  2,  6, 10, 14
 * cxi1:  1,  5,  9, 13,  3,  7, 11, 15
 * cxi2: 16, 20, 24, 28, 18, 22, 26, 30
 * cxi3: 17, 21, 25, 29, 19, 23, 27, 31
 */
int get_proc_affinities(uint32_t dev_id, int nprocs, int **affinities,
			char *aff_str)
{
	int rc;
	int dev_numa;
	int first_cpu;
	int phys_pkg_id;
	int phys_pkg_id_cmp;
	int i;
	int node;
	int *numa_arr = NULL;
	int numa_arr_len = 32;
	int numa_cnt = 0;
	struct bitmask *cpumask = NULL;
	int cpumask_len;
	int cpu;
	int **numa_cpu_lists = NULL;
	int *numa_cpu_cnts = NULL;
	int cpu_cnt;
	DIR *dp;
	struct dirent *dir;
	int num_competing_devs = 0;
	int competing_dev;
	int *aff = NULL;
	int aff_len = 0;
	int cpu_offset = 0;
	bool cpu_assigned;
	int first;
	int last;
	bool first_write = true;
	int writes = 0;
	int write_offset = 0;

	/* Determine our device's NUMA node */
	rc = read_int_from_file(DEV_NUMA_FILE, (int)dev_id, &dev_numa);
	if (rc)
		goto done;

	/* Get first CPU from our device's NUMA node */
	rc = read_int_from_file(NUMA_CPULIST_FILE, dev_numa, &first_cpu);
	if (rc)
		goto done;

	/* Get package ID of first CPU */
	rc = read_int_from_file(CPU_PHYS_PKG_FILE, first_cpu, &phys_pkg_id);
	if (rc)
		goto done;

	/* Determine which NUMA nodes share the same socket by comparing
	 * physical_package_ids of the first CPU from each.
	 */
	numa_arr = malloc(numa_arr_len * sizeof(*numa_arr));
	if (!numa_arr)
		err(1, "Failed to allocate NUMA node array");
	errno = 0;
	dp = opendir(NUMA_DIR);
	if (!dp) {
		fprintf(stderr, "opendir('%s'): %s\n", NUMA_DIR,
			strerror(errno));
		goto done;
	}
	while ((dir = readdir(dp))) {
		rc = sscanf(dir->d_name, "node%d", &node);
		if (rc != 1)
			continue;
		first_cpu = -1;
		rc = maybe_read_int_from_file(NUMA_CPULIST_FILE, node,
					      &first_cpu, true);
		if (rc)
			goto done;
		/* Ignore NUMA nodes with no CPUs */
		if (first_cpu == -1)
			continue;
		rc = read_int_from_file(CPU_PHYS_PKG_FILE, first_cpu,
					&phys_pkg_id_cmp);
		if (rc)
			goto done;
		if (phys_pkg_id_cmp == phys_pkg_id) {
			if (numa_cnt == numa_arr_len) {
				numa_arr_len *= 2;
				numa_arr = realloc(numa_arr,
						   numa_arr_len *
							   sizeof(*numa_arr));
			}
			numa_arr[numa_cnt++] = node;
		}
	}
	numa_arr = realloc(numa_arr, numa_cnt * sizeof(*numa_arr));
	if (!numa_arr)
		err(1, "Failed to reallocate NUMA node array");

	/* Get CPU list for each NUMA node */
	numa_cpu_lists = calloc(numa_cnt, (sizeof(*numa_cpu_lists)));
	numa_cpu_cnts = malloc(numa_cnt * (sizeof(*numa_cpu_cnts)));
	cpumask = numa_allocate_cpumask();
	cpumask_len = cpumask->size;
	if (!numa_cpu_lists || !numa_cpu_cnts)
		err(1, "Failed to allocate NUMA core arrays");
	for (i = 0; i < numa_cnt; i++) {
		numa_cpu_lists[i] =
			malloc(cpumask_len * sizeof(*numa_cpu_lists[i]));
		if (!numa_cpu_lists[i])
			err(1, "Failed to allocate NUMA core array");
		cpu_cnt = 0;
		errno = 0;
		rc = numa_node_to_cpus(numa_arr[i], cpumask);
		if (rc) {
			fprintf(stderr, "numa_node_to_cpus(%d): %s\n",
				numa_arr[i], strerror(errno));
			goto done;
		}

		for (cpu = 0; cpu < cpumask_len; cpu++) {
			if (!numa_bitmask_isbitset(cpumask, cpu))
				continue;
			/* Exclude threads */
			rc = read_int_from_file(CPU_THREADS_FILE, cpu,
						&first_cpu);
			if (rc)
				goto done;
			if (cpu == first_cpu)
				numa_cpu_lists[i][cpu_cnt++] = cpu;
		}
		numa_cpu_lists[i] =
			realloc(numa_cpu_lists[i],
				cpu_cnt * sizeof(*numa_cpu_lists[i]));
		if (!numa_cpu_lists[i])
			err(1, "Failed to reallocate NUMA core array");
		numa_cpu_cnts[i] = cpu_cnt;
	}

	/* Determine how many CXI devices are associated with NUMA nodes from
	 * the same socket. We want to assign cores that don't overlap in case
	 * this tool is being executed for multiple devices at the same time.
	 */
	errno = 0;
	dp = opendir(DEV_DIR);
	if (!dp) {
		fprintf(stderr, "opendir('%s'): %s\n", DEV_DIR,
			strerror(errno));
		goto done;
	}
	while ((dir = readdir(dp))) {
		rc = sscanf(dir->d_name, "cxi%d", &competing_dev);
		if (rc != 1)
			continue;
		rc = read_int_from_file(DEV_NUMA_FILE, competing_dev,
					&dev_numa);
		if (rc) {
			closedir(dp);
			goto done;
		}
		for (i = 0; i < numa_cnt; i++) {
			if (numa_arr[i] == dev_numa) {
				/* Offset assigns devs different cores */
				if (competing_dev == (int)dev_id)
					cpu_offset = num_competing_devs;
				num_competing_devs++;
				break;
			}
		}
	}
	closedir(dp);

	/* Assign affinities by repeatedly looping through our socket's NUMA
	 * nodes, increasing the core offset each loop.
	 */
	aff = malloc(nprocs * sizeof(*aff));
	if (!aff)
		err(1, "Failed to allocate core affinity array");
	cpu_assigned = true;
	while (aff_len < nprocs && cpu_assigned) {
		cpu_assigned = false;
		for (i = 0; i < numa_cnt && aff_len < nprocs; i++) {
			if (cpu_offset < numa_cpu_cnts[i]) {
				aff[aff_len++] = numa_cpu_lists[i][cpu_offset];
				cpu_assigned = true;
			}
		}
		cpu_offset += num_competing_devs;
	}

	/* Sort for the sake of building a concise header string */
	qsort(aff, aff_len, sizeof(*aff), compare_ints);

	/* Build cpu list string for header */
	first = -1;
	last = -1;
	for (i = 0; i < aff_len; i++) {
		if (i == (aff_len - 1))
			writes = 1;
		if (first == -1) {
			first = aff[i];
			last = first;
		} else if (aff[i] == (last + 1)) {
			last = aff[i];
		} else {
			writes++;
		}
		while (writes > 0 && write_offset < MAX_AFF_STR_LEN) {
			if (last == first)
				write_offset +=
					snprintf(&aff_str[write_offset],
						 MAX_AFF_STR_LEN - write_offset,
						 first_write ? "%d" : ",%d",
						 first);
			else
				write_offset += snprintf(
					&aff_str[write_offset],
					MAX_AFF_STR_LEN - write_offset,
					first_write ? "%d-%d" : ",%d-%d", first,
					last);
			first_write = false;
			writes--;
			first = aff[i];
			last = aff[i];
		}
	}

done:
	if (aff_len) {
		*affinities = realloc(aff, aff_len * sizeof(*aff));
		if (!*affinities)
			err(1, "Failed to reallocate affinities array");
	} else {
		if (aff)
			free(aff);
	}
	if (numa_arr)
		free(numa_arr);
	if (numa_cpu_lists) {
		for (i = 0; i < numa_cnt; i++) {
			if (numa_cpu_lists[i])
				free(numa_cpu_lists[i]);
		}
		free(numa_cpu_lists);
	}
	if (numa_cpu_cnts)
		free(numa_cpu_cnts);
	if (cpumask)
		numa_free_cpumask(cpumask);

	return aff_len;
}

void usage(void)
{
	printf("Monitor Cassini NIC temperature and power consumption while stressing\n");
	printf("the chip with RDMA writes.\n");
	printf("Requirements:\n");
	printf("  1. The NIC must be able to initiate writes to itself, either by being\n");
	printf("  configured in internal loopback mode, or by having a link partner that is\n");
	printf("  configured to allow routing packets back to their source.\n");
	printf("  2. When configured in internal loopback mode, the --no-hrp option must be\n");
	printf("  used.\n");
	printf("  3. When testing a dual-NIC card, the diagnostic should be run for each NIC\n");
	printf("  concurrently or the power target will not be reached.\n");
	printf("\n");
	printf("Options:\n");
	printf("  -d, --device=DEV       Device name (default: \"cxi0\")\n");
	printf("  -v, --svc-id=SVC_ID    Service ID (default: 1)\n");
	printf("  -t, --tx-gpu=GPU       GPU index for allocating TX buf (default: no GPU)\n");
	printf("  -r, --rx-gpu=GPU       GPU index for allocating RX buf (default: no GPU)\n");
	printf("  -g, --gpu-type=TYPE    GPU type (AMD or NVIDIA or INTEL) (default type determined\n");
	printf("                         by discovered GPU files on system)\n");
	printf("  -P, --procs=PROCS      Number of write-generating processes\n");
	printf("                         (default: 1/4 of available processors)\n");
	printf("  -c, --cpu-list=LIST    Processors to use when assigning affinities to\n");
	printf("                         write-generating processes (default assigned based\n");
	printf("                         on device number and socket count)\n");
	printf("  -D, --duration=SEC     Run for the specified number of seconds (default: %u)\n",
	       DURATION_DFLT);
	printf("  -i, --interval=INT     Interval in seconds to check and print sensor\n");
	printf("                         readings (default: %u)\n",
	       INTERVAL_DFLT);
	printf("  -s, --size=SIZE        RDMA Write size to use (default: %u)\n",
	       MSG_SIZE_DFLT);
	printf("                         The maximum size is %u\n", MAX_SIZE);
	printf("  -l, --list-size=LSIZE  Number of writes per iteration, all pushed to\n");
	printf("                         the Tx CQ prior to initiating xfer (default: %u)\n",
	       LIST_SIZE_DFLT);
	printf("      --no-hrp           Do not use High Rate Puts for sizes <= %u bytes\n",
	       PORTALS_MTU);
	printf("      --no-idc           Do not use Immediate Data Cmds for high rate put\n");
	printf("                         sizes <= %u bytes and matching put sizes <= %u bytes\n",
	       MAX_IDC_RESTRICTED_SIZE, MAX_IDC_UNRESTRICTED_SIZE);
	printf("      --no-ple           Append a single use-once list entry for every write\n");
	printf("                         Note: Combining this option with large LSIZE and PROCS\n");
	printf("                         values may results in NO_SPACE errors\n");
	printf("  -h, --help             Print this help text and exit\n");
	printf("  -V, --version          Print the version and exit\n");
}

int main(int argc, char **argv)
{
	int rc;
	int run_rc = 0;
	int c;
	char *endptr;
	int option_index;
	struct heatsink_opts opts = { 0 };
	struct heatsink_ctx ctx = { 0 };
	uint32_t dev_id = 0;
	uint32_t svc_id = CXI_DEFAULT_SVC_ID;
	cpu_set_t mask;
	int p;
	char sem_p_name[MAX_SEM_NAME];
	char sem_c_name[MAX_SEM_NAME];
	int status;
	int nprocs;
	int *affinities = NULL;
	int aff_len = 0;
	char aff_str[MAX_AFF_STR_LEN];
	struct sigaction action = {};
	int count;
	long tmp;

	opts.msg_size = MSG_SIZE_DFLT;
	opts.list_size = LIST_SIZE_DFLT;
	opts.interval = INTERVAL_DFLT;
	opts.duration = DURATION_DFLT;
	opts.tx_gpu = -1;
	opts.rx_gpu = -1;

	/* Set default GPU type based on discovered GPU files. */
#if HAVE_HIP_SUPPORT
	opts.gpu_type = AMD;
#elif defined(HAVE_CUDA_SUPPORT)
	opts.gpu_type = NVIDIA;
#elif defined(HAVE_ZE_SUPPORT)
	opts.gpu_type = INTEL;
#else
	opts.gpu_type = -1;
#endif

	nprocs = get_nprocs();

	struct option long_options[] = {
		{ "no-hrp", no_argument, &opts.no_hrp, 1 },
		{ "no-idc", no_argument, &opts.no_idc, 1 },
		{ "no-ple", no_argument, &opts.no_ple, 1 },
		{ "procs", required_argument, NULL, 'P' },
		{ "device", required_argument, NULL, 'd' },
		{ "tx-gpu", required_argument, NULL, 't' },
		{ "rx-gpu", required_argument, NULL, 'r' },
		{ "gpu-type", required_argument, NULL, 'g' },
		{ "cpu-list", required_argument, NULL, 'c' },
		{ "duration", required_argument, NULL, 'D' },
		{ "interval", required_argument, NULL, 'i' },
		{ "size", required_argument, NULL, 's' },
		{ "list-size", required_argument, NULL, 'l' },
		{ "help", no_argument, NULL, 'h' },
		{ "version", no_argument, NULL, 'V' },
		{ NULL, 0, NULL, 0 }
	};

	while (1) {
		c = getopt_long(argc, argv, "P:v:d:t:r:g:c:D:i:s:l:hV",
				long_options, &option_index);
		if (c == -1)
			break;

		switch (c) {
		case 0:
			break;
		case 'P':
			errno = 0;
			endptr = NULL;
			opts.procs = strtoul(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 || opts.procs == 0)
				errx(1, "Invalid process count: %s", optarg);
			break;
		case 'd':
			if (strlen(optarg) < 4 || strncmp(optarg, "cxi", 3))
				errx(1, "Invalid device name: %s", optarg);
			optarg += 3;

			errno = 0;
			endptr = NULL;
			dev_id = strtoul(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0)
				errx(1, "Invalid device name: cxi%s", optarg);
			break;
		case 'v':
			errno = 0;
			endptr = NULL;
			tmp = strtol(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0 || endptr == optarg ||
			    tmp < 1 || tmp > INT_MAX)
				errx(1, "Invalid svc_id: %s", optarg);
			svc_id = tmp;
			break;
		case 't':
			errno = 0;
			endptr = NULL;
			opts.tx_gpu = strtol(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 || opts.tx_gpu < 0)
				errx(1, "Invalid src gpu device: %s", optarg);
			ctx.use_tx_gpu = true;
			break;
		case 'r':
			errno = 0;
			endptr = NULL;
			opts.rx_gpu = strtol(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 || opts.rx_gpu < 0)
				errx(1, "Invalid dest gpu device: %s", optarg);
			ctx.use_rx_gpu = true;
			break;
		case 'g':
			opts.gpu_type = get_gpu_type(optarg);
			if (opts.gpu_type < 0)
				errx(1,
				     "Invalid gpu type: %s. Must be AMD or NVIDIA",
				     optarg);
			break;
		case 'c':
			aff_len = parse_proc_affinities(optarg, nprocs,
							&affinities);
			if (!aff_len)
				errx(1, "Invalid CPU list: %s", optarg);
			snprintf(aff_str, MAX_AFF_STR_LEN, "%s", optarg);
			break;
		case 'D':
			errno = 0;
			endptr = NULL;
			opts.duration = strtoul(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 || opts.duration == 0)
				errx(1, "Invalid duration: %s", optarg);
			break;
		case 'i':
			errno = 0;
			endptr = NULL;
			opts.interval = strtoul(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 || opts.interval == 0)
				errx(1, "Invalid measurement intreval: %s",
				     optarg);
			break;
		case 's':
			errno = 0;
			endptr = NULL;
			opts.msg_size = strtoul(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 ||
			    opts.msg_size > MAX_SIZE)
				errx(1, "Invalid size: %s", optarg);
			break;
		case 'l':
			errno = 0;
			endptr = NULL;
			opts.list_size = strtoul(optarg, &endptr, 0);
			if (errno != 0 || *endptr != 0 || opts.list_size == 0 ||
			    (opts.list_size > ((MAX_CQ_DEPTH / 4) - 1)))
				errx(1, "Invalid Tx CQ depth: %s", optarg);
			break;
		case 'h':
			usage();
			exit(0);
		case 'V':
			printf("%s version: %s\n", name, version);
			exit(0);
		case '?':
		default:
			usage();
			exit(1);
			break;
		}
	}
	if (optind < argc)
		errx(1, "Unexpected argument: %s", argv[optind]);
	if (opts.interval > opts.duration)
		opts.interval = opts.duration;

	if ((ctx.use_tx_gpu || ctx.use_rx_gpu) && opts.gpu_type < 0)
		errx(1, "Invalid GPU type or unable to find GPU libraries");

	rc = get_board_type(dev_id);
	if (rc < 0)
		errx(1, "Unable to determine board type: %s", strerror(-rc));
	else if (rc == 0)
		errx(1, "Unable to determine board type");
	ctx.board_type = rc;

	if (!aff_len) {
		if (!opts.procs)
			opts.procs = nprocs / 4;
		aff_len = get_proc_affinities(dev_id, opts.procs, &affinities,
					      aff_str);
		if (!aff_len) {
			fprintf(stderr, "Unable to pick cores to use.\n");
			fprintf(stderr, "Use '-c' to pick them manually\n");
			return 1;
		}
	}
	opts.procs = aff_len; /* No sharing cores */
	opts.procs++; /* proc 0 doesn't generate traffic */

	/* Let parent process stop the children */
	action.sa_handler = sigusr1_handler;
	sigemptyset(&action.sa_mask);
	sigaction(SIGUSR1, &action, NULL);

	/* Some things are mutually exclusive */
	ctx.use_hrp = !opts.no_hrp && opts.msg_size <= PORTALS_MTU;
	ctx.use_idc =
		!opts.no_idc && !ctx.use_tx_gpu &&
		(opts.msg_size <= MAX_IDC_UNRESTRICTED_SIZE ||
		 (ctx.use_hrp && opts.msg_size <= MAX_IDC_RESTRICTED_SIZE));

	/* Print out the configuration */
	if (ctx.board_type == BRD_BRAZOS) {
		snprintf(results_header, MAX_HDR_LEN,
			 "%*s  %*s  %*s  %*s  %*s  %*s  %*s  %*s", TIME_W,
			 "Time[s]", RATE_W, "Rate[GB/s]", VDD_W, "VDD[W]",
			 AVDD_W, "AVDD[W]", QSFP_P_W, "QSFP[W]", TEMPS_W,
			 "ASIC_0[°C]", QSFP_VR_W, "QSFP_VR[°C]", QSFP_INT_W,
			 "QSFP_INT[°C]");
		/* -3 for degree symbols */
		header_len = strlen(results_header) - 3;
	} else if (ctx.board_type == BRD_SAWTOOTH) {
		snprintf(results_header, MAX_HDR_LEN,
			 "%*s  %*s  %*s  %*s  %*s  %*s", TIME_W, "Time[s]",
			 RATE_W, "Rate[GB/s]", VDD_W, "VDD[W]", AVDD_W,
			 "AVDD[W]", TEMPS_W, "ASIC_0[°C]", TEMPS_W,
			 "ASIC_1[°C]");
		/* -2 for degree symbols */
		header_len = strlen(results_header) - 2;
	} else if (ctx.board_type == BRD_KENNEBEC) {
		snprintf(results_header, MAX_HDR_LEN,
			 "%*s  %*s  %*s  %*s  %*s  %*s  %*s", TIME_W, "Time[s]",
			 RATE_W, "Rate[GB/s]", VDD_W, "VDD[W]", AVDD_W,
			 "AVDD[W]", TRVDD_W, "TRVDD[W]", TEMPS_W, "ASIC_0[°C]",
			 QSFP_INT_W, "QSFP_INT[°C]");
		/* -2 for degree symbols */
		header_len = strlen(results_header) - 2;
	} else if (ctx.board_type == BRD_WASHINGTON) {
		snprintf(results_header, MAX_HDR_LEN,
			 "%*s  %*s  %*s  %*s  %*s  %*s  %*s", TIME_W, "Time[s]",
			 RATE_W, "Rate[GB/s]", VDD_W, "VDD[W]", AVDD_W,
			 "AVDD[W]", TRVDD_W, "TRVDD[W]", TEMPS_W, "ASIC_0[°C]",
			 TEMPS_W, "ASIC_1[°C]");
		/* -2 for degree symbols */
		header_len = strlen(results_header) - 2;
	} else if (ctx.board_type == BRD_SOUHEGAN) {
		snprintf(results_header, MAX_HDR_LEN,
			 "%*s  %*s  %*s  %*s  %*s  %*s  %*s %*s",
			 TIME_W, "Time[s]", RATE_W, "Rate[GB/s]",
			 VDD_W, "VDD[W]", AVDD_W, "AVDD[W]",
			 TRVDD_W, "TRVDD[W]", TEMPS_W, "ASIC_0[°C]", TEMPS_W,
			 "ASIC_1[°C]", OSFP_INT_W, "OSFP_INT[°C]);");
		/* -3 for degree symbols */
		header_len = strlen(results_header) - 3;
	}
	print_separator(header_len);
	printf("    CXI Heatsink Test\n");
	printf("Device          : cxi%u\n", dev_id);
	printf("Service ID      : %u\n", svc_id);
	if (ctx.use_tx_gpu)
		printf("TX Mem          : GPU %d\n", opts.tx_gpu);
	else
		printf("TX Mem          : System\n");
	if (ctx.use_rx_gpu)
		printf("RX Mem          : GPU %d\n", opts.rx_gpu);
	else
		printf("RX Mem          : System\n");
	if (ctx.use_tx_gpu || ctx.use_rx_gpu)
		printf("GPU Type        : %s\n", gpu_names[opts.gpu_type]);
	printf("Duration        : %lu seconds\n", opts.duration);
	printf("Sensor Interval : %lu seconds\n", opts.interval);
	printf("TX/RX Processes : %u\n", opts.procs - 1);
	printf("Processor List  : %s\n", aff_len ? aff_str : "0");
	printf("RDMA Write Size : %lu\n", opts.msg_size);
	printf("List Size       : %u\n", opts.list_size);
	if (opts.no_hrp)
		printf("HRP             : Disabled\n");
	else if (ctx.use_hrp)
		printf("HRP             : Enabled\n");
	else
		printf("HRP             : Disabled - Not Applicable\n");
	if (opts.no_idc)
		printf("IDC             : Disabled\n");
	else if (ctx.use_idc)
		printf("IDC             : Enabled\n");
	else
		printf("IDC             : Disabled - Not Applicable\n");
	printf("Persistent LEs  : %s\n", opts.no_ple ? "Disabled" : "Enabled");

	/* Fork */
	putenv("CXI_FORK_SAFE=1");
	CPU_ZERO(&mask);
	snprintf(sem_p_name, MAX_SEM_NAME, "cxi_heatsink_p_cxi%u", dev_id);
	snprintf(sem_c_name, MAX_SEM_NAME, "cxi_heatsink_c_cxi%u", dev_id);
	ctx.sem_parent = sem_open(sem_p_name, O_CREAT | O_EXCL, 0660, 0);
	if (ctx.sem_parent == SEM_FAILED) {
		rc = -errno;
		fprintf(stderr, "sem_open(%s): %s\n", sem_p_name,
			strerror(-rc));
		goto cleanup;
	}
	ctx.sem_children = sem_open(sem_c_name, O_CREAT | O_EXCL, 0660, 0);
	if (ctx.sem_children == SEM_FAILED) {
		rc = -errno;
		fprintf(stderr, "sem_open(%s): %s\n", sem_c_name,
			strerror(-rc));
		goto cleanup;
	}
	fflush(stdout);
	for (p = 0; p < (opts.procs - 1); p++) {
		rc = fork();
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "fork: %s\n", strerror(-rc));
			goto cleanup;
		}
		if (rc)
			continue;
		else
			break;
	}

	/* Set core affinities */
	ctx.proc = opts.procs - p - 1; /* Set parent to process index 0 */
	if (!ctx.proc || !aff_len)
		CPU_SET(0, &mask);
	else
		CPU_SET(affinities[(ctx.proc - 1) % aff_len], &mask);
	rc = sched_setaffinity(0, sizeof(mask), &mask);
	if (rc) {
		rc = -errno;
		fprintf(stderr, "sched_setaffinity: %s\n", strerror(-rc));
		goto cleanup;
	}

	/* Initialize GPU library if we are using GPU memory */
	if (ctx.proc && (ctx.use_tx_gpu || ctx.use_rx_gpu)) {
		rc = gpu_lib_init(opts.gpu_type);
		if (rc < 0) {
			fprintf(stderr, "Failed to init GPU lib: %s\n",
				strerror(-rc));
			goto cleanup;
		}
		if (ctx.proc == 1) {
			count = get_gpu_device_count();
			printf("Found %d GPU(s)\n", count);
		}
	}

	/* Allocate CXI context */
	rc = ctx_alloc(&ctx.cxi, dev_id, svc_id);
	if (rc < 0) {
		fprintf(stderr, "Failed to init CXI context: %s\n",
			strerror(-rc));
		goto cleanup;
	}

	if (!ctx.proc) {
		/* Map CSRs */
		printf("Local Address   : NIC 0x%x VNI %u\n",
		       ctx.cxi.loc_addr.nic, ctx.cxi.vni);
		printf("Board Type      : %s-NIC\n",
		       ((ctx.board_type == BRD_BRAZOS ||
			 ctx.board_type == BRD_KENNEBEC) ?
				"Single" :
				"Dual"));
		rc = cxil_map_csr(ctx.cxi.dev);
		if (rc) {
			fprintf(stderr, "Failed to map CSRs: %s\n",
				strerror(-rc));
			/* Probably not root, sync then quit */
			kill(0, SIGUSR1);
			run_finished = 1;
		}

		/* Open files for reading sensors */
		rc = open_sensor_files(&ctx, dev_id);
		if (rc) {
			fprintf(stderr,
				"Failed to open sensor reading files: %s\n",
				strerror(-rc));
			goto cleanup;
		}
	}

	/* Synchronize */
	rc = synchronize(&ctx, &opts);
	if (rc < 0) {
		if (!ctx.proc) {
			if (rc != -EINTR)
				fprintf(stderr,
					"Failed to sync processes before run\n");
			else
				rc = -ECANCELED;
		}
		goto cleanup;
	}
	if (run_finished)
		goto wait_cleanup;

	/* Allocate remaining CXI resources */
	if (ctx.proc) {
		rc = heatsink_alloc(&ctx, &opts);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate CXI resources: %s\n",
				strerror(-rc));
			goto cleanup;
		}
	}

	/* Synchronize */
	rc = synchronize(&ctx, &opts);
	if (rc < 0) {
		if (!ctx.proc) {
			if (rc != -EINTR)
				fprintf(stderr,
					"Failed to sync processes before run\n");
			else
				rc = -ECANCELED;
		}
		goto cleanup;
	}

	if (!ctx.proc) {
		print_separator(header_len);
		printf("%s\n", results_header);
	}

	/* Run */
	run_rc = run_heatsink_check(&ctx, &opts);

	/* Synchronize */
	if (ctx.proc && getppid() == 1)
		goto cleanup;
	rc = synchronize(&ctx, &opts);
	if (rc < 0) {
		if (!ctx.proc) {
			if (rc != -EINTR)
				fprintf(stderr,
					"Failed to sync processes after run\n");
			else
				rc = -ECANCELED;
		}
		goto cleanup;
	}

wait_cleanup:
	/* Wait for children to exit */
	if (!ctx.proc)
		for (p = 1; p < opts.procs; p++)
			wait(&status);

cleanup:
	if (ctx.cxi.dev)
		ctx_destroy(&ctx.cxi);
	if (ctx.proc && (ctx.use_tx_gpu || ctx.use_rx_gpu))
		gpu_lib_fini(opts.gpu_type);
	if (ctx.sem_children) {
		sem_unlink(sem_c_name);
		sem_close(ctx.sem_children);
	}
	if (ctx.sem_parent) {
		sem_unlink(sem_p_name);
		sem_close(ctx.sem_parent);
	}
	if (!ctx.proc)
		close_sensor_files(&ctx);
	if (affinities)
		free(affinities);

	if (rc || run_rc < 0)
		return 1;
	else if (run_rc > 0)
		return 2; /* No errors, but failed sensor targets */
	else
		return 0;
}
