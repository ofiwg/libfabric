/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

#define _GNU_SOURCE
#define FUSE_USE_VERSION 30

#include <pthread.h>
#include <fuse.h>
#include <fuse/fuse_common.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include "rh.h"

char *rh_stats_dir = "/run/cxi";
static struct retry_handler *fs_rh;

/* Callback to get file/directory attributes
 * Define the permissions for reading any directory and update link count
 * Link count of a directory is 2 + the number of directories immediately contained by it
 */
static int stats_getattr(const char *path, struct stat *st)
{
	st->st_uid = getuid();
	st->st_gid = getgid();
	st->st_atime = time(NULL);
	st->st_mtime = time(NULL);

	/* Root of stats FS */
	if (!strcmp(path, "/")) {
		st->st_mode = S_IFDIR | 0755;
		st->st_nlink = 3;
	/* Directory of config stats */
	} else if (!strcmp(path, "/config")) {
		st->st_mode = S_IFDIR | 0755;
		st->st_nlink = 2;
	/* Regular Files */
	} else {
		st->st_mode = S_IFREG | 0644;
		st->st_nlink = 1;
		st->st_size = 1024;
	}

	return 0;
}

/* Callback to get directory contents - what does ls show
 * Add both sub directories and files
 */
static int stats_readdir(const char *path, void *buffer,
			 fuse_fill_dir_t filler, off_t offset,
			 struct fuse_file_info *fi)
{
	filler(buffer, ".", NULL, 0);
	filler(buffer, "..", NULL, 0);

	if (!strcmp(path, "/")) {
		filler(buffer, "config", NULL, 0);
		filler(buffer, "spt_timeouts", NULL, 0);
		filler(buffer, "spt_timeouts_u", NULL, 0);
		filler(buffer, "spt_timeouts_o", NULL, 0);
		filler(buffer, "connections_cancelled", NULL, 0);
		filler(buffer, "pkts_cancelled_u", NULL, 0);
		filler(buffer, "pkts_cancelled_o", NULL, 0);
		filler(buffer, "cancel_no_matching_conn", NULL, 0);
		filler(buffer, "cancel_resource_busy", NULL, 0);
		filler(buffer, "cancel_trs_pend_rsp", NULL, 0);
		filler(buffer, "cancel_tct_closed", NULL, 0);
		filler(buffer, "nacks", NULL, 0);
		filler(buffer, "nack_no_target_trs", NULL, 0);
		filler(buffer, "nack_no_target_mst", NULL, 0);
		filler(buffer, "nack_no_target_conn", NULL, 0);
		filler(buffer, "nack_no_matching_conn", NULL, 0);
		filler(buffer, "nack_resource_busy", NULL, 0);
		filler(buffer, "nack_trs_pend_rsp", NULL, 0);
		filler(buffer, "nack_sequence_error", NULL, 0);
		filler(buffer, "sct_timeouts", NULL, 0);
		filler(buffer, "tct_timeouts", NULL, 0);
		filler(buffer, "accel_close_complete", NULL, 0);
		filler(buffer, "ignored_sct_timeouts", NULL, 0);
		filler(buffer, "srb_in_use", NULL, 0);
		filler(buffer, "spt_in_use", NULL, 0);
		filler(buffer, "smt_in_use", NULL, 0);
		filler(buffer, "sct_in_use", NULL, 0);
		filler(buffer, "trs_in_use", NULL, 0);
		filler(buffer, "mst_in_use", NULL, 0);
		filler(buffer, "tct_in_use", NULL, 0);
		filler(buffer, "rh_sct_status_change", NULL, 0);
		filler(buffer, "nid_value", NULL, 0);
		filler(buffer, "nid_tree_count", NULL, 0);
		filler(buffer, "max_nid_tree_count", NULL, 0);
		filler(buffer, "switch_tree_count", NULL, 0);
		filler(buffer, "max_switch_tree_count", NULL, 0);
		filler(buffer, "parked_nids", NULL, 0);
		filler(buffer, "max_parked_nids", NULL, 0);
		filler(buffer, "parked_switches", NULL, 0);
		filler(buffer, "max_parked_switches", NULL, 0);
	} else if (!strcmp(path, "/config")) {
		filler(buffer, "config_file_path", NULL, 0);
		filler(buffer, "max_fabric_packet_age", NULL, 0);
		filler(buffer, "max_spt_retries", NULL, 0);
		filler(buffer, "max_no_matching_conn_retries", NULL, 0);
		filler(buffer, "max_resource_busy_retries", NULL, 0);
		filler(buffer, "max_trs_pend_rsp_retries", NULL, 0);
		filler(buffer, "max_batch_size", NULL, 0);
		filler(buffer, "initial_batch_size", NULL, 0);
		filler(buffer, "backoff_multiplier", NULL, 0);
		filler(buffer, "nack_backoff_start", NULL, 0);
		filler(buffer, "tct_wait_time", NULL, 0);
		filler(buffer, "pause_wait_time", NULL, 0);
		filler(buffer, "cancel_spt_wait_time", NULL, 0);
		filler(buffer, "peer_tct_free_wait_time", NULL, 0);
		filler(buffer, "down_nid_wait_time", NULL, 0);
		filler(buffer, "log_increment", NULL, 0);
	}

	return 0;
}

/* Callback to get essence of a file - what do we display */
static int stats_read(const char *path, char *buffer, size_t size,
		      off_t offset, struct fuse_file_info *fi)
{
	union c_pct_prf_sct_status sct_status;
	union c_pct_prf_spt_status spt_status;
	union c_pct_prf_tct_status tct_status;
	union c_pct_prf_smt_status smt_status;
	union c_pct_prf_srb_status srb_status;
	union c_pct_prf_mst_status mst_status;
	int value;
	int len;

	if (!strcmp(path, "/spt_timeouts"))
		value = fs_rh->stats.event_spt_timeout;
	else if (!strcmp(path, "/spt_timeouts_u"))
		value = fs_rh->stats.event_spt_timeout_u;
	else if (!strcmp(path, "/spt_timeouts_o"))
		value = fs_rh->stats.event_spt_timeout_o;
	else if (!strcmp(path, "/connections_cancelled"))
		value = fs_rh->stats.connections_cancelled;
	else if (!strcmp(path, "/pkts_cancelled_u"))
		value = fs_rh->stats.pkts_cancelled_u;
	else if (!strcmp(path, "/pkts_cancelled_o"))
		value = fs_rh->stats.pkts_cancelled_o;
	else if (!strcmp(path, "/cancel_no_matching_conn"))
		value = fs_rh->stats.cancel_no_matching_conn;
	else if (!strcmp(path, "/cancel_resource_busy"))
		value = fs_rh->stats.cancel_resource_busy;
	else if (!strcmp(path, "/cancel_trs_pend_rsp"))
		value = fs_rh->stats.cancel_trs_pend_rsp;
	else if (!strcmp(path, "/cancel_tct_closed"))
		value = fs_rh->stats.cancel_tct_closed;
	else if (!strcmp(path, "/nacks"))
		value = fs_rh->stats.event_nack;
	else if (!strcmp(path, "/nack_no_target_trs"))
		value = fs_rh->stats.nack_no_target_trs;
	else if (!strcmp(path, "/nack_no_target_mst"))
		value = fs_rh->stats.nack_no_target_mst;
	else if (!strcmp(path, "/nack_no_target_conn"))
		value = fs_rh->stats.nack_no_target_conn;
	else if (!strcmp(path, "/nack_no_matching_conn"))
		value = fs_rh->stats.nack_no_matching_conn;
	else if (!strcmp(path, "/nack_resource_busy"))
		value = fs_rh->stats.nack_resource_busy;
	else if (!strcmp(path, "/nack_trs_pend_rsp"))
		value = fs_rh->stats.nack_trs_pend_rsp;
	else if (!strcmp(path, "/nack_sequence_error"))
		value = fs_rh->stats.nack_sequence_error;
	else if (!strcmp(path, "/sct_timeouts"))
		value = fs_rh->stats.event_sct_timeout;
	else if (!strcmp(path, "/tct_timeouts"))
		value = fs_rh->stats.event_tct_timeout;
	else if (!strcmp(path, "/accel_close_complete"))
		value = fs_rh->stats.event_accel_close_complete;
	else if (!strcmp(path, "/ignored_sct_timeouts"))
		value = fs_rh->stats.ignored_sct_timeouts;
	else if (!strcmp(path, "/rh_sct_status_change"))
		value = fs_rh->stats.rh_sct_status_change;
	else if (!strcmp(path, "/srb_in_use")) {
		cxil_read_csr(fs_rh->dev, C_PCT_PRF_SRB_STATUS,
			      &srb_status, sizeof(srb_status));
		value = srb_status.srb_in_use;
	} else if (!strcmp(path, "/spt_in_use")) {
		cxil_read_csr(fs_rh->dev, C_PCT_PRF_SPT_STATUS,
			      &spt_status, sizeof(spt_status));
		value = spt_status.spt_in_use;
	} else if (!strcmp(path, "/smt_in_use")) {
		cxil_read_csr(fs_rh->dev, C_PCT_PRF_SMT_STATUS,
			      &smt_status, sizeof(smt_status));
		value = smt_status.smt_in_use;
	} else if (!strcmp(path, "/sct_in_use")) {
		cxil_read_csr(fs_rh->dev, C_PCT_PRF_SCT_STATUS,
			      &sct_status, sizeof(sct_status));
		value = sct_status.sct_in_use;
	} else if (!strcmp(path, "/trs_in_use")) {
		if (fs_rh->is_c1) {
			union c1_pct_prf_trs_status trs_status;

			cxil_read_csr(fs_rh->dev, C1_PCT_PRF_TRS_STATUS,
				      &trs_status, sizeof(trs_status));
			value = trs_status.trs_in_use;
		} else {
			union c2_pct_prf_trs_status trs_status;

			cxil_read_csr(fs_rh->dev, C2_PCT_PRF_TRS_STATUS,
				      &trs_status, sizeof(trs_status));
			value = trs_status.trs_in_use;
		}
	} else if (!strcmp(path, "/mst_in_use")) {
		cxil_read_csr(fs_rh->dev, C_PCT_PRF_MST_STATUS,
			      &mst_status, sizeof(mst_status));
		value = mst_status.mst_in_use;
	} else if (!strcmp(path, "/tct_in_use")) {
		cxil_read_csr(fs_rh->dev, C_PCT_PRF_TCT_STATUS,
			      &tct_status, sizeof(tct_status));
		value = tct_status.tct_in_use;
	} else if (!strcmp(path, "/nid_value")) {
		value = fs_rh->dev->info.nid;
	} else if (!strcmp(path, "/reset_counters")) {
		// Showing an empty string
		len = snprintf(buffer, size, "%s\n", "");
		return len < size ? len : size;
	} else if (!strcmp(path, "/config/config_file_path")) {
		len = snprintf(buffer, size, "%s\n", config_file);
		return len < size ? len : size;
	} else if (!strcmp(path, "/config/max_fabric_packet_age")) {
		value = max_fabric_packet_age;
	} else if (!strcmp(path, "/config/max_spt_retries")) {
		value = max_spt_retries;
	} else if (!strcmp(path, "/config/max_no_matching_conn_retries")) {
		value = max_no_matching_conn_retries;
	} else if (!strcmp(path, "/config/max_resource_busy_retries")) {
		value = max_resource_busy_retries;
	} else if (!strcmp(path, "/config/max_trs_pend_rsp_retries")) {
		value = max_trs_pend_rsp_retries;
	} else if (!strcmp(path, "/config/max_batch_size")) {
		value = max_batch_size;
	} else if (!strcmp(path, "/config/initial_batch_size")) {
		value = initial_batch_size;
	} else if (!strcmp(path, "/config/backoff_multiplier")) {
		value = backoff_multiplier;
	} else if (!strcmp(path, "/config/nack_backoff_start")) {
		value = nack_backoff_start;
	} else if (!strcmp(path, "/config/tct_wait_time")) {
		value = tct_wait_time.tv_sec;
	} else if (!strcmp(path, "/config/pause_wait_time")) {
		value = pause_wait_time.tv_sec;
	} else if (!strcmp(path, "/config/cancel_spt_wait_time")) {
		value = cancel_spt_wait_time.tv_sec;
	} else if (!strcmp(path, "/config/peer_tct_free_wait_time")) {
		value = peer_tct_free_wait_time.tv_sec;
	} else if (!strcmp(path, "/config/down_nid_wait_time")) {
		value = down_nid_wait_time.tv_sec;
	} else if (!strcmp(path, "/config/log_increment")) {
		value = fs_rh->log_increment;
	} else if (!strcmp(path, "/nid_tree_count")) {
		value = fs_rh->nid_tree_count;
	} else if (!strcmp(path, "/max_nid_tree_count")) {
		value = fs_rh->stats.max_nid_tree_count;
	} else if (!strcmp(path, "/switch_tree_count")) {
		value = fs_rh->switch_tree_count;
	} else if (!strcmp(path, "/max_switch_tree_count")) {
		value = fs_rh->stats.max_switch_tree_count;
	} else if (!strcmp(path, "/parked_nids")) {
		value = fs_rh->parked_nids;
	} else if (!strcmp(path, "/max_parked_nids")) {
		value = fs_rh->stats.max_parked_nids;
	} else if (!strcmp(path, "/parked_switches")) {
		value = fs_rh->parked_switches;
	} else if (!strcmp(path, "/max_parked_switches")) {
		value = fs_rh->stats.max_parked_switches;
	} else {
		return -1;
	}
	len = snprintf(buffer, size, "%d\n", value);

	return len < size ? len : size;
}

/* Callback to get content of a file - what do we write */
static int stats_write(const char *path, const char *buffer, size_t size,
		      off_t offset, struct fuse_file_info *fi)
{
	int value;

	/* Write 0 to reset_counters file to reset the stat values */
	if (!strcmp(path, "/reset_counters")) {
		if (!strncmp(buffer, "0", size - 1)) {
			memset(&(fs_rh->stats), 0, sizeof(fs_rh->stats));
			fprintf(stderr, "Reset counters success.\n");
		} else {
			fprintf(stderr, "Reset counters failed. Please write 0 to reset.\n");
		}
	} else if (!strcmp(path, "/config/log_increment")) {
		value = atoi(buffer);
		if (value < -7 || value > 7)
			fprintf(stderr, "Invalid log_increment level provided\n");
		else
			fs_rh->log_increment = value;
	} else {
		fprintf(stderr, "Cannot write file: %s\n", path);
		return -ENOENT;
	}
	return size;
}

static int stats_truncate(const char *path, off_t size)
{
	if (!strcmp(path, "/config/log_increment"))
		return 0;
	if (!strcmp(path, "/reset_counters"))
		return 0;

	fprintf(stderr, "Cannot truncate file: %s\n", path);
	return -ENOENT;
}

const struct fuse_operations ops = {
	.getattr	= stats_getattr,
	.readdir	= stats_readdir,
	.read		= stats_read,
	.write		= stats_write,
	.truncate	= stats_truncate,
};

void *stats_fs(void *arg)
{
	fuse_loop(fs_rh->stats_fuse);

	return NULL;
}

int stats_init(struct retry_handler *rh)
{
	int ret;
	char *tmp;

	/* Set global RH */
	fs_rh = rh;

	/* Append dev_id to stats path*/
	ret = asprintf(&tmp, "%s/cxi%d", rh_stats_dir, rh->dev_id);
	if (ret == -1) {
		rh_printf(rh, LOG_ERR,
			  "Updating mount path failed: %d\n", ret);
		goto err;
	}

	if (rh->cfg_stats)
		free(rh_stats_dir);

	rh_stats_dir = tmp;

	struct fuse_args args = FUSE_ARGS_INIT(0, NULL);

	/* First arg requires a name for the fs */
	ret = fuse_opt_add_arg(&args, "rh_stats");
	if (ret) {
		rh_printf(rh, LOG_ERR, "Couldn't add fuse fs name opt\n");
		goto err;
	}

	/* Allow non-root users to access stats FS */
	ret = fuse_opt_add_arg(&args, "-oallow_other");
	if (ret) {
		rh_printf(rh, LOG_ERR,
			  "Couldn't add fuse allow_other opt\n");
		goto free_args;
	}

	rh->stats_chan = fuse_mount(rh_stats_dir, &args);
	if (rh->stats_chan == NULL) {
		rh_printf(rh, LOG_ERR,
			  "Mount to %s failed\n", rh_stats_dir);
		ret = -1;
		goto free_args;
	}

	rh->stats_fuse = fuse_new(rh->stats_chan, NULL, &ops, sizeof(ops), rh);
	if (rh->stats_fuse == NULL) {
		ret = -1;
		goto free_chan;
	}

	ret = pthread_create(&rh->stats_thread, NULL, stats_fs, (void *)rh);
	if (ret)
		goto free_fuse;

	fuse_opt_free_args(&args);

	return 0;

free_fuse:
	fuse_destroy(rh->stats_fuse);
free_chan:
	fuse_unmount(rh_stats_dir, rh->stats_chan);
free_args:
	fuse_opt_free_args(&args);
err:
	return ret;
}

void stats_fini(void)
{
	void *thread_rc;
	int ret;

	/* interrupt fuse_loop() */
	fuse_unmount(rh_stats_dir, fs_rh->stats_chan);

	ret = pthread_join(fs_rh->stats_thread, &thread_rc);
	if (ret)
		perror("stats fini pthread_join failed");

	fuse_destroy(fs_rh->stats_fuse);

	free(rh_stats_dir);
}
