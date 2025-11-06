/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2020 Hewlett Packard Enterprise Development LP
 */

/* Configuration parser for the Cassini retry handler. */

#include <math.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>

#include "rh.h"

/* Read the configuration file and change the default settings */
int read_config(const char *filename, struct retry_handler *rh)
{
	unsigned int rc;
	config_setting_t *settings;
	int intconf;
	double dconf;
	double y;
	config_t cfg;
	const char *sconf;
	DIR *dir;
	int cfg_options;
	int array_size;
	int i;

	config_init(&cfg);
	cfg_options = config_get_options(&cfg);
	cfg_options |= CONFIG_OPTION_AUTOCONVERT;
	config_set_options(&cfg, cfg_options);

	if (!config_read_file(&cfg, filename)) {
		if (config_error_line(&cfg)) {
			rh_printf(rh, LOG_ERR, "config error %s:%d: %s\n",
				  config_error_file(&cfg), config_error_line(&cfg),
				  config_error_text(&cfg));

			rc = 1;
			goto out;
		}

		/* File not found. Use defaults. */
		rh_printf(rh, LOG_ERR, "Using default config. Couldn't read config file '%s': %s\n",
			  filename, config_error_text(&cfg));
		rc = 0;
		goto out;
	} else {
		rh_printf(rh, LOG_NOTICE, "using config file \"%s\"\n", filename);
	}

	settings = config_lookup(&cfg, "max_spt_retries");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf > MAX_SPT_RETRIES_LIMIT) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_spt_retries\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		max_spt_retries = intconf;
	}

	settings = config_lookup(&cfg, "max_fabric_packet_age");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf < 0) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_fabric_packet_age\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}

		/* Value of zero reuses default value. */
		if (intconf > 0)
			max_fabric_packet_age = intconf;
	}

	settings = config_lookup(&cfg, "unorder_pkt_min_retry_delay");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf) {
			if (intconf < 0 || intconf < max_fabric_packet_age) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"unorder_pkt_min_retry_delay\" value %u\n",
					  config_setting_source_line(settings),
					  intconf);
				rc = 1;
				goto out;
			}

			unorder_pkt_min_retry_delay = intconf;
		}
	}

	settings = config_lookup(&cfg, "max_no_matching_conn_retries");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf > 20) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_no_matching_conn_retries\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		max_no_matching_conn_retries = intconf;
	}

	settings = config_lookup(&cfg, "max_resource_busy_retries");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf > 20) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_resource_busy_retries\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		max_resource_busy_retries = intconf;
	}

	settings = config_lookup(&cfg, "max_trs_pend_rsp_retries");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf > 20) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_trs_pend_rsp_retries\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		max_trs_pend_rsp_retries = intconf;
	}

	/* max_sct_close_retries deprecates max_sct_retries */
	settings = config_lookup(&cfg, "max_sct_close_retries");
	if (!settings)
		settings = config_lookup(&cfg, "max_sct_retries");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf < 0 || intconf > 7) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_sct_close_retries\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		max_sct_close_retries = intconf;
	}

	settings = config_lookup(&cfg, "max_batch_size");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf > 64) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"max_batch_size\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		max_batch_size = intconf;
	}

	settings = config_lookup(&cfg, "initial_batch_size");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf > max_batch_size) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"initial_batch_size\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		initial_batch_size = intconf;
	}

	settings = config_lookup(&cfg, "nack_backoff_start");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf < 0 || intconf > 64) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"nack_backoff_start\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		nack_backoff_start = intconf;
	}

	settings = config_lookup(&cfg, "backoff_multiplier");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf < 1 || intconf > 10) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"backoff_multiplier\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		backoff_multiplier = intconf;
	}

	settings = config_lookup(&cfg, "timeout_backoff_factor");
	if (settings)
		rh_printf(rh, LOG_NOTICE, "timeout_backoff_factor is deprecated, use retry_intervals\n");

	settings = config_lookup(&cfg, "timeout_backoff_multiplier");
	if (settings)
		rh_printf(rh, LOG_NOTICE, "timeout_backoff_multiplier is deprecated, use retry_intervals\n");

	settings = config_lookup(&cfg, "spt_timeout_epoch");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf && (intconf < 11 || intconf > 36)) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"spt_timeout_epoch\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		user_spt_timeout_epoch = intconf;
	}

	settings = config_lookup(&cfg, "sct_idle_epoch");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf && (intconf < 12 || intconf > 36)) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"sct_idle_epoch\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		user_sct_idle_epoch = intconf;
	}

	settings = config_lookup(&cfg, "sct_close_epoch");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf && (intconf < 12 || intconf > 36)) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"sct_close_epoch\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		user_sct_close_epoch = intconf;
	}

	settings = config_lookup(&cfg, "tct_timeout_epoch");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf && (intconf < 14 || intconf > 38)) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"tct_timeout_epoch\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		user_tct_timeout_epoch = intconf;
	}

	settings = config_lookup(&cfg, "tct_wait_time");
	if (settings) {
		dconf = config_setting_get_float(settings);
		if (dconf <= 0 || dconf > 100) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"tct_wait_time\" value %e\n",
				  config_setting_source_line(settings), dconf);
			rc = 1;
			goto out;
		}
		tct_wait_time.tv_usec = modf(dconf, &y) * 1000000;
		tct_wait_time.tv_sec = y;
	}

	settings = config_lookup(&cfg, "pause_wait_time");
	if (settings) {
		dconf = config_setting_get_float(settings);
		if (dconf <= 0 || dconf > 10) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"pause_wait_time\" value %e\n",
				  config_setting_source_line(settings), dconf);
			rc = 1;
			goto out;
		}
		pause_wait_time.tv_usec = modf(dconf, &y) * 1000000;
		pause_wait_time.tv_sec = y;
	}

	settings = config_lookup(&cfg, "cancel_spt_wait_time");
	if (settings) {
		dconf = config_setting_get_float(settings);
		if (dconf <= 0 || dconf > 100) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"cancel_spt_wait_time\" value %e\n",
				  config_setting_source_line(settings), dconf);
			rc = 1;
			goto out;
		}
		cancel_spt_wait_time.tv_usec = modf(dconf, &y) * 1000000;
		cancel_spt_wait_time.tv_sec = y;
	}

	settings = config_lookup(&cfg, "peer_tct_free_wait_time");
	if (settings) {
		dconf = config_setting_get_float(settings);
		if (dconf <= 0 || dconf > 600) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"peer_tct_free_wait_time\" value %e\n",
				  config_setting_source_line(settings), dconf);
			rc = 1;
			goto out;
		}
		peer_tct_free_wait_time.tv_usec = modf(dconf, &y) * 1000000;
		peer_tct_free_wait_time.tv_sec = y;
	}

	settings = config_lookup(&cfg, "down_nid_wait_time");
	if (settings) {
		dconf = config_setting_get_float(settings);
		if (dconf < 0 || dconf > 600) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"down_nid_wait_time\" value %e\n",
				  config_setting_source_line(settings), dconf);
			rc = 1;
			goto out;
		}
		down_nid_wait_time.tv_usec = modf(dconf, &y) * 1000000;
		down_nid_wait_time.tv_sec = y;
	}

	settings = config_lookup(&cfg, "down_nid_spt_timeout_epoch");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf) {
			if (intconf < 11 || intconf > 36) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"down_nid_spt_timeout_epoch\" value %u\n",
					  config_setting_source_line(settings),
					  intconf);
				rc = 1;
				goto out;
			}

			down_nid_spt_timeout_epoch = intconf;
		}
	}

	settings = config_lookup(&cfg, "down_nid_get_packets_inflight");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf) {
			if (intconf < 1 || intconf > 2047) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"down_nid_get_packets_inflight\" value %u\n",
					  config_setting_source_line(settings),
					  intconf);
				rc = 1;
				goto out;
			}

			down_nid_get_packets_inflight = intconf;
		}
	}

	settings = config_lookup(&cfg, "down_nid_put_packets_inflight");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf) {
			if (intconf < 1 || intconf > 2047) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"down_nid_put_packets_inflight\" value %u\n",
					  config_setting_source_line(settings),
					  intconf);
				rc = 1;
				goto out;
			}

			down_nid_put_packets_inflight = intconf;
		}
	}

	settings = config_lookup(&cfg, "down_switch_nid_count");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf < 0) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"down_switch_nid_count\" value %u\n",
				  config_setting_source_line(settings),
				  intconf);
			rc = 1;
			goto out;
		}

		down_switch_nid_count = intconf;
	}

	settings = config_lookup(&cfg, "down_nid_pkt_count");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf) {
			if (intconf < 0) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"down_nid_pkt_count\" value %u\n",
					  config_setting_source_line(settings),
					  intconf);
				rc = 1;
				goto out;
			}

			down_nid_pkt_count = intconf;
		}
	}

	settings = config_lookup(&cfg, "switch_id_mask");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf) {
			if (intconf < 0 || intconf > DFA_MAX) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"switch_id_mask\" value %u\n",
					  config_setting_source_line(settings),
					  intconf);
				rc = 1;
				goto out;
			}

			switch_id_mask = intconf;
		}
	}

	settings = config_lookup(&cfg, "sct_stable_wait_time");
	if (settings) {
		dconf = config_setting_get_float(settings);
		if (dconf < 0.001 - __FLT_EPSILON__ || dconf > 0.1 + __FLT_EPSILON__) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"sct_stable_wait_time\" value %e\n",
				  config_setting_source_line(settings), dconf);
			rc = 1;
			goto out;
		}
		sct_stable_wait_time.tv_usec = modf(dconf, &y) * 1000000;
		sct_stable_wait_time.tv_sec = y;
	}

	settings = config_lookup(&cfg, "retry_intervals");
	if (settings) {
		array_size = config_setting_length(settings);
		/* Making sure that input size equals to max_spt_retries */
		if (array_size <= 0 || array_size != max_spt_retries) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"retry_intervals\" size %u\n",
				  config_setting_source_line(settings), array_size);
			rc = 1;
			goto out;
		}
		for (i = 0; i < array_size; ++i) {
			dconf = config_setting_get_float_elem(settings, i);
			intconf = dconf * 1000000;
			if (intconf < 0) {
				rh_printf(rh, LOG_ERR, "config error line %d, invalid \"retry_intervals\" value %e\n",
					  config_setting_source_line(settings), dconf);
				rc = 1;
				goto out;
			}
			retry_interval_values_us[i] = intconf;
		}
	}

	settings = config_lookup(&cfg, "allowed_retry_time_percent");
	if (settings) {
		intconf = config_setting_get_int(settings);
		if (intconf <= 0 || intconf >= 100) {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"allowed_retry_time_percent\" value %u\n",
				  config_setting_source_line(settings), intconf);
			rc = 1;
			goto out;
		}
		allowed_retry_time_percent = intconf;
	}

	settings = config_lookup(&cfg, "rh_stats_dir");
	if (settings) {
		sconf = config_setting_get_string(settings);
		dir = opendir(sconf);
		if (dir) {
			closedir(dir);
			rh_stats_dir = strdup(sconf);
			rh->cfg_stats = true;
		} else {
			rh_printf(rh, LOG_ERR, "config error line %d, invalid \"rh_stats_dir\" value %s\n",
				  config_setting_source_line(settings), sconf);
		}
	}
	rc = 0;

out:
	config_destroy(&cfg);

	return rc;
}
