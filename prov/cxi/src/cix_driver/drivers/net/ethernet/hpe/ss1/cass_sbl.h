/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020-2024 Hewlett Packard Enterprise Development LP */

#ifndef _CXI_SBL_H
#define _CXI_SBL_H

#include <linux/phy/phy.h>
#include "uapi/sbl.h"

#define PCA_SHASTA_BRAZOS_PASS_1_PN        "102325100"
#define PCA_SHASTA_BRAZOS_PASS_2_PN        "102325101"
#define PCA_SHASTA_BRAZOS_PASS_3_PN        "102325102"
#define PCA_SHASTA_BRAZOS_PASS_4_PN        "P41345-"

#define PCA_SHASTA_SAWTOOTH_PASS_1_PN      "102251000"
#define PCA_SHASTA_SAWTOOTH_PASS_2_PN      "102251001"
#define PCA_SHASTA_SAWTOOTH_PASS_3_PN      "P43012-"

#define CASS_LINK_ASYNC_DOWN_TIMEOUT              5000
#define CASS_LINK_ASYNC_DOWN_INTERVAL              500
#define CASS_LINK_ASYNC_RESET_TIMEOUT             5000
#define CASS_LINK_ASYNC_RESET_INTERVAL             500
#define CASS_PORT_DEFAULT_OPT_FLAGS                  0

#define CASS_PCS_NUM_FECL_CNTRS                           8

/**
 * @brief default link options flags
 */
#define CASS_LINK_ATTR_DEFAULT_OPT_FLAGS                  0
#define CASS_SBL_PORT_CONFIG1_PRE                         4
#define CASS_SBL_PORT_CONFIG2_ATTEN                       12
#define CASS_SBL_PORT_CONFIG2_GS1                         0
#define CASS_SBL_PORT_CONFIG2_NUM_INTR                    4
#define CASS_SBL_PORT_CONFIG2_INTR0                       0x2c
#define CASS_SBL_PORT_CONFIG2_DATA0                       0x0909
#define CASS_SBL_PORT_CONFIG2_INTR1                       0x6c
#define CASS_SBL_PORT_CONFIG2_DATA1                       0x3
#define CASS_SBL_PORT_CONFIG2_INTR2                       0x2c
#define CASS_SBL_PORT_CONFIG2_DATA2                       0x090a
#define CASS_SBL_PORT_CONFIG2_INTR3                       0x6c
#define CASS_SBL_PORT_CONFIG2_DATA3                       0xa
#define CASS_SBL_PORT_CONFIG3_ATTEN                       6
#define CASS_SBL_PORT_CONFIG3_PRE                         4
#define CASS_SBL_PORT_CONFIG3_POST                        0
#define CASS_SBL_PORT_CONFIG3_GS1                         0

#define CASS_SBL_SC_MASKED_OUT                            0
#define CASS_SBL_SC_MASKED_IN                             0xffffffffffffffff
#define CASS_SBL_SC_NA                                    0x0
#define CASS_SBL_SC_LP_SWITCH                             0x1
#define CASS_SBL_SC_MT_ELEC                               0x1
#define CASS_SBL_SC_MT_OPT_ANY                            0x6
#define CASS_SBL_SC_LBM_ANY                               0xff
#define CASS_SBL_SC_LM_100                                0x2
#define CASS_SBL_SC_LM_200                                0x1

/* NOTE: don't forget to add sysfs entries if changing this */
#define CASS_SBL_LINK_RESTART_TIME_SIZE 10

/* Link up FEC check */
#define CASS_SBL_UP_MIN_FEC_WINDOW             250    /* ms */

/**
 * @brief software counter names initialisers
 *
 *   provides a name for each counter index
 *
 */
#define CASS_SBL_COUNTER_NAMES  "link_up", \
				"link_up_tries", \
				"link_down", \
				"link_down_tries", \
				"link_async_down", \
				"link_async_down_tries", \
				"link_reset", \
				"link_reset_tries", \
				"link_auto_restart", \
				"link_down_origin_config", \
				"link_down_origin_bl_lfault", \
				"link_down_origin_bl_rfault", \
				"link_down_origin_bl_align", \
				"link_down_origin_bl_down", \
				"link_down_origin_bl_hiser", \
				"link_down_origin_bl_llr", \
				"link_down_origin_bl_unknown", \
				"link_down_origin_lmon_ucw", \
				"link_down_origin_lmon_ccw", \
				"link_down_origin_lmon_llr_tx_replay", \
				"link_down_origin_headshell_removed", \
				"link_down_origin_headshell_error", \
				"link_down_origin_media_removed", \
				"link_down_origin_cmd", \
				SBL_COUNTERS_NAME

#define CASS_LMON_COUNTER_NAMES "lmon_wakeup", \
				"lmon_active", \
				"lmon_limiter"

/**
 * enum cass_port_subtype - The sub-type of a port
 * @CASS_PORT_SUBTYPE_INVALID: Invalid
 * @CASS_PORT_SUBTYPE_LOCAL:   Local group fabric port
 * @CASS_PORT_SUBTYPE_GLOBAL:  Global (inter-group) fabric port
 * @CASS_PORT_SUBTYPE_IEEE:    Standard Ethernet NIC
 * @CASS_PORT_SUBTYPE_CASSINI: Cassini NIC
 *
 * The sub-type of a port
 */
enum cass_port_subtype {
	CASS_PORT_SUBTYPE_INVALID = 0,
	CASS_PORT_SUBTYPE_LOCAL,
	CASS_PORT_SUBTYPE_GLOBAL,
	CASS_PORT_SUBTYPE_IEEE,
	CASS_PORT_SUBTYPE_CASSINI,
};

/**
 * enum cass_port_command - Commands that can be issued to a port
 * @CASS_CMD_NULL:       Invalid
 * @CASS_CMD_NOOP:       Do nothing (for testing)
 * @CASS_CMD_PORT_START: Request port starts
 * @CASS_CMD_PORT_STOP:  Request port stops
 * @CASS_CMD_PORT_RESET: Request port reset
 * @CASS_CMD_LINK_UP:    Request link bring-up
 * @CASS_CMD_LINK_DOWN:  Request link bring-down
 * @CASS_CMD_LINK_RESET: Request link resets
 *
 * Commands that can be issued to a port
 */
enum cass_port_command {
	CASS_CMD_NULL = 0,
	CASS_CMD_NOOP,
	CASS_CMD_PORT_START,
	CASS_CMD_PORT_STOP,
	CASS_CMD_PORT_RESET,
	CASS_CMD_LINK_UP,
	CASS_CMD_LINK_DOWN,
	CASS_CMD_LINK_RESET,
};

/**
 * enum cass_port_status - The status of a port
 * @CASS_PORT_STATUS_UNKNOWN:          State unknown
 * @CASS_PORT_STATUS_NOT_PRESENT:      Port not present in hardware
 * @CASS_PORT_STATUS_UNCONFIGURED:     Port is not configured
 * @CASS_PORT_STATUS_STARTING:         Port Starting
 * @CASS_PORT_STATUS_UP:               Port Up
 * @CASS_PORT_STATUS_RUNNING:          Port Up and Link Up
 * @CASS_PORT_STATUS_RUNNING_DEGRADED: Port Up and Link Up but link degraded
 * @CASS_PORT_STATUS_STOPPING:         Port Stopping
 * @CASS_PORT_STATUS_DOWN:             Port is down and ready to start
 * @CASS_PORT_STATUS_RESETTING:        Port is resetting
 * @CASS_PORT_STATUS_ERROR:            Port has an Error
 *
 * The status of a port
 */
enum cass_port_status {
	CASS_PORT_STATUS_UNKNOWN          = 0,
	CASS_PORT_STATUS_NOT_PRESENT      = 1<<0,
	CASS_PORT_STATUS_UNCONFIGURED     = 1<<1,
	CASS_PORT_STATUS_STARTING         = 1<<2,
	CASS_PORT_STATUS_UP               = 1<<3,
	CASS_PORT_STATUS_RUNNING          = 1<<4,
	CASS_PORT_STATUS_RUNNING_DEGRADED = 1<<5,
	CASS_PORT_STATUS_STOPPING         = 1<<6,
	CASS_PORT_STATUS_DOWN             = 1<<7,
	CASS_PORT_STATUS_RESETTING        = 1<<8,
	CASS_PORT_STATUS_ERROR            = 1<<9,
};

/**
 * enum cass_lmon_direction - link monitor direction flags
 *
 * @CASS_LMON_DIRECTION_INVALID: Invalid
 * @CASS_LMON_DIRECTION_NONE:    no action requested
 * @CASS_LMON_DIRECTION_UP:      port should try to come up
 * @CASS_LMON_DIRECTION_DOWN:    port should try to come down
 * @CASS_LMON_DIRECTION_RESET:   port should reset
 */
enum cass_lmon_direction {
	CASS_LMON_DIRECTION_INVALID = 0,
	CASS_LMON_DIRECTION_NONE,
	CASS_LMON_DIRECTION_UP,
	CASS_LMON_DIRECTION_DOWN,
	CASS_LMON_DIRECTION_RESET,
};

/**
 * enum cass_link_status - Status of the link
 *
 * @CASS_LINK_STATUS_UNKNOWN:		State unknown
 * @CASS_LINK_STATUS_UNCONFIGURED:	Link is not configured
 * @CASS_LINK_STATUS_STARTING:		Link bring-up in progress
 * @CASS_LINK_STATUS_UP:		Link Up
 * @CASS_LINK_STATUS_STOPPING:		Link bring-down in progress
 * @CASS_LINK_STATUS_DOWN:		Link Down
 * @CASS_LINK_STATUS_RESETTING:		Link reset in progress
 * @CASS_LINK_STATUS_ERROR:		Link has an error
 */
enum cass_link_status {
	CASS_LINK_STATUS_UNKNOWN      = 0,
	CASS_LINK_STATUS_UNCONFIGURED = 1<<0,
	CASS_LINK_STATUS_STARTING     = 1<<1,
	CASS_LINK_STATUS_UP           = 1<<2,
	CASS_LINK_STATUS_STOPPING     = 1<<3,
	CASS_LINK_STATUS_DOWN         = 1<<4,
	CASS_LINK_STATUS_RESETTING    = 1<<5,
	CASS_LINK_STATUS_ERROR        = 1<<6,
};

/**
 * enum cass_link_status - Link configuration option flags
 * @CASS_LINK_OPT_UP_AUTO_RESTART:	Keep trying to bring up a link that has
 *					gone down
 * @CASS_LINK_OPT_NO_SHUTDOWN:		Keep trying to bring up a link that has
 *					gone down
 * @CASS_LINK_OPT_RESTART_ON_INSERT:    Allow link restart if up before
 *                                      headshell remove
 *
 * Link configuration option flags
 */
enum cass_link_options {
	CASS_LINK_OPT_UP_AUTO_RESTART    = 1<<1,
	CASS_LINK_OPT_NO_SHUTDOWN        = CASS_LINK_OPT_UP_AUTO_RESTART,
	CASS_LINK_OPT_RESTART_ON_INSERT  = 1<<2,
};

/**
 * enum cass_link_down_origin - The reason the link went down
 * @CASS_DOWN_ORIGIN_UNKNOWN:		Origin is not known
 * @CASS_DOWN_ORIGIN_NONE:		not coming down
 * @CASS_DOWN_ORIGIN_CONFIG:		down after config
 * @CASS_DOWN_ORIGIN_BL_LFAULT:		base link - local fault
 * @CASS_DOWN_ORIGIN_BL_RFAULT:		base link - remote fault
 * @CASS_DOWN_ORIGIN_BL_ALIGN:		base link - alignment lost
 * @CASS_DOWN_ORIGIN_BL_DOWN:		base link - link down
 * @CASS_DOWN_ORIGIN_BL_HISER:		base link - high serdes errors
 * @CASS_DOWN_ORIGIN_BL_LLR:		base link - llr max buffer
 * @CASS_DOWN_ORIGIN_BL_UNKNOWN:	base link - some other reason
 * @CASS_DOWN_ORIGIN_LMON_UCW:		lmon - high uncorrected fec error rate
 * @CASS_DOWN_ORIGIN_LMON_CCW:		lmon - high corrected fec error rate
 * @CASS_DOWN_ORIGIN_LLR_TX_REPLAY:	lmon - high llr_tx_replay fec error rate
 * @CASS_DOWN_ORIGIN_HEADSHELL_REMOVED: headshell removed
 * @CASS_DOWN_ORIGIN_HEADSHELL_ERROR:   headshell fault
 * @CASS_DOWN_ORIGIN_MEDIA_REMOVED:     media removed
 * @CASS_DOWN_ORIGIN_CMD:               direct command
 *
 * The last reason why the link has gone down.
 */
enum cass_link_down_origin {
	CASS_DOWN_ORIGIN_UNKNOWN,
	CASS_DOWN_ORIGIN_NONE,
	CASS_DOWN_ORIGIN_CONFIG,
	CASS_DOWN_ORIGIN_BL_LFAULT,
	CASS_DOWN_ORIGIN_BL_RFAULT,
	CASS_DOWN_ORIGIN_BL_ALIGN,
	CASS_DOWN_ORIGIN_BL_DOWN,
	CASS_DOWN_ORIGIN_BL_HISER,
	CASS_DOWN_ORIGIN_BL_LLR,
	CASS_DOWN_ORIGIN_BL_UNKNOWN,
	CASS_DOWN_ORIGIN_LMON_UCW,
	CASS_DOWN_ORIGIN_LMON_CCW,
	CASS_DOWN_ORIGIN_LLR_TX_REPLAY,
	CASS_DOWN_ORIGIN_HEADSHELL_REMOVED,
	CASS_DOWN_ORIGIN_HEADSHELL_ERROR,
	CASS_DOWN_ORIGIN_MEDIA_REMOVED,
	CASS_DOWN_ORIGIN_CMD,
};

/**
 * enum cass_port_config_flags - Configure/unconfigured state
 * @CASS_TYPE_CONFIGURED:	type info has been configured
 * @CASS_PORT_CONFIGURED:	port info  has been configured
 * @CASS_LINK_CONFIGURED:	link info has been configured
 * @CASS_MEDIA_CONFIGURED:	media info has been configured
 *
 * Configured/uncofigured state for tpml
 */
enum cass_port_config_flags {
	CASS_TYPE_CONFIGURED   = (1<<0),
	CASS_PORT_CONFIGURED   = (1<<1),
	CASS_LINK_CONFIGURED   = (1<<2),
	CASS_MEDIA_CONFIGURED  = (1<<3),
};

/**
 * enum cass_phy_mode - Phy mode for Cassini SerDes
 * @PHY_MODE_NORMAL:		normal operation
 * @PHY_MODE_TX_DISABLED:	Transmitter disabled
 * @PHY_MODE_LOW_POWER:		SerDes in low power mode
 * @PHY_MODE_OFF:		SerDes powered off
 * @PHY_MODE_SPECIAL:		special operation flag
 *
 * Phy mode for Cassini SerDes
 */
enum cass_phy_mode {
	PHY_MODE_NORMAL		= 0,
	PHY_MODE_TX_DISABLED	= 1,
	PHY_MODE_LOW_POWER	= 2,
	PHY_MODE_OFF		= 4,
	PHY_MODE_SPECIAL	= 8,
};

/**
 * enum cass_pause_type - Pause type for Cassini
 * @CASS_PAUSE_TYPE_INVALID:	invalid pause type
 * @CASS_PAUSE_TYPE_NONE:		no pause specified
 * @CASS_PAUSE_TYPE_GLOBAL:		standard pause
 * @CASS_PAUSE_TYPE_802_3X:		standard pause
 * @CASS_PAUSE_TYPE_PFC:		priority flow control pause
 * @CASS_PAUSE_TYPE_802_1QBB:	priority flow control pause
 *
 * Pause type for Cassini
 */
enum cass_pause_type {
	CASS_PAUSE_TYPE_INVALID  = 0,
	CASS_PAUSE_TYPE_NONE,
	CASS_PAUSE_TYPE_GLOBAL,
	CASS_PAUSE_TYPE_802_3X   = CASS_PAUSE_TYPE_GLOBAL,
	CASS_PAUSE_TYPE_PFC,
	CASS_PAUSE_TYPE_802_1QBB = CASS_PAUSE_TYPE_PFC,
};

/**
 * enum cass_sbl_counters_idx - Software counter indexes
 *
 * Counters cannot be reset
 *
 * @link_up:                            link reached up state
 * @link_up_tries:                      link up called
 * @link_down:                          link reached down state
 * @link_down_tries:                    link down called
 * @link_async_down:                    link async down triggers down
 * @link_async_down_tries:              link async down called
 * @link_reset:                         link reset completed
 * @link_reset_tries:                   link reset called
 * @link_auto_restart:                  lmon link auto-restarted
 * @link_down_origin_config:            after config
 * @link_down_origin_bl_lfault:         pcs local fault
 * @link_down_origin_bl_rfault:         pcr remote fault
 * @link_down_origin_bl_align:          pcs align
 * @link_down_origin_bl_down:           pcs link down
 * @link_down_origin_bl_hiser:          pcs high serdes error rate
 * @link_down_origin_bl_llr:            llr error
 * @link_down_origin_bl_unknown:        base link unknown
 * @link_down_origin_lmon_ucw:          lmon high ucw rate
 * @link_down_origin_lmon_ccw:          lmon high ccw rate
 * @link_down_origin_lmon_llr_tx_replay:lmon high llr tx replay rate
 * @link_down_origin_headshell_removed: headshell removed
 * @link_down_origin_headshell_error:   headshell in error
 * @link_down_origin_media_removed:     media removed
 * @link_down_origin_cmd:               user command
 * @SBL_COUNTERS:                       sbl counters defined in sbl_counters.h
 * @CASS_SBL_NUM_COUNTERS:              the number of counters
 */
enum cass_sbl_counters_idx {
	link_up,
	link_up_tries,
	link_down,
	link_down_tries,
	link_async_down,
	link_async_down_tries,
	link_reset,
	link_reset_tries,
	link_auto_restart,
	link_down_origin_config,
	link_down_origin_bl_lfault,
	link_down_origin_bl_rfault,
	link_down_origin_bl_align,
	link_down_origin_bl_down,
	link_down_origin_bl_hiser,
	link_down_origin_bl_llr,
	link_down_origin_bl_unknown,
	link_down_origin_lmon_ucw,
	link_down_origin_lmon_ccw,
	link_down_origin_lmon_llr_tx_replay,
	link_down_origin_headshell_removed,
	link_down_origin_headshell_error,
	link_down_origin_media_removed,
	link_down_origin_cmd,
	SBL_COUNTERS,
	CASS_SBL_NUM_COUNTERS,
};

/**
 * enum cass_counters_idx - Software counter indexes
 *
 * Counters cannot be reset
 *
 * @lmon_wakeup:                        lmon wakeup
 * @lmon_active:                        lmon active
 * @lmon_limiter:                       lmon limiter
 */
enum cass_lmon_counters_idx {
	lmon_wakeup,
	lmon_active,
	lmon_limiter,
	CASS_LMON_NUM_COUNTERS,
};

/**
 * struct cass_link_attr - Attributes for link configuration
 * @bl:                 base-link attributes
 * @options:            option flags for links
 * @el:                 ether link specific attributes;
 * @el.mattr:           cable related configuration
 *
 * These attributes must be configured for a link.
 * Different setting will be required for fabric or Ethernet links
 * The anonymous union is not required by base link.
 */
struct cass_link_attr {
	struct sbl_base_link_attr bl;
	__u32 options;
	struct sbl_media_attr mattr;
};

/**
 * struct cass_port - port-specific values
 *
 * @subtype:                  ieee or casssini
 * @lock:                     TODO what lock type?
 * @lmon:                     link monitor thread
 * @lmon_dirn:                lmon direction
 * @lmon_wq:                  wq for waking the link monitor
 * @lmon_active:              lmon is awake
 * @lmon_limiter_on:          rate limiter is active
 * @lmon_up_pause:            link up is paused
 * @lmon_wake_jiffies:        end of wake count period
 * @lmon_wake_cnt:            wakeups in period
 * @lmon_counters:            sw counter block (can never be cleared)
 * @config_state:             what has been configured
 * @lattr:                    link related configuration
 * @lstate:                   link state
 * @prev_lstate:              previous link state
 * @lerr:                     error number that caused error state
 * @hstate:                   headshell state
 * @link_down_origin:         reason we are going down
 * @link_restart_count:       auto link restarts since last up command
 * @link_restart_time_idx     auto link restarts current buffer index
 * @link_restart_time_buf     auto link restarts timestamp buffer
 * @pause_lock:               lock on pause state access
 * @pause_type:               type of pause (global/pfc)
 * @tx_pause:                 enable tx pause
 * @rx_pause:                 enable rx pause
 */
struct cass_port {
	int subtype;
	spinlock_t lock;
	struct task_struct *lmon;
	int lmon_dirn;
	wait_queue_head_t lmon_wq;
	bool lmon_active;
	bool lmon_limiter_on;
	bool lmon_up_pause;
	unsigned long lmon_wake_jiffies;
	int lmon_wake_cnt;
	atomic_t *lmon_counters;
	u32 config_state;
	struct cass_link_attr lattr;
	u32 lstate;
	u32 prev_lstate;
	u32 lerr;
	u32 hstate;
	u32 link_down_origin;
	u32 link_restart_count;
	u32 link_restart_time_idx;
	time64_t link_restart_time_buf[CASS_SBL_LINK_RESTART_TIME_SIZE];
	spinlock_t pause_lock;
	u32 pause_type;
	bool tx_pause;
	bool rx_pause;
	time64_t start_time;
};

int cass_sbl_init(struct cass_dev *hw);
int cass_sbl_link_start(struct cass_dev *hw);
void cass_sbl_link_fini(struct cass_dev *hw);
int cass_sbl_power_on(struct cass_dev *hw);
int cass_sbl_power_off(struct cass_dev *hw);
void cass_sbl_set_defaults(struct cass_dev *hw);
int cass_sbl_configure(struct cass_dev *hw);
int cass_sbl_reset(struct cass_dev *hw);

/*
 * link monitors (lmon)
 */
int  cass_lmon_start_all(struct cass_dev *hw);
int  cass_lmon_stop_all(struct cass_dev *hw);
void cass_lmon_kill_all(struct cass_dev *hw);
int cass_lmon_get_dirn(struct cass_dev *hw);
void cass_lmon_set_dirn(struct cass_dev *hw, int dirn);
bool cass_lmon_get_active(struct cass_dev *hw);
void cass_lmon_set_active(struct cass_dev *hw, bool state);
void cass_lmon_set_up_pause(struct cass_dev *hw, bool state);
bool cass_lmon_get_up_pause(struct cass_dev *hw);
int  cass_lmon_request_up(struct cass_dev *hw);
int  cass_lmon_request_down(struct cass_dev *hw);
int  cass_lmon_request_reset(struct cass_dev *hw);
const char *cass_lmon_dirn_str(enum cass_lmon_direction dirn);

/*
 * misc
 */
int  cass_link_get_down_origin(struct cass_dev *hw);
void cass_link_set_down_origin(struct cass_dev *hw, int origin);
const char *cass_link_state_str(enum cass_link_status state);
const char *cass_link_down_origin_str(u32 origin);
void cass_tc_get_hni_pause_cfg(struct cass_dev *hw);

/*
 * Port database
 */
int cass_port_new_port_db(struct cass_dev *hw);
void cass_port_del_port_db(struct cass_dev *hw);

/*
 * link state access
 */
int  cass_link_get_state(struct cass_dev *hw);
void cass_link_set_state(struct cass_dev *hw, int state, int err);

/*
 * set link led
 */
void cass_link_set_led(struct cass_dev *hw);

/*
 * sysfs
 */
int cass_link_sysfs_sprint(struct cass_dev *hw, char *buf, size_t size);
int cass_pause_sysfs_sprint(struct cass_dev *hw, char *buf, size_t size);

/*
 * Text output helpers
 */
const char *cass_port_subtype_str(enum cass_port_subtype subtype);
const char *cass_link_state_str(enum cass_link_status state);
const char *cass_pause_type_str(enum cass_pause_type type);

/*
 * principal link functions
 */
void cass_link_async_down(struct cass_dev *hw, u32 origin);
int  cass_link_async_down_wait(struct cass_dev *hw, u32 origin);
int cass_link_async_reset_wait(struct cass_dev *hw, u32 origin);

void cass_sbl_mode_get(struct cass_dev *hw, struct cxi_link_info *link_info);
void cass_sbl_mode_set(struct cass_dev *hw, const struct cxi_link_info *link_info);
void cass_sbl_pml_recovery_set(struct cass_dev *hw, bool set);
void cass_sbl_get_debug_flags(struct cass_dev *hw, u32 *flags);
void cass_sbl_set_debug_flags(struct cass_dev *hw, u32 clr_flags, u32 set_flags);
void cass_sbl_link_mode_to_speed(u32 link_mode, int *speed);
void cass_sbl_link_speed_to_mode(int speed, u32 *link_mode);

int cass_sbl_media_config(struct cass_dev *hw, void *attr);
int cass_sbl_media_unconfig(struct cass_dev *hw);
int cass_sbl_link_config(struct cass_dev *hw);
bool cass_sbl_is_link_up(struct cass_dev *hw);

bool cass_sbl_pml_pcs_aligned(struct cass_dev *hw);
void cass_sbl_exit(struct cass_dev *hw);

/*
 * sbl counters
 */
void cass_sbl_counters_init(struct cass_dev *hw);
void cass_sbl_counters_term(struct cass_dev *hw);
void cass_sbl_counters_down_origin_inc(struct cass_dev *hw, int down_origin);
void cass_sbl_counters_update(struct cass_dev *hw);

/*
 * lmon counters
 */
void cass_lmon_counters_init(struct cass_dev *hw);
void cass_lmon_counters_term(struct cass_dev *hw);

int cass_sbl_set_eth_name(struct cass_dev *hw, const char *name);
#endif	/* _CXI_SBL_H */
