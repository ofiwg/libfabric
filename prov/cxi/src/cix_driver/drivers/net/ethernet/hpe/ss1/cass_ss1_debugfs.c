// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018,2025 Hewlett Packard Enterprise Development LP */

/* cxi-ss1 debug fs*/
#include <linux/etherdevice.h>
#include <linux/seq_file.h>
#include <linux/sbl.h>
#include "cass_core.h"
#include "cass_ss1_debugfs.h"

#define DEBUGFS_BUFSIZE        1000

/* Dump UC logging to debugfs */
static int uc_log(struct seq_file *s, void *unused)
{
	struct cass_dev *hw = s->private;
	int rc;

	if (!hw->uc_present)
		return 0;

	mutex_lock(&hw->uc_mbox_mutex);

	while (!seq_has_overflowed(s)) {
		uc_prepare_comm(hw);

		hw->uc_req.cmd = CUC_CMD_GET_LOG;
		hw->uc_req.count = 1;

		rc = uc_wait_for_response(hw);
		if (rc || hw->uc_resp.count <= 1)
			break;

		hw->uc_resp.data[CUC_DATA_BYTES - 1] = 0;
		seq_printf(s, "%s\n", hw->uc_resp.data);
	}

	mutex_unlock(&hw->uc_mbox_mutex);

	return 0;
}

static int uc_log_debug_dev_open(struct inode *inode, struct file *file)
{
	return single_open(file, uc_log, inode->i_private);
}

const struct file_operations uc_fops = {
	.owner = THIS_MODULE,
	.open = uc_log_debug_dev_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

/*
 * print out pause state for debugfs diags
 *
 */
static void cass_pause_debugfs_print(struct cass_dev *hw, struct seq_file *s)
{
	u32 pause_type;
	bool tx_pause;
	bool rx_pause;

	spin_lock(&hw->port->pause_lock);
	pause_type = hw->port->pause_type;
	tx_pause = hw->port->tx_pause;
	rx_pause = hw->port->rx_pause;
	spin_unlock(&hw->port->pause_lock);

	seq_printf(s, "pause: %s", cass_pause_type_str(pause_type));
	switch (pause_type) {
	case CASS_PAUSE_TYPE_GLOBAL:
	case CASS_PAUSE_TYPE_PFC:
		seq_printf(s, ", tx %s, rx %s",
			tx_pause ? "on" : "off", rx_pause ? "on" : "off");
		break;
	default:
		break;
	}
	seq_puts(s, "\n");
}

static const int rsrc_dump_order[] = {
	CXI_RESOURCE_AC,
	CXI_RESOURCE_CT,
	CXI_RESOURCE_EQ,
	CXI_RESOURCE_PTLTE,
	CXI_RESOURCE_TGQ,
	CXI_RESOURCE_TXQ,
	CXI_RESOURCE_PE0_LE,
	CXI_RESOURCE_PE1_LE,
	CXI_RESOURCE_PE2_LE,
	CXI_RESOURCE_PE3_LE,
	CXI_RESOURCE_TLE
};

static int dump_rgroups(struct cass_dev *hw, struct seq_file *s)
{
	int i;
	int rc;
	unsigned long index;
	struct cxi_rgroup *rgroup;
	struct cxi_resource_entry *entry;

	seq_puts(s, "Rgroups:");
	for_each_rgroup(index, rgroup) {
		seq_printf(s, "\nID: %u\n", rgroup->id);
		seq_printf(s, "  Name:%s\n", cxi_rgroup_name(rgroup));
		seq_printf(s, "  State:%s\n",
			   cxi_rgroup_is_enabled(rgroup) ? "Enabled" : "Disabled");
		seq_printf(s, "  LNIs/RGID:%d\n",
			   cxi_rgroup_lnis_per_rgid(rgroup));
		seq_printf(s, "  System service:%d\n",
			   cxi_rgroup_system_service(rgroup));
		seq_printf(s, "  Counter pool ID:%d\n",
			   cxi_rgroup_cntr_pool_id(rgroup));
		seq_printf(s, "  LE pool IDs: %d %d %d %d  TLE pool ID: %d\n",
			   rgroup->pools.le_pool_id[0],
			   rgroup->pools.le_pool_id[1],
			   rgroup->pools.le_pool_id[2],
			   rgroup->pools.le_pool_id[3],
			   rgroup->pools.tle_pool_id);

		cxi_rgroup_print_ac_entry_info(rgroup, s);

		seq_puts(s, "           ACs     CTs     EQs    PTEs    TGQs    TXQs    LE0s    LE1s    LE2s    LE3s    TLEs\n");
		seq_puts(s, "  max   ");
		for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++) {
			rc = cxi_rgroup_get_resource_entry(rgroup,
							   rsrc_dump_order[i],
							   &entry);
			seq_printf(s, "%6lu  ", rc ? 0 : entry->limits.max);
		}
		seq_puts(s, "\n");

		seq_puts(s, "  res   ");
		for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++) {
			rc = cxi_rgroup_get_resource_entry(rgroup,
							   rsrc_dump_order[i],
							   &entry);
			seq_printf(s, "%6lu  ", rc ? 0 : entry->limits.reserved);
		}
		seq_puts(s, "\n");

		seq_puts(s, "Alloc   ");
		for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++) {
			rc = cxi_rgroup_get_resource_entry(rgroup,
							   rsrc_dump_order[i],
							   &entry);
			seq_printf(s, "%6lu  ", rc ? 0 : entry->limits.in_use);
		}
		seq_puts(s, "\n");
	}

	cxi_rx_profile_print(s);
	cxi_tx_profile_print(s);

	return 0;
}

static int dump_services(struct seq_file *s, void *unused)
{
	int i;
	ulong value;
	struct cass_dev *hw = s->private;
	struct cxi_resource_use *rsrc_use = hw->resource_use;

	spin_lock(&hw->rgrp_lock);

	seq_puts(s, "Resources\n");
	seq_puts(s, "           ACs     CTs     EQs    PTEs    TGQs    TXQs    LE0s    LE1s    LE2s    LE3s    TLEs\n");

	seq_puts(s, "  Max ");
	for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++)
		seq_printf(s, "  %6lu", rsrc_use[rsrc_dump_order[i]].max);
	seq_puts(s, "\n");

	seq_puts(s, "  Res ");
	for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++)
		seq_printf(s, "  %6lu", rsrc_use[rsrc_dump_order[i]].reserved);
	seq_puts(s, "\n");

	seq_puts(s, "Alloc ");
	for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++)
		seq_printf(s, "  %6lu", rsrc_use[rsrc_dump_order[i]].in_use);
	seq_puts(s, "\n");

	seq_puts(s, "Avail ");
	for (i = 0; i < ARRAY_SIZE(rsrc_dump_order); i++) {
		value = rsrc_use[rsrc_dump_order[i]].shared -
			rsrc_use[rsrc_dump_order[i]].shared_use;
		seq_printf(s, "  %6lu", value);
	}
	seq_puts(s, "\n\n");

	dump_rgroups(hw, s);

	spin_unlock(&hw->rgrp_lock);

	return 0;
}

static int dump_service_debug_dev_open(struct inode *inode, struct file *file)
{
	return single_open(file, dump_services, inode->i_private);
}

static const struct file_operations svc_debug_fops = {
	.owner = THIS_MODULE,
	.open = dump_service_debug_dev_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

void svc_service_debugfs_create(struct cass_dev *hw)
{
	hw->svc_debug = debugfs_create_file("services", 0444, hw->debug_dir,
					hw, &svc_debug_fops);
}

static int cass_port_uptime_str(time64_t time, char *buf, int len)
{
	int days;
	int hours;
	int mins;
	int secs;
	int bytes;

	/* divide time by seconds per day to get days */
	days  = time / (24*60*60);
	time %= 24*60*60;

	/* divide remainder by seconds per hour to get hours */
	hours = time / (60*60);
	time %= 60*60;

	/* divide remainder by seconds per minute to get minutes */
	mins  = time / 60;
	time %= 60;

	/* remainder in seconds */
	secs  = time;

	if (days)
		bytes = snprintf(buf, len, "%dd %dh %dm %ds", days, hours, mins, secs);
	else if (hours)
		bytes = snprintf(buf, len, "%dh %dm %ds", hours, mins, secs);
	else if (mins)
		bytes = snprintf(buf, len, "%dm %ds", mins, secs);
	else
		bytes = snprintf(buf, len, "%ds", secs);

	if (bytes > len)
		return -ENOSPC;
	else
		return 0;
}
/*
 * return the time (s) that we have been running for
 * if we are not running return 0;
 */
static time64_t cass_port_uptime_get(struct cass_dev *hw)
{
	if (hw->port->lstate == CASS_LINK_STATUS_UP)
		return  ktime_get_seconds() - hw->port->start_time;

	return 0;
}

static void cass_uptime_debugfs_print(struct cass_dev *hw, struct seq_file *s)
{
	time64_t uptime;
	char uptime_str[64];
	int err;

	uptime = cass_port_uptime_get(hw);
	if (!uptime)
		return;

	err = cass_port_uptime_str(uptime, uptime_str, sizeof(uptime_str));
	if (err) {
		cxidev_err(&hw->cdev, "get uptime str failed [%d]\n", err);
		return;
	}

	seq_printf(s, "uptime: %lld (%s)\n", uptime, uptime_str);
}

static void cass_lmon_debugfs_print(struct cass_dev *hw, struct seq_file *s)
{
	struct task_struct *lmon;
	int limiter;
	int dirn;
	bool active;
	u32 restart_count;

	spin_lock(&hw->port->lock);
	lmon  = hw->port->lmon;
	limiter  = hw->port->lmon_limiter_on;
	dirn  = hw->port->lmon_dirn;
	active = hw->port->lmon_active;
	restart_count = hw->port->link_restart_count;
	spin_unlock(&hw->port->lock);

	if (!lmon)
		return;

	seq_printf(s, "lmon: dirn %s%s%s", cass_lmon_dirn_str(dirn),
		   active ? ", active" : "", limiter ? ", (limited)" : "");
	if (cass_lmon_get_up_pause(hw))
		seq_puts(s, ", up-paused");
	if (hw->port->lattr.options & CASS_LINK_OPT_UP_AUTO_RESTART) {
		seq_puts(s, ", up-auto-restart");
		if (restart_count && (dirn == CASS_LMON_DIRECTION_UP))
			seq_printf(s, " (%d)", restart_count);
	}
	seq_puts(s, "\n");
}

static void cass_link_debugfs_print(struct cass_dev *hw, struct seq_file *s)
{
	int state;
	int down_origin;
	int lerr;

	spin_lock(&hw->port->lock);
	state = hw->port->lstate;
	lerr = hw->port->lerr;
	down_origin = hw->port->link_down_origin;
	spin_unlock(&hw->port->lock);

	seq_printf(s, "link state: %s", cass_link_state_str(state));

	if (state == CASS_LINK_STATUS_ERROR)
		seq_printf(s, " [%d]\n", lerr);
	else if (state == CASS_LINK_STATUS_DOWN)
		seq_printf(s, " (%s)\n",
			   cass_link_down_origin_str(down_origin));
	else
		seq_puts(s, "\n");
}

static int cass_port_show(struct seq_file *s, void *unused)
{
	struct cass_dev *hw = s->private;
	char buf[DEBUGFS_BUFSIZE];
	int nic_num = hw->uc_nic;

	if (!cass_version(hw, CASSINI_1)) {
		seq_puts(s, "NOT IMPLEMENTED!!\n");
		return 0;
	}

	seq_printf(s, "** %d **\n", nic_num);

	/* config */
	spin_lock(&hw->port->lock);
	seq_printf(s, "configured: %c%c%c%c\n",
		 (hw->port->config_state & CASS_TYPE_CONFIGURED)   ? 't' : '-',
		 (hw->port->config_state & CASS_PORT_CONFIGURED)   ? 'p' : '-',
		 (hw->port->config_state & CASS_MEDIA_CONFIGURED)  ? 'm' : '-',
		 (hw->port->config_state & CASS_LINK_CONFIGURED)   ? 'l' : '-');

	/* port */
	seq_printf(s, "type: ether (%s)\n",
		   cass_port_subtype_str(hw->port->subtype));
	spin_unlock(&hw->port->lock);

	/* uptime */
	cass_uptime_debugfs_print(hw, s);

	/* media */
	memset(buf, 0, DEBUGFS_BUFSIZE);
	sbl_media_sysfs_sprint(hw->sbl, 0, buf, DEBUGFS_BUFSIZE);
	seq_puts(s, buf);

	/* pause */
	cass_pause_debugfs_print(hw, s);

	/* serdes */
	memset(buf, 0, DEBUGFS_BUFSIZE);
	sbl_serdes_sysfs_sprint(hw->sbl, 0, buf, DEBUGFS_BUFSIZE);
	seq_puts(s, buf);

	/* pcs */
	memset(buf, 0, DEBUGFS_BUFSIZE);
	sbl_pml_pcs_sysfs_sprint(hw->sbl, 0, buf, DEBUGFS_BUFSIZE);
	seq_puts(s, buf);

	/* lane degrade status */
	memset(buf, 0, DEBUGFS_BUFSIZE);
	sbl_pml_pcs_lane_degrade_sysfs_sprint(hw->sbl, 0, buf, DEBUGFS_BUFSIZE);
	seq_puts(s, buf);

	/* base link */
	memset(buf, 0, DEBUGFS_BUFSIZE);
	sbl_base_link_sysfs_sprint(hw->sbl, 0, buf, DEBUGFS_BUFSIZE);
	seq_puts(s, buf);

	/* base link */
	memset(buf, 0, DEBUGFS_BUFSIZE);
	sbl_fec_sysfs_sprint(hw->sbl, 0, buf, DEBUGFS_BUFSIZE);
	seq_puts(s, buf);

	/* link */
	cass_link_debugfs_print(hw, s);
	cass_lmon_debugfs_print(hw, s);

	return 0;
}

static int cass_port_open(struct inode *inode, struct file *file)
{
	return single_open(file, cass_port_show, inode->i_private);
}

static const struct file_operations cass_port_fops = {
	.owner = THIS_MODULE,
	.open = cass_port_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

void cass_port_debugfs_init(struct cass_dev *hw)
{
	debugfs_create_file("port", 0444, hw->debug_dir, hw, &cass_port_fops);
}

static int cass_sbl_counters_show(struct seq_file *s, void *unused)
{
	struct cass_dev *hw = s->private;
	static char *sbl_counter_names[CASS_SBL_NUM_COUNTERS] = { CASS_SBL_COUNTER_NAMES };
	static char *lmon_counter_names[CASS_LMON_NUM_COUNTERS] = { CASS_LMON_COUNTER_NAMES };
	int i;

	cass_sbl_counters_update(hw);

	for (i = 0; i < CASS_SBL_NUM_COUNTERS; ++i)
		seq_printf(s, "%s %d\n", sbl_counter_names[i], atomic_read(&hw->sbl_counters[i]));

	for (i = 0; i < CASS_LMON_NUM_COUNTERS; ++i)
		seq_printf(s, "%s %d\n", lmon_counter_names[i],
				atomic_read(&hw->port->lmon_counters[i]));

	return 0;
}

static int cass_sbl_counters_open(struct inode *inode, struct file *file)
{
	return single_open(file, cass_sbl_counters_show, inode->i_private);
}

static const struct file_operations cass_sbl_counters_fops = {
	.owner = THIS_MODULE,
	.open = cass_sbl_counters_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

void cass_sbl_counters_debugfs_init(struct cass_dev *hw)
{
	debugfs_create_file("counters", 0444, hw->debug_dir, hw, &cass_sbl_counters_fops);
}

void domain_debugfs_create(int vni, int pid, struct cass_dev *hw,
				struct cxi_domain_priv *domain_priv)
{
	char name[30];

	sprintf(name, "%u_%u", vni, pid);
	domain_priv->debug_dir = debugfs_create_dir(name, hw->domain_dir);
	debugfs_create_u32("vni", 0444, domain_priv->debug_dir,
			   &domain_priv->domain.vni);
}

void pt_debugfs_create(int id, struct cxi_pte_priv *pt, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv)
{
	char name[30];
	char path[30];

	sprintf(name, "%u", id);
	pt->debug_dir = debugfs_create_dir(name, hw->pt_dir);
	sprintf(path, "../../../pt/%u", id);
	pt->lni_dir = debugfs_create_symlink(name, lni_priv->pt_dir,
					     path);
}


void lni_debugfs_create(int id, struct cass_dev *hw, struct cxi_lni_priv *lni_priv)
{
	char name[30];

	sprintf(name, "%d", id);
	lni_priv->debug_dir = debugfs_create_dir(name, hw->lni_dir);
	debugfs_create_u32("id", 0444, lni_priv->debug_dir, &lni_priv->lni.id);
	debugfs_create_u32("rgid", 0444, lni_priv->debug_dir, &lni_priv->lni.rgid);
	debugfs_create_u32("pid", 0444, lni_priv->debug_dir, &lni_priv->pid);

	lni_priv->cq_dir = debugfs_create_dir("cq", lni_priv->debug_dir);
	lni_priv->pt_dir = debugfs_create_dir("pt", lni_priv->debug_dir);
	lni_priv->eq_dir = debugfs_create_dir("eq", lni_priv->debug_dir);
	lni_priv->ct_dir = debugfs_create_dir("ct", lni_priv->debug_dir);
	lni_priv->ac_dir = debugfs_create_dir("ac", lni_priv->debug_dir);
}

static int eq_debugfs_info(struct seq_file *s, void *unused)
{
	struct cxi_eq_priv *eq = s->private;

	seq_printf(s, "EQ id: %u\n", eq->eq.eqn);
	seq_puts(s, "event MSI vector: ");
	if (eq->event_msi_irq) {
		const struct cass_irq *irq = eq->event_msi_irq;

		seq_printf(s, "%s\n", irq->irq_name);
	} else {
		seq_puts(s, "none\n");
	}

	seq_puts(s, "status MSI vector: ");
	if (eq->status_msi_irq) {
		const struct cass_irq *irq = eq->status_msi_irq;

		seq_printf(s, "%s\n", irq->irq_name);
	} else {
		seq_puts(s, "none\n");
	}

	seq_printf(s, "slots: %lu\n", eq->queue_size);
	seq_printf(s, "flags: %llx\n", eq->attr.flags);

	return 0;
}

static int debug_eq_open(struct inode *inode, struct file *file)
{
	return single_open(file, eq_debugfs_info, inode->i_private);
}

static const struct file_operations eq_fops = {
	.owner = THIS_MODULE,
	.open = debug_eq_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

void eq_debugfs_create(int id, struct cxi_eq_priv *eq, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv)
{
	char name[30];
	char path[30];

	sprintf(name, "%u", id);
	eq->debug_file = debugfs_create_file(name, 0444, hw->eq_dir,
					     eq, &eq_fops);
	sprintf(path, "../../../eq/%u", id);
	eq->lni_dir = debugfs_create_symlink(name, lni_priv->eq_dir,
						     path);
}

static int cass_dmac_desc_sets_show(struct seq_file *s, void *unused)
{
	struct cass_dev *hw = s->private;
	int i;
	int j;
	u16 count;
	u16 index;
	u16 numused;
	const char *name;
	union c_pi_cfg_dmac_desc desc;
	s64 dst;
	s32 src;
	s32 len;
	s16 rem;

	mutex_lock(&hw->dmac.lock);
	for (i = 0 ; i < DMAC_DESC_SET_COUNT ; ++i) {
		count	= hw->dmac.desc_sets[i].count;
		index	= hw->dmac.desc_sets[i].index;
		numused = hw->dmac.desc_sets[i].numused;
		name    = hw->dmac.desc_sets[i].name;
		seq_printf(s, "desc_set[%02d]: count=%u index=%u numused=%u name='%s'\n",
			   i, count, index, numused,
			   name == NULL ? "" : name);
		if (count != 0)
			for (j = 0; j < count; ++j) {
				cass_read(hw, C_PI_CFG_DMAC_DESC(index + j), &desc,
					  sizeof(desc));
				dst =  desc.qw[0];
				src =  desc.qw[1] & 0xffffffff;
				len = (desc.qw[1] >> 32) & 0x000fffff;
				rem = (desc.qw[1] >> (32 + 20)) & 0x0fff;
				seq_printf(s, "\tdesc[%04u]=0x%03x,0x%05x,0x%08x,0x%016llx %s\n",
					   index + j, rem, len, src, dst,
					   (((dst & 0x7f) == 0) ? "(dst 128-byte aligned)" : ""));
			}
	}
	mutex_unlock(&hw->dmac.lock);
	return 0;
}

static int cass_dmac_desc_sets_open(struct inode *inode, struct file *file)
{
	return single_open(file, cass_dmac_desc_sets_show,
			   inode->i_private);
}

static const struct file_operations cass_dmac_desc_sets_fops = {
	.owner = THIS_MODULE,
	.open = cass_dmac_desc_sets_open,
	.read = seq_read,
	.llseek	 = seq_lseek,
	.release = single_release,
};

void cass_dmac_debugfs_init(struct cass_dev *hw)
{
	debugfs_create_file("dmac-desc-sets", 0444, hw->debug_dir,
			    hw, &cass_dmac_desc_sets_fops);
}

static int tc_cfg_open(struct inode *inode, struct file *file)
{
	return single_open(file, tc_cfg_show, inode->i_private);
}

static const struct file_operations tc_cfg_fops = {
	.owner = THIS_MODULE,
	.open = tc_cfg_open,
	.read = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

void cass_probe_debugfs_init(struct cass_dev *hw)
{
	/* Various debugfs entries */
	hw->debug_dir = debugfs_create_dir(hw->cdev.name, cxi_debug_dir);
	hw->lni_dir = debugfs_create_dir("lni", hw->debug_dir);
	hw->stats_dir = debugfs_create_dir("stats", hw->debug_dir);
	hw->atu_dir = debugfs_create_dir("atu", hw->debug_dir);
	debugfs_create_atomic_t("lni", 0444, hw->stats_dir, &hw->stats.lni);
	debugfs_create_atomic_t("domain", 0444, hw->stats_dir,
				&hw->stats.domain);
	debugfs_create_atomic_t("eq", 0444, hw->stats_dir, &hw->stats.eq);
	debugfs_create_atomic_t("txq", 0444, hw->stats_dir, &hw->stats.txq);
	debugfs_create_atomic_t("tgq", 0444, hw->stats_dir, &hw->stats.tgq);
	debugfs_create_atomic_t("pt", 0444, hw->stats_dir, &hw->stats.pt);
	debugfs_create_atomic_t("ct", 0444, hw->stats_dir, &hw->stats.ct);
	debugfs_create_atomic_t("ac", 0444, hw->stats_dir, &hw->stats.ac);
	debugfs_create_atomic_t("md", 0444, hw->stats_dir, &hw->stats.md);

	hw->domain_dir = debugfs_create_dir("domain", hw->debug_dir);
	hw->eq_dir = debugfs_create_dir("eq", hw->debug_dir);
	hw->cq_dir = debugfs_create_dir("cq", hw->debug_dir);
	hw->pt_dir = debugfs_create_dir("pt", hw->debug_dir);
	hw->ct_dir = debugfs_create_dir("ct", hw->debug_dir);
	hw->ac_dir = debugfs_create_dir("ac", hw->debug_dir);

	debugfs_create_file("tc_cfg", 0444, hw->debug_dir, hw, &tc_cfg_fops);
	debugfs_create_atomic_t("error_inject", 0644, hw->atu_dir,
				&hw->atu_error_inject);
	debugfs_create_atomic_t("odp_requests", 0644, hw->atu_dir,
				&hw->atu_odp_requests);
	debugfs_create_atomic_t("atu_odp_fails", 0644, hw->atu_dir,
				&hw->atu_odp_fails);
	debugfs_create_atomic_t("atu_prb_expired", 0644, hw->atu_dir,
				&hw->atu_prb_expired);

	/* ODP Decouple stats */
	debugfs_create_file("decouple_stats", 0644, hw->atu_dir, hw,
			    &decouple_stats_fops);
	debugfs_create_file("odp_sw_decouple", 0644, hw->atu_dir, hw,
			    &sw_decouple_fops);

	debugfs_create_file("uc_log", 0444, hw->debug_dir, hw, &uc_fops);

	/* setup SBL debugfs interface */
	cass_port_debugfs_init(hw);

	if (cass_version(hw, CASSINI_1))
		cass_sbl_counters_debugfs_init(hw); /* Port counters */

	cass_dmac_debugfs_init(hw);
}

void cq_debugfs_create(int id, struct cxi_cq_priv *cq, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv)
{
	char name[30];
	char path[30];

	sprintf(name, "%u", id);
	cq->debug_dir = debugfs_create_dir(name, hw->cq_dir);
	sprintf(path, "../../../cq/%u", id);
	cq->lni_dir = debugfs_create_symlink(name, lni_priv->cq_dir, path);
	debugfs_create_u32("id", 0444, cq->debug_dir, &cq->cass_cq.idx);
}

void ct_debugfs_setup(int id, struct cxi_ct_priv *ct_priv, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv)
{
	char name[30];
	char path[30];

	sprintf(name, "%u", id);
	ct_priv->debug_dir = debugfs_create_dir(name, hw->ct_dir);
	sprintf(path, "../../../ct/%u", id);
	ct_priv->lni_dir = debugfs_create_symlink(name, lni_priv->ct_dir,
						  path);
}

void ac_debugfs_create(int id, struct cass_ac *cac, struct cass_dev *hw,
		       struct cxi_lni_priv *lni_priv)
{
	char name[30];
	char path[30];

	sprintf(name, "%u", id);
	cac->debug_dir = debugfs_create_dir(name, hw->ac_dir);
	sprintf(path, "../../../ac/%u", id);
	cac->lni_dir = debugfs_create_symlink(name, lni_priv->ac_dir, path);
}
