// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/* Cassini CSR error handlers */

#include <linux/interrupt.h>
#include <linux/types.h>
#include <linux/version.h>

#define CREATE_TRACE_POINTS

#include "cass_core.h"
#include "cassini_error_defs.h"

#include "cass_trace.h"

/* Prepare a netlink message, queue it, and trigger the tasklet to
 * send it. The message cannot be sent from this function directly
 * because it is called from the interrupt handler.
 */
static void send_nl_msg(const struct cxi_dev *cdev, const struct csr_info *info,
			const struct flg_err_info *flg_info, unsigned int bitn)
{
	struct sk_buff *msg;
	size_t msg_size;
	void *hdr;
	int rc;

	msg_size = nla_total_size(2 * nla_attr_size(sizeof(u32)) +
				  nla_attr_size(sizeof(u8)));
	msg = genlmsg_new(msg_size, GFP_ATOMIC);
	if (!msg) {
		cxidev_err(cdev, "Can't allocate GENL skb\n");
		return;
	}

	hdr = genlmsg_put(msg, 0, 0, &cxierr_genl_family,
			  0, CXIERR_ERR_MSG);
	if (!hdr) {
		cxidev_err(cdev, "genl put error");
		goto fail;
	}

	rc = nla_put_u32(msg, CXIERR_GENL_ATTR_CSR_FLG, info->flg);
	if (rc) {
		cxidev_err(cdev, "nla put str error %d\n", rc);
		goto fail;
	}

	rc = nla_put_u8(msg, CXIERR_GENL_ATTR_BIT, bitn);
	if (rc) {
		cxidev_err(cdev, "nla put str error %d\n", rc);
		goto fail;
	}

	rc = nla_put_u32(msg, CXIERR_GENL_ATTR_DEV_NUM, cdev->cxi_num);
	if (rc) {
		cxidev_err(cdev, "nla put str error %d\n", rc);
		goto fail;
	}

	genlmsg_end(msg, hdr);

	genlmsg_multicast(&cxierr_genl_family, msg, 0, 0, GFP_ATOMIC);

	return;

fail:
	genlmsg_cancel(msg, hdr);
	nlmsg_free(msg);
}

/* Collect the information needed to analyze an error flag.
 *
 * Some fields in error CSRs are arrays (PFC_FIFO_OFLW in
 * C_HNI_ERR_FLG for instance). bit_index is their index.
 */
static void collect_flg_info(struct cass_dev *hw,
			     const struct flg_err_info *flg_info,
			     unsigned int bit_index)
{
	int i;
	int idx = 0;

	hw->err_info_count = 0;

	for (idx = 0;
	     idx < C_MAX_CSR_ERR_INFO && flg_info->err_info[idx].offset != 0;
	     idx++) {
		struct err_info *info = &hw->err_info[idx];
		void __iomem *base =
				cass_csr(hw, flg_info->err_info[idx].offset);
		int count = flg_info->err_info[idx].size / 8;

		/* The ERR_INFO CSRs are up to 5 64-bits values, but
		 * that may change.
		 */
		if (cxidev_WARN_ONCE(&hw->cdev, count > 5,
				     "err_info CSR %x is too big: %d\n",
				     flg_info->err_info[idx].offset, count)) {
			count = 5;
		}

		for (i = 0; i < count; i++, base += sizeof(u64))
			info->data[i] = readq(base);

		info->count = count;

		hw->err_info_count++;
	}

	hw->err_cntrs_count = 0;

	for (idx = 0;
	     idx < C_MAX_CSR_CNTRS && flg_info->cntr[idx].offset != 0;
	     idx++) {
		void __iomem *base;

		if (flg_info->cntr[idx].index == C_ERR_INDEX_NONE)
			base = cass_csr(hw, flg_info->cntr[idx].offset);
		else
			base = cass_csr(hw, flg_info->cntr[idx].offset +
					bit_index * sizeof(u64));

		hw->cntrs_val[idx] = readq(base);
		hw->err_cntrs_count++;
	}
}

/*
 * eventually move this logging macro  as an addition or a replacement
 * for	_cxi_log() in cass_core.h.
 */
#define _dump_flg_info_logger(loglevel, hw, fmt, ...)			\
	do {								\
		switch (loglevel) {					\
		case LOGLEVEL_EMERG:					\
			dev_emerg(&(hw)->cdev.pdev->dev,		\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_ALERT:					\
			dev_alert(&(hw)->cdev.pdev->dev,		\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_CRIT:					\
			dev_crit(&(hw)->cdev.pdev->dev,			\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_ERR:					\
			dev_err(&(hw)->cdev.pdev->dev,			\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_WARNING:					\
			dev_warn(&(hw)->cdev.pdev->dev,			\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_NOTICE:					\
			dev_notice(&(hw)->cdev.pdev->dev,		\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_INFO:					\
			dev_info(&(hw)->cdev.pdev->dev,			\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		case LOGLEVEL_DEBUG:					\
			dev_dbg(&(hw)->cdev.pdev->dev,			\
				"%s[%s]: " fmt,				\
				(hw)->cdev.name, (hw)->cdev.eth_name,	\
				##__VA_ARGS__);				\
			break;						\
		}							\
	} while (0)

/* Emit the collected information for a given error flag to the kernel
 * log.
 */
static void dump_flg_info(struct cass_dev *hw,
			  unsigned int irq, bool is_ext,
			  const struct csr_info *info,
			  const struct flg_err_info *flg_info,
			  unsigned int bitn)
{
	int i;
	char *first_err;
	int log_level = 0;
	static const char * const error_event_names[] = {
		"C_EC_SFTWR",
		"C_EC_INFO",
		"C_EC_TRS_NS",
		"C_EC_TRS_S",
		"C_EC_TRNSNT_NS",
		"C_EC_TRNSNT_S",
		"C_EC_BADCON_NS",
		"C_EC_BADCON_S",
		"C_EC_DEGRD_NS",
		"C_EC_COR",
		"C_EC_UNCOR_NS",
		"C_EC_UNCOR_S",
		"C_EC_CRIT",
	};
	const char *errname;

	if (test_bit(bitn, hw->first_flg)) {
		first_err = hw->err_info_str;
		sprintf(hw->err_info_str, "(was first error at %lu:%09lu)",
			(unsigned long)hw->first_flg_ts.seconds,
			(unsigned long)hw->first_flg_ts.nanoseconds);
	} else {
		first_err = "";
	}

	if (test_bit(bitn,
		     hw->sfs_err_flg[irq][is_ext ? 1 : 0].no_print_mask.mask))
		return;

	/* Find the log level for that error. Not all errors reported
	 * by the device are bad.
	 */
	switch (flg_info->ec) {
	case C_EC_SFTWR:
	case C_EC_INFO:
	case C_EC_TRS_NS:
	case C_EC_TRS_S:
	case C_EC_TRNSNT_NS:
	case C_EC_TRNSNT_S:
		errname = error_event_names[flg_info->ec];
		log_level = LOGLEVEL_INFO;
		break;

	case C_EC_BADCON_NS:
	case C_EC_BADCON_S:
	case C_EC_COR:
		errname = error_event_names[flg_info->ec];
		log_level = LOGLEVEL_WARNING;
		break;

	case C_EC_DEGRD_NS:
		errname = error_event_names[flg_info->ec];
		log_level = LOGLEVEL_ERR;
		break;

	case C_EC_UNCOR_NS:
	case C_EC_UNCOR_S:
		errname = error_event_names[flg_info->ec];
		if (strcmp("llr_eopb", flg_info->bit_name) == 0)
			log_level = LOGLEVEL_INFO;
		else
			log_level = LOGLEVEL_CRIT;
		break;

	case C_EC_CRIT:
		errname = error_event_names[flg_info->ec];
		log_level = LOGLEVEL_CRIT;
		break;

	default:
		errname = "unrecognized c_error_class value";
		log_level = LOGLEVEL_CRIT;
		break;
	}

	_dump_flg_info_logger(log_level, hw, "%s: %s error: %s (%d) %s\n",
			      errname, info->csr_name, flg_info->bit_name,
			      bitn, first_err);

	if (hw->err_info_count) {
		int idx;

		for (idx = 0; idx < hw->err_info_count; idx++) {
			struct err_info *info = &hw->err_info[idx];
			char *p = hw->err_info_str;

			*p = '\0';

			for (i = 0; i < info->count; i++)
				p += sprintf(p, " %016llx", info->data[i]);

			_dump_flg_info_logger(log_level, hw, "%s: %s%s\n",
					      errname,
					      flg_info->err_info[idx].name,
					      hw->err_info_str);
		}
	}

	if (hw->err_cntrs_count) {
		int idx;

		for (idx = 0; idx < hw->err_cntrs_count; idx++)
			_dump_flg_info_logger(log_level, hw, "%s: %s: %llu\n",
					      errname,
					      flg_info->cntr[idx].name,
					      hw->cntrs_val[idx]);
	}
}

static void trace_err_info(struct cass_dev *hw,
			   const struct csr_info *info,
			   const struct flg_err_info *flg_info,
			   unsigned int bitn)
{
	unsigned long seconds;
	unsigned long nanoseconds;

	if (test_bit(bitn, hw->first_flg)) {
		seconds = hw->first_flg_ts.seconds;
		nanoseconds = hw->first_flg_ts.nanoseconds;
	} else {
		seconds = 0;
		nanoseconds = 0;
	}

	trace_cass_err(info->flg, bitn, seconds, nanoseconds,
		       flg_info, hw->err_info, hw->cntrs_val);
}

/* Check whether some registered handlers are interested in that bit. */
static void
process_registered_handlers(struct cass_dev *hw, unsigned int irq, bool is_ext,
			    unsigned int bitn)
{
	struct cxi_reg_err_flg *err_flg;

	if (list_empty(&hw->err_flg_list))
		return;

	mutex_lock(&hw->err_flg_mutex);
	list_for_each_entry(err_flg, &hw->err_flg_list, list) {
		if ((irq == err_flg->irq) &&
		    (is_ext == err_flg->is_ext) &&
		    test_bit(bitn, err_flg->err_flags.mask))
			err_flg->cb(hw, irq, is_ext, bitn);
	}
	mutex_unlock(&hw->err_flg_mutex);
}

/* Called for each bit in an ERR_FLG (or EXT_ERR_FLG) CSR.
 *
 * irq is which interrupt was triggered (C_PI_IPD_IRQA_MSIX_INT to
 *   C_OXE_IRQB_MSIX_INT.)
 * is_ext indicates whether this is the normal CSR or its EXT part.
 * bitn is the bit number (0 to 63) that has been set in that CSR.
 */
static void process_err_flg(struct cass_dev *hw, unsigned int irq, bool is_ext,
			    const struct csr_info *info, unsigned int bitn)
{
	const struct flg_err_info *flg_info = info->flg_info;

	/* Both for_each_set_bit and flg_info->bit go
	 * up. Match the range.
	 */
	while (flg_info->bit + flg_info->num_bits <= bitn)
		flg_info++;

	if (flg_info->bit <= bitn) {
		collect_flg_info(hw, flg_info, bitn - flg_info->bit);

		/* TODO?: once the API stabilizes, it might be
		 * possible to have these 3 handlers as clients of
		 * process_registered_handlers().
		 */

		dump_flg_info(hw, irq, is_ext, info, flg_info, bitn);
		send_nl_msg(&hw->cdev, info, flg_info, bitn);
		trace_err_info(hw, info, flg_info, bitn);

		process_registered_handlers(hw, irq, is_ext, bitn);
	} else {
		cxidev_info(&hw->cdev, "%s error: unhandled bit %u\n",
			    info->csr_name, bitn);
		flg_info = NULL;
	}

	if (!__ratelimit(&hw->err_rl[irq][bitn])) {
		struct sfs_err_flg *sfs = &hw->sfs_err_flg[irq][is_ext ? 1 : 0];

		if (!test_bit(bitn, sfs->no_auto_mask.mask)) {
			union err_flags bitmask = {};

			cxidev_info(&hw->cdev,
				    "Too many interrupts for %s.%s. Masking.\n",
				    info->csr_name,
				    flg_info ? flg_info->bit_name : "unknown");

			set_bit(bitn, bitmask.mask);
			cxi_disable_hw_errors(hw, irq, is_ext, bitmask.mask);
		}
	}
}

static ssize_t mask_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
	struct sfs_err_flg *sfs = container_of(kobj, struct sfs_err_flg, kobj);

	return scnprintf(buf, PAGE_SIZE, "%*pb\n",
			 sfs->bitlen, sfs->err_irq_mask);
}

static ssize_t no_print_mask_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	struct sfs_err_flg *sfs = container_of(kobj, struct sfs_err_flg, kobj);

	return scnprintf(buf, PAGE_SIZE, "%*pb\n",
			 sfs->bitlen, sfs->no_print_mask.mask);
}

static ssize_t no_print_mask_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t len)
{
	struct sfs_err_flg *sfs = container_of(kobj, struct sfs_err_flg, kobj);
	DECLARE_BITMAP(new, MAX_ERR_FLAG_BITLEN);
	int ret;

	ret = bitmap_parse(buf, len, new, sfs->bitlen);
	if (ret)
		return ret;

	bitmap_copy(sfs->no_print_mask.mask, new, sfs->bitlen);

	return len;
}

static ssize_t no_auto_mask_show(struct kobject *kobj,
				 struct kobj_attribute *attr, char *buf)
{
	struct sfs_err_flg *sfs = container_of(kobj, struct sfs_err_flg, kobj);

	return scnprintf(buf, PAGE_SIZE, "%*pb\n",
			 sfs->bitlen, sfs->no_auto_mask.mask);
}

static ssize_t no_auto_mask_store(struct kobject *kobj,
				  struct kobj_attribute *attr,
				  const char *buf, size_t len)
{
	struct sfs_err_flg *sfs = container_of(kobj, struct sfs_err_flg, kobj);
	DECLARE_BITMAP(new, MAX_ERR_FLAG_BITLEN);
	int ret;

	ret = bitmap_parse(buf, len, new, sfs->bitlen);
	if (ret)
		return ret;

	bitmap_copy(sfs->no_auto_mask.mask, new, sfs->bitlen);

	return len;
}

#define CSR_ERROR_ATTR_RO(_name) \
	static struct kobj_attribute _name##_attr = __ATTR_RO(_name)

#define CSR_ERROR_ATTR_RW(_name)					\
	static struct kobj_attribute _name##_attr = __ATTR_RW(_name)

CSR_ERROR_ATTR_RO(mask);
CSR_ERROR_ATTR_RW(no_print_mask);
CSR_ERROR_ATTR_RW(no_auto_mask);

static struct attribute *irq_attrs[] = {
	&mask_attr.attr,
	&no_print_mask_attr.attr,
	&no_auto_mask_attr.attr,
	NULL
};
ATTRIBUTE_GROUPS(irq);

static struct kobj_type irq_kobj_type = {
	.sysfs_ops  = &kobj_sysfs_ops,
	.default_groups = irq_groups,
};

/* Define which error CSRs are connected to which IRQs.
 * pf1 = prefix for the main CSR (ie. C1_IXE_ERR_FLG)
 * pf2 = prefix for the sub-CSRs (ie. C_IXE_ERR_FIRST_FLG)
 * The main can be different between C1/C2, but the other don't have
 * the named fields in them, so they are identical. This is messy.
 */
#define CSRERR1(pf1, pf2, name, irq, namelo) {				\
	.csr_name = #pf1 "_" #name, .flg = pf1##_##name##_ERR_FLG,	\
	.mask = pf2##_##name##_ERR_IRQ##irq##_MSK,			\
	.clr = pf2##_##name##_ERR_CLR, .flg_info = namelo##_err_info,	\
	.first_flg = pf2##_##name##_ERR_FIRST_FLG,			\
	.first_flg_ts = pf2##_##name##_ERR_FIRST_FLG_TS,		\
	.csr_name_lo = #namelo,						\
	.bitlen = (pf1##_##name##_ERR_FLG_SIZE * BITS_PER_BYTE)		\
}

#define CSRERR2(pf1, pf2, name, irq, namelo) {				\
	.csr_name = #pf1 "_" #name, .flg = pf1##_##name##_ERR_FLG,	\
	.mask = pf2##_##name##_ERR_IRQ##irq##_MSK,			\
	.clr = pf2##_##name##_ERR_CLR, .flg_info = c2_##namelo##_err_info, \
	.first_flg = pf2##_##name##_ERR_FIRST_FLG,			\
	.first_flg_ts = pf2##_##name##_ERR_FIRST_FLG_TS,		\
	.csr_name_lo = #namelo,						\
	.bitlen = (pf1##_##name##_ERR_FLG_SIZE * BITS_PER_BYTE)		\
}

#define CSRERRSS2(pf1, pf2, name, irq, namelo) {				\
	.csr_name = #pf1 "_" #name, .flg = pf1##_##name##_ERR_FLG,	\
	.mask = pf2##_##name##_ERR_IRQ##irq##_MSK,			\
	.clr = pf2##_##name##_ERR_CLR, .flg_info = ss2_##namelo##_err_info, \
	.first_flg = pf2##_##name##_ERR_FIRST_FLG,			\
	.first_flg_ts = pf2##_##name##_ERR_FIRST_FLG_TS,		\
	.csr_name_lo = #namelo,						\
	.bitlen = (pf1##_##name##_ERR_FLG_SIZE * BITS_PER_BYTE)		\
}

static const struct csr_info c1_handlers[NUM_ERR_INTS][2] = {
	{ CSRERR1(C, C, PI_IPD, A, pi_ipd),
	  CSRERR1(C, C, PI_IPD_EXT, A, pi_ipd_ext) },
	{ CSRERR1(C, C, PI, A, pi), CSRERR1(C, C, PI_EXT, A, pi_ext) },
	{ CSRERR1(C, C, MB, A, mb), CSRERR1(C, C, MB_EXT, A, mb_ext) },
	{ CSRERR1(C, C, CQ, A, cq), CSRERR1(C, C, CQ_EXT, A, cq_ext) },
	{ CSRERR1(C1, C, PCT, A, pct), CSRERR1(C, C, PCT_EXT, A, pct_ext) },
	{ CSRERR1(C1, C, HNI, A, hni), },
	{ CSRERR1(C1, C1, HNI_PML, A, hni_pml) },
	{ CSRERR1(C, C, RMU, A, rmu) },
	{ CSRERR1(C, C, IXE, A, ixe), CSRERR1(C, C, IXE_EXT, A, ixe_ext) },
	{ CSRERR1(C, C, ATU, A, atu), CSRERR1(C, C, ATU_EXT, A, atu_ext) },
	{ CSRERR1(C, C, EE, A, ee), CSRERR1(C, C, EE_EXT, A, ee_ext) },
	{ CSRERR1(C, C, PARBS, A, parbs),
	  CSRERR1(C, C, PARBS_EXT, A, parbs_ext) },
	{ CSRERR1(C, C, LPE, A, lpe) },
	{ CSRERR1(C, C, MST, A, mst) },
	{ CSRERR1(C, C, OXE, A, oxe) },
	{ CSRERR1(C, C, PI_IPD, B, pi_ipd),
	  CSRERR1(C, C, PI_IPD_EXT, B, pi_ipd_ext) },
	{ CSRERR1(C, C, PI, B, pi), CSRERR1(C, C, PI_EXT, B, pi_ext) },
	{ CSRERR1(C, C, MB, B, mb), CSRERR1(C, C, MB_EXT, B, mb_ext) },
	{ CSRERR1(C, C, CQ, B, cq), CSRERR1(C, C, CQ_EXT, B, cq_ext) },
	{ CSRERR1(C1, C, PCT, B, pct), CSRERR1(C, C, PCT_EXT, B, pct_ext) },
	{ CSRERR1(C1, C, HNI, B, hni) },
	{ CSRERR1(C1, C1, HNI_PML, B, hni_pml) },
	{ CSRERR1(C, C, RMU, B, rmu) },
	{ CSRERR1(C, C, IXE, B, ixe), CSRERR1(C, C, IXE_EXT, B, ixe_ext) },
	{ CSRERR1(C, C, ATU, B, atu), CSRERR1(C, C, ATU_EXT, B, atu_ext) },
	{ CSRERR1(C, C, EE, B, ee), CSRERR1(C, C, EE_EXT, B, ee_ext) },
	{ CSRERR1(C, C, PARBS, B, parbs),
	  CSRERR1(C, C, PARBS_EXT, B, parbs_ext) },
	{ CSRERR1(C, C, LPE, B, lpe) },
	{ CSRERR1(C, C, MST, B, mst) },
	{ CSRERR1(C, C, OXE, B, oxe) },
};

static const struct csr_info c2_handlers[NUM_ERR_INTS][2] = {
	{ CSRERR2(C, C, PI_IPD, A, pi_ipd),
	  CSRERR2(C, C, PI_IPD_EXT, A, pi_ipd_ext) },
	{ CSRERR2(C, C, PI, A, pi), CSRERR2(C, C, PI_EXT, A, pi_ext) },
	{ CSRERR2(C, C, MB, A, mb), CSRERR2(C, C, MB_EXT, A, mb_ext) },
	{ CSRERR2(C, C, CQ, A, cq), CSRERR2(C, C, CQ_EXT, A, cq_ext) },
	{ CSRERR2(C2, C, PCT, A, pct), CSRERR2(C, C, PCT_EXT, A, pct_ext) },
	{ CSRERR2(C2, C, HNI, A, hni), CSRERR2(C2, C2, HNI_EXT, A, hni_ext) },
	{ CSRERRSS2(SS2, SS2, PORT_PML, A, port_pml) },
	{ CSRERR2(C, C, RMU, A, rmu) },
	{ CSRERR2(C, C, IXE, A, ixe), CSRERR2(C, C, IXE_EXT, A, ixe_ext) },
	{ CSRERR2(C, C, ATU, A, atu), CSRERR2(C, C, ATU_EXT, A, atu_ext) },
	{ CSRERR2(C, C, EE, A, ee), CSRERR2(C, C, EE_EXT, A, ee_ext) },
	{ CSRERR2(C, C, PARBS, A, parbs),
	  CSRERR2(C, C, PARBS_EXT, A, parbs_ext) },
	{ CSRERR2(C, C, LPE, A, lpe) },
	{ CSRERR2(C, C, MST, A, mst) },
	{ CSRERR2(C, C, OXE, A, oxe) },
	{ CSRERR2(C, C, PI_IPD, B, pi_ipd),
	  CSRERR2(C, C, PI_IPD_EXT, B, pi_ipd_ext) },
	{ CSRERR2(C, C, PI, B, pi), CSRERR2(C, C, PI_EXT, B, pi_ext) },
	{ CSRERR2(C, C, MB, B, mb), CSRERR2(C, C, MB_EXT, B, mb_ext) },
	{ CSRERR2(C, C, CQ, B, cq), CSRERR2(C, C, CQ_EXT, B, cq_ext) },
	{ CSRERR2(C2, C, PCT, B, pct), CSRERR2(C, C, PCT_EXT, B, pct_ext) },
	{ CSRERR2(C2, C, HNI, B, hni), CSRERR2(C2, C2, HNI_EXT, B, hni_ext) },
	{ CSRERRSS2(SS2, SS2, PORT_PML, B, port_pml) },
	{ CSRERR2(C, C, RMU, B, rmu) },
	{ CSRERR2(C, C, IXE, B, ixe), CSRERR2(C, C, IXE_EXT, B, ixe_ext) },
	{ CSRERR2(C, C, ATU, B, atu), CSRERR2(C, C, ATU_EXT, B, atu_ext) },
	{ CSRERR2(C, C, EE, B, ee), CSRERR2(C, C, EE_EXT, B, ee_ext) },
	{ CSRERR2(C, C, PARBS, B, parbs),
	  CSRERR2(C, C, PARBS_EXT, B, parbs_ext) },
	{ CSRERR2(C, C, LPE, B, lpe) },
	{ CSRERR2(C, C, MST, B, mst) },
	{ CSRERR2(C, C, OXE, B, oxe) },
};

/* Whether the extended error flag exists */
#define INFO(irq, idx) ((*hw->err_handlers)[irq][idx])
#define HAS_EXT(irq) (INFO(irq, 1).csr_name)

static void process_err_irq(struct cass_dev *hw, int irq)
{
	union err_flags err_flag[2];
	DECLARE_BITMAP(mask, MAX_ERR_FLAG_BITLEN);
	int i;
	unsigned int bitn;

	while (true) {
		bool empty[2] = { true, true };

		/* Read the normal CSR and the extended CSR if it
		 * exists. Bits that are not valid must be ignored.
		 */
		for (i = 0; i < 2; i++) {
			const struct csr_info *info = &INFO(irq, i);

			if (i == 0 || HAS_EXT(irq)) {
				cass_read(hw, info->flg,
					  &err_flag[i].mask, info->bitlen / BITS_PER_BYTE);

				cass_read(hw, info->mask, mask, info->bitlen / BITS_PER_BYTE);
				bitmap_andnot(err_flag[i].mask, err_flag[i].mask, mask,
					      info->bitlen);

				if (!bitmap_empty(err_flag[i].mask, info->bitlen))
					empty[i] = false;
			} else {
				bitmap_zero(err_flag[i].mask, info->bitlen);
			}
		}

		/* A new error might have been raised since being
		 * cleared in the previous loop. We must ensure the
		 * error flags are 0 before leaving.
		 */
		if (empty[0] && empty[1])
			return;

		for (i = 0; i < 2; i++) {
			const struct csr_info *info = &INFO(irq, i);

			if (empty[i])
				continue;

			cass_read(hw, info->first_flg,
				  hw->first_flg, info->bitlen / BITS_PER_BYTE);

			cass_read(hw, info->first_flg_ts,
				  &hw->first_flg_ts, sizeof(hw->first_flg_ts));

			for_each_set_bit(bitn, err_flag[i].mask, info->bitlen)
				process_err_flg(hw, irq, i == 1, info, bitn);

			/* Clear the error bits */
			cass_write(hw, info->clr, &err_flag[i],
				   info->bitlen / BITS_PER_BYTE);
		}
	}
}

/* Worker to process all raised interrupts */
static void error_irq_worker(struct work_struct *work)
{
	struct cass_dev *hw = container_of(work, struct cass_dev, err_irq_work);
	int irq;
	unsigned long irqs_raised;

	irqs_raised = xchg(&hw->err_irq_raised, 0);

	for_each_set_bit(irq, &irqs_raised, NUM_ERR_INTS)
		process_err_irq(hw, irq);
}

/* Main interrupt handler for error registers */
static irqreturn_t error_irq_cb(int vec, void *context)
{
	struct cass_dev *hw = context;
	int irq;

	/* Find the irq from the vector. This could be more efficient,
	 * but these interrupts should be fairly rare.
	 */
	for (irq = 0; irq < NUM_ERR_INTS; irq++) {
		if (hw->err_irq_vecs[irq] == vec)
			break;
	}

	if (irq == NUM_ERR_INTS) {
		cxidev_warn_once(&hw->cdev, "Error IRQ not found for vector %u\n",
				 vec);
		return IRQ_HANDLED;
	}

	set_bit(irq, &hw->err_irq_raised);
	queue_work(hw->err_irq_wq, &hw->err_irq_work);

	return IRQ_HANDLED;
}

static int cxierr_msg(struct sk_buff *skb, struct genl_info *info)
{
	return 0;
}

static const struct nla_policy cxierr_genl_policy[CXIERR_GENL_ATTR_MAX] = {
	[CXIERR_GENL_ATTR_DEV_NUM] = { .type = NLA_U32 },
	[CXIERR_GENL_ATTR_CSR_FLG] = { .type = NLA_U32 },
	[CXIERR_GENL_ATTR_BIT] = { .type = NLA_U8 },
};

static const struct genl_ops cxierr_genl_ops[] = {
	{
		.cmd = CXIERR_ERR_MSG,
#if KERNEL_VERSION(4, 18, 0) > LINUX_VERSION_CODE
		.policy = cxierr_genl_policy,
#endif
		.doit = cxierr_msg,
		.flags = GENL_ADMIN_PERM,
	},
};

static const struct genl_multicast_group cxierr_mcast_grps[] = {
	{ .name = CXIERR_GENL_MCAST_GROUP_NAME, },
};

struct genl_family cxierr_genl_family __ro_after_init = {
	.module = THIS_MODULE,
	.name = CXIERR_GENL_FAMILY_NAME,
	.version = CXIERR_GENL_VERSION,
	.maxattr = CXIERR_GENL_ATTR_MAX - 1,
#if KERNEL_VERSION(4, 18, 0) <= LINUX_VERSION_CODE
	.policy = cxierr_genl_policy,
#endif
	.ops = cxierr_genl_ops,
	.n_ops = ARRAY_SIZE(cxierr_genl_ops),
	.mcgrps = cxierr_mcast_grps,
	.n_mcgrps = ARRAY_SIZE(cxierr_mcast_grps),
};

static void free_err_irqs(struct cass_dev *hw)
{
	static const u64 mask;
	int irq;

	for (irq = 0; irq < NUM_ERR_INTS; irq++) {
		int vec = hw->err_irq_vecs[irq];

		if (vec < 0)
			continue;

		/* Reset their masks */
		cass_write(hw, INFO(irq, 0).mask, &mask, sizeof(mask));
		if (HAS_EXT(irq))
			cass_write(hw, INFO(irq, 1).mask, &mask, sizeof(mask));

		free_irq(vec, hw);

		hw->err_irq_vecs[irq] = -1;
	}
}

/**
 * cxi_register_hw_errors() - register a callback for error bits
 *
 * When an error interrupt is generated, one or two CSR (ERR_FLG) have
 * information. This allows a caller to be informed, through a
 * callback, when some of these bits are set.
 *
 * @hw: the device
 * @reg_err_flg: caller-initialized parameters (bitmask, callback, ...)
 */
void cxi_register_hw_errors(struct cass_dev *hw,
			    struct cxi_reg_err_flg *reg_err_flg)
{
	mutex_lock(&hw->err_flg_mutex);
	list_add_tail(&reg_err_flg->list, &hw->err_flg_list);
	mutex_unlock(&hw->err_flg_mutex);
}

/**
 * cxi_unregister_hw_errors() - un-register a callback for error bits
 *
 * @hw: the device
 * @reg_err_flg: same structure used to register
 */
void cxi_unregister_hw_errors(struct cass_dev *hw,
			      struct cxi_reg_err_flg *reg_err_flg)
{
	mutex_lock(&hw->err_flg_mutex);
	list_del(&reg_err_flg->list);
	mutex_unlock(&hw->err_flg_mutex);
}

/**
 * cxi_enable_hw_errors() - enable interrupts for some error bits
 *
 * @hw: the device
 * @irq: the IRQ/BLOCK (eg. C_HNI_PML_IRQA_MSIX_INT)
 * @is_ext: whether or not the error flag is in the EXT_FLG CSR
 * @bitmask: bitmask of interrupts to enable
 */
void cxi_enable_hw_errors(struct cass_dev *hw, unsigned int irq, bool is_ext,
			  const unsigned long *bitmask)
{
	unsigned int idx = is_ext ? 1 : 0;
	const struct csr_info *info = &INFO(irq, idx);
	struct sfs_err_flg *sfs = &hw->sfs_err_flg[irq][idx];

	spin_lock_bh(&hw->sfs_err_flg_lock);
	bitmap_andnot(sfs->err_irq_mask, sfs->err_irq_mask,
		      bitmask, sfs->bitlen);
	cass_write(hw, info->mask, &sfs->err_irq_mask,
		   sfs->bitlen / BITS_PER_BYTE);
	spin_unlock_bh(&hw->sfs_err_flg_lock);
}

/**
 * cxi_disable_hw_errors() - disable interrupts for some error bits
 *
 * @hw: the device
 * @irq: the IRQ/BLOCK (eg. C_HNI_PML_IRQA_MSIX_INT)
 * @is_ext: whether or not the error flag is in the EXT_FLG CSR
 * @bitmask: bitmask of interrupts to disable
 */
void cxi_disable_hw_errors(struct cass_dev *hw, unsigned int irq, bool is_ext,
			   const unsigned long *bitmask)
{
	unsigned int idx = is_ext ? 1 : 0;
	const struct csr_info *info = &INFO(irq, idx);
	struct sfs_err_flg *sfs = &hw->sfs_err_flg[irq][idx];

	spin_lock_bh(&hw->sfs_err_flg_lock);
	bitmap_or(sfs->err_irq_mask, sfs->err_irq_mask, bitmask, sfs->bitlen);
	cass_write(hw, info->mask, &sfs->err_irq_mask,
		   sfs->bitlen / BITS_PER_BYTE);
	spin_unlock_bh(&hw->sfs_err_flg_lock);
}

static void alloc_kobjs_and_hw_errors(struct cass_dev *hw)
{
	struct sfs_err_flg *sfs;
	DECLARE_BITMAP(bitmask, MAX_ERR_FLAG_BITLEN);
	int irq = 0;
	int rc;
	int i;

	bitmap_fill(bitmask, MAX_ERR_FLAG_BITLEN);

	/* Only register with IrqA. Unmask all IrqA and mask all IrqB
	 * interrupts.
	 */
	for (irq = 0; irq < NUM_ERR_INTS; irq++) {
		for (i = 0; i < 2; i++) {
			struct kobject *dkobj = hw->err_flgs_dir_kobj[0];
			const struct csr_info *info = &INFO(irq, i);

			if (info->csr_name == NULL)
				continue;

			sfs = &hw->sfs_err_flg[irq][i];

			/* Clear all existing error conditions */
			cass_write(hw, info->clr, &bitmask,
				   sfs->bitlen / BITS_PER_BYTE);

			if (irq < NUM_ERR_INTS / 2) {
				/* IrqA */
				rc = kobject_init_and_add(&sfs->kobj,
							  &irq_kobj_type,
							  dkobj,
							  "%s",
							  info->csr_name_lo);
				if (rc)
					cxidev_warn(&hw->cdev,
						    "Can't add kobject for error CSR %s\n",
						    info->csr_name);

				cxi_enable_hw_errors(hw, irq, i, bitmask);
			} else {
				/* IrqB */
				cxi_disable_hw_errors(hw, irq, i, bitmask);
			}
		}
	}
}

static void free_kobjs_and_hw_errors(struct cass_dev *hw)
{
	struct sfs_err_flg *sfs;
	DECLARE_BITMAP(bitmask, MAX_ERR_FLAG_BITLEN);
	int irq = 0;
	int i;

	bitmap_fill(bitmask, MAX_ERR_FLAG_BITLEN);

	for (irq = 0; irq < NUM_ERR_INTS; irq++) {
		for (i = 0; i < 2; i++) {
			const struct csr_info *info = &INFO(irq, i);

			if (info->csr_name == NULL)
				continue;

			sfs = &hw->sfs_err_flg[irq][i];

			if (irq < NUM_ERR_INTS / 2) {
				/* IrqA */
				kobject_put(&sfs->kobj);
			}

			cxi_disable_hw_errors(hw, irq, i, bitmask);
		}
	}
}

int register_error_handlers(struct cass_dev *hw)
{
	struct sfs_err_flg *sfs;
	union err_flags bitmask;
	int irq = 0;
	int rc;
	int i;
	int j;

	if (cass_version(hw, CASSINI_1))
		hw->err_handlers = &c1_handlers;
	else
		hw->err_handlers = &c2_handlers;

	for (irq = 0; irq < NUM_ERR_INTS; irq++)
		hw->err_irq_vecs[irq] = -1;

	for (i = 0; i < NUM_ERR_INTS; i++) {
		for (j = 0; j < MAX_ERR_FLAG_BITLEN; j++) {
			ratelimit_default_init(&hw->err_rl[i][j]);
			ratelimit_set_flags(&hw->err_rl[i][j],
					    RATELIMIT_MSG_ON_RELEASE);
		}
	}

	for (irq = 0; irq < NUM_ERR_INTS; irq++) {
		for (i = 0; i < 2; i++) {
			const struct csr_info *info = &INFO(irq, i);
			struct sfs_err_flg *sfs = &hw->sfs_err_flg[irq][i];

			if (info->csr_name == NULL)
				continue;

			sfs->bitlen = info->bitlen;
		}
	}

	hw->err_flgs_dir_kobj[0] =
		kobject_create_and_add("err_flgs_irqa",
				       &hw->cdev.pdev->dev.kobj);
	if (hw->err_flgs_dir_kobj[0] == NULL)
		return -ENOMEM;

	alloc_kobjs_and_hw_errors(hw);

	/* Hardcode some options. */

	/* PI.UC_ATTENTION[] should not be printed, and never masked. */
	sfs = &hw->sfs_err_flg[C_PI_IRQA_MSIX_INT][0];
	sfs->no_print_mask.pi.uc_attention = 0b11;
	sfs->no_auto_mask.pi.uc_attention = 0b11;

	/* Mask some PCIe errors that are monitored by the driver's
	 * PCIe monitor, and other errors we don't want to see.
	 */
	sfs = &hw->sfs_err_flg[C_PI_IPD_IRQA_MSIX_INT][1];
	bitmap_zero(bitmask.mask, sfs->bitlen);
	bitmask.pi_ipd_ext.ip_cfg_bad_tlp_err_sts = 1;
	bitmask.pi_ipd_ext.ip_cfg_bad_dllp_err_sts = 1;
	bitmask.pi_ipd_ext.ip_cfg_replay_timer_timeout_err_sts = 1;
	bitmask.pi_ipd_ext.ip_cfg_replay_number_rollover_err_sts = 1;
	bitmask.pi_ipd_ext.pri_rbyp_abort = 1;
	bitmap_copy(sfs->no_print_mask.mask, bitmask.mask, sfs->bitlen);
	cxi_disable_hw_errors(hw, C_PI_IPD_IRQA_MSIX_INT, true, bitmask.mask);

	sfs = &hw->sfs_err_flg[C_PI_IRQA_MSIX_INT][1];
	sfs->no_print_mask.pi_ext.pri_rarb_xlat_rbyp_abort_error = 1;

	/* ATU.PRB_EXPIRED should not be printed, and never
	 * masked. Ignore some other.
	 */
	sfs = &hw->sfs_err_flg[C_ATU_IRQA_MSIX_INT][0];
	sfs->no_print_mask.atu.prb_expired = 1;
	sfs->no_auto_mask.atu.prb_expired = 1;
	sfs->no_print_mask.atu.invalid_ac = 1;
	sfs->no_print_mask.atu.no_translation = 1;

	bitmap_zero(bitmask.mask, sfs->bitlen);
	bitmask.atu.invalid_ac = 1;
	bitmask.atu.no_translation = 1;
	cxi_disable_hw_errors(hw, C_ATU_IRQA_MSIX_INT, false, bitmask.mask);

	/* Ignore some RMU errors. */
	sfs = &hw->sfs_err_flg[C_RMU_IRQA_MSIX_INT][0];
	sfs->no_print_mask.rmu.enet_frame_rejected = 1;
	sfs->no_print_mask.rmu.ptl_no_ptlte = 1;

	bitmap_copy(bitmask.mask, sfs->no_print_mask.mask, sfs->bitlen);
	cxi_disable_hw_errors(hw, C_RMU_IRQA_MSIX_INT, false, bitmask.mask);

	/* Ignore some LPE errors. */
	sfs = &hw->sfs_err_flg[C_LPE_IRQA_MSIX_INT][0];
	sfs->no_print_mask.lpe.entry_not_found = 1;
	sfs->no_print_mask.lpe.pt_disabled_dis = 1;
	sfs->no_print_mask.lpe.pt_disabled_fc = 1;
	sfs->no_print_mask.lpe.src_error = 1;

	bitmap_copy(bitmask.mask, sfs->no_print_mask.mask, sfs->bitlen);
	cxi_disable_hw_errors(hw, C_LPE_IRQA_MSIX_INT, false, bitmask.mask);

	/* Some PCT EXT errors can be ignored in C1*/
	if (cass_version(hw, CASSINI_1)) {
		sfs = &hw->sfs_err_flg[C_PCT_IRQA_MSIX_INT][1];
		bitmap_zero(bitmask.mask, sfs->bitlen);
		bitmask.pct_ext.sct_tbl_rd_misc_unused = 1;
		bitmask.pct_ext.sct_tbl_rd_ram_unused = 1;
		cxi_disable_hw_errors(hw, C_PCT_IRQA_MSIX_INT, true, bitmask.mask);
	}

	/* Mask some EE errors and don't print them */
	sfs = &hw->sfs_err_flg[C_EE_IRQA_MSIX_INT][0];
	sfs->no_print_mask.ee.eq_dsabld_event = 1;

	bitmap_zero(bitmask.mask, sfs->bitlen);
	bitmask.ee.eq_dsabld_event = 1;
	cxi_disable_hw_errors(hw, C_EE_IRQA_MSIX_INT, false, bitmask.mask);

	/* Disable some link management error flags in HNI_PML. These
	 * bits are controlled by the link driver.
	 */
	if (cass_version(hw, CASSINI_2)) {
		sfs = &hw->sfs_err_flg[C_HNI_IRQA_MSIX_INT][0];
		bitmap_zero(bitmask.mask, sfs->bitlen);
		bitmask.c2_hni.llr_eopb = 1;
		bitmap_copy(sfs->no_auto_mask.mask, bitmask.mask, sfs->bitlen);
		cxi_disable_hw_errors(hw, C_HNI_IRQA_MSIX_INT, false, bitmask.mask);

		sfs = &hw->sfs_err_flg[C_HNI_IRQA_MSIX_INT][1];
		bitmap_zero(bitmask.mask, sfs->bitlen);
		bitmask.c2_hni_ext.pmi_ack = 1;
		bitmap_copy(sfs->no_auto_mask.mask, bitmask.mask, sfs->bitlen);
		cxi_disable_hw_errors(hw, C_HNI_IRQA_MSIX_INT, true, bitmask.mask);
	}

	sfs = &hw->sfs_err_flg[C_HNI_PML_IRQA_MSIX_INT][0];
	bitmap_zero(bitmask.mask, sfs->bitlen);
	if (cass_version(hw, CASSINI_1)) {
		sfs->no_print_mask.c1_hni_pml.autoneg_page_received = 1;
		sfs->no_print_mask.c1_hni_pml.autoneg_complete = 1;
		sfs->no_print_mask.c1_hni_pml.pcs_lanes_locked = 1;
		sfs->no_print_mask.c1_hni_pml.pcs_aligned = 1;
		sfs->no_print_mask.c1_hni_pml.pcs_ready = 1;
		sfs->no_print_mask.c1_hni_pml.pcs_link_down = 1;

		bitmask.c1_hni_pml.autoneg_page_received = 1;
		bitmask.c1_hni_pml.autoneg_complete = 1;
		bitmask.c1_hni_pml.pcs_hi_ser = 1;
		bitmask.c1_hni_pml.pcs_link_down = 1;
		bitmask.c1_hni_pml.llr_replay_at_max = 1;
		bitmask.c1_hni_pml.pcs_lanes_locked = 1;
		bitmask.c1_hni_pml.pcs_aligned = 1;
		bitmask.c1_hni_pml.pcs_ready = 1;
	} else {
		bitmask.ss2_port_pml.autoneg_page_received_0 = 1;
		bitmask.ss2_port_pml.autoneg_failed_0 = 1;

		bitmask.ss2_port_pml.llr_init_complete_0 = 1;
		bitmask.ss2_port_pml.llr_init_fail_0 = 1;
		bitmask.ss2_port_pml.llr_loop_time_0 = 1;
		bitmask.ss2_port_pml.llr_loop_time_fail_0 = 1;
		bitmask.ss2_port_pml.llr_replay_at_max_0 = 1;
		bitmask.ss2_port_pml.llr_tx_dp_err = 1;
		bitmask.ss2_port_pml.llr_rx_dp_err = 1;

		bitmask.ss2_port_pml.pcs_aligned_0 = 1;
		bitmask.ss2_port_pml.pcs_hi_ser_0 = 1;
		bitmask.ss2_port_pml.pcs_lanes_locked_0 = 1;
		bitmask.ss2_port_pml.pcs_link_down_0 = 1;
		bitmask.ss2_port_pml.pcs_link_down_lf_0 = 1;
		bitmask.ss2_port_pml.pcs_link_down_rf_0 = 1;
		bitmask.ss2_port_pml.pcs_link_up_0 = 1;
		bitmask.ss2_port_pml.pcs_tx_dp_err = 1;
		bitmask.ss2_port_pml.pcs_rx_dp_err = 1;

		bitmask.ss2_port_pml.mac_tx_dp_err = 1;
		bitmask.ss2_port_pml.mac_rx_dp_err = 1;
		bitmask.ss2_port_pml.mac_rx_fcs32_err_0 = 1;
		bitmask.ss2_port_pml.mac_rx_framing_err_0 = 1;
		bitmask.ss2_port_pml.mac_rx_preamble_err_0 = 1;
	}

	bitmap_copy(sfs->no_auto_mask.mask, bitmask.mask, sfs->bitlen);
	cxi_disable_hw_errors(hw, C_HNI_PML_IRQA_MSIX_INT, false, bitmask.mask);

	/* Allocate the work queue with a single worker. */
	hw->err_irq_wq = alloc_workqueue("error_wq", WQ_UNBOUND, 1);
	if (!hw->err_irq_wq) {
		rc = -ENOMEM;
		goto free_err_flg_kobjs;
	}

	INIT_WORK(&hw->err_irq_work, error_irq_worker);

	/* Register all 15 IRQA interrupts. */
	hw->err_irq_mask = 0x7fff;

	/* Register and activate all the error reporting
	 * interrupts.
	 */
	for_each_set_bit(irq, &hw->err_irq_mask, NUM_ERR_INTS) {
		int vec;

		vec = pci_irq_vector(hw->cdev.pdev, irq);
		hw->err_irq_vecs[irq] = vec;

		scnprintf(hw->err_int_names[irq], sizeof(hw->err_int_names[irq]),
			  "%s_err_%s", hw->cdev.name, INFO(irq, 0).csr_name_lo);
		rc = request_irq(vec, error_irq_cb, 0,
				 hw->err_int_names[irq], hw);
		if (rc) {
			cxidev_err(&hw->cdev, "Failed to request IRQ for events.\n");
			hw->err_irq_vecs[irq] = -1;
			goto remove;
		}
	}

	return 0;

remove:
	free_err_irqs(hw);
	destroy_workqueue(hw->err_irq_wq);
	hw->err_irq_wq = NULL;

free_err_flg_kobjs:
	free_kobjs_and_hw_errors(hw);

	kobject_put(hw->err_flgs_dir_kobj[0]);

	return rc;
}

void deregister_error_handlers(struct cass_dev *hw)
{
	free_err_irqs(hw);
	destroy_workqueue(hw->err_irq_wq);
	hw->err_irq_wq = NULL;
	free_kobjs_and_hw_errors(hw);
	kobject_put(hw->err_flgs_dir_kobj[0]);
}
