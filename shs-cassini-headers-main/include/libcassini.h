// SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
/*
 * Copyright 2018 Cray Inc. All rights reserved
 *
 * Cassini hardware definitions and accessors. This is the header to
 * include Cassini hardware support.
 *
 * This header is only for entities that control the hardware (the
 * kernel core driver, a bios driver, ...) Applications only include
 * cxi_prov_hw.h.
 */

#ifndef __CXI_CASSINI_H
#define __CXI_CASSINI_H

#ifndef __KERNEL__
#include <stdbool.h>
#endif

#include "cassini_user_defs.h"
#include "cxi_prov_hw.h"

/* Assume that shadow CSRs are at same offset across all blocks. This
 * is not true for IXE and SS2_PORT_PML, but shadow CSRs are not used
 * for these.
 */
#define SHADOW_OFFSET C_ATU_MSC_SHADOW_OFFSET
#define SHADOW_ACTION_OFFSET C_ATU_MSC_SHADOW_ACTION_OFFSET

struct hni_data4_full_amo {
	struct c_port_unrestricted_hdr full;
	struct c_port_amo_payload payload;
} __attribute__((packed));

/* IPV4 and generic Portals header. */
struct hni_header4 {
	struct c_port_fab_hdr hdr;
	union {
		struct c_pkt_type		pkt_type;
		struct c_port_unrestricted_hdr	full;
		struct c_port_small_msg_hdr	small_msg;
		struct c_port_continuation_hdr	cont;
		struct c_port_restricted_hdr	restricted;
		struct hni_data4_full_amo	full_amo;
	};
};

/* Small packet format. */
struct hni_header_vs {
	struct c_port_fab_vs_hdr	hdr;
	struct c_pkt_type		pkt_type;
	/* May need to expand union */
	union {
		struct c_port_restricted_hdr	req;
	};
};

/* IPV6 header. */
struct hni_header6 {
	uint32_t	version  : 4;
	uint32_t	tc       : 8;
	uint32_t	flowLab  :20;	/* Flow Label */
	uint16_t	length;		/* Payload length */
	uint8_t		nextHead;	/* Next Header */
	uint8_t		hopLimit;
	uint32_t	srcIP[4];
	uint32_t	dstIP[4];
};

struct hni_ver {
    uint8_t unused : 4;
    uint8_t ver    : 4;
};

union hni_header {
    struct hni_ver      ver;
	struct hni_header4   h4;
	struct hni_header_vs vs;
	struct hni_header6   h6;
};

/**
 * cass_flush_pci() - Flush the PCI bus for the device
 *
 * Since PCI bus writes are posted asynchronously, force them to reach
 * the device. This is needed when some resources are allocated or
 * freed. Any CSR read is good.
 */
static inline void cass_flush_pci(struct cass_dev *dev)
{
	union c_mb_sts_rev rev;

	cass_read(dev, C_MB_STS_REV, &rev, sizeof(rev));
}

/**
 * cass_shadow_write_prepare() - copy the CSR data to the shadow region.
 *
 * This is called before cass_shadow_write_do() to prepare the shadow
 * write. It is useful when the same data must be copied to many CSRs
 * through the shadow region. Otherwise it is easier to use
 * cass_shadow_write().
 */
static inline void
cass_shadow_write_prepare(struct cass_dev *dev, unsigned int block_base,
			  const void *csr_value,
			  unsigned int csr_length)
{
	cass_write(dev, block_base + SHADOW_OFFSET, csr_value, csr_length);
}

/**
 * cass_shadow_write_do() - Perform a shadow write
 *
 * Ring the corresponding CSR. cass_shadow_write_prepare() must be
 * called first to prepare the data to write to the CSR.
 */
static inline void
cass_shadow_write_do(struct cass_dev *dev, unsigned int block_base,
		     unsigned int addr_offset)
{
	union c_msc_shadow_action shadow = {
		.write = 1,
		.addr_offset = addr_offset
	};

	cass_write(dev, block_base + SHADOW_ACTION_OFFSET, &shadow,
		   sizeof(shadow));
}

/**
 * cass_shadow_write() - Perform a shadow write
 */
static inline void
cass_shadow_write(struct cass_dev *dev, unsigned int block_base,
		  unsigned int addr_offset, const void *csr_value,
		  unsigned int csr_length)
{
	cass_shadow_write_prepare(dev, block_base, csr_value, csr_length);
	cass_shadow_write_do(dev, block_base, addr_offset);
}

/**
 * cass_shadow_read() - Perform a shadow read
 */
static inline void
cass_shadow_read(struct cass_dev *dev, unsigned int block_base,
		 unsigned int addr_offset, void *csr_value,
		 unsigned int csr_length)
{
	union c_msc_shadow_action shadow = {
		.write = 0,
		.addr_offset = addr_offset
	};

	cass_write(dev, block_base + SHADOW_ACTION_OFFSET, &shadow,
		   sizeof(shadow));

	cass_read(dev, block_base + SHADOW_OFFSET, csr_value, csr_length);
}

/**
 * cass_tx_cq_init() - Configure a transmit CQ
 */
static inline void cass_tx_cq_init(struct cass_dev *dev, struct cxi_cq *cq,
				   unsigned int cq_num,
				   const union c_cq_txq_tc_table *tc,
				   const union c_cq_txq_base_table *cfg)
{
	const union c_cq_txq_wrptr_table wrptr = {
		.mem_q_wr_ptr = C_CQ_FIRST_WR_PTR,
	};

	cass_write(dev, C_CQ_TXQ_TC_TABLE(cq_num), tc, sizeof(*tc));
	cass_write(dev, C_CQ_TXQ_WRPTR_TABLE(cq_num), &wrptr, sizeof(wrptr));

	cass_shadow_write(dev, C_CQ_BASE, C_CQ_TXQ_BASE_TABLE(cq_num),
			  cfg, sizeof(*cfg));
}

/**
 * cass_tgt_cq_init() - Configure a target CQ
 */
static inline void
cass_tgt_cq_init(struct cass_dev *dev, struct cxi_cq *cq,
		 unsigned int cq_num, const union c_cq_tgq_table *cfg)
{
	const union c_cq_tgq_wrptr_table wrptr = {
		.mem_q_wr_ptr = C_CQ_FIRST_WR_PTR,
	};

	cass_write(dev, C_CQ_TGQ_WRPTR_TABLE(cq_num), &wrptr, sizeof(wrptr));

	cass_shadow_write(dev, C_CQ_BASE, C_CQ_TGQ_TABLE(cq_num),
			  cfg, sizeof(*cfg));
}

/**
 * cass_tx_cq_enable() - Enable a transmit CQ
 */
static inline void
cass_tx_cq_enable(struct cass_dev *dev, unsigned int cq_num)
{
	const union c_cq_txq_enable cfg = {
		.txq_enable = 1,
	};

	cass_write(dev, C_CQ_TXQ_ENABLE(cq_num), &cfg, sizeof(cfg));

	cass_flush_pci(dev);
}

/**
 * cass_tgt_cq_enable() - Enable a target CQ
 */
static inline void
cass_tgt_cq_enable(struct cass_dev *dev, unsigned int cq_num)
{
	const union c_cq_tgq_enable cfg = {
		.tgq_enable = 1,
	};

	cass_write(dev, C_CQ_TGQ_ENABLE(cq_num), &cfg, sizeof(cfg));

	cass_flush_pci(dev);
}

/**
 * cass_tx_cq_disable() - Disable a transmit CQ
 */
static inline void
cass_tx_cq_disable(struct cass_dev *dev, unsigned int cq_num)
{
	const union c_cq_txq_enable cfg = {
			.drain = 1,
	};

	cass_write(dev, C_CQ_TXQ_ENABLE(cq_num), &cfg, sizeof(cfg));

	cass_flush_pci(dev);
}

/**
 * cass_tgt_cq_disable() - Disable a target CQ
 */
static inline void
cass_tgt_cq_disable(struct cass_dev *dev, unsigned int cq_num)
{
	const union c_cq_tgq_enable cfg = {
		.drain = 1,
	};

	cass_write(dev, C_CQ_TGQ_ENABLE(cq_num), &cfg, sizeof(cfg));

	cass_flush_pci(dev);
}

/**
 * cass_eq_set_event_format() - Set default event size
 *
 * Tell Cassini to use short or large events for the EQ. By default,
 * Cassini will use short event format if possible.
 *
 * Since the CSR accessed is shared by 32 EQs, access to this function
 * must be serialized.
 *
 * @initr_use_long: If set, always use long format initiator events
 * @trgt_use_long: If set, always use long format target events
 */
static inline void cass_eq_set_event_format(struct cass_dev *dev,
					    unsigned int eq_num,
					    bool initr_use_long,
					    bool trgt_use_long)
{
	unsigned int index = eq_num / 32;
	unsigned int offset = eq_num % 32 * 2;
	uint64_t mask = 0b11UL << offset;
	union c_ee_cfg_long_evnt_ovr_table ovr;

	cass_read(dev, C_EE_CFG_LONG_EVNT_OVR_TABLE(index),
		  &ovr.qw, sizeof(ovr));

	ovr.qw &= ~mask;
	mask = (uint64_t)(trgt_use_long << 1 | initr_use_long) << offset;
	ovr.qw |= mask;

	cass_write(dev, C_EE_CFG_LONG_EVNT_OVR_TABLE(index),
		   &ovr.qw, sizeof(ovr));
}

/**
 * cass_eq_set_periodic_timestamp() - Enable/disable timestamp event generation
 *
 * Since the CSR accessed is shared by 64 EQs, access to this function
 * must be serialized.
 *
 * @dev: the Cassini device
 * @eq_num: the EQ index
 * @enable: whether to enable or disable the timestamp event generation
 */
static inline void cass_eq_set_periodic_timestamp(struct cass_dev *dev,
						  unsigned int eq_num,
						  bool enable)
{
	uint64_t mask = 1ULL << (eq_num & 0x3f);
	union c_ee_cfg_periodic_tstamp_table cfg_periodic_tstamp;

	cass_read(dev, C_EE_CFG_PERIODIC_TSTAMP_TABLE(eq_num / 64),
		  &cfg_periodic_tstamp, sizeof(cfg_periodic_tstamp));

	if (enable)
		cfg_periodic_tstamp.n63_n0_enable_periodic_tstamp |= mask;
	else
		cfg_periodic_tstamp.n63_n0_enable_periodic_tstamp &= ~mask;

	cass_write(dev, C_EE_CFG_PERIODIC_TSTAMP_TABLE(eq_num / 64),
		   &cfg_periodic_tstamp, sizeof(cfg_periodic_tstamp));
}

/**
 * cass_eq_init() - Initializes an event queue.
 *
 * @dev: Cassini device
 * @eq_num: Event queue number
 * @rgid: Resource group ID to assign the EQ to
 * @cfg: Event queue initial configuration
 * @initr_use_long: If true, always use long format for initiator events
 * @trgt_use_long: If true, always use long format for initiator events
 *
 * Caller must ensure that C_EE_CFG_INIT_EQ_HW_STATE is ready to be
 * used by checking its pending bit beforehand.
 */
static inline void cass_eq_init(struct cass_dev *dev, unsigned int eq_num,
				uint8_t rgid, union c_ee_cfg_eq_descriptor *cfg,
				bool initr_use_long, bool trgt_use_long)
{
	static const union c_ee_cfg_eq_sw_state sw_state = {};
	const union c_ee_cfg_init_eq_hw_state hw_state = {
		.eq_handle = eq_num
	};
	const union c_cq_cfg_eq_rgid_table cfg_rgid = { .rgid = rgid };
	uint64_t csr;

	cass_write(dev, C_CQ_CFG_EQ_RGID_TABLE(eq_num),
		   &cfg_rgid, sizeof(cfg_rgid));

	csr = C_EE_CFG_EQ_SW_STATE(eq_num);
	cass_write(dev, csr, &sw_state, sizeof(sw_state));

	csr = C_EE_CFG_INIT_EQ_HW_STATE;
	cass_write(dev, csr, &hw_state, sizeof(hw_state));

	/* Initialize the event queue descriptor */
	cass_shadow_write(dev, C_EE_BASE, C_EE_CFG_EQ_DESCRIPTOR(eq_num),
			  cfg, sizeof(*cfg));

	cass_eq_set_event_format(dev, eq_num, initr_use_long, trgt_use_long);
	cass_eq_set_periodic_timestamp(dev, eq_num, false);

	cass_flush_pci(dev);
}

/**
 * cass_eq_clear() - Clear the event queue configuration.
 *
 * Note: cass_eq_init() must be called before cass_eq_clear().
 */
static inline void cass_eq_clear(struct cass_dev *dev, struct cxi_eq *evtq)
{
	static const union c_ee_cfg_eq_descriptor cfg = {};
	static const union c_cq_cfg_eq_rgid_table cfg_rgid = {
		.rgid = C_RESERVED_RGID
	};

	cass_shadow_write(dev, C_EE_BASE, C_EE_CFG_EQ_DESCRIPTOR(evtq->eqn),
			  &cfg, sizeof(cfg));

	cass_write(dev, C_CQ_CFG_EQ_RGID_TABLE(evtq->eqn),
		   &cfg_rgid, sizeof(cfg_rgid));

	cass_flush_pci(dev);
}

/**
 * cass_read_lpe_reserve_pool() - Get reserved entries in PE's pool
 *
 * pe is 0 to 3, pool is 0 to 15.
 */
static inline void
cass_read_lpe_reserve_pool(struct cass_dev *dev, unsigned int pe,
			   unsigned int pool,
			   union c_lpe_cfg_pe_le_pools *alloc)
{
	uint64_t csr;

	csr = C_LPE_CFG_PE_LE_POOLS(pe * (C_LPE_CFG_PE_LE_POOLS_ENTRIES / C_PE_COUNT) + pool);

	cass_read(dev, csr, alloc, sizeof(*alloc));
}

/**
 * cass_config_lpe_reserve_pool() - Reserve some entries in a PE's pool.
 *
 * Reserve some entries in pool 'pool' for Process Engine 'pe'.
 * Return 0 if the adapter accepted the new setting, or a negative
 * errno on error.
 *
 * pe is 0 to 3, pool is 0 to 15.
 */
static inline int
cass_config_lpe_reserve_pool(struct cass_dev *dev, unsigned int pe,
			     unsigned int pool,
			     const union c_lpe_cfg_pe_le_pools *alloc)
{
	uint64_t csr;
	union c_lpe_cfg_pe_le_pools alloc_check = {};

	csr = C_LPE_CFG_PE_LE_POOLS(pe * (C_LPE_CFG_PE_LE_POOLS_ENTRIES / C_PE_COUNT) + pool);

	cass_write(dev, csr, alloc, sizeof(*alloc));

	/* Check the write was successful, as the spec recommends. */
	cass_read(dev, csr, &alloc_check, sizeof(alloc_check));

	if (alloc_check.num_reserved == alloc->num_reserved &&
	    alloc_check.max_alloc == alloc->max_alloc) {
		return 0;
	} else {
		return -EINVAL;
	}
}

/**
 * cass_lpe_reserve_pool_sts() - Get the number of used entries in a PE pool.
 *
 * pe is 0 to 3, pool is 0 to 15.
 */
static inline void
cass_lpe_reserve_pool_sts(struct cass_dev *dev, unsigned int pe,
			  unsigned int pool,
			  union c_lpe_sts_pe_le_alloc *query)
{
	uint64_t csr;

	csr = C_LPE_STS_PE_LE_ALLOC(pe * (C_LPE_STS_PE_LE_ALLOC_ENTRIES / C_PE_COUNT) + pool);
	cass_read(dev, csr, query, sizeof(*query));
}

/**
 * cass_read_lpe_shared_pool() - Get the number of shared list entries in a PE.
 *
 * pe is 0 to 3.
 */
static inline void
cass_read_lpe_shared_pool(struct cass_dev *dev, unsigned int pe,
			  union c_lpe_cfg_pe_le_shared *shared)
{
	uint64_t csr;

	csr = C_LPE_CFG_PE_LE_SHARED(pe);

	cass_read(dev, csr, shared, sizeof(*shared));
}

/**
 * cass_config_lpe_shared_pool() - Configure the shared list entries in a PE.
 *
 * pe is 0 to 3.
 */
static inline int
cass_config_lpe_shared_pool(struct cass_dev *dev, unsigned int pe,
			    union c_lpe_cfg_pe_le_shared *shared)
{
	union c_lpe_cfg_pe_le_shared shared_check = {};
	uint64_t csr;

	csr = C_LPE_CFG_PE_LE_SHARED(pe);

	cass_write(dev, csr, shared, sizeof(*shared));

	/* Check the write was successful, as the spec recommends. */
	cass_read(dev, csr, &shared_check, sizeof(shared_check));

	if (shared_check.num_total == shared->num_total &&
	    shared_check.num_shared == shared->num_shared) {
		return 0;
	} else {
		return -EINVAL;
	}
}

/**
 * cass_read_le_state() - Read a list entry (LE) state
 *
 * Since this function uses C_LPE_MSC_SHADOW, its access must be
 * serialized.
 */
static inline void cass_read_le_state(struct cass_dev *dev, unsigned int le_idx,
				      union c_lpe_sts_list_entries *le_sts)
{
	cass_shadow_read(dev, C_LPE_BASE, C_LPE_STS_LIST_ENTRIES_OFFSET(le_idx),
			 le_sts, sizeof(*le_sts));
}

/**
 * cass_read_pte() - Read a Table Entry (PtlTE)
 *
 * Since this function uses C_LPE_MSC_SHADOW, its access must be
 * serialized.
 */
static inline void cass_read_pte(struct cass_dev *dev, unsigned int pte_num,
				 union c_lpe_cfg_ptl_table *ptl_table)
{
	cass_shadow_read(dev, C_LPE_BASE, C_LPE_CFG_PTL_TABLE_OFFSET(pte_num),
			 ptl_table, sizeof(*ptl_table));
}

/**
 * cass_config_pte() - Configure a Table Entry (PtlTE)
 *
 * Since this function uses C_LPE_MSC_SHADOW, its access must be
 * serialized.
 */
static inline void cass_config_pte(struct cass_dev *dev, unsigned int pte_num,
				   const union c_lpe_cfg_ptl_table *ptl_table)
{
	cass_shadow_write(dev, C_LPE_BASE, C_LPE_CFG_PTL_TABLE_OFFSET(pte_num),
			  ptl_table, sizeof(*ptl_table));
}

/**
 * cass_config_ac() - Configure an Address Context entry (AC)
 *
 * Since this function uses C_ATU_MSC_SHADOW, its access must be
 * serialized.
 */
static inline void cass_config_ac(struct cass_dev *dev, unsigned int ac_num,
				  const union c_atu_cfg_ac_table *atu_ac)
{
	cass_shadow_write(dev, C_ATU_BASE, C_ATU_CFG_AC_TABLE_OFFSET(ac_num),
			  atu_ac, sizeof(*atu_ac));
}

/**
 * cass_config_pi_acxt() - Configure a PI Address Context entry (AC)
 */
static inline void cass_config_pi_acxt(struct cass_dev *dev,
				       unsigned int ac_num,
				       const union c_pi_cfg_acxt *pi_acxt)
{
	cass_write(dev, C_PI_CFG_ACXT(ac_num), pi_acxt, sizeof(*pi_acxt));
}

/**
 * cass_invalidate_vni_list() - Invalidate a VNI list entry.
 */
static inline void cass_invalidate_vni_list(struct cass_dev *dev,
					    unsigned int idx)
{
	const union c_rmu_cfg_vni_list_invalidate
		rmu_cfg_vni_list_invalidate = {
		.invalidate = 1
	};

	cass_write(dev, C_RMU_CFG_VNI_LIST_INVALIDATE(idx),
		   &rmu_cfg_vni_list_invalidate,
		   sizeof(rmu_cfg_vni_list_invalidate));
}

/**
 * cass_config_matching_vni_list() - Configure a VNI entry in the VNI list.
 *
 * The VNI is returned with cass_invalidate_vni_list().
 *
 * The X and Y CSRs are used to configure the VNI value for which the
 * VNI List entry matches. If an exact match is required, the value to be
 * matched is written to the X CSR array entry and the one's complement
 * of the value to be matched is written to the Y CSR array entry.
 *
 * If any bits of the value are to be ignored when checking for a match,
 * those bits are written with the value 0 in both the X and Y CSR array entries.
 * X and Y of the same bit position should not both be written to the value 1;
 * doing so leads to increased power consumption. However, it is acceptable
 * for X and Y of the same bit position to both be 1 while the entry is in the
 * invalid state and is being transitioned to a new value.
 */
static inline void cass_config_matching_vni_list(struct cass_dev *dev,
						 unsigned int idx,
						 unsigned int vni,
						 unsigned int ignore)
{
	union c_rmu_cfg_vni_list rmu_cfg_vni_list;

	/* Invalidate first */
	cass_invalidate_vni_list(dev, idx);

	/* Remove the ignore bits from the VNI. */
	vni &= ~ignore;

	rmu_cfg_vni_list.vni = vni;
	cass_write(dev, C_RMU_CFG_VNI_LIST_X(idx),
		   &rmu_cfg_vni_list, sizeof(rmu_cfg_vni_list));

	rmu_cfg_vni_list.vni = ~vni & ~ignore;
	cass_write(dev, C_RMU_CFG_VNI_LIST_Y(idx),
		   &rmu_cfg_vni_list, sizeof(rmu_cfg_vni_list));
}

/**
 * cass_config_vni_list() - Configure a VNI entry in the VNI list.
 *
 * Only exact matches are currently supported. The VNI is returned
 * with cass_invalidate_vni_list().
 */
static inline void cass_config_vni_list(struct cass_dev *dev,
					unsigned int idx,
					unsigned int vni)
{
	cass_config_matching_vni_list(dev, idx, vni, 0);
}

/**
 * cass_config_portal_index_table() - Set a portal index table entry.
 *
 * The corresponding entry must be invalidated first. Since each CSR
 * addresses 4 different portal entries, and only one is changed,
 * serialization must be ensured.
 *
 * @dev: Cassini device
 * @plist_idx: portal list index
 * @phys_ptn: portal number allocated on device
 */
static inline void
cass_config_portal_index_table(struct cass_dev *dev, unsigned int plist_idx,
			       unsigned int phys_ptn)
{
	union c_rmu_cfg_portal_index_table portal_index_table = {};
	const unsigned int idx =
		plist_idx / C_RMU_CFG_PORTAL_INDEX_TABLE_ARRAY_SIZE;
	const unsigned int offset =
		plist_idx % C_RMU_CFG_PORTAL_INDEX_TABLE_ARRAY_SIZE;

	/* Read, modify, write */
	cass_read(dev, C_RMU_CFG_PORTAL_INDEX_TABLE(idx),
		  &portal_index_table, sizeof(portal_index_table));

	portal_index_table.e[offset].phys_portal_table_idx = phys_ptn;

	cass_write(dev, C_RMU_CFG_PORTAL_INDEX_TABLE(idx),
		   &portal_index_table, sizeof(portal_index_table));
}

/**
 * cass_invalidate_portal_list() - Invalidate a Portal list entry
 */
static inline void cass_invalidate_portal_list(struct cass_dev *dev,
					       unsigned int idx)
{
	const union c_rmu_cfg_portal_list_invalidate
		rmu_cfg_portal_list_invalidate = {
		.invalidate = 1
	};

	cass_write(dev, C_RMU_CFG_PORTAL_LIST_INVALIDATE(idx),
		   &rmu_cfg_portal_list_invalidate,
		   sizeof(rmu_cfg_portal_list_invalidate));
}

/**
 * cass_config_portal_list() - Configure a Portal list entry in the
 * portals list, and its corresponding entry in the RMU portal index
 * table.
 *
 * Only exact matches are supported.
 *
 * Since each portal index table CSR addresses 4 different portal
 * entries, and only one is changed, serialization must be ensured.
 *
 * @dev: Cassini device
 * @plist_idx: portal list index
 * @phys_ptn: portal number allocated on device
 * @cfg: pointer to the portal list configuration
 */
static inline void
cass_config_portal_list(struct cass_dev *dev, unsigned int plist_idx,
			unsigned int phys_ptn,
			const union c_rmu_cfg_portal_list *cfg)
{
	union c_rmu_cfg_portal_list cfg_y;

	/* Invalidate first */
	cass_invalidate_portal_list(dev, plist_idx);

	cass_config_portal_index_table(dev, plist_idx, phys_ptn);

	/* Activate the portal list */
	cass_write(dev, C_RMU_CFG_PORTAL_LIST_X(plist_idx), cfg, sizeof(*cfg));

	cfg_y = *cfg;
	cfg_y.index_ext = ~cfg->index_ext;
	cfg_y.multicast_id = ~cfg->multicast_id;
	cfg_y.is_multicast = ~cfg->is_multicast;
	cfg_y.vni_list_idx = ~cfg->vni_list_idx;
	cass_write(dev, C_RMU_CFG_PORTAL_LIST_Y(plist_idx),
		   &cfg_y, sizeof(cfg_y));
}

/**
 * cass_invalidate_set_list() - Invalidate a Set List entry.
 *
 * @dev: Cassini device
 * @set_list_idx: Set List and Set Ctrl indexes
 */
static inline void cass_invalidate_set_list(struct cass_dev *dev,
					    unsigned int set_list_idx)
{
	static const union c_rmu_cfg_ptlte_set_list_invalidate
		rmu_cfg_ptlte_set_list_invalidate = { .invalidate = 1 };

	cass_write(dev, C_RMU_CFG_PTLTE_SET_LIST_INVALIDATE(set_list_idx),
		   &rmu_cfg_ptlte_set_list_invalidate,
		   sizeof(rmu_cfg_ptlte_set_list_invalidate));
}

/**
 * cass_config_set_ctrl() - Configure a Set Ctrl entry
 *
 * @dev: Cassini device
 * @set_list_idx: Set Ctrl indexes
 */
static inline void
cass_config_set_ctrl(struct cass_dev *dev,
		     unsigned int set_list_idx,
		     const struct c_rmu_cfg_ptlte_set_ctrl_table_entry *entry)
{
	unsigned int set_ctrl_idx = set_list_idx /
		C_RMU_CFG_PTLTE_SET_CTRL_TABLE_ARRAY_SIZE;
	unsigned int set_ctrl_entry = set_list_idx %
		C_RMU_CFG_PTLTE_SET_CTRL_TABLE_ARRAY_SIZE;
	union c_rmu_cfg_ptlte_set_ctrl_table set_ctrl;

	cass_read(dev, C_RMU_CFG_PTLTE_SET_CTRL_TABLE(set_ctrl_idx),
		  &set_ctrl, sizeof(set_ctrl));

	set_ctrl.e[set_ctrl_entry] = *entry;

	cass_write(dev, C_RMU_CFG_PTLTE_SET_CTRL_TABLE(set_ctrl_idx),
		   &set_ctrl, sizeof(set_ctrl));
}

/**
 * cass_config_indir_entry() - Configure an indirection table
 *
 * @dev: Cassini device
 * @indir_idx: Index in the indirection table
 * @ptlte_idx: Portal table entry to direct to
 */
static inline void cass_config_indir_entry(struct cass_dev *dev,
					   unsigned int indir_idx,
					   unsigned int ptlte_idx)
{
	union c_rmu_cfg_portal_index_indir_table indir_table;
	unsigned int indir_table_idx = indir_idx /
		C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_ARRAY_SIZE;
	unsigned int indir_table_entry = indir_idx %
		C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_ARRAY_SIZE;

	cass_read(dev, C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE(indir_table_idx),
		  &indir_table, sizeof(indir_table));

	indir_table.e[indir_table_entry].phys_portal_table_idx = ptlte_idx;

	cass_write(dev, C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE(indir_table_idx),
		   &indir_table, sizeof(indir_table));
}

/**
 * cass_config_set_list() - Configure a Set List entry
 *
 * Each Ethernet device needs a default entry to catch packets that do
 * not hash to anything. Program the Set List, Set Ctrl and
 * indirection entries.
 *
 * @dev: Cassini device
 * @set_list_idx: Set List and Set Ctrl indexes
 * @ptlte_idx: Portal table entry
 * @cfg: how to program the set list
 * @cfg_mask: valid bits for @cfg
 *
 * Since this function uses C_RMU_MSC_SHADOW, its access must be
 * serialized.
 */
static inline void
cass_config_set_list(struct cass_dev *dev, unsigned int set_list_idx,
		     unsigned int ptlte_idx,
		     const union c_rmu_cfg_ptlte_set_list *cfg,
		     const union c_rmu_cfg_ptlte_set_list *cfg_mask)
{
	static const struct c_rmu_cfg_ptlte_set_ctrl_table_entry set_ctrl = {};

	/* Invalidate set list first */
	cass_invalidate_set_list(dev, set_list_idx);

	/* Clear the control table access */
	cass_config_set_ctrl(dev, set_list_idx, &set_ctrl);

	/* Modify the indirection table with the physical PtlTE. */
	cass_config_indir_entry(dev, 2048 + set_list_idx, ptlte_idx);

	/* Enable the set list */
	cass_write(dev, C_RMU_CFG_PTLTE_SET_LIST_X(set_list_idx),
		   cfg, sizeof(*cfg));

	cass_shadow_write(dev, C_RMU_BASE,
			  C_RMU_CFG_PTLTE_SET_LIST_Y_OFFSET(set_list_idx),
			  cfg_mask, sizeof(*cfg_mask));
}

/**
 * cass_ct_enable() - Configure and enable a counting event.
 *
 * @dev: Cassini device
 * @ctn: Counter number
 * @rgid: Resource group ID to assign the CT to
 * @wb: Writeback buffer
 * @wb_addr: Address for writeback counter (physical or virtual)
 * @wb_ac: Address context for @wb_addr
 * @ct_mmio: Base address of CT doorbell memory
 * @ct: User counter structure to initialize
 */
static inline void
cass_ct_enable(struct cass_dev *dev, unsigned int ctn, uint8_t rgid,
	       uint64_t wb_addr, uint16_t wb_ac)
{
	static const union c_cq_cfg_ct_enable enable = { .ct_enable = 1, };
	const union c_cq_cfg_ct_rgid_table cfg = { .rgid = rgid };
	const union c_cq_cfg_ct_wb_mem_addr mem = {
		.wb_mem_addr = wb_addr >> 3,
		.wb_mem_context = wb_ac,
	};

	cass_write(dev, C_CQ_CFG_CT_RGID_TABLE(ctn), &cfg, sizeof(cfg));
	cass_write(dev, C_CQ_CFG_CT_ENABLE(ctn), &enable, sizeof(enable));

	/* Enable the writeback area (if it exists) after enabling the
	 * CT. Cassini is triggering a writeback during reset which
	 * the driver doesn't want to deal with.
	 */
	cass_write(dev, C_CQ_CFG_CT_WB_MEM_ADDR(ctn), &mem, sizeof(mem));

	cass_flush_pci(dev);
}

/**
 * cass_ct_disable() - Disable a counting event (CT).
 *
 * @dev: Cassini device
 * @ctn: Counter number
 */
static inline void cass_ct_disable(struct cass_dev *dev, unsigned int ctn)
{
	static const union c_cq_cfg_ct_enable enable = {};
	static const union c_cq_cfg_ct_rgid_table cfg = {};
	static const union c_cq_cfg_ct_wb_mem_addr mem = {};

	cass_write(dev, C_CQ_CFG_CT_WB_MEM_ADDR(ctn), &mem, sizeof(mem));
	cass_write(dev, C_CQ_CFG_CT_ENABLE(ctn), &enable, sizeof(enable));
	cass_write(dev, C_CQ_CFG_CT_RGID_TABLE(ctn), &cfg, sizeof(cfg));

	cass_flush_pci(dev);
}

/**
 * cass_assign_ptlte_to_rgid() - Assign a portal entry to a resource group.
 *
 * @dev: Cassini device
 * @pte: The portal table entry
 * @rgid: Resource group ID
 */
static inline void cass_assign_ptlte_to_rgid(struct cass_dev *dev,
					     unsigned int pte, uint8_t rgid)
{
	union c_cq_cfg_ptlte_rgid_table cfg = { .rgid = rgid };

	cass_write(dev, C_CQ_CFG_PTLTE_RGID_TABLE(pte), &cfg, sizeof(cfg));
}

/**
 * cass_set_cp_table() - Set a communication profile table entry.
 *
 * @dev: Cassini device
 * @cid: Communication profile ID
 * @cfg: Desired configuration
 */
static inline void cass_set_cp_table(struct cass_dev *dev, uint8_t cid,
				     const union c_cq_cfg_cp_table *cfg)
{
	cass_write(dev, C_CQ_CFG_CP_TABLE(cid), cfg, sizeof(*cfg));
}

/**
 * cass_set_cp_fl_table() - Set a communication profile flow table entry.
 *
 * @dev: Cassini device
 * @cid: Communication profile ID
 * @cfg: Desired configuration
 */
static inline void cass_set_cp_fl_table(struct cass_dev *dev, uint8_t cid,
					const union c_cq_cfg_cp_fl_table *cfg)
{
	cass_write(dev, C_CQ_CFG_CP_FL_TABLE(cid), cfg, sizeof(*cfg));
}

/**
 * cass_set_pfq_tc_map() - Map prefetch queue to traffic class.
 *
 * @dev: Cassini device
 * @pfq_id: Prefetch queue ID
 * @cfg: Desired configuration
 */
static inline void cass_set_pfq_tc_map(struct cass_dev *dev, uint8_t pfq_id,
				       const union c_cq_cfg_pfq_tc_map *cfg)
{
	cass_write(dev, C_CQ_CFG_PFQ_TC_MAP(pfq_id), cfg, sizeof(*cfg));
}

/**
 * cass_set_acid() - Set an ACID for an RGID
 *
 * There are 8 ACIDs per RGID.
 *
 * @dev: Cassini device
 * @rgid: Resource group ID
 * @lac:  Index in the RGID. 0 to 7.
 * @acid: The ACID to store in the RGID
 */
static inline void cass_set_acid(struct cass_dev *dev, uint8_t rgid,
				 unsigned int lac, uint16_t acid)
{
	const union c_cq_cfg_ac_array cfg = { .ac = acid };

	cass_write(dev, C_CQ_CFG_AC_ARRAY(rgid * C_NUM_LACS + lac),
		   &cfg, sizeof(cfg));

	cass_flush_pci(dev);
}

/**
 * cass_set_cid() - Set a CID for an RGID
 *
 * There are 16 CIDs in a RGID.
 *
 * @dev: Cassini device
 * @rgid: Resource group ID
 * @lcid_idx: Index in the RGID. 0 to 15.
 * @cid: The CID to store in the RGID
 */
static inline void cass_set_cid(struct cass_dev *dev, uint8_t rgid,
				unsigned int lcid_idx, uint8_t cid)
{
	union c_cq_cfg_cid_array cfg = { .cid = cid };

	cass_write(dev, C_CQ_CFG_CID_ARRAY(rgid * 16 + lcid_idx),
		   &cfg, sizeof(cfg));
}

/**
 * cass_set_vf_pf_ints() - Enable or disable a VF interrupt on the host
 *
 * Part of the PF/VF communication. This allows the given VF to
 * generate interrupt 31 on the host to signify a message is
 * ready. This is called by the owner of the PF.
 *
 * @dev: Cassini device
 * @vf_num: which VF interrupt to set
 * @enable: whether to enable or disable the VF interrupt
 */
static inline void cass_set_vf_pf_int(struct cass_dev *dev,
				      unsigned int vf_num,
				      bool enable)
{
	union c_pi_ipd_cfg_vf_pf_irq_mask cfg_vf_pf_irq_mask;

	cass_read(dev , C_PI_IPD_CFG_VF_PF_IRQ_MASK, &cfg_vf_pf_irq_mask,
		  sizeof(cfg_vf_pf_irq_mask));
	if (enable)
		cfg_vf_pf_irq_mask.mask &= ~(1 << vf_num);
	else
		cfg_vf_pf_irq_mask.mask |= (1 << vf_num);
	cass_write(dev, C_PI_IPD_CFG_VF_PF_IRQ_MASK, &cfg_vf_pf_irq_mask,
		   sizeof(cfg_vf_pf_irq_mask));
}

/**
 * cass_clear_vf_pf_int() - Clear a pending VF interrupt
 *
 * Called by the owner of the PF after processing an interrupt from a
 * VF.
 *
 * @dev: Cassini device
 * @vf_num: which VF interrupt to clear
 */
static inline void cass_clear_vf_pf_int(struct cass_dev *dev,
					unsigned int vf_num)
{
	union c_pi_ipd_cfg_vf_pf_irq_clr cfg_vf_pf_irq_clr = {
		.clr = (1 << vf_num),
	};

	cass_write(dev, C_PI_IPD_CFG_VF_PF_IRQ_CLR,
		   &cfg_vf_pf_irq_clr, sizeof(cfg_vf_pf_irq_clr));
}

/**
 * cass_config_sriovt() - Configure access to a resource from a VF
 *
 * @dev: Cassini device
 * @enable: whether to enable or disable the access from the VF
 * @vf_num: which VF owns that resource (if enabled)
 * @res_index: resource index
 *
 * The resource index is either:
 *   -    0 + a transmit CQ id
 *   - 1024 + a target CQ id
 *   - 2048 + a CT id
 *   - 4096 + an EQ id
 */
static inline void cass_config_sriovt(struct cass_dev *dev,
				      bool enable,
				      unsigned int vf_num,
				      unsigned res_index)
{
	const union c_pi_cfg_sriovt cfg_sriovt = {
		.vf_en = enable,
		.vf_num = vf_num,
	};

	cass_write(dev, C_PI_CFG_SRIOVT(res_index),
		   &cfg_sriovt, sizeof(cfg_sriovt));
}

/* Initializes simple structures that are 8 bytes wide. */
#define CASS_INIT_SIMPLE_TABLE(up, lo) {			\
	unsigned int i;						\
	const union lo lo = {};					\
	_Static_assert(sizeof(lo) == 8, "Invalid size");	\
	for (i = 0; i < up ## _ENTRIES; i++)			\
		cass_write(dev, up(i), &lo, sizeof(lo));	\
	}

/*
 * Initializes simple structures that are larger than 8 bytes. Shadow
 * memory must be used.
 */
#define CASS_INIT_SHADOW_TABLE(up, lo, block_base) {		\
	unsigned int i;						\
	const union lo lo = {};					\
	_Static_assert(sizeof(lo) > 8, "Invalid size");			\
	cass_shadow_write_prepare(dev, block_base, &lo, sizeof(lo));	\
	for (i = 0; i < up ## _ENTRIES; i++)			\
		cass_shadow_write_do(dev, block_base, up ## _OFFSET(i));\
	}

/**
 * cass_init() - Mandatory initialization of a Cassini chip
 *
 * Used after a power on or a reset.
 */
static inline void cass_init(struct cass_dev *dev)
{
	unsigned int i;
	static const union {
		union c_rmu_cfg_portal_list rmu_cfg_portal_list;
		union c_rmu_cfg_ptlte_set_list rmu_cfg_ptlte_set_list;
		union c_cq_cfg_tg_thresh cq_cfg_tg_thresh;
		union c_lpe_cfg_get_ctrl lpe_cfg_get_ctrl;
	} u = {};
	static const union c_rmu_cfg_ptlte_set_list rmu_cfg_ptlte_set_list_mask = {
		.qw = { 0xfffffffffffffffULL, 0xfffffffffffffffULL,
			0xfffffffffffffffULL, 0xfffffffffffffffULL }
	};
	static const union c_ee_cfg_latency_monitor cfg_latency_monitor = {
		// TODO - these are working values.  They should be reviewed.
		.thresh_a      = 128,
		.thresh_b      = 196,
		.lt_limit      = 100,
		.lt_limit_a    =  40,
		.lt_limit_b    =  20,
		.granularity   =  50,
		.granularity_a =  20,
		.granularity_b =  10,
		.hysteresis_a  = 100,
		.hysteresis_b  = 150,
	};
	union c_ee_cfg_timestamp_freq cfg_timestamp_freq = {.clk_divider = 1000};
	union c_cq_cfg_ct_wb_mem_addr cfg_ct_wb_mem_addr;
	union c_mb_sts_rev rev;

	/*
	 * All CSRs that are accessible through the shadow space must
	 * be initialized.
	 *
	 * TODO: this is not complete, as some CSRs are not defined
	 * yet (at least PARBS and PI).
	 */

	CASS_INIT_SIMPLE_TABLE(C_RMU_CFG_PORTAL_INDEX_TABLE,
			       c_rmu_cfg_portal_index_table);
	CASS_INIT_SIMPLE_TABLE(C_RMU_CFG_PTLTE_SET_CTRL_TABLE,
			       c_rmu_cfg_ptlte_set_ctrl_table);
	CASS_INIT_SIMPLE_TABLE(C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE,
			       c_rmu_cfg_portal_index_indir_table);

	for (i = 0; i < C_RMU_CFG_PTLTE_SET_LIST_ENTRIES; i++) {
		cass_config_set_list(dev, i, 0, &u.rmu_cfg_ptlte_set_list,
					   &rmu_cfg_ptlte_set_list_mask);
		cass_invalidate_set_list(dev, i);
	}

	for (i = 0; i < C_RMU_CFG_VNI_LIST_ENTRIES; i++) {
		cass_config_matching_vni_list(dev, i, 0, 0);
		cass_invalidate_vni_list(dev, i);
	}

	for (i = 0; i < C_RMU_CFG_PORTAL_LIST_ENTRIES; i++) {
		cass_config_portal_list(dev, i, 0, &u.rmu_cfg_portal_list);
		cass_invalidate_portal_list(dev, i);
	}

	CASS_INIT_SHADOW_TABLE(C_ATU_CFG_AC_TABLE,
			       c_atu_cfg_ac_table, C_ATU_BASE);
	CASS_INIT_SIMPLE_TABLE(C_ATU_CFG_CQ_TABLE, c_atu_cfg_cq_table);

	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_CP_TABLE, c_cq_cfg_cp_table);
	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_CT_WB_MEM_ADDR, c_cq_cfg_ct_wb_mem_addr);

	CASS_INIT_SHADOW_TABLE(C_CQ_TXQ_BASE_TABLE, c_cq_txq_base_table,
			       C_CQ_BASE);
	CASS_INIT_SHADOW_TABLE(C_CQ_TGQ_TABLE, c_cq_tgq_table, C_CQ_BASE);
	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_CID_ARRAY, c_cq_cfg_cid_array);
	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_CT_RGID_TABLE, c_cq_cfg_ct_rgid_table);
	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_EQ_RGID_TABLE, c_cq_cfg_eq_rgid_table);
	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_AC_ARRAY, c_cq_cfg_ac_array);
	CASS_INIT_SIMPLE_TABLE(C_CQ_CFG_PTLTE_RGID_TABLE, c_cq_cfg_ptlte_rgid_table);

	CASS_INIT_SHADOW_TABLE(C_LPE_CFG_PTL_TABLE, c_lpe_cfg_ptl_table,
			       C_LPE_BASE);
	CASS_INIT_SHADOW_TABLE(C_LPE_DBG_MRR, c_lpe_dbg_mrr, C_LPE_BASE);
	CASS_INIT_SHADOW_TABLE(C_LPE_DBG_RGET_RAM, c_lpe_dbg_rget_ram, C_LPE_BASE);
	CASS_INIT_SHADOW_TABLE(C_LPE_STS_LIST_ENTRIES, c_lpe_sts_list_entries,
			       C_LPE_BASE);
	cass_write(dev, C_LPE_CFG_GET_CTRL, &u.lpe_cfg_get_ctrl,
		   sizeof(u.lpe_cfg_get_ctrl));

	CASS_INIT_SHADOW_TABLE(C_EE_CFG_EQ_DESCRIPTOR,
			       c_ee_cfg_eq_descriptor, C_EE_BASE);
	CASS_INIT_SIMPLE_TABLE(C_EE_CFG_LONG_EVNT_OVR_TABLE,
			       c_ee_cfg_long_evnt_ovr_table);
	CASS_INIT_SIMPLE_TABLE(C_EE_CFG_PERIODIC_TSTAMP_TABLE,
			       c_ee_cfg_periodic_tstamp_table);
	CASS_INIT_SIMPLE_TABLE(C_EE_DBG_ECB_SIDEBAND, c_ee_dbg_ecb_sideband);
	CASS_INIT_SHADOW_TABLE(C_EE_DBG_ECB, c_ee_dbg_ecb, C_EE_BASE);
	CASS_INIT_SIMPLE_TABLE(C_EE_DBG_TRNSLTN_RSP, c_ee_dbg_trnsltn_rsp);
	CASS_INIT_SIMPLE_TABLE(C_EE_CFG_EQ_SW_STATE, c_ee_cfg_eq_sw_state);
	CASS_INIT_SHADOW_TABLE(C_EE_CFG_STS_EQ_HW_STATE,
			       c_ee_cfg_sts_eq_hw_state, C_EE_BASE);
	cass_write(dev, C_EE_CFG_LATENCY_MONITOR, &cfg_latency_monitor,
		   sizeof(cfg_latency_monitor));
	cass_write(dev, C_EE_CFG_TIMESTAMP_FREQ, &cfg_timestamp_freq,
		   sizeof(cfg_timestamp_freq));

	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SCT_RAM0, c_pct_cfg_sct_ram0);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SCT_RAM1, c_pct_cfg_sct_ram1);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SCT_RAM2, c_pct_cfg_sct_ram2);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SCT_RAM3, c_pct_cfg_sct_ram3);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SCT_RAM4, c_pct_cfg_sct_ram4);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SMT_RAM0, c_pct_cfg_smt_ram0);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SMT_RAM1, c_pct_cfg_smt_ram1);
	CASS_INIT_SHADOW_TABLE(C_PCT_CFG_SPT_RAM0, c_pct_cfg_spt_ram0, C_PCT_BASE);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SPT_RAM1, c_pct_cfg_spt_ram1);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_SPT_RAM2, c_pct_cfg_spt_ram2);
	CASS_INIT_SIMPLE_TABLE(C_PCT_CFG_TCT_RAM, c_pct_cfg_tct_ram);

	cass_read(dev, C_MB_STS_REV, &rev, sizeof(rev));

	if (rev.vendor_id == 0x17db && rev.device_id == 0x501) {
		/* See Cassini ERRATA-3248. CTs can be disabled due to a bug
		 * unless C_CQ_CFG_CT_WB_MEM_ADDR has been read once.
		 * Must read back one CT MEM ADDR. */
		cass_read(dev, C_CQ_CFG_CT_WB_MEM_ADDR(0), &cfg_ct_wb_mem_addr,
			  sizeof(cfg_ct_wb_mem_addr));

		CASS_INIT_SIMPLE_TABLE(C1_PCT_CFG_TRS_RAM0,
				       c_pct_cfg_trs_ram0);
		CASS_INIT_SHADOW_TABLE(C1_PCT_CFG_TRS_RAM1,
				       c_pct_cfg_trs_ram1, C_PCT_BASE);
	} else {
		CASS_INIT_SIMPLE_TABLE(C2_PCT_CFG_TRS_RAM0,
				       c_pct_cfg_trs_ram0);
		CASS_INIT_SHADOW_TABLE(C2_PCT_CFG_TRS_RAM1,
				       c_pct_cfg_trs_ram1, C_PCT_BASE);
	}

	CASS_INIT_SHADOW_TABLE(C_MST_DBG_MST_TABLE, c_mst_dbg_mst_table, C_MST_BASE);
}

#undef CASS_INIT_SIMPLE_TABLE
#undef CASS_INIT_SHADOW_TABLE

#endif	/* __CXI_CASSINI_H */
