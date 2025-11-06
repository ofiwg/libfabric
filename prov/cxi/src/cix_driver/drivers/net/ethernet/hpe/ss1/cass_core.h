/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020-2024,2025 Hewlett Packard Enterprise Development LP */

#ifndef _CASS_CORE_H
#define _CASS_CORE_H

#include <linux/device.h>
#include <linux/iova.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/pci.h>
#include <linux/rbtree.h>
#include <linux/version.h>
#include <linux/workqueue.h>
#include <net/netlink.h>
#include <net/genetlink.h>
#include <linux/ptp_clock_kernel.h>
#include <linux/etherdevice.h>
#include <linux/completion.h>
#include <linux/hrtimer.h>
#include <linux/hwmon.h>
#include <linux/socket.h>

#include "cxi_link.h"
#include "cxi_core.h"
#include "cuc_cxi.h"	/* from firmware_cassini/ */
#include "cass_sbl.h"
#include "pldm_cxi.h"
#include "cass_sl.h"
#include "cxi_qos_profiles.h"
#include "cass_rx_tx_profile.h"
#include "cass_rgroup.h"
#include "cxi_config.h"

#define PCI_VENDOR_ID_CRAY      0x17db
#define PCI_DEVICE_ID_CASSINI_1 0x0501
#define PCI_VENDOR_ID_HPE       0x1590
#define PCI_DEVICE_ID_CASSINI_2 0x0371

#define CXI_MODULE_NAME "cxi_ss1"

/* The following are IPv6 next header values used for IPv6 extension headers.
 * https://www.iana.org/assignments/ipv6-parameters/ipv6-parameters.xhtml
 */
#define IPV6_EH_HOPOPTS 0
#define IPV6_EH_ROUTE 43
#define IPV6_EH_FRAG 44
#define IPV6_EH_ESP 50
#define IPV6_EH_AH 51
#define IPV6_EH_OPTS 60
#define IPV6_EH_MOBILITY 135
#define IPV6_EH_HIP 139
#define IPV6_EH_SHIM6 140
#define IPV6_EH_RSVD1 253
#define IPV6_EH_RSVD2 254

#define HW_PLATFORM(_hw) ((_hw)->rev.platform)
#define HW_PLATFORM_ASIC(_hw) (HW_PLATFORM(_hw) == C_PLATFORM_ASIC)
#define HW_PLATFORM_Z1(_hw) (HW_PLATFORM(_hw) == C_PLATFORM_Z1)
#define HW_PLATFORM_NETSIM(_hw) (HW_PLATFORM(_hw) == C_PLATFORM_NETSIM)

#define C_MAX_CSR_ERR_INFO 3
#define C_MAX_CSR_CNTRS   3

/* HPC MTU is fixed at 2k. The largest HPC header is 64 bytes. */
#define C_MAX_HPC_MTU (2048 + 64)

/* PCI BARs containing the CSRs and CQ/EQ areas. */
#define MMIO_BAR 0

/* Expected length of PCI BAR 0 for a Physical Function. Used to differentiate
 * between Virtual Function (512M BAR 0) and Physical Function (2G BAR 0) under
 * SR-IOV.
 */
#define MMIO_BAR_LEN_PF 2147483648

/* Number of error interrupts */
#define NUM_ERR_INTS 30

#define PORTALS_MAX_FRAME_SIZE 2111U
#define ETHERNET_MAX_FRAME_SIZE 9022U
#define TSC_DEFAULT_FILL_QTY 16U

#define PLEC_SIZE 256U

enum cass_phy_state {
	CASS_PHY_DOWN,
	CASS_PHY_READY,
	CASS_PHY_HALTED,
	CASS_PHY_UP,
	CASS_PHY_RUNNING,
	CASS_PHY_NOLINK,
	CASS_PHY_HEADSHELL_REMOVED,
};

#ifdef pr_fmt
#undef pr_fmt
#endif
#define pr_fmt(fmt) KBUILD_MODNAME ":%s:%d " fmt, __func__, __LINE__

#define atu_debug(fmt, ...) \
do {						\
	if (more_debug)				\
		pr_debug(fmt, ##__VA_ARGS__);	\
} while (0)

#if defined(RHEL8_7)
static inline u64 ALLOC_IOVA_FAST(struct iova_domain *iovad, unsigned long len,
				  unsigned long pfn)
{
	struct iova *tiova = alloc_iova(iovad, len, pfn, true);

	if (!tiova)
		return 0;

	return tiova->pfn_lo;
};

#define FREE_IOVA_FAST(iovad, pfn, len) free_iova(iovad, pfn)
#else

#define ALLOC_IOVA_FAST(iovad, len, end) alloc_iova_fast(iovad, len, end, true)
#define FREE_IOVA_FAST free_iova_fast

#endif /* ! RHEL8_7 */

#define INIT_IOVA_DOMAIN(iovad, ps, base, end) \
		init_iova_domain(iovad, ps, base)

#define ATU_HMM_PFN_TO_PFN(hpfn) page_to_pfn(hmm_pfn_to_page(hpfn))

#define ATU_PFN_INVALID(hpfn) \
	(!(hpfn & HMM_PFN_VALID) || \
	 !(hpfn & HMM_PFN_WRITE) || \
	  (hpfn & HMM_PFN_ERROR))
#define ATU_PFN_WRITE(hpfn) ((hpfn & HMM_PFN_WRITE))
#define ATU_PFN_VALID(hpfn) ((hpfn & HMM_PFN_VALID))

#if (defined(RHEL_MAJOR) && (RHEL_MAJOR == 8 && RHEL_MINOR >= 7))
/* pin_user_pages_fast & co. have been backported */
#elif LINUX_VERSION_CODE < KERNEL_VERSION(5, 5, 0)
#define pin_user_pages_fast get_user_pages_fast
#define unpin_user_pages put_user_pages
#define unpin_user_page put_user_page
#endif

#if (KERNEL_VERSION(5, 7, 0) > LINUX_VERSION_CODE) && \
	!defined(SLES15SP3) && !defined(RHEL_MAJOR)
static inline void mmap_read_lock(struct mm_struct *mm)
{
	down_read(&mm->mmap_sem);
}

static inline void mmap_read_unlock(struct mm_struct *mm)
{
	up_read(&mm->mmap_sem);
}
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 5, 0) || \
    (defined(RHEL_MAJOR) && (RHEL_MAJOR == 9 && RHEL_MINOR >= 4))
/* AER is automatically enabled by the PCI core. */
static inline int pci_enable_pcie_error_reporting(void *p) { return 0; }
static inline void pci_disable_pcie_error_reporting(void *p) {}
#endif

#define ATU_FAULT_RETRY_MAX 16
#define ATU_PTE_PTR (1 << 6)
#define ATU_PTE_READ (1UL << 57)
#define ATU_PTE_WRITE (1UL << 58)
#define ATU_PTE_LEAF (1UL << 59)
#define ATU_PTE_PRESENT (1UL << 60)
#define ATU_HMM_RANGE_TIMEOUT 1000
#define ATU_PTE_MASK (~(ATU_PTE_READ | ATU_PTE_WRITE | ATU_PTE_LEAF | \
				ATU_PTE_PRESENT))

#define MASK(x) (BIT(x) - 1)
#define ATU_PTEPTR_SHIFT 6
#define PTE_ADDR(pte) ((pte).ptr << ATU_PTEPTR_SHIFT)
#define pfn2phys __pfn_to_phys
#define PHYS_TO_PAGE(pa) pfn_to_page(__phys_to_pfn(pa))

#define C_ADDR_SHIFT 12
#define ATU_PTE_ENTRY_SIZE    sizeof(union a_pte)
#define flsl(x) (fls64(x) - 1)
#define ffsl(x) (fls64((x) & -(x)) - 1)
#define ATUCQ_TIMEOUT (odp_sw_decouple ? decouple_timeout_us : atucq_timeout_us)
#define ATU_IBW_TIMEOUT (5 * MSEC_PER_SEC * USEC_PER_MSEC)
#define ATU_ODPQ_SPACE_TIMEOUT (100 * USEC_PER_MSEC)
#define ATU_STS_ATUCQ_IDLE_TIMEOUT (10 * MSEC_PER_SEC * USEC_PER_MSEC)
#define ATU_STS_IBW_IDLE_TIMEOUT (1000 * MSEC_PER_SEC * USEC_PER_MSEC)
#define ATUCQ_ENTRY(index) (index & (C_ATU_CFG_CQ_TABLE_ENTRIES - 1))
/* Based on the AC base page size */
#define ATU_PMD_ORDER(cac) (cac->huge_shift - cac->page_shift)
#define ATU_PMD_NR(cac) BIT(ATU_PMD_ORDER(cac))
#define ATU_PTE_ORDER(cac) (cac->page_shift - PAGE_SHIFT)
#define ATU_PTE_NR(cac) BIT(ATU_PTE_ORDER(cac))
#define PFN_INC(cac, leaf) (leaf ? ATU_PMD_NR(cac) : 1)
/* Based on the kernel base page size */
#define K_PMD_ORDER(cac) (cac->huge_shift - PAGE_SHIFT)
#define K_PMD_NR(cac) BIT(K_PMD_ORDER(cac))
#define KPFN_INC(cac, leaf) (leaf ? K_PMD_NR(cac) : ATU_PTE_NR(cac))
#define ATU_FLAGS_MASK ~(CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_ALLOC_MD | \
			 CXI_MAP_FAULT | CXI_MAP_PREFETCH)
#define ATUCQ_INVALIDATE_ALL 0x200000000000000UL
#define MAX_PG_TABLE_SIZE 15
#define MIN_PG_TABLE_SIZE 3
#define MIN_DEVICE_PG_TABLE_SIZE (MIN_PG_TABLE_SIZE + 2)
#define INVALID_DMABUF_FD (-1)

/* Reserve an AC for physical mapping to be shared by all processes */
#define ATU_PHYS_AC (C_ATU_CFG_AC_TABLE_ENTRIES - 1)
#define ATU_NTA_DEF_PI_ACXT ATU_PHYS_AC
#define ATU_FLUSH_REQ_DELAY 0x3ff

#ifndef VMA_ITERATOR
#define for_each_vma(vmi, vma) for (; vma; vma = vma->vm_next)
#endif /* VMA_ITERATOR */

#define CASS_NUM_LE_POOLS (C_LPE_CFG_PE_LE_POOLS_ENTRIES / C_PE_COUNT)
#define CASS_MIN_POOL_TLES 8
#define ACS_AVAIL (ATU_PHYS_AC - 1)
#define CTS_AVAIL (C_NUM_CTS - 1) /* CT 0 invalid */
#define EQS_AVAIL (C_NUM_EQS - 1) /* EQ 0 invalid */

#define CASS_MAX_IRQ_NAME 24

/* Current devices have up to 31 sensors */
#define CASS_MAX_SENSORS 32

#define CASS2_MAX_PACKETS_INFLIGHT 128U
#define CASS1_MAX_PACKETS_INFLIGHT 64U

#define DEFAULT_LE_POOL_ID 0
#define DEFAULT_TLE_POOL_ID 0

enum cass_link_speed {
	CASS_SPEED_UNKNOWN = PCI_SPEED_UNKNOWN,
	CASS_SPEED_2_5GT = PCIE_SPEED_2_5GT,
	CASS_SPEED_5_0GT = PCIE_SPEED_5_0GT,
	CASS_SPEED_8_0GT = PCIE_SPEED_8_0GT,
	CASS_SPEED_16_0GT = PCIE_SPEED_16_0GT,
	CASS_SPEED_32_0GT = PCIE_SPEED_32_0GT,

	CASS_SPEED_20_0GT,
	CASS_SPEED_25_0GT,
};

/* Cassini clock frequency */
#define C1_CLK_FREQ_HZ 1000000000ULL /* 1 GHz */
#define C2_CLK_FREQ_HZ 1100000000ULL /* 1.1 GHz */

extern const struct file_operations uc_fops;

struct ac_map_opts {
	u64 va_start;
	u64 va_end;
	size_t va_len;
	u32 flags;
	u64 iova;
	dma_addr_t *dma_addrs;
	int page_shift;
	int huge_shift;
	int ptg_mode;
	bool hugepage_test;
	bool is_huge_page;
	struct cxi_md_priv *md_priv;
};

union a_pte {
	struct c_page_table_entry pte;
	u64 qw;
};

struct l1_entry {
	/* The array of PTEs */
	struct page *l1_pages;
	dma_addr_t l1_pages_dma_addr;

	/* Bitmap of used PTEs */
	unsigned long *bitmap;
};

struct cass_nta {
	/* nta root page table pointer virtual address */
	union a_pte *root_ptr;
	struct page *root_page;
	dma_addr_t root_page_dma_addr;
	int root_order;
	int l1_order;
	int numa_node;

	/* Each L0 PDE has its own bitmap of allocated L1 PTE. */
	int l0_entries;
	int l1_entries;
	struct l1_entry l1[];
};

struct cass_ac {
	struct cxi_ac ac;
	struct list_head list;
	struct list_head md_list;
	struct mutex md_mutex;
	struct cxi_lni_priv *lni_priv;

	struct mutex ac_mutex;

	/* copy of the atu ac struct */
	union c_atu_cfg_ac_table cfg_ac;
	u32 flags;

	int page_shift;
	int huge_shift;
	int ptg_mode;
	bool hugepage_test;

	struct cass_nta *nta;

	u64 iova_base;
	u64 iova_end;
	u64 iova_len;

	struct iova_domain *iovad;
	struct dentry *debug_dir;
	struct dentry *lni_dir;
	refcount_t refcount;
};

struct cass_p2p_ops {
	int (*get_pages)(struct ac_map_opts *m_opts);
	void (*put_pages)(struct cxi_md_priv *md_priv);
	int (*is_device_mem)(uintptr_t va, size_t len, int *page_shift);
};

/* Value written by the ATU CQ in wait_rsp_data when completing
 * commands.
 */
#define WAIT_RSP_DATA 0xf00fbeefULL

struct cass_atu_cq {
	/* Protects access to the ATU CQ and wait_rsp_data */
	struct mutex atu_cq_mutex;
	struct mutex atu_ib_mutex;

	struct wait_rsp_data {
		u64 cmp ____cacheline_aligned;
		u64 ib ____cacheline_aligned;
	} wait_rsp_data;

	dma_addr_t rsp_dma_addr;
	dma_addr_t cmp_wait_rsp_addr;
	dma_addr_t ib_wait_rsp_addr;

	char cmpl_wait_int_name[CASS_MAX_IRQ_NAME];
};

struct sts_idle {
	struct cass_dev *hw;
	bool ib_wait;
};

#define MAX_VFMSG_SIZE 122

/* A VF to PF message header */
struct vf_pf_msg_hdr {
	unsigned int len;	/* whole message length */
	int rc;			/* return code */
};

/* Traffic Class descriptions. */
struct cass_tc_cfg {
	struct list_head tc_entry;

	unsigned int tc;
	enum cxi_traffic_class_type tc_type;

	/* Restricted request DSCP. */
	unsigned int res_req_dscp;

	/* Unrestricted request DSCP. */
	unsigned int unres_req_dscp;

	/* Restricted response DSCP. */
	unsigned int res_rsp_dscp;

	/* Unrestricted response DSCP. */
	unsigned int unres_rsp_dscp;

	/* Request PCP. */
	unsigned int req_pcp;

	/* Response PCP. */
	unsigned int rsp_pcp;

	/* CQ OCU Set Index */
	unsigned int ocuset;

	/* Traffic class value specific to CQ configuration. */
	unsigned int cq_tc;

	/* Buffer Class */
	unsigned int req_bc;
};

struct cass_rsrc_info {
	u16 res;
	u16 shared_total;
	u16 shared_in_use;
};

struct cass_irq {
	struct atomic_notifier_head nh;
	int idx;
	struct cpumask mask;
	char irq_name[CASS_MAX_IRQ_NAME];
	atomic_t refcount;
	int vec;
	struct mutex lock;
};

/* Maximum size of an error flag CSR. For Cassini-1, they are all 64
 * bits. For Cassini-2, C_HNI_PML_ERR_FLG is 256 bits long.
 */
#define MAX_ERR_FLAG_BITLEN 256
union err_flags {
	DECLARE_BITMAP(mask, MAX_ERR_FLAG_BITLEN);
	union c_pi_err_flg pi;
	union c_pi_ext_err_flg pi_ext;
	union c_pi_ipd_ext_err_flg pi_ipd_ext;
	union c_atu_err_flg atu;
	union c_atu_ext_err_flg atu_ext;
	union c1_hni_err_flg c1_hni;
	union c2_hni_err_flg c2_hni;
	union c2_hni_ext_err_flg c2_hni_ext;
	union c1_hni_pml_err_flg c1_hni_pml;
	union ss2_port_pml_err_flg ss2_port_pml;
	union c_rmu_err_flg rmu;
	union c_lpe_err_flg lpe;
	union c_pct_ext_err_flg pct_ext;
	union c_ee_err_flg ee;
};

struct cxi_reg_err_flg {
	struct list_head list;
	unsigned int irq;
	bool is_ext;
	union err_flags err_flags;
	/* Callback. irq is the MSI-X interrupt index (0 to 29), and
	 * is_ext indicates whether the flag is set in the ERR_FLG or
	 * EXT_ERR_FLG CSR. bitn is the bit index in abovementioned
	 * CSR.
	 */
	void (*cb)(struct cass_dev *hw, unsigned int irq,
		   bool is_ext, unsigned int bitn);
};

/* For error CSR processing */
struct flg_err_info;
struct csr_info {
	const char *csr_name;
	const char *csr_name_lo; /* same, in lower case */

	/* Various CSR offsets for that error. */
	unsigned int flg;
	unsigned int mask;
	unsigned int clr;
	unsigned int first_flg;
	unsigned int first_flg_ts;
	unsigned int bitlen;
	const struct flg_err_info *flg_info;
};

extern const struct file_operations decouple_stats_fops;
extern const struct file_operations sw_decouple_fops;

/*
 * there currently are not that many entities that need to reserve
 * sets of dma descriptors.
 *
 * Currently, cassini telemetry/counters (with test/experimental code)
 * uses five dmac descriptor sets and rosetta link monitoring (lmon)
 * may use two dma descriptor sets.
 *
 */
#define DMAC_DESC_SET_COUNT			16

struct dmac_desc_set {
	u16 count;      /* total number of descs */
	u16 index;      /* index of first desc */
	u16 numused;    /* number of descs written */
	const char *name;	/* descriptive name to identify use */
};

struct cass_vf {
	struct task_struct *task;
	struct socket *sock;

	/* Index and back-pointer to hardware struct for use by VF handler thread */
	struct cass_dev *hw;
	int vf_idx;

	/* Per-VF message buffers */
	char request[MAX_VFMSG_SIZE];
	char reply[MAX_VFMSG_SIZE];
};

/* Private hardware data for Cassini. This is not seen by clients. */
struct cass_dev {
	/* Embed a cxi device. */
	struct cxi_dev cdev;
	struct cxi_cfg_group_item *cfg_dev_dir;

	/* Register base in BAR0 */
	void __iomem *regs;
	unsigned long regs_base;

	/* /sys/class interface */
	struct device class_dev;

	/* Device type and revision */
	union c_mb_sts_rev rev;
	bool reduced_pct_tables;

	/* TODO: For development only. Netsim VFs are not really
	 * supported.
	 */
	bool with_vf_support;

	/* Number of VFs currently configured in the PF. */
	unsigned int num_vfs;

	/* Interrupt vector used for the PF/ VF communication */
	unsigned int pf_vf_vec;
	char pf_vf_int_name[CASS_MAX_IRQ_NAME];
	struct completion pf_to_vf_comp;

	/* Virtual function communication socket for SR-IOV. Used by PF to
	 * listen for incoming VF connections, used by VF to initiate connection
	 */
	struct socket *vf_sock;
	struct task_struct *vf_listener;
	struct cass_vf vfs[C_NUM_VFS];
	cxi_msg_relay_t msg_relay;
	void *msg_relay_data;
	struct mutex msg_relay_lock;

	/* PCIe info */
	bool esm_active;
	u8 pcie_link_speed;
	bool pci_disabled;

	struct {
		/* Monitoring */
		struct delayed_work task;

		/* Acceptable number of correctable errors per minute */
		unsigned int corr_err_min;

		/* PCIe Vendor Specific Information position */
		unsigned int event_counter_control;
		unsigned int event_counter_data;

		/* Number of correctable and uncorrectable errors seen. */
		u64 uncorr_err;
		u64 corr_err;
	} pcie_mon;

	/* Error interrupts handling. Represent each of the 30 error
	 * interrupts by a bit in err_irq_mask.
	 */
	unsigned long err_irq_mask;
	unsigned long err_irq_raised;
	int err_irq_vecs[NUM_ERR_INTS];
	char err_int_names[NUM_ERR_INTS][CASS_MAX_IRQ_NAME];
	struct workqueue_struct *err_irq_wq;
	struct work_struct err_irq_work;
	struct kobject *err_flgs_dir_kobj[2]; /* IRQA and IRQB */
	const struct csr_info (*err_handlers)[NUM_ERR_INTS][2];

	spinlock_t sfs_err_flg_lock; /* protects mask update */
	struct sfs_err_flg {
		struct kobject kobj;

		/* IRQ A mask for the error flag CSR. */
		DECLARE_BITMAP(err_irq_mask, MAX_ERR_FLAG_BITLEN);
		unsigned int bitlen;

		/* Which bits must never be automatically masked */
		union err_flags no_auto_mask;

		/* Which bits must never be printed in logs. Some error bits
		 * can be raised during normal operation (eg. UC_INTERRUPT)
		 * and should not appear in logs.
		 */
		union err_flags no_print_mask;
	} sfs_err_flg[NUM_ERR_INTS][2];

	/* Clients registered with the error interrupt handler, for
	 * certain bits
	 */
	struct mutex err_flg_mutex;
	struct list_head err_flg_list;

	/* Temporary storage for CSR error processing. As there is
	 * only one work at any given time, there is no race to access
	 * these.
	 */
	struct {
		struct err_info {
			u64 data[5];
			int count;
		} err_info[C_MAX_CSR_ERR_INFO];
		unsigned int err_info_count;
		u64 cntrs_val[C_MAX_CSR_CNTRS];
		unsigned int err_cntrs_count;
		char err_info_str[100];
		DECLARE_BITMAP(first_flg, MAX_ERR_FLAG_BITLEN);
		union c_any_err_first_flg_ts first_flg_ts;
	};

	/* Each error is rate limited. If too many interrupts happen
	 * for the same error bit, then the bit will be permanently
	 * masked.
	 */
	struct ratelimit_state err_rl[NUM_ERR_INTS][MAX_ERR_FLAG_BITLEN];

	/* Debugfs directories for that device */
	struct dentry *debug_dir;
	struct dentry *port_dir;
	struct dentry *stats_dir;
	struct dentry *lni_dir;
	struct dentry *atu_dir;
	struct dentry *decouple_dir;

	/* Count users of this device. */
	refcount_t refcount;

	/* Keep track of NIs. TODO: keep as that may help debugging,
	 * but should be removed eventually.
	 */
	struct list_head lni_list;
	spinlock_t lni_lock;

	struct {
		atomic_t lni;
		atomic_t domain;
		atomic_t eq;
		atomic_t txq;
		atomic_t tgq;
		atomic_t pt;
		atomic_t ct;
		atomic_t ac;
		atomic_t md;
	} stats;

	struct dentry *domain_dir;
	struct dentry *eq_dir;
	struct dentry *cq_dir;
	struct dentry *pt_dir;
	struct dentry *ct_dir;
	struct dentry *ac_dir;

	/* Pending LNI cleanups */
	struct list_head lni_cleanups_list;
	struct delayed_work lni_cleanups_work;
	struct mutex lni_cleanups_lock;

	/* Number of IRQs available to that device. */
	unsigned int num_irqs;

	unsigned int atu_cq_vec;
	unsigned int atu_pri_vec;

	/* Completion IRQs */
	unsigned int num_comp_irqs;
	struct cass_irq *comp_irqs;

	/* RMU table allocation bitmap. Each entry in the RMU table
	 * contains a different VNI.
	 */
	struct ida rmu_index_table;
	struct ida domain_table;
	spinlock_t rmu_lock;
	struct rb_root rmu_tree;

	/* Domains. domain_tree contains every registered combinations
	 * of VNI+PID as these must be unique.
	 */
	spinlock_t domain_lock;
	struct rb_root domain_tree;

	/* EQ allocation bitmap */
	spinlock_t eq_shadow_lock;
	struct ida eq_index_table;
	struct mutex init_eq_hw_state;

	/* Which EQ is registered with the PCT block. Protected by
	 * init_eq_hw_state.
	 */
	unsigned int pct_eq_n;

	/* Communication Profile Table (CPT) and containers. The table,
	 * cpt_index_table, resides on the adapter and is used to reserve
	 * one of the hardware resource. The RB tree, cp_cont_tree, is used to
	 * store the allocated communication containers. Each container is
	 * identified by a VNI and traffic class. Each container may contain
	 * multiple communication profiles which map to an index in the CPT.
	 * The lock, cp_lock, protects the tree.
	 */
	struct ida cp_table;
	struct mutex cp_lock;

	/* Write once, read only list of configured traffic classes. */
	struct list_head tc_list;

	/* Next resource to be used for traffic class configuration. */
	unsigned int next_oxe_bc;
	unsigned int next_oxe_branch_bucket;
	unsigned int next_oxe_leaf_bucket;
	unsigned int next_cq_tc;
	unsigned int next_cq_ocuset;
	unsigned int next_cq_ocu;
	unsigned int next_ixe_wrq;
	unsigned int next_ixe_fq;
	unsigned int next_lpe_fq;

	u8 tsc_fill_qty;
	struct qos_profile qos;
	struct kobject tcs_kobj;

	/* CQ allocation bitmaps. */
	spinlock_t cq_shadow_lock;
	struct mutex cq_init_lock;
	struct ida cq_table;

	/* CT allocations */
	struct mutex ct_init_lock;

	/* Multicast uniqueness table */
	struct ida multicast_table;

	/* PT allocation bitmap */
	spinlock_t lpe_shadow_lock;
	struct ida pte_table;
	struct ida pt_index_table;
	unsigned int dmac_pt_id;
	struct mutex mst_table_lock;
	struct mutex pte_transition_sm_lock;
	union c_mst_dbg_mst_table *mst_entries;
	dma_addr_t mst_entries_dma_addr;
	atomic_t plec_count;

	/* LE pool allocation bitmap */
	struct ida le_pool_ids[C_PE_COUNT];
	struct ida tle_pool_ids;

	/* Ethernet Set List */
	struct ida set_list_table;
	struct cxi_tx_profile eth_tx_profile;

	/* Protects C_RMU_CFG_PORTAL_LIST_X/Y. */
	spinlock_t rmu_portal_list_lock;

	/* RGID management */
	struct xarray rgid_array;
	refcount_t rgids_refcount;
	struct ida lni_table;

	/* ATU info */
	void *oxe_dummy_addr;
	dma_addr_t oxe_dummy_dma_addr;
	struct cass_ac *cac_table[C_ATU_CFG_AC_TABLE_ENTRIES];
	struct ida atu_table;
	struct cass_atu_cq atu_cq;
	spinlock_t atu_shadow_lock;
	atomic_t atu_error_inject;
	atomic_t atu_odp_requests;
	atomic_t atu_odp_fails;
	atomic_t atu_prb_expired;
	char odp_pri_int_name[CASS_MAX_IRQ_NAME];
	int pri_rd_ptr;
	struct work_struct pri_work;
	struct workqueue_struct *pri_wq;
	struct c_page_request_entry *page_request_table;
	dma_addr_t prt_dma_addr;
	struct cxi_reg_err_flg atu_err;
	struct work_struct prb_work;
	struct workqueue_struct *prb_wq;
	struct mutex odpq_mutex;
	struct list_head mm_lni_list;
	struct mutex mm_mutex;
	struct ida md_index_table;
	bool ac_filter_disabled;
	/* Flag to indicate that the IOMMU callback is in place to support
	 * C1 ODP.
	 */
	bool ats_c1_odp_enable;

	/* Number of invalidations due to memory deregistration */
	atomic_t dcpl_md_clear_inval;
	/* Number of invalidations due to the mmu notifier for NTA mode */
	atomic_t dcpl_nta_mn_inval;
	/* Number of invalidations due to the mmu notifier (via IOMMU callback)
	 * for ATS mode
	 */
	atomic_t dcpl_ats_mn_inval;
	/* Completion wait count */
	atomic_t dcpl_comp_wait;
	/* ODP decoupling entered - Completion wait has timed out */
	atomic_t dcpl_entered;
	/* ixe_epoch_x_cntr was 0 */
	atomic_t dcpl_ixe_cntr_0;
	/* ixe_epoch_x_cntr was not initially 0 but it decremented to 0 */
	atomic_t dcpl_ixe_cntr_dec_0;
	/* ee_epoch_x_cntr count of entering dec loop */
	atomic_t dcpl_ee_cntr_dec_0;
	/* ee_epoch_x_cntr is stuck */
	atomic_t dcpl_ee_cntr_stuck;
	/* oxe_epoch_x_cntr count of entering dec loop */
	atomic_t dcpl_oxe_cntr_dec_0;
	/* oxe_epoch_x_cntr is stuck */
	atomic_t dcpl_oxe_cntr_stuck;
	/* ixe_epoch_x_cntr is stuck after hw decoupling enabled */
	atomic_t dcpl_ixe_cntr_stuck;
	/* decouple process successful after hw decoupling enabled */
	atomic_t dcpl_success;
	/* ib_epoch_x_cntr decremented to 0 in step 6.2 */
	atomic_t dcpl_ibw_cntr_dec_0_62;
	/* ib_epoch_x_cntr count of dec loops in step 6.2 */
	atomic_t dcpl_ibw_cntr_dec_0_62_count;
	/* ib_epoch_x_cntr decremented to 0 in step 6.4 */
	atomic_t dcpl_ibw_cntr_dec_0_64;
	/* ib_epoch_x_cntr count of dec loops in step 6.4 */
	atomic_t dcpl_ibw_cntr_dec_0_64_count;
	/* ib_epoch_x_cntr is stuck */
	atomic_t dcpl_step7;
	/* IBW is active and ib_epoch_x_cntr is stuck */
	atomic_t dcpl_ibw_active_stuck;
	/* How long waiting to get the lock */
	atomic_t dcpl_ibw_idle_wait;
	/* IBW counter is inside of trylock loop */
	atomic_t dcpl_ibw_cntr_is_0;
	/* Decouple issued inbound wait */
	atomic_t dcpl_ibw_issued;
	/* Maximum time spent in decoupling */
	ktime_t dcpl_max_time;
	/* Histogram of time spent in decoupling */
	#define DBINS 12
	#define FIRST_BIN (64 * 1024)
	int dcpl_time[DBINS];
	/* Maximum time spent finding an MD */
	ktime_t pri_max_md_time;
	/* Histogram of time spent in finding MD during a fault */
	#define MDBINS 8
	#define FIRST_MDBIN (8 * 1024)
	int pri_md_time[MDBINS];
	/* Maximum time spent servicing a fault */
	ktime_t pri_max_fault_time;
	/* Histogram of time spent servicing a fault */
	#define FBINS 7
	#define FIRST_FBIN (8 * 1024)
	int pri_fault_time[FBINS];

	/* Counting event bitmap. */
	struct ida ct_table;

	/* Precision Time Protocol Clock */
	struct ptp_clock_info ptp_info;
	struct ptp_clock *ptp_clock;
	spinlock_t rtc_lock;

	/* MST locks. */
	spinlock_t mst_match_done_lock;
	struct mutex mst_update_lock;

	/* Micro-controller */
	bool uc_present;
	int uc_platform;	/* CUC_BOARD_TYPE_xxx */
	struct cxi_reg_err_flg uc_err;
	struct completion uc_attention0_comp;
	struct cuc_pkt uc_req;
	struct cuc_pkt uc_resp;
	struct workqueue_struct *uc_attn1_wq;
	struct work_struct uc_attn1_work;
	struct mutex uc_mbox_mutex;
	s8 uc_nic;		/* NIC index from uC point of view */
	u8 default_mac_addr[ETH_ALEN];
	struct kobject uc_kobj;
	char *fw_versions[7];	/* indexed by enum casuc_fw_target */
	bool fru_info_valid;	/* FRU query was successful; do not query again. */
	char *fru_info[PLDM_FRU_FIELD_VENDOR_IANA + 1];
	struct idr pldm_sensors;
	unsigned int pldm_sensors_count;
	struct kobject *pldm_sensors_kobj;
	unsigned int uc_dmac_id;
	struct kobject port_kobj;
	struct kobject port_num_kobj;
	bool qsfp_beacon_active;
	bool qsfp_over_temp;
	bool qsfp_bad;

	/* HW mon */
	struct {
		struct device *dev;
		struct hwmon_chip_info chip_info;
		const struct hwmon_channel_info *all_types[hwmon_max];

		struct hwmon_channel_info info[5];

		/* Enough room for 30 sensors. */
		u32 config[CASS_MAX_SENSORS + hwmon_max];
	} hwmon;

	/* RX and TX Profiles */
	struct cxi_rxtx_profile_list rx_profile_list;
	struct cxi_rxtx_profile_list tx_profile_list;
	struct mutex tx_profile_get_lock;
	struct mutex rx_profile_get_lock;

	/* Services */
	struct mutex svc_lock;
	struct idr svc_ids;
	struct dentry *svc_debug;
	struct list_head svc_list;
	unsigned int svc_count;
	struct cxi_resource_use resource_use[CXI_RESOURCE_MAX];
	spinlock_t rgrp_lock;

	/* Resource Groups */
	struct cxi_rgroup_list rgroup_list;

	/* HNI */
	struct cxi_reg_err_flg hni_pause_err;
	struct work_struct pause_timeout_work;
	struct cxi_reg_err_flg hni_uncor_err;
	struct cxi_reg_err_flg hni_pml_uncor_err;

	/* PML */
	struct cxi_reg_err_flg pml_err;
	struct {
		spinlock_t lock;
		enum cass_phy_state state;
		struct delayed_work state_queue;
		atomic_t cancel_state_machine;
	} phy;

	u32 max_eth_rxsize;

	/* Ethernet TX timestamp shift. 0 to 3, stays at zero but
	 * could become a kernel parameter.
	 */
	int tx_timestamp_shift;

	/* base link support */
	struct sbl_inst *sbl;           /* slingshot base-link instance */
	atomic_t *sbl_counters;         /* sbl link counters */
	bool link_config_dirty;         /* link config != running config */
	spinlock_t sbl_state_lock;      /* link state lock */
	struct cass_port *port;         /* port db array */
	const struct cxi_link_ops *link_ops;

	/* SBus config */
	struct mutex sbus_mutex;      /* serialise SBus config accesses */

	/* Rendezvous get control config */
	struct mutex get_ctrl_mutex;  /* serialize writes from sysfs */

	/* AMO remap sysfs mutex. */
	struct mutex amo_remap_to_pcie_fadd_mutex;

	/* sysfs support */
	struct kobject properties_kobj;
	struct kobject fru_kobj;
	struct kobject link_restarts_kobj;

	/* debugfs support */
	struct attribute_group  port_group;
	struct attribute        **port_attrs;
	struct device_attribute *port_dev_attrs;
	struct device_attribute port_attr;

	/* Current copy of the QSFP eeprom, with its length. If length
	 * is 0, the data is not valid.
	 */
	struct mutex qsfp_eeprom_lock;
	u8 qsfp_eeprom_page0[ETH_MODULE_SFF_8436_LEN];
	u8 qsfp_eeprom_page1[ETH_MODULE_SFF_8436_LEN];
	u8 qsfp_format;
	unsigned int qsfp_eeprom_page_len;

	/*
	 * telemetry for cassini
	 */
	struct {
		struct kobject		kobj_items;
		struct telem_info	*info;
	} telemetry;

	/*
	 * dma controller
	 */
	struct {
		struct mutex			lock;
		struct completion		irupt;
		char				irupt_name[CASS_MAX_IRQ_NAME];
		struct c_pi_dmac_cdesc		*cpl_desc;
		dma_addr_t                      cpl_desc_dma_addr;
		struct dmac_desc_set		*desc_sets;
		unsigned long			*desc_map;
	} dmac;

	/* sl */
	struct cass_sl_dev sl;
};

static inline struct cass_dev *cxi_to_cass_dev(struct cxi_dev *dev)
{
	return container_of(dev, struct cass_dev, cdev);
}

/* Communication profile. */
struct cass_cp {
	struct cass_dev *hw;
	struct cxi_tx_profile *tx_profile;

	/* VNI/PCP and TC identify a unique communication profile. */
	unsigned int vni_pcp;
	enum cxi_traffic_class tc;
	enum cxi_traffic_class_type tc_type;

	/* Communication profile table ID. */
	unsigned int id;

	/* CP list ID. */
	unsigned int list_id;

	/* Number of users of this profile. */
	refcount_t ref;
};

struct pldm_sensor {
	struct cass_dev *hw;
	unsigned int id;
	struct kobject kobj;
	char name[AUX_NAME_MAX];
	struct numeric_sensor_pdr num;

	/* Mapping when inserted into hwmon */
	struct {
		enum hwmon_sensor_types type;
		int channel;
		int multiplier;
	} hwmon;
};

/* From struct getsensorreading_rsp */
struct pldm_sensor_reading {
	u8 operational_state;
	u8 present_state;
	u8 previous_state;
	s64 present_reading;
};

/* Read one of the fields from a PLDM declaration */
#define get_pldm_value(sensor, field) ({				\
	const int s = sensor->num.sensor_data_size;			\
	s64 value;							\
									\
	if (s == PLDM_DATA_SIZE_UINT8 || s == PLDM_DATA_SIZE_SINT8)	\
		value = sensor->num.ssd.ssd8.field;			\
	else if (s == PLDM_DATA_SIZE_UINT16 ||				\
		 s == PLDM_DATA_SIZE_SINT16)				\
		value = sensor->num.ssd.ssd16.field;			\
	else								\
		value = sensor->num.ssd.ssd32.field;			\
									\
	value;								\
})

static inline void __iomem *cass_csr(struct cass_dev *hw, u64 offset)
{
	BUG_ON(offset > C_MEMORG_CSR_SIZE);
	return hw->regs + offset;
}

/**
 * cass_write() - Write to the adapter in 64-bits words
 *
 * Modify a control status register (CSR) on BAR0.
 *
 * @hw: the device
 * @offset: offset of the CSR in the region
 * @data_in: buffer with the data to write
 * @len: length to write, in bytes. Must be a multiple of 8.
 */
static inline void cass_write(struct cass_dev *hw, u64 offset,
			      const void *data_in, size_t len)
{
	void __iomem *base = cass_csr(hw, offset);
	const u64 *data = data_in;
	int i;

	len /= sizeof(u64);

	for (i = 0; i < len; i++, base += sizeof(u64))
		writeq(data[i], base);
}

/**
 * cass_read() - Read from the adapter in 64-bits words
 *
 * Read a CSR on BAR0.
 *
 * @hw: the device
 * @offset: offset of the CSR in the region
 * @data_out: buffer to store the result
 * @len: length to read, in bytes. Must be a multiple of 8.
 */
static inline void cass_read(struct cass_dev *hw, u64 offset, void *data_out, size_t len)
{
	void __iomem *base = cass_csr(hw, offset);
	u64 *data = data_out;
	int i;

	len /= sizeof(u64);

	for (i = 0; i < len; i++, base += sizeof(u64))
		data[i] = readq(base);
}

/**
 * cass_clear() - Writes zeroes to a CSR
 *
 * @hw: the device
 * @offset: offset of the CSR in the region
 * @len: length to clear, in bytes. Must be a multiple of 8.
 */
static inline void cass_clear(struct cass_dev *hw, u64 offset, size_t len)
{
	void __iomem *base = cass_csr(hw, offset);
	int i;

	len /= sizeof(u64);

	for (i = 0; i < len; i++, base += sizeof(u64))
		writeq(0, base);
}

/* must come after cass_read/write/clear definitions. */
#include "libcassini.h"

static inline bool cass_version(const struct cass_dev *hw,
				enum cassini_version version)
{
	return cassini_version(&hw->cdev.prop, version);
}

static inline void cass_cond_lock(struct mutex *lock, bool need_lock)
{
	if (need_lock)
		mutex_lock(lock);
}

static inline void cass_cond_unlock(struct mutex *lock, bool need_lock)
{
	if (need_lock)
		mutex_unlock(lock);
}

static inline void cass_cp_lock(struct cass_dev *hw)
{
	mutex_lock(&hw->cp_lock);
}

static inline void cass_cp_unlock(struct cass_dev *hw)
{
	mutex_unlock(&hw->cp_lock);
}

void cass_device_put_pages(struct cxi_md_priv *md_priv);
int cxi_dmabuf_get_pages(struct ac_map_opts *m_opts);
void cxi_dmabuf_put_pages(struct cxi_md_priv *md_priv);

int cass_ixe_set_amo_remap_to_pcie_fadd(struct cass_dev *hw, int amo_remap);
void cass_pte_init(struct cass_dev *hw);
void cass_pte_set_get_ctrl(struct cass_dev *hw);
int cass_atu_init(struct cass_dev *hw);
void cass_atu_fini(struct cass_dev *hw);
void cass_acs_disable(struct cxi_lni_priv *lni_priv);
void cass_acs_free(struct cxi_lni_priv *lni_priv);
void cass_cq_init(struct cass_dev *hw);
void cass_ee_init(struct cass_dev *hw);
int cass_svc_init(struct cass_dev *hw);
void cass_svc_fini(struct cass_dev *hw);
int cass_tc_find(struct cass_dev *hw, enum cxi_traffic_class tc,
		 enum cxi_traffic_class_type tc_type, u8 ethernet_pcp,
		 struct cass_tc_cfg *tc_cfg);
int cass_tc_init(struct cass_dev *hw);
void cass_tc_fini(struct cass_dev *hw);
void traffic_shaping_cfg(struct cass_dev *hw);
void cass_tc_set_tx_pause_all(struct cass_dev *hw, bool enable);
void cass_tc_set_rx_pause_all(struct cass_dev *hw, bool enable);
void cass_clear_cps(struct cxi_tx_profile *tx_profile);

static inline bool is_vni_valid(unsigned int vni)
{
	/* A VNI is 16 bits and VNI 0 is invalid */
	return vni && !(vni & ~0xffff);
}

int register_error_handlers(struct cass_dev *hw);
void deregister_error_handlers(struct cass_dev *hw);
int cass_sriov_configure(struct pci_dev *pdev, int num_vfs);
int cass_vf_init(struct cass_dev *hw);
void cass_vf_fini(struct cass_dev *hw);

#define ATU_CFG_AC_TABLE_MB_SHIFT 15
/* Based on note in MEM_SIZE description - MEM_BASE must be aligned to
 * 8x the largest configured page size.
 * Derivative 1 Page Size as described in Table 364 - Page Table Sizes
 */
#define ATU_IOVA_BASE_SHIFT(ac) (12 + ac.base_pg_size + ac.pg_table_size + 3)
#define ATU_IOVA_MASK(ac) (~MASK(ATU_IOVA_BASE_SHIFT(ac)))
/* Maximum addressable memory range (56 bits) of Cassini. */
#define ATU_MAX_ADDR_MEM_BITS 56
#define ATU_LAC_SHIFT (ATU_MAX_ADDR_MEM_BITS - 3)
#define ATU_VA_MASK(va) (va & MASK(ATU_MAX_ADDR_MEM_BITS))
#define IS_DEVICE_MEM(md_priv) (md_priv->flags & CXI_MAP_DEVICE)

extern struct mutex dev_list_mutex;
extern int more_debug;
extern struct genl_family cxierr_genl_family;
extern struct cass_p2p_ops p2p_ops;
extern bool odp_sw_decouple;
extern bool ats_c1_override;

/* Add the LAC to the top 3 bits of the IOVA so they are unique */
static inline u64 cass_iova_base(struct cass_ac *cac)
{
	u64 iova_base;

	iova_base = (u64)get_random_u32() << ATU_IOVA_BASE_SHIFT(cac->cfg_ac);
	iova_base &= ~(0x7UL << ATU_LAC_SHIFT);
	iova_base |= (u64)cac->ac.lac << ATU_LAC_SHIFT;

	return iova_base;
}

void __iomem *cass_csr(struct cass_dev *hw, u64 offset);
int cass_nta_cq_init(struct cass_dev *hw);
void cass_nta_cq_fini(struct cass_dev *hw);
int cass_nta_pri_init(struct cass_dev *hw);
int cass_sts_idle(struct sts_idle *sidle);
void cass_nta_pri_fini(struct cass_dev *hw);
int cass_inbound_wait(struct cass_dev *hw, bool wait_for_response);
u64 cass_nta_iova_alloc(struct cass_ac *cac, const struct ac_map_opts *m_opts);
void cass_nta_iova_fini(struct cass_ac *cac);
int cass_nta_iova_init(struct cass_ac *cac, size_t va_len);
void cass_nta_free(struct cass_dev *hw, struct cass_nta *nta);
void cass_l1_tables_free(struct cass_dev *hw, struct cass_nta *nta);
void cass_clear_range(const struct cxi_md_priv *md_priv, u64 iova, u64 len);
void cass_md_clear(struct cxi_md_priv *md_prv, bool inval, bool need_lock);
void cass_dma_unmap_pages(struct cxi_md_priv *md_priv);
int cass_dma_addr_mirror(dma_addr_t dma_addr, u64 iova, struct cass_ac *cac,
			 u32 flags, bool is_huge_page, bool *invalidate);
int cass_mirror_fault(const struct ac_map_opts *m_opts, u64 *pfns, int count,
		      uintptr_t addr, size_t len);
int cass_mirror_odp(const struct ac_map_opts *m_opts, struct cass_ac *cac,
		    int npfns, u64 addr);
int cass_mirror_device(struct cxi_md_priv *md_priv,
		       const struct ac_map_opts *m_opts);
int cass_mirror_range(struct cxi_md_priv *md_priv, struct ac_map_opts *m_opts);
void cass_invalidate_range(const struct cass_ac *cac, u64 iova,
			   size_t length);
int cass_nta_init(struct cxi_lni_priv *lni_priv, struct ac_map_opts *m_opts,
		  struct cass_ac *cac);
int cass_nta_mirror_sgt(struct cxi_md_priv *md_priv, bool need_lock);
int cass_nta_mirror_kern(struct cxi_md_priv *md_priv,
			 const struct iov_iter *iter, bool need_lock);
int cass_pin(const struct cass_ac *cac, struct page **pages, int npages,
	     u64 addr, bool write);
int cass_mirror_hp(const struct ac_map_opts *m_opts, struct cxi_md_priv *md_priv);
int cass_pfns_mirror(struct cxi_md_priv *md_priv, const struct ac_map_opts *m_opts,
		     u64 *pfns, int npfns, bool is_huge_page);
int cass_vma_write_flag(struct mm_struct *mm, ulong start, ulong end,
			u32 flags);
int cass_pin_mirror(struct cxi_md_priv *md_priv, struct ac_map_opts *m_opts);
void cass_notifier_cleanup(struct cxi_md_priv *md_priv);
int cass_odp_decouple(struct cass_dev *hw);
int cass_odp_supported(struct cass_dev *hw, u32 flags);
int cass_mmu_notifier_insert(struct cxi_md_priv *md_priv,
			     const struct ac_map_opts *m_opts);
int cass_is_device_memory(struct cass_dev *hw, struct ac_map_opts *m_opts,
			  uintptr_t va, size_t len);
int cass_device_get_pages(struct ac_map_opts *m_opts);
void cass_device_hugepage_size(int contig_cnt, struct ac_map_opts *m_opts);
void cass_align_start_len(struct ac_map_opts *m_opts, uintptr_t va, size_t len,
			  int align_shift);
int cass_cpu_page_size(struct cass_dev *hw, struct ac_map_opts *m_opts,
		       struct mm_struct *mm, uintptr_t va,
		       const struct cxi_md_hints *hints, int *align_shift);
int cass_ptp_init(struct cass_dev *hw);
void cass_ptp_fini(struct cass_dev *hw);
struct cxi_md_priv *cass_device_md(struct cxi_lni_priv *lni_priv, u64 va,
				   size_t len, dma_addr_t *dma_addr);
int cass_ats_init(struct cxi_lni_priv *lni_priv, struct ac_map_opts *m_opts,
		  struct cass_ac *cac);
int cass_ats_md_init(struct cxi_md_priv *md_priv, const struct ac_map_opts *m_opts);
void cass_amd_iommu_inval_cb_init(struct pci_dev *pdev);
void cass_unbind_ac(struct pci_dev *pdev, int pasid);
void cass_iommu_fini(struct pci_dev *pdev);
void cass_iommu_init(struct cass_dev *hw);

void cass_hni_init(struct cass_dev *hw);
void cass_hni_fini(struct cass_dev *hw);
int cass_cable_scan(struct cass_dev *hw);
void update_oxe_link_up(struct cass_dev *hw);
void update_hni_link_up(struct cass_dev *hw);
int cass_create_tc_sysfs(struct cass_dev *hw);
void cass_destroy_tc_sysfs(struct cass_dev *hw);

int cass_ixe_init(struct cass_dev *hw);

int cass_register_uc(struct cass_dev *hw);
void cass_unregister_uc(struct cass_dev *hw);
int cass_register_port(struct cass_dev *hw);
void cass_unregister_port(struct cass_dev *hw);
int uc_cmd_get_nic_id(struct cass_dev *hw);
int uc_cmd_get_mac(struct cass_dev *hw);
int uc_cmd_get_fru(struct cass_dev *hw);
void uc_cmd_update_ier(struct cass_dev *hw, u32 set_bits, u32 clear_bits);
int uc_cmd_qsfp_write(struct cass_dev *hw, u8 page, u8 addr,
		      const u8 *data, size_t data_len);
void uc_prepare_comm(struct cass_dev *hw);
int uc_wait_for_response(struct cass_dev *hw);
int cxi_get_qsfp_data(struct cxi_dev *cdev, u32 offset,
		      u32 len, u32 page, u8 *data);
void uc_cmd_set_link_leds(struct cass_dev *hw, enum casuc_led_states led_state);
int update_sensor(struct pldm_sensor *sensor,
		  struct pldm_sensor_reading *result);

void finalize_pt_cleanups(struct cxi_lni_priv *lni, bool force);
void finalize_cq_cleanups(struct cxi_lni_priv *lni, bool force);
void finalize_eq_cleanups(struct cxi_lni_priv *lni);
void finalize_ct_cleanups(struct cxi_lni_priv *lni);
bool lni_cleanups(struct cass_dev *hw, bool force);
void lni_cleanups_work(struct work_struct *work);
void cass_rgid_init(struct cass_dev *hw);
void cass_rgid_fini(struct cass_dev *hw);
int cass_lac_get(struct cass_dev *hw, int id);
void cass_lac_put(struct cass_dev *hw, int id, int lac);
int cass_rgid_get(struct cass_dev *hw, struct cxi_rgroup *rgroup);
void cass_rgid_put(struct cass_dev *hw, int id);
int cass_lcid_get(struct cass_dev *hw, struct cxi_cp_priv *cp_priv, int rgid);
void cass_lcid_put(struct cass_dev *hw, int rgid, int lcid);
struct cxi_cp_priv *cass_cp_find(struct cass_dev *hw, int rgid, int lcid);
struct cxi_cp_priv *cass_cp_rgid_find(struct cass_dev *hw, int rgid,
				      unsigned int vni_pcp, unsigned int tc,
				      enum cxi_traffic_class_type tc_type);

int cass_telem_init(struct cass_dev *hw);
void cass_telem_fini(struct cass_dev *hw);

int create_sysfs_properties(struct cass_dev *hw);
void destroy_sysfs_properties(struct cass_dev *hw);

void cass_phy_start(struct cass_dev *hw, bool force_reconfig);
void cass_phy_stop(struct cass_dev *hw, bool block);
void cass_phy_bounce(struct cass_dev *hw);
void cass_phy_trigger_machine(struct cass_dev *hw);
void cass_phy_link_up(struct cass_dev *hw);
void cass_phy_link_down(struct cass_dev *hw);

extern unsigned int pe_total_les;
void cass_cfg_le_pools(struct cass_dev *hw, int pool_id, int pe,
		       const struct cxi_limits *les, bool release);
void cass_cfg_tle_pool(struct cass_dev *hw, int pool_id,
		       const struct cxi_limits *tles, bool release);
void cass_tle_init(struct cass_dev *hw);

int cass_dmac_init(struct cass_dev *hw);
void cass_dmac_fini(struct cass_dev *hw);

int cass_irq_init(struct cass_dev *hw);
void cass_irq_fini(struct cass_dev *hw);
struct cass_irq *cass_comp_irq_attach(struct cass_dev *hw,
				      const struct cpumask *mask,
				      struct notifier_block *nb);
void cass_comp_irq_detach(struct cass_dev *hw, struct cass_irq *irq,
			  struct notifier_block *nb);

void cxi_register_hw_errors(struct cass_dev *hw,
			    struct cxi_reg_err_flg *reg_err_flg);
void cxi_unregister_hw_errors(struct cass_dev *hw,
			      struct cxi_reg_err_flg *reg_err_flg);
void cxi_enable_hw_errors(struct cass_dev *hw, unsigned int irq, bool is_ext,
			  const unsigned long *bitmask);
void cxi_disable_hw_errors(struct cass_dev *hw, unsigned int irq, bool is_ext,
			   const unsigned long *bitmask);

bool cass_phy_is_headshell_removed(struct cass_dev *hw);
void cass_phy_set_state(enum cass_phy_state state, struct cass_dev *hw);

void start_pcie_monitoring(struct cass_dev *hw);
void stop_pcie_monitoring(struct cass_dev *hw);
void cass_set_outstanding_limit(struct cass_dev *hw);

#endif	/* _CASS_CORE_H */
