/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2022-2024 Hewlett Packard Enterprise Development LP */

#ifndef _CXI_CORE_H
#define _CXI_CORE_H

#include <linux/device.h>
#include <linux/types.h>
#include <linux/hmm.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/refcount.h>
#include <linux/version.h>
#include <linux/mmu_notifier.h>

#define DFA_EP_BITS 17
#define MAX_PID_BITS 9
#define MIN_PID_BITS (MAX_PID_BITS - 3)
#define XSTR(x) STR(x)
#define STR(x) #x

/* Some kernel specific defines */
#if (defined(RHEL_MAJOR) && RHEL_MAJOR == 8 && RHEL_MINOR >= 9)
#define RHEL8_9_PLUS
#define HAVE_GET_SINGLETON
#define HAVE_IOVA_INIT_RCACHES
#elif (defined(RHEL_MAJOR) && RHEL_MAJOR == 9 && RHEL_MINOR >= 3)
#define RHEL9_3_PLUS
#define HAVE_GET_SINGLETON
#define HAVE_IOVA_INIT_RCACHES
#elif defined(CONFIG_SUSE_PRODUCT_SLE) && CONFIG_SUSE_VERSION == 15 && \
	CONFIG_SUSE_PATCHLEVEL == 4
#elif defined(CONFIG_SUSE_PRODUCT_SLE) && CONFIG_SUSE_VERSION == 15 && \
	CONFIG_SUSE_PATCHLEVEL >= 5
#define HAVE_GET_SINGLETON
#define HAVE_IOVA_INIT_RCACHES
#include <linux/mmu_notifier.h>
#endif

#if (defined(RHEL_MAJOR) && RHEL_MAJOR == 9 && RHEL_MINOR >= 6)
#define HAVE_KERNEL_ETHTOOL_TS_INFO
#endif

#if defined(CONFIG_SUSE_PRODUCT_SLE) && CONFIG_SUSE_VERSION == 15 && \
	CONFIG_SUSE_PATCHLEVEL >= 7
#define HAVE_KERNEL_ETHTOOL_TS_INFO
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 17, 0)
#define HAVE_GET_SINGLETON
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 18, 0)
#define HAVE_IOVA_INIT_RCACHES
#endif

#define dev_emerg_always(dev, fmt, ...)	  dev_emerg(dev, fmt, ##__VA_ARGS__)
#define dev_alert_always(dev, fmt, ...)	  dev_alert(dev, fmt, ##__VA_ARGS__)
#define dev_crit_always(dev, fmt, ...)	  dev_crit(dev, fmt, ##__VA_ARGS__)
#define dev_err_always(dev, fmt, ...)	  dev_err(dev, fmt, ##__VA_ARGS__)
#define dev_warn_always(dev, fmt, ...)	  dev_warn(dev, fmt, ##__VA_ARGS__)
#define dev_notice_always(dev, fmt, ...)  dev_notice(dev, fmt, ##__VA_ARGS__)
#define dev_info_always(dev, fmt, ...)	  dev_info(dev, fmt, ##__VA_ARGS__)
#define dev_dbg_always(dev, fmt, ...)	  dev_dbg(dev, fmt, ##__VA_ARGS__)

/*
 * This is the main or core cassini cxi logging macro.	It is not intended to
 * be used directly anywhere except the cxidev_*() macros.  The whole purpose
 * behind this logging is to use '"%s: " fmt' and '"%s/%s: " fmt' to take
 * advantage of C string constant concatenation to add common information to
 * the logged cassini cxi messages.
 */
#define _cxi_log(freq, severity, cdev, fmt, ...)			\
	dev_##severity##_##freq(&(cdev)->pdev->dev,		\
				"%s[%s] " fmt,				\
				(cdev)->name, (cdev)->eth_name,		\
				##__VA_ARGS__)

/*
 * the cassini log equivalent of dev_*() log
 */
#define cxidev_emerg(cdev, fmt, ...)\
		_cxi_log(always, emerg, cdev, fmt, ##__VA_ARGS__)
#define cxidev_alert(cdev, fmt, ...)\
		_cxi_log(always, alert, cdev, fmt, ##__VA_ARGS__)
#define cxidev_crit(cdev, fmt, ...)\
		_cxi_log(always, crit, cdev, fmt, ##__VA_ARGS__)
#define cxidev_err(cdev, fmt, ...)\
		_cxi_log(always, err, cdev, fmt, ##__VA_ARGS__)
#define cxidev_warn(cdev, fmt, ...)\
		_cxi_log(always, warn, cdev, fmt, ##__VA_ARGS__)
#define cxidev_notice(cdev, fmt, ...)\
		_cxi_log(always, notice, cdev, fmt, ##__VA_ARGS__)
#define cxidev_info(cdev, fmt, ...)\
		_cxi_log(always, info, cdev, fmt, ##__VA_ARGS__)
#define cxidev_dbg(cdev, fmt, ...)\
		_cxi_log(always, dbg, cdev, fmt, ##__VA_ARGS__)

#define cxidev_WARN(cdev, fmt, ...)\
		dev_WARN(&(cdev)->cdev.pdev->dev,			\
			 "%s[%s]: " fmt, (cdev)->cdev.name,	\
			 (cdev)->cdev.eth_name,			\
			 ##__VA_ARGS__)

/*
 * the cassini log equivalent of dev_*_once()
 */
#define cxidev_emerg_once(cdev, fmt, ...)\
		_cxi_log(once, emerg, cdev, fmt, ##__VA_ARGS__)
#define cxidev_alert_once(cdev, fmt, ...)\
		_cxi_log(once, alert, cdev, fmt, ##__VA_ARGS__)
#define cxidev_crit_once(cdev, fmt, ...)\
		_cxi_log(once, crit, cdev, fmt, ##__VA_ARGS__)
#define cxidev_err_once(cdev, fmt, ...)\
		_cxi_log(once, err, cdev, fmt, ##__VA_ARGS__)
#define cxidev_warn_once(cdev, fmt, ...)\
		_cxi_log(once, warn, cdev, fmt, ##__VA_ARGS__)
#define cxidev_notice_once(cdev, fmt, ...)\
		_cxi_log(once, notice, cdev, fmt, ##__VA_ARGS__)
#define cxidev_info_once(cdev, fmt, ...)\
		_cxi_log(once, info, cdev, fmt, ##__VA_ARGS__)
#define cxidev_dbg_once(cdev, fmt, ...)\
		_cxi_log(once, dbg, cdev, fmt, ##__VA_ARGS__)

#define cxidev_WARN_ONCE(cdev, condition, fmt, ...)\
		dev_WARN_ONCE(&(cdev)->pdev->dev, (condition),	\
			      "%s[%s]: " fmt, (cdev)->name,		\
			      (cdev)->eth_name,			\
			      ##__VA_ARGS__)

/*
 * the cassini log equivalent of dev_*_ratelimited()
 */
#define cxidev_emerg_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, emerg, cdev, fmt, ##__VA_ARGS__)
#define cxidev_alert_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, alert, cdev, fmt, ##__VA_ARGS__)
#define cxidev_crit_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, crit, cdev, fmt, ##__VA_ARGS__)
#define cxidev_err_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, err, cdev, fmt, ##__VA_ARGS__)
#define cxidev_warn_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, warn, cdev, fmt, ##__VA_ARGS__)
#define cxidev_notice_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, notice, cdev, fmt, ##__VA_ARGS__)
#define cxidev_info_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, info, cdev, fmt, ##__VA_ARGS__)
#define cxidev_dbg_ratelimited(cdev, fmt, ...)\
		_cxi_log(ratelimited, dbg, cdev, fmt, ##__VA_ARGS__)

extern unsigned int pid_bits;
extern unsigned int system_type_identifier;
extern unsigned int min_free_shift;
extern unsigned int vni_matching;
extern unsigned int tg_threshold[4];
extern int untagged_eth_pcp;
extern bool switch_connected;

extern atomic_t cxi_num;
extern struct class cxi_class;
extern struct dentry *cxi_debug_dir;

extern void cass_disable_device(struct pci_dev *pdev);
extern int hw_register(void);
extern void hw_unregister(void);
extern int cxi_configfs_subsys_init(void);
extern void cxi_configfs_exit(void);

void cxi_add_device(struct cxi_dev *cdev);
void cxi_remove_device(struct cxi_dev *cdev);
void cxi_send_async_event(struct cxi_dev *cdev, enum cxi_async_event event);
void cxi_apply_for_all(void (*callback)(struct cxi_dev *dev, void *p),
		       void *p);

void cxi_p2p_fini(void);

#define CXI_DEFAULT_LNIS_PER_RGID 1

/* Service */
struct cxi_svc_priv {
	struct list_head list;
	struct cxi_svc_desc svc_desc;
	struct cxi_rgroup *rgroup;
	struct cxi_rx_profile *rx_profile[CXI_SVC_MAX_VNIS];
	struct cxi_tx_profile *tx_profile[CXI_SVC_MAX_VNIS];
};

/* Logical Network Interface */
struct cxi_lni_priv {
	struct cxi_dev *dev;
	struct list_head list;

	/* User LNI object. */
	struct cxi_lni lni;

	/* Users of this LNI. */
	refcount_t refcount;

	/* Service associated with this LNI. */
	struct cxi_rgroup *rgroup;

	/* Keep track of associated resources. TODO: keep as that may
	 * help debugging, but should be removed eventually.
	 */
	struct list_head domain_list;
	struct list_head eq_list;
	struct list_head cq_list;
	struct list_head pt_list;
	struct list_head ac_list;
	struct list_head ct_list;
	struct list_head reserved_pids;
	spinlock_t res_lock;
	struct mutex ac_list_mutex;

	struct ida lac_table;
	atomic_t lpe_pe_num;

	struct dentry *debug_dir;
	struct dentry *cq_dir;
	struct dentry *pt_dir;
	struct dentry *eq_dir;
	struct dentry *ct_dir;
	struct dentry *ac_dir;

	/* Pending resources to cleanup. Use res_lock to serialize. */
	struct list_head pt_cleanups_list;
	struct list_head cq_cleanups_list;
	struct list_head eq_cleanups_list;
	struct list_head ct_cleanups_list;

	u32 pid;
};

struct cass_vni;

struct cxi_reserved_pids {
	struct list_head entry;
	struct cxi_rx_profile *rx_profile;
	DECLARE_BITMAP(table, 1 << MAX_PID_BITS);
};

struct ct_wb_desc {
	struct page *page;
	dma_addr_t wb_dma_addr;
	struct cxi_md_priv *md_priv;
	bool is_map_page;
};

struct cxi_ct_priv {
	struct list_head entry;
	struct cxi_lni_priv *lni_priv;
	struct dentry *debug_dir;
	struct dentry *lni_dir;
	struct ct_wb_desc wb_desc;
	struct cxi_ct ct;
	bool is_user;

	/* Doorbell MMIO address */
	void __iomem *ct_mmio;
};

struct cxi_domain_priv {
	struct cxi_domain domain;
	struct cxi_lni_priv *lni_priv;
	struct list_head list;
	struct dentry *debug_dir;
	struct rb_node node;	/* attach to domain_tree */
	struct cxi_rx_profile *rx_profile;

	/* Users of this domain */
	refcount_t refcount;
};

struct cxi_md_priv {
	struct list_head md_entry; /* entry in a CAC md_list */
	struct cxi_lni_priv *lni_priv;
	struct cass_ac *cac;
	struct page **pages;
	struct sg_table *sgt;
	struct sg_table *dmabuf_sgt;
	struct device *device;

	/* DMA buffer file descriptor. This is passed in by user-space. */
	int dmabuf_fd;

	/* Offset into the DMA buffer where PFN mapping should begin. */
	unsigned long dmabuf_offset;

	/* User request DMA buffer len. This should not be aligned. */
	unsigned long dmabuf_length;

	struct dma_buf *dmabuf;
	struct dma_buf_attachment *dmabuf_attach;

	/* Users of this MD */
	refcount_t refcount;

	struct cxi_md md;
	/* Original length of the MD */
	size_t olen;
	u32 flags;
	/* Need to lock when initially mirroring page tables */
	bool need_lock;
	/* SG table is owned by an external user (e.g. kfabric) */
	bool external_sgt_owner;
	struct mmu_interval_notifier mn_sub;
	/* GPU direct info */
	void *p2p_info;
	bool cleanup_done;
};

struct cxi_cp_priv {
	struct cass_cp *cass_cp;
	struct cxi_cp cp;
	refcount_t refcount;
	unsigned int rgid;
};

/* EQ buffer description */
struct eq_buf_desc {
	const struct cxi_md *md; /* Virtual buffer MD. */
	void *events;
	size_t events_len;

	struct page **pages;

	dma_addr_t dma_addr; /* DMA buffer address. */
};

struct cxi_eq_priv {
	struct cxi_lni_priv *lni_priv;
	struct list_head list;
	struct dentry *debug_file;
	struct dentry *lni_dir;

	bool reused;

	/* Users of this EQ */
	refcount_t refcount;

	/* EQ buffer description */
	struct eq_buf_desc active;

	/* Doorbell MMIO address */
	void __iomem *eq_mmio;

	/* Current EQ descriptor state */
	union c_ee_cfg_eq_descriptor cfg;

	/* Copy of the creation attributes */
	struct cxi_eq_attr attr;

	/* Public event queue */
	struct cxi_eq eq;

	/* Interrupt vectors, and callbacks */
	struct cass_irq *event_msi_irq;
	void (*event_cb)(void *context);
	void *event_cb_data;
	struct notifier_block event_nb;

	struct cass_irq *status_msi_irq;
	void (*status_cb)(void *context);
	void *status_cb_data;
	struct notifier_block status_nb;

	/* Number of slots in queue. */
	size_t queue_size;

	/* Resize buffer attributes */
	struct mutex resize_mutex;
	bool resized;
	struct eq_buf_desc resize;
};

struct cxi_cq_priv {
	struct cxi_lni_priv *lni_priv;
	struct list_head list;
	struct dentry *debug_dir;
	struct dentry *lni_dir;

	u32 flags;

	/* EQ for error reporting. May be NULL. */
	struct cxi_eq_priv *eq;

	/* DMA mapped CQ */
	size_t cmds_len;
	size_t cmds_order;
	struct page *cmds_pages;
	void *cmds;
	dma_addr_t cmds_dma_addr;

	/* Doorbell MMIO address */
	void __iomem *cq_mmio;

	/* HW CQ structure */
	struct cxi_cq cass_cq;
};

struct cxi_pte_priv {
	struct cxi_lni_priv *lni_priv;
	struct list_head list;
	struct dentry *debug_dir;
	struct dentry *lni_dir;

	struct cxi_pte pte;

	struct cxi_eq_priv *eq;

	/* Users of this portal table entries. */
	refcount_t refcount;

	/* Pte is flow control enabled. */
	bool fc_enabled;
	bool plec_enabled;

	/* pt_n -> mcast_id mapping */
	unsigned int mcast_n;

	/* Keep track of used MST entries when releasing the PTe. The
	 * PTe can only be released when all MSTs attached to it have
	 * been recycled.
	 */
	DECLARE_BITMAP(mst_rc_update, C_MST_DBG_MST_TABLE_ENTRIES);

	int pe_num;
	int le_pool;
};

void cxi_free_resource(struct cxi_dev *dev, struct cxi_svc_priv *svc_priv,
		       enum cxi_rsrc_type type);
int cxi_alloc_resource(struct cxi_dev *dev, struct cxi_svc_priv *svc_priv,
		       enum cxi_rsrc_type type);
unsigned int cxi_lni_get_pe_num(struct cxi_lni_priv *lni);

struct pci_dev *cxi_get_vf0_dev(void);

void cxi_domain_lni_cleanup(struct cxi_lni_priv *lni_priv);

int nvidia_p2p_init(void);
void nvidia_p2p_fini(void);

int amd_p2p_init(void);
void amd_p2p_fini(void);

int cxi_dma_buf_init(void);
void cxi_dma_buf_fini(void);

#include "cxi_configuration.h"

#endif	/* _CXI_CORE_H */
