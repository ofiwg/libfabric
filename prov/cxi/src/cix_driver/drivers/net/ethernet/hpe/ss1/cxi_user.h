/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2021,2024 Hewlett Packard Enterprise Development LP */

#ifndef __CXI_USER_H
#define __CXI_USER_H

#include <linux/device.h>
#include <linux/cdev.h>

#include <uapi/ethernet/cxi-abi.h>

/* Size of the biggest command */
#define MAX_REQ_SIZE 112

#ifdef CONFIG_ARM64
extern struct static_key_false avoid_writecombine;
#endif

/* Keep track of known devices. Protected by dev_list_mutex. There is
 * one instance per Cassini device.
 */
struct ucxi {
	/* List of registered devices */
	struct list_head dev_list;

	/* The CXI device. As it can go away at anytime, we protect
	 * its access with RCU. When the device goes away (such as
	 * hotplug or unbind), open/write/close will return -EIO.
	 */
	struct cxi_dev __rcu *dev;
	struct srcu_struct srcu;

	/* Reference on this object. Prevents freeing it too early. */
	struct kobject kobj;

	/* Device entry in /dev */
	struct cdev cdev;
	struct device *udev;
	unsigned int minor;

	/* Client IDs */
	spinlock_t lock;
	struct idr client_idr;
	struct kobject *clients_kobj;

	/* sysfs directory for wait object */
	struct kobject *wait_kobjs;

	/* For a PF device, list of the VF clients */
	struct user_client *vf_clients[C_NUM_VFS];
};

/* Wait object */
struct ucxi_wait {
	/* Entry for that wait object in sysfs. */
	struct kobject kobj;
	struct kernfs_node *dirent;
	bool going_away;
};

/* Keep track of user clients. One instance is created for every
 * opened file.
 */
struct user_client {
	/* What device was opened. */
	struct ucxi *ucxi;
	struct file *fileptr;

	unsigned int id;

	/* Whether this client represents a VF */
	bool is_vf;
	u8 vf_num;

	/* sysfs objects */
	struct kobject *kobj;
	struct kobject *wait_objs_kobj;

	spinlock_t mmap_offset_lock;
	off_t mmap_offset;

	spinlock_t pending_lock; /* guard pending_mmaps */
	struct list_head pending_mmaps;
	struct cxi_mmap_info *csrs_mminfo;

	struct mutex eq_resize_mutex;

	/* Resources allocated by that client. The IDR is usually returned to
	 * userspace for identification purposes.
	 */
	rwlock_t res_lock;
	struct idr lni_idr;
	struct idr cp_idr;
	struct idr domain_idr;
	struct idr cq_idr;
	struct idr md_idr;
	struct idr eq_idr;
	struct idr pte_idr;
	struct idr pte_map_idr;
	struct idr wait_idr;
	struct idr ct_idr;

	/*
	 * for now, max of one counter pool id per client.
	 */
	unsigned int cntr_pool_id;
};

struct cxi_mmap_info;

struct ucxi_obj {
	/* How many object depend on this object */
	atomic_t refs;
	atomic_t mappings;

	struct cxi_mmap_info *mminfo;

	/* Pointer to the core object */
	union {
		struct cxi_cp *cp;
		struct cxi_lni *lni;
		struct cxi_domain *domain;
		struct cxi_cq *cq;
		struct cxi_md *md;
		struct cxi_eq *eq;
		struct cxi_pte *pte;
		struct ucxi_wait *wait;
		struct cxi_ct *ct;
		unsigned int pte_index;
	};

	/* Dependencies. e.g. a CQ depends on several CPs and an LNI */
	unsigned int num_deps;
	struct ucxi_obj *deps[];
};

enum cxiu_mmap_type {
	MMAP_PHYSICAL,
	MMAP_LOGICAL,
	MMAP_VIRTUAL,
	MMAP_LOGICAL_RO,
};

/* Each time the kernel allocates a structure that needs to be shared
 * with userspace, it creates this cxi_mmap_info, and passes the
 * cxi_mminfo to userspace. The application then calls mmap with that
 * information, and the kernel can retrieve the right pages to map.
 */
struct cxi_mmap_info {
	struct list_head pending_mmaps;

	struct cxi_mminfo mminfo;

	/* Object this mapping is attached to. The mmap code will get
	 * a reference on it when the object is mmap'ed, to prevent
	 * the object from being freed when there is still a userspace
	 * mapping on it.
	 */
	struct ucxi_obj *obj;

	/* Whether the object is a physical address (eg. something in
	 * the PCI BAR), a kernel logical address (eg. a CQ), or a
	 * kernel virtual address (eg EQ mapped to the user)
	 */
	enum cxiu_mmap_type mmap_type;
	union {
		/* Pages array corresponding to a logical address */
		struct page *pages;

		/* Physical address (in a PCI BAR) */
		phys_addr_t phys;

		/* Memory corresponding to a virtual user address */
		void *vma_addr;
	};

	bool wc;

	/*
	 * The start and end vm addresses from struct vm_area_struct to
	 * ucxi_mmap() routine.  Zero indicates the instance is not currently
	 * memory mapped.  If pending_mmaps is empty then the instances was
	 * unmapped else the instance is still waiting to be mapped.
	 */
	unsigned long vm_start;
	unsigned long vm_end;
};

void fill_mmap_info(struct user_client *client, struct cxi_mmap_info *mminfo,
		    unsigned long addr, size_t size, enum cxiu_mmap_type type);
int ucxi_mmap(struct file *filp, struct vm_area_struct *vma);

struct cmd_info {
	unsigned int req_size;
	const char *name;
	int (*handler)(struct user_client *client, const void *cmd_in,
		       void *resp, size_t *resp_len);
	bool admin_only;
};

#endif	/* __CXI_USER_H */
