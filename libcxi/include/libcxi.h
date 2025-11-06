/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Copyright 2020,2024 Hewlett Packard Enterprise Development LP */

#ifndef __LIBCXI_H__
#define __LIBCXI_H__

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CXIL_API __attribute__((visibility("default")))

#ifndef __user
#define __user
#endif

#include "cxi_prov_hw.h"
#include "uapi/misc/cxi.h"
#include "cassini_cntr_defs.h"

#define CXIL_DEVNAME_MAX 13	/* "cxi" + up to 10 digits */
#define CXIL_DRVNAME_MAX 8
#define CXIL_FRUDESC_MAX 16

struct cxil_devinfo {
	unsigned int dev_id;
	union {
		unsigned int nic_addr; /* obsolete */
		unsigned int nid;
	};
	unsigned int pid_bits;
	unsigned int pid_count;
	unsigned int pid_granule;
	unsigned int min_free_shift;
	unsigned int rdzv_get_idx;

	char device_name[CXIL_DEVNAME_MAX+1];
	char driver_name[CXIL_DRVNAME_MAX+1];
	unsigned int vendor_id;
	unsigned int device_id;
	unsigned int device_rev;
	unsigned int device_proto;
	unsigned int device_platform;

	uint16_t num_ptes;
	uint16_t num_txqs;
	uint16_t num_tgqs;
	uint16_t num_eqs;
	uint16_t num_cts;
	uint16_t num_acs;
	uint16_t num_tles;
	uint16_t num_les;

	uint16_t pci_domain;
	uint8_t pci_bus;
	uint8_t pci_device;
	uint8_t pci_function;

	size_t link_mtu;
	size_t link_speed;
	uint8_t link_state;

	int uc_nic;

	unsigned int pct_eq;

	/* Cassini version (CASSINI_1_0, ...) */
	enum cassini_version cassini_version;

	/* type of board: "Brazos", ... */
	char fru_description[CXIL_FRUDESC_MAX];

	bool is_vf;		/* PCIe PF or VF */

	/* System info (mix or homogeneous) */
	enum system_type_identifier system_type_identifier;
};

struct cxil_pte {
	unsigned int ptn;
};

struct cxil_domain {
	unsigned int vni;
	unsigned int pid;
};

struct cxil_dev {
	struct cxil_devinfo info;
};

struct cxil_lni {
	unsigned int id;
};

struct cxil_pte;
struct cxil_pte_map;
struct cxil_wait_obj;

struct cxil_device_list {
	unsigned int count;
	struct cxil_devinfo info[];
};

struct cxil_svc_list {
	unsigned int count;
	struct cxi_svc_desc descs[];
};

struct cxil_svc_rsrc_list {
	unsigned int count;
	struct cxi_rsrc_use rsrcs[];
};

/**
 * @brief Tests if the CXI retry handler is running for a device.
 *
 * @param devinfo Device info for the device to test
 *
 * @return True is returned if a retry handler is running for the device.
 */
static inline bool cxil_rh_running(struct cxil_devinfo *devinfo)
{
	return devinfo->pct_eq != C_EQ_NONE;
}

/**
 * @brief Retrieves the list of available CXI devices
 *
 * @param dev_list A pointer to a device_list pointer
 *
 * @return On success, zero is returned and dev_list points to a newly
 *         allocated structure. Otherwise, a negative errno value is
 *         returned indicating the error.
 *
 * The returned structure should be freed by calling cxil_free_device_list().
 */
CXIL_API int cxil_get_device_list(struct cxil_device_list **dev_list);

/**
 * @brief Frees the device structure allocated by cxil_get_device_list()
 *
 * @param dev_list A pointer to the device_list
 */
CXIL_API void cxil_free_device_list(struct cxil_device_list *dev_list);

/**
 * @brief Opens a CXI Device object.  A Device represents a single physical
 *        network device.
 *
 * @param dev_id The ID of the network device to open
 * @param dev The new CXI Device object
 *
 * @return On success, zero is returned and the new Device is pointed to by the
 *         dev parameter.  Otherwise, a negative errno value is returned
 *         indicating the error.
 */
CXIL_API int cxil_open_device(uint32_t dev_id, struct cxil_dev **dev);

/**
 * @brief Destroys a CXI Device object.
 *
 * @param dev The CXI Device object to destroy
 */
CXIL_API void cxil_close_device(struct cxil_dev *dev);

/**
 * @brief Gets a svc_descriptor from its ID.
 *
 * @param dev The CXI Device
 * @param svc_id The ID returned from cxil_svc_alloc
 * @param svc_desc Destination pointer for the svc_desc
 *
 * @return On success, zero is returned and svc_desc will contain
 *         info regarding the kernel's view of this service.
 *         Otherwise, a negative errno value is
 *         returned indicating the error.
 *
 */
CXIL_API int cxil_get_svc(struct cxil_dev *dev, unsigned int svc_id,
			  struct cxi_svc_desc *svc_desc);

/**
 * @brief Retrieves the list of active services
 *
 * @param dev The CXI Device
 * @param svc_list Destination pointer for the service list
 *
 * @return On success, zero is returned and svc_list points to a newly
 *         allocated structure. Otherwise, a negative errno value is
 *         returned indicating the error.
 *
 * The returned structure should be freed by calling cxil_free_svc_list().
 */
CXIL_API int cxil_get_svc_list(struct cxil_dev *dev,
			       struct cxil_svc_list **svc_list);

/**
 *@brief Frees the device structure allocated by cxil_get_svc_list()
 *
 * @param svc_list A pointer to the service_list
 */
CXIL_API void cxil_free_svc_list(struct cxil_svc_list *svc_list);

/**
 * @brief Gets service resource usage cxi_svc_rsrc_use from its ID.
 *
 * @param dev The CXI Device
 * @param svc_id The ID returned from cxil_svc_alloc
 * @param rsrcs Destination pointer for cxi_rsrc_use
 *
 * @return On success, zero is returned and rsrcs will contain
 *         info regarding the kernel's view of this service's
 *         resource usage.
 *         Otherwise, a negative errno value is
 *         returned indicating the error.
 *
 */
CXIL_API int cxil_get_svc_rsrc_use(struct cxil_dev *dev, unsigned int svc_id,
				   struct cxi_rsrc_use *rsrcs);

/**
 * @brief Retrieves list with resource usage info for active services
 *
 * @param dev The CXI Device
 * @param rsrc_list Destination pointer for the service list
 *
 * @return On success, zero is returned and rsrc_list points to a newly
 *         allocated structure. Otherwise, a negative errno value is
 *         returned indicating the error.
 *
 * The returned structure should be freed by calling cxil_free_rsrc_list().
 */
CXIL_API int cxil_get_svc_rsrc_list(struct cxil_dev *dev,
				    struct cxil_svc_rsrc_list **rsrc_list);

/**
 *@brief Frees the device structure allocated by cxil_get_svc_rsrc_list()
 *
 * @param rsrc_list A pointer to the service_list
 */
CXIL_API void cxil_free_svc_rsrc_list(struct cxil_svc_rsrc_list *rsrc_list);


/**
 * @brief Update a CXI service.
 *
 * Updating a service is a privileged operation.
 *
 * @param dev The Cassini Device
 * @param desc Pointer to a service descriptor that contains the updates to
 *             the descriptor that was initially returned by cxil_alloc_svc.
 *             Currently all updates are honored except changes to any
 *             resource_limits.
 * @param fail_info Pointer to a structure to which detailed information
 *                  will be written if service allocation fails.
 *                  May be NULL.
 *                  -- Currently unused. Always NULL.
 *
 * @return On success, 0 is returned. Otherwise a negative errno value
 *         is returned indicating the error.
 */
CXIL_API int cxil_update_svc(struct cxil_dev *dev,
			     const struct cxi_svc_desc *desc,
			     struct cxi_svc_fail_info *fail_info);
/**
 * @brief Allocates a CXI service. Every network interface is associated with a
 *        service. A service defines a set of VNIs, TCs, and resources that
 *        members have access to.
 *
 * Service allocation is a privileged operation. Services allow an
 * administrator to control access to VNIs and TCs, and to partition local
 * resources.
 *
 * @param dev The Cassini Device
 * @param desc Pointer to a service descriptor that contains requests for
 *             various resources and optionally identifies member processes,
 *             tcs, vnis, etc. See cxi_svc_desc.
 * @param fail_info Pointer to a structure to which detailed information
 *                  will be written if service allocation fails.
 *                  May be NULL.
 *
 * @return On success, svc_id > 0 is returned. Otherwise a negative
 *         errno value is returned indicating the error.
 */
CXIL_API int cxil_alloc_svc(struct cxil_dev *dev,
			    const struct cxi_svc_desc *desc,
			    struct cxi_svc_fail_info *fail_info);

/**
 * @brief Destroys a CXI Service and releases reserved resources.
 *
 * @param dev The Cassini device
 * @param svc_id The Service ID returned initially from cxil_alloc_svc
 *
 * @return On success, zero is returned. Otherwise, a negative errno value
 *         is returned indicating the error.
 */
CXIL_API int cxil_destroy_svc(struct cxil_dev *dev, unsigned int svc_id);

/**
 * @brief Sets lnis_per_rgid (lpr) of a service.
 *
 * @param dev The CXI Device
 * @param svc_id The ID returned from cxil_svc_alloc
 * @param lnis_per_rgid Number of processes per resource group (Cassini RGID)
 *
 * @return On success, zero is returned and the lnis_per_rgid will be set
 *         in the service indicated by the svc_id. Otherwise, a negative
 *         errno value is returned indicating the error.
 *
 */
CXIL_API int cxil_set_svc_lpr(struct cxil_dev *dev, unsigned int svc_id,
			      unsigned int lnis_per_rgid);

/**
 * @brief Gets lnis_per_rgid (lpr) of a service.
 *
 * @param dev The CXI Device
 * @param svc_id The ID returned from cxil_svc_alloc
 * @param lnis_per_rgid Number of processes per resource group (Cassini RGID)
 *
 * @return On success, lpr of the indicated svc_id is returned.
 *         Otherwise, a negative errno value is returned indicating the error.
 *
 */
CXIL_API int cxil_get_svc_lpr(struct cxil_dev *dev, unsigned int svc_id);

/**
 * @brief Enable or disable a service.
 *
 * @param dev The CXI Device
 * @param svc_id The ID of the service to update
 * @param enable True to enable, false to disable
 *
 * @return On success, zero is returned.
 *         Otherwise, a negative errno value is returned indicating the error.
 */
CXIL_API int cxil_svc_enable(struct cxil_dev *dev, unsigned int svc_id,
			     bool enable);

/**
 * @brief Enable or disable exclusive_cp mode for a service.
 *
 * @param dev The CXI Device
 * @param svc_id The ID of the service to update
 * @param exclusive_cp True to enable, false to disable
 *
 * @return On success, zero is returned.
 *         Otherwise, a negative errno value is returned indicating the error.
 */
CXIL_API int cxil_svc_set_exclusive_cp(struct cxil_dev *dev,
				       unsigned int svc_id,
				       bool exclusive_cp);

/**
 * @brief Query whether exclusive_cp mode is enabled for a service.
 *
 * @param dev The CXI Device
 * @param svc_id The ID of the service to query
 * @param exclusive_cp Pointer to bool to receive the exclusive_cp mode
 *
 * @return On success, zero is returned and exclusive_cp is set appropriately.
 *         Otherwise, a negative errno value is returned indicating the error.
 */
CXIL_API int cxil_svc_get_exclusive_cp(struct cxil_dev *dev,
				       unsigned int svc_id, bool *exclusive_cp);

/**
 * @brief Sets a VNI range for a service.
 *
 * The provided range must be exactly representable as a mask/match pair.
 * Requirements:
 *   - The number of values in the range must be a power of two
 *     (1, 2, 4, 8, 16, ...).
 *   - The first value in the range (vni_min) must be a multiple of the
 *     range size.
 *   - The svc must not have the restricted_vnis bit set.
 *
 * For example:
 *   64–127: 64 values, starting value (64) is a multiple of the
 *           range size (64), so the range is valid.
 *   32–95 : 64 values, starting value (32) is not a multiple of the
 *           range size (64), so the range is invalid.
 *
 * @param dev The CXI Device
 * @param svc_id The ID returned from cxil_svc_alloc
 * @param vni_min Minimum VNI value (inclusive)
 * @param vni_max Maximum VNI value (inclusive)
 *
 * @return On success, 0 is returned and the vni range will be set
 *         in the service indicated by the svc_id. Otherwise, a negative
 *         errno value is returned indicating the error.
 *
 */
CXIL_API int cxil_svc_set_vni_range(struct cxil_dev *dev, unsigned int svc_id,
				    uint16_t vni_min, uint16_t vni_max);

/**
 * @brief Gets a VNI range of a service.
 *
 * @param dev The CXI Device
 * @param svc_id The ID returned from cxil_svc_alloc
 * @param_out vni_min Destination pointer for the minimum VNI value
 * @param_out vni_max Destination pointer for the maximum VNI value
 *
 * @return On success, 0 is returned and the vni_min and vni_max will be
 *         returned for svc_id. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_svc_get_vni_range(struct cxil_dev *dev, unsigned int svc_id,
				    uint16_t *vni_min, uint16_t *vni_max);

/**
 * @brief Allocates a CXI LNI (Logical Network Interface) object.  An LNI is a
 *        logical group of hardware resources on a single network device which
 *        belong to a single process.
 *
 * @param dev The device used to allocate the LNI
 * @param lni The new CXI LNI object
 * @param svc_id ID of the service to associate with this LNI. To use an
 *               unrestricted service, pass in CXI_DEFAULT_SVC_ID.
 *
 * @return On success, zero is returned and the new LNI is pointed to by the lni
 *         parameter.  Otherwise, a negative errno value is returned indicating
 *         the error.
 */
CXIL_API int cxil_alloc_lni(struct cxil_dev *dev, struct cxil_lni **lni,
			    unsigned int svc_id);

/**
 * @brief Destroys a CXI LNI object.
 *
 * @param lni The CXI LNI object to destroy
 *
 * @return On success, zero is returned. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_destroy_lni(struct cxil_lni *lni);

/**
 * @brief Allocate a communication profile. Communication profiles can be used
 *        allocate a transmit command queue or to change the communication
 *        profile for active transmit command queues by using the LCID
 *        embedded within the communication profile structure. Users should
 *        manage which transmit command queues are using which communication
 *        profile.
 *
 * @param lni LNI of communication profile
 * @param vni VNI of communication profile
 * @param tc Traffic Class label of communication profile
 * @param tc_type Traffic Class label type of communication profile
 * @param cp New CXI communication profile object
 *
 * @return On success, zero is returned and the new communication profile is
 *         pointed to by the cp parameter.  Otherwise, a negative errno value
 *         is returned indicating the error.
 */
CXIL_API int cxil_alloc_cp(struct cxil_lni *lni, unsigned int vni,
			   enum cxi_traffic_class tc,
			   enum cxi_traffic_class_type tc_type,
			   struct cxi_cp **cp);

/**
 * @brief Destroy a communication profile.
 *
 * @param cp CXI communication profile object to destroy
 *
 * @return On success, zero is returned. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_destroy_cp(struct cxi_cp *cp);

/**
 * @brief Modify an exclusive communication profile with new VNI
 *
 * @param lni LNI of communication profile
 * @param cp Communication profile to modify
 * @param vni new value of VNI to be set for this communication profile
 *
 * @return On success, zero is returned and the new VNI for the communication
 *         profile is updated. Otherwise, a negative errno value
 *         is returned indicating the error.
 */
CXIL_API int cxil_modify_cp(struct cxil_lni *lni, struct cxi_cp *cp,
			    unsigned int vni);

/**
 * @brief Atomically reserve a contiguous range of VNI PIDs.
 *
 * cxil_alloc_domain() is used to allocate a Domain using a reserved PID.
 * Reserved PIDs are released when the LNI is destroyed.
 *
 * This interface supports clients need to atomically allocate a range of PIDs.
 * It is not required for domain allocation.
 *
 * @param lni The LNI object used for the PID reservation.
 * @param vni The VNI used for the PID reservation.
 * @param pid The base PID value to reserve.
 * @param count The number of PIDs to reserve.
 *
 * @return On success, the first reserved PID value is returned. Otherwise, a
 *         negative errno value is returned indicating the error.
 */
CXIL_API int cxil_reserve_domain(struct cxil_lni *lni, unsigned int vni,
				 unsigned int pid, unsigned int count);

/**
 * @brief Allocates a CXI Domain object. A CXI Domain encapsulates a range of
 *        logical endpoints on a CXI NIC. The endpoints are defined by a VNI
 *        (Virtual Network Identifier) and PID (Process/VNI Partition ID). A
 *        Domain spans 'pid_granule' logical endpoints. pid_granule is a
 *        property of the device. If the reserved value C_PID_ANY is
 *        supplied as the PID, an unused PID value is automatically assigned to
 *        the Domain.
 *
 * @param lni The LNI object used to allocate the Domain
 * @param vni The VNI value used for Domain allocation
 * @param pid The PID value used for Domain allocation
 * @param domain The new CXI Domain object
 *
 * @return On success, zero is returned and the new Domain is pointed to by the
 *         domain parameter.  Otherwise, a negative errno value is returned
 *         indicating the error.
 */
CXIL_API int cxil_alloc_domain(struct cxil_lni *lni, unsigned int vni,
			       unsigned int pid, struct cxil_domain **domain);

/**
 * @brief Destroys a CXI Domain object.
 *
 * @param domain The CXI Domain object to destroy
 *
 * @return On success, zero is returned. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_destroy_domain(struct cxil_domain *domain);

/**
 * @brief Allocates a Cassini Command Queue (CMDQ) object.  CMDQs are used for
 *        scheduling transmit and receive side DMA operations to a CXI NIC.
 *
 * @param lni The LNI object used to allocate the CMDQ
 * @param evtq Optional event queue, to receive CQ command errors.
 * @param opts Command queue options (size, type of CQ, ...)
 * @param cmdq The new CMDQ object.
 *
 * @return On success, zero is returned and the new CMDQ is pointed to by the
 *         cmdq parameter.  Otherwise, a negative errno value is returned
 *         indicating the error.
 */
CXIL_API int cxil_alloc_cmdq(struct cxil_lni *lni, struct cxi_eq *evtq,
			     const struct cxi_cq_alloc_opts *opts,
			     struct cxi_cq **cmdq);

/**
 * @brief Destroys a Cassini Command Queue (CMDQ) object.
 *
 * @param cmdq The CMDQ object to destroy
 *
 * @return On success, zero is returned. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_destroy_cmdq(struct cxi_cq *cmdq);


/**
 * @brief Get current Cassini Command Queue (CMDQ) ack counter value.
 *
 * @param cmdq The CMDQ object to get the ack counter from.
 * @param ack_counter Current ack counter value set upon success.
 *
 * @return On success, zero is returned and ack_counter is set. Otherwise, a
 *	   negative errno value is returned indicating the error.
 */
CXIL_API int cxil_cmdq_ack_counter(struct cxi_cq *cmdq,
				   unsigned int *ack_counter);

/**
 * @brief Maps virtual addresses into IO address space.
 *
 * @param lni The LNI object
 * @param va The virtual address to map
 * @param len The size of the allocated virtual memory
 * @param flags Mapping flags
 * @param hints Hints used for:
 *        Dmabuf info:
 *                hint->dmabuf_fd
 *                hint->dmabuf_offset
 *                hint->dmabuf_valid
 *        Hugepage size hint for sparse ODP registrations:
 *                hint->huge_shift is the requested hugepage shift value
 *                hint->page_shift must be 0 and flags must not contain
 *                CXI_MAP_PIN, CXI_MAP_FAULT or CXI_MAP_PREFETCH
 *        Testing hugepage capabilities of Cassini
 *                hint->page_shift - requested page size shift value
 *                hint->huge_shift - requested hugepage shift value
 * @param md The Memory Descriptor object
 *
 * @return int On success, zero is returned and the iova structure contains the
 *         mapped iova data.  Otherwise, a negative errno value is returned
 *         indicating the error.
 */
CXIL_API int cxil_map(struct cxil_lni *lni, void *va, size_t len,
		      uint32_t flags, struct cxi_md_hints *hints,
		      struct cxi_md **md);

/**
 * @brief Unmaps virtual addresses from IO address space.
 *
 * @param md The Memory Descriptor object to unmap
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_unmap(struct cxi_md *md);

/**
 * @brief Fault in pages of virtual address range.
 *
 * @param md The Memory Descriptor object to reference.
 * @param va The virtual address to map
 * @param len The size of the allocated virtual memory
 * @param flags Mapping flags
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_update_md(struct cxi_md *md, void *va, size_t len,
			    uint32_t flags);

/**
 * @brief Allocates a PtlTE.
 *
 * @param lni The LNI object
 * @param evtq Pointer (optional) to an EQ structure
 * @param opts PtlTE options
 * @param pte Pointer to an uninitialized pte pointer
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_alloc_pte(struct cxil_lni *lni, struct cxi_eq *evtq,
			    struct cxi_pt_alloc_opts *opts,
			    struct cxil_pte **pte);

/**
 * @brief Frees a PtlTE.
 *
 * @param pte Initialized pointer returned by cxil_alloc_pte()
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_destroy_pte(struct cxil_pte *pte);

/**
 * @brief Maps a PtlTE to a specific domain and offset.
 *
 * @param pte Initialized pte pointer from cxil_alloc_pte()
 * @param domain Pointer to a domain structure
 * @param pid_offset Offset of the portal in the domain's PID slice
 * @param is_multicast true if address is multicast, otherwise false
 * @param pte_map Pointer to uninitialized pte_map pointer
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_map_pte(struct cxil_pte *pte, struct cxil_domain *domain,
			  unsigned int pid_offset, bool is_multicast,
			  struct cxil_pte_map **pte_map);

/**
 * @brief Unmaps a PtlTE.
 *
 * @param pte_map Initialized pte_map pointer returned by cxil_map_pte()
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_unmap_pte(struct cxil_pte_map *pte_map);

/**
 * @brief Invalidate outstanding messages on a PtlTE list.
 *
 * Invalidate outstanding messages on a PtlTE list with the provided buffer_id.
 * Cassini does not invalidate outstanding messages associated with an LE that
 * is unlinked manually (via a TGQ Unlink command). This function must be used
 * to invalidate messages after manually unlinking an LE and before freeing the
 * LE buffer to avoid data corruption.
 *
 * Note: If the PtlTE is configured with a EQ, target DMA operation events may
 * complete with a RC of C_RC_MST_CANCELLED. These target events should be
 * ignored by the user.
 *
 * @param pte Initialized pte pointer from cxil_alloc_pte().
 * @param buffer_id Buffer ID used when LE was appended.
 * @param list Portal list used when LE was appended.
 *
 * @return int On success, returns zero. Otherwise, a negative errno value is
 *         returned indicating the error.
 */
CXIL_API int cxil_invalidate_pte_le(struct cxil_pte *pte,
				    unsigned int buffer_id,
				    enum c_ptl_list list);
/**
 * @brief Get a struct that contains the following stats associated with a PTE
 * drop_count:    number of dropped messages targeting the PTE since it was last
 *                disabled. drop_count is used to re-enable a PTE which was
 *                disabled due to flow control.
 * state:         state of the PTE.
 * les_reserved:  number of reserved les for the pool associated with the PTE.
 * les_allocated: current usage count of les allocated from the pool associated
 *                with the PTE.
 * les_max:       the max number of les available for the pool associated with
 *                the PTE. This is a sum of number of les_reserved and the
 *                number of shared entries.
 *
 * @param pte Initialized pte pointer from cxil_alloc_pte().
 * @param status struct for resulting information.
 *
 * @return int on Success, returns zero. Otherwise, a negative errno value is
 * returned indicating the error.
 */
CXIL_API int cxil_pte_status(struct cxil_pte *pte,
			     struct cxi_pte_status *status);

/**
 * @brief Transition a disabled PTE to software managed mode
 *
 * @param pte Initialized pte pointer from cxil_alloc_pte().
 * @param drop_count Expected drop_count
 *
 * @return int Returns 0 on Success. Returns -EINVAL if the PTE is not
 * disabled. Returns -ETIMEDOUT if the hardware has issues. Returns
 * -EAGAIN if the PTE drop_count doesn't match the drop_count given,
 * in which case the command should be tried again until success.
 */
CXIL_API int cxil_pte_transition_sm(struct cxil_pte *pte,
				    unsigned int drop_count);

/* libcxi clients also use cxi_cq and cxi_eq structures.  Those
 * structures may be passed to hardware access functions defined in
 * cxi_prov_hw.h.
 */

/**
 * @brief Allocate a CXI event queue.
 *
 * @param lni The LNI object used to allocate the event queue.
 * @param md The MD to associate with the queue.
 * @param attr EQ allocation attributes.
 * @param event_wait Wait object used for asynchronous event notification.
 * @param status_wait Wait object used for asynchronous status update
 * notification.
 * @param evtq The new event queue object.
 *
 * @return int On success, 0 is returned and the new Event Queue is pointed to
 * by the evtq parameter.
 */
CXIL_API int cxil_alloc_evtq(struct cxil_lni *lni, const struct cxi_md *md,
			     const struct cxi_eq_attr *attr,
			     struct cxil_wait_obj *event_wait,
			     struct cxil_wait_obj *status_wait,
			     struct cxi_eq **evtq);

/**
 * @brief Destroy a CXI Event Queue
 *
 * @param evtq The Event Queue object to destroy
 */
CXIL_API int cxil_destroy_evtq(struct cxi_eq *evtq);

/**
 * @brief Adjust a CXI Event Queue reserved FC field value.
 *
 * Software can use an Event Queue reserved FC value to reserve Event Queue
 * space from being used for network receive related events. Allowing software
 * to adjust this value can enable software to implement schemes to prevent
 * Event Queue overruns from happening for non-network receive related events.
 * Note that hardware will ensure Event Queue overruns do not occur for network
 * receive related events.
 *
 * @param evtq The Event Queue to adjust.
 * @param value Value to adjust the current reserved FC value by. Can be
 * positive or negative.
 *
 * @return int On success, the current reserved FC value. On error, -EINVAL if
 * value is valid, or -ENOSPC if a valid value could not be applied.
 */
CXIL_API int cxil_evtq_adjust_reserved_fc(struct cxi_eq *evtq, int value);

/**
 * @brief Resize a CXI Event Queue buffer
 *
 * Resizing an Event Queue is a multi-step process. The first step is to call
 * cxi_eq_resize() to pass a new event buffer to the device. After this call,
 * evtq will continue to reference the old EQ buffer. The device may write a
 * small number of events to the old EQ buffer followed by a special
 * C_EVENT_EQ_SWITCH event to indicate that hardware has transitioned to
 * writing to the new EQ buffer. When this event is detected, software must
 * call cxi_eq_resize_complete() in order to start reading events from the new
 * EQ buffer.
 *
 * The new event queue buffer must use the same translation mechanism as was
 * used to allocate the EQ. If translation is used, the Addressing Context (AC)
 * used by the new MD must match the MD used to allocate the EQ.
 *
 * The new event queue buffer must be cleared before calling cxi_eq_resize().
 *
 * @param evtq The Event Queue to resize.
 * @param queue The new event queue buffer. Must be page aligned.
 * @param queue_len The new event queue buffer length in bytes. Must be page
 * aligned.
 * @param queue_md The new event queue memory descriptor.
 *
 * @return int On success, 0 is returned and a resized buffer has been
 * submitted to the device.
 */
CXIL_API int cxil_evtq_resize(struct cxi_eq *evtq, void *queue,
			      size_t queue_len, struct cxi_md *queue_md);

/**
 * @brief Complete resizing a CXI Event Queue buffer
 *
 * cxil_evtq_resize_complete() must be called after an EQ has been resized
 * using cxil_evtq_resize() and a C_EVENT_EQ_SWITCH event was delivered. See
 * the documentation for cxil_evtq_resize().
 *
 * @param evtq The Event Queue to being resized.
 *
 * @return int On success, 0 is returned and evtq references the new, resized
 * buffer.
 */
CXIL_API int cxil_evtq_resize_complete(struct cxi_eq *evtq);

/**
 * @brief Clear a wait object event
 *
 * @param wait the wait object to clear
 */
CXIL_API void cxil_clear_wait_obj(struct cxil_wait_obj *wait);

/**
 * @brief Allocate a wait object
 *
 * @param lni The LNI object used to allocate the wait object
 * @param wait The new wait object
 * @return int On success, 0 is returned and the new wait object is
 * pointed to by the wait parameter.
 */
CXIL_API int cxil_alloc_wait_obj(struct cxil_lni *lni,
				 struct cxil_wait_obj **wait);
/**
 * @brief Destroy a wait object
 *
 * @param wait The wait object to destroy
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_destroy_wait_obj(struct cxil_wait_obj *wait);

/**
 * @brief Get the wait object's file descriptor to poll on.
 *
 * @param wait The wait object
 * @return The file descriptor to pool on.
 */
CXIL_API int cxil_get_wait_obj_fd(struct cxil_wait_obj *wait);

/**
 * @brief Allocate a CXI counting event
 *
 * It is the user's responsibility to initialize the buffer pointed to
 * by wb.
 *
 * @param lni The LNI object used to allocate the wait object
 * @param wb The writeback buffer the counting event should use
 * @param ct The new counting event object
 * @return On success, 0 is returned and the new counting event object is
 * pointed to by the ct parameter.
 */
CXIL_API int cxil_alloc_ct(struct cxil_lni *lni, struct c_ct_writeback *wb,
			   struct cxi_ct **ct);

/**
 * @brief Update a CXI counting event with a new wb pointer
 *
 * @param ct The counting event object
 * @param wb The new writeback buffer the counting event should use
 * @return On success, 0 is returned and the counting event object is
 * now using the new writeback address.
 */
CXIL_API int cxil_ct_wb_update(struct cxi_ct *ct, struct c_ct_writeback *wb);

/**
 * @brief Destroy a counting event object
 *
 * @param ct The counting event object to destroy
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_destroy_ct(struct cxi_ct *ct);

/**
 * @brief Map the CSRs belonging to the device into userspace
 *
 * @param dev The Cassini device
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_map_csr(struct cxil_dev *dev);

/**
 * @brief Read a CSR
 *
 * @param dev The Cassini device
 * @param csr The CSR to read (C_MB_STS_REV, ...)
 * @param value Buffer to store the value read.
 * @param csr_len Size of value in bytes. Must be a multiple of 8.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_read_csr(struct cxil_dev *dev, unsigned int csr,
			   void *value, size_t csr_len);

/**
 * @brief Write a CSR
 *
 * @param dev The Cassini device
 * @param csr The CSR to write
 * @param value Buffer containing the value to store.
 * @param csr_len Size of value in bytes. Must be a multiple of 8.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_write_csr(struct cxil_dev *dev, unsigned int csr,
			    const void *value, size_t csr_len);
/**
 * @brief Byte/8-bit Write a CSR. Only use for CSRs that specify SWW8
 *
 * @param dev The Cassini device
 * @param csr The CSR to write
 * @param offset Offset in bytes into the CSR for desired field.
 * @param value Buffer containing the value to store.
 * @param csr_len Size of CSR in bytes. Must be a multiple of 8.
 *                Only used for sanity checking.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */

CXIL_API int cxil_write8_csr(struct cxil_dev *dev, unsigned int csr,
			     unsigned int offset, const void *value,
			     size_t csr_len);

/**
 * @brief Read a Cassini performance counter
 *
 * @param dev The Cassini device.
 * @param cntr The counter to read.
 * @param value Buffer to store the counter value.
 * @param ts Timestamp of the counter sample.
 *
 * Valid counter values are enumerated in enum c_cntr_type.
 * Timestamp is only returned if ts is non-NULL.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_read_cntr(struct cxil_dev *dev, unsigned int cntr,
			    uint64_t *value, struct timespec *ts);

/**
 * @brief Get one or more Cassini performance counters
 *
 * @param dev The Cassini device.
 * @param count The size of cntr and value arrays.
 * @param cntr The counters to read.
 * @param value Buffer to store the counter values.
 * @param ts Timestamp of the counter sample.
 *
 * Timestamp is only returned if ts is non-NULL.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_read_n_cntrs(struct cxil_dev *dev, unsigned int count,
			       const enum c_cntr_type *cntr, uint64_t *value,
			       struct timespec *ts);

/**
 * @brief Get all Cassini performance counters
 *
 * @param dev The Cassini device.
 * @param value Buffer to store the C_CNTR_SIZE counter values.
 * @param ts Timestamp of the counter sample.
 *
 * Timestamp is only returned if ts is non-NULL.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_read_all_cntrs(struct cxil_dev *dev, uint64_t *value,
				 struct timespec *ts);

/**
 * @brief Call Cassini inbound wait to flush internal buffers.
 *
 * @param dev The Cassini device.
 *
 * @return On success, 0 is returned. Otherwise, a negative errno
 *         value is returned indicating the error.
 */
CXIL_API int cxil_inbound_wait(struct cxil_dev *dev);

/**
 * @brief Perform an SBus operation
 *
 * Performs target SBus operation on the single Cassini SBus ring
 *
 * @param dev The Cassini device
 * @param params The SBus command parameters
 * @param rsp_data pointer to store response data
 * @param result_code pointer to store result code from SBus request
 * @param overrun pointer to store request overrun condition
 *
 * @return 0 on success, negative errno on failure
 */
CXIL_API int cxil_sbus_op(struct cxil_dev *dev,
			  const struct cxi_sbus_op_params *params,
			  uint32_t *rsp_data, uint8_t *result_code,
			  uint8_t *overrun);

/* Passthrough for cxil_sbus_op(). Do not use. */
CXIL_API int cxil_sbus_op_compat(struct cxil_dev *dev, int ring,
				 uint32_t req_data, uint8_t data_addr,
				 uint8_t rx_addr, uint8_t command,
				 uint32_t *rsp_data, uint8_t *result_code,
				 uint8_t *overrun, int timeout,
				 unsigned int flags);

/**
 * @brief Perform an sbus op reset
 *
 * Note this doesn't reset the sbus - all it does is clear the MB
 * accessor registers.
 *
 * @param dev The Cassini device.
 *
 * @return 0 on success, negative errno on failure
 */
CXIL_API int cxil_sbus_op_reset(struct cxil_dev *dev);

/* Passthrough for cxil_sbus_op(). Do not use. */
CXIL_API int cxil_sbus_op_reset_compat(struct cxil_dev *dev, int ring);

/**
 * @brief Perform a SERDES operation
 *
 * @param dev The Cassini device.
 *
 * @return 0 on success, negative errno on failure
 */
CXIL_API int cxil_serdes_op(struct cxil_dev *dev, int port_num,
			    uint64_t serdes_sel, uint64_t op, uint64_t data,
			    uint16_t *result, int timeout, unsigned int flags);

/**
 * @brief Get device NIC amo to PCIe fetch add remap value
 *
 * @param dev The Cassini device.
 *
 * @return 0 on success in addition to setting the amo_remap_to_pcie_fadd
 * variable, negative errno on failure.
 */
CXIL_API int cxil_get_amo_remap_to_pcie_fadd(struct cxil_dev *dev,
					     int *amo_remap_to_pcie_fadd);

/**
 * @brief Get the page size of a virtual address
 *
 * @param base The virtual address
 *
 * @return page size or errno
 */
CXIL_API size_t cxil_page_size(void *base);

/**
 * @brief Get whether the kernel supports copy-on-fork
 *
 * @return true if supported; otherwise false
 */
CXIL_API bool cxil_is_copy_on_fork(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __LIBCXI_H__ */
