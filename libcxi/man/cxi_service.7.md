---
title: CXI_SERVICE(7) Version 1.0.0 | Cassini Service API
date: 2022-07-11
---

# NAME

Cassini Service API

# SYNOPSIS

An API for configuring a Cassini NIC to take advantage of:

- Network Application Isolation (VNIs)

- Traffic Classes (TCs)

- Access Control

- Local Resource Partitioning


# DESCRIPTION

A CXI Service is a container of requirements and rules for any
entity that needs to utilize a Cassini NIC.
It defines:

- Which specific NIC resources are needed and how many.

- What traffic classes can be utilized.

- Which VNIs can be used.

- Which users or groups should be allowed to access the aforementioned resources.

In other words, it defines the level of service needed by some set of users or apps.

This API is made available through libcxi, and is used on a per NIC basis.

## Interfaces
_cxil_alloc_svc()_ -  Allocate a CXI Service.

_cxil_destroy_svc()_ - Destroy a CXI Service, frees any reserved resources.

_cxil_update_svc()_ - Updates a CXI Service.

_cxil_get_svc()_ - Given a service ID, get a copy of the associated service descriptor.

_cxil_get_svc_list()_ - Get a list of all service descriptors.

_cxil_free_svc_list()_ - Free list allocated by cxil_get_svc_list().

_cxil_get_svc_rsrc_use()_ - Given a service ID, get resource usage information for the service.

_cxil_get_svc_rsrc_list()_ - Get resource usage information for all services.

_cxil_free_svc_rsrc_list()_ - Free list allocated by cxil_get_svc_rsrc_list().

_cxil_svc_enable()_ - Enable/Disable a CXI Service.

_cxil_svc_set_exclusive_cp()_ - Set the exclusive_cp bit for the CXI Service.

_cxil_svc_get_exclusive_cp()_ - Get the exclusive_cp bit for the CXI Service.

_cxil_svc_set_vni_range()_ - Set a VNI range for a CXI service.

_cxil_svc_get_vni_range()_ - Get the VNI range for a CXI service.

1. **Default service**

The Cassini driver provides a "default service" (ID: 1), which provides
unfettered access to a Cassini NIC. It provides access to all NIC resources,
allows use of any VNI or TC, and all users and groups may utilize it.

This default service can be useful for validation purposes, but it is
inherently unsafe. Admins may enable or disable this default service by
using the _cxi_service_ command line utility.

2. **Allocating a service** - _cxil_alloc_svc()_

```
int cxil_alloc_svc(struct cxil_dev *dev_in,
                   const struct cxi_svc_desc *desc,
                   struct cxi_svc_fail_info *fail_info);
```

This function is used to allocate a CXI service. Service allocation is a
privileged operation that requires the capability CAP_SYS_ADMIN.
The caller must first fill out a service descriptor (_cxi_svc_desc_)
to define the rules and requirements for the service they wish to allocate.
This descriptor is passed to the kernel which determines if all the
requests in the descriptor can be honored.

Upon success a service ID greater than 0 is returned. Otherwise a negative
errno value is returned indicating the error.
If the failure was due to a lack of requested HW resources,
the struct _cxi_svc_fail_info_, will contain detailed information about resources
that could not be successfully reserved.

The example below would request a service with no rules or limitations,
like the default service.
```
struct cxi_svc_fail_info fail_info = {};
struct cxi_svc_desc svc_desc = {};
rc = cxil_alloc_svc(dev, &svc_desc, &fail_info);
```

2.1 **Service descriptor** - _cxi_svc_desc_

The struct cxi_svc_desc is the container of requests made to the Cassini driver.
```
struct cxi_svc_desc {
         uint8_t
                 /* Limit access to member processes */
                 restricted_members:1,

                 /* Limit access to defined VNIs */
                 restricted_vnis:1,

                 /* Limit access to defined TCs */
                 restricted_tcs:1,

                 /* Limit access to resources */
                 resource_limits:1,

                 /* Whether a service should be enabled.
                  * Services are enabled by default upon creation.
                  */
                 enable:1,

                 /* Differentiates system and user services */
                 is_system_svc:1,

                 /* Counter Pool ID - Unimplemented, field currently ignored */
                 cntr_pool_id:2;

         /* How many VNIs provided by the user are valid.
          * Must be non-zero if restricted_vnis is true.
          */
         uint8_t num_vld_vnis;

         /* VNIs allowed by this service */
         uint16_t vnis[CXI_SVC_MAX_VNIS];

         bool tcs[CXI_TC_MAX];

         struct {
                 union cxi_svc_member {
                         __kernel_uid_t  uid;
                         __kernel_gid_t  gid;
                 } svc_member;
                 enum cxi_svc_member_type type;
         } members[CXI_SVC_MAX_MEMBERS];

         struct cxi_rsrc_limits limits;

         unsigned int svc_id;
 };
 ```
Common settings that can be requested when allocating a service are described below.

2.1.1 **Restricting members**

A service can be allocated that limits access to specific UIDs or GIDs.
A combination of CXI_SVC_MAX_MEMBERS UIDs or GIDs may be provided.
To do so, set "restricted_members=1" and fill out the "members" structure.

```
/* SVC Member */
struct {
        union cxi_svc_member {
                __kernel_uid_t  uid;
                __kernel_gid_t  gid;
        } svc_member;
        enum cxi_svc_member_type type;
} members[CXI_SVC_MAX_MEMBERS];
```

```
/* SVC Member Types */
enum cxi_svc_member_type {
	CXI_SVC_MEMBER_IGNORE,
	CXI_SVC_MEMBER_UID,
	CXI_SVC_MEMBER_GID,

	CXI_SVC_MEMBER_MAX,
};
```

For example, to limit use of this service to users that belong to group 100,
set up the service descriptor as follows:
```
struct cxi_svc_desc svc_desc = {};
svc_desc.restricted_members = 1;
svc_desc.members[0].svc_member.gid = 100;
svc_desc.members[0].type = CXI_SVC_MEMBER_GID;
```

2.1.2 **Restricting traffic classes**

A service can be created that limits access to specific HPC Traffic
Classes. A value of true written to an index in the boolean array "tcs"
indicates that a particular TC should be enabled for a service.

```
enum cxi_traffic_class {
	/* HRP traffic classes. */
	CXI_TC_DEDICATED_ACCESS,
	CXI_TC_LOW_LATENCY,
	CXI_TC_BULK_DATA,
	CXI_TC_BEST_EFFORT,

	/* Ethernet specific traffic class. */
	CXI_TC_ETH,
	CXI_TC_MAX,
}
```

For example, to enable the BEST_EFFORT and DEDICATED_ACCESS traffic classes,
set up the service descriptor as follows:
```
struct cxi_svc_desc svc_desc = {};
svc_desc.restricted_tcs = 1;
svc_desc.tcs[CXI_TC_DEDICATED_ACCESS] = true;
svc_desc.tcs[CXI_TC_BEST_EFFORT] = true;
```

2.1.3 **VNIs**

A service can either specify up to 4 distinct VNIs or 1 VNI "Range". The "restricted_vnis" bit in the CXI Service descriptor controls this behavior.

2.1.3.1 **VNI Range**

A VNI Range is a contiguous block of VNIs that can be used for communication.

Workflow/Requirements to set a VNI range:

 - The number of values in the range must be a power of two (1, 2, 4, 8, 16, ...).

 - The first value in the range (vni_min) must be a multiple of the range size.

 - VNI Range size is (vni_max - vni_min + 1).

 - The CXI Service should be set up with restricted_vnis=0. This will result in a
   CXI Service that is created in a disabled state.

 - A subsequent call to cxil_svc_set_vni_range() sets the desired VNI range and enables
   the CXI Service.

2.1.3.2 **Restricting VNI**

A service can be created that limits access to certain VNIs.
Up to CXI_SVC_MAX_VNIS VNIs can be specified.
Users must explicitly indicate how many VNIs they wish to utilize.

For example, to limit a service to only have access to VNIs 1 and 2
set up the service descriptor as follows:
```
struct cxi_svc_desc svc_desc = {};
svc_desc.restricted_vnis = 1;
svc_desc.num_vld_vnis = 2
svc_desc.vnis[0] = 1;
svc_desc.vnis[1] = 2;
```

2.1.4  **Restricting resources**

There are many Cassini HW resources that are essential for jobs to function properly.
These resources initially belong to a shared pool. Resources may be reserved for use
exclusively by a particular service. Similarly it is possible to cap the amount of a
resources that a service can utilize.

Relevant structures:
```
enum cxi_rsrc_type {
         CXI_RSRC_TYPE_PTE,
         CXI_RSRC_TYPE_TXQ,
         CXI_RSRC_TYPE_TGQ,
         CXI_RSRC_TYPE_EQ,
         CXI_RSRC_TYPE_CT,
         CXI_RSRC_TYPE_LE,
         CXI_RSRC_TYPE_TLE,
         CXI_RSRC_TYPE_AC,

         CXI_RSRC_TYPE_MAX,
};
```

```
struct cxi_limits {
         uint16_t max;
         uint16_t res;
};
```

```
struct cxi_rsrc_limits {
        union {
                struct {
                        struct cxi_limits ptes;
                        struct cxi_limits txqs;
                        struct cxi_limits tgqs;
                        struct cxi_limits eqs;
                        struct cxi_limits cts;
                        struct cxi_limits les;
                        struct cxi_limits tles;
                        struct cxi_limits acs;
                };
                struct cxi_limits type[CXI_RSRC_TYPE_MAX];
        };
};

```

Example:
```
struct cxi_rsrc_limits = {
	.txqs = {
		.max = 1024,
                .res = 1024,
        },
        .eqs = {
        	.max = 10,
                .res = 5,
		},
};
struct cxi_svc_fail_info fail_info = {};
struct cxi_svc_desc svc_desc = {
	.resource_limits = 1,
	.limits = limits,
};
```

Setting max=0 would disallow usage of a particular resource.
Hence (unlike the contrived example above) max values should
be explicitly specified for each resource type.

To see how many resources are advertised by a Cassini NIC, refer to
_struct cxil_dev_info_ which contains information about each resource.

Unlike other resources, LE and TLE reservations are backed by a
limited number of HW "pools". There are 16 LE pools and 4 TLE pools.
This means only 16 services can be created that reserve LEs, and only
4 services can be created that reserve TLEs.

2.1.5 **Exclusive CP**

The exclusive_cp bit disables sharing of Communication Profiles (CP).
Exclusive CP is not allowed if "restricted_vnis" bit is set in the CXI service descriptor.

2.2 **Fail info** - _cxi_fail_info_

If service allocation fails due to lack of resource availability (-ENOSPC), detailed information about
which resources were unavailable is provided in the struct _cxi_fail_info_.

```
 struct cxi_svc_fail_info {
         /* If a reservation was requested for a CXI_RSRC_TYPE_X and allocation
          * failed, its entry in this array will reflect how many of said
          * resource were actually available to reserve.
          */
         uint16_t rsrc_avail[CXI_RSRC_TYPE_MAX];

         /* True if relevant resources were requested, but none were available. */
         bool no_le_pools;
         bool no_tle_pools;
         bool no_cntr_pools;
 };
```
The information in this structure is only valid for resources that were actually requested.
If a user attempted to reserve LEs but not ACs, fail_info.rsrc_avail[CXI_RSRC_TYPE_LE]
would contain valid info, but fail_info.rsrc_avail[CXI_RSRC_TYPE_AC] should not be referenced.

If cxil_alloc_svc returned a valid svc_id, fail_info should not be referenced.

3. **Deleting a service** - _cxil_destroy_svc()_

```
cxil_destroy_svc(struct cxil_dev *dev, unsigned int svc_id);
```

This function is used to destroy a CXI service. Service deletion is a
privileged operation that requires the capability CAP_SYS_ADMIN.
Destroying a service will release any reserved resources associated with the service
back to a shared pool that can by utilized by other services.

Upon success 0 is returned. Otherwise a negative errno value is returned indicating the error.

A service cannot be destroyed if there are still active references to it,
i.e. no allocated LNIs may reference this service. In addition, the default service
cannot be deleted. However, the default service may be disabled via the _cxi_service_ command line utility.


4. **Updating a service** - _cxil_update_svc()_

```
cxil_update_svc(struct cxil_dev *dev,
                const struct cxi_svc_desc *desc,
                struct cxi_svc_fail_info *fail_info);
```

This function is used to update an existing CXI service. Updating a service is
a privileged operation that requires the capability CAP_SYS_ADMIN.

The usage of this function mirrors _cxil_alloc_svc()_. A user fills out a service descriptor
with the needed changes to an existing service. Typically a user should first call _cxil_get_svc()_
to get the latest version of a descriptor from the kernel, make necessary changes in the returned copy
of the descriptor, then finally call _cxil_update_svc()_.

Modifications to resource reservations are not currently supported.
The _cxi_service_ command line tool provides a simple wrapper to update a service.

5. **Get a service descriptor from service ID** - _cxil_get_svc()_

```
cxil_get_svc(struct cxil_dev *dev, unsigned int svc_id,
             struct cxi_svc_desc *svc_desc);
```

If a valid service ID is passed into this function, information about the service
is stored into the provided service descriptor.
If no service is found with the provided ID, an error is returned.

Example:
```
int rc;
struct cxi_svc_desc desc;
rc = cxil_get_svc(dev, CXI_DEFAULT_SVC_ID, &desc);
```

6. **Get a list of all service descriptors** - _cxil_get_svc_list()_

```
int cxil_get_svc_list(struct cxil_dev *dev,
                      struct cxil_svc_list **svc_list);
```

This function will query the driver for information about all services associated
with a given device. Memory will be allocated on the users behalf for the service
list. The "count" field will indicate how many service descriptors have been copied
in. Must call _cxil_free_svc_list()_ to free memory during cleanup.

```
struct cxil_svc_list {
	unsigned int count;
	struct cxi_svc_desc descs[];
};
```
Example:
```
int rc;
struct cxil_svc_list *list = NULL;
rc = cxil_get_svc_list(dev, &list);
```

7. **Free list of all service descriptors** - _cxil_free_svc_list()_

```
void cxil_free_svc_list(struct cxil_svc_list *svc_list);
```

8. **Get resource usage information for a particular service** - _cxil_get_svc_rsrc_use()_

If a valid service ID is passed into this function, information about the resource usage by the
associated service is stored into the provided structure.
If no service is found with the provided ID, an error is returned.

Example:
```
int rc;
struct cxi_rsrc_use rsrc_use;
rc = cxil_get_svc(dev, CXI_DEFAULT_SVC_ID, &rsrc_use);
```

9. **Get resource usage information for all services** - _cxil_get_svc_rsrc_list()_

This function will query the driver for information regarding the resources that are being utilized
by each service associated with a given device. Memory will be allocated on the users behalf for the
rsrc_use list. The "count" field will indicate the number of rsrc_use structs that have been copied in.
Must call _cxil_free_svc_rsrc_list()_ to free memory during cleanup.

```
struct cxil_svc_rsrc_list {
	unsigned int count;
	struct cxi_rsrc_use rsrcs[];
};
```
```
struct cxi_rsrc_use {
	unsigned int svc_id;
	uint16_t in_use[CXI_RSRC_TYPE_MAX];
};
```
Example:
```
int rc;
struct cxil_svc_rsrc_list *rsrc_list = NULL;
rc = cxil_get_svc_list(dev, &rsrc_list);
```

10. **Free list of resource usage info** - _cxil_free_svc_rsrc_list()_

```
void cxil_free_svc_rsrc_list(struct cxil_svc_rsrc_list *rsrc_list)
```

11. **Enable/disable the CXI Service** - _cxil_svc_enable()_

```
int cxil_svc_enable(struct cxil_dev *dev_in, unsigned int svc_id,
                    bool enable)
```

To enable set the "enable" bool to true.
```
int rc;
rc = cxil_svc_enable(dev, svc_id, true);
```

To disable set the "enable" bool to false.
```
int rc;
rc = cxil_svc_enable(dev, svc_id, false);
```

12. **Set exclusive_cp bit for a CXI Service** - _cxil_svc_set_exclusive_cp()_

Set the "exclusive_cp" bit for a service.

```
int cxil_svc_set_exclusive_cp(struct cxil_dev *dev_in,
                              unsigned int svc_id,
                              bool exclusive_cp)
```

13. **Get exclusive_cp bit for a CXI Service** - _cxil_svc_get_exclusive_cp()_

Get the "exclusive_cp" bit for a service.

```
int cxil_svc_get_exclusive_cp(struct cxil_dev *dev_in,
                              unsigned int svc_id, bool *exclusive_cp)
```

14. **Set a VNI range for a CXI Service** - _cxil_svc_set_vni_range()_

Service ID of service to be updated.
Minimum VNI value (inclusive)
Maximum VNI value (inclusive)

```
int cxil_svc_set_vni_range(struct cxil_dev *dev_in,
                           unsigned int svc_id, uint16_t vni_min,
                           uint16_t vni_max)
```

15. **Get the VNI range for a CXI Service** - _cxil_svc_get_vni_range()_

Get the VNI range associated with a service.
Service ID of service to query.
Pointer to store minimum VNI value (inclusive)
Pointer to store maximum VNI value (inclusive)

```
int cxil_svc_get_vni_range(struct cxil_dev *dev_in,
                           unsigned int svc_id, uint16_t *vni_min,
                           uint16_t *vni_max)
```

# FILES

_uapi/misc/cxi.h_
```
Where cxi service related structures are defined.
```

_libcxi.h_
```
Where cxi service related functions are defined.
```

# SEE ALSO

**cxi_service**(1)
