/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 Los Alamos National Security, LLC.
 *                    All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <sys/types.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "cxip.h"

#include "ofi_osd.h"
#include "ofi_util.h"

/* A picture is worth ...:
 *
 * Shared (named)	Unshared (unnamed)
 * +-----------+	+-----------+
 * | table_hdr |	| table_hdr |
 * +-----------+	+-----------+
 * | idx_arr   |	| table     |
 * +-----------+	+-----------+
 * | table     |
 * +-----------+
 *
 * idx_arr is ONLY used with shared (named) AVs, is not present
 * for unnamed. This is used to share the AV keys (indices).
 *
 * table contains the fabric-specific addresses for TABLE and
 * MAP. Both TABLE and MAP implementations are the same: both
 * use a key (index) equal to the index of the fabric-specific
 * address in 'table'.
 */

#define	_NOCHECK(_x_)		do {/*_x_ prechecked*/} while (0)
#define	_DEADNULL(_x_)		do {/*If _x_ is 0, already dead*/} while (0)
#define	_CHECKNULL(_x_, _do_, ...)	do {		\
		if (!(_x_)) {				\
			CXIP_LOG_ERROR("Bad arguments: " __VA_ARGS__);	\
			_do_;				\
		}					\
	} while (0)

#define	READONLY(av)	(av->attr.flags & FI_READ)

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_AV, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_AV, __VA_ARGS__)

#define	CXIP_IS_SHARED_AV(shared) ((shared) ? 1 : 0)

#define	CXIP_AV_INDEX_SZ(count, shared)					\
		(CXIP_IS_SHARED_AV(shared) * count * sizeof(uint64_t))

#define	CXIP_AV_TABLE_SZ(count, shared)					\
		(sizeof(struct cxip_av_table_hdr) +			\
		 CXIP_AV_INDEX_SZ(count, shared) +			\
		 (count * sizeof(struct cxip_addr)))

/**
 * Return a pointer to the 'table' portion of the AV data.
 *
 * @param av AV pointer
 * @param count number of addresses in table
 *
 * @return struct cxip_addr* pointer to table of cxip_addr structures
 */
static inline struct cxip_addr *cxip_update_av_table(struct cxip_av *av,
						     size_t count)
{
	_NOCHECK(av);

	return (struct cxip_addr *)((char *)av->table_hdr +
				    CXIP_AV_INDEX_SZ(count, av->shared) +
				    sizeof(struct cxip_av_table_hdr));
}

/**
 * Resize (expand) the AV data.
 *
 * This uses the original requested size of the table as the increment.
 *
 * This is only called during insertion, which is only allowed for writable
 * tables. Read-only tables will not attempt to expand the memory.
 *
 * @param av AV pointer
 *
 * @return int 0 on success, <0 on error
 */
static int cxip_resize_av_table(struct cxip_av *av)
{
	void *new_addr;
	size_t old_count;
	size_t new_count;
	size_t old_sz;
	size_t new_sz;

	_NOCHECK(av);

	/* Increase the size */
	old_count = av->table_hdr->size;
	new_count = old_count + av->attr.count;
	old_sz = CXIP_AV_TABLE_SZ(old_count, av->shared);
	new_sz = CXIP_AV_TABLE_SZ(new_count, av->shared);

	if (av->shared) {
		struct cxip_addr *newtable;
		size_t cpysiz;
		size_t clrsiz;

		/* Examples on the web show adjusting the file size before
		 * remapping. The order doesn't seem to be important, however.
		 */
		if (ftruncate(av->shm.shared_fd, new_sz)) {
			CXIP_LOG_ERROR("shared memory truncate: %s\n",
				       strerror(errno));
			return -FI_ENOMEM;
		}
		new_addr =
			mremap(av->table_hdr, old_sz, new_sz, MREMAP_MAYMOVE);
		if (!new_addr || new_addr == MAP_FAILED) {
			CXIP_LOG_ERROR("shared memory remap: %s\n",
				       strerror(errno));
			return -FI_ENOMEM;
		}

		/* Adjust all of these pointers to old locations. */
		av->table_hdr = new_addr;
		av->idx_arr = (uint64_t *)(av->table_hdr + 1);
		av->table = cxip_update_av_table(av, old_count);

		/* Remap has copied the old data into the new (larger) space,
		 * but we need to open a gap between the idx_arr and the table.
		 * So we copy the table data to the new table location, then
		 * invalidate the spaces we opened up.
		 */
		newtable = cxip_update_av_table(av, new_count);
		cpysiz = old_count * sizeof(struct cxip_addr);
		clrsiz = (char *)newtable - (char *)av->table;
		memmove(newtable, av->table, cpysiz);
		memset(av->table, 0xff, clrsiz);
		av->table = newtable;

	} else {
		/* Memory is not shared, so we can do what we want. */
		new_addr = realloc(av->table_hdr, new_sz);
		if (!new_addr) {
			CXIP_LOG_ERROR("memory realloc: %s\n", strerror(errno));
			return -FI_ENOMEM;
		}

		/* Not shared, idx_arr == NULL. */
		av->table_hdr = new_addr;
		av->table = cxip_update_av_table(av, new_count);
	}

	av->table_hdr->size = new_count;

	return 0;
}

/**
 * Return index to the next empty slot in the AV table.
 *
 * @param av AV pointer
 *
 * @return int index on success, <0 if no empty slots
 */
static int cxip_av_get_next_index(struct cxip_av *av)
{
	uint64_t index;

	_NOCHECK(av);

	/* Advance the 'stored' index until we find an open slot. This will
	 * return the index of the open slot, and will leave the 'stored' index
	 * at the next index.
	 */
	while (av->table_hdr->stored < av->table_hdr->size) {
		index = av->table_hdr->stored++;
		if (!av->table[index].valid)
			return index;
	}

	/* Could not find an empty slot. */
	return -1;
}

/**
 * Insert an address into the AV.
 *
 * @param av AV pointer
 * @param addr pointer to list of FSAs
 * @param fi_addr (optional) pointer to space for return indexes
 * @param count number of FSAs in list
 * @param flags insertion flags
 * @param context insertion context
 *
 * @return int number of addresses inserted, <0 on error
 */
static int cxip_check_table_in(struct cxip_av *av, struct cxip_addr *addr,
			       fi_addr_t *fi_addr, size_t count, uint64_t flags,
			       void *context)
{
	struct cxip_addr *av_addr;
	int index;
	int i, ret;

	_NOCHECK(av && addr);

	/* Cannot modify a read-only map. */
	if (READONLY(av))
		return -FI_EINVAL;

	ret = 0;
	for (i = 0; i < count; i++) {
		/* Normally O(1), worst-case O(N). */
		index = cxip_av_get_next_index(av);
		if (index < 0) {
			if (cxip_resize_av_table(av))
				return -FI_ENOMEM;
			index = cxip_av_get_next_index(av);
			if (index < 0) {
				if (fi_addr)
					fi_addr[i] = FI_ADDR_NOTAVAIL;
				continue;
			}
		}

		/* Copy the FSA -> table */
		av_addr = &av->table[index];
		memcpy(av_addr, &addr[i], sizeof(struct cxip_addr));

		/* If keeping an index, insert that, too */
		if (av->idx_arr)
			av->idx_arr[index] = index;

		CXIP_LOG_DBG("inserted 0x%x:%u\n", av_addr->nic, av_addr->pid);

		/* If caller wants it, return the index */
		if (fi_addr)
			fi_addr[i] = (fi_addr_t)index;

		/* Prevent overwrite */
		av_addr->valid = 1;
		ret++;
	}

	return ret;
}

/**
 * Insert addresses into the AV table.
 *
 * @param avfid generic fid_av pointer
 * @param addr pointer to list of FSAs
 * @param fi_addr (optional) pointer to space for return indexes
 * @param count number of FSAs in list
 * @param flags insertion flags
 * @param context insertion context
 *
 * @return int number of addresses inserted, <0 on error
 */
static int cxip_av_insert(struct fid_av *avfid, const void *addr, size_t count,
			  fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct cxip_av *av;

	_CHECKNULL(avfid && addr, return -FI_EINVAL, "fid=%p, addr=%p\n", avfid,
		   addr);

	av = container_of(avfid, struct cxip_av, av_fid);
	return cxip_check_table_in(av, (struct cxip_addr *)addr, fi_addr, count,
				   flags, context);
}

/**
 * Insert a single 'service' address into the AV table.
 *
 * @param avfid generic fid_av pointer
 * @param node node identifier
 * @param service service identifier
 * @param fi_addr AV index for an address
 * @param addr space to contain FSA
 * @param addrlen space containing length
 *
 * @return int number of addresses inserted (1), <0 on error
 */
static int cxip_av_insertsvc(struct fid_av *avfid, const char *node,
			     const char *service, fi_addr_t *fi_addr,
			     uint64_t flags, void *context)
{
	int ret;
	struct cxip_av *av;
	struct cxip_addr addr;

	_CHECKNULL(avfid && service, return -FI_EINVAL, "fid=%p, service=%p\n",
		   avfid, service);

	av = container_of(avfid, struct cxip_av, av_fid);

	ret = cxip_parse_addr(node, service, &addr);
	if (ret)
		return ret;

	ret = cxip_check_table_in(av, &addr, fi_addr, 1, flags, context);

	return ret;
}

/* Fast, internal look up function. */
int _cxip_av_lookup(struct cxip_av *av, fi_addr_t fi_addr,
		    struct cxip_addr *addr)
{
	struct cxip_addr *av_addr;
	uint64_t index = ((uint64_t)fi_addr & av->mask);

	av_addr = &av->table[index];
	if (!av_addr->valid) {
		CXIP_LOG_ERROR("requested address is invalid");
		return -FI_EINVAL;
	}

	*addr = *av_addr;
	return FI_SUCCESS;
}

/**
 * Look up an address in the AV table.
 *
 * 'addrlen' must contain the size of the 'addr' structure in bytes on entry,
 * and will contain the actual size of the FSA structure on return.
 *
 * @param avfid generic fid_av pointer
 * @param fi_addr AV index for an address
 * @param addr space to contain FSA
 * @param addrlen space containing length
 *
 * @return int 0 on success, <0 on error
 */
static int cxip_av_lookup(struct fid_av *avfid, fi_addr_t fi_addr, void *addr,
			  size_t *addrlen)
{
	uint64_t index;
	struct cxip_av *av;
	struct cxip_addr av_addr;
	int ret;

	_CHECKNULL(avfid && addr && addrlen, return -FI_EINVAL,
		   "fid=%p, addr=%p, addrlen=%p\n", avfid, addr, addrlen);

	av = container_of(avfid, struct cxip_av, av_fid);
	index = ((uint64_t)fi_addr & av->mask);
	if (index >= av->table_hdr->size) {
		CXIP_LOG_ERROR("requested address is invalid");
		return -FI_EINVAL;
	}

	ret = _cxip_av_lookup(av, fi_addr, &av_addr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("Failed to look up FI addr: %lu", fi_addr);
		return ret;
	}

	memcpy(addr, &av_addr, MIN(*addrlen, av->addrlen));
	*addrlen = av->addrlen;

	return 0;
}

/**
 * Remove an FSA from the AV table.
 *
 * This leaves 'holes' in the table, which will be filled first on any new
 * insertion.
 *
 * @param avfid generic fid_av pointer
 * @param fi_addr pointer to a list of AV indexes
 * @param count number of indexes in the list
 * @param flags removal flags
 *
 * @return int 0 on success, <0 on error
 */
static int cxip_av_remove(struct fid_av *avfid, fi_addr_t *fi_addr,
			  size_t count, uint64_t flags)
{
	struct cxip_av *av;
	struct cxip_addr *av_addr;
	size_t i;

	_CHECKNULL(avfid && fi_addr, return -FI_EINVAL, "fid=%p, fi_addr=%p\n",
		   avfid, fi_addr);

	av = container_of(avfid, struct cxip_av, av_fid);

	for (i = 0; i < count; i++) {
		uint64_t index;

		index = ((uint64_t)fi_addr[i] & av->mask);
		if (index >= av->table_hdr->size)
			continue;

		/* Clobber the FSA */
		av_addr = &av->table[index];
		av_addr->valid = 0;

		/* If keeping an index, clobber that, too */
		if (av->idx_arr)
			av->idx_arr[index] = (uint64_t)-1;
	}

	/* Reset the 'stored' pointer. If we delete an entry, we reset the
	 * pointer to 0, so that it will search for 'holes' in the existing
	 * memory.
	 */
	av->table_hdr->stored = 0;

	return 0;
}

/**
 * Produce a readable version of an FSA.
 *
 * @param avfid generic fid_av pointer
 * @param addr pointer to an FSA
 * @param buf buffer to contain text string
 * @param len maximum length of the text
 *
 * @return const char* pointer to buf
 */
static const char *cxip_av_straddr(struct fid_av *avfid, const void *addr,
				   char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_CXI, addr);
}

/**
 * Close and destroy the AV structure.
 *
 * @param fid generic file system FID
 *
 * @return int 0 (success)
 */
static int cxip_av_close(struct fid *fid)
{
	struct cxip_av *av;
	int ret = 0;

	_DEADNULL(fid);

	av = container_of(fid, struct cxip_av, av_fid.fid);
	if (ofi_atomic_get32(&av->ref))
		return -FI_EBUSY;

	if (!av->shared) {
		/* This is our memory, so we can simply free it */
		free(av->table_hdr);
	} else {
		struct util_shm *shm = &av->shm;

		/* This is shared memory. The ofi_shm_unmap() call will destroy
		 * the global shared memory name on the first call. There should
		 * be exactly one writer, and when it closes, it SHOULD prevent
		 * new connections. But there may be multiple readers, and when
		 * they close, they should release their own resources, but
		 * should not preclude new readers from attaching.
		 *
		 * We accomplish this by a trick from reading the libfabric
		 * code: if the shm->name structure is NULL, the unmap will not
		 * destroy the underlying FS name.
		 */
		if (READONLY(av)) {
			/* Do not destroy the pseudo-file name */
			free((void *)shm->name);
			shm->name = NULL;
		}

		ret = ofi_shm_unmap(&av->shm);
		if (ret) {
			CXIP_LOG_ERROR("unmap failed: %s\n",
				       strerror(ofi_syserr()));
		}
	}

	ofi_atomic_dec32(&av->domain->ref);
	fastlock_destroy(&av->list_lock);
	free(av);

	return 0;
}

static struct fi_ops cxip_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_av cxip_am_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = cxip_av_insert,
	.insertsvc = cxip_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = cxip_av_remove,
	.lookup = cxip_av_lookup,
	.straddr = cxip_av_straddr
};

static struct fi_ops_av cxip_at_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = cxip_av_insert,
	.insertsvc = cxip_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = cxip_av_remove,
	.lookup = cxip_av_lookup,
	.straddr = cxip_av_straddr
};

/**
 * Open a new AV structure.
 *
 * @param domain fabric domain pointer
 * @param attr AV attributes
 * @param avp pointer to storage for the fid_av pointer
 * @param context context pointer for creation
 *
 * @return int 0 on success, <0 on failure
 */
int cxip_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **avp, void *context)
{
	int ret = 0;
	struct cxip_av *av = NULL;
	struct fi_ops_av *avops;
	struct cxip_domain *dom;
	size_t table_sz;
	size_t addrlen;

	_DEADNULL(domain);
	_CHECKNULL(avp && attr, return -FI_EINVAL, "fidp=%p, attr=%p\n", avp,
		   attr);

	/* Do-no-harm checking comes first. See if anything will cause
	 * us to bail out without changing any global states.
	 */

	if ((attr->flags & FI_READ) && !attr->name) {
		CXIP_LOG_ERROR("Invalid read-only and non-shared\n");
		return -FI_EINVAL;
	}

	if (attr->rx_ctx_bits > CXIP_EP_MAX_CTX_BITS) {
		CXIP_LOG_ERROR("Invalid rx_ctx_bits\n");
		return -FI_EINVAL;
	}

	switch (attr->type) {
	case FI_AV_MAP:
		avops = &cxip_am_ops;
		break;
	case FI_AV_TABLE:
		avops = &cxip_at_ops;
		break;
	case FI_AV_UNSPEC:
		/* override and report back through attr */
		attr->type = FI_AV_TABLE;
		avops = &cxip_at_ops;
		break;
	default:
		CXIP_LOG_ERROR("Invalid FI_AV type\n");
		return -FI_EINVAL;
	}

	dom = container_of(domain, struct cxip_domain, dom_fid);
	if (dom->attr.av_type != FI_AV_UNSPEC &&
	    dom->attr.av_type != attr->type) {
		CXIP_LOG_ERROR("Domain incompatible with CXI\n");
		return -FI_EINVAL;
	}

	switch (dom->info.addr_format) {
	case FI_ADDR_CXI:
		addrlen = sizeof(struct cxip_addr);
		break;
	default:
		CXIP_LOG_ERROR("Invalid address format\n");
		return -FI_EINVAL;
	}

	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	/* All subsequent failures must exit through err label  */

	/* Local copy of current attributes. Once copied, we use our local copy
	 * for all references below. At the end, we apply any changes that
	 * should be reported back to the caller through attr.
	 */
	av->attr = *attr;

	/* Initialize */
	av->domain = dom;

	av->av_fid.fid.fclass = FI_CLASS_AV;
	av->av_fid.fid.context = context;
	av->av_fid.fid.ops = &cxip_av_fi_ops;
	av->av_fid.ops = avops;

	av->addrlen = addrlen;

	av->rx_ctx_bits = av->attr.rx_ctx_bits;
	av->mask = av->attr.rx_ctx_bits ?
		((uint64_t)1 << (64 - av->attr.rx_ctx_bits)) - 1 : ~0;

	dlist_init(&av->ep_list);
	fastlock_init(&av->list_lock);
	ofi_atomic_initialize32(&av->ref, 0);

	/* zero count is allowed, means default size */
	if (!av->attr.count)
		av->attr.count = cxip_av_def_sz;

	/* Allocate memory */
	if (av->attr.name) {
		/* Shared memory requested */
		av->shared = 1;
		table_sz = CXIP_AV_TABLE_SZ(av->attr.count, av->shared);

		/* Create the shared memory */
		ret = ofi_shm_map(&av->shm, av->attr.name,
				  READONLY(av) ? 0 : table_sz, READONLY(av),
				  (void **)&av->table_hdr);
		if (ret || av->table_hdr == MAP_FAILED) {
			CXIP_LOG_ERROR("map failed\n");
			ret = -FI_ENOMEM;
			goto err;
		}

		/* first aligned byte after header */
		av->idx_arr = (uint64_t *)(av->table_hdr + 1);

		if (READONLY(av)) {
			/* return the current size and map_addr */
			av->attr.count = av->table_hdr->size;
			av->attr.map_addr = av->idx_arr;
			table_sz =
				CXIP_AV_TABLE_SZ(av->attr.count, av->attr.name);
		} else {
			/* take count as a hint */
			av->table_hdr->size = av->attr.count;
			/* invalidate all index values */
			memset(av->idx_arr, 0xff,
			       av->attr.count * sizeof(uint64_t));
		}
	} else {
		/* simple allocation */
		table_sz = CXIP_AV_TABLE_SZ(av->attr.count, av->shared);
		av->table_hdr = calloc(1, table_sz);
		if (!av->table_hdr) {
			ret = -FI_ENOMEM;
			goto err;
		}

		av->table_hdr->size = av->attr.count;
	}

	/* locate the table data */
	av->table = cxip_update_av_table(av, av->table_hdr->size);

	/* Do not trust persistence of attr->name after return */
	av->attr.name = NULL;

	/* return selected values through attr */
	attr->count = av->attr.count;
	attr->map_addr = av->attr.map_addr;

	/* increment domain reference count */
	ofi_atomic_inc32(&dom->ref);

	/* Return the AV pointer */
	*avp = &av->av_fid;
	return 0;

err:
	free(av);
	return ret;
}
