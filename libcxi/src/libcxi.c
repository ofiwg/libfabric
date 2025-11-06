/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Copyright 2018-2024 Hewlett Packard Enterprise Development LP */

/* User space-CXI device interaction */

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>
#include <poll.h>
#include <time.h>

#include "libcxi_priv.h"
#include "uapi/misc/cxi.h"
#include "cassini_cntr_desc.h"

#ifndef NSEC_PER_SEC
#define NSEC_PER_SEC 1000000000
#endif

/**
 * @brief General purpose device write function
 *
 * @param dev Cassini device
 * @param cmd Command pointer
 * @param len Length of command
 * @return On success, 0. Else, negative errno.
 */
static int device_write(const struct cxil_dev_priv *dev,
			const void *cmd, size_t len)
{
	int rc;

	rc = write(dev->fd, cmd, len);
	if (rc != len) {
		if (rc < 0)
			return -errno;

		/* Truncated writes shouldn't occur. Thus, treat as an error. */
		return -EIO;
	}

	return 0;
}

/* Open a CXI Device */
CXIL_API int cxil_open_device(uint32_t dev_id, struct cxil_dev **dev)
{
	int ret;
	struct cxil_dev_priv *new_dev;
	char *dev_name;

	if (!dev)
		return -EINVAL;

	new_dev = calloc(1, sizeof(*new_dev));
	if (new_dev == NULL)
		return -errno;

	ret = asprintf(&dev_name, "/dev/cxi%u", dev_id);
	if (ret == -1) {
		/* there is a question whether asprintf() sets errno */
		ret = errno ? -errno : -ENOMEM;
		goto err_free_dev;
	}

	new_dev->fd = open(dev_name, O_RDWR | O_CLOEXEC);
	if (new_dev->fd == -1) {
		ret = -errno;
		goto err_free_dev_name;
	}
	free(dev_name);

	*dev = &new_dev->dev;
	ret = cxil_query_devinfo(dev_id, &new_dev->dev);
	if (ret)
		goto err_free_dev;

	if (getenv("CXI_FORK_SAFE")) {
		ret = cxil_fork_init();
		if (ret)
			fprintf(stderr, "cxil_fork_init() failed: %d\n", ret);
	}

	return 0;

err_free_dev_name:
	free(dev_name);
err_free_dev:
	free(new_dev);
	return ret;
}

/* Close a CXI Device */
CXIL_API void cxil_close_device(struct cxil_dev *dev_in)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;

	if (dev) {
		if (dev->mapped_csrs)
			munmap(dev->mapped_csrs,
			       dev->mapped_csrs_size);
		close(dev->fd);
		free(dev);
	}
}

/* Get resource usage of all active Services */
CXIL_API int cxil_get_svc_rsrc_list(struct cxil_dev *dev_in,
				    struct cxil_svc_rsrc_list **rsrc_list)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_rsrc_list_get_resp resp;
	struct cxi_svc_rsrc_list_get_cmd cmd = {
		.op = CXI_OP_SVC_RSRC_LIST_GET,
		.resp = &resp,
		.count = 0,
	};
	struct cxil_svc_rsrc_list *list;
	size_t list_size;
	int rc;

	if (!dev_in)
		return -EINVAL;

	/* Initial write to get number of services. */
	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

resize:
	/* Allocate space for resource list */
	list_size = sizeof(struct cxil_svc_rsrc_list) +
		resp.count * sizeof(struct cxi_rsrc_use);
	list = calloc(1, list_size);
	if (list == NULL)
		return -errno;

	cmd.count = resp.count;
	cmd.rsrc_list = list->rsrcs;
	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_list;

	/* Kernel service list grew. Redo. */
	if (resp.count > cmd.count) {
		free(list);
		goto resize;
	}

	*rsrc_list = list;
	list->count = resp.count;

	return 0;
free_list:
	free(list);
	return rc;
}

CXIL_API void cxil_free_svc_rsrc_list(struct cxil_svc_rsrc_list *rsrc_list)
{
	 free(rsrc_list);
}

/* Get a rsrc_use from a svc_id */
CXIL_API int cxil_get_svc_rsrc_use(struct cxil_dev *dev_in,
				   unsigned int svc_id,
				   struct cxi_rsrc_use *rsrcs)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_rsrc_get_resp resp;
	struct cxi_svc_rsrc_get_cmd cmd = {
		.op = CXI_OP_SVC_RSRC_GET,
		.resp = &resp,
	};
	int rc;

	if (!dev_in || !rsrcs)
		return -EINVAL;

	if (rsrcs == NULL)
		return -EINVAL;

	cmd.svc_id = svc_id;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	*rsrcs = resp.rsrcs;

	return 0;
}

/* Get list of all active Services */
CXIL_API int cxil_get_svc_list(struct cxil_dev *dev_in,
			       struct cxil_svc_list **svc_list)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_list_get_resp resp;
	struct cxi_svc_list_get_cmd cmd = {
		.op = CXI_OP_SVC_LIST_GET,
		.resp = &resp,
		.count = 0,
	};
	struct cxil_svc_list *list;
	size_t list_size;
	int rc;

	if (!dev_in)
		return -EINVAL;

	/* Initial write to get number of descriptors. */
	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

resize:
	/* Allocate space for service list */
	list_size = sizeof(struct cxil_svc_list) +
		resp.count * sizeof(struct cxi_svc_desc);
	list = calloc(1, list_size);
	if (list == NULL)
		return -errno;

	cmd.count = resp.count;
	cmd.svc_list = list->descs;
	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_list;

	/* Kernel service list grew. Redo. */
	if (resp.count > cmd.count) {
		free(list);
		goto resize;
	}

	*svc_list = list;
	list->count = resp.count;

	return 0;
free_list:
	free(list);
	return rc;
}

CXIL_API void cxil_free_svc_list(struct cxil_svc_list *svc_list)
{
	 free(svc_list);
}

/* Get a svc_desc from a svc_id */
CXIL_API int cxil_get_svc(struct cxil_dev *dev_in,
			  unsigned int svc_id,
			  struct cxi_svc_desc *desc)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_get_resp resp;
	struct cxi_svc_get_cmd cmd = {
		.op = CXI_OP_SVC_GET,
		.resp = &resp,
	};
	int rc;

	if (!dev_in || !desc)
		return -EINVAL;

	if (desc == NULL)
		return -EINVAL;

	cmd.svc_id = svc_id;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	*desc = resp.svc_desc;

	return 0;
}

/* Update a CXI Service */
CXIL_API int cxil_update_svc(struct cxil_dev *dev_in,
			     const struct cxi_svc_desc *desc,
			     struct cxi_svc_fail_info *fail_info)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_alloc_resp resp;
	struct cxi_svc_alloc_cmd cmd = {
		.op = CXI_OP_SVC_UPDATE,
		.resp = &resp,
	};
	int rc;

	if (!dev_in || !desc)
		return -EINVAL;

	cmd.svc_desc = *desc;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc) {
		if (fail_info)
			*fail_info = resp.fail_info;
		return rc;
	}

	return 0;
}

/* Allocate a CXI Service */
CXIL_API int cxil_alloc_svc(struct cxil_dev *dev_in,
			    const struct cxi_svc_desc *desc,
			    struct cxi_svc_fail_info *fail_info)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_alloc_resp resp;
	struct cxi_svc_alloc_cmd cmd = {
		.op = CXI_OP_SVC_ALLOC,
		.resp = &resp,
	};
	int rc;

	if (!dev_in || !desc)
		return -EINVAL;

	cmd.svc_desc = *desc;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc) {
		if (fail_info)
			*fail_info = resp.fail_info;
		return rc;
	}

	return resp.svc_id;
}

/* Destroy a CXI Service */
int cxil_destroy_svc(struct cxil_dev *dev_in, unsigned int svc_id)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_destroy_cmd cmd = {
		.op = CXI_OP_SVC_DESTROY,
		.svc_id = svc_id,
	};
	int rc;

	if (svc_id < 1)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));

	return rc;
}

/* Update a CXI Service with LNIs per RGID */
CXIL_API int cxil_set_svc_lpr(struct cxil_dev *dev_in, unsigned int svc_id,
			      unsigned int lnis_per_rgid)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_lpr_cmd cmd = {
		.op = CXI_OP_SVC_SET_LPR,
		.svc_id = svc_id,
		.lnis_per_rgid = lnis_per_rgid,
	};

	if (!dev_in)
		return -EINVAL;

	return device_write(dev, &cmd, sizeof(cmd));
}

/* Get the LNIs per RGID of a CXI Service */
CXIL_API int cxil_get_svc_lpr(struct cxil_dev *dev_in, unsigned int svc_id)
{
	int rc;
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_get_value_resp resp = {};
	struct cxi_svc_lpr_cmd cmd = {
		.op = CXI_OP_SVC_GET_LPR,
		.svc_id = svc_id,
		.resp = &resp,
	};

	if (!dev_in)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc < 0)
		return rc;

	return resp.value;
}

CXIL_API int cxil_svc_enable(struct cxil_dev *dev_in, unsigned int svc_id,
			     bool enable)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_enable_cmd cmd = {
		.op = CXI_OP_SVC_ENABLE,
		.svc_id = svc_id,
		.enable = enable,
	};

	if (!dev_in)
		return -EINVAL;

	if (svc_id < 1)
		return -EINVAL;

	return device_write(dev, &cmd, sizeof(cmd));
}

CXIL_API int cxil_svc_set_exclusive_cp(struct cxil_dev *dev_in,
				       unsigned int svc_id,
				       bool exclusive_cp)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_set_exclusive_cp_cmd cmd = {
		.op = CXI_OP_SVC_SET_EXCLUSIVE_CP,
		.svc_id = svc_id,
		.exclusive_cp = exclusive_cp,
	};

	if (!dev_in)
		return -EINVAL;

	if (svc_id < 1)
		return -EINVAL;

	return device_write(dev, &cmd, sizeof(cmd));
}

CXIL_API int cxil_svc_get_exclusive_cp(struct cxil_dev *dev_in,
				       unsigned int svc_id, bool *exclusive_cp)
{
	int rc;
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_get_value_resp resp = {};
	struct cxi_svc_get_exclusive_cp_cmd cmd = {
		.op = CXI_OP_SVC_GET_EXCLUSIVE_CP,
		.svc_id = svc_id,
		.resp = &resp,
	};

	if (!dev_in)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc < 0)
		return rc;

	*exclusive_cp = resp.value;

	return rc;
}

CXIL_API int cxil_svc_set_vni_range(struct cxil_dev *dev_in,
				   unsigned int svc_id, uint16_t vni_min,
				   uint16_t vni_max)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_vni_range_cmd cmd = {
		.op = CXI_OP_SVC_SET_VNI_RANGE,
		.svc_id = svc_id,
		.vni_min = vni_min,
		.vni_max = vni_max,
	};

	if (!dev_in)
		return -EINVAL;

	if (svc_id < 1)
		return -EINVAL;

	return device_write(dev, &cmd, sizeof(cmd));
}

CXIL_API int cxil_svc_get_vni_range(struct cxil_dev *dev_in,
				   unsigned int svc_id, uint16_t *vni_min,
				   uint16_t *vni_max)
{
	int rc;
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_svc_get_vni_range_resp resp = {};
	struct cxi_svc_vni_range_cmd cmd = {
		.op = CXI_OP_SVC_GET_VNI_RANGE,
		.svc_id = svc_id,
		.resp = &resp,
	};

	if (!dev_in || !vni_min || !vni_max)
		return -EINVAL;

	if (svc_id < 1)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	*vni_min = resp.vni_min;
	*vni_max = resp.vni_max;

	return 0;
}

/* Allocate a CXI LNI */
CXIL_API int cxil_alloc_lni(struct cxil_dev *dev_in, struct cxil_lni **lni,
			    unsigned int svc_id)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_lni_alloc_cmd cmd = {};
	struct cxi_lni_alloc_resp resp;
	struct cxil_lni_priv *lni_priv;
	int rc;

	if (!dev_in || !lni || !svc_id)
		return -EINVAL;

	lni_priv = calloc(1, sizeof(*lni_priv));
	if (lni_priv == NULL)
		return -errno;

	cmd.op = CXI_OP_LNI_ALLOC;
	cmd.resp = &resp;
	cmd.svc_id = svc_id;
	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_lni;

	lni_priv->dev = dev;
	lni_priv->lni.id = resp.lni;

	*lni = &lni_priv->lni;

	return 0;

free_lni:
	free(lni_priv);
	return rc;
}

/* Destroy a CXI LNI */
CXIL_API int cxil_destroy_lni(struct cxil_lni *lni)
{
	struct cxi_lni_free_cmd cmd = {
		.op = CXI_OP_LNI_FREE,
	};
	int rc;
	struct cxil_lni_priv *lni_priv;

	if (!lni)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	cmd.lni = lni->id;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	free(lni_priv);
	return 0;
}

/* Allocate a CXI communication profile */
CXIL_API int cxil_alloc_cp(struct cxil_lni *lni, unsigned int vni,
			   enum cxi_traffic_class tc,
			   enum cxi_traffic_class_type tc_type,
			   struct cxi_cp **cp)
{
	struct cxi_cp_alloc_resp resp;
	struct cxi_cp_alloc_cmd cmd = {};
	int rc;
	struct cxil_cp_priv *cp_priv;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !cp)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	cp_priv = calloc(1, sizeof(*cp_priv));
	if (!cp_priv)
		return -errno;

	cmd.op = CXI_OP_CP_ALLOC;
	cmd.resp = &resp;
	cmd.lni = lni->id;
	cmd.vni = vni;
	cmd.tc = tc;
	cmd.tc_type = tc_type;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_cp;

	cp_priv->cp.vni = vni;
	cp_priv->cp.tc = tc;
	cp_priv->cp.tc_type = tc_type;
	cp_priv->cp.lcid = resp.lcid;
	cp_priv->lni_priv = lni_priv;
	cp_priv->cp_hndl = resp.cp_hndl;

	*cp = &cp_priv->cp;

	return 0;

free_cp:
	free(cp_priv);
	return rc;
}

/* Destroy a CXI communication profile */
CXIL_API int cxil_destroy_cp(struct cxi_cp *cp)
{
	struct cxi_cp_free_cmd cmd = {};
	struct cxil_cp_priv *cp_priv;
	int rc;

	if (!cp)
		return -EINVAL;

	cp_priv = container_of(cp, struct cxil_cp_priv, cp);

	cmd.op = CXI_OP_CP_FREE;
	cmd.cp_hndl = cp_priv->cp_hndl;

	rc = device_write(cp_priv->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	free(cp_priv);

	return 0;
}

/* Modify an exclusive CXI communication profile */
CXIL_API int cxil_modify_cp(struct cxil_lni *lni, struct cxi_cp *cp,
			    unsigned int vni)
{
	struct cxi_cp_modify_cmd cmd = {};
	struct cxil_cp_priv *cp_priv;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !cp)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);
	cp_priv = container_of(cp, struct cxil_cp_priv, cp);

	cmd.op = CXI_OP_CP_MODIFY;
	cmd.cp_hndl = cp_priv->cp_hndl;
	cmd.vni = vni;

	return device_write(lni_priv->dev, &cmd, sizeof(cmd));
}

/* Atomically reserve a contiguous range of VNI PIDs. */
CXIL_API int cxil_reserve_domain(struct cxil_lni *lni, unsigned int vni,
				 unsigned int pid, unsigned int count)
{
	struct cxi_domain_reserve_cmd cmd = {};
	struct cxi_domain_reserve_resp resp;
	struct cxil_lni_priv *lni_priv;
	int rc;

	if (!lni)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	cmd.op = CXI_OP_DOMAIN_RESERVE;
	cmd.resp = &resp;
	cmd.lni = lni->id;
	cmd.vni = vni;
	cmd.pid = pid;
	cmd.count = count;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc < 0)
		return rc;

	return resp.pid;
}

/* Allocate a CXI Domain */
CXIL_API int cxil_alloc_domain(struct cxil_lni *lni, unsigned int vni,
			       unsigned int pid, struct cxil_domain **domain)
{
	struct cxi_domain_alloc_cmd cmd = {};
	struct cxi_domain_alloc_resp resp;
	struct cxil_domain_priv *new_domain;
	int rc;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !domain)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	new_domain = calloc(1, sizeof(*new_domain));
	if (new_domain == NULL)
		return -errno;

	cmd.op = CXI_OP_DOMAIN_ALLOC;
	cmd.resp = &resp;
	cmd.lni = lni->id;
	cmd.vni = vni;
	cmd.pid = pid;
	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_domain;

	new_domain->domain.vni = resp.vni;
	new_domain->domain.pid = resp.pid;
	new_domain->lni_priv = lni_priv;
	new_domain->domain_hndl = resp.domain;
	*domain = &new_domain->domain;

	return 0;

free_domain:
	free(new_domain);
	return rc;
}

/* Destroy a CXI Domain */
CXIL_API int cxil_destroy_domain(struct cxil_domain *domain)
{
	struct cxil_domain_priv *domain_priv;
	struct cxi_domain_free_cmd cmd = {
		.op = CXI_OP_DOMAIN_FREE,
	};
	int rc;

	if (!domain)
		return -EINVAL;

	domain_priv = container_of(domain, struct cxil_domain_priv, domain);

	cmd.domain = domain_priv->domain_hndl;

	rc = device_write(domain_priv->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	free(domain_priv);
	return 0;
}

/* Free Command Queue driver resources */
static int free_cmdq(const struct cxil_dev_priv *dev, int cmdq_id)
{
	int rc;
	struct cxi_cq_free_cmd cmd = {
		.op = CXI_OP_CQ_FREE,
		.cq = cmdq_id,
	};

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	return 0;
}

static int free_ct(const struct cxil_dev_priv *dev, int ctn)
{
	int rc;
	struct cxi_ct_free_cmd cmd = {
		.op = CXI_OP_CT_FREE,
		.ctn = ctn,
	};

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	return 0;
}

/* Allocate a CXI counting event. */
CXIL_API int cxil_alloc_ct(struct cxil_lni *lni, struct c_ct_writeback *wb,
			   struct cxi_ct **ct)
{
	struct cxi_ct_alloc_cmd cmd = {};
	struct cxi_ct_alloc_resp resp;
	int rc;
	struct cxil_ct *cct;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !ct)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	cct = calloc(1, sizeof(*cct));
	if (!cct)
		return -errno;

	/* Get a counting event. */
	cmd.op = CXI_OP_CT_ALLOC;
	cmd.resp = &resp;
	cmd.lni = lni->id;
	cmd.wb = wb;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_cct;

	cct->lni_priv = lni_priv;
	cct->ct_hndl = resp.ctn;
	cct->doorbell_len = resp.doorbell.size;

	cct->doorbell = mmap(NULL, resp.doorbell.size,
			     PROT_READ | PROT_WRITE, MAP_SHARED,
			     lni_priv->dev->fd, resp.doorbell.offset);
	if (cct->doorbell == MAP_FAILED) {
		rc = -errno;
		goto release_cct;
	}

	cxi_ct_init(&cct->ct, wb, cct->ct_hndl, cct->doorbell);

	*ct = &cct->ct;

	return 0;

release_cct:
	free_ct(cct->lni_priv->dev, cct->ct_hndl);
free_cct:
	free(cct);

	return rc;
}

/* Update a CXI counting event with a new writeback buffer. */
CXIL_API int cxil_ct_wb_update(struct cxi_ct *ct, struct c_ct_writeback *wb)
{
	int rc;
	struct cxil_ct *cct;
	struct cxi_ct_wb_update_cmd cmd = {
		.op = CXI_OP_CT_WB_UPDATE,
		.wb = wb,
	};

	if (!ct || !wb)
		return -EINVAL;

	cct = container_of(ct, struct cxil_ct, ct);

	cmd.ctn = ct->ctn;

	rc = device_write(cct->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	ct->wb = wb;

	return 0;
}

/* Destroy a CXI counting event. */
CXIL_API int cxil_destroy_ct(struct cxi_ct *ct)
{
	struct cxil_ct *cct;
	int rc;

	if (!ct)
		return -EINVAL;

	cct = container_of(ct, struct cxil_ct, ct);

	if (cct->doorbell && munmap(cct->doorbell, cct->doorbell_len))
		return -errno;
	cct->doorbell = NULL;

	rc = free_ct(cct->lni_priv->dev, cct->ct_hndl);
	if (rc)
		return rc;

	free(cct);

	return 0;
};

/* Allocate a CXI CMDQ */
CXIL_API int cxil_alloc_cmdq(struct cxil_lni *lni, struct cxi_eq *evtq,
			     const struct cxi_cq_alloc_opts *opts,
			     struct cxi_cq **cmdq)
{
	struct cxi_cq_alloc_cmd cmd = {};
	struct cxi_cq_alloc_resp resp;
	int rc;
	struct cxil_cq *ccq;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !cmdq)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	ccq = calloc(1, sizeof(*ccq));
	if (ccq == NULL)
		return -errno;

	/* Get a CMDQ */
	cmd.op = CXI_OP_CQ_ALLOC;
	cmd.resp = &resp;
	cmd.lni = lni->id;
	cmd.eq = evtq ? evtq->eqn : C_EQ_NONE;
	cmd.opts = *opts;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_ccq;

	ccq->lni_priv = lni_priv;
	ccq->cmdq_hndl = resp.cq;
	ccq->size_req = opts->count;
	ccq->cmds_len = resp.cmds.size;
	ccq->csr_len = resp.wp_addr.size;
	ccq->flags = opts->flags;

	/* mmaping queue */
	ccq->cmds = mmap(NULL, resp.cmds.size,
			 PROT_READ | PROT_WRITE, MAP_SHARED, lni_priv->dev->fd,
			 resp.cmds.offset);
	if (ccq->cmds == MAP_FAILED) {
		rc = -errno;
		goto free_cmdq;
	}

	/* mmaping csr block */
	ccq->csr = mmap(NULL, resp.wp_addr.size,
			PROT_READ | PROT_WRITE, MAP_SHARED, lni_priv->dev->fd,
			resp.wp_addr.offset);
	if (ccq->csr == MAP_FAILED) {
		rc = -errno;
		goto unmap_cmds;
	}

	cxi_cq_init(&ccq->hw, ccq->cmds, resp.count, ccq->csr,
		    ccq->cmdq_hndl);

	*cmdq = &ccq->hw;

	return 0;

unmap_cmds:
	munmap(ccq->cmds, ccq->cmds_len);
free_cmdq:
	free_cmdq(lni_priv->dev, resp.cq);
free_ccq:
	free(ccq);
	return rc;
}

/* Destroy a CXI CMDQ */
CXIL_API int cxil_destroy_cmdq(struct cxi_cq *cmdq)
{
	struct cxil_cq *ccq;
	int rc;

	if (!cmdq)
		return -EINVAL;

	ccq = container_of(cmdq, struct cxil_cq, hw);

	/* Rationale:
	 *
	 * munmap() must be called before freeing the CQ, which will fail if the
	 * memory is still mapped: this is a security consideration.
	 *
	 * munmap() compromises the CQ. If the munmap() succeeds, but the
	 * free_cmdq() call fails, the CQ exists, but is unusable. If this call
	 * fails for any reason, there is an application problem, mostly likely
	 * an elevated reference count on the CQ.
	 *
	 * Test code will force failures, so this is written to be safely called
	 * again after the underlying forcing-condition has been removed.
	 */
	if (ccq->csr && munmap(ccq->csr, ccq->csr_len))
		return -errno;
	ccq->csr = NULL;

	if (ccq->cmds && munmap(ccq->cmds, ccq->cmds_len))
		return -errno;
	ccq->cmds = NULL;

	rc = free_cmdq(ccq->lni_priv->dev, ccq->cmdq_hndl);
	if (rc)
		return rc;

	free(ccq);

	return 0;
}

CXIL_API int cxil_cmdq_ack_counter(struct cxi_cq *cmdq,
				   unsigned int *ack_counter)
{
	struct cxi_cq_ack_counter_resp resp;
	struct cxi_cq_ack_counter_cmd cmd = {};
	struct cxil_cq *ccq;
	int rc;

	if (!cmdq || !ack_counter)
		return -EINVAL;

	ccq = container_of(cmdq, struct cxil_cq, hw);

	cmd.op = CXI_OP_CQ_ACK_COUNTER;
	cmd.resp = &resp;
	cmd.cq = ccq->cmdq_hndl;

	rc = device_write(ccq->lni_priv->dev, &cmd, sizeof(cmd));
	if (!rc)
		*ack_counter = resp.ack_counter;

	return rc;
}

/* Perform a map operation */
CXIL_API int cxil_map(struct cxil_lni *lni, void *va, size_t len,
		      uint32_t flags, struct cxi_md_hints *hints,
		      struct cxi_md **md)
{
	struct cxil_md_priv *md_priv;
	struct cxi_atu_map_cmd cmd = {};
	struct cxi_atu_map_resp resp;
	int rc;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !md)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	md_priv = calloc(1, sizeof(*md_priv));
	if (md_priv == NULL)
		return -errno;

	cmd.op = CXI_OP_ATU_MAP;
	cmd.resp = &resp;
	cmd.lni = lni->id;
	cmd.va = (uint64_t)va;
	cmd.len = len;
	cmd.flags = flags;

	if (hints)
		cmd.hints = *hints;

	if (flags & CXI_MAP_PIN && !(flags & CXI_MAP_DEVICE) &&
	    cxil_dontfork_range(va, len)) {
		rc = -EFAULT;
		goto err_dontfork;
	}

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto err_write;

	md_priv->lni_priv = lni_priv;
	md_priv->md_hndl = resp.id;
	md_priv->md = resp.md;
	md_priv->flags = flags;

	*md = &md_priv->md;

	return 0;

err_write:
	fprintf(stderr, "cxil_map: write error\n");
	if ((flags & (CXI_MAP_PIN | CXI_MAP_DEVICE)) == CXI_MAP_PIN)
		cxil_dofork_range(va, len);
err_dontfork:
	free(md_priv);

	return rc;
}

/* Perform an unmap operation */
CXIL_API int cxil_unmap(struct cxi_md *md)
{
	struct cxil_md_priv *md_priv;
	struct cxi_atu_unmap_cmd cmd;
	int rc;

	if (!md)
		return -EINVAL;

	md_priv = container_of(md, struct cxil_md_priv, md);

	cmd.op = CXI_OP_ATU_UNMAP;
	cmd.id = md_priv->md_hndl;
	rc = device_write(md_priv->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	if ((md_priv->flags & (CXI_MAP_PIN | CXI_MAP_DEVICE)) == CXI_MAP_PIN)
		cxil_dofork_range((void *)md_priv->md.va, md_priv->md.len);

	free(md_priv);

	return 0;
}

CXIL_API int cxil_update_md(struct cxi_md *md, void *va, size_t len,
			    uint32_t flags)
{
	int rc;
	struct cxil_md_priv *md_priv;
	struct cxi_atu_update_md_cmd cmd;

	if (!md)
		return -EINVAL;

	md_priv = container_of(md, struct cxil_md_priv, md);

	cmd.op = CXI_OP_ATU_UPDATE_MD;
	cmd.id = md_priv->md_hndl;
	cmd.va = (uint64_t)va;
	cmd.len = len;
	cmd.flags = flags;

	rc = device_write(md_priv->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	return 0;
}

CXIL_API void cxil_clear_wait_obj(struct cxil_wait_obj *wait)
{
	char buf;

	/* There's nothing we can do if an error is returned. This
	 * should never happen since it's a virtual device.
	 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
	pread(wait->fd, &buf, 1, 0);
#pragma GCC diagnostic pop
}

static int release_wait_obj(const struct cxil_dev_priv *dev, unsigned int wait)
{
	struct cxi_wait_free_cmd cmd = {
		.op = CXI_OP_WAIT_FREE,
		.wait = wait,
	};
	int rc;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	return 0;
}

CXIL_API int cxil_alloc_wait_obj(struct cxil_lni *lni,
				 struct cxil_wait_obj **wait_out)
{
	struct cxi_wait_alloc_cmd cmd = {};
	struct cxi_wait_alloc_resp resp;
	int rc;
	struct cxil_wait_obj *wait;
	char *int_fname;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !wait_out)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	wait = calloc(1, sizeof(*wait));
	if (!wait)
		return -errno;

	wait->lni_priv = lni_priv;

	cmd.op = CXI_OP_WAIT_ALLOC;
	cmd.resp = &resp;
	cmd.lni = lni->id;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_wait;

	wait->wait = resp.wait;

	/* Connect to interrupt notifier */
	rc = asprintf(&int_fname,
		      "/sys/class/cxi_user/%s/clients/%u/wait/%u/intr",
		      lni_priv->dev->dev.info.device_name,
		      resp.client_id,
		      resp.wait);
	if (rc == -1) {
		rc = -ENOMEM;
		goto free_wait_driver;
	}

	wait->fd = open(int_fname, O_RDONLY);
	if (wait->fd == -1) {
		rc = -errno;
		goto free_fname;
	}

	free(int_fname);

	/* Read to clear the file creation event */
	cxil_clear_wait_obj(wait);

	*wait_out = wait;

	return 0;

free_fname:
	free(int_fname);

free_wait_driver:
	release_wait_obj(wait->lni_priv->dev, wait->wait);

free_wait:
	free(wait);
	return rc;
}

CXIL_API int cxil_destroy_wait_obj(struct cxil_wait_obj *wait)
{
	if (!wait)
		return -EINVAL;

	close(wait->fd);

	release_wait_obj(wait->lni_priv->dev, wait->wait);

	free(wait);

	return 0;
}

CXIL_API int cxil_get_wait_obj_fd(struct cxil_wait_obj *wait)
{
	return wait->fd;
}

/* Free Event Queue driver resources */
static int free_evtq(const struct cxil_dev_priv *dev, int evtq_id)
{
	int rc;
	struct cxi_eq_free_cmd cmd = {
		.op = CXI_OP_EQ_FREE,
		.eq = evtq_id,
	};

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	return 0;
}

/* Allocate a CXI Event Queue */
CXIL_API int cxil_alloc_evtq(struct cxil_lni *lni, const struct cxi_md *md,
			     const struct cxi_eq_attr *attr,
			     struct cxil_wait_obj *event_wait,
			     struct cxil_wait_obj *status_wait,
			     struct cxi_eq **evtq)
{
	struct cxi_eq_alloc_resp resp;
	struct cxil_eq *eq;
	struct cxil_md_priv *md_priv;
	int rc;
	struct cxi_eq_alloc_cmd cmd = {
		.op = CXI_OP_EQ_ALLOC,
		.resp = &resp,
	};
	struct cxil_lni_priv *lni_priv;

	if (!lni || !attr || !evtq)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	if (!(attr->flags & CXI_EQ_PASSTHROUGH)) {
		if (!md)
			return -EINVAL;

		md_priv = container_of(md, struct cxil_md_priv, md);
	} else {
		md_priv = NULL;
	}

	if (!attr->queue)
		return -EINVAL;

	memset(attr->queue, 0, attr->queue_len);

	eq = calloc(1, sizeof(*eq));
	if (!eq)
		return -errno;

	/* Allocate event queue driver resources */
	cmd.lni = lni->id;
	cmd.queue_md = md_priv ? md_priv->md_hndl : CXI_MD_NONE;
	cmd.event_wait = event_wait ? event_wait->wait : CXI_WAIT_NONE;
	cmd.status_wait = status_wait ? status_wait->wait : CXI_WAIT_NONE;
	cmd.attr = *attr;
	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_eq_mem;

	/* mmaps the event queue SW state descriptor */
	eq->csr = mmap(NULL, resp.csr.size,
			PROT_READ | PROT_WRITE, MAP_SHARED, lni_priv->dev->fd,
			resp.csr.offset);
	if (eq->csr == MAP_FAILED) {
		rc = -errno;
		goto free_eq_driver;
	}

	/* Initialize event queue state information */
	eq->lni_priv = lni_priv;
	eq->evtq_hndl = resp.eq;
	eq->evts = attr->queue;
	eq->evts_len = attr->queue_len;
	eq->evts_md = md_priv;
	eq->csr_len = resp.csr.size;

	cxi_eq_init(&eq->hw, eq->evts, eq->evts_len, eq->evtq_hndl, eq->csr);

	*evtq = &eq->hw;

	return 0;

free_eq_driver:
	free_evtq(lni_priv->dev, resp.eq);
free_eq_mem:
	free(eq);
	return rc;
}

/* Destroy a CXI Event Queue */
CXIL_API int cxil_destroy_evtq(struct cxi_eq *evtq)
{
	struct cxil_eq *eq;
	int rc;

	if (!evtq)
		return -EINVAL;

	eq = container_of(evtq, struct cxil_eq, hw);

	/* Rationale: see cxil_destroy_cmdq() */
	if (eq->csr && munmap(eq->csr, eq->csr_len))
		return -errno;
	eq->csr = NULL;

	/* Free driver resources */
	rc = free_evtq(eq->lni_priv->dev, eq->evtq_hndl);
	if (rc)
		return rc;

	/* Unmap the Event Queue CSRs and memory from userspace */
	free(eq);

	return 0;
}

/* Adjust CXI Event Queue reserved FC field. */
CXIL_API int cxil_evtq_adjust_reserved_fc(struct cxi_eq *evtq, int value)
{
	struct cxi_eq_adjust_reserved_fc_resp resp;
	struct cxi_eq_adjust_reserved_fc_cmd cmd = {
		.op = CXI_OP_EQ_ADJUST_RESERVED_FC,
		.value = value,
		.resp = &resp,
	};
	struct cxil_eq *eq;
	int rc;

	if (!evtq)
		return -EINVAL;

	eq = container_of(evtq, struct cxil_eq, hw);

	cmd.eq_hndl = eq->evtq_hndl;
	rc = device_write(eq->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;
	return resp.reserved_fc;
}

/* Resize a CXI Event Queue */
CXIL_API int cxil_evtq_resize(struct cxi_eq *evtq, void *queue,
			      size_t queue_len, struct cxi_md *queue_md)
{
	struct cxil_eq *eq;
	struct cxil_md_priv *md_priv;
	int rc;
	struct cxi_eq_resize_cmd cmd = {
		.op = CXI_OP_EQ_RESIZE,
		.queue = queue,
		.queue_len = queue_len,
	};

	if (!evtq)
		return -EINVAL;

	eq = container_of(evtq, struct cxil_eq, hw);

	if (queue_md)
		md_priv = container_of(queue_md, struct cxil_md_priv, md);
	else
		md_priv = NULL;

	cmd.eq_hndl = eq->evtq_hndl;
	cmd.queue_md = md_priv ? md_priv->md_hndl : CXI_MD_NONE;
	rc = device_write(eq->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	eq->resized_evts = queue;
	eq->resized_evts_len = queue_len;
	eq->resized_evts_md = md_priv;

	return 0;
}

/* Complete a CXI Event Queue resize */
CXIL_API int cxil_evtq_resize_complete(struct cxi_eq *evtq)
{
	struct cxil_eq *eq;
	int rc;
	struct cxi_eq_resize_complete_cmd cmd = {
		.op = CXI_OP_EQ_RESIZE_COMPLETE,
	};

	if (!evtq)
		return -EINVAL;

	eq = container_of(evtq, struct cxil_eq, hw);

	cmd.eq_hndl = eq->evtq_hndl;
	rc = device_write(eq->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	/* Re-initialize event queue pointers and flip the active buffer */
	eq->evts = eq->resized_evts;
	eq->evts_len = eq->resized_evts_len;
	eq->evts_md = eq->resized_evts_md;

	cxi_eq_init(&eq->hw, eq->evts, eq->evts_len, eq->evtq_hndl, eq->csr);
	evtq->sw_state.reading_buffer_b = !evtq->sw_state.reading_buffer_b;

	return 0;
}

/* Free PtlTE (Portal Table Entry) driver resources */
static int free_pte(const struct cxil_dev_priv *dev, int pte_id)
{
	int rc;
	struct cxi_pte_free_cmd cmd = {
		.op = CXI_OP_PTE_FREE,
		.pte_number = pte_id,
	};

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	return 0;
}

/* Allocate a PtlTE (Portal Table Entry) structure */
CXIL_API int cxil_alloc_pte(struct cxil_lni *lni, struct cxi_eq *evtq,
			    struct cxi_pt_alloc_opts *opts,
			    struct cxil_pte **pte)
{
	struct cxi_pte_alloc_cmd cmd;
	struct cxi_pte_alloc_resp resp;
	struct cxil_pte_priv *pte_priv;
	int evtq_hndl = C_EQ_NONE;
	int rc;
	struct cxil_lni_priv *lni_priv;

	if (!lni || !opts || !pte)
		return -EINVAL;

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);

	pte_priv = calloc(1, sizeof(*pte_priv));
	if (pte_priv == NULL)
		return -errno;

	if (evtq) {
		struct cxil_eq *eq;
		eq = container_of(evtq, struct cxil_eq, hw);
		evtq_hndl = eq->evtq_hndl;
	}

	cmd.op = CXI_OP_PTE_ALLOC;
	cmd.resp = &resp;
	cmd.lni_hndl = lni->id;
	cmd.evtq_hndl = evtq_hndl;
	cmd.opts = *opts;

	rc = device_write(lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_pte_priv;

	pte_priv->lni_priv = lni_priv;
	pte_priv->pte.ptn = resp.pte_number;

	*pte = &pte_priv->pte;

	return 0;

free_pte_priv:
	free(pte_priv);
	return rc;
}

/* Destroy a PtlTE (Portal Table Entry) structure */
CXIL_API int cxil_destroy_pte(struct cxil_pte *pte)
{
	struct cxil_pte_priv *pte_priv;

	if (!pte)
		return -EINVAL;

	pte_priv = container_of(pte, struct cxil_pte_priv, pte);

	if (free_pte(pte_priv->lni_priv->dev, pte_priv->pte.ptn))
		return -errno;

	free(pte_priv);

	return 0;
}

CXIL_API int cxil_map_pte(struct cxil_pte *pte, struct cxil_domain *domain,
			  unsigned int pid_offset, bool is_multicast,
			  struct cxil_pte_map **pte_map)
{
	struct cxi_pte_map_cmd cmd;
	struct cxi_pte_map_resp resp;
	struct cxil_pte_map *cpte_map;
	struct cxil_domain_priv *domain_priv;
	struct cxil_pte_priv *pte_priv;
	int rc;

	if (!pte || !domain || !pte_map)
		return -EINVAL;

	cpte_map = calloc(1, sizeof(*cpte_map));
	if (cpte_map == NULL)
		return -errno;

	domain_priv = container_of(domain, struct cxil_domain_priv, domain);
	pte_priv = container_of(pte, struct cxil_pte_priv, pte);

	cmd.op = CXI_OP_PTE_MAP;
	cmd.resp = &resp;
	cmd.pte_number = pte->ptn;
	cmd.domain_hndl = domain_priv->domain_hndl;
	cmd.pid_offset = pid_offset;
	cmd.is_multicast = is_multicast;

	rc = device_write(pte_priv->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		goto free_cpte_map;

	cpte_map->lni_priv = pte_priv->lni_priv;
	cpte_map->pte_index = resp.pte_index;

	*pte_map = cpte_map;

	return 0;

free_cpte_map:
	free(cpte_map);
	return rc;
}

CXIL_API int cxil_unmap_pte(struct cxil_pte_map *pte_map)
{
	struct cxi_pte_unmap_cmd cmd;
	int rc;

	if (!pte_map)
		return -EINVAL;

	cmd.op = CXI_OP_PTE_UNMAP;
	cmd.pte_index = pte_map->pte_index;

	rc = device_write(pte_map->lni_priv->dev, &cmd, sizeof(cmd));
	if (rc)
		return rc;

	free(pte_map);

	return 0;
}

CXIL_API int cxil_invalidate_pte_le(struct cxil_pte *pte,
				    unsigned int buffer_id,
				    enum c_ptl_list list)
{
	struct cxi_pte_le_invalidate_cmd cmd;
	int rc;
	struct cxil_pte_priv *pte_priv;

	if (!pte)
		return -EINVAL;

	pte_priv = container_of(pte, struct cxil_pte_priv, pte);

	cmd.op = CXI_OP_PTE_LE_INVALIDATE;
	cmd.pte_index = pte->ptn;
	cmd.buffer_id = buffer_id;
	cmd.list = list;

	rc = device_write(pte_priv->lni_priv->dev, &cmd, sizeof(cmd));

	return rc;
}

/* Get a collection of PTE Stats */
CXIL_API int cxil_pte_status(struct cxil_pte *pte,
			     struct cxi_pte_status *status)
{
	struct cxi_pte_status_resp resp;
	struct cxi_pte_status_cmd cmd = {
		.op = CXI_OP_PTE_STATUS,
		.resp = &resp,
	};
	int rc;
	struct cxil_pte_priv *pte_priv;

	if (!pte || !status)
		return -EINVAL;

	pte_priv = container_of(pte, struct cxil_pte_priv, pte);

	resp.status = *status;
	cmd.pte_index = pte->ptn;

	rc = device_write(pte_priv->lni_priv->dev, &cmd, sizeof(cmd));
	if (!rc)
		*status = resp.status;

	return rc;
}

CXIL_API int cxil_pte_transition_sm(struct cxil_pte *pte,
				    unsigned int drop_count)
{
	struct cxil_pte_priv *pte_priv =
		container_of(pte, struct cxil_pte_priv, pte);
	struct cxi_pte_transition_sm_cmd cmd = {
		.op = CXI_OP_PTE_TRANSITION_SM,
		.drop_count = drop_count,
	};
	int rc;

	if (!pte)
		return -EINVAL;

	cmd.pte_index = pte->ptn;
	rc = device_write(pte_priv->lni_priv->dev, &cmd, sizeof(cmd));

	return rc;
}

static bool valid_csr(const struct cxil_dev_priv *dev,
		      unsigned int csr, size_t csr_len)
{
	return dev->mapped_csrs && (csr_len % sizeof(uint64_t) == 0) &&
		csr < dev->mapped_csrs_size &&
		csr + csr_len <= dev->mapped_csrs_size;
}

/* Read a CSR */
CXIL_API int cxil_read_csr(struct cxil_dev *dev_in, unsigned int csr,
			   void *value, size_t csr_len)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	const uint64_t *src;
	uint64_t *dst;
	size_t i;

	if (!valid_csr(dev, csr, csr_len))
		return -EINVAL;

	src = (uint64_t *)((char *)dev->mapped_csrs + csr);
	dst = value;

	csr_len /= sizeof(uint64_t);

	for (i = 0; i < csr_len; i++)
		*dst++ = *src++;

	return 0;
}

/* Write a CSR */
CXIL_API int cxil_write_csr(struct cxil_dev *dev_in, unsigned int csr,
			    const void *value, size_t csr_len)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	const uint64_t *src;
	uint64_t *dst;
	size_t i;

	if (!valid_csr(dev, csr, csr_len))
		return -EINVAL;

	src = value;
	dst = (uint64_t *)((char *)dev->mapped_csrs + csr);

	csr_len /= sizeof(uint64_t);

	for (i = 0; i < csr_len; i++)
		*dst++ = *src++;

	return 0;
}

/* Byte Write a CSR */
CXIL_API int cxil_write8_csr(struct cxil_dev *dev_in, unsigned int csr,
			     unsigned int offset, const void *value,
			     size_t csr_len)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	const uint8_t *src;
	uint8_t *dst;

	if (!valid_csr(dev, csr, csr_len))
		return -EINVAL;

	if (offset >= csr_len)
		return -EINVAL;

	src = ((uint8_t *)value) + offset;
	dst = ((uint8_t *)dev->mapped_csrs) + csr + offset;

	*dst = *src;

	return 0;
}

/* Map the CSRs belonging to the device into userspace */
CXIL_API int cxil_map_csr(struct cxil_dev *dev_in)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_map_csrs_resp resp;
	int rc;
	struct cxi_map_csrs_cmd cmd = {
		.op = CXI_OP_MAP_CSRS,
		.resp = &resp,
	};
	void *csr;

	if (!dev || dev->mapped_csrs != NULL)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));
	if (rc)
		goto err;

	/* mmaps the event queue SW state descriptor */
	csr = mmap(NULL, resp.csr.size,
		   PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
		   resp.csr.offset);

	if (csr == MAP_FAILED) {
		rc = -errno;
		goto err;
	}

	dev->mapped_csrs = csr;
	dev->mapped_csrs_size = resp.csr.size;

	return 0;

err:
	return rc;
}

static int read_sysfs_cntrs_individually(struct cxil_dev *dev,
					 unsigned int count,
					 const enum c_cntr_type *cntr,
					 uint64_t *value,
					 struct timespec *ts)
{
	int rc;
	char fname[100];
	int i;
	int64_t secs;
	uint64_t nsecs;
	FILE *f;
	int c;

	for (i = 0 ; i < count ; ++i) {
		if (count == C1_CNTR_SIZE && cntr == NULL) {
			/*
			 * this is a special case to signal
			 * retrieving all the items.
			 */
			c = i;
			if (c1_cntr_descs[c].name == NULL) {
				value[i] = 0;
				continue;
			}
		} else {
			/*
			 * typical case to read one or more items.
			 */
			c = cntr[i];
			if (c < 0 || c >= C1_CNTR_SIZE ||
			    c1_cntr_descs[c].name == NULL)
				return -EINVAL;
		}

		rc = snprintf(fname, sizeof(fname),
			      "/sys/class/cxi/%s/device/telemetry/%s",
			      dev->info.device_name,
			      c1_cntr_descs[c].name);
		if (rc < 0)
			return -errno;

		f = fopen(fname, "r");
		if (f == NULL)
			return -errno;

		rc = fscanf(f, "%lu@%ld.%lu", &value[i], &secs, &nsecs);
		if (rc != 3)
			return -EINVAL;

		(void)fclose(f);
	}
	if (ts != NULL) {
		ts->tv_sec = secs;
		ts->tv_nsec = nsecs;
	}
	return 0;
}

/*
 * Read a Cassini counter
 */
CXIL_API int cxil_read_cntr(struct cxil_dev *dev, unsigned int cntr,
			    uint64_t *value, struct timespec *ts)
{
	/*
	 * Even though the for-loop in cntr_get_common() will also verify cntr
	 * is valid, do early error checking here since the cost/overhead of
	 * this check is very little.
	 */
	if (dev == NULL || value == NULL || cntr >= C1_CNTR_SIZE ||
	    c1_cntr_descs[cntr].name == NULL)
		return -EINVAL;
	return read_sysfs_cntrs_individually(dev, 1, &cntr, value, ts);
}

/*
 * Read a collection/array of Cassini counters
 */
CXIL_API int cxil_read_n_cntrs(struct cxil_dev *dev, unsigned int count,
			       const enum c_cntr_type *cntr, uint64_t *value,
			       struct timespec *ts)
{
	/*
	 * To have count greater than C1_CNTR_COUNT, there would either have to
	 * be duplicate or erroneous counters in 'cntr'.  There is no benefit
	 * to having duplicates, so disallow such usage.
	 */
	if (dev == NULL || cntr == NULL || value == NULL || count == 0 ||
	    count > C1_CNTR_COUNT)
		return -EINVAL;
	return read_sysfs_cntrs_individually(dev, count, cntr, value, ts);
}

/*
 * Read all of Cassini counters
 *
 * value must have space for C_CNTR_SIZE uint64_t
 */
CXIL_API int cxil_read_all_cntrs(struct cxil_dev *dev, uint64_t *value,
				 struct timespec *ts)
{
	if (dev == NULL || value == NULL)
		return -EINVAL;
	return read_sysfs_cntrs_individually(dev, C1_CNTR_SIZE, NULL, value,
					     ts);
}

CXIL_API int cxil_inbound_wait(struct cxil_dev *dev_in)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_inbound_wait_cmd cmd = {
		.op = CXI_OP_INBOUND_WAIT,
	};

	if (!dev)
		return -EINVAL;

	return device_write(dev, &cmd, sizeof(cmd));
}

/* Perform an SBus operation */
CXIL_API int cxil_sbus_op(struct cxil_dev *dev_in,
			  const struct cxi_sbus_op_params *params,
			  uint32_t *rsp_data, uint8_t *result_code,
			  uint8_t *overrun)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *)dev_in;
	struct cxi_sbus_op_resp resp;
	struct cxi_sbus_op_cmd cmd = {
		.op = CXI_OP_SBUS_OP,
		.resp = &resp,
		.params = *params,
	};
	int rc;

	if (!dev)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));

	if (rc == 0) {
		*rsp_data = resp.rsp_data;
		*result_code = resp.result_code;
		*overrun = resp.overrun;
	}

	return rc;
}

/* Passthrough for cxil_sbus_op(), for compatibility with existing
 * tools. Do not use.
 * TODO: remove eventually.
 */
CXIL_API int cxil_sbus_op_compat(struct cxil_dev *dev_in, int ring,
				 uint32_t req_data, uint8_t data_addr,
				 uint8_t rx_addr, uint8_t command,
				 uint32_t *rsp_data, uint8_t *result_code,
				 uint8_t *overrun, int timeout,
				 unsigned int flags)
{
	struct cxi_sbus_op_params params = {
		.req_data = req_data,
		.data_addr = data_addr,
		.rx_addr = rx_addr,
		.command = command,
		.timeout = timeout,
	};

	if (flags & SBL_FLAG_DELAY_3US)
		params.delay = 3;
	else if (flags & SBL_FLAG_DELAY_4US)
		params.delay = 4;
	else if (flags & SBL_FLAG_DELAY_5US)
		params.delay = 5;
	else if (flags & SBL_FLAG_DELAY_10US)
		params.delay = 10;
	else if (flags & SBL_FLAG_DELAY_20US)
		params.delay = 20;
	else if (flags & SBL_FLAG_DELAY_50US)
		params.delay = 50;
	else if (flags & SBL_FLAG_DELAY_100US)
		params.delay = 100;
	else
		return -EINVAL;

	if (flags & SBL_FLAG_INTERVAL_1MS)
		params.poll_interval = 1;
	else if (flags & SBL_FLAG_INTERVAL_10MS)
		params.poll_interval = 10;
	else if (flags & SBL_FLAG_INTERVAL_100MS)
		params.poll_interval = 100;
	else if (flags & SBL_FLAG_INTERVAL_1S)
		params.poll_interval = 1000;
	else
		return -EINVAL;

	return cxil_sbus_op(dev_in, &params, rsp_data, result_code, overrun);
}

/* Perform an sbus op reset */
CXIL_API int cxil_sbus_op_reset(struct cxil_dev *dev_in)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *)dev_in;
	struct cxi_sbus_op_reset_cmd cmd = {
		.op = CXI_OP_SBUS_OP_RESET,
	};

	if (!dev)
		return -EINVAL;

	return device_write(dev, &cmd, sizeof(cmd));
}

/* Passthrough for cxil_sbus_op(), for compatibility with existing
 * tools. Do not use.
 * TODO: remove eventually.
 */
CXIL_API int cxil_sbus_op_reset_compat(struct cxil_dev *dev, int ring)
{
	return cxil_sbus_op_reset(dev);
}

CXIL_API int cxil_serdes_op(struct cxil_dev *dev_in, int port_num,
			    uint64_t serdes_sel, uint64_t op, uint64_t data,
			    uint16_t *result, int timeout, unsigned int flags)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *)dev_in;
	struct cxi_serdes_op_resp resp;
	struct cxi_serdes_op_cmd cmd = {
		.op = CXI_OP_SERDES_OP,
		.resp = &resp,
		.serdes_sel = serdes_sel,
		.serdes_op = op,
		.data = data,
		.timeout = timeout,
		.flags = flags,
	};
	int rc;

	if (!dev)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));

	if (rc == 0)
		*result = resp.result;

	return rc;
}

int cxil_get_dev_info(struct cxil_dev *dev_in,
		      struct cxi_dev_info_use *devinfo)
{
	struct cxil_dev_priv *dev = (struct cxil_dev_priv *) dev_in;
	struct cxi_dev_info_get_resp resp;
	struct cxi_dev_info_get_cmd cmd = {
		.op = CXI_OP_DEV_INFO_GET,
		.resp = &resp,
	};
	int rc;

	if (!dev_in || !devinfo)
		return -EINVAL;

	if (devinfo == NULL)
		return -EINVAL;

	rc = device_write(dev, &cmd, sizeof(cmd));

	if (rc)
		return rc;

	*devinfo = resp.devinfo;

	return 0;
}
