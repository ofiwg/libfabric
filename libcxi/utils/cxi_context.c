/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2023 Hewlett Packard Enterprise Development LP
 */

/* CXI resource allocation and cleanup for CXI benchmarks
 *
 * This file implements an interface to the functions used to allocate CXI
 * resources. As each is allocated, a pointer to it and a pointer to its
 * cleanup function are appended to an array. This allows a single function
 * (ctx_destroy) to clean/free each resource in the opposite order of their
 * allocation.
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <err.h>
#include <sys/random.h>
#include <sys/mman.h>
#include <linux/mman.h>

#include "utils_common.h"

#define MAX_FUNC_NAME 32

struct cleanup_handle {
	void *obj;
	int (*clean_ret)(void *ptr);
	void (*clean)(void *ptr);
	char func_name[MAX_FUNC_NAME];
};

static struct cleanup_handle *clean_hdl_arr;
static int clean_hdl_count;

int s_page_size;

int get_page_size(void)
{
	if (!s_page_size)
		s_page_size = sysconf(_SC_PAGE_SIZE);

	return s_page_size;
}

/* Register a newly allocated resource to be cleaned */
static void register_for_cleanup(void *obj, int (*clean_ret)(void *),
				 void (*clean)(void *), char *func_name)
{
	struct cleanup_handle *tmp;
	int new_count = clean_hdl_count + 1;

	tmp = realloc(clean_hdl_arr, (new_count * sizeof(*clean_hdl_arr)));
	if (!tmp)
		err(1, "Failed to (re)allocate CXI cleanup array");

	clean_hdl_arr = tmp;
	clean_hdl_arr[clean_hdl_count].obj = obj;
	clean_hdl_arr[clean_hdl_count].clean_ret = clean_ret;
	clean_hdl_arr[clean_hdl_count].clean = clean;
	strncpy(clean_hdl_arr[clean_hdl_count].func_name, func_name,
		MAX_FUNC_NAME);
	clean_hdl_arr[clean_hdl_count].func_name[MAX_FUNC_NAME - 1] = '\0';
	clean_hdl_count = new_count;
}

/* Allocate base context resources */
int ctx_alloc(struct cxi_context *ctx, uint32_t dev_id, uint32_t svc_id)
{
	int rc;
	struct cxi_svc_desc svc_desc = {};

	get_page_size();

	if (!ctx)
		return -EINVAL;
	memset(ctx, 0, sizeof(*ctx));

	ctx->dev_id = dev_id;

	/* device */
	rc = cxil_open_device(ctx->dev_id, &ctx->dev);
	if (rc) {
		fprintf(stderr, "Failed to open CXI device: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(ctx->dev, NULL,
			     (void (*)(void *))cxil_close_device,
			     "cxil_close_device");

	/* verify svc_id exists */
	rc = cxil_get_svc(ctx->dev, svc_id, &svc_desc);
	if (rc) {
		fprintf(stderr, "Failed to get service %d: %s\n",
			svc_id, strerror(-rc));
		return rc;
	}

	/* If the provided service does not restrict VNIs, use one of the
	 * default VNIs. Default VNIs can be determined by checking the
	 * default service, if it exists. If the default service does not
	 * restrict VNIs, pick any non-zero VNI.
	 */
	if (!svc_desc.restricted_vnis && svc_id != CXI_DEFAULT_SVC_ID) {
		/* ensure we can get the default service */
		rc = cxil_get_svc(ctx->dev, CXI_DEFAULT_SVC_ID, &svc_desc);
		if (rc) {
			fprintf(stderr,
				"Failed to get default service %d: %s\n",
				CXI_DEFAULT_SVC_ID, strerror(-rc));
			return rc;
		}
	}
	if (!svc_desc.restricted_vnis) {
		ctx->vni = 1;
	} else {
		if (svc_desc.vnis[0] == 0) {
			fprintf(stderr, "VNI 0 is not a valid VNI.\n");
			return -EINVAL;
		}
		ctx->vni = svc_desc.vnis[0];
	}

	/* LNI */
	rc = cxil_alloc_lni(ctx->dev, &ctx->lni, svc_id);
	if (rc) {
		fprintf(stderr, "Failed to allocate LNI: %s\n", strerror(-rc));
		return rc;
	}
	register_for_cleanup(ctx->lni, (int (*)(void *))cxil_destroy_lni, NULL,
			     "cxil_destroy_lni");

	/* domain */
	rc = cxil_alloc_domain(ctx->lni, ctx->vni, C_PID_ANY, &ctx->dom);
	if (rc) {
		fprintf(stderr, "Failed to allocate Domain: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(ctx->dom, (int (*)(void *))cxil_destroy_domain,
			     NULL, "cxil_destroy_domain");

	ctx->loc_addr.nic = ctx->dev->info.nid;
	ctx->loc_addr.pid = ctx->dom->pid;

	return 0;
}

/* Clean all resources in reverse order of their allocation */
void ctx_destroy(struct cxi_context *ctx)
{
	int err;
	struct cleanup_handle *clean_hdl;

	if (!ctx) {
		fprintf(stderr, "Cannot destroy NULL context\n");
		return;
	}

	while (clean_hdl_count-- > 0) {
		clean_hdl = &clean_hdl_arr[clean_hdl_count];

		if (clean_hdl->clean_ret != NULL) {
			err = clean_hdl->clean_ret(clean_hdl->obj);
			if (err < 0)
				fprintf(stderr, "%s failed: %s\n",
					clean_hdl->func_name, strerror(-err));
		}

		if (clean_hdl->clean != NULL)
			clean_hdl->clean(clean_hdl->obj);
	}
	free(clean_hdl_arr);
	clean_hdl_arr = NULL;
}

/* Allocate a Communication Profile */
int ctx_alloc_cp(struct cxi_context *ctx, enum cxi_traffic_class tc,
		 enum cxi_traffic_class_type tc_type, struct cxi_cp **cp)
{
	int rc;
	struct cxi_cp *cp_tmp;

	if (!ctx || !cp)
		return -EINVAL;

	rc = cxil_alloc_cp(ctx->lni, ctx->vni, tc, tc_type, &cp_tmp);
	if (rc < 0) {
		fprintf(stderr,
			"Failed to allocate Communication Profile: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(cp_tmp, (int (*)(void *))cxil_destroy_cp, NULL,
			     "cxil_destroy_cp");

	*cp = cp_tmp;

	return rc;
}

/* Pattern the given system buffer */
static int pattern_buf(void *buf, size_t buf_len, enum ctx_buf_pat pattern)
{
	int rc;
	size_t pat_len;

	switch (pattern) {
	case CTX_BUF_PAT_URAND:
		pat_len = 0;
		do {
			rc = getrandom((void *)((uintptr_t)buf + pat_len),
				       (buf_len - pat_len), 0);
			if (rc < 0) {
				warn("Failed to randomize buffer");
				break;
			}
			pat_len += rc;
		} while (pat_len < buf_len);

		break;
	case CTX_BUF_PAT_A5:
		memset(buf, 0xa5, buf_len);
		break;
	case CTX_BUF_PAT_ZERO:
		memset(buf, 0, buf_len);
		break;
	default:
		warn("Invalid buffer pattern specified. Buffer not patterned.");
	case CTX_BUF_PAT_NONE:
		break;
	}

	return 0;
}

/* Helper function to fit munmap into the CXI cleanup pattern */
static void munmap_buf(void *ctx_buf)
{
	int rc;
	struct ctx_buffer *buf = (struct ctx_buffer *)ctx_buf;

	rc = munmap(buf->buf, buf->len);
	if (rc)
		err(1, "munmap failed");
}

/* Allocate, pattern, and map a page-aligned buffer */
int ctx_alloc_buf(struct cxi_context *ctx, size_t buf_len,
		  enum ctx_buf_pat pattern, enum hugepage_type hp,
		  struct ctx_buffer **buf)
{
	int rc;
	struct ctx_buffer *buf_tmp;
	int flags;

	if (!ctx || !buf)
		return -EINVAL;

	buf_tmp = malloc(sizeof(struct ctx_buffer));
	if (!buf_tmp)
		err(1, "Failed to allocate data buffer");
	register_for_cleanup(buf_tmp, NULL, free, "free");

	flags = MAP_PRIVATE | MAP_ANONYMOUS;
	if (hp == HP_2M) {
		flags |= MAP_HUGETLB | MAP_HUGE_2MB;
		buf_len = NEXT_MULTIPLE(buf_len, TWO_MB);
	} else if (hp == HP_1G) {
		flags |= MAP_HUGETLB | MAP_HUGE_1GB;
		buf_len = NEXT_MULTIPLE(buf_len, ONE_GB);
	} else {
		buf_len = NEXT_MULTIPLE(buf_len, s_page_size);
	}
	buf_tmp->len = buf_len;
	buf_tmp->buf = mmap(NULL, buf_len, PROT_READ | PROT_WRITE,
			    flags, -1, 0);
	if (!buf_tmp->buf)
		err(1, "Failed to allocate data buffer");
	register_for_cleanup(buf_tmp, NULL, munmap_buf, "munmap");

	rc = pattern_buf(buf_tmp->buf, buf_len, pattern);
	if (rc)
		return rc;

	rc = cxil_map(ctx->lni, buf_tmp->buf, buf_len,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &buf_tmp->md);
	if (rc) {
		fprintf(stderr, "Failed to map CXI buffer: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(buf_tmp->md, (int (*)(void *))cxil_unmap, NULL,
			     "cxil_unmap");

	*buf = buf_tmp;

	return 0;
}

#if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT)
/* Pattern the given GPU buffer */
static int pattern_gpu_buf(void *gpu_buf, size_t buf_len,
			   enum ctx_buf_pat pattern)
{
	int rc = 0;
	size_t pat_len;
	void *host_buf = NULL;

	switch (pattern) {
	case CTX_BUF_PAT_URAND:
		/* This could be done on-device, but would require more work */
		host_buf = malloc(buf_len);
		if (!host_buf)
			err(1, "Failed to allocate buffer for randomization");
		pat_len = 0;
		do {
			rc = getrandom((void *)((uintptr_t)host_buf + pat_len),
				       (buf_len - pat_len), 0);
			if (rc < 0) {
				warn("Failed to randomize buffer");
				break;
			}
			pat_len += rc;
		} while (pat_len < buf_len);

		rc = gpu_memcpy(gpu_buf, host_buf, buf_len, g_memcpy_kind_htod);
		break;
	case CTX_BUF_PAT_A5:
		rc = gpu_memset(gpu_buf, 0xa5, buf_len);
		break;
	case CTX_BUF_PAT_ZERO:
		rc = gpu_memset(gpu_buf, 0, buf_len);
		break;
	default:
		warn("Invalid buffer pattern specified. Buffer not patterned.");
	case CTX_BUF_PAT_NONE:
		break;
	}

	if (host_buf)
		free(host_buf);
	return rc;
}
#endif /* defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT) */

/* Allocate, pattern, and map a GPU buffer */
int ctx_alloc_gpu_buf(struct cxi_context *ctx, size_t buf_len,
		      enum ctx_buf_pat pattern, struct ctx_buffer **buf,
		      int gpu_id)
{
#if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT)
	int rc;
	struct ctx_buffer *buf_tmp;
	void *hints_ptr = NULL;
	struct cxi_md_hints hints = {};

	if (!ctx || !buf)
		return -EINVAL;

	/* set GPU device prior to allocating buffer */
	rc = set_gpu_device(gpu_id);
	if (rc)
		return rc;

	buf_tmp = malloc(sizeof(struct ctx_buffer));
	if (!buf_tmp)
		err(1, "Failed to allocate ctx_buffer");
	register_for_cleanup(buf_tmp, NULL, free, "free");

	rc = gpu_malloc(buf_tmp, buf_len);
	if (rc)
		return rc;
	register_for_cleanup(buf_tmp->buf, gpu_free, NULL, "gpu_free");

	rc = pattern_gpu_buf(buf_tmp->buf, buf_len, pattern);
	if (rc)
		return rc;

	if (g_mem_properties) {
		void *base_addr;
		size_t size;
		int dma_buf_fd = 0;

		base_addr = buf_tmp->buf;
		size = buf_tmp->len;
		rc = g_mem_properties(buf_tmp->buf, &base_addr, &size, &dma_buf_fd);
		if (rc)
			return rc;
		if (dma_buf_fd) {
			hints.dmabuf_fd = dma_buf_fd;
			hints.dmabuf_valid = true;
			hints_ptr = &hints;
		}
        }

	rc = cxil_map(ctx->lni, buf_tmp->buf, buf_len,
		      (CXI_MAP_PIN | CXI_MAP_DEVICE | CXI_MAP_WRITE |
		       CXI_MAP_READ),
		      hints_ptr, &buf_tmp->md);
	if (rc) {
		fprintf(stderr, "cxil_map() failed %d\n", rc);
		return rc;
	}
	register_for_cleanup(buf_tmp->md, (int (*)(void *))cxil_unmap, NULL,
			     "cxil_unmap");

	*buf = buf_tmp;
	return 0;
#else
	err(1, "GPU buffer allocation failed. GPU library not found.");
#endif /* defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT) */
}

/* Allocate an Event Queue */
int ctx_alloc_eq(struct cxi_context *ctx, struct cxi_eq_attr *attr,
		 struct cxi_md *md, struct cxi_eq **eq)
{
	int rc;
	struct cxi_eq *eq_tmp;
	struct cxil_wait_obj *ev_wait = NULL;
	struct cxil_wait_obj *sts_wait = NULL;

	if (!ctx || !eq)
		return -EINVAL;

	rc = cxil_alloc_evtq(ctx->lni, md, attr, ev_wait, sts_wait, &eq_tmp);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Event Queue: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(eq_tmp, (int (*)(void *))cxil_destroy_evtq, NULL,
			     "cxil_destroy_evtq");

	*eq = eq_tmp;

	return rc;
}

/* Allocate an Event Counter */
int ctx_alloc_ct(struct cxi_context *ctx, struct cxi_ct **ct)
{
	int rc;
	struct cxi_ct *ct_tmp;
	struct c_ct_writeback *wb = NULL;

	if (!ctx || !ct)
		return -EINVAL;

	rc = cxil_alloc_ct(ctx->lni, wb, &ct_tmp);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Event Counter: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(ct_tmp, (int (*)(void *))cxil_destroy_ct, NULL,
			     "cxil_destroy_ct");

	cxi_ct_reset_success(ct_tmp);
	cxi_ct_reset_failure(ct_tmp);
	*ct = ct_tmp;

	return rc;
}

/* Allocate a Command Queue */
int ctx_alloc_cq(struct cxi_context *ctx, struct cxi_eq *eq,
		 struct cxi_cq_alloc_opts *opts, struct cxi_cq **cq)
{
	int rc;
	struct cxi_cq *cq_tmp;

	if (!ctx || !cq)
		return -EINVAL;

	rc = cxil_alloc_cmdq(ctx->lni, eq, opts, &cq_tmp);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Command Queue: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(cq_tmp, (int (*)(void *))cxil_destroy_cmdq, NULL,
			     "cxil_destroy_cmdq");

	*cq = cq_tmp;

	return rc;
}

/* Allocate a Portals Table Entry */
int ctx_alloc_pte(struct cxi_context *ctx, struct cxi_eq *eq,
		  struct cxi_pt_alloc_opts *opts, int pid_offset,
		  struct cxil_pte **pte)
{
	int rc;
	struct cxil_pte *pte_tmp;
	struct cxil_pte_map *map_tmp;
	bool is_multicast = false;

	if (!ctx || !pte)
		return -EINVAL;

	rc = cxil_alloc_pte(ctx->lni, eq, opts, &pte_tmp);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Portals Table Entry: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(pte_tmp, (int (*)(void *))cxil_destroy_pte, NULL,
			     "cxil_destroy_pte");

	rc = cxil_map_pte(pte_tmp, ctx->dom, pid_offset, is_multicast,
			  &map_tmp);
	if (rc < 0) {
		fprintf(stderr, "Failed to map Portals Table Entry: %s\n",
			strerror(-rc));
		return rc;
	}
	register_for_cleanup(map_tmp, (int (*)(void *))cxil_unmap_pte, NULL,
			     "cxil_unmap_pte");

	*pte = pte_tmp;

	return rc;
}
