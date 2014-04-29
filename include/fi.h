/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _FI_H_
#define _FI_H_

#include <endian.h>
#include <byteswap.h>
#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_atomic.h>


#ifdef __cplusplus
extern "C" {
#endif

#define PFX "libfabric: "

#ifdef INCLUDE_VALGRIND
#   include <valgrind/memcheck.h>
#   ifndef VALGRIND_MAKE_MEM_DEFINED
#      warning "Valgrind requested, but VALGRIND_MAKE_MEM_DEFINED undefined"
#   endif
#endif

#ifndef VALGRIND_MAKE_MEM_DEFINED
#   define VALGRIND_MAKE_MEM_DEFINED(addr, len)
#endif

#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#else
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#endif

#define max(a, b) ((a) > (b) ? a : b)
#define min(a, b) ((a) < (b) ? a : b)

struct fi_prov {
	struct fi_prov		*next;
	struct fi_ops_prov	*ops;
};

struct uv_dev {
	struct uv_dev		*next;
	char			sysfs_name[FI_NAME_MAX];
	char			dev_name[FI_NAME_MAX];
	char			sysfs_path[FI_PATH_MAX];
	char			dev_path[FI_PATH_MAX];
};

extern struct fid_fabric *g_fabric;
extern int uv_abi_ver;
extern struct uv_dev *udev_head, *udev_tail;

int  fi_init(void);

void uv_ini(void);
void uv_fini(void);
int  uv_init(void);

void ibv_ini(void);
void ibv_fini(void);

void ucma_ini(void);
void ucma_fini(void);
int  ucma_init(void);

void rdma_cm_ini(void);
void rdma_cm_fini(void);

void mlx4_ini(void);
void mlx4_fini(void);

#ifdef HAVE_PSM
void psmx_ini(void);
void psmx_fini(void);
#else
#define psmx_ini()
#define psmx_fini()
#endif

const char *fi_sysfs_path(void);
int fi_read_file(const char *dir, const char *file, char *buf, size_t size);
void __fi_freeinfo(struct fi_info *info);
int fi_poll_fd(int fd);
int fi_sockaddr_len(struct sockaddr *addr);
size_t fi_datatype_size(enum fi_datatype datatype);

#ifndef SYSCONFDIR
#define SYSCONFDIR "/etc"
#endif
#ifndef RDMADIR
#define RDMADIR "rdma"
#endif
#define RDMA_CONF_DIR  SYSCONFDIR "/" RDMADIR
#define FI_CONF_DIR RDMA_CONF_DIR "/fabric"


#ifdef __cplusplus
}
#endif

#endif /* _FI_H_ */
