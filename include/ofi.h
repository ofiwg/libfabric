/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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

#ifndef _OFI_H_
#define _OFI_H_

#include "config.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <netinet/in.h>
#include <ifaddrs.h>

#include <ofi_abi.h>
#include <ofi_file.h>
#include <ofi_lock.h>
#include <ofi_atom.h>
#include <ofi_mem.h>
#include <ofi_net.h>
#include <rdma/providers/fi_prov.h>
#include <rdma/providers/fi_log.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>

#include <ofi_osd.h>


#ifdef __cplusplus
extern "C" {
#endif

#define OFI_CORE_PROV_ONLY	(1ULL << 59)

#define sizeof_field(type, field) sizeof(((type *)0)->field)

#ifndef MIN
#define MIN(a, b) \
	({ typeof (a) _a = (a); \
		typeof (b) _b = (b); \
		_a < _b ? _a : _b; })
#endif

#ifndef MAX
#define MAX(a, b) \
	({ typeof (a) _a = (a); \
		typeof (b) _b = (b); \
		_a > _b ? _a : _b; })
#endif


/*
 * CPU specific features
 */
enum {
	OFI_CLWB_REG		= 2,
	OFI_CLWB_BIT		= (1 << 24),
	OFI_CLFLUSHOPT_REG	= 1,
	OFI_CLFLUSHOPT_BIT	= (1 << 24),
	OFI_CLFLUSH_REG		= 3,
	OFI_CLFLUSH_BIT		= (1 << 23),
};

int ofi_cpu_supports(unsigned func, unsigned reg, unsigned bit);


/* Restrict to size of struct fi_context */
struct fi_prov_context {
	int disable_logging;
	int is_util_prov;
};

struct fi_filter {
	char **names;
	int negated;
};

extern struct fi_filter prov_log_filter;
extern struct fi_provider core_prov;

void ofi_create_filter(struct fi_filter *filter, const char *env_name);
void ofi_free_filter(struct fi_filter *filter);
int ofi_apply_filter(struct fi_filter *filter, const char *name);

void fi_log_init(void);
void fi_log_fini(void);
void fi_param_init(void);
void fi_param_fini(void);
void fi_param_undefine(const struct fi_provider *provider);

const char *ofi_hex_str(const uint8_t *data, size_t len);

static inline uint64_t roundup_power_of_two(uint64_t n)
{
	if (!n || !(n & (n - 1)))
		return n;
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n |= n >> 32;
	n++;
	return n;
}

static inline size_t fi_get_aligned_sz(size_t size, size_t alignment)
{
	return ((size % alignment) == 0) ?
		size : ((size / alignment) + 1) * alignment;
}

#define FI_TAG_GENERIC	0xAAAAAAAAAAAAAAAAULL


uint64_t ofi_max_tag(uint64_t mem_tag_format);
uint64_t ofi_tag_format(uint64_t max_tag);
uint8_t ofi_msb(uint64_t num);

int ofi_send_allowed(uint64_t caps);
int ofi_recv_allowed(uint64_t caps);
int ofi_rma_initiate_allowed(uint64_t caps);
int ofi_rma_target_allowed(uint64_t caps);
int ofi_ep_bind_valid(const struct fi_provider *prov, struct fid *bfid,
		      uint64_t flags);
int ofi_check_rx_mode(const struct fi_info *info, uint64_t flags);

uint64_t fi_gettime_ms(void);
uint64_t fi_gettime_us(void);


#define OFI_ENUM_VAL(X) X
#define OFI_STR(X) #X
#define OFI_STR_INT(X) OFI_STR(X)


/*
 * Key Index
 */

/*
 * The key_idx object and related functions can be used to generate unique keys
 * from an index. The key and index would refer to an object defined by the user.
 * A local endpoint can exchange this key with a remote endpoint in the first message.
 * The remote endpoint would then use this key in subsequent messages to reference
 * the correct object at the local endpoint.
 */
struct ofi_key_idx {
	uint64_t seq_no;
	/* The uniqueness of the generated key would depend on how many bits are
	 * used for the index */
	uint8_t idx_bits;
};

static inline void ofi_key_idx_init(struct ofi_key_idx *key_idx, uint8_t idx_bits)
{
	key_idx->seq_no = 0;
	key_idx->idx_bits = idx_bits;
}

static inline uint64_t ofi_idx2key(struct ofi_key_idx *key_idx, uint64_t idx)
{
	return ((++(key_idx->seq_no)) << key_idx->idx_bits) | idx;
}

static inline uint64_t ofi_key2idx(struct ofi_key_idx *key_idx, uint64_t key)
{
	return key & ((1ULL << key_idx->idx_bits) - 1);
}


#ifdef __cplusplus
}
#endif

#endif /* _OFI_H_ */
