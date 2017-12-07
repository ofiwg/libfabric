/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#ifndef _FI_H_
#define _FI_H_

#include "config.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <netinet/in.h>
#include <ifaddrs.h>

#include <fi_abi.h>
#include <fi_file.h>
#include <fi_lock.h>
#include <fi_atom.h>
#include <fi_mem.h>
#include <rdma/providers/fi_prov.h>
#include <rdma/providers/fi_log.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>

#include <fi_osd.h>


#ifdef __cplusplus
extern "C" {
#endif

#define OFI_CORE_PROV_ONLY	(1ULL << 59)

/*
 * OS X doesn't have __BYTE_ORDER, Linux usually has BYTE_ORDER but not under
 * all features.h flags
 */
#if !defined(BYTE_ORDER)
#  if defined(__BYTE_ORDER) && \
      defined(__LITTLE_ENDIAN) && \
      defined(__BIG_ENDIAN)
#    define BYTE_ORDER __BYTE_ORDER
#    define LITTLE_ENDIAN __LITTLE_ENDIAN
#    define BIG_ENDIAN __BIG_ENDIAN
#  else
#    error "cannot determine endianness!"
#  endif
#endif

#if BYTE_ORDER == LITTLE_ENDIAN
#ifndef htonll
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
#endif
#ifndef ntohll
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#endif
#else
#ifndef htonll
static inline uint64_t htonll(uint64_t x) { return x; }
#endif
#ifndef ntohll
static inline uint64_t ntohll(uint64_t x) { return x; }
#endif
#endif

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


uint64_t fi_tag_bits(uint64_t mem_tag_format);
uint64_t fi_tag_format(uint64_t tag_bits);
int fi_size_bits(uint64_t num);

int ofi_send_allowed(uint64_t caps);
int ofi_recv_allowed(uint64_t caps);
int ofi_rma_initiate_allowed(uint64_t caps);
int ofi_rma_target_allowed(uint64_t caps);
int ofi_ep_bind_valid(const struct fi_provider *prov, struct fid *bfid,
		      uint64_t flags);

uint64_t fi_gettime_ms(void);
uint64_t fi_gettime_us(void);

/*
 * Address utility functions
 */

#ifndef AF_IB
#define AF_IB 27
#endif

#define ofi_sin_addr(addr) (((struct sockaddr_in *)(addr))->sin_addr)
#define ofi_sin_port(addr) (((struct sockaddr_in *)(addr))->sin_port)

#define ofi_sin6_addr(addr) (((struct sockaddr_in6 *)(addr))->sin6_addr)
#define ofi_sin6_port(addr) (((struct sockaddr_in6 *)(addr))->sin6_port)

#define OFI_ADDRSTRLEN (INET6_ADDRSTRLEN + 50)

#define OFI_ENUM_VAL(X) X
#define OFI_STR(X) #X
#define OFI_STR_INT(X) OFI_STR(X)

static inline size_t ofi_sizeofaddr(const struct sockaddr *address)
{
	return (address->sa_family == AF_INET ?
		sizeof(struct sockaddr_in) :
		sizeof(struct sockaddr_in6));
}

static inline int ofi_equals_ipaddr(const struct sockaddr_in *addr1,
				    const struct sockaddr_in *addr2)
{
        return (addr1->sin_addr.s_addr == addr2->sin_addr.s_addr);
}

static inline int ofi_equals_sockaddr(const struct sockaddr_in *addr1,
				      const struct sockaddr_in *addr2)
{
        return (ofi_equals_ipaddr(addr1, addr2) &&
        	(addr1->sin_port == addr2->sin_port));
}

static inline int ofi_translate_addr_format(int family)
{
	switch (family) {
	case AF_INET:
		return FI_SOCKADDR_IN;
	case AF_INET6:
		return FI_SOCKADDR_IN6;
	case AF_IB:
		return FI_SOCKADDR_IB;
	default:
		return FI_FORMAT_UNSPEC;
	}
}

const char *ofi_straddr(char *buf, size_t *len,
			uint32_t addr_format, const void *addr);

/* Returns allocated address to caller.  Caller must free.  */
int ofi_str_toaddr(const char *str, uint32_t *addr_format,
		   void **addr, size_t *len);

int ofi_addr_cmp(const struct fi_provider *prov, const struct sockaddr *sa1,
		const struct sockaddr *sa2);
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

int ofi_getifaddrs(struct ifaddrs **ifap);

#ifdef __cplusplus
}
#endif

#endif /* _FI_H_ */
