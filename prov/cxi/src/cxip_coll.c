/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2020-2022 Cray Inc. All rights reserved.
 * Copyright (c) 2021-2023 Hewlett Packard Enterprise Development LP
 * Support for accelerated collective reductions.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <endian.h>
#include <sys/time.h>
#include <sys/types.h>
#include <math.h>

#include <ofi_list.h>
#include <ofi.h>
#include <fenv.h>

#include "cxip.h"

#ifndef _MM_GET_FLUSH_ZERO_MODE
#define _MM_GET_FLUSH_ZERO_MODE() ({0;})
#endif

/* see cxit_trace_enable() in each test framework */
#define	TRACE		CXIP_TRACE
#define	TRACE_JOIN	CXIP_TRACE

// TODO regularize usage of these
#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

/* must all be 0 in production code */
#define __chk_pkts	1
#define __trc_pkts	0
#define __trc_data	0

#define	MAGIC		0x677d

/****************************************************************************
 * Reduction packet for hardware accelerated collectives:
 *
 *  +----------------------------------------------------------+
 *  | BYTES | Mnemonic    | Definition                         |
 *  +----------------------------------------------------------+
 *  | 48:17 | RED_PAYLOAD | Reduction payload, always 32 bytes |
 *  | 16:5  | RED_HDR     | Reduction Header (below)           |
 *  | 4:0   | RED_PADDING | Padding                            |
 *  +----------------------------------------------------------+
 *
 *  Reduction header format:
 *  --------------------------------------------------------
 *  | Field          | Description              | Bit | Size (bits)
 *  --------------------------------------------------------
 *  | rt_seqno       | Sequence number          |  0  | 10 |
 *  | rt_arm         | Multicast arm command    | 10  |  1 |
 *  | rt_op          | Reduction operation      | 11  |  6 |
 *  | rt_count       | Number of contributions  | 17  | 20 |
 *  | rt_resno       | Result number            | 37  | 10 |
 *  | rt_rc          | result code              | 47  |  4 |
 *  | rt_repsum_m    | Reproducible sum M value | 51  |  8 |
 *  | rt_repsum_ovfl | Reproducible sum M ovfl  | 59  |  2 |
 *  | rt_pad         | Pad to 64 bits           | 61  |  3 |
 *  | rt_cookie      | Cookie value             | 64  | 32 |
 *  --------------------------------------------------------
 *
 * Note that this header is a 12-byte object, and "network-defined order" means
 * big-endian for the entire 12-byte object. Thus, bytes must be swapped so
 * that the MSByte of rt_cookie appears at byte 0, and the LS 8 bits of
 * rt_seqno appear in byte 11.
 *
 * The cookie is ignored by reduction hardware, and is used as follows:
 *
 * mcast_id is the 13-bit multicast hardware address. This is not necessary
 * given one PTE per multicast address: all request structures used for posting
 * receive buffers will receive events from only that multicast. If underlying
 * drivers are changed to allow a single PTE to be mapped to multiple multicast
 * addresses, the mcast_id field can be used to disambiguate packets intended
 * for different collective trees. It's also useful when debugging.
 *
 * red_id is used to disambiguate packets delivered for different concurrent
 * reductions.
 *
 * magic is a magic number used to positively identify this packet as a
 * reduction packet. The basic send/receive code could be used for other kinds
 * of restricted IDC packets. At present, all such packets are discarded.
 *
 * retry is a control bit that can be invoked by the hw root node to initiate a
 * retransmission of the data from the leaves, if packets are lost.
 */
struct cxip_coll_cookie {
	uint32_t mcast_id:13;
	uint32_t red_id:3;
	uint32_t magic: 15;
	uint32_t retry: 1;
} __attribute__((__packed__));           /* size  4b */

/* Packed header bits and cookie from above */
struct cxip_coll_hdr {
        uint64_t seqno:10;
        uint64_t arm:1;
        uint64_t op:6;
        uint64_t redcnt:20;
        uint64_t resno:10;
        uint64_t red_rc:4;
        uint64_t repsum_m:8;
        uint64_t repsum_ovflid:2;
        uint64_t pad:3;
        struct cxip_coll_cookie cookie;
} __attribute__((__packed__));		/* size 12b */

/* The following structure is 49 bytes in size, and all of the fields align
 * properly for network transmission.
 */
struct red_pkt {
	uint8_t pad[5];			/* size  5b offset  0b */
	struct cxip_coll_hdr hdr;	/* size 12b offset  5b */
	uint8_t data[32];		/* size 32b offset 17b */
} __attribute__((__packed__));		/* size 49b */

/* Swap byte order in an object of any size. Works for even or odd counts */
static inline
void _swapbyteorder(void *ptr, int count)
{
	uint8_t *p1 = (uint8_t *)ptr;
	uint8_t *p2 = p1 + count - 1;
	uint8_t swp;
	while (p1 < p2) {
		swp = *p1;
		*p1 = *p2;
		*p2 = swp;
		p1++;
		p2--;
	}
}

/**
 * Reformat the packet to accommodate network-ordering (big-endian) Rosetta
 * expectations, versus little-endian Intel processing.
 *
 * Note in particular that the header bytes are treated as a single 12-byte
 * object, rather than an 8-byte followed by a 4-byte, i.e. the last byte of the
 * cookie is the first byte of the data processed by Rosetta. Note also that
 * there is a 5-byte pad at the beginning of the packet, not included in the
 * byte-swapping.
 *
 * This is done in-place for convenience. For reductions, it is copied to a
 * properly-aligned data structure for mathematical operations.
 */
static inline
void _swappkt(struct red_pkt *pkt)
{
#if (BYTE_ORDER == LITTLE_ENDIAN)
	uint64_t *data = (uint64_t *)pkt->data;
	int i;

	_swapbyteorder(&pkt->hdr, sizeof(pkt->hdr));
	for (i = 0; i < 4; i++)
		_swapbyteorder(&data[i], 8);
#else
#error "Unsupported processor byte ordering"
#endif
}

/**
 * Verificaton of the packet structure, normally disabled. Sizes and offsets
 * cannot be checked at compile time. If the structure is wrong, this will
 * call abort().
 */
#define FLDOFFSET(base, fld)	((uint8_t *)&base.fld - (uint8_t *)&base)
__attribute__((unused)) static inline
void check_red_pkt(void)
{
#if __chk_pkts
	static int checked = 0;
	struct red_pkt pkt;
	uint64_t len, exp;
	uint8_t *ptr, offset;
	int i, err = 0;


	if (checked)
		return;
	checked = 1;

	len = sizeof(pkt);
	exp = 49;
	if (len != exp) {
		TRACE("sizeof(pkt) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = sizeof(pkt.pad);
	exp = 5;
	if (len != exp) {
		TRACE("sizeof(pkt.pad) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = sizeof(pkt.hdr);
	exp = 12;
	if (len != exp) {
		TRACE("sizeof(pkt.hdr) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = sizeof(pkt.data);
	exp = 32;
	if (len != exp) {
		TRACE("sizeof(pkt.data) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = FLDOFFSET(pkt, hdr);
	exp = 5;
	if (len != exp) {
		TRACE("offset(pkt.hdr) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = FLDOFFSET(pkt, data);
	exp = 17;
	if (len != exp) {
		TRACE("offset(pkt.data) = %ld, exp %ld\n", len, exp);
		err++;
	}

	/* Arbitrary value between 1,15 inclusive, ensure non-zero fill */
	offset = 13;

	/* Fill, swap, and confirm integrity of all 49 bytes */
	ptr = (uint8_t *)&pkt;
	for (i = 0; i < sizeof(pkt); i++)
		ptr[i] = i + offset;
	_swappkt(&pkt);
	_swappkt(&pkt);
	for (i = 0; i < sizeof(pkt); i++)
		if (ptr[i] != i + offset) {
			TRACE("pkt[%d] = %d, exp %d\n", i, ptr[i], i + offset);
			err++;
		}

	if (err) {
		TRACE("*** INVALID STRUCTURE see above ***\n");
		abort();
	}
#endif
}

__attribute__((unused)) static inline
void _dump_red_pkt(struct red_pkt *pkt, char *dir)
{
#if __trc_pkts
	const uint64_t *data = (const uint64_t *)pkt->data;
	int i;

	TRACE("---------------\n");
	TRACE("Reduction packet (%s):\n", dir);
	TRACE("  seqno        = %d\n", pkt->hdr.seqno);
	TRACE("  arm          = %d\n", pkt->hdr.arm);
	TRACE("  op           = %d\n", pkt->hdr.op);
	TRACE("  redcnt       = %d\n", pkt->hdr.redcnt);
	TRACE("  resno        = %d\n", pkt->hdr.resno);
	TRACE("  red_rc       = %d\n", pkt->hdr.red_rc);
	TRACE("  repsum_m     = %d\n", pkt->hdr.repsum_m);
	TRACE("  repsum_ovflid= %d\n", pkt->hdr.repsum_ovflid);
	TRACE("  cookie --\n");
	TRACE("   .mcast_id   = %08x\n", pkt->hdr.cookie.mcast_id);
	TRACE("   .red_id     = %08x\n", pkt->hdr.cookie.red_id);
	TRACE("   .magic      = %08x\n", pkt->hdr.cookie.magic);
	TRACE("   .retry      = %08x\n", pkt->hdr.cookie.retry);
	for (i = 0; i < 4; i++)
		TRACE("  ival[%d]     = %016lx\n", i, data[i]);
	TRACE("---------------\n");
#endif
}

/****************************************************************************
 * Reduction operators for accelerated collectives.
 *
 * The array lookup is faster than a switch. Non-static initialization makes
 * this adaptive to changes in header files (e.g. new opcodes in FI).
 */
#define COLL_OPCODE_BARRIER		0x00
#define COLL_OPCODE_BIT_AND		0x01
#define COLL_OPCODE_BIT_OR		0x02
#define COLL_OPCODE_BIT_XOR		0x03
#define COLL_OPCODE_INT_MIN		0x10
#define COLL_OPCODE_INT_MAX		0x11
#define COLL_OPCODE_INT_MINMAXLOC	0x12
#define COLL_OPCODE_INT_SUM		0x14
#define COLL_OPCODE_FLT_MIN		0x20
#define COLL_OPCODE_FLT_MAX		0x21
#define COLL_OPCODE_FLT_MINMAXLOC	0x22
#define COLL_OPCODE_FLT_MINNUM		0x24
#define COLL_OPCODE_FLT_MAXNUM		0x25
#define COLL_OPCODE_FLT_MINMAXNUMLOC	0x26
#define COLL_OPCODE_FLT_SUM_NOFTZ_RND0	0x28
#define COLL_OPCODE_FLT_SUM_NOFTZ_RND1	0x29
#define COLL_OPCODE_FLT_SUM_NOFTZ_RND2	0x2a
#define COLL_OPCODE_FLT_SUM_NOFTZ_RND3	0x2b
#define COLL_OPCODE_FLT_SUM_FTZ_RND0	0x2c
#define COLL_OPCODE_FLT_SUM_FTZ_RND1	0x2d
#define COLL_OPCODE_FLT_SUM_FTZ_RND2	0x2e
#define COLL_OPCODE_FLT_SUM_FTZ_RND3	0x2f
#define COLL_OPCODE_FLT_REPSUM		0x30
#define COLL_OPCODE_MAX			0x31

/* Convert exported op values to Rosetta opcodes */
static cxip_coll_op_t _int8_16_32_op_to_opcode[CXI_FI_OP_LAST];
static cxip_coll_op_t _int64_op_to_opcode[CXI_FI_OP_LAST];
static cxip_coll_op_t _flt_op_to_opcode[CXI_FI_OP_LAST];
static enum c_return_code _cxip_rc_to_cxi_rc[16];
static enum cxip_coll_redtype _cxi_op_to_redtype[COLL_OPCODE_MAX];

/* One-time dynamic initialization of FI to CXI opcode.
 */
void cxip_coll_populate_opcodes(void)
{
	int rnd, ftz, i;

	if ((int)CXI_FI_MINMAXLOC < (int)FI_ATOMIC_OP_LAST) {
		CXIP_FATAL("Invalid CXI_FMINMAXLOC value\n");
	}
	for (i = 0; i < CXI_FI_OP_LAST; i++) {
		_int8_16_32_op_to_opcode[i] = -FI_EOPNOTSUPP;
		_int64_op_to_opcode[i] = -FI_EOPNOTSUPP;
		_flt_op_to_opcode[i] = -FI_EOPNOTSUPP;
		_cxi_op_to_redtype[i] = REDTYPE_BYT;
	}
	/* operations supported by 32, 16, and 8 bit integer operands */
	/* NOTE: executed as packed 64-bit quantities */
	_int8_16_32_op_to_opcode[FI_BOR] = COLL_OPCODE_BIT_OR;
	_int8_16_32_op_to_opcode[FI_BAND] = COLL_OPCODE_BIT_AND;
	_int8_16_32_op_to_opcode[FI_BXOR] = COLL_OPCODE_BIT_XOR;
	_int8_16_32_op_to_opcode[CXI_FI_BARRIER] = COLL_OPCODE_BARRIER;

	/* operations supported by 64 bit integer operands */
	_int64_op_to_opcode[FI_MIN] = COLL_OPCODE_INT_MIN;
	_int64_op_to_opcode[FI_MAX] = COLL_OPCODE_INT_MAX;
	_int64_op_to_opcode[FI_SUM] = COLL_OPCODE_INT_SUM;
	_int64_op_to_opcode[FI_BOR] = COLL_OPCODE_BIT_OR;
	_int64_op_to_opcode[FI_BAND] = COLL_OPCODE_BIT_AND;
	_int64_op_to_opcode[FI_BXOR] = COLL_OPCODE_BIT_XOR;
	_int64_op_to_opcode[CXI_FI_MINMAXLOC] = COLL_OPCODE_INT_MINMAXLOC;
	_int64_op_to_opcode[CXI_FI_BARRIER] = COLL_OPCODE_BARRIER;

	/* operations supported by 64 bit double operands */
	_flt_op_to_opcode[FI_MIN] = COLL_OPCODE_FLT_MIN;
	_flt_op_to_opcode[FI_MAX] = COLL_OPCODE_FLT_MAX;
	_flt_op_to_opcode[CXI_FI_MINMAXLOC] = COLL_OPCODE_FLT_MINMAXLOC;
	_flt_op_to_opcode[CXI_FI_MINNUM] = COLL_OPCODE_FLT_MINNUM;
	_flt_op_to_opcode[CXI_FI_MAXNUM] = COLL_OPCODE_FLT_MAXNUM;
	_flt_op_to_opcode[CXI_FI_MINMAXNUMLOC] = COLL_OPCODE_FLT_MINMAXNUMLOC;
	_flt_op_to_opcode[CXI_FI_REPSUM] = COLL_OPCODE_FLT_REPSUM;
	_flt_op_to_opcode[CXI_FI_BARRIER] = COLL_OPCODE_BARRIER;

	/* SUM operations supported by 64 bit double operands */
	rnd = fegetround();
	ftz = _MM_GET_FLUSH_ZERO_MODE();
	switch (rnd) {
	case FE_UPWARD:
		_flt_op_to_opcode[FI_SUM] = (ftz) ?
			COLL_OPCODE_FLT_SUM_FTZ_RND1 :
			COLL_OPCODE_FLT_SUM_NOFTZ_RND1;
		break;
	case FE_DOWNWARD:
		_flt_op_to_opcode[FI_SUM] = (ftz) ?
			COLL_OPCODE_FLT_SUM_FTZ_RND2 :
			COLL_OPCODE_FLT_SUM_NOFTZ_RND2;
		break;
	case FE_TOWARDZERO:
		_flt_op_to_opcode[FI_SUM] = (ftz) ?
			COLL_OPCODE_FLT_SUM_FTZ_RND3 :
			COLL_OPCODE_FLT_SUM_NOFTZ_RND3;
		break;
	case FE_TONEAREST:
		_flt_op_to_opcode[FI_SUM] = (ftz) ?
			COLL_OPCODE_FLT_SUM_FTZ_RND0 :
			COLL_OPCODE_FLT_SUM_NOFTZ_RND0;
		break;
	default:
		CXIP_FATAL("Invalid fegetround() return = %d\n", rnd);
	}

	/* cxi_opcode to redtype translation */
	_cxi_op_to_redtype[COLL_OPCODE_BIT_OR] = REDTYPE_INT;
	_cxi_op_to_redtype[COLL_OPCODE_BIT_AND] = REDTYPE_INT;
	_cxi_op_to_redtype[COLL_OPCODE_BIT_XOR] = REDTYPE_INT;
	_cxi_op_to_redtype[COLL_OPCODE_INT_MIN] = REDTYPE_INT;
	_cxi_op_to_redtype[COLL_OPCODE_INT_MAX] = REDTYPE_INT;
	_cxi_op_to_redtype[COLL_OPCODE_INT_SUM] = REDTYPE_INT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_MIN] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_MAX] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_MINNUM] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_MAXNUM] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_NOFTZ_RND0] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_NOFTZ_RND1] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_NOFTZ_RND2] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_NOFTZ_RND3] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_FTZ_RND0] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_FTZ_RND1] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_FTZ_RND2] = REDTYPE_FLT;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_SUM_FTZ_RND3] = REDTYPE_FLT;

	_cxi_op_to_redtype[COLL_OPCODE_INT_MINMAXLOC] = REDTYPE_IMINMAX;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_MINMAXLOC] = REDTYPE_FMINMAX;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_MINMAXNUMLOC] = REDTYPE_FMINMAX;
	_cxi_op_to_redtype[COLL_OPCODE_FLT_REPSUM] = REDTYPE_REPSUM;

	for (i = 0; i < 16; i++)
		_cxip_rc_to_cxi_rc[i] = C_RC_AMO_ALIGN_ERROR;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_SUCCESS] = C_RC_OK;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_FLT_INEXACT] = C_RC_AMO_FP_INEXACT;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_FLT_OVERFLOW] = C_RC_AMO_FP_OVERFLOW;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_FLT_INVALID] = C_RC_AMO_FP_INVALID;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_REP_INEXACT] = C_RC_AMO_FP_INEXACT;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_INT_OVERFLOW] = C_RC_AMO_FP_OVERFLOW;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_CONTR_OVERFLOW] = C_RC_AMO_LENGTH_ERROR;
	_cxip_rc_to_cxi_rc[CXIP_COLL_RC_OP_MISMATCH] = C_RC_AMO_INVAL_OP_ERROR;
}

/* Convert CXI opcode to reduction data type */
static inline
enum cxip_coll_redtype _opcode_to_redtype(cxip_coll_op_t cxi_opcode)
{
	return _cxi_op_to_redtype[cxi_opcode];
}

/* Convert FI opcode to CXI opcode, depending on FI data type */
int cxip_fi2cxi_opcode(enum fi_op op, enum fi_datatype datatype)
{
	int opcode;

	switch (datatype) {
	case FI_UINT8:
	case FI_UINT16:
	case FI_UINT32:
		opcode = _int8_16_32_op_to_opcode[op];
		break;
	case FI_UINT64:
		opcode = _int64_op_to_opcode[op];
		break;
	case FI_DOUBLE:
		opcode = _flt_op_to_opcode[op];
		break;
	default:
		opcode = -FI_EOPNOTSUPP;
		break;
	}
	return opcode;
}

/* Determine FI datatype size */
static inline
int _get_cxi_data_bytcnt(cxip_coll_op_t cxi_opcode,
			 enum fi_datatype datatype, size_t count)
{
	int size;

	switch (datatype) {
	case FI_UINT8:
		size = sizeof(uint8_t);
		break;
	case FI_UINT16:
		size = sizeof(uint16_t);
		break;
	case FI_UINT32:
		size = sizeof(uint32_t);
		break;
	case FI_UINT64:
		size = sizeof(uint64_t);
		break;
	case FI_DOUBLE:
		size = sizeof(double);
		break;
	default:
		return -FI_EOPNOTSUPP;
	}
	size *= count;
	if (size > CXIP_COLL_MAX_TX_SIZE)
		return -FI_EINVAL;
	return size;
}

/****************************************************************************
 * SEND operation (restricted Put to a remote PTE)
 */

/* Forward references */
static void _progress_coll(struct cxip_coll_reduction *reduction,
			   struct red_pkt *pkt);
static ssize_t _coll_append_buffer(struct cxip_coll_pte *coll_pte,
				   struct cxip_coll_buf *buf);

/* Generate a dfa and index extension for a reduction */
static int _gen_tx_dfa(struct cxip_coll_reduction *reduction,
		       int av_set_idx, union c_fab_addr *dfa,
		       uint8_t *index_ext, bool *is_mcast)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_av_set *av_set;
	struct cxip_addr dest_caddr;
	fi_addr_t dest_addr;
	int pid_bits;
	int idx_ext;
	int ret;

	ep_obj = reduction->mc_obj->ep_obj;
	av_set = reduction->mc_obj->av_set;

	/* Send address */
	switch (av_set->comm_key.keytype) {
	case COMM_KEY_MULTICAST:
		/* - destination == multicast ID
		 * - idx_ext == 0
		 * - dfa == multicast destination
		 * - index_ext == 0
		 */
		if (is_netsim(ep_obj)) {
			CXIP_WARN("NETSIM does not support mcast\n");
			return -FI_EINVAL;
		}
		idx_ext = 0;
		cxi_build_mcast_dfa(av_set->comm_key.mcast.mcast_addr,
				    reduction->red_id, idx_ext,
				    dfa, index_ext);
		*is_mcast = true;
		break;
	case COMM_KEY_UNICAST:
		/* - destination == remote node in av_set
		 * - idx_ext == CXIP_PTL_IDX_COLL
		 * - dfa = remote nic
		 * - index_ext == CXIP_PTL_IDX_COLL
		 */
		if (av_set_idx >= av_set->fi_addr_cnt) {
			CXIP_WARN("av_set_idx out-of-range\n");
			return -FI_EINVAL;
		}
		dest_addr = av_set->fi_addr_ary[av_set_idx];
		ret = _cxip_av_lookup(ep_obj->av, dest_addr, &dest_caddr);
		if (ret != FI_SUCCESS)
			return ret;
		pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
		cxi_build_dfa(dest_caddr.nic, dest_caddr.pid, pid_bits,
			      CXIP_PTL_IDX_COLL, dfa, index_ext);
		*is_mcast = false;
		break;
	case COMM_KEY_RANK:
		/* - destination == source NIC
		 * - idx_ext == extended PID
		 * - dfa == source NIC
		 * - index_ext == idx_ext offset beyond RXCs (5-bit range)
		 */
		if (av_set_idx >= av_set->fi_addr_cnt) {
			CXIP_WARN("av_set_idx out-of-range\n");
			return -FI_EINVAL;
		}
		dest_caddr = ep_obj->src_addr;
		pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
		idx_ext = CXIP_PTL_IDX_COLL + av_set_idx;
		cxi_build_dfa(dest_caddr.nic, dest_caddr.pid, pid_bits,
			      idx_ext, dfa, index_ext);
		*is_mcast = false;
		break;
	default:
		CXIP_WARN("unexpected comm_key type: %d\n",
			  av_set->comm_key.keytype);
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

/**
 * Issue a restricted Put to the destination address.
 * If md is NULL, this performs an IDC Put, otherwise it issues a DMA Put.
 *
 * Exported for unit testing.
 *
 * This will return -FI_EAGAIN on transient errors.
 */
int cxip_coll_send(struct cxip_coll_reduction *reduction,
		   int av_set_idx, const void *buffer, size_t buflen,
		   struct cxi_md *md)
{
	union c_cmdu cmd = {};
	struct cxip_coll_mc *mc_obj;
	struct cxip_ep_obj *ep_obj;
	struct cxip_cmdq *cmdq;
	union c_fab_addr dfa;
	uint8_t index_ext;
	bool is_mcast;
	int ret;

	if (!buffer) {
		CXIP_INFO("no buffer\n");
		return -FI_EINVAL;
	}

	mc_obj = reduction->mc_obj;
	ep_obj = mc_obj->ep_obj;
	cmdq = ep_obj->coll.tx_cmdq;

	ret = _gen_tx_dfa(reduction, av_set_idx, &dfa, &index_ext, &is_mcast);
	if (ret)
		return ret;

	if (cxip_evtq_saturated(ep_obj->coll.tx_evtq)) {
		CXIP_DBG("TX HW EQ saturated\n");
		return -FI_EAGAIN;
	}

#if ENABLE_DEBUG
	if (reduction->drop_send) {
		reduction->drop_send = false;
		goto drop_pkt;
	}
#endif

	if (md) {
		cmd.full_dma.command.opcode = C_CMD_PUT;
		cmd.full_dma.event_send_disable = 1;
		cmd.full_dma.event_success_disable = 1;
		cmd.full_dma.restricted = 1;
		cmd.full_dma.reduction = is_mcast;
		cmd.full_dma.index_ext = index_ext;
		cmd.full_dma.eq = cxip_evtq_eqn(ep_obj->coll.tx_evtq);
		cmd.full_dma.dfa = dfa;
		cmd.full_dma.lac = md->lac;
		cmd.full_dma.local_addr = CXI_VA_TO_IOVA(md, buffer);
		cmd.full_dma.request_len = buflen;

		/* this uses cached values, returns -FI_EAGAIN if queue full */
		ret = cxip_txq_cp_set(cmdq, ep_obj->auth_key.vni,
				      mc_obj->tc, mc_obj->tc_type);
		if (ret)
			goto err;

		ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &cmd.full_dma);
	} else {
		cmd.c_state.event_send_disable = 1;
		cmd.c_state.event_success_disable = 1;
		cmd.c_state.restricted = 1;
		cmd.c_state.reduction = is_mcast;
		cmd.c_state.index_ext = index_ext;
		cmd.c_state.eq = cxip_evtq_eqn(ep_obj->coll.tx_evtq);
		cmd.c_state.initiator = CXI_MATCH_ID(
			ep_obj->domain->iface->dev->info.pid_bits,
			ep_obj->src_addr.pid, ep_obj->src_addr.nic);

		/* this uses cached values, returns -FI_EAGAIN if queue full */
		ret = cxip_txq_cp_set(cmdq, ep_obj->auth_key.vni,
				      mc_obj->tc, mc_obj->tc_type);
		if (ret)
			goto err;

		/* returns -FI_EAGAIN on failure */
		ret = cxip_cmdq_emit_c_state(cmdq, &cmd.c_state);
		if (ret) {
			ret = -FI_EAGAIN;
			goto err;
		}

		memset(&cmd.idc_put, 0, sizeof(cmd.idc_put));
		cmd.idc_put.idc_header.dfa = dfa;
		ret = cxi_cq_emit_idc_put(cmdq->dev_cmdq, &cmd.idc_put,
					  buffer, buflen);
		if (ret) {
			ret = -FI_EAGAIN;
			goto err;
		}
	}

	if (ret) {
		/* Return error according to Domain Resource Management */
		ret = -FI_EAGAIN;
		goto err;
	}

	cxi_cq_ring(cmdq->dev_cmdq);

#if ENABLE_DEBUG
drop_pkt:
#endif
	ret = FI_SUCCESS;
	ofi_atomic_inc32(&reduction->mc_obj->send_cnt);

err:
	return ret;
}

/****************************************************************************
 * RECV operation (of restricted Put to a local PTE)
 */

/* Report success/error results of an RX event through RX CQ / counters, and
 * roll over the buffers if appropriate.
 *
 * NOTE: req may be invalid after this call.
 *
 * Caller must hold ep_obj->lock.
 */
static void _coll_rx_req_report(struct cxip_req *req)
{
	size_t overflow;
	int err, ret;

	req->flags &= (FI_RECV | FI_COMPLETION);

	/* Interpret results */
	overflow = req->coll.hw_req_len - req->data_len;
	if (req->coll.cxi_rc == C_RC_OK && req->coll.isred && !overflow) {
		/* receive success */
		if (req->flags & FI_COMPLETION) {
			/* failure means progression is hung */
			ret = cxip_cq_req_complete(req);
			if (ret)
				CXIP_FATAL(
				    "cxip_cq_req_complete failed: %d\n", ret);
		}

		if (req->coll.coll_pte->ep_obj->coll.rx_cntr) {
			/* failure means counts cannot be trusted */
			ret = cxip_cntr_mod(
				req->coll.coll_pte->ep_obj->coll.rx_cntr, 1,
				false, false);
			if (ret)
				CXIP_WARN(
					"Failed success cxip_cntr_mod: %d\n",
					ret);
		}
	} else {
		/* failure */
		if (req->coll.cxi_rc != C_RC_OK) {
			/* real network error of some sort */
			err = FI_EIO;
			CXIP_WARN("Request error: %p (err: %d, %s)\n",
				  req, err, cxi_rc_to_str(err));
		} else if (overflow) {
			/* can only happen on very large packet (> 64 bytes) */
			err = FI_EMSGSIZE;
			CXIP_WARN("Request truncated: %p (err: %d, %s)\n",
				  req, err, cxi_rc_to_str(err));
		} else {
			/* non-reduction packet */
			err = FI_ENOMSG;
			CXIP_INFO("Not reduction pkt: %p (err: %d, %s)\n",
				  req, err, cxi_rc_to_str(err));
		}

		/* failure means progression is hung */
		ret = cxip_cq_req_error(req, overflow, err,
					req->coll.cxi_rc,
					NULL, 0);
		if (ret)
			CXIP_FATAL("cxip_cq_req_error: %d\n", ret);

		if (req->coll.coll_pte->ep_obj->coll.rx_cntr) {
			/* failure means counts cannot be trusted */
			ret = cxip_cntr_mod(
				req->coll.coll_pte->ep_obj->coll.rx_cntr, 1,
				false, true);
			if (ret)
				CXIP_WARN("cxip_cntr_mod: %d\n", ret);
		}
	}

	/* manage buffer rollover */
	if (req->coll.mrecv_space <
	    req->coll.coll_pte->ep_obj->coll.min_multi_recv) {
		struct cxip_coll_pte *coll_pte = req->coll.coll_pte;
		struct cxip_coll_buf *buf = req->coll.coll_buf;

		/* Will be re-incremented when LINK is received */
		ofi_atomic_dec32(&coll_pte->buf_cnt);
		ofi_atomic_inc32(&coll_pte->buf_swap_cnt);

		/* Re-use this buffer in the hardware */
		ret = _coll_append_buffer(coll_pte, buf);
		if (ret != FI_SUCCESS)
			CXIP_WARN("Re-link buffer failed: %d\n", ret);

		/* Hardware has silently unlinked this */
		cxip_evtq_req_free(req);
	}
}

/* Evaluate PUT receive request to see if this is a reduction packet */
static void _coll_rx_progress(struct cxip_req *req,
			      const union c_event *event)
{
	struct cxip_coll_mc *mc_obj;
	struct cxip_coll_reduction *reduction;
	struct red_pkt *pkt;

	mc_obj = req->coll.coll_pte->mc_obj;
	ofi_atomic_inc32(&mc_obj->recv_cnt);

	/* If not the right size, don't swap bytes */
	if (req->data_len != sizeof(struct red_pkt)) {
		CXIP_INFO("Bad coll packet size: %ld\n", req->data_len);
		return;
	}

	/* If swap doesn't look like reduction packet, swap back */
	pkt = (struct red_pkt *)req->buf;
	_swappkt(pkt);
	if (pkt->hdr.cookie.magic != MAGIC)
	{
		CXIP_INFO("Bad coll MAGIC: %x\n", pkt->hdr.cookie.magic);
		_swappkt(pkt);
		return;
	}

	/* This is a reduction packet */
	req->coll.isred = true;
	reduction = &mc_obj->reduction[pkt->hdr.cookie.red_id];

#if ENABLE_DEBUG
	/* Test case, simulate packet dropped in-flight */
	if (reduction->drop_recv) {
		reduction->drop_recv = false;
		return;
	}
#endif

	/* Progress the reduction */
	_dump_red_pkt(pkt, "recv");
	ofi_atomic_inc32(&mc_obj->pkt_cnt);
	_progress_coll(reduction, pkt);
}

/* Event-handling callback for posted receive buffers */
static int _coll_recv_cb(struct cxip_req *req, const union c_event *event)
{
	req->coll.cxi_rc = cxi_tgt_event_rc(event);
	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* Enabled */
		if (req->coll.cxi_rc != C_RC_OK) {
			CXIP_WARN("LINK error rc: %d\n", req->coll.cxi_rc);
			break;
		}
		CXIP_DBG("LINK event seen\n");
		ofi_atomic_inc32(&req->coll.coll_pte->buf_cnt);
		break;
	case C_EVENT_UNLINK:
		/* Normally disabled, errors only */
		req->coll.cxi_rc = cxi_tgt_event_rc(event);
		if (req->coll.cxi_rc != C_RC_OK) {
			CXIP_WARN("UNLINK error rc: %d\n", req->coll.cxi_rc);
			break;
		}
		CXIP_DBG("UNLINK event seen\n");
		break;
	case C_EVENT_PUT:
		req->coll.isred = false;
		req->coll.cxi_rc = cxi_tgt_event_rc(event);
		if (req->coll.cxi_rc != C_RC_OK) {
			CXIP_WARN("PUT error rc: %d\n", req->coll.cxi_rc);
			break;
		}
		CXIP_DBG("PUT event seen\n");
		req->buf = (uint64_t)(CXI_IOVA_TO_VA(
					req->coll.coll_buf->cxi_md->md,
					event->tgt_long.start));
		req->coll.mrecv_space -= event->tgt_long.mlength;
		req->coll.hw_req_len = event->tgt_long.rlength;
		req->data_len = event->tgt_long.mlength;
		_coll_rx_progress(req, event);
		_coll_rx_req_report(req);
		break;
	default:
		req->coll.cxi_rc = cxi_tgt_event_rc(event);
		CXIP_WARN(CXIP_UNEXPECTED_EVENT,
			  cxi_event_to_str(event),
			  cxi_rc_to_str(req->coll.cxi_rc));
		break;
	}

	return FI_SUCCESS;
}

/* Inject a hardware LE append. Does not generate HW LINK event unless error. */
static int _hw_coll_recv(struct cxip_coll_pte *coll_pte, struct cxip_req *req)
{
	uint32_t le_flags;
	uint64_t recv_iova;
	int ret;

	/* C_LE_MANAGE_LOCAL makes Cassini ignore initiator remote_offset in all
	 * Puts, and causes automatic UNLINK when buffer capacity drops below
	 * CXIP_COLL_MIN_FREE.
	 *
	 * C_LE_EVENT_UNLINK_DISABLE prevents generation of UNLINK events. We
	 * detect UNLINK by counting packets, and presume automatic UNLINK drops
	 * below CXIP_COLL_MIN_FREE.
	 *
	 * C_LE_EVENT_UNLINK_DISABLE prevents UNLINK events from being
	 * generated. Hardware performs UNLINK automatically when buffer
	 * capacity is below CXIP_COLL_MIN_FREE.
	 *
	 * C_LE_OP_PUT indicates this is an input buffer that responses to PUT.
	 *
	 * C_LE_NO_TRUNCATE is not used, because all packets are a fixed size,
	 * and CXIP_COLL_MIN_FREE is sufficient to guarantee space for one new
	 * reduction packet.
	 */
	le_flags = C_LE_EVENT_UNLINK_DISABLE | C_LE_OP_PUT | C_LE_MANAGE_LOCAL;

	recv_iova = CXI_VA_TO_IOVA(req->coll.coll_buf->cxi_md->md,
				   (uint64_t)req->coll.coll_buf->buffer);

	ret = cxip_pte_append(coll_pte->pte,
			      recv_iova,
			      req->coll.coll_buf->bufsiz,
			      req->coll.coll_buf->cxi_md->md->lac,
			      C_PTL_LIST_PRIORITY,
			      req->req_id,
			      0, 0, 0,
			      coll_pte->ep_obj->rxc.min_multi_recv,
			      le_flags, coll_pte->ep_obj->coll.rx_cntr,
			      coll_pte->ep_obj->coll.rx_cmdq,
			      true);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("PTE append inject failed: %d\n", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/* Append a receive buffer to the PTE, with callback to handle receives.
 *
 * Caller must hold ep_obj->lock.
 */
static ssize_t _coll_append_buffer(struct cxip_coll_pte *coll_pte,
				   struct cxip_coll_buf *buf)
{
	struct cxip_req *req;
	int ret;

	if (buf->bufsiz && !buf->buffer) {
		CXIP_INFO("no buffer\n");
		return -FI_EINVAL;
	}

	/* Allocate and populate a new request
	 * Sets:
	 * - req->cq
	 * - req->req_id to request index
	 * - req->req_ctx to passed context (buf)
	 * - req->discard to false
	 * - Inserts into the cq->req_list
	 */
	req = cxip_evtq_req_alloc(coll_pte->ep_obj->coll.rx_evtq, 1, buf);
	if (!req) {
		ret = -FI_ENOMEM;
		goto recv_unmap;
	}

	/* CQ event fields, set according to fi_cq.3
	 *   - set by provider
	 *   - returned to user in completion event
	 * uint64_t context;	// operation context
	 * uint64_t flags;	// operation flags
	 * uint64_t data_len;	// received data length
	 * uint64_t buf;	// receive buf offset
	 * uint64_t data;	// receive REMOTE_CQ_DATA
	 * uint64_t tag;	// receive tag value on matching interface
	 * fi_addr_t addr;	// sender address (if known) ???
	 */

	/* Request parameters */
	req->type = CXIP_REQ_COLL;
	req->flags = (FI_RECV | FI_COMPLETION);
	req->cb = _coll_recv_cb;
	req->triggered = false;
	req->trig_thresh = 0;
	req->trig_cntr = NULL;
	req->context = (uint64_t)buf;
	req->data_len = 0;
	req->buf = (uint64_t)buf->buffer;
	req->data = 0;
	req->tag = 0;
	req->coll.coll_pte = coll_pte;
	req->coll.coll_buf = buf;
	req->coll.mrecv_space = req->coll.coll_buf->bufsiz;

	/* Returns FI_SUCCESS or FI_EAGAIN */
	ret = _hw_coll_recv(coll_pte, req);
	if (ret != FI_SUCCESS)
		goto recv_dequeue;

	return FI_SUCCESS;

recv_dequeue:
	cxip_evtq_req_free(req);

recv_unmap:
	cxip_unmap(buf->cxi_md);
	return ret;
}

/****************************************************************************
 * PTE management functions.
 */

/* PTE state-change callback */
 __attribute__((__unused__))
static void _coll_pte_cb(struct cxip_pte *pte, const union c_event *event)
{
	switch (pte->state) {
	case C_PTLTE_ENABLED:
	case C_PTLTE_DISABLED:
		break;
	default:
		CXIP_FATAL("Unexpected state received: %u\n", pte->state);
	}
}

/* Enable a collective PTE. Wait for completion. */
static inline
int _coll_pte_enable(struct cxip_coll_pte *coll_pte, uint32_t drop_count)
{
	return cxip_pte_set_state_wait(coll_pte->pte,
				       coll_pte->ep_obj->coll.rx_cmdq,
				       coll_pte->ep_obj->coll.rx_evtq,
				       C_PTLTE_ENABLED, drop_count);
}

/* Disable a collective PTE. Wait for completion */
static inline
int _coll_pte_disable(struct cxip_coll_pte *coll_pte)
{
	return cxip_pte_set_state_wait(coll_pte->pte,
				       coll_pte->ep_obj->coll.rx_cmdq,
				       coll_pte->ep_obj->coll.rx_evtq,
				       C_PTLTE_DISABLED, 0);
}

/* Destroy and unmap all buffers used by the collectives PTE.
 *
 * Caller must hold ep_obj->lock.
 */
static void _coll_destroy_buffers(struct cxip_coll_pte *coll_pte)
{
	struct dlist_entry *list = &coll_pte->buf_list;
	struct cxip_coll_buf *buf;

	while (!dlist_empty(list)) {
		dlist_pop_front(list, struct cxip_coll_buf, buf, buf_entry);
		cxip_unmap(buf->cxi_md);
		free(buf);
	}
}

/* Adds 'count' buffers of 'size' bytes to the collecives PTE. This succeeds
 * fully, or it fails and removes all added buffers.
 */
static int _coll_add_buffers(struct cxip_coll_pte *coll_pte, size_t size,
			     size_t count)
{
	struct cxip_coll_buf *buf;
	int ret, i;

	if (count < CXIP_COLL_MIN_RX_BUFS) {
		CXIP_INFO("Buffer count %ld < minimum (%d)\n",
			  count, CXIP_COLL_MIN_RX_BUFS);
		return -FI_EINVAL;
	}

	if (size < CXIP_COLL_MIN_RX_SIZE) {
		CXIP_INFO("Buffer size %ld < minimum (%d)\n",
			  size, CXIP_COLL_MIN_RX_SIZE);
		return -FI_EINVAL;
	}

	CXIP_DBG("Adding %ld buffers of size %ld\n", count, size);
	for (i = 0; i < count; i++) {
		buf = calloc(1, sizeof(*buf) + size);
		if (!buf) {
			ret = -FI_ENOMEM;
			goto out;
		}
		ret = cxip_map(coll_pte->ep_obj->domain, (void *)buf->buffer,
			       size, 0, &buf->cxi_md);
		if (ret)
			goto del_msg;
		buf->bufsiz = size;
		dlist_insert_tail(&buf->buf_entry, &coll_pte->buf_list);

		ret = _coll_append_buffer(coll_pte, buf);
		if (ret) {
			CXIP_WARN("Add buffer %d of %ld: %d\n",
				  i, count, ret);
			goto out;
		}
	}
	/* Block until PTE completes buffer appends */
	do {
		sched_yield();
		cxip_evtq_progress(coll_pte->ep_obj->coll.rx_evtq);
	} while (ofi_atomic_get32(&coll_pte->buf_cnt) < count);

	return FI_SUCCESS;
del_msg:
	free(buf);
out:
	_coll_destroy_buffers(coll_pte);
	return ret;
}

/****************************************************************************
 * Mathematical routines used for collective reductions.
 */

/* Set RC only if new is higher priority than old */
// TODO avoid branch:
// http://geeksforgeeks.org/
//     compute-the-minimum-or-maximum-of-two-integers-without-branching
#define SET_RED_RC(redrc, rc) do {if ((redrc)<(rc)) (redrc)=(rc);} while(0)

static inline
bool cxip_is_snan64(double d)
{
	/* This detection is universal IEEE */
	return isnan(d) && !(_dbl2bits(d) & 0x0008000000000000);
}

/* convert signalling NaN to quiet NaN */
static inline
bool _quiesce_nan(double *d)
{
	if (!cxip_is_snan64(*d))
		return false;
	*d = NAN;
	return true;
}

/**
 * Implement NaN comparison in RSDG 4.5.9.2.3 FLT_MIN and FLT_MAX
 *
 * Compares two doubles, replaces *d1 as appropriate, and indicates swap.
 *
 * If the values are normal doubles, less=true indicates we are looking for the
 * lesser of the two values, while less=false indicates we are looking for the
 * greater of the two values. The appropriate value will be swapped into *d1 if
 * necessary.
 *
 * If any of the values are NaN, this will give preference to the NaN, ignoring
 * less, and if both are NaN, this will give preference to sNaN over qNan.
 *
 * The return value can be used when associating an index with the value.
 *
 * Note that since this quiets any signalling NaNs, we need set the
 * CXIP_COLL_RC_FLT_INVALID error here.
 *
 * - return  0 indicates the values are equivalent, so use the smallest index.
 * - return +1 indicates the values were swapped, so use the second index.
 * - return -1 indicates no swap, so use the first index.
 */
static int swpnan1(double *d1, double d2, bool less, cxip_coll_rc_t *rc)
{
	bool nan1, nan2, snan1, snan2;

	// isnan() does not distinguish sNaN from qNaN
	nan1 = isnan(*d1);
	nan2 = isnan(d2);
	// Neither is NaN, so simple comparison
	if (!nan1 && !nan2) {
		if (*d1 == d2)
			return 0;
		if (less && (*d1 > d2)) {
			*d1 = d2;
			return 1;
		}
		if (!less && (*d1 < d2)) {
			*d1 = d2;
			return 1;
		}
		return -1;
	}

	// ----- FLT_MIN and FLT_MAX rules
	snan1 = _quiesce_nan(d1);
	snan2 = _quiesce_nan(&d2);
	if (snan1 || snan2)
		SET_RED_RC(*rc, CXIP_COLL_RC_FLT_INVALID);

	// Always give preference to any NaN over normal number
	if (!nan1 && nan2) {
		*d1 = d2;
		return 1;
	}
	if (nan1 && !nan2) {
		return -1;
	}
	// Both are NaN

	// Always prefers sNaN over qNaN or number
	if (!snan1 && snan2) {
		*d1 = d2;
		return 1;
	}
	if (snan1 && !snan2) {
		return -1;
	}
	// both snan || neither snan
	return 0;
}

/**
 * Implement NaN comparison in RSDG 4.5.9.2.4 FLT_MINNUM and FLT_MAXNUM
 *
 * Only associative mode is supported. The old IEEE mode is incorrect, and has
 * been deprecated.
 *
 * Compares two doubles, replaces *d1 as appropriate, and indicates swap.
 *
 * If the values are normal doubles, less=true indicates we are looking for the
 * lesser of the two values, while less=false indicates we are looking for the
 * greater of the two values. The appropriate value will be swapped into *d1 if
 * necessary.
 *
 * In general, this will give preference to real values over NaN, which is the
 * opposite of swpnan1() above. This will only return NaN if BOTH values in
 * the comparison are NaN.
 *
 * If either NaN is sNaN, this will set the CXIP_COLL_RC_FLT_INVALID error.
 *
 * The return value can be used when associating an index with the value.
 *
 * Note that since this quiets any signalling NaNs, we need set the
 * CXIP_COLL_RC_FLT_INVALID error.
 *
 * - return  0 indicates the values are equivalent, so use the smallest index.
 * - return +1 indicates the values were swapped, so use the second index.
 * - return -1 indicates no swap, so use the first index.
 */
static int swpnan2(double *d1, double d2, bool less, cxip_coll_rc_t *rc)
{
	bool nan1, nan2, snan1, snan2;

	// isnan() does not distinguish sNaN from qNaN
	nan1 = isnan(*d1);
	nan2 = isnan(d2);
	// Neither is NaN, so simple comparison
	if (!nan1 && !nan2) {
		if (*d1 == d2)
			return 0;
		if (less && (*d1 > d2)) {
			*d1 = d2;
			return 1;
		}
		if (!less && (*d1 < d2)) {
			*d1 = d2;
			return 1;
		}
		return -1;
	}

	// ----- FLT_MINNUM and FLT_MAXNUM rules
	// At least one is NaN, check for sNaN
	snan1 = _quiesce_nan(d1);
	snan2 = _quiesce_nan(&d2);
	if (snan1 || snan2)
		SET_RED_RC(*rc, CXIP_COLL_RC_FLT_INVALID);

	// return qNaN only if both are NaN
	if (nan1 && nan2)
		return 0;

	// Prefers number
	if (nan1) {
		*d1 = d2;
		return 1;
	}
	// Prefers number
	return -1;
}

/* Companion to swpnan1() and swpnan2() to swap associated indices */
static inline
void swpidx(uint64_t *i1, uint64_t i2, int swp)
{
	if (swp >= 0 && (swp > 0 || *i1 > i2))
		*i1 = i2;
}

/* Determine if double precision sum is exact. This shifts the value with the
 * lower exponent toward the MSBit by the amount of the bitwise overlap between
 * the final sum and the value that resulted in that sum. If any non-zero bits
 * remain in that smaller value, they were discarded during the summation, and
 * the result is inexact.
 */
static inline
bool exact(double rslt, double d)
{
	// TODO verify sign and shift
	unsigned long m1, m2;
	int s1, e1, s2, e2;
	int shft, dlte;
	bool ret;

	_decompose_dbl(rslt, &s1, &e1, &m1);
	_decompose_dbl(d, &s2, &e2, &m2);
	dlte = e1 - e2;

	if (dlte < 0) {
		shft = MIN(52 + dlte, 0);
		ret = !(m1 << shft);
	} else {
		shft= MIN(52 - dlte, 0);
		ret = !(m2 << shft);
	}
	return ret;
}

static inline
void _dump_coll_data(const char *tag, const struct cxip_coll_data *coll_data)
{
#if __trc_data
	int i;

	TRACE("=== Coll data: %s\n", tag);
	TRACE("  init    = %d\n", coll_data->initialized);
	TRACE("  red_op  = %d\n", coll_data->red_op);
	TRACE("  rec_rc  = %d\n", coll_data->red_rc);
	TRACE("  red_cnt = %d\n", coll_data->red_cnt);
	TRACE("  red_max = %d\n", coll_data->red_max);
	TRACE("  data:\n");
	for (i = 0; i < 4; i++)
		TRACE(" %016lx", coll_data->intval.ival[i]);
	TRACE("\n");
	TRACE("===================\n");
#endif
}

/* initialize coll_data structure from raw user data */
static void _init_coll_data(struct cxip_coll_data *coll_data, int opcode,
			    const void *user_data, int bytcnt)
{
	double d;
	int i;

	/* NOTE: snan can be directly injected here */
	memset(coll_data, 0, sizeof(*coll_data));
	if (user_data)
		memcpy(coll_data->databuf, user_data, bytcnt);
	coll_data->red_rc = 0;
	coll_data->red_op = opcode;
	switch (coll_data->red_op) {
	case COLL_OPCODE_FLT_MIN:
	case COLL_OPCODE_FLT_MAX:
	case COLL_OPCODE_FLT_MINNUM:
	case COLL_OPCODE_FLT_MAXNUM:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND0:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND1:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND2:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND3:
	case COLL_OPCODE_FLT_SUM_FTZ_RND0:
	case COLL_OPCODE_FLT_SUM_FTZ_RND1:
	case COLL_OPCODE_FLT_SUM_FTZ_RND2:
	case COLL_OPCODE_FLT_SUM_FTZ_RND3:
		/* evaluate all four doubles */
		for (i = 0; i < 4; i++) {
			if (cxip_is_snan64(coll_data->fltval.fval[i]))
				SET_RED_RC(coll_data->red_rc,
					   CXIP_COLL_RC_FLT_INVALID);
			if (isnan(coll_data->fltval.fval[i]) ||
			    isinf(coll_data->fltval.fval[i]))
				SET_RED_RC(coll_data->red_rc,
					   CXIP_COLL_RC_FLT_OVERFLOW);
		}
		break;
	case COLL_OPCODE_FLT_MINMAXLOC:
	case COLL_OPCODE_FLT_MINMAXNUMLOC:
		/* evaluate the two doubles */
		for (i = 0; i < 4; i += 2) {
			if (cxip_is_snan64(coll_data->fltval.fval[i]))
				SET_RED_RC(coll_data->red_rc,
				CXIP_COLL_RC_FLT_INVALID);
			if (isinf(coll_data->fltval.fval[i]))
				SET_RED_RC(coll_data->red_rc,
					   CXIP_COLL_RC_FLT_OVERFLOW);
		}
		break;
	case COLL_OPCODE_FLT_REPSUM:
		/* perform the conversion */
		d = coll_data->fltval.fval[0];
		cxip_dbl_to_rep(&coll_data->repsum, d);
		break;
	}
	coll_data->initialized = true;
}

/* reduce data into accumulator - can be used on uninitialized accumulator */
static void _reduce(struct cxip_coll_data *accum,
		    const struct cxip_coll_data *coll_data,
		    bool pre_reduce)
{
	int i, swp;

	/* Initialize with new data */
	if (!accum->initialized) {
		memcpy(accum, coll_data, sizeof(*accum));
		return;
	}

	/* copy new error to accumulator */
	SET_RED_RC(accum->red_rc, coll_data->red_rc);

	/* Pre-reduction never counts the contributions, and cannot overflow.
	 * Real reduction (send or receive) must count contributions.
	 * red_max is zero until after injection from this node.
	 */
	if (!pre_reduce) {
		accum->red_cnt += coll_data->red_cnt;
		if (!accum->red_max)
			accum->red_max = coll_data->red_max;
		if (accum->red_cnt > accum->red_max) {
			SET_RED_RC(accum->red_rc, CXIP_COLL_RC_CONTR_OVERFLOW);
			return;
		}
	}

	/* ops must always match, else don't apply data */
	if (accum->red_op != coll_data->red_op) {
		SET_RED_RC(accum->red_rc, CXIP_COLL_RC_OP_MISMATCH);
		return;
	}

	/* Perform the reduction in software */
	switch (accum->red_op) {
	case COLL_OPCODE_BARRIER:
		break;
	case COLL_OPCODE_BIT_AND:
		for (i = 0; i < 4; i++)
			accum->intval.ival[i] &= coll_data->intval.ival[i];
		/* overflow not possible */
		break;
	case COLL_OPCODE_BIT_OR:
		for (i = 0; i < 4; i++)
			accum->intval.ival[i] |= coll_data->intval.ival[i];
		/* overflow not possible */
		break;
	case COLL_OPCODE_BIT_XOR:
		for (i = 0; i < 4; i++)
			accum->intval.ival[i] ^= coll_data->intval.ival[i];
		/* overflow not possible */
		break;
	case COLL_OPCODE_INT_MIN:
		for (i = 0; i < 4; i++)
			if (accum->intval.ival[i] > coll_data->intval.ival[i])
				accum->intval.ival[i] = coll_data->intval.ival[i];
		/* overflow not possible */
		break;
	case COLL_OPCODE_INT_MAX:
		for (i = 0; i < 4; i++)
			if (accum->intval.ival[i] < coll_data->intval.ival[i])
				accum->intval.ival[i] = coll_data->intval.ival[i];
		/* overflow not possible */
		break;
	case COLL_OPCODE_INT_MINMAXLOC:
		/* RSDG 4.5.9.2.2 MINMAXLOC */
		/* return smallest value and its index */
		if (accum->intminmax.iminval > coll_data->intminmax.iminval) {
			accum->intminmax.iminval = coll_data->intminmax.iminval;
			accum->intminmax.iminidx = coll_data->intminmax.iminidx;
		} else
		/* return smallest index if values equal */
		if (accum->intminmax.iminval == coll_data->intminmax.iminval &&
		    accum->intminmax.iminidx > coll_data->intminmax.iminidx) {
			accum->intminmax.iminidx = coll_data->intminmax.iminidx;
		}

		/* return largest value and its index */
		if (accum->intminmax.imaxval < coll_data->intminmax.imaxval) {
			accum->intminmax.imaxval = coll_data->intminmax.imaxval;
			accum->intminmax.imaxidx = coll_data->intminmax.imaxidx;
		} else
		/* return smallest (yes) index if values equal */
		if (accum->intminmax.imaxval == coll_data->intminmax.imaxval &&
		    accum->intminmax.imaxidx > coll_data->intminmax.imaxidx) {
			accum->intminmax.imaxidx = coll_data->intminmax.imaxidx;
		}
		/* overflow not possible */
		break;
	case COLL_OPCODE_INT_SUM:
		for (i = 0; i < 4; i++) {
			bool newneg = (coll_data->intval.ival[i] < 0);
			bool oldneg = (accum->intval.ival[i] < 0);
			bool sumneg;
			accum->intval.ival[i] += coll_data->intval.ival[i];
			sumneg = (accum->intval.ival[i] < 0);
			/* if sum changed sign, and doesn't match new sign */
			if (sumneg != oldneg && sumneg != newneg)
				SET_RED_RC(accum->red_rc,
					   CXIP_COLL_RC_INT_OVERFLOW);
		}
		break;
	case COLL_OPCODE_FLT_MIN:
		/* RSDG 4.5.9.2.3 FLT_MIN and FLT_MAX */
		for (i = 0; i < 4; i++) {
			swpnan1(&accum->fltval.fval[i], coll_data->fltval.fval[i], 1,
				&accum->red_rc);
		}
		break;
	case COLL_OPCODE_FLT_MAX:
		/* RSDG 4.5.9.2.3 FLT_MIN and FLT_MAX */
		for (i = 0; i < 4; i++)
			swpnan1(&accum->fltval.fval[i], coll_data->fltval.fval[i], 0,
				&accum->red_rc);
		break;
	case COLL_OPCODE_FLT_MINMAXLOC:
		/* RSDG 4.5.9.2.3 FLT_MIN and FLT_MAX */
		swp = swpnan1(&accum->fltminmax.fminval,
			      coll_data->fltminmax.fminval, 1, &accum->red_rc);
		swpidx(&accum->fltminmax.fminidx, coll_data->fltminmax.fminidx, swp);
		swp = swpnan1(&accum->fltminmax.fmaxval,
			      coll_data->fltminmax.fmaxval, 0, &accum->red_rc);
		swpidx(&accum->fltminmax.fmaxidx, coll_data->fltminmax.fmaxidx, swp);
		break;
	case COLL_OPCODE_FLT_MINNUM:
		/* RSDG 4.5.9.2.4 FLT_MINNUM and FLT_MAXNUM */
		for (i = 0; i < 4; i++) {
			swpnan2(&accum->fltval.fval[i], coll_data->fltval.fval[i], 1,
				&accum->red_rc);
		}
		break;
	case COLL_OPCODE_FLT_MAXNUM:
		/* RSDG 4.5.9.2.4 FLT_MINNUM and FLT_MAXNUM */
		for (i = 0; i < 4; i++) {
			swpnan2(&accum->fltval.fval[i], coll_data->fltval.fval[i], 0,
				&accum->red_rc);
		}
		break;
	case COLL_OPCODE_FLT_MINMAXNUMLOC:
		/* RSDG 4.5.9.2.4 FLT_MINNUM and FLT_MAXNUM */
		swp = swpnan2(&accum->fltminmax.fminval,
			      coll_data->fltminmax.fminval, 1, &accum->red_rc);
		swpidx(&accum->fltminmax.fminidx, coll_data->fltminmax.fminidx, swp);
		swp = swpnan2(&accum->fltminmax.fmaxval,
			      coll_data->fltminmax.fmaxval, 0, &accum->red_rc);
		swpidx(&accum->fltminmax.fmaxidx, coll_data->fltminmax.fmaxidx, swp);
		break;
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND0:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND1:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND2:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND3:
		/* Rosetta opcode has been chosen according to the current
		 * rounding mode for this application, so all we need to do is
		 * add the numbers.
		 */
		for (i = 0; i < 4; i++) {
			/* NOTE: arithmetic operations will quiesce snan */
			accum->fltval.fval[i] +=  coll_data->fltval.fval[i];

			if (!exact(accum->fltval.fval[i],
				   coll_data->fltval.fval[i]))
				SET_RED_RC(accum->red_rc,
					   CXIP_COLL_RC_FLT_INEXACT);
			if (isinf(accum->fltval.fval[i]))
				SET_RED_RC(accum->red_rc,
					   CXIP_COLL_RC_FLT_OVERFLOW);
		}
		break;
	case COLL_OPCODE_FLT_SUM_FTZ_RND0:
	case COLL_OPCODE_FLT_SUM_FTZ_RND1:
	case COLL_OPCODE_FLT_SUM_FTZ_RND2:
	case COLL_OPCODE_FLT_SUM_FTZ_RND3:
		/* Rosetta opcode has been chosen according to the current
		 * rounding mode for this application, so all we need to do is
		 * add the numbers.
		 */
		for (i = 0; i < 4; i++) {
			/* NOTE: arithmetic operations will quiesce snan */
			accum->fltval.fval[i] +=  coll_data->fltval.fval[i];

			if (!exact(accum->fltval.fval[i],
				   coll_data->fltval.fval[i]))
				SET_RED_RC(accum->red_rc,
					   CXIP_COLL_RC_FLT_INEXACT);
			if (isinf(accum->fltval.fval[i]))
				SET_RED_RC(accum->red_rc,
					   CXIP_COLL_RC_FLT_OVERFLOW);
		}
		break;
	case COLL_OPCODE_FLT_REPSUM:
		cxip_rep_add(&accum->repsum, &coll_data->repsum);
		break;
	}
}

/****************************************************************************
 * Reduction packet management.
 */

/**
 * Prevent setting the ARM bit on a root packet.
 *
 * This is used in testing to suppress Rosetta collective operations, forcing
 * all leaf packets to arrive at the root, creating an incast.
 */
int cxip_coll_arm_disable(struct fid_mc *mc, bool disable)
{
	struct cxip_coll_mc *mc_obj = (struct cxip_coll_mc *)mc;
	int old = mc_obj->arm_disable;

	mc_obj->arm_disable = disable;

	return old;
}

/**
 * Limit the reduction ID values.
 *
 * Reduction ID values do round-robin over an adjustable range of values. This
 * is useful in testing to force all reductions to use reduction id zero (set
 * max_red_id to 1), but could be used in production to use only a subset of
 * reduction IDs to limit fabric resource exhaustion when concurrent reductions
 * are used.
 */
void cxip_coll_limit_red_id(struct fid_mc *mc, int max_red_id)
{
	struct cxip_coll_mc *mc_obj = (struct cxip_coll_mc *)mc;

	if (max_red_id < 1)
		max_red_id = 1;
	if (max_red_id > CXIP_COLL_MAX_CONCUR)
		max_red_id = CXIP_COLL_MAX_CONCUR;
	mc_obj->max_red_id = max_red_id;
}

/* drop the next packet sent */
void cxip_coll_drop_send(struct cxip_coll_reduction *reduction)
{
	reduction->drop_send = true;
}

/* drop the next packet received */
void cxip_coll_drop_recv(struct cxip_coll_reduction *reduction)
{
	reduction->drop_recv = true;
}

/* Return true if this node is the hwroot node */
static inline
bool is_hw_root(struct cxip_coll_mc *mc_obj)
{
	return (mc_obj->hwroot_idx == mc_obj->mynode_idx);
}

/* Simulated unicast send of multiple packets as root node to leaf nodes */
static inline
ssize_t _send_pkt_as_root(struct cxip_coll_reduction *reduction, bool retry)
{
	int i, ret;

	for (i = 0; i < reduction->mc_obj->av_set->fi_addr_cnt; i++) {
		if (i == reduction->mc_obj->mynode_idx &&
		    reduction->mc_obj->av_set->fi_addr_cnt > 1)
			continue;
		ret = cxip_coll_send(reduction, i,
				     reduction->tx_msg,
				     sizeof(struct red_pkt),
				     reduction->mc_obj->reduction_md);
		if (ret)
			return ret;
	}
	return FI_SUCCESS;
}

/* Simulated unicast send of single packet as leaf node to root node */
static inline
ssize_t _send_pkt_as_leaf(struct cxip_coll_reduction *reduction, bool retry)
{
	return cxip_coll_send(reduction, reduction->mc_obj->hwroot_idx,
			      reduction->tx_msg, sizeof(struct red_pkt),
			      reduction->mc_obj->reduction_md);
}

/* Multicast send of single packet from root or leaf node */
static inline
ssize_t _send_pkt_mc(struct cxip_coll_reduction *reduction, bool retry)
{
	return cxip_coll_send(reduction, 0,
			      reduction->tx_msg,
			      sizeof(struct red_pkt),
			      reduction->mc_obj->reduction_md);
}

/* Send packet from root or leaf node as appropriate */
static inline
ssize_t _send_pkt(struct cxip_coll_reduction *reduction, bool retry)
{
	int ret;

	if (reduction->mc_obj->av_set->comm_key.keytype == COMM_KEY_MULTICAST) {
		ret = _send_pkt_mc(reduction, retry);
	} else if (is_hw_root(reduction->mc_obj)) {
		ret = _send_pkt_as_root(reduction, retry);
	} else {
		ret = _send_pkt_as_leaf(reduction, retry);
	}
	return ret;
}

/* prepare and issue the reduction packet */
int cxip_coll_send_red_pkt(struct cxip_coll_reduction *reduction,
			   const struct cxip_coll_data *coll_data,
			   bool arm, bool retry)
{
	struct red_pkt *pkt;
	int ret;

	pkt = (struct red_pkt *)reduction->tx_msg;

	memset(&pkt->hdr, 0, sizeof(pkt->hdr));
	pkt->hdr.arm = arm;
	pkt->hdr.seqno = reduction->seqno;
	pkt->hdr.resno = reduction->resno;
	pkt->hdr.cookie.mcast_id = reduction->mc_obj->mcast_addr;
	pkt->hdr.cookie.red_id = reduction->red_id;
	pkt->hdr.cookie.retry = retry;
	pkt->hdr.cookie.magic = MAGIC;

	if (coll_data) {
		pkt->hdr.redcnt = coll_data->red_cnt;
		pkt->hdr.op = coll_data->red_op;
		pkt->hdr.red_rc = coll_data->red_rc;
		/* repsum has some additional information that must be set */
		if (_opcode_to_redtype(coll_data->red_op) == REDTYPE_REPSUM) {
			pkt->hdr.repsum_m = coll_data->repsum.M;
			pkt->hdr.repsum_ovflid = coll_data->repsum.overflow_id;
		}
		memcpy(pkt->data, &coll_data->databuf, CXIP_COLL_MAX_TX_SIZE);
	} else {
		pkt->hdr.redcnt = 0;
		pkt->hdr.op = 0;
		pkt->hdr.red_rc = 0;
		pkt->hdr.repsum_m = 0;
		pkt->hdr.repsum_ovflid = 0;
		memset(pkt->data, 0, CXIP_COLL_MAX_TX_SIZE);
	}
	_dump_red_pkt(pkt, "send");
	_swappkt(pkt);

	/* -FI_EAGAIN means HW queue is full, should self-clear */
	do {
		ret = _send_pkt(reduction, retry);
	} while (ret == -FI_EAGAIN);
	/* any other error is a serious config/hardware issue */
	if (ret)
		CXIP_WARN("Fatal send error = %d\n", ret);

	return ret;
}

/* Post a reduction completion request to the collective completion queue */
static void _post_coll_complete(struct cxip_coll_reduction *reduction)
{
	struct cxip_req *req;
	int ret;

	/* Indicates collective completion by writing to the endpoint TX CQ */
	req = reduction->op_inject_req;
	reduction->op_inject_req = NULL;
	if (!req) {
		TRACE("red_id=%2d NO REQ\n", reduction->red_id);
		return;
	}

	if (reduction->accum.red_rc == CXIP_COLL_RC_SUCCESS) {
		ret = cxip_cq_req_complete(req);
	} else {
		ret = cxip_cq_req_error(req, 0,
			_cxip_rc_to_cxi_rc[reduction->accum.red_rc],
			reduction->accum.red_rc, NULL, 0);
	}
	cxip_evtq_req_free(req);

	if (ret) {
		/* Is this possible? The only error is -FI_ENOMEM. It looks like
		 * send is blocked with -FI_EAGAIN until we are guaranteed EQ
		 * space in the queue. Display and ignore.
		 */
		CXIP_WARN("Attempt to post completion failed %s\n",
			   fi_strerror(-ret));
	}
}

/* unpack reduction data from a reduction packet */
static void _unpack_red_data(struct cxip_coll_data *coll_data,
			     const struct red_pkt *pkt)
{
	memcpy(coll_data->databuf, pkt->data, 32);
	coll_data->repsum.M = pkt->hdr.repsum_m;
	coll_data->repsum.overflow_id = pkt->hdr.repsum_ovflid;
	coll_data->red_op = pkt->hdr.op;
	coll_data->red_cnt = pkt->hdr.redcnt;
	coll_data->red_rc = pkt->hdr.red_rc;
	coll_data->initialized = true;
}

/****************************************************************************
 * Collective State Machine
 *
 * The basic flow is:
 *   - all nodes reach a common reduction call (at different times)
 *   - leaf nodes send their data, to be reduced, and block, polling CQ
 *   - root node prepares for the reduction, and blocks, polling CQ
 *   - root node receives leaf packets and reduces them, until all received
 *   - root node sends Arm Packet with final result, and unblocks
 *   - leaf nodes receive Arm Packet with final result, and unblock
 *
 * The Rosetta acceleration comes from the Arm Packet, which speculatively arms
 * the Rosetta tree for the NEXT operation. This persists until a timeout
 * expires. The timeout is specified when the multicast tree is created by the
 * Rosetta configuration service, and cannot be modified after join is complete.
 *
 * If the next collective operation occurs within the timeout, the leaf results
 * will be reduced in reduction engines by Rosetta as they move up the tree,
 * reducing the number of packets received by the root.
 *
 * If the reduction engine times out with partial results, it forwards the
 * partial results, and all subsequent results are passed directly to the next
 * Rosetta.
 *
 * The first leaf contribution to reach a reduction engine establishes the
 * reduction operation. All subsequent contributions must use the same
 * operation, or Rosetta returns an error.
 *
 * There are eight reduction_id values, which can be used to acquire and use up
 * to eight independent reduction engines (REs) at each upstream port of each
 * Rosetta switch in the collective tree.
 *
 * We use a round-robin selection of reduction id values. There is a small race
 * condition among the leaf nodes as the result is distributed from the root. If
 * another reduction were to be initiated during this race, the leaf nodes would
 * be in disagreement as to which reduction IDs were free for the new reduction.
 * To avoid this, we use a deterministic algorithm (round-robin) so that the
 * "next" reduction id is always predetermined for each reduction.
 *
 * Ordering of requests and responses will the same on all nodes.
 *
 * Ordering of requests is required of the application. If requests are ordered
 * differently on different nodes, results are undefined, and it is considered
 * an application error.
 *
 * Ordering of responses is guaranteed by the mc_obj->tail_red_id value, which
 * is advanced after the reduction completes. This ordering is required to
 * ensure that the round-robin is observed
 */

/* modular increment/decrement */
#define INCMOD(val, mod)	do {(val)=((val)+1)%(mod);} while (0)
#define DECMOD(val, mod)	do {(val)=((val)+(mod)-1)%(mod);} while (0)

/* MONOTONIC timestamp operations for timeouts/retries */
static inline
void _tsget(struct timespec *ts)
{
	clock_gettime(CLOCK_MONOTONIC, ts);
}

static inline
void _tsadd(struct timespec *ts, const struct timespec *dt)
{
	ts->tv_sec += dt->tv_sec;
	ts->tv_nsec += dt->tv_nsec;
	if (ts->tv_nsec >= 1000000000L) {
		ts->tv_sec += 1;
		ts->tv_nsec -= 1000000000L;
	}
}

/* Set a timespec at expiration time (future) */
static inline
void _tsset(struct cxip_coll_reduction *reduction)
{
	_tsget(&reduction->tv_expires);
	_tsadd(&reduction->tv_expires, &reduction->mc_obj->timeout);
}

/* Used to prevent first-use incast */
static inline
bool _is_red_first_time(struct cxip_coll_reduction *reduction)
{
	return (reduction->tv_expires.tv_sec == 0L &&
	    	reduction->tv_expires.tv_nsec == 0L);
}

/* Used to reduce incast congestion during run */
static inline
bool _is_red_timed_out(struct cxip_coll_reduction *reduction)
{
	struct timespec tsnow;

	if (_is_red_first_time(reduction))
		return true;
	_tsget(&tsnow);
	if (tsnow.tv_sec < reduction->tv_expires.tv_sec)
		return false;
	if (tsnow.tv_sec > reduction->tv_expires.tv_sec)
		return true;
	if (tsnow.tv_nsec < reduction->tv_expires.tv_nsec)
		return false;
	return true;
}

/* Root node state machine.
 * !pkt means this is progressing from injection call (e.g. fi_reduce())
 *  pkt means this is progressing from event callback (leaf packet)
 */
static void _progress_root(struct cxip_coll_reduction *reduction,
			   struct red_pkt *pkt)
{
	struct cxip_coll_mc *mc_obj = reduction->mc_obj;
	struct cxip_coll_data coll_data;
	ssize_t ret;

	/* State machine disabled for testing */
	if (reduction->coll_state != CXIP_COLL_STATE_READY)
		return;

	/* Injection or packet arrival after root timeout initiates a retry */
	if (_is_red_timed_out(reduction)) {
		/* reset reduction for retry send */
		reduction->seqno = mc_obj->seqno;
		INCMOD(mc_obj->seqno, CXIP_COLL_MAX_SEQNO);
		ofi_atomic_inc32(&mc_obj->tmout_cnt);

		_tsset(reduction);
		ret = cxip_coll_send_red_pkt(reduction, NULL,
					     !mc_obj->arm_disable, true);
		if (ret) {
			SET_RED_RC(reduction->accum.red_rc,
				   CXIP_COLL_RC_TX_FAILURE);
			reduction->completed = true;
			goto post_complete;
		}
		return;
	}

	/* If injection, nothing more to do */
	if (!pkt)
		return;

	/* Leaf packet receipt, process */
	_dump_red_pkt(pkt, "Rrcv");

	/* Drop out-of-date packets */
	if (pkt->hdr.resno != reduction->seqno) {
		ofi_atomic_inc32(&mc_obj->seq_err_cnt);
		return;
	}

	/* capture and reduce packet information */
	_unpack_red_data(&coll_data, pkt);
	_reduce(&reduction->accum, &coll_data, false);
	_dump_coll_data("after leaf contrib to root", &reduction->accum);

	/* check for reduction complete */
	if (reduction->accum.red_cnt == reduction->accum.red_max) {
		/* copy reduction result to user result buffer */
		if (reduction->op_rslt_data && reduction->op_data_bytcnt) {
			memcpy(reduction->op_rslt_data,
			       reduction->accum.databuf,
			       reduction->op_data_bytcnt);
		}

		/* send reduction result to leaves, arming the next pass */
		reduction->seqno = mc_obj->seqno;
		INCMOD(mc_obj->seqno, CXIP_COLL_MAX_SEQNO);
		reduction->completed = true;

		_tsset(reduction);
		ret = cxip_coll_send_red_pkt(reduction, &reduction->accum,
					     !mc_obj->arm_disable, false);
		if (ret)
			SET_RED_RC(reduction->accum.red_rc,
				   CXIP_COLL_RC_TX_FAILURE);
	}

post_complete:
	/* Post completions in injection order */
	reduction = &mc_obj->reduction[mc_obj->tail_red_id];
	while (reduction->in_use && reduction->completed) {
		/* Reduction completed on root */
		_post_coll_complete(reduction);
		reduction->in_use = false;

		/* Advance to the next reduction */
		INCMOD(mc_obj->tail_red_id, mc_obj->max_red_id);
		reduction = &mc_obj->reduction[mc_obj->tail_red_id];
	}
}

/* Leaf node state machine.
 * !pkt means this is progressing from injection call (e.g. fi_reduce())
 *  pkt means this is progressing from event callback (receipt of packet)
 */
static void _progress_leaf(struct cxip_coll_reduction *reduction,
			   struct red_pkt *pkt)
{
	struct cxip_coll_mc *mc_obj = reduction->mc_obj;
	struct cxip_coll_data coll_data;
	int ret;

	/* state machine disabled for testing */
	if (reduction->coll_state != CXIP_COLL_STATE_READY)
		return;

	/* if reduction packet, reset timer, seqno, honor retry */
	if (pkt) {
		_dump_red_pkt(pkt, "Lrcv");

		_tsset(reduction);
		reduction->seqno = pkt->hdr.seqno;
		reduction->resno = pkt->hdr.seqno;
		if (pkt->hdr.cookie.retry)
			reduction->pktsent = false;
	}

	/* leaves lead with sending a packet */
	if (!reduction->pktsent) {
		/* Avoid first-use incast, retry guaranteed */
		if (_is_red_first_time(reduction))
			return;

		/* Send leaf data */
		ret = cxip_coll_send_red_pkt(reduction, &reduction->accum,
					     false, false);
		if (ret) {
			SET_RED_RC(reduction->accum.red_rc,
				   CXIP_COLL_RC_TX_FAILURE);
			reduction->completed = true;
			goto post_complete;
		}
		reduction->pktsent = true;
	}

	/* If no incoming reduction packet, we are done */
	if (!pkt)
		return;

	/* If packet has no reduction count (retry), done */
	if (!pkt->hdr.redcnt)
		return;

	/* Capture final reduction data in user-pointer */
	SET_RED_RC(reduction->accum.red_rc, pkt->hdr.red_rc);
	if (reduction->op_rslt_data) {
		_unpack_red_data(&coll_data, pkt);
		memcpy(reduction->op_rslt_data,
			&coll_data.databuf,
			reduction->op_data_bytcnt);
	}

post_complete:
	/* Reduction completed on leaf */
	reduction->completed = true;

	/* Post completions in injection order */
	reduction = &mc_obj->reduction[mc_obj->tail_red_id];
	while (reduction->in_use && reduction->completed) {
		_post_coll_complete(reduction);
		reduction->in_use = false;
		INCMOD(mc_obj->tail_red_id, mc_obj->max_red_id);
		reduction = &mc_obj->reduction[mc_obj->tail_red_id];
	}
}

/* Root or leaf progress state machine.
 */
static void _progress_coll(struct cxip_coll_reduction *reduction,
			   struct red_pkt *pkt)
{
	if (is_hw_root(reduction->mc_obj))
		_progress_root(reduction, pkt);
	else
		_progress_leaf(reduction, pkt);
}

/* Debugging only */
static int *_injected_red_id_buf;
void cxip_capture_red_id(int *red_id_buf)
{
	_injected_red_id_buf = red_id_buf;
}

/* Generic collective pre-reduction into cxip_coll_data structure */
static void
_cxip_coll_prereduce(int cxi_opcode, const void *op_send_data,
		     void *op_rslt_data, size_t sendcnt, uint64_t flags)
{
	struct cxip_coll_data *accum = op_rslt_data;
	struct cxip_coll_data coll_data;
	const struct cxip_coll_data *coll_data_ptr;

	/* Convert user data to local coll_data structure */
	if (flags & FI_CXI_PRE_REDUCED) {
		coll_data_ptr = op_send_data;
	} else {
		_init_coll_data(&coll_data, cxi_opcode, op_send_data,
				sendcnt);
		coll_data_ptr = &coll_data;
	}
	_dump_coll_data("coll_data initialized", coll_data_ptr);

	/* pre-reduce data into accumulator */
	_reduce(accum, coll_data_ptr, true);
}

/* Generic collective injection into fabric.
 *
 * Reduction ID is normally hidden. Can be exposed by calling _capture_red_id()
 * just before calling a reduction operation.
 */
static ssize_t
_cxip_coll_inject(struct cxip_coll_mc *mc_obj, int cxi_opcode,
		  const void *op_send_data, void *op_rslt_data,
		  size_t bytcnt, uint64_t flags, void *context)
{
	struct cxip_coll_reduction *reduction;
	struct cxip_coll_data coll_data;
	struct cxip_req *req;
	int ret;

	ofi_genlock_lock(&mc_obj->ep_obj->lock);

	/* must observe strict round-robin across all nodes */
	reduction = &mc_obj->reduction[mc_obj->next_red_id];
	if (reduction->in_use) {
		ret = -FI_EAGAIN;
		goto quit;
	}

	/* acquire a request structure */
	req = cxip_evtq_req_alloc(mc_obj->ep_obj->coll.tx_evtq, 1, NULL);
	if (!req) {
		ret = -FI_EAGAIN;
		goto quit;
	}

	/* Used for debugging */
	if (_injected_red_id_buf) {
		*_injected_red_id_buf = reduction->red_id;
		_injected_red_id_buf = NULL;
	}

	/* advance next_red_id, reserving this one for us */
	INCMOD(mc_obj->next_red_id, mc_obj->max_red_id);
	reduction->in_use = true;

	/* Set up the reduction structure */
	reduction->pktsent = false;
	reduction->completed = false;
	reduction->accum.initialized = false;
	reduction->op_rslt_data = op_rslt_data;
	reduction->op_data_bytcnt = bytcnt;
	reduction->op_context = context;
	reduction->op_inject_req = req;
	reduction->op_inject_req->context = (uint64_t)context;

	/* Convert user data to local coll_data structure */
	if (flags & FI_CXI_PRE_REDUCED)
		memcpy(&coll_data, op_send_data, sizeof(coll_data));
	else
		_init_coll_data(&coll_data, cxi_opcode, op_send_data, bytcnt);

	_dump_coll_data("coll_data initialized", &coll_data);

	/* reduce data into accumulator */
	coll_data.red_cnt = 1;
	coll_data.red_max = mc_obj->av_set->fi_addr_cnt;
	_reduce(&reduction->accum, &coll_data, false);

	/* Progress the collective */
	_progress_coll(reduction, NULL);
	ret = FI_SUCCESS;

quit:
	ofi_genlock_unlock(&mc_obj->ep_obj->lock);
	return ret;
}

/* Get the mc_obj from ep/coll_addr and check for consistency */
static inline
ssize_t _get_mc_obj(struct fid_ep *ep, fi_addr_t coll_addr,
		    struct cxip_coll_mc **mc_obj)
{
	struct cxip_ep *cxi_ep;

	if (!ep) {
		CXIP_WARN("Collective requires ep\n");
		return -FI_EINVAL;
	}

	if (!coll_addr) {
		CXIP_WARN("Collective requires coll_addr\n");
		return -FI_EINVAL;
	}

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	*mc_obj = (struct cxip_coll_mc *)((uintptr_t)coll_addr);

	if ((*mc_obj)->ep_obj != cxi_ep->ep_obj) {
		CXIP_WARN("Multicast does not belong to ep\n");
		return -FI_EINVAL;
	}

	if (!(*mc_obj)->is_joined) {
		CXIP_WARN("Multicast collective not joined\n");
		return -FI_EOPBADSTATE;
	}

	return FI_SUCCESS;
}

/* get payload byte count and check for consistency */
static inline
ssize_t _get_bytcnt(int cxi_opcode, enum fi_datatype datatype,
		    const void *buf, size_t count)
{
	ssize_t bytcnt;

	if (cxi_opcode < 0) {
		CXIP_WARN("opcode not supported\n");
		return -FI_EINVAL;
	}

	if (!buf || count <= 0L) {
		CXIP_WARN("buffer required\n");
		return -FI_EINVAL;
	}

	bytcnt = _get_cxi_data_bytcnt(cxi_opcode, datatype, count);
	if (bytcnt < 0)
		CXIP_WARN("opcode does not support datatype\n");

	return bytcnt;
}

ssize_t cxip_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context)
{
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode;
	ssize_t ret;

	/* barrier requires mc_obj */
	ret = _get_mc_obj(ep, coll_addr, &mc_obj);
	if (ret)
		return ret;

	cxi_opcode = COLL_OPCODE_BARRIER;

	return _cxip_coll_inject(mc_obj, cxi_opcode, NULL, NULL, 0, 0, context);
}

ssize_t cxip_broadcast(struct fid_ep *ep, void *buf, size_t count,
		       void *desc, fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, uint64_t flags,
		       void *context)
{
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode, bytcnt;
	ssize_t ret;

	if (flags & (FI_MORE|FI_CXI_PRE_REDUCED)) {
		CXIP_WARN("Illegal flags for broadcast\n");
		return -FI_EINVAL;
	}

	cxi_opcode = COLL_OPCODE_BIT_OR;
	bytcnt = _get_bytcnt(cxi_opcode, datatype, buf, count);
	if (bytcnt < 0)
		return -FI_EINVAL;

	/* broadcast requires mc_obj */
	ret = _get_mc_obj(ep, coll_addr, &mc_obj);
	if (ret)
		return ret;

	/* only root node contributes data, others contribute 0 */
	if (root_addr != mc_obj->mynode_fiaddr)
		memset(buf, 0, bytcnt);

	/* buf serves as source and result */
	return _cxip_coll_inject(mc_obj, cxi_opcode, buf, buf, bytcnt,
				flags, context);
}

ssize_t cxip_reduce(struct fid_ep *ep, const void *buf, size_t count,
		    void *desc, void *result, void *result_desc,
		    fi_addr_t coll_addr, fi_addr_t root_addr,
		    enum fi_datatype datatype, enum fi_op op, uint64_t flags,
		    void *context)
{
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode;
	ssize_t bytcnt, ret;

	cxi_opcode = cxip_fi2cxi_opcode(op, datatype);
	bytcnt = _get_bytcnt(cxi_opcode, datatype, buf, count);
	if (bytcnt < 0)
		return (ssize_t)bytcnt;

	/* FI_MORE requires target buffer, succeeds immediately */
	if (flags & FI_MORE) {
		if (!result) {
			CXIP_WARN("result required with FI_MORE\n");
			return -FI_EINVAL;
		}
		_cxip_coll_prereduce(cxi_opcode, buf, result, bytcnt, flags);
		return FI_SUCCESS;
	}

	/* otherwise reduce requires mc_obj */
	ret = _get_mc_obj(ep, coll_addr, &mc_obj);
	if (ret)
		return ret;

	/* root requires a result buffer */
	if (!result && (mc_obj->mynode_fiaddr == root_addr)) {
		CXIP_WARN("reduce root result required\n");
		return -FI_EINVAL;
	}

	return _cxip_coll_inject(mc_obj, cxi_opcode, buf, result, bytcnt,
				flags, context);
}

ssize_t cxip_allreduce(struct fid_ep *ep, const void *buf, size_t count,
		       void *desc, void *result, void *result_desc,
		       fi_addr_t coll_addr, enum fi_datatype datatype,
		       enum fi_op op, uint64_t flags, void *context)
{
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode, bytcnt;
	ssize_t ret;

	cxi_opcode = cxip_fi2cxi_opcode(op, datatype);
	bytcnt = _get_bytcnt(cxi_opcode, datatype, buf, count);
	if (bytcnt < 0)
		return bytcnt;

	/* result required in all cases */
	if (!result) {
		CXIP_WARN("result required with FI_MORE\n");
		return -FI_EINVAL;
	}

	/* FI_MORE succeeds immediately */
	if (flags & FI_MORE) {
		_cxip_coll_prereduce(cxi_opcode, buf, result, bytcnt, flags);
		return FI_SUCCESS;
	}

	/* otherwise reduce requires mc_obj */
	ret = _get_mc_obj(ep, coll_addr, &mc_obj);
	if (ret)
		return ret;

	return _cxip_coll_inject(mc_obj, cxi_opcode, buf, result, bytcnt,
				flags, context);
}

/****************************************************************************
 * JOIN COLLECTIVE STATE MACHINE
 */

/* Packed structure to fit information into zbcoll broadcast payload */
union pack_mcast {
	uint64_t uint64;
	struct {
		uint64_t mcast_addr: 16;// maximum anticipated multicast
		uint64_t hwroot_idx: 27;// 128M endpoints in tree
		uint64_t valid: 1;	// success flag
		uint64_t pad2: 20;	// needed by zbcoll
	} __attribute__((__packed__));
};

/* State structure for carrying data through the join sequence */
struct cxip_join_state {
	struct cxip_ep_obj *ep_obj;	// ep object
	struct cxip_av_set *av_set;	// av set for this collective
	struct cxip_coll_mc *mc_obj;	// mc object for this collective
	struct cxip_zbcoll_obj *zb;	// zb object associated with state
	struct fid_mc **mc;		// user pointer to return mc_obj
	void *context;			// user context for concurrent joins
	uint64_t join_flags;		// user-supplied libfabric join flags
	union pack_mcast bcast_data;	// packed multicast data
	bool is_mcast;			// set if using Rosetta multicast tree
	bool create_mcast;		// set to create Rosetta multicast tree
	bool creating_mcast;		// set once CURL has been initiated
	bool created_mcast;		// set once CURL has been completed
	bool created_ptlte;		// set once PtlTE is initialized
	int mynode_idx;			// index within the fi_addr[] list
	int mynode_fiaddr;		// fi_addr of this node
	int simrank;			// simulated rank of NIC
	int pid_idx;			// pid_idx used by ptl_te
	int prov_errno;			// collective provider error
	int sched_state;		// scheduled
	struct dlist_entry sched_link;	// link to scheduled actions
};

/* State structure for recovering data from CURL response */
struct cxip_curl_mcast_usrptr {
	struct cxip_join_state *jstate;	// join state
	int mcast_id;			// multicast address
	int hwroot_rank;		// hardware root index
};

static int _close_mc(struct fid *fid);

/* multicast object operational functions */
static struct fi_ops mc_ops = {
	.size = sizeof(struct fi_ops),
	.close = _close_mc,
};

/* Close multicast collective object */
static int _close_mc(struct fid *fid)
{
	struct cxip_coll_mc *mc_obj;
	int ret;

	TRACE_JOIN("%s entry\n", __func__);
	mc_obj = container_of(fid, struct cxip_coll_mc, mc_fid.fid);
	if (mc_obj->coll_pte) {
		do {
			ret = _coll_pte_disable(mc_obj->coll_pte);
		} while (ret == -FI_EAGAIN);

		_coll_destroy_buffers(mc_obj->coll_pte);
		cxip_pte_free(mc_obj->coll_pte->pte);
		free(mc_obj->coll_pte);
	}
	if (mc_obj->reduction_md)
		cxil_unmap(mc_obj->reduction_md);

	mc_obj->av_set->mc_obj = NULL;
	ofi_atomic_dec32(&mc_obj->ep_obj->coll.num_mc);
	free(mc_obj);

	return FI_SUCCESS;
}

/**
 * Utility routine to set up the collective framework. Any failures are reported
 * to all endpoints with C_RC_PTLTE_NOT_FOUND as the prov_errno.
 *
 * Caller must hold ep_obj->lock.
 */
static int _initialize_mc(void *ptr)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.do_space_check = 1,
		.en_restricted_unicast_lm = 1,
	};
	struct cxip_join_state *jstate = ptr;
	struct cxip_ep_obj *ep_obj = jstate->ep_obj;
	struct cxip_av_set *av_set = jstate->av_set;
	struct cxip_coll_mc *mc_obj;
	struct cxip_coll_pte *coll_pte;
	struct cxip_cmdq *cmdq;
	int red_id;
	int ret;

	TRACE_JOIN("%s entry\n", __func__);

	ret = -FI_ENOMEM;
	mc_obj = calloc(1, sizeof(*av_set->mc_obj));
	if (!mc_obj)
		goto fail;

	coll_pte = calloc(1, sizeof(*coll_pte));
	if (!coll_pte)
		goto fail;
	mc_obj->coll_pte = coll_pte;

	/* link ep_obj to mc_obj (1 to many) */
	mc_obj->ep_obj = ep_obj;
	ofi_atomic_inc32(&ep_obj->coll.num_mc);

	/* link av_set to mc_obj (one to one) */
	av_set->mc_obj = mc_obj;
	mc_obj->av_set = av_set;

	/* initialize coll_pte */
	coll_pte->ep_obj = ep_obj;
	coll_pte->mc_obj = mc_obj;
	dlist_init(&coll_pte->buf_list);
	ofi_atomic_initialize32(&coll_pte->buf_cnt, 0);
	ofi_atomic_initialize32(&coll_pte->buf_swap_cnt, 0);

	/* initialize mc_obj */
	mc_obj->mc_fid.fid.fclass = FI_CLASS_MC;
	mc_obj->mc_fid.fid.context = mc_obj;
	mc_obj->mc_fid.fid.ops = &mc_ops;
	mc_obj->mc_fid.fi_addr = (fi_addr_t)(uintptr_t)mc_obj;
	mc_obj->ep_obj = ep_obj;
	mc_obj->av_set = av_set;
	mc_obj->coll_pte = coll_pte;
	mc_obj->hwroot_idx = jstate->bcast_data.hwroot_idx;
	mc_obj->mcast_addr = jstate->bcast_data.mcast_addr;
	mc_obj->mynode_idx = jstate->mynode_idx;
	mc_obj->mynode_fiaddr = jstate->mynode_fiaddr;
	mc_obj->max_red_id = CXIP_COLL_MAX_CONCUR;
	mc_obj->arm_disable = false;
	mc_obj->timeout.tv_sec = 1;
	mc_obj->timeout.tv_nsec = 0;
	mc_obj->tc = CXI_TC_BEST_EFFORT;
	for (red_id = 0; red_id < CXIP_COLL_MAX_CONCUR; red_id++) {
		struct cxip_coll_reduction *reduction;

		reduction = &mc_obj->reduction[red_id];
		reduction->coll_state = CXIP_COLL_STATE_READY;
		reduction->mc_obj = mc_obj;
		reduction->red_id = red_id;
		reduction->in_use = false;
		reduction->completed = false;
	}
	ofi_spin_init(&mc_obj->lock);
	ofi_atomic_initialize32(&mc_obj->send_cnt, 0);
	ofi_atomic_initialize32(&mc_obj->recv_cnt, 0);
	ofi_atomic_initialize32(&mc_obj->pkt_cnt, 0);
	ofi_atomic_initialize32(&mc_obj->seq_err_cnt, 0);
	ofi_atomic_initialize32(&mc_obj->tmout_cnt, 0);

	/* map entire reduction block if using DMA */
	if (cxip_env.coll_use_dma_put) {
		/* EXPERIMENTAL */
		ret = cxil_map(ep_obj->domain->lni->lni,
			       mc_obj->reduction,
			       sizeof(mc_obj->reduction),
			       CXI_MAP_PIN  | CXI_MAP_READ | CXI_MAP_WRITE,
			       NULL, &mc_obj->reduction_md);
		if (ret)
			goto fail;
	}

	/* bind PTE to domain */
	ret = cxip_pte_alloc(ep_obj->if_dom, ep_obj->coll.rx_evtq->eq,
			     jstate->pid_idx, jstate->is_mcast, &pt_opts,
			     _coll_pte_cb, coll_pte, &coll_pte->pte);
	if (ret)
		goto fail;

	/* enable the PTE */
	ret = _coll_pte_enable(coll_pte, CXIP_PTE_IGNORE_DROPS);
	if (ret)
		goto fail;

	/* add buffers to the PTE */
	ret = _coll_add_buffers(coll_pte,
				ep_obj->coll.buffer_size,
				ep_obj->coll.buffer_count);
	if (ret)
		goto fail;

	/* define the traffic class */
	// TODO revisit for LOW_LATENCY
	if (is_hw_root(mc_obj))
		mc_obj->tc_type = CXI_TC_TYPE_DEFAULT;
	else if (is_netsim(ep_obj))
		mc_obj->tc_type = CXI_TC_TYPE_DEFAULT;
	else
		mc_obj->tc_type = CXI_TC_TYPE_COLL_LEAF;

	/* Set this now to instantiate cmdq CP */
	cmdq = ep_obj->coll.tx_cmdq;

	ret = cxip_txq_cp_set(cmdq, ep_obj->auth_key.vni,
			      mc_obj->tc, mc_obj->tc_type);

	if (ret)
		goto fail;

	/* Last to set */
	mc_obj->is_joined = true;

	jstate->mc_obj = mc_obj;
	*jstate->mc = &mc_obj->mc_fid;
	TRACE_JOIN("%s: initialized mc[%d] to %p\n",
		   __func__, jstate->simrank, *jstate->mc);

	return FI_SUCCESS;

fail:
	if (mc_obj)
		_close_mc(&mc_obj->mc_fid.fid);
	return ret;
}

/**
 * CURL callback function upon completion of a request.
 *
 * This sets jstate->created_mcast, even if the operation fails.
 * This sets jstate->bcast_data.valid if the address is valid.
 */
static void _cxip_create_mcast_cb(struct cxip_curl_handle *handle)
{
	struct cxip_curl_mcast_usrptr *curl_usrptr = handle->usrptr;
	struct cxip_join_state *jstate = curl_usrptr->jstate;
	struct json_object *json_obj;
	struct cxip_addr caddr;
	const char *hwrootstr;
	int mcaddr, hwroot;
	uint32_t b2, b1, b0, n;
	int i, ret;

	/* Creation process is done */
	TRACE_JOIN("CURL COMPLETED!\n");
	jstate->created_mcast = true;

	switch (handle->status) {
	case 200:
	case 201:
		/* CURL succeeded, parse response */
		TRACE_JOIN("CURL PARSE RESPONSE:\n%s\n", handle->response);
		if (!(json_obj = json_tokener_parse(handle->response)))
			break;
		if (cxip_json_int("mcastID", json_obj, &mcaddr))
			break;
		if (cxip_json_string("hwRoot", json_obj, &hwrootstr))
			break;

		n = sscanf(hwrootstr, "%x:%x:%x", &b2, &b1, &b0);
		if (n < 3 || b2 > 0xf || b1 > 0xff || b2 > 0xff)
			break;
		hwroot = (b2 << 16) + (b1 << 8) + b0;

		TRACE_JOIN("mcastID=%d hwRoot='%s'=%x\n", mcaddr, hwrootstr,
			   hwroot);
		for (i = 0; i < jstate->av_set->fi_addr_cnt; i++) {
			ret = _cxip_av_lookup(
					jstate->av_set->cxi_av,
					jstate->av_set->fi_addr_ary[i],
					&caddr);
			if (ret < 0)
				continue;
			TRACE_JOIN("test %d == %d\n", hwroot, caddr.nic);
			if (hwroot == caddr.nic)
				break;
		}
		TRACE_JOIN("final index=%d\n", i);
		if (i >= jstate->av_set->fi_addr_cnt) {
			TRACE_JOIN("multicast HWroot not found in av_set\n");
			break;
		}
		if (mcaddr >= 8192) {
			/* Flask test API: switch to UNICAST */
			jstate->bcast_data.valid = true;
			jstate->bcast_data.hwroot_idx = i;
			jstate->bcast_data.mcast_addr =
				jstate->ep_obj->src_addr.nic;
			jstate->is_mcast = false;
		} else {
			/* Production MCAST address */
			jstate->bcast_data.valid = true;
			jstate->bcast_data.hwroot_idx = i;
			jstate->bcast_data.mcast_addr = (uint32_t)mcaddr;
			jstate->is_mcast = true;
		}
		/* This succeeded */
		TRACE_JOIN("curl: mcaddr   =%08x\n",
			   jstate->bcast_data.mcast_addr);
		TRACE_JOIN("curl: hwrootidx=%d\n",
			   jstate->bcast_data.hwroot_idx);
		break;
	default:
		TRACE_JOIN("ERRMSK SET CURL error %ld!\n", handle->status);
		if (handle->response)
			TRACE_JOIN("ERROR RESPONSE:\n%s\n", handle->response);
		break;
	}
	free(curl_usrptr);
}

/**
 * Start a CURL request for a multicast address.
 *
 * This sets jstate->created_mcast if the operation does not start.
 * This jstate->bcast_data.valid will remain false.
 */
static void _start_curl(void *ptr)
{
	struct cxip_curl_mcast_usrptr *curl_usrptr;
	struct cxip_join_state *jstate = ptr;
	static const char *json_fmt =
		"{'macs':[%s],'jobID':'%s','jobStepID':'%s','timeout':%ld}";
	struct cxip_addr caddr;
	char *jsonreq, *mac, *p;
	char url[256];
	int i, len, ret;

	/* early exit will attempt to free these */
	curl_usrptr = NULL;
	jsonreq = NULL;
	mac = NULL;

	/* acquire the environment variables needed */
	TRACE_JOIN("jobid  = %s\n", cxip_env.coll_job_id);
	TRACE_JOIN("stepid = %s\n", cxip_env.coll_step_id);
	TRACE_JOIN("fmurl  = %s\n", cxip_env.coll_fabric_mgr_url);
	TRACE_JOIN("token  = %s\n", cxip_env.coll_fabric_mgr_token);
	TRACE_JOIN("tmout  = %ld\n", cxip_env.coll_timeout_usec);

	ret = -FI_EINVAL;
	if (!cxip_env.coll_job_id ||
	    !cxip_env.coll_fabric_mgr_url ||
	    !cxip_env.coll_fabric_mgr_token) {
		TRACE_JOIN("Check environment variables\n");
		goto quit;
	}

	len = snprintf(url, sizeof(url), "%s/fabric/collectives/multicast",
			cxip_env.coll_fabric_mgr_url);
	if (len >= sizeof(url)) {
		TRACE_JOIN("Failed to construct CURL address\n");
		goto quit;
	}

	/* five hex digits per mac, two colons, two quotes, comma */
	ret = -FI_ENOMEM;
	p = mac = malloc(10*jstate->av_set->fi_addr_cnt + 1);
	if (!mac)
		goto quit;
	for (i = 0; i < jstate->av_set->fi_addr_cnt; i++) {
		ret = _cxip_av_lookup(
				jstate->av_set->cxi_av,
				jstate->av_set->fi_addr_ary[i], &caddr);
		if (ret < 0)
			goto quit;
		p += sprintf(p, "'%01X:%02X:%02X',",
		 		(caddr.nic >> 16) & 0xf,
				(caddr.nic >> 8) & 0xff,
				(caddr.nic) & 0xff);

	}
	*(--p) = 0;

	/* generate the CURL JSON request */
	ret = asprintf(&jsonreq, json_fmt, mac,
			cxip_env.coll_job_id,
			cxip_env.coll_step_id,
			cxip_env.coll_timeout_usec);
	if (ret < 0)
		goto quit;
	single_to_double_quote(jsonreq);
	TRACE_JOIN("JSON = %s\n", jsonreq);

	/* create the mcast address */
	ret = -FI_ENOMEM;
	curl_usrptr = calloc(1, sizeof(*curl_usrptr));
	if (!curl_usrptr)
		goto quit;
	/* dispatch CURL request */
	curl_usrptr->jstate = jstate;
	if (cxip_trap_search(jstate->mynode_idx, CXIP_TRAP_CURLSND, &ret))
		goto quit;
	ret = cxip_curl_perform(url, jsonreq, cxip_env.coll_fabric_mgr_token, 0,
				CURL_POST, false, _cxip_create_mcast_cb,
				curl_usrptr);
quit:
	free(mac);
	free(jsonreq);
	if (ret < 0) {
		TRACE_JOIN("CURL execution failed\n");
		free(curl_usrptr);
		jstate->created_mcast = true;
		/* jstate->bcast_data.valid == false */
	}
}

/****************************************************************************
 * State machine for performing fi_join_collective()
 *
 * The zbcoll operations use unrestricted packets, will re-route dynamically,
 * and manage NAK retries automatically, so they are resistant to dropped
 * packets and other transient errors. They will not (and should not) time out:
 * a persistently unresponsive endpoint in the collective tree will cause the
 * collective join to block indefinitely.
 *
 * Each state operation returns without doing any retries. The state machine
 * progress table will decide whether to retry the operation.
 *
 * Each state operation must set zb->error as follows:
 * - FI_SUCCESS - continues the state machine
 * - FI_EAGAIN  - retries the same state
 * - other      - fails the join operation
 *
 * The prov_errno value is used to carry the state error, and is one of the
 * following:
 *
 * - C_RC_NO_EVENT              no error
 * - C_RC_INVALID_DFA_FORMAT    CURL multicast address request failed
 * - C_RC_PTLTE_NOT_FOUND       PtlTE could not be initialized
 */

/**
 * Join state machine.
 *
 * The state machine walks through the following functions in the order shown.
 * If the return code is success, it advances to the next state.
 * If the return code is -FI_EAGAIN, it repeats the current state.
 * If the return code is anything else, the join operation fails.
 */

/* append a jstate to the zbcoll scheduler */
static void _append_sched(struct cxip_zbcoll_obj *zb, void *usrptr)
{
	struct cxip_ep_coll_obj *coll_obj = &zb->ep_obj->coll;
	struct cxip_join_state *jstate = usrptr;

	dlist_ts_insert_tail(&coll_obj->sched_list, &jstate->sched_link);
}

static void _noop(void *ptr)
{
	TRACE_JOIN("%s: entry\n", __func__);
}

static void _start_getgroup(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;

	TRACE_JOIN("%s on %d: entry\n", __func__, jstate->mynode_idx);

	if (cxip_trap_search(jstate->mynode_idx, CXIP_TRAP_GETGRP, &zb->error))
		goto quit;
	/* zb->error == FI_SUCCESS, -FI_EAGAIN, -FI_EINVAL */
	zb->error = cxip_zbcoll_getgroup(zb);
quit:
	TRACE_JOIN("getgroup error = %d\n", zb->error);
	if (zb->error)
		_append_sched(zb, jstate);
}

static void _finish_getgroup(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;

	TRACE_JOIN("%s on %d: entry\n", __func__, jstate->mynode_idx);
	_append_sched(zb, jstate);	// _start_bcast
}

static void _start_bcast(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;

	TRACE_JOIN("%s: entry\n", __func__);

	/* error will indicate that the multicast request fails */
	jstate->prov_errno = C_RC_INVALID_DFA_FORMAT;
	/* presume bcast_data is valid from static initialization */
	jstate->bcast_data.valid = true;
 	/* at most one endpoint will have create_mcast == true */
	if (jstate->create_mcast) {
		/* first call (only) initiates CURL request */
		if (!jstate->creating_mcast) {
			/* presume bcast_data is invalid until created */
			jstate->bcast_data.valid = false;
			jstate->creating_mcast = true;
			_start_curl(jstate);
		}
		/* every call checks to see if CURL is complete */
		if (!jstate->created_mcast) {
			zb->error = -FI_EAGAIN;
			goto quit;
		}
	}
	if (cxip_trap_search(jstate->mynode_idx, CXIP_TRAP_BCAST, &zb->error))
		goto quit;
	/* zb->error == FI_SUCCESS, -FI_EAGAIN, -FI_EINVAL */
	zb->error = cxip_zbcoll_broadcast(zb, &jstate->bcast_data.uint64);
quit:
	if (zb->error)
		_append_sched(zb, jstate);
}

static void _finish_bcast(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;
	int ret;

	/* all NICs have same data, if invalid, fail */
	if (!jstate->bcast_data.valid)
		goto quit;
	/* error indicates that attempt to configure fails */
	jstate->prov_errno = C_RC_PTLTE_NOT_FOUND;
	if (cxip_trap_search(jstate->mynode_idx, CXIP_TRAP_INITPTE, &ret))
		goto quit;
	ret = _initialize_mc(jstate);
quit:
	/* if initialization fails, invalidate bcast_data */
	if (ret != FI_SUCCESS)
		jstate->bcast_data.valid = false;
	_append_sched(zb, jstate);	// _start_reduce
}

static void _start_reduce(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;

	/* reduce ANDs bcast_data, if any invalid, all become invalid */
	if (cxip_trap_search(jstate->mynode_idx, CXIP_TRAP_REDUCE, &zb->error))
		goto quit;
	/* zb->error == FI_SUCCESS, -FI_EAGAIN, -FI_EINVAL */
	zb->error = cxip_zbcoll_reduce(zb, &jstate->bcast_data.uint64);
quit:
	if (zb->error)
		_append_sched(zb, jstate);
}

static void _finish_reduce(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;

	TRACE_JOIN("%s: entry\n", __func__);

	/* if bcast_data is valid, we can clear the error */
	if (jstate->bcast_data.valid)
		jstate->prov_errno = C_RC_NO_EVENT;

	TRACE_JOIN("%s: prov_errno=0x%x\n", __func__, jstate->prov_errno);
	_append_sched(zb, jstate);	// _start_cleanup
}

static void _start_cleanup(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct fi_eq_err_entry entry = {};
	size_t size = sizeof(entry);
	uint64_t flags = 0L;
	int ret;

	TRACE_JOIN("%s: entry\n", __func__);
	if (jstate) {
		entry.fid = (jstate->mc_obj) ?
				&jstate->mc_obj->mc_fid.fid : NULL;
		entry.context = jstate->context;

		if (jstate->prov_errno != C_RC_NO_EVENT) {
			size = sizeof(struct fi_eq_err_entry);
			entry.data = FI_JOIN_COMPLETE;
			entry.err = -FI_EAVAIL;
			entry.prov_errno = jstate->prov_errno;
			flags |= UTIL_FLAG_ERROR;
		}
		ret = ofi_eq_write(&jstate->ep_obj->eq->util_eq.eq_fid,
				FI_JOIN_COMPLETE, &entry,
				size, flags);
		if (ret < 0)
			CXIP_INFO("FATAL ERROR: cannot post to EQ\n");
		cxip_zbcoll_free(jstate->zb);
	}
	free(jstate);
}

typedef	void (*sched_func)(void *ptr);

enum state_code {
	state_init,
	start_getgroup,
	finish_getgroup,
	start_bcast,
	finish_bcast,
	start_reduce,
	finish_reduce,
	start_cleanup,
	state_done
};

const char *state_name[] = {
	"state_init",
	"start_getgroup",
	"finish_getgroup",
	"start_bcast",
	"finish_bcast",
	"start_reduce",
	"finish_reduce",
	"start_cleanup",
	"state_done"
};
sched_func state_func[] = {
	_noop,
	_start_getgroup,
	_finish_getgroup,
	_start_bcast,
	_finish_bcast,
	_start_reduce,
	_finish_reduce,
	_start_cleanup,
	_noop,
};

/**
 * State progression table
 *
 * Row is the current state.
 * Col contains states reachable from this state on success/again/fail.
 */
static enum state_code progress_state[][3] = {
	/* STATE              SUCCESS         EAGAIN         FAIL */
	/* state_init     */ {start_getgroup, start_cleanup, start_cleanup},
	/* start_getgroup */ {finish_getgroup,start_getgroup,start_cleanup},
	/* finish_getgroup*/ {start_bcast,    start_cleanup, start_cleanup},
	/* start_bcast    */ {finish_bcast,   start_bcast,   start_cleanup},
	/* finish_bcast   */ {start_reduce,   start_cleanup, start_cleanup},
	/* start_reduce   */ {finish_reduce,  start_reduce,  start_cleanup},
	/* finish_reduce  */ {start_cleanup,  start_cleanup, start_cleanup},
	/* start_cleanup  */ {state_done,     state_done,    state_done},
	/* state_done     */ {state_done,     state_done,    state_done},
};

/* Advance the state and run scheduled operations */
static void _progress_sched(struct cxip_join_state *jstate)
{
	struct cxip_zbcoll_obj *zb = jstate->zb;
	enum state_code *codes;

	TRACE_JOIN("entry jstate[%d]=%s, error=%d\n",
		jstate->simrank, state_name[jstate->sched_state], zb->error);

	/* acquire the success/again/fail state codes for current state */
	codes = progress_state[jstate->sched_state];
	switch (zb->error) {
	case FI_SUCCESS:
		/* last operation succeeded */
		jstate->sched_state = codes[0];
		break;
	case -FI_EBUSY:
	case -FI_EAGAIN:
		/* last operation needs a retry */
		TRACE_JOIN("busy retry\n");
		jstate->sched_state = codes[1];
		break;
	default:
		/* last operation failed */
		jstate->sched_state = codes[2];
		break;
	}
	TRACE_JOIN("----> jstate[%d]=%s\n",
		jstate->simrank, state_name[jstate->sched_state]);

	/* execute the new state function */
	state_func[jstate->sched_state](jstate);
}

/* Process the schedule list and dispatch next scheduled operation */
static void _progress_join(struct cxip_ep_obj *ep_obj)
{
	struct cxip_ep_coll_obj *coll_obj = &ep_obj->coll;
	struct cxip_join_state *jstate = NULL;

	dlist_ts_pop_front(&coll_obj->sched_list,
			   struct cxip_join_state,
			   jstate, sched_link);

	if (jstate)
		_progress_sched(jstate);
}

/* During join, determine my index position in the av_set */
static unsigned int _caddr_to_idx(struct cxip_av_set *av_set,
				  struct cxip_addr caddr)
{
	struct cxip_addr addr;
	size_t size = sizeof(addr);
	int i, ret;

	for (i = 0; i < av_set->fi_addr_cnt; i++) {
		ret = fi_av_lookup(&av_set->cxi_av->av_fid,
				   av_set->fi_addr_ary[i],
				   &addr, &size);
		if (ret)
			return ret;
		if (CXIP_ADDR_EQUAL(addr, caddr))
			return i;
	}
	return -FI_EADDRNOTAVAIL;
}

/**
 * fi_join_collective() implementation.
 *
 * Calling syntax is defined by libfabric.
 *
 * This is a multi-stage collective operation, progressed by calling TX/RX CQs
 * and the EQ for the endpoint.
 *
 * We go through the following steps:
 *
 * 1) allocate a join state for this operation
 * 2) allocate zbcoll object
 * 3) get a collective group identifier
 * 4) generate a multicast tree from NIC 0
 * 5) broadcast multicast address from NIC 0
 * 6) reduce error mask across all NICs
 * 7) cleanup
 *
 * This needs to be a non-blocking process, to support concurrent joins that are
 * driven by CQ/EQ polling. If the initial fi_join_collective() call returns
 * success, meaning the operation was initiated, actual completion success or
 * failure must be captured through EQ polling.
 *
 * Any transient errors associated with collective communications should be
 * retried internally, and indefinitely: once into the join state machine, there
 * is no reasonable way to re-enter, and if there is a synchronization delay,
 * there is no "reasonable" upper bound on the delay. If a NIC is unresponsive,
 * for instance, this will wait forever.
 *
 * Non-transient errors, such as failure to create the multicast address, or
 * setting up the multicast support, are distributed to all NICs through the
 * final zbcoll reduce function, and result in an immediate failure of the join
 * across all NICs.
 *
 * There are four operational models, one for production, and three for testing.
 *
 * In all cases, there must be one join for every NIC address in the av_set
 * fi_addr_ary, and the collective proceeds among these joined endpoints.
 *
 * COMM_KEY_RANK tests using a single process on a single Cassini, which
 * supplies the src/tgt, but different pid_idx values, representing different
 * PTLTE objects, each with its own buffers. The zbcoll operations are performed
 * using linked zb objects, which represent a single zbcoll collective, so each
 * zb callback function is called only once for the entire set, yet must provide
 * a unique mc return value and FI_COLL_COMPLETE event for each joined object.
 * We manage this with the simstates array, which associates the simulated rank
 * with the state pointer, so that upon completion, we can provide all of the
 * return pointers and events.
 *
 * COMM_KEY_UNICAST tests on multiple nodes on a real network, but without any
 * multicast support. It initializes one mc object on each node, and designates
 * the first node in the multicast list, fiaddr[0], as the hardware root node.
 * fiaddr[1..N] send directly to fiaddr[0], and fiaddr[0] sends to each of the
 * other addresses in a simulated broadcast. This is not expected to be
 * performant, but it does exercise a necessary incast edge case, and it fully
 * exercises the collectives software across multiple nodes.
 *
 * COMM_KEY_MULTICAST is a fully-functioning model, but requires that an
 * external application prepare the multicast address on the fabric before
 * calling fi_join_collective() on any node. This information must be supplied
 * through the av_set->comm_key structure.
 *
 * COMM_KEY_NONE is the production model, in which fi_join_collective() creates
 * the multicast address by making a CURL call to the fabric manager REST API.
 * fiaddr[0] manages the CURL call, and broadcasts the results to all of the
 * other objects across the collective group.
 */
int cxip_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			 const struct fid_av_set *coll_av_set,
			 uint64_t flags, struct fid_mc **mc, void *context)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_av_set *av_set;
	struct cxip_join_state *jstate;
	struct cxip_zbcoll_obj *zb;
	bool link_zb;
	int ret;

	check_red_pkt();

	TRACE_JOIN("%s: entry\n", __func__);
	/* Validate arguments */
	if (!ep || !coll_av_set || !mc || coll_addr != FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;
	/* flags are ignored, per util_coll.c example code
	 * Only FI_SCATTER is documented, and applies to fi_query_collective().
	 */

	cxip_ep = container_of(ep, struct cxip_ep, ep.fid);
	av_set = container_of(coll_av_set, struct cxip_av_set, av_set_fid);

	ep_obj = cxip_ep->ep_obj;

	/* allocate state to pass arguments through callbacks */
	jstate = calloc(1, sizeof(*jstate));
	if (! jstate)
		return -FI_ENOMEM;
	/* all errors after this must goto fail for cleanup */

	jstate->ep_obj = ep_obj;
	jstate->av_set = av_set;
	jstate->mc = mc;
	jstate->context = context;
	jstate->join_flags = flags;
	jstate->sched_state = state_init;

	/* rank 0 (av_set->fi_addr_cnt[0]) does zb broadcast, so all nodes will
	 * share whatever bcast_data rank 0 ends up with.
	 */

	ret = -FI_EINVAL;
	switch (av_set->comm_key.keytype) {
	case COMM_KEY_NONE:
		/* Production case, acquire multicast from FM */
		if (is_netsim(ep_obj)) {
			CXIP_INFO("NETSIM COMM_KEY_NONE not supported\n");
			goto fail;
		}
		TRACE_JOIN("%s: MULTICAST CURL model setup\n", __func__);
		jstate->mynode_idx =
			_caddr_to_idx(av_set, ep_obj->src_addr);
		jstate->mynode_fiaddr =
			av_set->fi_addr_ary[jstate->mynode_idx];
		jstate->simrank = ZB_NOSIM;
		jstate->pid_idx = CXIP_PTL_IDX_COLL;
		jstate->bcast_data.hwroot_idx = 0;
		jstate->bcast_data.mcast_addr = 0;
		jstate->bcast_data.valid = false;
		jstate->is_mcast = true;
		jstate->create_mcast = (jstate->mynode_idx == 0);
		link_zb = false;
		break;
	case COMM_KEY_MULTICAST:
		/* Real network test with predefined multicast address */
		if (is_netsim(ep_obj)) {
			CXIP_INFO("NETSIM COMM_KEY_MULTICAST not supported\n");
			goto fail;
		}
		TRACE_JOIN("%s: MULTICAST prefab model setup\n", __func__);
		jstate->mynode_idx =
			_caddr_to_idx(av_set, ep_obj->src_addr);
		jstate->mynode_fiaddr =
			av_set->fi_addr_ary[jstate->mynode_idx];
		jstate->simrank = ZB_NOSIM;
		jstate->pid_idx = CXIP_PTL_IDX_COLL;
		jstate->bcast_data.hwroot_idx =
			av_set->comm_key.mcast.hwroot_idx;
		jstate->bcast_data.mcast_addr =
			av_set->comm_key.mcast.mcast_addr;
		jstate->bcast_data.valid = true;
		jstate->is_mcast = true;
		jstate->create_mcast = false;
		link_zb = false;
		break;
	case COMM_KEY_UNICAST:
		/* Real network test without multicast address */
		if (is_netsim(ep_obj)) {
			CXIP_INFO("NETSIM COMM_KEY_UNICAST not supported\n");
			goto fail;
		}
		TRACE_JOIN("%s: UNICAST model setup\n", __func__);
		jstate->mynode_idx =
			_caddr_to_idx(av_set, ep_obj->src_addr);
		jstate->mynode_fiaddr =
			av_set->fi_addr_ary[jstate->mynode_idx];
		jstate->simrank = ZB_NOSIM;
		jstate->pid_idx = CXIP_PTL_IDX_COLL;
		jstate->bcast_data.hwroot_idx =
			av_set->comm_key.ucast.hwroot_idx;
		jstate->bcast_data.mcast_addr = ep_obj->src_addr.nic;
		jstate->bcast_data.valid = true;
		jstate->is_mcast = false;
		jstate->create_mcast = false;
		link_zb = false;
		break;
	case COMM_KEY_RANK:
		/* Single process simulation, can run under NETSIM */
		TRACE_JOIN("%s: COMM_KEY_RANK detected\n", __func__);
		jstate->mynode_idx = av_set->comm_key.rank.rank;
		jstate->mynode_fiaddr = (fi_addr_t)jstate->mynode_idx;
		jstate->simrank = jstate->mynode_idx;
		jstate->pid_idx = CXIP_PTL_IDX_COLL + jstate->simrank;
		jstate->bcast_data.hwroot_idx = 0;
		jstate->bcast_data.mcast_addr = ep_obj->src_addr.nic;
		jstate->bcast_data.valid = true;
		jstate->is_mcast = false;
		jstate->create_mcast = false;
		link_zb = true;
		break;
	default:
		CXIP_INFO("unexpected comm_key keytype: %d\n",
			  av_set->comm_key.keytype);
		goto fail;
	}

	/* Acquire a zbcoll identifier */
	TRACE_JOIN("%s: allocate zb\n", __func__);
	ret = cxip_zbcoll_alloc(jstate->ep_obj,
				jstate->av_set->fi_addr_cnt,
				jstate->av_set->fi_addr_ary,
				jstate->simrank, &zb);
	TRACE_JOIN("%s: returned=%d\n", __func__, ret);
	if (ret)
		goto fail;

	/* Install the callback function for zb collectives */
	TRACE_JOIN("%s: cxip_zbcoll_set_user_cb\n", __func__);
	cxip_zbcoll_set_user_cb(zb, _append_sched, jstate);

	/* If COMM_KEY_RANK, join is called for each rank */
	if (link_zb) {
		static struct cxip_zbcoll_obj *zb0 = NULL;
		static int zb0_count = 0;
		int rank = av_set->comm_key.rank.rank;

		/* first call sets the zb0 simulated endpoint */
		TRACE_JOIN("%s: rank = %d, zb0_count=%d\n", __func__, rank, zb0_count);
		if (!zb0_count++) {
			/* first must be rank 0 */
			if (rank != 0) {
				TRACE_JOIN("%s: rank %d not 0\n", __func__, rank);
				ret = -FI_EINVAL;
				goto fail;
			}
			zb0 = zb;
			TRACE_JOIN("%s: zb0=%p zb=%p\n", __func__, zb0, zb);
		}
		/* link this zb to zb0 */
		ret = cxip_zbcoll_simlink(zb0, zb);
		if (ret) {
			TRACE_JOIN("%s: return=%d\n", __func__, ret);
			return ret;
		}
		/* after the last, we need to reset this */
		if (zb0_count == av_set->fi_addr_cnt) {
			zb0_count = 0;
			zb0 = NULL;
		}
	}

	jstate->zb = zb;
	_append_sched(zb, jstate);

	return FI_SUCCESS;

fail:
	/* this path returns error, does not post to EQ */
	TRACE_JOIN("fail cxip_join_collective\n");
	cxip_zbcoll_free(zb);
	free(jstate);

	return ret;
}

/* Exported to be called by EQ read function */
void cxip_coll_progress_join(struct cxip_ep_obj *ep_obj)
{
	ofi_genlock_lock(&ep_obj->lock);

	/* progress the work schedule */
	_progress_join(ep_obj);

	/* don't want handle returned, callback function manages it */
	cxip_curl_progress(NULL);

	/* progress the underlying zbcoll */
	cxip_ep_zbcoll_progress(ep_obj);

	ofi_genlock_unlock(&ep_obj->lock);
}

/* Reset all of the diagnostic counters */
void cxip_coll_reset_mc_ctrs(struct fid_mc *mc)
{
	struct cxip_coll_mc *mc_obj = (struct cxip_coll_mc *)mc;

	ofi_atomic_set32(&mc_obj->send_cnt, 0);
	ofi_atomic_set32(&mc_obj->recv_cnt, 0);
	ofi_atomic_set32(&mc_obj->pkt_cnt, 0);
	ofi_atomic_set32(&mc_obj->seq_err_cnt, 0);
	ofi_atomic_set32(&mc_obj->tmout_cnt, 0);
}

/****************************************************************************
 * Manage the static coll structure in the EP. Because of its specialized
 * nature, it made sense to manage it here, rather than in the EP module.
 */

struct fi_ops_collective cxip_collective_ops = {
	.size = sizeof(struct fi_ops_collective),
	.barrier = cxip_barrier,
	.broadcast = cxip_broadcast,
	.alltoall = fi_coll_no_alltoall,
	.allreduce = cxip_allreduce,
	.allgather = fi_coll_no_allgather,
	.reduce_scatter = fi_coll_no_reduce_scatter,
	.reduce = cxip_reduce,
	.scatter = fi_coll_no_scatter,
	.gather = fi_coll_no_gather,
	.msg = fi_coll_no_msg,
};

struct fi_ops_collective cxip_collective_no_ops = {
	.size = sizeof(struct fi_ops_collective),
	.barrier = fi_coll_no_barrier,
	.broadcast = fi_coll_no_broadcast,
	.alltoall = fi_coll_no_alltoall,
	.allreduce = fi_coll_no_allreduce,
	.allgather = fi_coll_no_allgather,
	.reduce_scatter = fi_coll_no_reduce_scatter,
	.reduce = fi_coll_no_reduce,
	.scatter = fi_coll_no_scatter,
	.gather = fi_coll_no_gather,
	.msg = fi_coll_no_msg,
};


/* Close collectives - call during EP close */
void cxip_coll_close(struct cxip_ep_obj *ep_obj)
{
}

/**
 * Initialize collectives - call during EP init */
void cxip_coll_init(struct cxip_ep_obj *ep_obj)
{
	cxip_coll_populate_opcodes();

	dlist_ts_init(&ep_obj->coll.sched_list);
	ep_obj->coll.rx_cmdq = NULL;
	ep_obj->coll.tx_cmdq = NULL;
	ep_obj->coll.rx_cntr = NULL;
	ep_obj->coll.tx_cntr = NULL;
	ep_obj->coll.rx_evtq = NULL;
	ep_obj->coll.tx_evtq = NULL;
	ep_obj->coll.min_multi_recv = CXIP_COLL_MIN_FREE;
	ep_obj->coll.buffer_count = CXIP_COLL_MIN_RX_BUFS;
	ep_obj->coll.buffer_size = CXIP_COLL_MIN_RX_SIZE;

	ofi_atomic_initialize32(&ep_obj->coll.num_mc, 0);
}

/**
 * Enable collectives - call from EP enable.
 */
int cxip_coll_enable(struct cxip_ep *ep)
{
	struct cxip_ep_obj *ep_obj = ep->ep_obj;

	if (ep_obj->coll.enabled)
		return FI_SUCCESS;

	if (!(ep_obj->caps & FI_COLLECTIVE)) {
		CXIP_INFO("FI_COLLECTIVE not requested\n");
		return FI_SUCCESS;
	}

	/* A read-only or write-only endpoint is legal */
	if (!(ofi_recv_allowed(ep_obj->rxc.attr.caps) &&
	      ofi_send_allowed(ep_obj->txc.attr.caps))) {
		CXIP_INFO("EP not recv/send, collectives not enabled\n");
		return FI_SUCCESS;
	}

	/* Sanity checks */
	if (ep_obj->coll.buffer_size == 0)
		return -FI_EINVAL;
	if (ep_obj->coll.buffer_count == 0)
		return -FI_EINVAL;
	if (ep_obj->coll.min_multi_recv == 0)
		return -FI_EINVAL;
	if (ep_obj->coll.min_multi_recv >= ep_obj->coll.buffer_size)
		return -FI_EINVAL;

	/* Bind all STD EP objects to the coll object */
	ep_obj->coll.rx_cmdq = ep_obj->rxc.rx_cmdq;
	ep_obj->coll.tx_cmdq = ep_obj->txc.tx_cmdq;
	ep_obj->coll.rx_cntr = ep_obj->rxc.recv_cntr;
	ep_obj->coll.tx_cntr = ep_obj->txc.send_cntr;
	ep_obj->coll.rx_evtq = &ep_obj->rxc.rx_evtq;
	ep_obj->coll.tx_evtq = &ep_obj->txc.tx_evtq;
	ep_obj->coll.eq = ep_obj->eq;

	ep->ep.collective = &cxip_collective_ops;
	ep_obj->coll.enabled = true;

	return FI_SUCCESS;
}

/* Disable collectives - call from EP disable */
int cxip_coll_disable(struct cxip_ep_obj *ep_obj)
{
	if (!ep_obj->coll.enabled)
		return FI_SUCCESS;

	ep_obj->coll.enabled = false;
	ep_obj->coll.rx_cmdq = NULL;
	ep_obj->coll.tx_cmdq = NULL;
	ep_obj->coll.rx_cntr = NULL;
	ep_obj->coll.tx_cntr = NULL;
	ep_obj->coll.rx_evtq = NULL;
	ep_obj->coll.tx_evtq = NULL;
	ep_obj->coll.eq = NULL;

	return FI_SUCCESS;
}
