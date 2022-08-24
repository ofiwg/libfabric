/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2020-2022 Cray Inc. All rights reserved.
 */

/* Support for Restricted Nomatch Put.
 */


/****************************************************************************
 * Environment variables (provisional)
 *
 * CXIP_COLL_USE_DMA_PUT=1 causes DMA Put to be used, instead of IDC Put
 *
 * CXIP_COLL_DUMP_PKT=1 causes all packets to be displayed to stdout
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

#include <ofi_list.h>
#include <ofi.h>
#include <fenv.h>
#include <xmmintrin.h>

#include "cxip.h"

/* Undefine this to remove development code */
#undef	DEVELOPER

#ifdef DEVELOPER
#define	IDX	reduction->mc_obj->mynode_index
#define	PRT(fmt, ...)	printf("%d: %-16s " fmt, IDX, __func__, ## __VA_ARGS__)
#else	/* not DEVELOPER */
#define	PRT(fmt, ...)
#endif	/* DEVELOPER */

/* set cxip_trace_fn=printf to enable */
#define	trc CXIP_TRACE

#define	MAGIC		0x1776

// TODO regularize usage of these
#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

static ssize_t _coll_append_buffer(struct cxip_coll_pte *coll_pte,
				   struct cxip_coll_buf *buf);

/****************************************************************************
 * Reduction packet for Cassini:
 *
 *  +----------------------------------------------------------+
 *  | BYTES | Mnemonic    | Definition                         |
 *  +----------------------------------------------------------+
 *  | 48:17 | RED_PAYLOAD | Reduction payload, always 32 bytes |
 *  | 16:5  | RED_HDR     | Reduction Header (below)           |
 *  | 4:0   | RED_PADDING | Padding                            |
 *  +----------------------------------------------------------+
 *
 *  Reduction header format, from Table 95 in CSDG 5.7.2, Table 24 in RSDG
 *  4.5.9.4:
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
 *  Note that this is a 12-byte object, and "network-defined order" means
 *  big-endian for the entire 12-byte object. Thus, bytes must be swapped so
 *  that the MSByte of rt_cookie appears at byte 0, and the LS 8 bits of
 *  rt_seqno appear in byte 11.
 */

/**
 * Collective cookie structure.
 *
 * mcast_id is not necessary given one PTE per multicast address: all request
 * structures used for posting receive buffers will receive events from
 * only that multicast. If underlying drivers are changed to allow a single PTE
 * to be mapped to multiple multicast addresses, the mcast_id field will be
 * needed to disambiguate packets.
 *
 * red_id is needed to disambiguate packets delivered for different concurrent
 * reductions.
 *
 * magic is a magic number used to identify this packet as a reduction packet.
 * The basic send/receive code can be used for other kinds of restricted IDC
 * packets.
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

/**
 * Packed header bits and cookie from above.
 */
struct cxip_coll_hdr {
        uint64_t seqno:10;
        uint64_t arm:1;
        uint64_t op:6;
        uint64_t redcnt:20;
        uint64_t resno:10;
        uint64_t red_rc:4;
        uint64_t repsum_m:8;
        uint64_t repsum_ovfl:2;
        uint64_t pad:3;
        struct cxip_coll_cookie cookie;
} __attribute__((__packed__));		/* size 12b */

/**
 * The following structure is 49 bytes in size, and all of the fields align
 * properly.
 */
struct red_pkt {
	uint8_t pad[5];			/* size  5b offset  0b */
	struct cxip_coll_hdr hdr;	/* size 12b offset  5b */
	union cxip_coll_data data;	/* size 32b offset 17b */
} __attribute__((__packed__));		/* size 49b */


/**
 * Swap byte order in an object of any size. Works for even or odd counts.
 *
 * @param ptr : pointer to a stream of bytes
 * @param count : number of bytes to swap
 */
static inline void _swapbyteorder(void *ptr, int count)
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
 * @param pkt : reduction packet
 */
static inline void _swappkt(struct red_pkt *pkt)
{
#if (BYTE_ORDER == LITTLE_ENDIAN)
	int i;
	_swapbyteorder(&pkt->hdr, sizeof(pkt->hdr));
	for (i = 0; i < 4; i++)
		_swapbyteorder(&pkt->data.ival[i], 8);
#else
#error "Unsupported processor byte ordering"
#endif
}

#ifdef DEVELOPER
/**
 * Verificaton of the packet structure, normally disabled. Sizes and offsets
 * cannot be checked at compile time.
 */
#define	FLDOFFSET(base, fld)	((uint8_t *)&base.fld - (uint8_t *)&base)
static inline int check_red_pkt(void)
{
	static int checked;
	struct red_pkt pkt;
	uint64_t len, exp;
	uint8_t *ptr;
	int i, err = 0;

	if (checked)
		return 0;
	checked = 1;

	len = sizeof(pkt);
	exp = 49;
	if (len != exp) {
		printf("sizeof(pkt) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = sizeof(pkt.pad);
	exp = 5;
	if (len != exp) {
		printf("sizeof(pkt.pad) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = sizeof(pkt.hdr);
	exp = 12;
	if (len != exp) {
		printf("sizeof(pkt.hdr) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = sizeof(pkt.data);
	exp = 32;
	if (len != exp) {
		printf("sizeof(pkt.data) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = FLDOFFSET(pkt, hdr);
	exp = 5;
	if (len != exp) {
		printf("offset(pkt.hdr) = %ld, exp %ld\n", len, exp);
		err++;
	}
	len = FLDOFFSET(pkt, data);
	exp = 17;
	if (len != exp) {
		printf("offset(pkt.data) = %ld, exp %ld\n", len, exp);
		err++;
	}

	ptr = (uint8_t *)&pkt;
	for (i = 0; i < sizeof(pkt); i++)
		ptr[i] = i + 13;
	_swappkt(&pkt);
	_swappkt(&pkt);
	for (i = 0; i < sizeof(pkt); i++)
		if (ptr[i] != i + 13) {
			printf("pkt[%d] = %d, exp %d\n", i, ptr[i], i + 13);
			err++;
		}

	if (err)
		printf("*** Structures are wrong, cannot continue ***\n");
	else
		printf("structures are properly aligned\n");
	fflush(stdout);

	return err;
}
#else	/* not DEVELOPER */
static inline int check_red_pkt(void) { return 0; }
#endif	/* DEVELOPER */

#ifdef	DEVELOPER
__attribute__((unused))
static void _dump_red_data(const void *buf, const char *hdr)
{
	const union cxip_coll_data *data = (const union cxip_coll_data *)buf;
	int i;
	if (hdr)
		printf("%s\n", hdr);
	for (i = 0; i < 4; i++)
		printf("  ival[%d]     = %016lx\n", i, data->ival[i]);
}

#if 0
static void _reset_mc_ctrs(struct cxip_coll_mc *mc_obj)
{

}
#endif

__attribute__((unused))
static void _dump_red_pkt(struct red_pkt *pkt, char *dir)
{
	if (!getenv("CXIP_COLL_DUMP_PKT"))	// TODO remove or make cxi env
		return;

	printf("---------------\n");
	printf("Reduction packet (%s):\n", dir);
	printf("  seqno       = %d\n", pkt->hdr.seqno);
	printf("  arm         = %d\n", pkt->hdr.arm);
	printf("  op          = %d\n", pkt->hdr.op);
	printf("  redcnt      = %d\n", pkt->hdr.redcnt);
	printf("  resno       = %d\n", pkt->hdr.resno);
	printf("  red_rc      = %d\n", pkt->hdr.red_rc);
	printf("  repsum_m    = %d\n", pkt->hdr.repsum_m);
	printf("  repsum_ovfl = %d\n", pkt->hdr.repsum_ovfl);
	printf("  cookie --\n");
	printf("   .mcast_id  = %08x\n", pkt->hdr.cookie.mcast_id);
	printf("   .red_id    = %08x\n", pkt->hdr.cookie.red_id);
	printf("   .magic     = %08x\n", pkt->hdr.cookie.magic);
	printf("   .retry     = %08x\n", pkt->hdr.cookie.retry);
	_dump_red_data(pkt->data.databuf, NULL);
	printf("---------------\n");
	fflush(stdout);
}
#else	/* not DEVELOPER */
__attribute__((unused))
static void _dump_red_pkt(struct red_pkt *pkt, char *dir) {}
#endif	/* DEVELOPER */

/****************************************************************************
 * Static conversions, initialized once at at startup.
 */

/**
 * Opcodes for collective operations supported by Rosetta.
 *
 * CXI opcode implies data type.
 */

#define	COLL_OPCODE_BARRIER		0x00
#define	COLL_OPCODE_BIT_AND		0x01
#define	COLL_OPCODE_BIT_OR		0x02
#define	COLL_OPCODE_BIT_XOR		0x03
#define	COLL_OPCODE_INT_MIN		0x10
#define	COLL_OPCODE_INT_MAX		0x11
#define	COLL_OPCODE_INT_MINMAXLOC	0x12
#define	COLL_OPCODE_INT_SUM		0x14
#define	COLL_OPCODE_FLT_MIN		0x20
#define	COLL_OPCODE_FLT_MAX		0x21
#define	COLL_OPCODE_FLT_MINMAXLOC	0x22
#define	COLL_OPCODE_FLT_MINNUM		0x24
#define	COLL_OPCODE_FLT_MAXNUM		0x25
#define	COLL_OPCODE_FLT_MINMAXNUMLOC	0x26
#define	COLL_OPCODE_FLT_SUM_NOFTZ_RND0	0x28
#define	COLL_OPCODE_FLT_SUM_NOFTZ_RND1	0x29
#define	COLL_OPCODE_FLT_SUM_NOFTZ_RND2	0x2a
#define	COLL_OPCODE_FLT_SUM_NOFTZ_RND3	0x2b
#define	COLL_OPCODE_FLT_SUM_FTZ_RND0	0x2c
#define	COLL_OPCODE_FLT_SUM_FTZ_RND1	0x2d
#define	COLL_OPCODE_FLT_SUM_FTZ_RND2	0x2e
#define	COLL_OPCODE_FLT_SUM_FTZ_RND3	0x2f
#define	COLL_OPCODE_FLT_REPSUM		0x30

/* Convert exported op values to Rosetta opcodes */
static unsigned int _int8_16_32_op_to_opcode[CXI_FI_OP_LAST];
static unsigned int _int64_op_to_opcode[CXI_FI_OP_LAST];
static unsigned int _flt_op_to_opcode[CXI_FI_OP_LAST];

/* One-time dynamic initialization of FI to CXI opcode.
 *
 * The array lookup is faster than a switch. Non-static initialization makes
 * this adaptive to changes in header files (e.g. new opcodes in FI).
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
	if (cxip_env.coll_use_repsum) {
		_flt_op_to_opcode[FI_SUM] = COLL_OPCODE_FLT_REPSUM;
	} else {
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
	}
}

/* Convert FI opcode to CXI opcode, depending on FI data type */
int cxip_fi2cxi_opcode(int op, int datatype)
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

/* Determine datatype size */
static inline int _get_cxi_datasize(enum fi_datatype datatype, size_t count)
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

/* Return true if this node is the hwroot node.
 */
static inline bool is_hw_root(struct cxip_coll_mc *mc_obj)
{
	return (mc_obj->hwroot_index == mc_obj->mynode_index);
}

/****************************************************************************
 * SEND operation (restricted IDC Put to a remote PTE)
 */

static void _progress_coll(struct cxip_coll_reduction *reduction,
			   struct red_pkt *pkt);

/* Generate a dfa and index extension for a reduction.
 */
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
		/* - dest_addr == multicast ID
		 * - idx_ext == 0
		 * - dfa == multicast destination
		 * - index_ext == 0
		 */
		if (is_netsim(ep_obj)) {
			CXIP_INFO("NETSIM does not support mcast\n");
			return -FI_EINVAL;
		}
		idx_ext = 0;
		cxi_build_mcast_dfa(av_set->comm_key.mcast.mcast_id,
				    reduction->red_id, idx_ext,
				    dfa, index_ext);
		*is_mcast = true;
		break;
	case COMM_KEY_UNICAST:
		/* - dest_addr == destination AV index
		 * - idx_ext == CXIP_PTL_IDX_COLL
		 * - dfa = remote nic
		 * - index_ext == CXIP_PTL_IDX_COLL
		 */
		if (av_set_idx >= av_set->fi_addr_cnt) {
			CXIP_INFO("av_set_idx out-of-range\n");
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
		/* - dest_addr == multicast object index
		 * - idx_ext == multicast object index
		 * - dfa == source NIC
		 * - index_ext == idx_ext offset beyond RXCs
		 */
		if (av_set_idx >= av_set->fi_addr_cnt) {
			CXIP_INFO("av_set_idx out-of-range\n");
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
		CXIP_INFO("unexpected comm_key type: %d\n",
			  av_set->comm_key.keytype);
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

/**
 * Issue a restricted Put to the destination address.
 *
 * If md is NULL, this will perform an IDC Put, otherwise it will issue a DMA
 * Put.
 *
 * Exported for unit testing.
 *
 * @param reduction - reduction object
 * @param av_set_idx - index of address in av_set
 * @param buffer - buffer containing data to send
 * @param buflen - byte count of data in buffer
 * @param md - IOVA memory descriptor, or NULL
 *
 * @return int - return code
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

	if (cxip_cq_saturated(ep_obj->coll.tx_cq)) {
		CXIP_DBG("CQ saturated\n");
		return -FI_EAGAIN;
	}

	if (md) {
		cmd.full_dma.command.opcode = C_CMD_PUT;
		cmd.full_dma.event_send_disable = 1;
		cmd.full_dma.event_success_disable = 1;
		cmd.full_dma.restricted = 1;
		cmd.full_dma.reduction = is_mcast;
		cmd.full_dma.index_ext = index_ext;
		cmd.full_dma.eq = cxip_cq_tx_eqn(ep_obj->coll.tx_cq);
		cmd.full_dma.dfa = dfa;
		cmd.full_dma.lac = md->lac;
		cmd.full_dma.local_addr = CXI_VA_TO_IOVA(md, buffer);
		cmd.full_dma.request_len = buflen;

		ofi_spin_lock(&cmdq->lock);

		/* this uses cached values */
		ret = cxip_txq_cp_set(cmdq, ep_obj->auth_key.vni,
				      mc_obj->tc, mc_obj->tc_type);
		if (ret)
			goto err_unlock;

		ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &cmd.full_dma);
	} else {
		cmd.c_state.event_send_disable = 1;
		cmd.c_state.event_success_disable = 1;
		cmd.c_state.restricted = 1;
		cmd.c_state.reduction = is_mcast;
		cmd.c_state.index_ext = index_ext;
		cmd.c_state.eq = cxip_cq_tx_eqn(ep_obj->coll.tx_cq);
		cmd.c_state.initiator = CXI_MATCH_ID(
			ep_obj->domain->iface->dev->info.pid_bits,
			ep_obj->src_addr.pid, ep_obj->src_addr.nic);

		ofi_spin_lock(&cmdq->lock);

		/* this uses cached values */
		ret = cxip_txq_cp_set(cmdq, ep_obj->auth_key.vni,
				      mc_obj->tc, mc_obj->tc_type);
		if (ret)
			goto err_unlock;

		ret = cxip_cmdq_emit_c_state(cmdq, &cmd.c_state);
		if (ret)
			goto err_unlock;

		memset(&cmd.idc_put, 0, sizeof(cmd.idc_put));
		cmd.idc_put.idc_header.dfa = dfa;
		ret = cxi_cq_emit_idc_put(cmdq->dev_cmdq, &cmd.idc_put,
					  buffer, buflen);
	}

	if (ret) {
		/* Return error according to Domain Resource Management
		 */
		ret = -FI_EAGAIN;
		goto err_unlock;
	}

	cxi_cq_ring(cmdq->dev_cmdq);
	ret = FI_SUCCESS;

	ofi_atomic_inc32(&reduction->mc_obj->send_cnt);

err_unlock:
	ofi_spin_unlock(&cmdq->lock);
	return ret;
}

/****************************************************************************
 * RECV operation (restricted IDC Put to a local PTE)
 */

/* Report success/error results of an RX event through RX CQ / counters, and
 * roll over the buffers if appropriate.
 *
 * NOTE: req may be invalid after this call.
 */
static void _coll_rx_req_report(struct cxip_req *req)
{
	size_t overflow;
	int err, ret;

	req->flags &= (FI_RECV | FI_COMPLETION | FI_COLLECTIVE);

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
		cxip_cq_req_free(req);
	}
}

/* Evaluate PUT request to see if this is a reduction packet.
 */
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

	/* Treat as a reduction packet */
	req->coll.isred = true;
	ofi_atomic_inc32(&mc_obj->pkt_cnt);
	reduction = &mc_obj->reduction[pkt->hdr.cookie.red_id];
	PRT("pkt received redid %d seqno %d\n", reduction->red_id,
	    pkt->hdr.seqno);
	_dump_red_pkt(pkt, "recv");
	_progress_coll(reduction, pkt);
}

/* Event-handling callback for posted receive buffers.
 */
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
		CXIP_WARN("Unexpected event type %d, rc: %d\n",
			  event->hdr.event_type, req->coll.cxi_rc);
		break;
	}

	return FI_SUCCESS;
}

/* Inject a hardware LE append. Does not generate HW LINK event unless error.
 */
static int _hw_coll_recv(struct cxip_coll_pte *coll_pte, struct cxip_req *req)
{
	uint32_t le_flags;
	uint64_t recv_iova;
	int ret;

	/* Always set manage_local in Receive LEs. This makes Cassini ignore
	 * initiator remote_offset in all Puts.
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
			      coll_pte->ep_obj->min_multi_recv,
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
	req = cxip_cq_req_alloc(coll_pte->ep_obj->coll.rx_cq, 1, buf);
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
	req->flags = (FI_RECV | FI_COMPLETION | FI_COLLECTIVE);
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
	cxip_cq_req_free(req);

recv_unmap:
	cxip_unmap(buf->cxi_md);
	return ret;
}

/****************************************************************************
 * PTE management functions.
 */

/* PTE state-change callback.
 */
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

/* Enable a collective PTE. Wait for completion.
 */
static inline int _coll_pte_enable(struct cxip_coll_pte *coll_pte,
				   uint32_t drop_count)
{
	return cxip_pte_set_state_wait(coll_pte->pte,
				       coll_pte->ep_obj->coll.rx_cmdq,
				       coll_pte->ep_obj->coll.rx_cq,
				       C_PTLTE_ENABLED, drop_count);
}

/* Disable a collective PTE. Wait for completion.
 */
static inline int _coll_pte_disable(struct cxip_coll_pte *coll_pte)
{
	return cxip_pte_set_state_wait(coll_pte->pte,
				       coll_pte->ep_obj->coll.rx_cmdq,
				       coll_pte->ep_obj->coll.rx_cq,
				       C_PTLTE_DISABLED, 0);
}

/* Destroy and unmap all buffers used by the collectives PTE.
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
 * fully, or it fails and removes all buffers.
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
	do {
		sched_yield();
		cxip_cq_progress(coll_pte->ep_obj->coll.rx_cq);
	} while (ofi_atomic_get32(&coll_pte->buf_cnt) < count);

	return FI_SUCCESS;
del_msg:
	free(buf);
out:
	_coll_destroy_buffers(coll_pte);
	return ret;
}

/****************************************************************************
 * Initialize, configure, enable, disable, and close the collective PTE.
 */

/**
 * Initialize the collectives structures.
 *
 * Must be done during EP initialization.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
void cxip_coll_init(struct cxip_ep_obj *ep_obj)
{
	cxip_coll_populate_opcodes();

	ep_obj->coll.rx_cmdq = NULL;
	ep_obj->coll.tx_cmdq = NULL;
	ep_obj->coll.rx_cntr = NULL;
	ep_obj->coll.tx_cntr = NULL;
	ep_obj->coll.rx_cq = NULL;
	ep_obj->coll.tx_cq = NULL;
	ep_obj->coll.min_multi_recv = CXIP_COLL_MIN_FREE;
	ep_obj->coll.buffer_count = CXIP_COLL_MIN_RX_BUFS;
	ep_obj->coll.buffer_size = CXIP_COLL_MIN_RX_SIZE;

	ofi_atomic_initialize32(&ep_obj->coll.mc_count, 0);
}

/**
 * Enable collectives.
 *
 * Must be preceded by cxip_coll_init(), called from STD EP enable.
 *
 * There is only one collectives object associated with an EP. It can be safely
 * enabled multiple times.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
int cxip_coll_enable(struct cxip_ep_obj *ep_obj)
{
	if (ep_obj->coll.enabled)
		return FI_SUCCESS;

	/* A read-only or write-only endpoint is legal */
	if (!(ofi_recv_allowed(ep_obj->rxcs[0]->attr.caps) &&
	      ofi_send_allowed(ep_obj->txcs[0]->attr.caps))) {
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
	ep_obj->coll.rx_cmdq = ep_obj->rxcs[0]->rx_cmdq;
	ep_obj->coll.tx_cmdq = ep_obj->txcs[0]->tx_cmdq;
	ep_obj->coll.rx_cntr = ep_obj->rxcs[0]->recv_cntr;
	ep_obj->coll.tx_cntr = ep_obj->txcs[0]->send_cntr;
	ep_obj->coll.rx_cq = ep_obj->rxcs[0]->recv_cq;
	ep_obj->coll.tx_cq = ep_obj->txcs[0]->send_cq;
	ep_obj->coll.eq = ep_obj->eq;

	ep_obj->coll.enabled = true;

	return FI_SUCCESS;
}

/**
 * Disable collectives.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
int cxip_coll_disable(struct cxip_ep_obj *ep_obj)
{
	if (!ep_obj->coll.enabled)
		return FI_SUCCESS;

	ep_obj->coll.enabled = false;
	// TODO: should this unlink EP objects?

	return FI_SUCCESS;
}

/**
 * Closes collectives and cleans up.
 *
 * Formal placeholder.
 *
 * Must be done during EP close.
 *
 * @param ep_obj - EP object
 */
void cxip_coll_close(struct cxip_ep_obj *ep_obj)
{

}

/* Write a Join Complete event to the endpoint EQ
 */
static void _post_join_complete(struct cxip_coll_mc *mc_obj, void *context,
			        int error)
{
	struct fi_eq_err_entry entry = {};
	size_t size = sizeof(struct fi_eq_entry);
	int ret;

	entry.fid = &mc_obj->mc_fid.fid;
	entry.context = context;

	if (error) {
		size = sizeof(struct fi_eq_err_entry);
		entry.err = error;
	}

	ret = ofi_eq_write(&mc_obj->ep_obj->eq->util_eq.eq_fid,
			   FI_JOIN_COMPLETE, &entry,
			   size, FI_COLLECTIVE);
	if (ret < 0)
		CXIP_INFO("FATAL ERROR: cannot post to EQ\n");
}

/****************************************************************************
 * Reduction packet management.
 */

static inline void _zcopy_pkt_data(void *tgt, const void *src, int len)
{
	if (tgt) {
		if (src)
			memcpy(tgt, src, len);
		else
			len = 0;
		memset((uint8_t *)tgt + len, 0, CXIP_COLL_MAX_TX_SIZE - len);
	}
}

/* Simulated unicast send of multiple packets as root node to leaf nodes.
 */
static ssize_t _send_pkt_as_root(struct cxip_coll_reduction *reduction,
					bool retry)
{
	int i, ret;

	for (i = 0; i < reduction->mc_obj->av_set->fi_addr_cnt; i++) {
		if (i == reduction->mc_obj->mynode_index &&
		    reduction->mc_obj->av_set->fi_addr_cnt > 1)
			continue;
		ret = cxip_coll_send(reduction, i,
				     reduction->tx_msg,
				     sizeof(struct red_pkt),
				     reduction->mc_obj->reduction_md);
		PRT("pkt sent tgt %d redid %d seqno %d ret %d\n",
		    i, reduction->red_id, reduction->seqno, ret);
		if (ret)
			return ret;
	}
	return FI_SUCCESS;
}

/* Simulated unicast send of single packet as leaf node to root node.
 */
static inline ssize_t _send_pkt_as_leaf(struct cxip_coll_reduction *reduction,
					bool retry)
{
	int ret;

	ret = cxip_coll_send(reduction, reduction->mc_obj->hwroot_index,
			     reduction->tx_msg, sizeof(struct red_pkt),
			     reduction->mc_obj->reduction_md);
	PRT("pkt sent tgt %d redid %d seqno %d ret %d\n",
	    reduction->mc_obj->hwroot_index, reduction->red_id,
	    reduction->seqno, ret);
	return ret;
}

/* Multicast send of single packet from root or leaf node.
 */
static inline ssize_t _send_pkt_mc(struct cxip_coll_reduction *reduction,
				   bool retry)
{
	return cxip_coll_send(reduction, 0,
			      reduction->tx_msg,
			      sizeof(struct red_pkt),
			      reduction->mc_obj->reduction_md);
}

/* Send packet from root or leaf node as appropriate.
 */
static inline ssize_t _send_pkt(struct cxip_coll_reduction *reduction,
				bool retry)
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

/**
 * Prevent setting the ARM bit on a root packet.
 *
 * This is used in testing to suppress Rosetta collective operations, forcing
 * all leaf packets to arrive at the root. It is of no use in production.
 */
// TODO Remove for production
int cxip_coll_arm_enable(struct fid_mc *mc, bool enable)
{
	struct cxip_coll_mc *mc_obj = (struct cxip_coll_mc *)mc;
	int old = mc_obj->arm_enable;

	mc_obj->arm_enable = enable;

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
 *
 * @param mc multicast object to limit
 * @param max_red_id maximum number of reduction ID values to use
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

int cxip_coll_send_red_pkt(struct cxip_coll_reduction *reduction,
			   int arm, size_t redcnt, int op, const void *data,
			   int len, enum cxip_coll_rc red_rc, bool retry)
{
	struct red_pkt *pkt;
	int ret;

	if (len > CXIP_COLL_MAX_TX_SIZE) {
		CXIP_INFO("length too large: %d\n", len);
		return -FI_EINVAL;
	}

	pkt = (struct red_pkt *)reduction->tx_msg;

	memset(&pkt->hdr, 0, sizeof(pkt->hdr));
	pkt->hdr.redcnt = redcnt;
	pkt->hdr.arm = arm;
	pkt->hdr.op = op;
	pkt->hdr.red_rc = red_rc;
	pkt->hdr.seqno = reduction->seqno;
	pkt->hdr.resno = reduction->resno;
	pkt->hdr.cookie.mcast_id = reduction->mc_obj->mcast_objid;
	pkt->hdr.cookie.red_id = reduction->red_id;
	pkt->hdr.cookie.retry = retry;
	pkt->hdr.cookie.magic = MAGIC;
	_zcopy_pkt_data(pkt->data.databuf, data, len);
	_dump_red_pkt(pkt, "send");
	_swappkt(pkt);

	/* -FI_EAGAIN means HW queue is full, should self-clear */
	do {
		ret = _send_pkt(reduction, retry);
	} while (ret == -FI_EAGAIN);

	return ret;
}

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

static inline enum c_return_code _red_to_cxi_error(enum cxip_coll_rc red_rc)
{
	switch (red_rc) {
	case CXIP_COLL_RC_SUCCESS:
		return C_RC_OK;
	case CXIP_COLL_RC_FLT_INEXACT:
	case CXIP_COLL_RC_REPSUM_INEXACT:
		return C_RC_AMO_FP_INEXACT;
	case CXIP_COLL_RC_FLT_OVERFLOW:
	case CXIP_COLL_RC_INT_OVERFLOW:
		return C_RC_AMO_FP_OVERFLOW;
	case CXIP_COLL_RC_FLT_INVALID:
		return C_RC_AMO_FP_INVALID;
	case CXIP_COLL_RC_CONTR_OVERFLOW:
		return C_RC_AMO_LENGTH_ERROR;
	case CXIP_COLL_RC_OP_MISMATCH:
		return C_RC_AMO_INVAL_OP_ERROR;
	default:
		return C_RC_AMO_ALIGN_ERROR;
	}
}

/* Post a reduction completion request to the collective TX CQ.
 */
static void _post_coll_complete(struct cxip_coll_reduction *reduction, int err)
{
	struct cxip_req *req;
	enum c_return_code cxi_rc;
	int ret;

	/* Indicates collective completion by writing to the endpoint TX CQ */
	req = reduction->op_inject_req;
	reduction->op_inject_req = NULL;
	if (req) {
		if (err != FI_SUCCESS) {
			ret = cxip_cq_req_error(req, 0, err, 0, NULL, 0);
		} else if (reduction->red_rc != CXIP_COLL_RC_SUCCESS) {
			cxi_rc = _red_to_cxi_error(reduction->red_rc);
			ret = cxip_cq_req_error(req, 0, FI_ENODATA, cxi_rc,
						NULL, 0);
		} else {
			ret = cxip_cq_req_complete(req);
		}
		if (ret < 0) {
			CXIP_WARN("Collective complete post: %d\n", ret);
		}
	}
}

/* Record only the first of multiple errors.
 */
static inline void _set_reduce_error(struct cxip_coll_reduction *reduction,
				     enum cxip_coll_rc red_rc)
{
	if (!reduction->red_rc)
		reduction->red_rc = red_rc;
}

/* Perform a reduction on the root in software.
 */
static int _root_reduce(struct cxip_coll_reduction *reduction,
			struct red_pkt *pkt, uint32_t exp_count)
{
	union cxip_coll_data *red_data;
	int i;

	/* first packet to arrive (root or leaf) sets up the reduction */
	red_data = (union cxip_coll_data *)reduction->red_data;
	if (!reduction->red_init) {
		memcpy(red_data, &pkt->data, CXIP_COLL_MAX_TX_SIZE);
		reduction->red_op = pkt->hdr.op;
		reduction->red_rc = pkt->hdr.red_rc;
		reduction->red_cnt = pkt->hdr.redcnt;
		reduction->red_init = true;
		goto out;
	}

	reduction->red_cnt += pkt->hdr.redcnt;
	if (pkt->hdr.red_rc != CXIP_COLL_RC_SUCCESS) {
		_set_reduce_error(reduction, pkt->hdr.red_rc);
		goto out;
	}

	if (pkt->hdr.op != reduction->red_op) {
		_set_reduce_error(reduction, CXIP_COLL_RC_OP_MISMATCH);
		goto out;
	}

	if (reduction->red_cnt > exp_count) {
		_set_reduce_error(reduction, CXIP_COLL_RC_CONTR_OVERFLOW);
		goto out;
	}

	switch (reduction->red_op) {
	case COLL_OPCODE_BARRIER:
		break;
	case COLL_OPCODE_BIT_AND:
		for (i = 0; i < 4; i++)
			red_data->ival[i] &= pkt->data.ival[i];
		break;
	case COLL_OPCODE_BIT_OR:
		for (i = 0; i < 4; i++)
			red_data->ival[i] |= pkt->data.ival[i];
		break;
	case COLL_OPCODE_BIT_XOR:
		for (i = 0; i < 4; i++)
			red_data->ival[i] ^= pkt->data.ival[i];
		break;
	case COLL_OPCODE_INT_MIN:
		for (i = 0; i < 4; i++)
			if (red_data->ival[i] > pkt->data.ival[i])
				red_data->ival[i] = pkt->data.ival[i];
		break;
	case COLL_OPCODE_INT_MAX:
		for (i = 0; i < 4; i++)
			if (red_data->ival[i] < pkt->data.ival[i])
				red_data->ival[i] = pkt->data.ival[i];
		break;
	case COLL_OPCODE_INT_MINMAXLOC:
		if (red_data->ival[0] > pkt->data.ival[0]) {
			red_data->ival[0] = pkt->data.ival[0];
			red_data->ival[1] = pkt->data.ival[1];
		}
		if (red_data->ival[2] < pkt->data.ival[2]) {
			red_data->ival[2] = pkt->data.ival[2];
			red_data->ival[3] = pkt->data.ival[3];
		}
		break;
	case COLL_OPCODE_INT_SUM:
		for (i = 0; i < 4; i++)
			red_data->ival[i] += pkt->data.ival[i];
		break;
	case COLL_OPCODE_FLT_MIN:
		for (i = 0; i < 4; i++)
			if (red_data->fval[i] > pkt->data.fval[i])
				red_data->fval[i] = pkt->data.fval[i];
		break;
	case COLL_OPCODE_FLT_MAX:
		for (i = 0; i < 4; i++)
			if (red_data->fval[i] < pkt->data.fval[i])
				red_data->fval[i] = pkt->data.fval[i];
		break;
	case COLL_OPCODE_FLT_MINMAXLOC:
		if (red_data->fval[0] > pkt->data.fval[0]) {
			red_data->fval[0] = pkt->data.fval[0];
			red_data->fval[1] = pkt->data.fval[1];
		}
		if (red_data->fval[2] < pkt->data.fval[2]) {
			red_data->fval[2] = pkt->data.fval[2];
			red_data->fval[3] = pkt->data.fval[3];
		}
		break;
	case COLL_OPCODE_FLT_MINNUM:
		// TODO
		break;
	case COLL_OPCODE_FLT_MAXNUM:
		// TODO
		break;
	case COLL_OPCODE_FLT_MINMAXNUMLOC:
		// TODO
		break;
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND0:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND1:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND2:
	case COLL_OPCODE_FLT_SUM_NOFTZ_RND3:
	case COLL_OPCODE_FLT_SUM_FTZ_RND0:
	case COLL_OPCODE_FLT_SUM_FTZ_RND1:
	case COLL_OPCODE_FLT_SUM_FTZ_RND2:
	case COLL_OPCODE_FLT_SUM_FTZ_RND3:
		/* Rosetta opcode has been chosen according to the current
		 * rounding mode for this application, so all we need to do is
		 * add the numbers.
		 */
		for (i = 0; i < 4; i++)
			red_data->fval[i] += pkt->data.fval[i];
		break;
	case COLL_OPCODE_FLT_REPSUM:
		// TODO
		break;
	}
out:
	return (reduction->red_cnt < exp_count) ? -1 : 0;
}

/****************************************************************************
 * Collective State Machine
 *
 * The basic flow is:
 *   - all nodes reach a reduction call (at different times)
 *   - leaf nodes send their data, to be reduced, and block, polling CQ
 *   - root node prepares for the reduction, and blocks, polling CQ
 *   - root node receives leaf packets and reduces them, until all received
 *   - root node sends Arm Packet with results, and unblocks
 *   - leaf nodes receive Arm Packet with results, and unblock
 *
 * The Rosetta acceleration comes from the Arm Packet, which speculatively
 * arms the Rosetta tree for the NEXT operation. This persists until a
 * timeout expires. The timeout is specified when the multicast tree is created
 * by the Rosetta configuration service, and cannot be adusted.
 *
 * If the next collective operation occurs within the timeout, the leaf results
 * will be reduced in reduction engines by Rosetta as they move up the tree,
 * reducing the number of packets received by the root.
 *
 * If the reduction engine times out with partial results, it forwards the
 * partial results, and all subsequent results are passed directly to the next
 * Rosetta.
 *
 * The timeout is set to 20.2 seconds (the maximum allowed) for the egress port
 * attached to the NIC, so any collective that completes within 20 seconds will
 * see only one packet, with N pre-reduced contributions, where N is the number
 * of leaf nodes. In expected use-cases, the reduction cycles will be sub-second
 * or perhaps sub-millisecond.
 *
 * The first leaf contribution to reach a reduction engine establishes the
 * reduction operation. All subsequent contributions must use the same
 * operation, or Rosetta returns an error.
 *
 * There are eight reduction_id values, which can be used to acquire and use
 * up to eight independent reduction engines (REs) at each upstream port of each
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
 * is advanced after the reduction completes.
 *
 * The algorithm below handles the general case of reordered response packets,
 * by forcing responses to occur in initiation-order.
 *
 * If a packet is dropped, it is up to the retry handler to initiate a retry or
 * a failure for tail_red_id, allowing subsequent reductions to complete.
 */

/* modular increment */
#define	INCMOD(val, mod)	do { val = (val + 1) % mod; } while (0)

/* MONOTONIC timestamp operations */
static inline
void tsget(struct timespec *ts0)
{
	clock_gettime(CLOCK_MONOTONIC, ts0);
}

static inline
void tsdiff(struct timespec *ts1, const struct timespec *ts0)
{
	ts1->tv_sec -= ts0->tv_sec;
	ts1->tv_nsec -= ts0->tv_nsec;
	if (ts1->tv_nsec < 0) {
		ts1->tv_sec--;
		ts1->tv_nsec += 1000000000L;
	}
}

static inline
int tscomp(const struct timespec *ts1, const struct timespec *ts0)
{
	if (ts1->tv_sec > ts0->tv_sec)
		return 1;
	if (ts1->tv_sec < ts0->tv_sec)
		return -1;
	if (ts1->tv_nsec > ts0->tv_nsec)
		return 1;
	if (ts1->tv_nsec < ts0->tv_nsec)
		return -1;
	return 0;
}

static inline
bool tszero(const struct timespec *ts0)
{
	return (!ts0->tv_sec && !ts0->tv_nsec);
}

static inline
bool _is_red_init(struct cxip_coll_reduction *reduction)
{
	return !tszero(&reduction->armtime);
}

static inline
bool _is_red_timed_out(struct cxip_coll_reduction *reduction)
{
	struct timespec tsnow;

	tsget(&tsnow);
	tsdiff(&tsnow, &reduction->armtime);
	return (tscomp(&tsnow, &reduction->mc_obj->timeout) >= 0);
}

static inline
struct red_pkt * _copy_user_to_pkt(void *packet,
				   struct cxip_coll_reduction *reduction)
{
	/* only root uses this */
	struct red_pkt *rootpkt = (struct red_pkt *)packet;

	rootpkt->hdr.redcnt = 1;
	rootpkt->hdr.seqno = reduction->seqno;
	rootpkt->hdr.resno = reduction->resno;
	rootpkt->hdr.red_rc = CXIP_COLL_RC_SUCCESS;
	rootpkt->hdr.op = reduction->op_code;
	_zcopy_pkt_data(rootpkt->data.databuf, reduction->op_send_data,
			reduction->op_data_len);

	return rootpkt;
}

static inline
void _copy_pkt_to_user(struct cxip_coll_reduction *reduction,
		       struct red_pkt *pkt)
{
	if (reduction->op_rslt_data && reduction->op_data_len) {
		memcpy(reduction->op_rslt_data, pkt->data.databuf,
		       reduction->op_data_len);
	}
}

static inline
void _copy_result_to_user(struct cxip_coll_reduction *reduction)
{
	if (reduction->op_rslt_data && reduction->op_data_len) {
		memcpy(reduction->op_rslt_data, reduction->red_data,
		       reduction->op_data_len);
	}
}

/* Root node state machine.
 * !pkt means this is progressing from injection call (e.g. fi_reduce())
 *  pkt means this is progressing from event callback (leaf packet)
 */
static void _progress_root(struct cxip_coll_reduction *reduction,
			   struct red_pkt *pkt)
{
	struct cxip_coll_mc *mc_obj = reduction->mc_obj;
	struct red_pkt *rootpkt = (struct red_pkt *)reduction->tx_msg;
	ssize_t ret;

	/* Drop packets until root is initialized. */
	if (reduction->coll_state != CXIP_COLL_STATE_READY)
		return;

	if (!pkt) {
		/* 'Receive' data packet with initial root data */
		pkt = _copy_user_to_pkt(reduction->tx_msg, reduction);
	} else {
		/* Drop old packets */
		if (pkt->hdr.resno != reduction->seqno) {
			ofi_atomic_inc32(&mc_obj->seq_err_cnt);
			return;
		}

		/* If a retry is needed */
		if (_is_red_timed_out(reduction)) {
			CXIP_DBG("RETRY collective packet\n");

			reduction->seqno = mc_obj->seqno;
			ret = cxip_coll_send_red_pkt(
				reduction, mc_obj->arm_enable,
				0, 0, NULL, 0, 0, true);
			if (ret) {
				/* fatal send error, collectives broken */
				CXIP_WARN("Collective send: %ld\n", ret);
				reduction->red_rc = 0;	// TODO error code?
				_post_coll_complete(reduction, ret);
				reduction->coll_state = CXIP_COLL_STATE_FAULT;
				return;
			}
			INCMOD(mc_obj->seqno, CXIP_COLL_MAX_SEQNO);
			tsget(&reduction->armtime);

			/* start reduction over */
			reduction->red_init = false;
			pkt = _copy_user_to_pkt(reduction->tx_msg, reduction);
		}
	}

	/* initialize or add to reduction */
	ret = _root_reduce(reduction, pkt, mc_obj->av_set->fi_addr_cnt);
	if (ret == 0) {
		/* reduction completed on root */
		rootpkt->hdr.red_rc = reduction->red_rc;
		reduction->completed = true;
	}

	/* Complete operations in injection order */
	reduction = &mc_obj->reduction[mc_obj->tail_red_id];
	while (reduction->in_use && reduction->completed) {

		/* copy reduction result to root user response buffer */
		_copy_result_to_user(reduction);

		/* send reduction result to leaves */
		reduction->seqno = mc_obj->seqno;
		ret = cxip_coll_send_red_pkt(reduction,
					     mc_obj->arm_enable,
					     reduction->red_cnt,
					     reduction->op_code,
					     reduction->op_rslt_data,
					     reduction->op_data_len,
					     reduction->red_rc,
					     false);
		if (ret) {
			/* fatal send error, leaves are hung */
			CXIP_WARN("Collective send: %ld\n", ret);
			reduction->red_rc = 0;	// TODO error code?
			_post_coll_complete(reduction, ret);
			reduction->coll_state = CXIP_COLL_STATE_FAULT;
			break;
		}
		INCMOD(mc_obj->seqno, CXIP_COLL_MAX_SEQNO);
		tsget(&reduction->armtime);

		/* Reduction completed on root */
		reduction->in_use = false;
		reduction->completed = false;
		reduction->red_init = false;
		_post_coll_complete(reduction, FI_SUCCESS);

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
	int ret;

	if (reduction->coll_state != CXIP_COLL_STATE_READY)
		return;

	/* initial state for leaf is always READY */

	if (!pkt) {
		/* Send leaf data to root */
		ret = cxip_coll_send_red_pkt(reduction, false, 1,
					     reduction->op_code,
					     reduction->op_send_data,
					     reduction->op_data_len,
					     reduction->red_rc,
					     false);
		if (ret) {
			/* fatal send error, root will time out and retry */
			CXIP_WARN("Collective send: %d\n", ret);
			return;
		}
	} else {
		/* Extract sequence number for next response */
		reduction->seqno = pkt->hdr.seqno;
		reduction->resno = pkt->hdr.seqno;

		/* Any packet from root re-arms the reduction */
		tsget(&reduction->armtime);

		/* Check for retry request */
		if (pkt->hdr.cookie.retry) {
			// TODO -- this needs to be expanded, see design
			/* Send the previous packet with new seqno */
			CXIP_DBG("leaf sending retry packet\n");
			pkt = (struct red_pkt *)&reduction->tx_msg;
			_swapbyteorder(&pkt->hdr, sizeof(pkt->hdr));
			pkt->hdr.seqno = reduction->seqno;
			pkt->hdr.resno = reduction->resno;
			_swapbyteorder(&pkt->hdr, sizeof(pkt->hdr));
			_send_pkt(reduction, true);
			/* do not change state, wait for next ARM */
			return;
		}

		/* Not a retry, no redcnt: just drop arming packet */
		if (pkt->hdr.redcnt == 0)
			return;

		/* Capture final reduction data */
		reduction->red_rc = pkt->hdr.red_rc;
		_copy_pkt_to_user(reduction, pkt);

		/* Reduction completed on leaf */
		reduction->completed = true;

		/* Complete operations in injection order */
		reduction = &mc_obj->reduction[mc_obj->tail_red_id];
		while (reduction->in_use && reduction->completed) {
			/* Reduction completed on leaf */
			reduction->in_use = false;
			reduction->completed = false;
			_post_coll_complete(reduction, FI_SUCCESS);

			/* Advance to the next reduction */
			INCMOD(mc_obj->tail_red_id, mc_obj->max_red_id);
			reduction = &mc_obj->reduction[mc_obj->tail_red_id];
		}
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

/* Generic collective request injection.
 *
 * Injection order must be the same on all nodes, and cannot be subject to
 * multithread race conditions that would potentially reorder the injection
 * calls: this would be a major application error, and results are
 * undefined. The application must ensure that any reduction initiation call
 * returns status before any other reduction initiation be attempted. So this
 * call can be considered process-atomic, and locks are not needed.
 */
ssize_t cxip_coll_inject(struct cxip_coll_mc *mc_obj,
			 enum fi_datatype datatype, int cxi_opcode,
			 const void *op_send_data, void *op_rslt_data,
			 size_t op_count, void *context, int *reduction_id)
{
	struct cxip_coll_reduction *reduction;
	struct cxip_req *req;
	int size, ret;

	if (!mc_obj->is_joined)
		return -FI_EOPBADSTATE;

	size = _get_cxi_datasize(datatype, op_count);
	if (size < 0)
		return size;

	reduction = &mc_obj->reduction[mc_obj->next_red_id];
	if (reduction->in_use)
		return -FI_EAGAIN;

	if (! _is_red_init(reduction)) {
		/* leaf has to wait for arm packet */
		if (!is_hw_root(mc_obj))
			return -FI_EAGAIN;

		/* root node arm no-op: redcnt == 0, retry == false */
		reduction->seqno = mc_obj->seqno;
		ret = cxip_coll_send_red_pkt(
			reduction, mc_obj->arm_enable,
			0, 0, NULL, 0, 0, false);
		if (ret) {
			PRT("send failure\n");
			return ret;
		}
		/* timer restarted, initialized */
		tsget(&reduction->armtime);
		/* root can continue now */
	}

	req = cxip_cq_req_alloc(mc_obj->ep_obj->coll.tx_cq, 1, NULL);
	if (!req)
		return -FI_ENOMEM;

	/* Advance to the next reduction id */
	INCMOD(mc_obj->next_red_id, mc_obj->max_red_id);

	/* Pass reduction parameters through the reduction structure */
	reduction->in_use = true;
	reduction->op_code = cxi_opcode;
	reduction->op_send_data = op_send_data;
	reduction->op_rslt_data = op_rslt_data;
	reduction->op_data_len = size;
	reduction->op_context = context;
	reduction->op_inject_req = req;
	reduction->op_inject_req->context = (uint64_t)context;

	if (reduction_id)
		*reduction_id = reduction->red_id;

	_progress_coll(reduction, NULL);
	return FI_SUCCESS;
}

ssize_t cxip_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_coll_mc *mc_obj;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	mc_obj = (struct cxip_coll_mc *) ((uintptr_t) coll_addr);

	if (mc_obj->ep_obj != cxi_ep->ep_obj) {
		CXIP_INFO("bad coll_addr\n");
		return -FI_EINVAL;
	}

	/* Use special opcode of -1 for barrier */
	ret = cxip_coll_inject(mc_obj, FI_UINT64, COLL_OPCODE_BARRIER,
			       NULL, NULL, 0, context, NULL);

	return ret;
}

/* NOTE: root_addr is index of node in fi_av_set list, i.e. local rank */
ssize_t cxip_broadcast(struct fid_ep *ep, void *buf, size_t count,
		       void *desc, fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, uint64_t flags,
		       void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_coll_mc *mc_obj;
	uint8_t src[CXIP_COLL_MAX_TX_SIZE];
	int size;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	mc_obj = (struct cxip_coll_mc *) ((uintptr_t) coll_addr);

	if (mc_obj->ep_obj != cxi_ep->ep_obj) {
		CXIP_INFO("bad coll_addr\n");
		return -FI_EINVAL;
	}

	size = _get_cxi_datasize(datatype, count);
	if (size < 0)
		return size;

	/* only root node contributes data */
	memset(src, 0, sizeof(src));
	if (root_addr == mc_obj->mynode_index)
		memcpy(src, buf, size);

	ret = cxip_coll_inject(mc_obj, datatype, COLL_OPCODE_BIT_OR, src, buf,
			       count, context, NULL);
	return ret;
}

/* NOTE: root_addr is index of node in fi_av_set list, i.e. local rank */
ssize_t cxip_reduce(struct fid_ep *ep, const void *buf, size_t count,
		    void *desc, void *result, void *result_desc,
		    fi_addr_t coll_addr, fi_addr_t root_addr,
		    enum fi_datatype datatype, enum fi_op op, uint64_t flags,
		    void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	mc_obj = (struct cxip_coll_mc *) ((uintptr_t) coll_addr);

	if (mc_obj->ep_obj != cxi_ep->ep_obj) {
		CXIP_INFO("bad coll_addr\n");
		return -FI_EINVAL;
	}

	if (root_addr != mc_obj->mynode_index)
		result = NULL;

	cxi_opcode = cxip_fi2cxi_opcode(op, datatype);
	if (cxi_opcode < 0) {
		CXIP_INFO("bad opcode %d\n", op);
		return cxi_opcode;
	}

	ret = cxip_coll_inject(mc_obj, datatype, cxi_opcode, buf, result,
			       count, context, NULL);

	return ret;
}

ssize_t cxip_allreduce(struct fid_ep *ep, const void *buf, size_t count,
		       void *desc, void *result, void *result_desc,
		       fi_addr_t coll_addr, enum fi_datatype datatype,
		       enum fi_op op, uint64_t flags, void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	mc_obj = (struct cxip_coll_mc *) ((uintptr_t) coll_addr);

	if (mc_obj->ep_obj != cxi_ep->ep_obj)
		return -FI_EINVAL;

	cxi_opcode = cxip_fi2cxi_opcode(op, datatype);
	if (cxi_opcode < 0) {
		CXIP_INFO("bad opcode %d\n", op);
		return cxi_opcode;
	}

	ret = cxip_coll_inject(mc_obj, datatype, cxi_opcode, buf, result,
			       count, context, NULL);

	return ret;
}

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

struct fi_ops_collective cxip_no_collective_ops = {
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

/****************************************************************************
 * Collective join operation.
 */

/* Close a multicast object.
 */
static int _close_mc(struct fid *fid)
{
	struct cxip_coll_mc *mc_obj;
	int ret;

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
	ofi_atomic_dec32(&mc_obj->av_set->ref);
	ofi_atomic_dec32(&mc_obj->ep_obj->coll.mc_count);
	free(mc_obj);

	return FI_SUCCESS;
}

static struct fi_ops mc_ops = {
	.size = sizeof(struct fi_ops),
	.close = _close_mc,
};

struct cxip_join_state {
	struct cxip_ep_obj *ep_obj;
	struct cxip_av_set *av_set;
	struct cxip_coll_mc *mc_obj;
	struct fid_mc **mc;
	void *context;
	uint64_t flags;
	uint64_t bcast_data;
	bool create_mcast;
	int mynode_idx;
	int hwroot_index;
	int mcast_addr;
	int mcast_objid;
	int simrank;
	int pid_idx;
};

struct _curl_usrptr {
	struct cxip_zbcoll_obj *zb;
	struct cxip_join_state *state;
};

static struct cxip_join_state **simstates;
static int num_simstates;

/**
 * @brief Utility routine to create and initialize the mc_obj.
 *
 * @param zb
 * @param statep
 */
static int _mc_initialize(struct cxip_zbcoll_obj *zb, void *statep)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.do_space_check = 1,
		.en_restricted_unicast_lm = 1,
	};
	struct cxip_join_state *state = statep;
	struct cxip_ep_obj *ep_obj = state->ep_obj;
	struct cxip_av_set *av_set = state->av_set;
	struct cxip_coll_mc *mc_obj;
	struct cxip_coll_pte *coll_pte;
	struct cxip_cmdq *cmdq;
	int red_id;
	int ret;

	/* Allocate the mc_obj, link, and adjust reference counts */
	mc_obj = calloc(1, sizeof(*av_set->mc_obj));
	if (!mc_obj)
		return -FI_ENOMEM;
	state->mc_obj = mc_obj;
	mc_obj->ep_obj = ep_obj;
	ofi_atomic_inc32(&ep_obj->coll.mc_count);

	av_set->mc_obj = mc_obj;
	mc_obj->av_set = av_set;
	ofi_atomic_inc32(&av_set->ref);

	/* Allocate the PTE structure and link */
	coll_pte = calloc(1, sizeof(*coll_pte));
	if (!coll_pte)
		return -FI_ENOMEM;

	/* initialize coll_pte */
	coll_pte->ep_obj = ep_obj;
	coll_pte->mc_obj = mc_obj;
	mc_obj->coll_pte = coll_pte;
	dlist_init(&coll_pte->buf_list);
	ofi_atomic_initialize32(&coll_pte->buf_cnt, 0);
	ofi_atomic_initialize32(&coll_pte->buf_swap_cnt, 0);
	// TODO should PTE create a reference count on EP?

	/* initialize mc_obj */
	mc_obj->mc_fid.fid.fclass = FI_CLASS_MC;
	mc_obj->mc_fid.fid.context = mc_obj;
	mc_obj->mc_fid.fid.ops = &mc_ops;
	mc_obj->mc_fid.fi_addr = (fi_addr_t)(uintptr_t)mc_obj;
	mc_obj->ep_obj = ep_obj;
	mc_obj->av_set = av_set;
	mc_obj->coll_pte = coll_pte;
	mc_obj->mcast_objid = state->mcast_objid;
	mc_obj->hwroot_index = state->hwroot_index;
	mc_obj->mcast_addr = state->mcast_addr;
	mc_obj->mynode_index = state->mynode_idx;
	mc_obj->max_red_id = CXIP_COLL_MAX_CONCUR;
	mc_obj->arm_enable = true;
	mc_obj->is_joined = true;
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

	/* map entire reduction block if using DMA */
	if (getenv("CXIP_COLL_USE_DMA_PUT")) {	// TODO move to cxi env
		/* EXPERIMENTAL */
		ret = cxil_map(ep_obj->domain->lni->lni,
			       mc_obj->reduction,
			       sizeof(mc_obj->reduction),
			       CXI_MAP_PIN  | CXI_MAP_READ | CXI_MAP_WRITE,
			       NULL, &mc_obj->reduction_md);
		if (ret)
			return ret;
	}

	/* bind PTE to domain */
	ret = cxip_pte_alloc(ep_obj->if_dom[0], ep_obj->coll.rx_cq->eq.eq,
			     state->pid_idx, state->create_mcast, &pt_opts,
			     _coll_pte_cb, coll_pte, &coll_pte->pte);
	if (ret)
		return ret;

	/* enable the PTE */
	ret = _coll_pte_enable(coll_pte, CXIP_PTE_IGNORE_DROPS);
	if (ret)
		return ret;

	/* add buffers to the PTE */
	ret = _coll_add_buffers(coll_pte,
				ep_obj->coll.buffer_size,
				ep_obj->coll.buffer_count);
	if (ret)
		return ret;

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
	ofi_spin_lock(&cmdq->lock);
	ret = cxip_txq_cp_set(cmdq, ep_obj->auth_key.vni,
			      mc_obj->tc, mc_obj->tc_type);
	ofi_spin_unlock(&cmdq->lock);
	if (ret)
		return ret;

	return FI_SUCCESS;
}

#if 0
static char *request_mcast_req(void)
{

}

static char *delete_mcast_req(int id)
{
}

static int mcast_parse(const char *response, int *reqid, int *mcast_id,
		       int *root_idx)
{

}

int cxip_request_mcast(const char *endpoint, bool verbose,
		       curlcomplete_t usrfunc, void *usrptr)
{

}

int cxip_delete_mcast(const char *endpoint, long reqid, bool verbose,
		      curlcomplete_t usrfunc, void *usrptr)
{

}
#endif

static void _cleanup_mcast(struct cxip_zbcoll_obj *zb, void *statep)
{
	struct cxip_join_state *state = statep;

	if (num_simstates) {
		int i;

		for (i = 0; i < num_simstates; i++) {
			state = simstates[i];
			_post_join_complete(state->mc_obj, state->context,
					    zb->error);
			if (state->mc_obj)
				fi_close(&state->mc_obj->mc_fid.fid);
		}
		free(simstates);
		simstates = NULL;
		num_simstates = 0;
	} else {
		_post_join_complete(state->mc_obj, state->context,
				    zb->error);
		if (state->mc_obj)
			fi_close(&state->mc_obj->mc_fid.fid);
	}
	cxip_zbcoll_free(zb);
	free(state);
}

#if 0
}
static void _append_sched(struct cxip_zbcoll_obj *zb, void *usrptr)
{

}

static void _noop(void *ptr)
{

}

static void _start_getgroup(void *ptr)
{

}
static inline void _finish_getgroup(void *ptr)
{

}

static void _curl_cb(struct cxip_curl_handle *handle)
{

	/* DO NOT FREE HANDLE */
	_append_sched(zb, jstate);
}

static void _start_curl(void *ptr)
{

}

static void _finish_curl(void *ptr)
{

}

static void _start_bcast(void *ptr)
{

}

static void _finish_bcast(void *ptr)
{
	int ret;

}

static void _start_reduce(void *ptr)
{

}

static void _finish_reduce(void *ptr)
{

}

static void _start_cleanup(void *ptr)
{
	struct cxip_join_state *jstate = ptr;
	struct cxip_zbcoll_obj *zb = jstate->zb;

	trc_join("%s: freeing jstate=%p\n", __func__, jstate);
	free(jstate);
}

#endif

static void _barrier_done(struct cxip_zbcoll_obj *zb, void *statep)
{
	struct cxip_join_state *state = statep;
	int ret;

	ret = zb->error;
	if (ret)
		goto fail;

	if (num_simstates) {
		int i;

		for (i = 0; i < num_simstates; i++) {
			state = simstates[i];
			*state->mc = &state->mc_obj->mc_fid;
			_post_join_complete(state->mc_obj, state->context,
					    FI_SUCCESS);
		}
		free(simstates);
		simstates = NULL;
		num_simstates = 0;
	} else {
		*state->mc = &state->mc_obj->mc_fid;
		_post_join_complete(state->mc_obj, state->context,
				    FI_SUCCESS);
	}

	/* we no longer need the zb object, or the state */
	cxip_zbcoll_free(zb);
	free(state);
	return;

fail:
	_cleanup_mcast(zb, state);
}

static void _broadcast_done(struct cxip_zbcoll_obj *zb, void *statep)
{
	struct cxip_join_state *state = statep;
	int ret;

	ret = zb->error;
	if (ret)
		goto fail;

	/* unpack the broadcast data */
	state->mcast_objid = 0;	// TODO
	state->hwroot_index = 0;
	state->mcast_addr = 0;

	/* initialize the multicast structure */
	ret = _mc_initialize(zb, state);
	if (ret)
		goto fail;

	/* initiate a barrier to synchronize again after setup */
	ret = cxip_zbcoll_push_cb(zb, _barrier_done, state);
	if (ret)
		goto fail;

	ret = cxip_zbcoll_barrier(zb);
	if (ret)
		goto fail;

	return;

fail:
	_cleanup_mcast(zb, state);
}

/* Process CURL information and broadcast it */
static void _curl_done(struct cxip_curl_handle *handle)
{
	struct _curl_usrptr *usrptr = (void *)handle->usrptr;
	struct cxip_zbcoll_obj *zb = usrptr->zb;
	struct cxip_join_state *state = usrptr->state;
	int ret;

	// parse handle->response
	state->bcast_data = 0x0;	// TODO pack data

	cxip_curl_free(handle);
	free(usrptr);

	/* Broadcast this information */
	ret = cxip_zbcoll_push_cb(zb, _broadcast_done, state);
	if (ret)
		goto fail;

	ret = cxip_zbcoll_broadcast(zb, &state->bcast_data);
	if (ret)
		goto fail;

	return;

fail:
	_cleanup_mcast(zb, state);
}

/**
 * @brief Process getgroup completion.
 *
 * After zb getgroup completes, there should be no further "normal" zb errors.
 *
 * In a production system, the process associated with fi_addr[0] is flagged
 * with create_mcast == true, and will issue the CURL request to the fabric
 * manager to acquire a new multicast address for this collective. All other
 * processes proceed immediately to zb_broadcast(). The completion function for
 * CURL will issue the zb_broadcast() once the multicast has been acquired, and
 * as it is on the zb root (fi_addr[0]), it will be used to broadcast the
 * multicast information.
 *
 * In any of the test systems, the broadcast address is explicitly or implicitly
 * known, so all processes proceed to zb_broadcast(). In this case, the
 * multicast address is already the same for all processes.
 *
 * @param zb     : zb coll object
 * @param statep : state containing variables
 */
static void _getgroup_done(struct cxip_zbcoll_obj *zb, void *statep)
{
	struct cxip_join_state *state = statep;
	int ret;

	ret = zb->error;
	if (ret)
		goto fail;

	/* If we don't need to create multicast, initialize */
	if (!state->create_mcast) {
		/* initialize the multicast structure */
		ret = _mc_initialize(zb, state);
		if (ret)
			goto fail;

		/* block on barrier */
		ret = cxip_zbcoll_push_cb(zb, _barrier_done, state);
		if (ret)
			goto fail;

		ret = cxip_zbcoll_barrier(zb);
		if (ret)
			goto fail;

		return;
	}

	/* mynode_idx == 0 is the broadcast sender, so create multicast first */
	if (state->mynode_idx == 0) {
		struct _curl_usrptr *usrptr;
		char *endpoint;
		char *request;

		endpoint = strdup("http://something");	// TODO
		request = strdup("something");		// TODO

		usrptr = calloc(1, sizeof(*usrptr));
		if (!usrptr) {
			zb->error = -FI_ENOMEM;
			goto fail;
		}
		usrptr->zb = zb;
		usrptr->state = state;

		ret = cxip_curl_perform(endpoint, request, 0, CURL_POST, false,
					_curl_done, usrptr);
		/* internal copies are made of endpoint and request */
		free(request);
		free(endpoint);
		if (ret) {
			free(usrptr);
			goto fail;
		}
		return;
	}

	/* All other endpoints go straight to broadcast and block */
	ret = cxip_zbcoll_push_cb(zb, _broadcast_done, state);
	if (ret)
		goto fail;

	ret = cxip_zbcoll_broadcast(zb, &state->bcast_data);
	if (ret)
		goto fail;

	return;

fail:
	_cleanup_mcast(zb, state);
}

#if 0

typedef	void (*sched_func)(void *ptr);

enum state_code {
};

const char *state_name[] = {
};

sched_func state_func[] = {
};

static enum state_code progress_state[][3] = {
};

static void _progress_sched(struct cxip_join_state *jstate)
{

}

static void _progress_join(struct cxip_ep_obj *ep_obj)
{

}
#endif

/**
 * @brief fi_join_collective() implementation.
 *
 * Calling syntax is defined by libfabric.
 *
 * This is a multi-stage collective operation, progressed by calling TX/RX CQs,
 * and the EQ for the endpoint. The basic sequence is:
 *
 * 1) get a collective group ID for zbcoll collectives
 * 2) acquire a collective address from the fabric manager
 * 3) broadcast the collective address to all endpoints
 * 4) initialize the mc_obj structure on all endpoints
 * 5) block on barrier until all endpoints are ready for collective data
 *
 * There are four operational models, one for production, and three for testing.
 *
 * In all cases, there must be one join for every address in the av_set
 * fi_addr_ary, and the collective proceeds among these joined objects.
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
 * fiaddr[1-N] send directly to fiaddr[0], and fiaddr[0] sends to each of the
 * other addresses in a simulated broadcast. This is not expected to be
 * performant, but it does exercise a necessary incast edge case, and it fully
 * exercises the collectives software across multiple nodes.
 *
 * COMM_KEY_MULTICAST is a fully-functioning model, but requires that an
 * external application prepare the multicast address on the fabric before
 * calling fi_join_collective() on any node. This information must be supplied
 * through the av_set->comm_key structure.
 *
 * COMM_KEY_NONE is the fully-function production model, in which
 * fi_join_collective() creates the multicast address by making a CURL call to
 * the fabric manager REST API. fiaddr[0] manages the CURL call, and broadcasts
 * the results to all of the other objects across the collective group.
 *
 * @param ep          : fabric endpoint fid
 * @param coll_addr   : FI_ADDR_NOTAVAIL (required)
 * @param coll_av_set : av_set fid
 * @param flags       : ignored
 * @param mc          : return pointer for mc object
 * @param context     : user context for concurrent joins
 * @return int error code
 */
int cxip_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			 const struct fid_av_set *coll_av_set,
			 uint64_t flags, struct fid_mc **mc, void *context)
{
	static uint32_t unicast_idcode = 0;

	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_av_set *av_set;
	struct cxip_join_state *state;
	struct cxip_zbcoll_obj *zb;
	bool link_zb;
	int ret;

	/* validate arguments */
	if (!ep || !coll_av_set || !mc || coll_addr != FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;
	/* flags are ignored */

	cxip_ep = container_of(ep, struct cxip_ep, ep.fid);
	av_set = container_of(coll_av_set, struct cxip_av_set, av_set_fid);

	ep_obj = cxip_ep->ep_obj;

	/* allocate state to pass arguments through callbacks */
	state = calloc(1, sizeof(*state));
	if (! state)
		return -FI_ENOMEM;
	/* all errors after this must goto fail for cleanup */

	zb = NULL;
	state->ep_obj = ep_obj;
	state->av_set = av_set;
	state->mc = mc;
	state->context = context;
	state->flags = flags;

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
		state->mynode_idx = _caddr_to_idx(av_set, ep_obj->src_addr);
		state->simrank = ZB_NOSIM;
		state->pid_idx = CXIP_PTL_IDX_COLL;
		state->mcast_objid = 0;
		state->mcast_addr = 0;
		state->hwroot_index = 0;
		state->create_mcast = true;
		link_zb = false;
		break;
	case COMM_KEY_MULTICAST:
		/* Real network test with predefined multicast address */
		if (is_netsim(ep_obj)) {
			CXIP_INFO("NETSIM COMM_KEY_MULTICAST not supported\n");
			goto fail;
		}
		state->mynode_idx = _caddr_to_idx(av_set, ep_obj->src_addr);
		state->simrank = ZB_NOSIM;
		state->pid_idx = CXIP_PTL_IDX_COLL;
		state->mcast_objid = av_set->comm_key.mcast.mcast_id;
		state->hwroot_index = av_set->comm_key.mcast.hwroot_idx;
		state->mcast_addr = av_set->comm_key.mcast.mcast_id;
		state->create_mcast = false;
		link_zb = false;
		break;
	case COMM_KEY_UNICAST:
		/* Real network test without multicast address */
		if (is_netsim(ep_obj)) {
			CXIP_INFO("NETSIM OMM_KEY_UNICAST not supported\n");
			goto fail;
		}
		state->mynode_idx = _caddr_to_idx(av_set, ep_obj->src_addr);
		state->simrank = state->mynode_idx;
		state->pid_idx = CXIP_PTL_IDX_COLL;
		state->mcast_objid = unicast_idcode++;
		state->mcast_addr = ep_obj->src_addr.nic;
		state->hwroot_index = 0;
		state->create_mcast = false;
		link_zb = false;
		break;
	case COMM_KEY_RANK:
		/* Single process simulation, can run under NETSIM */
		if (!simstates) {
			/* first join creates the entire array */
			if (av_set->comm_key.rank.rank != 0) {
				CXIP_INFO("Rank 0 must be first configured\n");
				goto fail;
			}
			num_simstates = av_set->fi_addr_cnt;
			simstates = calloc(num_simstates, sizeof(void *));
		}
		/* record this state in the array */
		simstates[av_set->comm_key.rank.rank] = state;
		state->mynode_idx = av_set->comm_key.rank.rank;
		state->simrank = state->mynode_idx;
		state->pid_idx = CXIP_PTL_IDX_COLL + state->simrank;
		state->mcast_objid = av_set->comm_key.rank.rank;
		state->mcast_addr = ep_obj->src_addr.nic;
		state->hwroot_index = 0;
		state->create_mcast = false;
		link_zb = true;
		break;
	default:
		CXIP_INFO("unexpected comm_key keytype: %d\n",
			  av_set->comm_key.keytype);
		goto fail;
	}

	/* Acquire a zbcoll identifier */
	ret = cxip_zbcoll_alloc(state->ep_obj,
				state->av_set->fi_addr_cnt,
				state->av_set->fi_addr_ary,
				state->simrank, &zb);
	if (ret)
		goto fail;
	if (link_zb) {
		static struct cxip_zbcoll_obj *zb0 = NULL;
		static int zb0_count = 0;

		if (!zb0)
			zb0 = zb;
		cxip_zbcoll_simlink(zb0, zb);
		if (++zb0_count == av_set->fi_addr_cnt) {
			zb0 = NULL;
			zb0_count = 0;
		}
	}

	/* Getgroup */
	ret = cxip_zbcoll_push_cb(zb, _getgroup_done, state);
	if (ret)
		goto fail;

	/* -FI_EAGAIN is a race, -FI_EBUSY is all in use */
	ret = cxip_zbcoll_getgroup(zb);
	if (ret)
		goto fail;

	return FI_SUCCESS;

fail:
	_cleanup_mcast(zb, state);
	return ret;
}

/* Reset all of the diagnostic counters atomically */
void cxip_coll_reset_mc_ctrs(struct fid_mc *mc)
{
	struct cxip_coll_mc *mc_obj = (struct cxip_coll_mc *)mc;

	ofi_spin_lock(&mc_obj->lock);
	ofi_atomic_set32(&mc_obj->send_cnt, 0);
	ofi_atomic_set32(&mc_obj->recv_cnt, 0);
	ofi_atomic_set32(&mc_obj->pkt_cnt, 0);
	ofi_atomic_set32(&mc_obj->seq_err_cnt, 0);
	ofi_spin_unlock(&mc_obj->lock);
}
