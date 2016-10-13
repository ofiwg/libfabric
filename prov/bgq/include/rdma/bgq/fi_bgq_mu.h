#ifndef _FI_PROV_BGQ_MU_H_
#define _FI_PROV_BGQ_MU_H_

#include <stdlib.h>
#include <stdio.h>

#include "rdma/bgq/fi_bgq_hwi.h"
#include "rdma/bgq/fi_bgq_spi.h"

#include "rdma/fi_errno.h"	// only for FI_* errno return codes
#include "rdma/fabric.h" // only for 'fi_addr_t' ... which is a typedef to uint64_t

#include "rdma/bgq/fi_bgq_l2atomic.h"

#define FI_BGQ_MU_RECFIFO_BYTES		(0x01 << 20)	/* 1 MB == 32K entries */
#define FI_BGQ_MU_RECFIFO_TAGGED_BYTES	(0x01 << 20)	/* 1 MB == 32K entries */
#define FI_BGQ_MU_RECFIFO_OTHER_BYTES	(0x01 << 15)	/* 32 KB == 1K entries */

#define FI_BGQ_MU_BAT_SUBGROUP_GLOBAL (65)
#define FI_BGQ_MU_BAT_ID_GLOBAL (FI_BGQ_MU_BAT_SUBGROUP_GLOBAL * BGQ_MU_NUM_DATA_COUNTERS_PER_SUBGROUP)
#define FI_BGQ_MU_BAT_ID_COUNTER (FI_BGQ_MU_BAT_ID_GLOBAL+1)
#define FI_BGQ_MU_BAT_ID_ZERO (FI_BGQ_MU_BAT_ID_COUNTER+1)
#define FI_BGQ_MU_BAT_ID_ONE (FI_BGQ_MU_BAT_ID_ZERO+1)
#define FI_BGQ_MU_BAT_ID_BLACKHOLE (FI_BGQ_MU_BAT_ID_ONE+1)

#define FI_BGQ_MUHWI_DESTINATION_MASK (0x073CF3C1ul)

union fi_bgq_addr {
	fi_addr_t			fi;
	uint64_t			raw;
	struct {
		union {
			MUHWI_Destination_t	Destination;
			struct {
				uint32_t	reserved	:  2;
				uint32_t	a		:  6;	/* only 3 bits are needed for Mira */
				uint32_t	b		:  6;	/* only 4 bits are needed for Mira */
				uint32_t	c		:  6;	/* only 4 bits are needed for Mira */
				uint32_t	d		:  6;	/* only 4 bits are needed for Mira */
				uint32_t	e		:  6;	/* only 1 bit is needed */
			};
		};
		uint16_t		fifo_map;			/* only 12 bits are needed for normal pt2pt; and only 10 bits for internode */
		uint16_t		is_local	:  1;		/* same as fifo_map::MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_LOCAL0 | fifo_map::MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_LOCAL1 */
		uint16_t		unused		:  7;
		uint16_t		rx		:  8;		/* node-scoped reception context identifier; see NOTE_MU_RECFIFO */
	};
};
//_Static_assert (sizeof(fi_addr_t) == sizeof(union fi_bgq_addr), "type size mismatch");

static inline
uint64_t fi_bgq_addr_is_local (fi_addr_t addr) {
	return (((uint64_t)addr) >> 15) & 0x01ull;
}

static inline
uint64_t fi_bgq_addr_rec_fifo_id (fi_addr_t addr) {
	return ((((uint64_t)addr) & 0x0FFull) << 1);
}

static inline
void fi_bgq_addr_dump (union fi_bgq_addr addr)
{
	fprintf(stderr, "==== %s ====\n", __func__);
	fprintf(stderr, "addr.a = %u\n", addr.a);
	fprintf(stderr, "addr.b = %u\n", addr.b);
	fprintf(stderr, "addr.c = %u\n", addr.c);
	fprintf(stderr, "addr.d = %u\n", addr.d);
	fprintf(stderr, "addr.e = %u\n", addr.e);
	fprintf(stderr, "addr.fifo_map = 0x%04x\n", addr.fifo_map);
	fprintf(stderr, "addr.is_local = %u\n", addr.is_local);
	fprintf(stderr, "addr.rx = %u\n", addr.rx);
	fprintf(stderr, "==== %s ====\n", __func__);
}


#define FI_BGQ_MU_PACKET_TYPE_TAG			(0x01ul<<1)
#define FI_BGQ_MU_PACKET_TYPE_INJECT			(0x01ul<<2)
#define FI_BGQ_MU_PACKET_TYPE_EAGER			(0x01ul<<3)
#define FI_BGQ_MU_PACKET_TYPE_RENDEZVOUS		(0x01ul<<4)
#define FI_BGQ_MU_PACKET_TYPE_RMA			(0x01ul<<5)
#define FI_BGQ_MU_PACKET_TYPE_ATOMIC			(0x01ul<<6)
#define FI_BGQ_MU_PACKET_TYPE_ACK			(0x01ul<<7)


/**
 * \brief MU packet header
 *
 * The MU packet header is consumed in many places and sometimes overloaded
 * for cache and memory allocation reasons.
 */
union fi_bgq_mu_packet_hdr {

	/* The torus packet header is 32 bytes. see: hwi/include/bqc/MU_PacketHeader.h */
	MUHWI_PacketHeader_t		muhwi;

	struct {
		/* The point-to-point header occupies bytes 0-11 of the packet header
		 * see: MUHWI_Pt2PtNetworkHeader_t in hwi/include/bqc/MU_Pt2PtNetworkHeader.h */
		uint64_t		reserved_0;
		uint32_t		reserved_1;

		/* The message unit header occupies bytes 12-31 of the packet header
		 * see: MUHWI_MessageUnitHeader_t in hwi/include/bqc/MU_MessageUnitHeader.h */
		uint16_t		reserved_2	: 10;
		uint16_t		unused_0	:  6;
		uint8_t			unused_1[18];
	} __attribute__((__packed__)) raw;

	struct {
		uint64_t		reserved_0;
		uint32_t		reserved_1;
		uint16_t		reserved_2	: 10;
		uint16_t		unused_0	:  6;

		uint8_t			unused_1;
		uint8_t			packet_type;		/* FI_BGQ_MU_PACKET_TYPE_*; all 8 bits are needed */
		uint64_t		unused_2[2];
	} __attribute__((__packed__)) common;

	struct {
		uint64_t		reserved_0;
		uint64_t		reserved_1	: 32;
		uint64_t		reserved_2	: 10;
		uint64_t		is_local	:  1;	/* used to specify fifo map */
		uint64_t		unused_0	:  3;
		uint64_t		message_length	: 10;	/* 0..512 bytes of payload data */
		uint64_t		reserved_3	:  8;	/* a.k.a. common::packet_type */

		MUHWI_Destination_t	origin;
		uint32_t		cntr_paddr_rsh3b;	/* 34b paddr, 8 byte aligned; See: NOTE_MU_PADDR */
		uint64_t		ofi_tag;
	} __attribute__((__packed__)) send;

	struct {
		uint64_t		reserved_0;
		uint32_t		reserved_1;
		uint16_t		reserved_2	: 10;
		uint16_t		is_local	:  1;	/* used to specify fifo map */
		uint16_t		niov_minus_1	:  5;	/* 1..32 mu iov elements in payload data */
		uint8_t			rget_inj_fifo_id;	/* 0..255 */
		uint8_t			reserved_3;		/* a.k.a. common::packet_type */

		union {
			uint32_t		origin_raw;
			MUHWI_Destination_t	origin;
			struct {
				uint32_t	fifo_am	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AM */
				uint32_t	fifo_ap	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AP */
				uint32_t	fifo_bm	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BM */
				uint32_t	fifo_bp	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BP */
				uint32_t	fifo_cm	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CM */
				uint32_t	a	:  3;
				uint32_t	unused0	:  2;
				uint32_t	b	:  4;
				uint32_t	unused1	:  2;
				uint32_t	c	:  4;
				uint32_t	unused2	:  2;
				uint32_t	d	:  4;
				uint32_t	fifo_cp	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CP */
				uint32_t	fifo_dm	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DM */
				uint32_t	fifo_dp	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DP */
				uint32_t	fifo_em	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_EM */
				uint32_t	fifo_ep	:  1;	/* MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_EP */
				uint32_t	e	:  1;
			};
		};
		uint32_t		cntr_paddr_rsh3b;	/* 34b paddr, 8 byte aligned; See: NOTE_MU_PADDR */
		uint64_t		ofi_tag;
	} __attribute__((__packed__)) rendezvous;

	struct {
		uint64_t		reserved_0;
		uint32_t		reserved_1;
		uint16_t		reserved_2	: 10;
		uint16_t		unused_0	:  6;
		uint8_t			message_length;		/* 0..8 bytes of immediate data; only 4 bits are actually needed */
		uint8_t			reserved_3;		/* a.k.a. common::packet_type */
		uint64_t		data;
		uint64_t		ofi_tag;
	} __attribute__((__packed__)) inject;

	struct {
		uint64_t		reserved_0;
		uint32_t		reserved_1;
		uint16_t		reserved_2	: 10;
		uint16_t		unused_0	:  6;

		uint8_t			unused_1;
		uint8_t			reserved_3;		/* a.k.a. common::packet_type (FI_BGQ_MU_PACKET_TYPE_ACK) */
		uint64_t		unused_2;
		uintptr_t		context;
	} __attribute__((__packed__)) ack;

	struct {
		uint64_t		reserved_0;
		uint32_t		reserved_1;
		uint16_t		reserved_2	: 10;
		uint16_t		unused_0	:  6;

		uint8_t			ndesc;			/* 0..8 descriptors */
		uint8_t			reserved_3;		/* a.k.a. common::packet_type (FI_BGQ_MU_PACKET_TYPE_RMA) */
		uint64_t		nbytes		: 16;	/* 0..512 bytes */
		uint64_t		unused_2	: 11;
		uint64_t		offset		: 37;	/* really only need 34 bits to reference all of physical memory (no mu atomics) TODO - FI_MR_SCALABLE uses virtual address as the offset? */
		uint64_t		key;			/* only 16 bits needed for FI_MR_SCALABLE */
	} __attribute__((__packed__)) rma;

	struct {
		uint64_t		reserved_0;
		uint32_t		reserved_1;
		uint32_t		reserved_2	: 10;
		uint32_t		unused_0	:  5;
		uint32_t		cntr_bat_id	:  9;
		uint32_t		reserved_3	:  8;	/* a.k.a. common::packet_type (FI_BGQ_MU_PACKET_TYPE_ATOMIC) */
		union {
			uint32_t		origin_raw;
			MUHWI_Destination_t	origin;
			struct {
				uint32_t	is_fetch	:  1;
				uint32_t	dt		:  4;	/* enum fi_datatype */
				uint32_t	a		:  3;	/* only 3 bits are needed for Mira */
				uint32_t	is_local	:  1;
				uint32_t	do_cntr		:  1;
				uint32_t	b		:  4;	/* only 4 bits are needed for Mira */
				uint32_t	unused_1	:  2;
				uint32_t	c		:  4;	/* only 4 bits are needed for Mira */
				uint32_t	unused_2	:  2;
				uint32_t	d		:  4;	/* only 4 bits are needed for Mira */
				uint32_t	op		:  5;	/* enum fi_op */
				uint32_t	e		:  1;	/* only 1 bit is needed for Mira */
			} __attribute__((__packed__));
		};
		uint16_t		nbytes_minus_1;			/* only 9 bits needed */
		uint16_t		key;				/* only 16 bits needed for FI_MR_SCALABLE; TODO 34 bits to hold paddr for FI_MR_BASIC */
		uint64_t		offset;				/* TODO FI_MR_BASIC* only needs 34 bits */
	} __attribute__((__packed__)) atomic;

} __attribute__((__aligned__(32)));

struct fi_bgq_mu_iov {
	uint64_t			message_length;
	uint64_t			src_paddr;
};

struct fi_bgq_mu_fetch_metadata {
	uint64_t			dst_paddr;
	uint64_t			cq_paddr;
	uint64_t			fifo_map;
	uint64_t			unused;
};

union fi_bgq_mu_packet_payload {
	uint8_t				byte[512];
	struct fi_bgq_mu_iov		mu_iov[32];
	struct {
		struct fi_bgq_mu_fetch_metadata	metadata;
		uint8_t				data[512-sizeof(struct fi_bgq_mu_fetch_metadata)];
	} atomic_fetch;
} __attribute__((__aligned__(32)));

struct fi_bgq_mu_packet {
	union {
		struct fi_bgq_mu_packet		*next;	/* first 8 bytes of the header is unused */
		union fi_bgq_mu_packet_hdr	hdr;
	};
	union fi_bgq_mu_packet_payload		payload;
} __attribute__((__aligned__(32)));


static inline uint64_t
fi_bgq_mu_packet_type_get (struct fi_bgq_mu_packet * pkt) {
	return pkt->hdr.common.packet_type;
}

static inline void
fi_bgq_mu_packet_type_set (union fi_bgq_mu_packet_hdr * hdr, const uint64_t packet_type) {
	hdr->common.packet_type = (uint8_t)packet_type;
}

static inline void
fi_bgq_mu_packet_rendezvous_origin (struct fi_bgq_mu_packet * pkt, MUHWI_Destination_t * out) {
	*((uint32_t*)out) = (((uint32_t)pkt->hdr.rendezvous.origin_raw) & 0x073CF3C1u);
}

static inline uint64_t
fi_bgq_mu_packet_rendezvous_fifomap (struct fi_bgq_mu_packet * pkt) {
	const uint32_t raw = (uint32_t)pkt->hdr.rendezvous.origin_raw;
	return (uint64_t) (((raw & 0x0000003Eu) << 5) | ((raw & 0xF8000000u) >> 16));
}

#define FI_BGQ_MU_DESCRIPTOR_UPDATE_BAT_TYPE_NONE	(0)
#define FI_BGQ_MU_DESCRIPTOR_UPDATE_BAT_TYPE_DST	(1)
#define FI_BGQ_MU_DESCRIPTOR_UPDATE_BAT_TYPE_SRC	(2)

union fi_bgq_mu_descriptor {

	/* The mu descriptor is 64 bytes. see: hwi/include/bqc/MU_Descriptor.h */
	MUHWI_Descriptor_t				muhwi_descriptor;

	struct {
		uint16_t				key_msb;
		uint8_t					update_type;		/* FI_BGQ_MU_DESCRIPTOR_UPDATE_BAT_TYPE_* */
		uint8_t					unused_0	:  7;
		uint8_t					reserved_0	:  1;

		uint32_t				unused_1	: 31;
		uint32_t				reserved_1	:  1;

		uint64_t				Pa_Payload;		/* 37 lsb are used */
		uint64_t				Message_Length;		/* 37 lsb (really, 35) are used */
		uint64_t				key_lsb		: 48;
		uint64_t				reserved_2	: 16;	/* a.k.a. Torus_FIFO_Map */

		union {
			MUHWI_Pt2PtNetworkHeader_t	muhwi_pt2ptnetworkheader;
			uint32_t			reserved_3[3];
		};

		union {
			MUHWI_MessageUnitHeader_t	muhwi_messageunitheader;
			struct {
				uint64_t		rec_payload_base_address_id	: 10;
				uint64_t		reserved_4			:  1;
				uint64_t		put_offset			: 37;
				uint64_t		unused_2			:  6;
				uint64_t		rec_counter_base_address_id	: 10;
				uint32_t		reserved_5[3];
			} __attribute__((__packed__)) dput;
		};

	} __attribute__((__packed__)) rma;

} __attribute__((__aligned__(64)));


static inline void
dump_descriptor (char * prefix, MUHWI_Descriptor_t * desc) {

	uint32_t * ptr = (uint32_t *)desc;
	fprintf(stderr, "%s [%p]: %08x %08x %08x %08x\n", prefix, ptr, *(ptr), *(ptr+1), *(ptr+2), *(ptr+3)); ptr+=4;
	fprintf(stderr, "%s [%p]: %08x %08x %08x %08x\n", prefix, ptr, *(ptr), *(ptr+1), *(ptr+2), *(ptr+3)); ptr+=4;
	fprintf(stderr, "%s [%p]: %08x %08x %08x %08x\n", prefix, ptr, *(ptr), *(ptr+1), *(ptr+2), *(ptr+3)); ptr+=4;
	fprintf(stderr, "%s [%p]: %08x %08x %08x %08x\n", prefix, ptr, *(ptr), *(ptr+1), *(ptr+2), *(ptr+3)); ptr+=4;

	fprintf(stderr, "%s descriptor dump at %p\n", prefix, (void*)desc);
	fprintf(stderr, "%s   .Half_Word0.Prefetch_Only .................. %d\n", prefix, desc->Half_Word0.Prefetch_Only);
	fprintf(stderr, "%s   .Half_Word1.Interrupt ...................... %d\n", prefix, desc->Half_Word1.Interrupt);
	fprintf(stderr, "%s   .Pa_Payload ................................ 0x%016lx\n", prefix, desc->Pa_Payload);
	fprintf(stderr, "%s   .Message_Length ............................ %lu\n", prefix, desc->Message_Length);
	fprintf(stderr, "%s   .Torus_FIFO_Map ............................ 0x%016lx\n", prefix, desc->Torus_FIFO_Map);
	fprintf(stderr, "%s   .PacketHeader.NetworkHeader.pt2pt\n", prefix);
	fprintf(stderr, "%s     .Data_Packet_Type ........................ 0x%02x\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Data_Packet_Type);
	fprintf(stderr, "%s     .Hints ................................... 0x%02x\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Hints);
	fprintf(stderr, "%s     .Byte2.Hint_E_plus ....................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Hint_E_plus);
	fprintf(stderr, "%s     .Byte2.Hint_E_minus ...................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Hint_E_minus);
	fprintf(stderr, "%s     .Byte2.Route_To_IO_Node .................. %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Route_To_IO_Node);
	fprintf(stderr, "%s     .Byte2.Return_From_IO_Node ............... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Return_From_IO_Node);
	fprintf(stderr, "%s     .Byte2.Dynamic ........................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Dynamic);
	fprintf(stderr, "%s     .Byte2.Deposit ........................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Deposit);
	fprintf(stderr, "%s     .Byte2.Interrupt ......................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte2.Interrupt);
	fprintf(stderr, "%s     .Byte3.Virtual_channel ................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte3.Virtual_channel);
	fprintf(stderr, "%s     .Byte3.Zone_Routing_Id ................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte3.Zone_Routing_Id);
	fprintf(stderr, "%s     .Byte3.Stay_On_Bubble .................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte3.Stay_On_Bubble);
	fprintf(stderr, "%s     .Destination.Destination.Reserved2 ....... %u\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Destination.Destination.Reserved2);
	fprintf(stderr, "%s     .Destination.Destination.A_Destination ... %u\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Destination.Destination.A_Destination);
	fprintf(stderr, "%s     .Destination.Destination.B_Destination ... %u\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Destination.Destination.B_Destination);
	fprintf(stderr, "%s     .Destination.Destination.C_Destination ... %u\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Destination.Destination.C_Destination);
	fprintf(stderr, "%s     .Destination.Destination.D_Destination ... %u\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Destination.Destination.D_Destination);
	fprintf(stderr, "%s     .Destination.Destination.E_Destination ... %u\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Destination.Destination.E_Destination);
	fprintf(stderr, "%s     .Byte8.Packet_Type ....................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte8.Packet_Type);
	fprintf(stderr, "%s     .Byte8.Reserved3 ......................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte8.Reserved3);
	fprintf(stderr, "%s     .Byte8.Size .............................. %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Byte8.Size);
	fprintf(stderr, "%s     .Injection_Info.Reserved4 ................ %hu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Injection_Info.Reserved4);
	fprintf(stderr, "%s     .Injection_Info.Skip ..................... %hhu\n", prefix, desc->PacketHeader.NetworkHeader.pt2pt.Injection_Info.Skip);
	if (desc->PacketHeader.NetworkHeader.pt2pt.Byte8.Packet_Type == 0) {
		fprintf(stderr, "%s   .PacketHeader.messageUnitHeader.Packet_Types\n", prefix);
		fprintf(stderr, "%s     .Memory_FIFO.Reserved1 ................... %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Memory_FIFO.Reserved1);
		fprintf(stderr, "%s     .Memory_FIFO.Rec_FIFO_Id ................. %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Memory_FIFO.Rec_FIFO_Id);
		fprintf(stderr, "%s     .Memory_FIFO.Unused1 ..................... %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Memory_FIFO.Unused1);
		fprintf(stderr, "%s     .Memory_FIFO.Put_Offset_MSB .............. 0x%08hx\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Memory_FIFO.Put_Offset_MSB);
		fprintf(stderr, "%s     .Memory_FIFO.Put_Offset_LSB .............. 0x%08x\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Memory_FIFO.Put_Offset_LSB);
	} else if (desc->PacketHeader.NetworkHeader.pt2pt.Byte8.Packet_Type == 1) {
		fprintf(stderr, "%s   .PacketHeader.messageUnitHeader.Packet_Types\n", prefix);
		fprintf(stderr, "%s     .Direct_Put.Rec_Payload_Base_Address_Id .. %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Rec_Payload_Base_Address_Id);
		fprintf(stderr, "%s     .Direct_Put.Pacing ....................... %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Pacing);
		fprintf(stderr, "%s     .Direct_Put.Put_Offset_MSB ............... 0x%08hx\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Put_Offset_MSB);
		fprintf(stderr, "%s     .Direct_Put.Put_Offset_LSB ............... 0x%08x\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Put_Offset_LSB);
		fprintf(stderr, "%s     .Direct_Put.Unused1 ...................... %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Unused1);
		fprintf(stderr, "%s     .Direct_Put.Rec_Counter_Base_Address_Id .. %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Rec_Counter_Base_Address_Id);
		fprintf(stderr, "%s     .Direct_Put.Counter_Offset ............... 0x%016lx\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Direct_Put.Counter_Offset);
	} else if (desc->PacketHeader.NetworkHeader.pt2pt.Byte8.Packet_Type == 2) {
		fprintf(stderr, "%s   .PacketHeader.messageUnitHeader.Packet_Types\n", prefix);
		fprintf(stderr, "%s     .Remote_Get.Rget_Inj_FIFO_Id ............. %hu\n", prefix, desc->PacketHeader.messageUnitHeader.Packet_Types.Remote_Get.Rget_Inj_FIFO_Id);
	}
	fflush(stderr);
}

#define DUMP_DESCRIPTOR(desc)							\
({										\
	char prefix[1024];							\
	snprintf(prefix, 1023, "%s:%s():%d", __FILE__, __func__, __LINE__);	\
	dump_descriptor(prefix, (desc));					\
})



#define FI_BGQ_MU_TORUS_INJFIFO_COUNT (10)
#define FI_BGQ_MU_LOCAL_INJFIFO_COUNT (6)




/* expensive .. not for critical path! */
static
inline uint16_t fi_bgq_mu_calculate_fifo_map(BG_CoordinateMapping_t local,
		uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
		uint32_t t) {

	/* calculate the signed coordinate difference between the source and
	 * destination torus coordinates
	 */
	ssize_t dA = (ssize_t)a - (ssize_t)local.a;
	ssize_t dB = (ssize_t)b - (ssize_t)local.b;
	ssize_t dC = (ssize_t)c - (ssize_t)local.c;
	ssize_t dD = (ssize_t)d - (ssize_t)local.d;
	ssize_t dE = (ssize_t)e - (ssize_t)local.e;

	/* select the fifo based on the t coordinate only if local */
	if ((dA | dB | dC | dD | dE) == 0) {
		return (t & 0x01) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_LOCAL0 : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_LOCAL1;
	}

	uint64_t dcr_value;
	dcr_value = DCRReadUser(ND_500_DCR(CTRL_CUTOFFS));

	Personality_t personality;
	Kernel_GetPersonality(&personality, sizeof(personality));

	/* select either A- or A+ if communicating only along the A dimension */
	if ((dB | dC | dD | dE) == 0) {
		if (ND_ENABLE_TORUS_DIM_A & personality.Network_Config.NetFlags) {
			uint64_t cutoff;
			if (dA > 0) {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__A_PLUS_get(dcr_value);
				return (a > cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AM : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AP;
			} else {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__A_MINUS_get(dcr_value);
				return (a < cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AM;
			}
		} else {
			return (dA > 0) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AM;
		}
	}

	/* select either B- or B+ if communicating only along the B dimension */
	if ((dA | dC | dD | dE) == 0) {
		if (ND_ENABLE_TORUS_DIM_B & personality.Network_Config.NetFlags) {
			uint64_t cutoff;
			if (dB > 0) {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__B_PLUS_get(dcr_value);
				return (b > cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BM : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BP;
			} else {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__B_MINUS_get(dcr_value);
				return (b < cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BM;
			}
		} else {
			return (dB > 0) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BM;
		}
	}

	/* select either C- or C+ if communicating only along the C dimension */
	if ((dA | dB | dD | dE) == 0) {
		if (ND_ENABLE_TORUS_DIM_C & personality.Network_Config.NetFlags) {
			uint64_t cutoff;
			if (dC > 0) {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__C_PLUS_get(dcr_value);
				return (c > cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CM : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CP;
			} else {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__C_MINUS_get(dcr_value);
				return (c < cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CM;
			}
		} else {
			return (dC > 0) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CM;
		}
	}

	/* select either D- or D+ if communicating only along the D dimension */
	if ((dA | dB | dC | dE) == 0) {
		if (ND_ENABLE_TORUS_DIM_D & personality.Network_Config.NetFlags) {
			uint64_t cutoff;
			if (dD > 0) {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__D_PLUS_get(dcr_value);
				return (d > cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DM : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DP;
			} else {
				cutoff = ND_500_DCR__CTRL_CUTOFFS__D_MINUS_get(dcr_value);
				return (d < cutoff) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DM;
			}
		} else {
			return (dD > 0) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DM;
		}
	}

	/* select either E- or E+ if communicating only along the E dimension */
	if ((dA | dB | dC | dD) == 0) {
		/* the maximum 'e' dimension size is 2 - and is a torus */
		return (t & 0x01) ? MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_EP : MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_EM;
	}

	/* communicating along diagonal */
	/* TODO - OPTIMIZE - round-robin the fifo picking based on destination */
	if (dA > 0) {
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AP;
	} else if (dA < 0)
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_AM;

	if (dB > 0) {
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BP;
	} else if (dB < 0)
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_BM;

	if (dC > 0) {
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CP;
	} else if (dC < 0)
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_CM;

	if (dD > 0) {
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DP;
	} else if(dD < 0)
		return MUHWI_DESCRIPTOR_TORUS_FIFO_MAP_DM;

	assert(0);
	return 0xFFFFu;
}


/* any coordinate specified as '-1' will be replaced with the corresponding
 * coordinate of "this process"
 */
static inline
void fi_bgq_create_addr (int8_t a_coord, int8_t b_coord,
	int8_t c_coord, int8_t d_coord, int8_t e_coord,
	int8_t t_coord, int8_t rx, fi_addr_t * addr)
{
	union fi_bgq_addr * bgq_addr = (union fi_bgq_addr *) addr;

	Personality_t p;
	Kernel_GetPersonality(&p, sizeof(Personality_t));

	bgq_addr->a = (a_coord == -1) ? p.Network_Config.Acoord : a_coord;
	bgq_addr->b = (b_coord == -1) ? p.Network_Config.Bcoord : b_coord;
	bgq_addr->c = (c_coord == -1) ? p.Network_Config.Ccoord : c_coord;
	bgq_addr->d = (d_coord == -1) ? p.Network_Config.Dcoord : d_coord;
	bgq_addr->e = (e_coord == -1) ? p.Network_Config.Ecoord : e_coord;

	if (t_coord == -1) t_coord = Kernel_MyTcoord();
	bgq_addr->rx = (rx == -1) ? (64 / Kernel_ProcessCount()) * t_coord : rx;
	bgq_addr->fifo_map = 0;	/* FIFO_Map to self? doesn't make sense, so set to zero to catch any usage and fix it */
	bgq_addr->is_local =
		(bgq_addr->a == p.Network_Config.Acoord) &&
		(bgq_addr->b == p.Network_Config.Bcoord) &&
		(bgq_addr->c == p.Network_Config.Ccoord) &&
		(bgq_addr->d == p.Network_Config.Dcoord) &&
		(bgq_addr->e == p.Network_Config.Ecoord);

	return;
}

static inline
void fi_bgq_create_addr_self (fi_addr_t * addr) {
	return fi_bgq_create_addr(-1, -1, -1, -1, -1, -1, -1, addr);
}

static inline
void fi_bgq_create_addr_self_cx (fi_addr_t * addr, int8_t cx) {
	return fi_bgq_create_addr(-1, -1, -1, -1, -1, -1, cx, addr);
}



#define FI_BGQ_DEBUG_MEMORY()					\
({								\
	fi_bgq_debug_memory(__FILE__, __func__, __LINE__);	\
})

static inline
void fi_bgq_debug_memory (char * file, const char * func, int line)
{
	uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;

	Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
	Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

	fprintf(stderr, "%s:%s():%d Allocated heap: %.2f MB, avail. heap: %.2f MB\n", file, func, line, (double)heap/(1024*1024),(double)heapavail/(1024*1024));
	fprintf(stderr, "%s:%s():%d Allocated stack: %.2f MB, avail. stack: %.2f MB\n", file, func, line, (double)stack/(1024*1024), (double)stackavail/(1024*1024));
	fprintf(stderr, "%s:%s():%d Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n", file, func, line, (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));

	return;
 }

static inline int fi_bgq_lock_if_required (struct l2atomic_lock *lock,
		int required)
{
	if (required) l2atomic_lock_acquire(lock);
	return 0;
}

static inline int fi_bgq_unlock_if_required (struct l2atomic_lock *lock,
		int required)
{
	if (required) l2atomic_lock_release(lock);
	return 0;
}

static inline uint64_t fi_bgq_cnk_vaddr2paddr(const void * vaddr, size_t len, uint64_t * paddr)
{
	Kernel_MemoryRegion_t cnk_mr;
	uint32_t cnk_rc;
	cnk_rc = Kernel_CreateMemoryRegion(&cnk_mr, (void *)vaddr, len);
	if (cnk_rc) return cnk_rc;

	*paddr = (uint64_t)cnk_mr.BasePa + ((uint64_t)vaddr - (uint64_t)cnk_mr.BaseVa);
	return 0;
}

enum fi_bgq_msync_type {
	FI_BGQ_MSYNC_TYPE_RW,
	FI_BGQ_MSYNC_TYPE_RO,
	FI_BGQ_MSYNC_TYPE_WO,
	FI_BGQ_MSYNC_TYPE_LAST
};

static inline void fi_bgq_msync(const enum fi_bgq_msync_type type)
{
	if (type == FI_BGQ_MSYNC_TYPE_RW || type == FI_BGQ_MSYNC_TYPE_WO) {
		/* this "l1p flush" hack is only needed to flush *writes*
		 * from a processor cache to the memory system */
		volatile uint64_t *mu_register =
			(volatile uint64_t *)(BGQ_MU_STATUS_CONTROL_REGS_START_OFFSET(0, 0) +
			0x030 - PHYMAP_PRIVILEGEDOFFSET);
		*mu_register = 0;
	}
	ppc_msync();
}

static inline void fi_bgq_mu_checks ()
{
	assert(sizeof(union fi_bgq_mu_packet_hdr) == sizeof(MUHWI_PacketHeader_t));
	assert(sizeof(union fi_bgq_addr) == sizeof(fi_addr_t));
	assert(sizeof(union fi_bgq_mu_descriptor) == sizeof(MUHWI_Descriptor_t));
}

/* ************************************************************************** *
 *
 * NOTE_MU_PADDR - The MU HWI documentation for MU descriptors states that
 * the physical address used for MU operations is 37 bits. However, the MSB
 * of this 37 bit physical address is used to indicate an atomic address and
 * will always be zero for normal physical addresses, and the maximum
 * physical address space depends on the amount of DRAM installed on the
 * compute nodes - which is only 16 GB. The physical addresses for main memory
 * begin at 0x0, and are contiguous until 64 GB, which means that the two
 * MSBs of the 36 bit physical address will always be zero.
 *
 * Unaligned non-atomic physical addresses can be safely specified using
 * only 34 bits in MU operations.
 *
 * Atomic physical addresses must be 8-byte-aligned which means that the
 * corresponding non-atomic physical address will always have the three
 * LSBs set to zero. A non-atomic physical address to be used for an atomic
 * physical address can be right-shifted 3 bits and can be safely specified
 * using only 31 bits when transferred as metadata. For MU operations the
 * physical address will be expanded to 37 bits as expected by the hardware.
 *
 * - MUHWI_Descriptor_t                 (hwi/include/bqc/MU_Descriptor.h)
 * - MUHWI_MessageUnitHeader_t          (hwi/include/bqc/MU_MessageUnitHeader.h)
 * - MUHWI_ATOMIC_ADDRESS_INDICATOR     (hwi/include/bqc/MU_Addressing.h)
 * - PHYMAP_MAXADDR_MAINMEMORY          (hwi/include/bqc/PhysicalMap.h)
 *
 * ************************************************************************** */

/* ************************************************************************** *
 *
 * NOTE_MU_RECFIFO - There are 16 "user" MU groups (not including the 17th MU
 * group which is normally used by cnk and agents) and there are 16 MU
 * reception fifos in each group (BGQ_MU_NUM_REC_FIFOS_PER_GROUP). There are
 * 2 MU reception fifos allocated to each ofi receive context - one for
 * "tagged" transfers (critical for MPI performance), and one for all other
 * transfers (not critical for MPI performance).
 *
 * This means that there are a maximum of 128 ofi receive contexts on a compute
 * node which must be allocated between all processes, domains, and endpoints
 * on that compute node.
 *
 * ************************************************************************** */

#endif /* _FI_PROV_BGQ_MU_H_ */

