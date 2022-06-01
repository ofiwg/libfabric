#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2015 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2015 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* Copyright (c) 2003-2014 Intel Corporation. All rights reserved. */

#ifndef PSM_HAL_GEN1_TYPES_H
#define PSM_HAL_GEN1_TYPES_H

/* some basic datatypes used throughout the gen1 HAL */

#define LAST_RHF_SEQNO 13

/* HAL assumes that the rx hdr q and the egr buff q are circular lists
 with two important indexes:

 head - software takes from this side of the circular list
 tail - hardware deposits new content here

The indexes advance in the list 0, 1, 2, 3, ... until they reach the value:
(number_of_entries_in_the_q-1), then the next value they take is 0.  And,
so, that is why these are called circular lists.

When the head idx == tail idx, that represents an empty circular list.

A completely full circular list is when:

    head_idx == (tail_idx + 1) % number_of_entries_in_the_q

Both indexes will always be in the range: 0 <= index < number_of_entries_in_the_q

After software receives the packet in the slot corresponding to the head idx,
and processes it completely, software will signal to the hardware that the slot
is available for re-use by retiring it - see api below for details.

Note that these are simplified assumptions for the benefit of the hardware independent
layer of PSM.  The actual implementation details are hidden in the hal instances.

Note that subcontexts have a collection of head / tail indexes for their use.

So, HAL supports the use of the following circular lists dealing with the
following entities:

1. Rx Hdr q - corresponding to hardware (software modifies head index, hardware modifies tail index).
2. Rx egr q - corresponding to hardware (software modifies head index, hardware modifies tail index).
3. Rx Hdr q - corresponding to a subcontext (software modifies both head and tail indexes).
4. Rx egr q - corresponding to a subcontext (software modifies both head and tail indexes).

Declare a type to indicate a circular list index:
*/
typedef uint32_t psm3_gen1_cl_idx;

typedef enum
{
	PSM3_GEN1_CL_Q_RX_HDR_Q      =  0, /* HW context for the rx hdr q. */
	PSM3_GEN1_CL_Q_RX_EGR_Q      =  1, /* HW context for the rx eager q. */
	/* Start of subcontexts (This is subcontext 0) */
	PSM3_GEN1_CL_Q_RX_HDR_Q_SC_0 =  2, /* Subcontext 0's rx hdr q. */
	PSM3_GEN1_CL_Q_RX_EGR_Q_SC_0 =  3, /* Subcontext 0's rx eager q. */

	/* Following SC 0's CL_Q's are the circular list q for subcontexts 1-7,
	   two per subcontext.  Even values are the rx hdr q for the subcontext
	   Odd value are for the eager q. */

/* Given a subcontext number (0-7), return the CL_Q for the RX HDR_Q: */
#define PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(SC) ((SC)*2 + PSM3_GEN1_CL_Q_RX_HDR_Q_SC_0)
/* Given a subcontext number (0-7), return the CL_Q for the RX EGR_Q: */
#define PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(SC) ((SC)*2 + PSM3_GEN1_CL_Q_RX_EGR_Q_SC_0)
} psm3_gen1_cl_q;

typedef struct
{
	volatile uint64_t *cl_q_head;
	volatile uint64_t *cl_q_tail;
	union
	{
		/* hdr_qe's are only present in *_RX_HDR_Q* CL Q types: */
		struct
		{
			uint32_t rx_hdrq_rhf_seq;
			uint32_t *p_rx_hdrq_rhf_seq;
			uint32_t *hdrq_base_addr;
		} hdr_qe;  /* header queue entry */
		/* egr_buffs's are only present in *_RX_EGR_Q* CL Q types: */
		void **egr_buffs;
	};
} psm3_gen1_cl_q_t;

typedef uint64_t psm3_gen1_raw_rhf_t;

typedef struct psm3_gen1_rhf_
{
	/* The first entity in rhf is the decomposed rhf.
	   psm3_gen1_get_receive_event(), we decompose the raw rhf
	   obtained from the hardware and deposit the data into this common
	   decomposed rhf, so the upper layers of psm can find the data in one
	   uniform place. */

	uint64_t decomposed_rhf;

	/* The second entry is the raw rhf that comes from the h/w.
	   The upper layers of psm should not use the raw rhf, instead use the
	   decomposed rhf above.  The raw rhf is intended for use by the HAL
	   instance only. */
	uint64_t raw_rhf;
} psm3_gen1_rhf_t;

#define PSM3_GEN1_RHF_ERR_ICRC_NBITS       1
#define PSM3_GEN1_RHF_ERR_ICRC_SHFTC      63
#define PSM3_GEN1_RHF_ERR_RSRV_NBITS       1
#define PSM3_GEN1_RHF_ERR_RSRV_SHFTC      62
#define PSM3_GEN1_RHF_ERR_ECC_NBITS        1
#define PSM3_GEN1_RHF_ERR_ECC_SHFTC       61
#define PSM3_GEN1_RHF_ERR_LEN_NBITS        1
#define PSM3_GEN1_RHF_ERR_LEN_SHFTC       60
#define PSM3_GEN1_RHF_ERR_TID_NBITS        1
#define PSM3_GEN1_RHF_ERR_TID_SHFTC       59
#define PSM3_GEN1_RHF_ERR_TFGEN_NBITS      1
#define PSM3_GEN1_RHF_ERR_TFGEN_SHFTC     58
#define PSM3_GEN1_RHF_ERR_TFSEQ_NBITS      1
#define PSM3_GEN1_RHF_ERR_TFSEQ_SHFTC     57
#define PSM3_GEN1_RHF_ERR_RTE_NBITS        3
#define PSM3_GEN1_RHF_ERR_RTE_SHFTC       56
#define PSM3_GEN1_RHF_ERR_DC_NBITS         1
#define PSM3_GEN1_RHF_ERR_DC_SHFTC        55
#define PSM3_GEN1_RHF_ERR_DCUN_NBITS       1
#define PSM3_GEN1_RHF_ERR_DCUN_SHFTC      54
#define PSM3_GEN1_RHF_ERR_KHDRLEN_NBITS    1
#define PSM3_GEN1_RHF_ERR_KHDRLEN_SHFTC   53
#define PSM3_GEN1_RHF_ALL_ERR_FLAGS_NBITS (PSM3_GEN1_RHF_ERR_ICRC_NBITS + PSM3_GEN1_RHF_ERR_RSRV_NBITS		\
					  	+ PSM3_GEN1_RHF_ERR_ECC_NBITS					\
						+ PSM3_GEN1_RHF_ERR_LEN_NBITS + PSM3_GEN1_RHF_ERR_TID_NBITS	\
						+ PSM3_GEN1_RHF_ERR_TFGEN_NBITS + PSM3_GEN1_RHF_ERR_TFSEQ_NBITS	\
						+ PSM3_GEN1_RHF_ERR_RTE_NBITS + PSM3_GEN1_RHF_ERR_DC_NBITS	\
						+ PSM3_GEN1_RHF_ERR_DCUN_NBITS + PSM3_GEN1_RHF_ERR_KHDRLEN_NBITS)
#define PSM3_GEN1_RHF_ALL_ERR_FLAGS_SHFTC 53
#define PSM3_GEN1_RHF_EGR_BUFF_OFF_NBITS  12
#define PSM3_GEN1_RHF_EGR_BUFF_OFF_SHFTC  32
#define PSM3_GEN1_RHF_SEQ_NBITS		  4
#define PSM3_GEN1_RHF_SEQ_SHFTC		 28
#define PSM3_GEN1_RHF_EGR_BUFF_IDX_NBITS  11
#define PSM3_GEN1_RHF_EGR_BUFF_IDX_SHFTC  16
#define PSM3_GEN1_RHF_USE_EGR_BUFF_NBITS   1
#define PSM3_GEN1_RHF_USE_EGR_BUFF_SHFTC  15
#define PSM3_GEN1_RHF_RX_TYPE_NBITS        3
#define PSM3_GEN1_RHF_RX_TYPE_SHFTC       12
#define PSM3_GEN1_RHF_PKT_LEN_NBITS       12
#define PSM3_GEN1_RHF_PKT_LEN_SHFTC        0

typedef enum {
	PSM3_GEN1_RHF_RX_TYPE_EXPECTED = 0,
	PSM3_GEN1_RHF_RX_TYPE_EAGER    = 1,
	PSM3_GEN1_RHF_RX_TYPE_NON_KD   = 2,
	PSM3_GEN1_RHF_RX_TYPE_ERROR    = 3
} psm3_gen1_rhf_rx_type;

#define PSM3_GEN1_RHF_UNPACK(A,NAME)				((uint32_t)((A.decomposed_rhf >>	\
									PSM3_GEN1_RHF_ ## NAME ## _SHFTC	\
									) &  PSMI_NBITS_TO_MASK(	\
									 PSM3_GEN1_RHF_ ## NAME ## _NBITS)))
/* define constants for the decomposed rhf error masks.
   Note how each of these are shifted by the ALL_ERR_FLAGS shift count. */

#define PSM3_GEN1_RHF_ERR_MASK_64(NAME)				((uint64_t)(((PSMI_NBITS_TO_MASK( \
									PSM3_GEN1_RHF_ERR_ ## NAME ## _NBITS) << \
									PSM3_GEN1_RHF_ERR_ ## NAME ## _SHFTC ))))
#define PSM3_GEN1_RHF_ERR_MASK_32(NAME)				((uint32_t)(PSM3_GEN1_RHF_ERR_MASK_64(NAME) >> \
									   PSM3_GEN1_RHF_ALL_ERR_FLAGS_SHFTC))
#define PSM3_GEN1_RHF_ERR_ICRC					PSM3_GEN1_RHF_ERR_MASK_32(ICRC)
#define PSM3_GEN1_RHF_ERR_ECC					PSM3_GEN1_RHF_ERR_MASK_32(ECC)
#define PSM3_GEN1_RHF_ERR_LEN					PSM3_GEN1_RHF_ERR_MASK_32(LEN)
#define PSM3_GEN1_RHF_ERR_TID					PSM3_GEN1_RHF_ERR_MASK_32(TID)
#define PSM3_GEN1_RHF_ERR_TFGEN					PSM3_GEN1_RHF_ERR_MASK_32(TFGEN)
#define PSM3_GEN1_RHF_ERR_TFSEQ					PSM3_GEN1_RHF_ERR_MASK_32(TFSEQ)
#define PSM3_GEN1_RHF_ERR_RTE					PSM3_GEN1_RHF_ERR_MASK_32(RTE)
#define PSM3_GEN1_RHF_ERR_DC					PSM3_GEN1_RHF_ERR_MASK_32(DC)
#define PSM3_GEN1_RHF_ERR_DCUN					PSM3_GEN1_RHF_ERR_MASK_32(DCUN)
#define PSM3_GEN1_RHF_ERR_KHDRLEN				PSM3_GEN1_RHF_ERR_MASK_32(KHDRLEN)

#define psm3_gen1_rhf_get_use_egr_buff(A)			PSM3_GEN1_RHF_UNPACK(A,USE_EGR_BUFF)
#define psm3_gen1_rhf_get_egr_buff_index(A)			PSM3_GEN1_RHF_UNPACK(A,EGR_BUFF_IDX)
#define psm3_gen1_rhf_get_egr_buff_offset(A)			PSM3_GEN1_RHF_UNPACK(A,EGR_BUFF_OFF)
#define psm3_gen1_rhf_get_packet_length(A)			(PSM3_GEN1_RHF_UNPACK(A,PKT_LEN)<<2)
#define psm3_gen1_rhf_get_all_err_flags(A)			PSM3_GEN1_RHF_UNPACK(A,ALL_ERR_FLAGS)
#define psm3_gen1_rhf_get_seq(A)					PSM3_GEN1_RHF_UNPACK(A,SEQ)

#define psm3_gen1_rhf_get_rx_type(A)				PSM3_GEN1_RHF_UNPACK(A,RX_TYPE)
#define PSM3_GEN1_RHF_PACK(NAME,VALUE)				((uint64_t)((((uint64_t)(VALUE)) & \
									PSMI_NBITS_TO_MASK( \
									PSM3_GEN1_RHF_ ## NAME ## _NBITS \
									)) << ( \
									PSM3_GEN1_RHF_ ## NAME ## _SHFTC )))
#endif /* PSM_HAL_GEN1_TYPES_H */
#endif /* PSM_OPA */
