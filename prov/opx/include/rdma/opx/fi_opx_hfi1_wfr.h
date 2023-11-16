/*
 * Copyright (C) 2024 Cornelis Networks.
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
#ifndef _OPX_HFI1_WFR_H_
#define _OPX_HFI1_WFR_H_

/* WFR unused define (documentation) */
#define OPX_PBC_WFR_STATICRCC_SHIFT         0          /* HFI_PBC_STATICRCC_SHIFT   */
#define OPX_PBC_WFR_STATICRCC_MASK          0xffff     /* HFI_PBC_STATICRCC_MASK    */

/* Fields that unused on WFR (zero will be OR'd) */
#define OPX_PBC_WFR_UNUSED            0UL

#define OPX_PBC_WFR_DLID(_dlid)      OPX_PBC_WFR_UNUSED
#define OPX_PBC_WFR_SCTXT(_ctx)      OPX_PBC_WFR_UNUSED
#define OPX_PBC_WFR_L2COMPRESSED(_c) OPX_PBC_WFR_UNUSED
#define OPX_PBC_WFR_PORTIDX(_pidx)   OPX_PBC_WFR_UNUSED
#define OPX_PBC_WFR_LRH_DLID_TO_PBC_DLID(_dlid)    OPX_PBC_WFR_UNUSED

#ifndef NDEBUG
/* Debug only JKR definition for this sanity check */
#define _OPX_PBC_JKR_L2TYPE_9B_          0b11

    __OPX_FORCE_INLINE__
    uint32_t opx_pbc_wfr_l2type(unsigned _type)
    {
    	/* Just verify WFR isn't attempting non-9B */
    	assert(_type == _OPX_PBC_JKR_L2TYPE_9B_);
    	return OPX_PBC_WFR_UNUSED;
    }
#define OPX_PBC_WFR_L2TYPE(_type)    opx_pbc_wfr_l2type(_type)
#else
#define OPX_PBC_WFR_L2TYPE(_type)    OPX_PBC_WFR_UNUSED
#endif

#define OPX_PBC_WFR_RUNTIME(_dlid, _pidx) OPX_PBC_WFR_UNUSED

/* Unused WFR field - always initialized with PBC to 0. 
  #define OPX_PBC_STATICRCC(srcc) (((unsigned long long)(dlid & OPX_PBC_WFR_STATICRCC_MASK) << OPX_PBC_WFR_STATICRCC_SHIFT) << OPX_PBC_MSB_SHIFT)
 */

/* WFR
 *
 * The RHF.RcvSeq field is located in LSB bits [31:28] and values are in
 * the range of (1..13) inclusive. A new packet is available when the
 * expected sequence number in the next header queue element matches
 * the RHF.RcvSeq field.
 *
 * Instead of shifting and masking the RHF bits to retrieve the
 * sequence number in the range of 1..13 (or, 0x1..0xD) use only a bit
 * mask to obtain the RHF sequence in the range of 0x10000000..0xD0000000.
 * In this scheme the expected sequence number is incremented by
 * 0x10000000 instead of 0x1.
 */

#define OPX_WFR_RHF_SEQ_NOT_MATCH(_seq, _rhf)   (_seq != (_rhf & 0xF0000000ul))
#define OPX_WFR_RHF_SEQ_INCREMENT(_seq)         ((_seq < 0xD0000000ul) * _seq + 0x10000000ul)
#define OPX_WFR_IS_ERRORED_RHF(_rhf)            (_rhf & 0xFFE0000000000000ul)
#define OPX_WFR_RHF_SEQ_MATCH(_seq, _rhf)       (_seq == (_rhf & 0xF0000000ul))
#define OPX_WFR_RHF_SEQ_INIT_VAL                (0x10000000ul)
#define OPX_WFR_RHF_IS_USE_EGR_BUF(_rhf)        ((_rhf & 0x00008000ul) == 0x00008000ul)
#define OPX_WFR_RHF_EGRBFR_INDEX_MASK           (0x7FF)
#define OPX_WFR_RHF_EGRBFR_INDEX_SHIFT          (16)
#define OPX_WFR_RHF_EGR_INDEX(_rhf)             ((_rhf >> OPX_WFR_RHF_EGRBFR_INDEX_SHIFT) & OPX_WFR_RHF_EGRBFR_INDEX_MASK)
#define OPX_WFR_RHF_EGR_OFFSET(_rhf)            ((_rhf >> 32) & 0x0FFFul)
#define OPX_WFR_RHF_HDRQ_OFFSET(_rhf)           ((_rhf >> (32 + 12)) & 0x01FFul)

#define OPX_WFR_RHF_LENERR     (0x40000000u)
#define OPX_WFR_RHF_KHDRLENERR (0x00400000u)

#endif
