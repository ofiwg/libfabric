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
#ifndef _OPX_HFI1_VERSION_H_
#define _OPX_HFI1_VERSION_H_

#include "rdma/opx/fi_opx_hfi1_wfr.h"
#include "rdma/opx/fi_opx_hfi1_jkr.h"


/*******************************************/
/* These are the same defines both WFR/JKR */
/*******************************************/
// RHF changes
// Common to both JKR/WFR

#define OPX_RHF_RCV_TYPE_EXPECTED_RCV(_rhf, _noop) ((_rhf & 0x00007000ul) == 0x00000000ul)
#define OPX_RHF_RCV_TYPE_EAGER_RCV(_rhf, _noop)    ((_rhf & 0x00001000ul) == 0x00001000ul)
#define OPX_RHF_RCV_TYPE_OTHER(_rhf, _noop)        ((_rhf & 0x00006000ul) != 0x00000000ul)


#define OPX_PBC_CR(cr, _noop) ((cr & FI_OPX_HFI1_PBC_CR_MASK) << FI_OPX_HFI1_PBC_CR_SHIFT)
#define OPX_PBC_LEN(len, _noop) (len)
#define OPX_PBC_VL(vl, _noop) ((vl & FI_OPX_HFI1_PBC_VL_MASK) << FI_OPX_HFI1_PBC_VL_SHIFT)

/* Note: double check JKR sc bits */
#define OPX_PBC_SC(sc, _noop) (((sc >> FI_OPX_HFI1_PBC_SC4_SHIFT) & FI_OPX_HFI1_PBC_SC4_MASK) << FI_OPX_HFI1_PBC_DCINFO_SHIFT)

/* PBC most significant bits shift (32 bits) defines */
#define OPX_MSB_SHIFT                   32


/***************************************************************/
/* Both JKR and WFR runtime is now supported (no longer doing  */
/* build-time constants)                                       */
/*                                                             */
/* Runtime support relies on a local variable "hfi1_type",     */
/*  which is likely passed down through macro and function     */
/*  constants which are selected/optimized inline with         */
/*  function tables.                                           */
/***************************************************************/

#define OPX_PBC_DLID(dlid, _hfi1_type)    ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_PBC_WFR_DLID(dlid) : OPX_PBC_JKR_DLID(dlid))

#define OPX_PBC_SCTXT(ctx, _hfi1_type)    ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_PBC_WFR_SCTXT(ctx) : OPX_PBC_JKR_SCTXT(ctx))

#define OPX_PBC_L2COMPRESSED(c, _hfi1_type)  ((_hfi1_type & OPX_HFI1_WFR) ?      \
    OPX_PBC_WFR_L2COMPRESSED(c) : OPX_PBC_JKR_L2COMPRESSED(c))

#define OPX_PBC_PORTIDX(pidx, _hfi1_type) ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_PBC_WFR_PORTIDX(pidx) : OPX_PBC_JKR_PORTIDX(pidx))

#define OPX_PBC_DLID_TO_PBC_DLID(dlid, _hfi1_type)  ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_PBC_WFR_DLID_TO_PBC_DLID(dlid) : OPX_PBC_JKR_DLID_TO_PBC_DLID(dlid))

#define OPX_PBC_L2TYPE(type, _hfi1_type) ((_hfi1_type & OPX_HFI1_WFR) ? \
    	OPX_PBC_WFR_L2TYPE(type) : OPX_PBC_JKR_L2TYPE(type))

/* One runtime check for mutiple fields - DLID, PORT, L2TYPE */
#define OPX_PBC_RUNTIME(_dlid, _pidx, _hfi1_type) ((_hfi1_type & OPX_HFI1_WFR) ?   \
     (OPX_PBC_WFR_DLID(_dlid) | OPX_PBC_WFR_PORTIDX(_pidx)) : \
 (OPX_PBC_JKR_DLID(_dlid) | OPX_PBC_JKR_PORTIDX(_pidx)))

/* Common BTH defines */

#define OPX_BTH_UNUSED 0  // Default unsupported values to 0

#define OPX_BTH_RC2(_rc2, _hfi1_type)    ((_hfi1_type & OPX_HFI1_JKR) ?         \
	OPX_BTH_JKR_RC2(_rc2) : OPX_BTH_UNUSED)
#define OPX_BTH_CSPEC(_cspec, _hfi1_type)   ((_hfi1_type & OPX_HFI1_JKR) ?         \
    OPX_BTH_JKR_CSPEC(_cspec) : OPX_BTH_UNUSED)
#define OPX_BTH_CSPEC_DEFAULT  OPX_BTH_UNUSED // Cspec is not used in 9B header
#define OPX_BTH_RC2_VAL(_hfi1_type)     ((_hfi1_type & OPX_HFI1_JKR) ?         \
    OPX_BTH_JKR_RC2_VAL     : OPX_BTH_UNUSED)

#define OPX_BTH_RX_SHIFT  56

/* Common RHF defines */

#define OPX_RHF_SEQ_NOT_MATCH(_seq, _rhf, _hfi1_type)   ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_SEQ_NOT_MATCH(_seq, _rhf) : OPX_JKR_RHF_SEQ_NOT_MATCH(_seq, _rhf))

#define OPX_RHF_SEQ_INCREMENT(_seq, _hfi1_type)    ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_SEQ_INCREMENT(_seq) : OPX_JKR_RHF_SEQ_INCREMENT(_seq))

#define OPX_IS_ERRORED_RHF(_rhf, _hfi1_type)       ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_IS_ERRORED_RHF(_rhf, _hfi1_type) : OPX_JKR_IS_ERRORED_RHF(_rhf, _hfi1_type))

#define OPX_RHF_SEQ_MATCH(_seq, _rhf, _hfi1_type)   ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_SEQ_MATCH(_seq, _rhf, _hfi1_type) : OPX_JKR_RHF_SEQ_MATCH(_seq, _rhf, _hfi1_type))

/* Init-time, let it use the variable - not optimized */
#define OPX_RHF_SEQ_INIT_VAL(_hfi1_type)   ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_SEQ_INIT_VAL : OPX_JKR_RHF_SEQ_INIT_VAL)

#define OPX_RHF_IS_USE_EGR_BUF(_rhf, _hfi1_type)   ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_IS_USE_EGR_BUF(_rhf) : OPX_JKR_RHF_IS_USE_EGR_BUF(_rhf))

#define OPX_RHF_EGR_INDEX(_rhf, _hfi1_type)      ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_EGR_INDEX(_rhf) : OPX_JKR_RHF_EGR_INDEX(_rhf))

#define OPX_RHF_EGR_OFFSET(_rhf, _hfi1_type)     ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_EGR_OFFSET(_rhf) : OPX_JKR_RHF_EGR_OFFSET(_rhf))

#define OPX_RHF_HDRQ_OFFSET(_rhf, _hfi1_type)    ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_HDRQ_OFFSET(_rhf) : OPX_JKR_RHF_HDRQ_OFFSET(_rhf))

#define OPX_RHE_DEBUG(_opx_ep, _rhe_ptr, _rhf_ptr, _rhf_msb, _rhf_lsb, _rhf_seq, _hdrq_offset, _rhf_rcvd, _hdr, _hfi1_type)    \
    ((_hfi1_type & OPX_HFI1_WFR) ?  \
    OPX_WFR_RHE_DEBUG(_opx_ep, _rhe_ptr, _rhf_ptr, _rhf_msb, _rhf_lsb, _rhf_seq, _hdrq_offset, _rhf_rcvd, _hdr, _hfi1_type) : \
    OPX_JKR_RHE_DEBUG(_opx_ep, _rhe_ptr, _rhf_ptr, _rhf_msb, _rhf_lsb, _rhf_seq, _hdrq_offset, _rhf_rcvd, _hdr, _hfi1_type))

#define OPX_RHF_CHECK_HEADER(_rhf_rcvd, _pktlen, _hfi1_type)     ((_hfi1_type & OPX_HFI1_WFR) ?         \
    OPX_WFR_RHF_CHECK_HEADER(_rhf_rcvd, _pktlen, _hfi1_type) : OPX_JKR_RHF_CHECK_HEADER(_rhf_rcvd, _pktlen, _hfi1_type))


#define OPX_HEADER_SIZE   (8 * 8)  // doesn't include PBC. For 9B it includes the unused_pad qw.

#endif



