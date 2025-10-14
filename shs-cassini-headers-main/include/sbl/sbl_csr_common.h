// SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
/*
 * Common utility defines.
 * Copyright 2020 Cray Inc. All rights reserved
 */
#ifndef _SBL_CSR_COMMON_H_
#define _SBL_CSR_COMMON_H_

/* Extra massaging of Cassini definitions */
#define C_CSR_MASK(type,member) ({uint64_t mask=0; union type t={0}; \
			t.member=-1; mask=t.qw; mask;})
#define C_CSR_WORD_MASK(type,member,word) ({uint64_t mask=0; union type t={0}; \
			t.member=-1; mask=t.qw[word]; mask;})
#define C_CSR_OFFSET(type,member) ({int offset=0; union type t={0}; \
			t.member=1; while (((t.qw >> offset) & 1) == 0) \
					    {offset++;} offset;})
#define C_CSR_WORD_OFFSET(type,member,word) ({int offset=0; union type t={0}; \
			t.member=1; while (((t.qw[word] >> offset) & 1) == 0) \
					    {offset++;} offset;})

/* UPDATE defines */
#define C_UPDATE(val, value, type, member) (((val)&(~C_CSR_MASK(type,member)))|C_SET(value,type,member))

/* SET defines */
#define C_SET(value, type, member) ((((unsigned long long) (value)) << C_CSR_OFFSET(type,member)) & C_CSR_MASK(type,member))

/* GET defines */
#define C_GET(value, type, member) (((value) & C_CSR_MASK(type,member)) >> C_CSR_OFFSET(type,member))
#define C_WORD_GET(value, type, member, word) (((value) & C_CSR_WORD_MASK(type,member,word)) >> C_CSR_WORD_OFFSET(type,member,word))

#endif /* _SBL_CSR_COMMON_H_ */
