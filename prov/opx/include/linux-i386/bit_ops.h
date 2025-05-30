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

#ifndef _HFI_i386_BIT_OPS_H
#define _HFI_i386_BIT_OPS_H

static __inline__ void ips_clear_bit(int nr, volatile unsigned long *addr)
{
	asm volatile(LOCK_PREFIX "btrl %1,%0" : "=m"(*addr) : "dIr"(nr));
}

static __inline__ void ips_change_bit(int nr, volatile unsigned long *addr)
{
	asm volatile(LOCK_PREFIX "btcl %1,%0" : "=m"(*addr) : "dIr"(nr));
}

static __inline__ int ips_test_and_set_bit(int nr, volatile unsigned long *addr)
{
	int oldbit;

	asm volatile(LOCK_PREFIX "btsl %2,%1\n\tsbbl %0,%0" : "=r"(oldbit), "=m"(*addr) : "dIr"(nr) : "memory");
	return oldbit;
}

static __inline__ void ips___clear_bit(int nr, volatile unsigned long *addr)
{
	asm volatile("btrl %1,%0" : "=m"(*addr) : "dIr"(nr));
}

static __inline__ void ips___change_bit(int nr, volatile unsigned long *addr)
{
	asm volatile("btcl %1,%0" : "=m"(*addr) : "dIr"(nr));
}

static __inline__ int ips___test_and_set_bit(int nr, volatile unsigned long *addr)
{
	int oldbit;

	asm volatile("btsl %2,%1\n\tsbbl %0,%0" : "=r"(oldbit), "=m"(*addr) : "dIr"(nr) : "memory");
	return oldbit;
}

#endif /* _HFI_i386_BIT_OPS_H */
