/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <gni_pub.h>
#include "gnix.h"
#include "gnix_util.h"
#include "fi.h"
#include "prov.h"

/**
 * @note  To make sure that static linking will work, there must be at
 *        least one symbol in the file that requires gnix_init.o to have
 *        to be linked in when building the executable. This insures the
 *        ctor will run even with static linking.
 */

atomic_t gnix_id_counter;
atomic_t file_id_counter;
#ifndef NDEBUG
/* don't think this needs to be in tls */
__thread pid_t gnix_debug_pid = ~(uint32_t) 0;
__thread uint32_t gnix_debug_tid = ~(uint32_t) 0;
atomic_t gnix_debug_next_tid;
#endif

uint8_t precomputed_crc_results[256];

static inline uint8_t __gni_crc_bits(uint8_t data)
{
  uint8_t lcrc = 0;

  if(data & 1)
    lcrc ^= 0x5e;
  if(data & 2)
    lcrc ^= 0xbc;
  if(data & 4)
    lcrc ^= 0x61;
  if(data & 8)
    lcrc ^= 0xc2;
  if(data & 0x10)
    lcrc ^= 0x9d;
  if(data & 0x20)
    lcrc ^= 0x23;
  if(data & 0x40)
    lcrc ^= 0x46;
  if(data & 0x80)
    lcrc ^= 0x8c;

  return lcrc;
}

void __setup_precomputed_crcs(void)
{
	int i;

	for (i = 0; i < 256; i++)
	{
		precomputed_crc_results[i] = __gni_crc_bits(i);
	}
}

/**
 * Initialization function for performing global setup
 */
__attribute__((constructor))
void gnix_init(void)
{
	__setup_precomputed_crcs();
	atomic_initialize(&gnix_id_counter, 0);
	atomic_initialize(&file_id_counter, 0);
#ifndef NDEBUG
	atomic_initialize(&gnix_debug_next_tid, 0);
#endif
}
