#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2021 Intel Corporation.

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

  Copyright(c) 2021 Intel Corporation.

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

/* Copyright (c) 2003-2021 Intel Corporation. All rights reserved. */

/* This file implements the HAL specific code for PSM PTL for ips RDMA */
#include "psm_user.h"
#include "psm2_hal.h"
#include "ptl_ips.h"
#include "psm_mq_internal.h"
#include "gen1_hal.h"

// The value returned is a bitmask of IPS_PROTOEXP_FLAG_* selections
// When reload==1, we refetch the env variable and reload the cached value
// While this can also be used to set additional flags (TID_DEBUG,
// RTS_CTS_INTERLEAVE and CTS_SERIALIZED), it should not.
// TID_DEBUG and CTS_SERIALIZED are automatically set when appropriate,
// and there is an env variable for RTS_CTS_INTERLEAVE.
unsigned psm3_gen1_parse_tid(int reload)
{
	union psmi_envvar_val envval;
	static int have_value = 0;
	static unsigned saved;

	// only parse once so doesn't appear in PSM3_VERBOSE_ENV multiple times
	if (!reload && have_value)
		return saved;

	psm3_getenv("PSM3_TID",
		"Tid proto flags (0 disables protocol)",
		PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT_FLAGS,
		(union psmi_envvar_val)IPS_PROTOEXP_FLAG_TID,
		&envval);
	saved = envval.e_uint;
	have_value = 1;
	return saved;
}
#endif /* PSM_OPA */
