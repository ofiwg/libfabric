/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2022 Intel Corporation.

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

  Copyright(c) 2022 Intel Corporation.

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

#ifndef UTILS_ENV_H
#define UTILS_ENV_H

#include "psm2_mock_testing.h"

/* we can only include low level headers here because this is
 * #included by utils_sysfs.c.  Can't pull in HAL headers or heap debug macros
 */

// we front end getenv with a check in the psm3.conf file
#define PSM3_ENV_FILENAME "/etc/psm3.conf"

int psm3_env_initialize(void);
void psm3_env_finalize(void);
char* psm3_env_get(const char *name);

/*
 * Parsing environment variables
 */

union psmi_envvar_val {
	void *e_void;
	char *e_str;
	int e_int;
	unsigned int e_uint;
	long e_long;
	unsigned long e_ulong;
	unsigned long long e_ulonglong;
};

#define PSMI_ENVVAR_LEVEL_USER	         1
#define PSMI_ENVVAR_LEVEL_HIDDEN         2
#define PSMI_ENVVAR_LEVEL_NEVER_PRINT    4

#define PSMI_ENVVAR_TYPE_YESNO		0
#define PSMI_ENVVAR_TYPE_STR		1
#define PSMI_ENVVAR_TYPE_INT		2
#define PSMI_ENVVAR_TYPE_UINT		3
#define PSMI_ENVVAR_TYPE_UINT_FLAGS	4
#define PSMI_ENVVAR_TYPE_LONG		5
#define PSMI_ENVVAR_TYPE_ULONG		6
#define PSMI_ENVVAR_TYPE_ULONG_FLAGS	7
#define PSMI_ENVVAR_TYPE_ULONG_ULONG    8
#define PSMI_ENVVAR_TYPE_STR_VAL_PAT    9
#define PSMI_ENVVAR_TYPE_STR_TUPLES    10

#define PSMI_ENVVAR_VAL_YES ((union psmi_envvar_val) 1)
#define PSMI_ENVVAR_VAL_NO  ((union psmi_envvar_val) 0)

int
MOCKABLE(psm3_getenv)(const char *name, const char *descr, int level,
		int type, union psmi_envvar_val defval,
		union psmi_envvar_val *newval);
MOCK_DCL_EPILOGUE(psm3_getenv);

/*
 * Parsing int and unsigned int parameters
 * 0 -> ok, *val updated
 * -1 -> empty string
 * -2 -> parse error
 */
int psm3_parse_str_int(const char *string, int *val);
int psm3_parse_str_uint(const char *string, unsigned int *val);

/*
 * Parse long parameters
 * -1 -> empty string
 * -2 -> parse error
 */
long psm3_parse_str_long(const char *str);

/*
 * Parsing yesno parameters
 * allows: yes/no, true/false, on/off, 1/0
 * -1 -> empty string
 * -2 -> parse error
 */
int psm3_parse_str_yesno(const char *str);

/*
 * Parsing int parameters set in string tuples.
 */
int psm3_parse_str_tuples(const char *str, int ntup, int *vals);

/* parse env of the form 'val' or 'val:' or 'val:pattern' */
int psm3_parse_val_pattern(const char *env, unsigned def, unsigned *val);

#if defined(PSM_VERBS) || defined(PSM_SOCKETS)
// return forced speed in mbps or 0 if not forced
unsigned long psm3_parse_force_speed();
// this can be overridden in compiler args, default is 32,000 mbps (32 gbps)
// if set to 0, no default (lack of speed is then fatal)
#ifndef PSM_DEFAULT_SPEED
#define PSM_DEFAULT_SPEED 32000
#endif
#endif

#endif /* UTILS_ENV_H */
