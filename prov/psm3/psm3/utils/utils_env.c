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

#include <malloc.h>             /* malloc_usable_size */
#include <fnmatch.h>
#include <ctype.h>
#include "psm_user.h"

static struct psm3_env_var {
	char *name;
	char *value;
} *psm3_env;

static unsigned psm3_env_count;
static unsigned psm3_env_alloc;
static unsigned psm3_env_initialized;

#define PSM3_ENV_INC 20	// number of additional entries per env realloc

// Absolute max non-comment lines in psm3.conf
// There are less then 200 unique PSM3 env variables so 1024 is more than enough
#define PSM3_ENV_LIMIT 1024

// parse the /etc/psm3.conf file
// info prints are used to warn about malformed lines ignored
// this is called early in psm3_init() so all info prints will still occur
// but PSM3_VERBOSE_ENV can be exported or placed early in the file to
// enable prints of values as they are read
// returns 0 on success, -1 on error
int psm3_env_initialize(void)
{
	FILE *f;
	char buf[1024];
	int env_log_level = 1;	// log controlled by TRACEMASK
	unsigned verb_env_val;

	if (psm3_env_initialized)
		return 0;	// already initialized

	// get verbosity level setting for env logging
	// if invalid syntax, will output warning when parse during psm3_getenv
	const char *verb_env = getenv("PSM3_VERBOSE_ENV");
	(void)psm3_parse_val_pattern(verb_env, 0, &verb_env_val);
	if (verb_env_val)
		env_log_level = 0;	// log at INFO level

	f = fopen(PSM3_ENV_FILENAME, "r");
	if (f == NULL) {
		if (errno == ENOENT) {
			// file is optional
			_HFI_ENVDBG(env_log_level, "%s: not found\n", PSM3_ENV_FILENAME);
			return 0;
		} else {
			_HFI_ERROR("Unable to open %s: %s\n",
					PSM3_ENV_FILENAME, strerror(errno));
			return -1;
		}
	}
	while (fgets(buf, sizeof(buf), f) != NULL) {
		int l = strlen(buf);
		int i, j;
		struct psm3_env_var var;
		char *p;

		if (! l) continue;	// should not happen
		if (buf[l-1] != '\n') {
			// can also occur for last line if no newline at end
			int c;
			c = fgetc(f);
			if (c != EOF) {
				// line too long, fgetc until read newline
				_HFI_INFO("%s: Ignoring line too long: '%s' ...\n",
						PSM3_ENV_FILENAME, buf);
				while (c != (int)(unsigned char)'\n' && (c = fgetc(f)) != EOF)
					;
				continue;
			}
		} else {
			buf[l-1] = '\0';	// drop newline
		}

		// drop any comment at end of line
		p = strchr(buf, '#');
		if (p)
			*p = '\0';

		// skip leading white space
		i = strspn(buf, " \t");

		if (buf[i] == '\0') continue;	// skip blank lines

		// drop any trailing whitespace
		l = strlen(&buf[i]);
		while (l>0 && isspace(buf[i+l-1])) {
			buf[i+l-1] = '\0';
			l--;
		}

		// length of variable name
		j = strspn(&buf[i], "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_");
		if (buf[i+j] != '=') {
			// malformed assignment,skip
			_HFI_INFO("%s: Ignoring malformed assignment: '%s'\n",
					PSM3_ENV_FILENAME, buf);
			continue;
		}
		buf[i+j] = '\0';

		if (psm3_env_count >= PSM3_ENV_LIMIT) {
			_HFI_ERROR("%s: Limit of %u entries\n",
						PSM3_ENV_FILENAME, PSM3_ENV_LIMIT);
			goto fail;
		}

		var.name = psmi_strdup(PSMI_EP_NONE, &buf[i]);
		if (! var.name) {
			_HFI_ERROR("%s: Unable to allocate memory for entry %s\n",
						PSM3_ENV_FILENAME, &buf[i]);
			goto fail;
		}
		// skip name and '=', rest is value
		var.value = psmi_strdup(PSMI_EP_NONE, &buf[i+j+1]);
		if (! var.value) {
			_HFI_ERROR("%s: Unable to allocate memory for entry %s\n",
						PSM3_ENV_FILENAME, &buf[i]);
			psmi_free(var.name);
			goto fail;
		}

		// allow psm3.env to set PSM3_VERBOSE_ENV when defaulted
		// if invalid syntax, will output warning when parse during psm3_getenv
		if (! verb_env && 0 == strcmp("PSM3_VERBOSE_ENV", var.name)) {
			(void)psm3_parse_val_pattern(var.value, 0, &verb_env_val);
			if (verb_env_val)
				env_log_level = 0;	// log at INFO level
		}
		// Note: we don't let PSM3_TRACEMASK affect this

		// this must be parsed in a constructor prior to this function,
		// so we ignore it here
		if (0 == strcmp(var.name, "PSM3_DISABLE_MMAP_MALLOC")) {
			_HFI_INFO("WARNING: %s Ignoring %s\n", PSM3_ENV_FILENAME,var.name);
			psmi_free(var.name);
			psmi_free(var.value);
			continue;
		}

		_HFI_ENVDBG(env_log_level, "%s: parsed %s='%s'\n",
				PSM3_ENV_FILENAME, var.name, var.value);

		if (psm3_env_count >= psm3_env_alloc) {
			unsigned n = psm3_env_alloc + PSM3_ENV_INC;
			void *p = psmi_realloc(PSMI_EP_NONE, UNDEFINED, psm3_env,
									sizeof(psm3_env[0])*n);
			if (! p) {
				_HFI_ERROR("%s: Unable to allocate memory for %u entries\n",
							PSM3_ENV_FILENAME, n);
				psmi_free(var.name);
				psmi_free(var.value);
				goto fail;
			}
			psm3_env_alloc = n;
			psm3_env = (struct psm3_env_var*)p;
		}
		psm3_env[psm3_env_count++] = var;
	}
	if (! feof(f) || ferror(f)) {
		_HFI_ERROR("Error reading %s\n", PSM3_ENV_FILENAME);
		goto fail;
	}
	fclose(f);
	psm3_env_initialized = 1;
	return 0;

fail:
	psm3_env_finalize();
	fclose(f);
	return -1;
}

void psm3_env_finalize(void)
{
	int i;

	if (! psm3_env_initialized)
		return;

	for (i=0; i < psm3_env_count; i++) {
		psmi_free(psm3_env[i].name);
		psmi_free(psm3_env[i].value);
	}
	psmi_free(psm3_env);
	psm3_env = NULL;
	psm3_env_count = 0;
	psm3_env_alloc = 0;
	psm3_env_initialized = 0;
}

// check getenv first and then psm3_env (/etc/psm3.env)
char *psm3_env_get(const char *name)
{
	char *ret = getenv(name);
	int i;

	if (ret || ! psm3_env_initialized || ! psm3_env_count)
		return ret;

	for (i=psm3_env_count-1; i >= 0;  i--) {
		if (0 == strcmp(name, psm3_env[i].name))
			return psm3_env[i].value;
	}
	return NULL;
}

/* _CONSUMED_ALL() is a macro which indicates if strtol() consumed all
   of the input passed to it. */
#define _CONSUMED_ALL(CHAR_PTR) (((CHAR_PTR) != NULL) && (*(CHAR_PTR) == 0))

// don't document that 3 and 3: and 3:pattern can output hidden params
const char *PSM3_VERBOSE_ENV_HELP =
				"Enable verbose output of environment variables. "
				"(0 - none, 1 - changed w/o help, 2 - user help, "
				"#: - limit output to rank 0, #:pattern - limit output "
				"to processes whose label matches "
#ifdef FNM_EXTMATCH
				"extended "
#endif
				"glob pattern";

/* If PSM3_VERBOSE_ENV is set in the environment, we determine
 * what its verbose level is and print the environment at "INFO"
 * level if the environment's level matches the desired printlevel.
 */
static int psmi_getenv_verblevel = -1;
static int psm3_getenv_is_verblevel(int printlevel)
{
	if (psmi_getenv_verblevel == -1) {
		// first call, parse PSM3_VERBOSE_ENV and set verbosity level
		// for other env vars (and itself)
		char *env = psm3_env_get("PSM3_VERBOSE_ENV");
		int nlevel = PSMI_ENVVAR_LEVEL_USER;
		unsigned verb_env_val;
		int ret = psm3_parse_val_pattern(env, 0, &verb_env_val);
		psmi_getenv_verblevel = verb_env_val;
		if (psmi_getenv_verblevel < 0 || psmi_getenv_verblevel > 3)
			psmi_getenv_verblevel = 2;
		if (psmi_getenv_verblevel > 0)
			nlevel = 0; /* output at INFO level */
		if (ret == -2)
			_HFI_ENVDBG(0, "Invalid value for %s ('%s') %-40s Using: %u\n",
				"PSM3_VERBOSE_ENV", env, PSM3_VERBOSE_ENV_HELP, verb_env_val);
		else if (psmi_getenv_verblevel == 1)
			_HFI_ENVDBG(0, " %-25s => '%s' (default was '%s')\n",
				"PSM3_VERBOSE_ENV", env?env:"", "0");
		else if (env && *env)
			_HFI_ENVDBG(nlevel, " %-25s %-40s => '%s' (default was '%s')\n",
				"PSM3_VERBOSE_ENV", PSM3_VERBOSE_ENV_HELP, env, "0");
		else	/* defaulted */
			_HFI_ENVDBG(nlevel,
				" %-25s %-40s => '%s'\n",
				"PSM3_VERBOSE_ENV", PSM3_VERBOSE_ENV_HELP, "0");
	}
	// printlevel is visibility of env (USER=1 or HIDDEN=2)
	// so at verbosity 1 and 2 output USER
	// at verbosity 3 output USER and HIDDEN
	return ((printlevel <= psmi_getenv_verblevel
			&& psmi_getenv_verblevel == 1)
		|| printlevel <= psmi_getenv_verblevel-1);
}

#define GETENV_PRINTF(_level, _fmt, ...)				\
	do {								\
		/* NEVER_PRINT disables output for deprecated variables */ \
		if ((_level & PSMI_ENVVAR_LEVEL_NEVER_PRINT) == 0)	\
		{							\
			int nlevel = _level;				\
			/* when enabled by VERBOSE_ENV, output at info (0), */ \
			/* otherwise output at nlevel (USER=1, HIDDEN=2 */ \
			if (psm3_getenv_is_verblevel(nlevel))		\
				nlevel = 0; /* output at INFO level */	\
			_HFI_ENVDBG(nlevel, _fmt, ##__VA_ARGS__);	\
		}							\
	} while (0)

// count number of fields in a str_tuple (field:field:....)
// The number is number of colons + 1
static int psm3_count_tuples(const char *str)
{
	int ret = 1;
	if (! str)
		return 0;
	while (*str) {
		if (*str == ':')
			ret++;
		str++;
	}
	return ret;
}

int
MOCKABLE(psm3_getenv)(const char *name, const char *descr, int level,
	    int type, union psmi_envvar_val defval,
	    union psmi_envvar_val *newval)
{
	int used_default = 0;
	union psmi_envvar_val tval;
	char *env = psm3_env_get(name);
#if _HFI_DEBUGGING
	int ishex = (type == PSMI_ENVVAR_TYPE_ULONG_FLAGS ||
		     type == PSMI_ENVVAR_TYPE_UINT_FLAGS);
#endif

	/* for verblevel 1 we only output non-default values with no help
	 * for verblevel>1 we promote to info (verblevel=2 promotes USER,
	 *		verblevel=3 promotes HIDDEN) and show help.
	 * for verblevel< 1 we don't promote anything and show help
	 */
#define _GETENV_PRINT(env, used_default, fmt, val, defval) \
	do {	\
		(void)psm3_getenv_is_verblevel(level);			\
		if (env && *env && used_default)				\
			_HFI_INFO("Invalid value for %s ('%s') %-40s Using: %s" fmt "\n", \
				name, env, descr, ishex ? "0x" : " ", val);	\
		else if (used_default && psmi_getenv_verblevel != 1)		\
			GETENV_PRINTF(level, "%s%-25s %-40s =>%s" fmt	\
				"\n", level > 1 ? "*" : " ", name,	\
				descr, ishex ? "0x" : " ", val);	\
		else if (! used_default && psmi_getenv_verblevel == 1)	\
			GETENV_PRINTF(1, "%s%-25s =>%s"			\
				fmt " (default was%s" fmt ")\n",	\
				level > 1 ? "*" : " ", name,		\
				ishex ? " 0x" : " ", val,		\
				ishex ? " 0x" : " ", defval);		\
		else if (! used_default && psmi_getenv_verblevel != 1)	\
			GETENV_PRINTF(1, "%s%-25s %-40s =>%s"		\
				fmt " (default was%s" fmt ")\n",	\
				level > 1 ? "*" : " ", name, descr,	\
				ishex ? " 0x" : " ", val,		\
				ishex ? " 0x" : " ", defval);		\
	} while (0)

#define _CONVERT_TO_NUM(DEST,TYPE,STRTOL)						\
	do {										\
		char *ep;								\
		/* Avoid base 8 (octal) on purpose, so don't pass in 0 for radix */	\
		DEST = (TYPE)STRTOL(env, &ep, 10);					\
		if (! _CONSUMED_ALL(ep)) {						\
			DEST = (TYPE)STRTOL(env, &ep, 16);				\
			if (! _CONSUMED_ALL(ep)) {					\
				used_default = 1;					\
				tval = defval;						\
			}								\
		}									\
	} while (0)

	switch (type) {
	case PSMI_ENVVAR_TYPE_YESNO:
		tval.e_int = psm3_parse_str_yesno(env);
		if (tval.e_int < 0) {
			tval = defval;
			used_default = 1;
		}
		_GETENV_PRINT(env, used_default, "%s", tval.e_int ? "YES" : "NO",
			      defval.e_int ? "YES" : "NO");
		break;

	case PSMI_ENVVAR_TYPE_STR:
		if (!env || *env == '\0') {
			tval = defval;
			used_default = 1;
		} else
			tval.e_str = env;
		_GETENV_PRINT(env, used_default, "'%s'", tval.e_str, defval.e_str);
		break;

	case PSMI_ENVVAR_TYPE_INT:
		if (!env || *env == '\0') {
			tval = defval;
			used_default = 1;
		} else {
			_CONVERT_TO_NUM(tval.e_int,int,strtol);
		}
		_GETENV_PRINT(env, used_default, "%d", tval.e_int, defval.e_int);
		break;

	case PSMI_ENVVAR_TYPE_UINT:
	case PSMI_ENVVAR_TYPE_UINT_FLAGS:
		if (!env || *env == '\0') {
			tval = defval;
			used_default = 1;
		} else {
			_CONVERT_TO_NUM(tval.e_int,unsigned int,strtoul);
		}
		if (type == PSMI_ENVVAR_TYPE_UINT_FLAGS)
			_GETENV_PRINT(env, used_default, "%x", tval.e_uint,
				      defval.e_uint);
		else
			_GETENV_PRINT(env, used_default, "%u", tval.e_uint,
				      defval.e_uint);
		break;

	case PSMI_ENVVAR_TYPE_LONG:
		if (!env || *env == '\0') {
			tval = defval;
			used_default = 1;
		} else {
			_CONVERT_TO_NUM(tval.e_long,long,strtol);
		}
		_GETENV_PRINT(env, used_default, "%ld", tval.e_long, defval.e_long);
		break;
	case PSMI_ENVVAR_TYPE_ULONG_ULONG:
		if (!env || *env == '\0') {
			tval = defval;
			used_default = 1;
		} else {
			_CONVERT_TO_NUM(tval.e_ulonglong,unsigned long long,strtoull);
		}
		_GETENV_PRINT(env, used_default, "%llu",
			      tval.e_ulonglong, defval.e_ulonglong);
		break;
	case PSMI_ENVVAR_TYPE_STR_VAL_PAT:
		{
			unsigned trash;
			// we parse just for syntax check, caller must parse again
			if (psm3_parse_val_pattern(env, 0, &trash) < 0) {
				tval = defval;
				used_default = 1;
			} else
				tval.e_str = env;
			_GETENV_PRINT(env, used_default, "'%s'", tval.e_str, defval.e_str);
		}
		break;
	case PSMI_ENVVAR_TYPE_STR_TUPLES:
		{
			// we parse just for syntax check, caller must parse again
			int vals[3];
			int ntup = psm3_count_tuples(defval.e_str);
			psmi_assert_always(ntup > 0 && ntup <= 3);
			// parse default into vals[] so can show what caller get
			(void)psm3_parse_str_tuples(defval.e_str, ntup, vals);
			switch (psm3_parse_str_tuples(env, ntup, vals)) {
			case -1:	// empty, use default
				tval = defval;
				used_default = 1;
				_GETENV_PRINT(env, 1, "'%s'", tval.e_str, defval.e_str);
				break;
			case -2:	// one or more fields with bad syntax, show what we have
				tval.e_str = env;
				// only 3 choices, so just bruteforce it
				switch (ntup) {
				case 1:
					_HFI_INFO("Invalid value for %s ('%s') %-40s Using: %d\n",
						name, env, descr, vals[0]);
					break;
				case 2:
					_HFI_INFO("Invalid value for %s ('%s') %-40s Using: %d:%d\n",
						name, env, descr, vals[0], vals[1]);
					break;
				case 3:
					_HFI_INFO("Invalid value for %s ('%s') %-40s Using: %d:%d:%d\n",
						name, env, descr, vals[0], vals[1], vals[2]);
					break;
				}
				break;
			default:	// valid string
				tval.e_str = env;
				_GETENV_PRINT(env, 0, "'%s'", tval.e_str, defval.e_str);
				break;
			}
		}
		break;
	case PSMI_ENVVAR_TYPE_ULONG:
	case PSMI_ENVVAR_TYPE_ULONG_FLAGS:
	default:
		if (!env || *env == '\0') {
			tval = defval;
			used_default = 1;
		} else {
			_CONVERT_TO_NUM(tval.e_ulong,unsigned long,strtoul);
		}
		if (type == PSMI_ENVVAR_TYPE_ULONG_FLAGS)
			_GETENV_PRINT(env, used_default, "%lx", tval.e_ulong,
				      defval.e_ulong);
		else
			_GETENV_PRINT(env, used_default, "%lu", tval.e_ulong,
				      defval.e_ulong);
		break;
	}
#undef _GETENV_PRINT
	*newval = tval;

	return used_default;
}
MOCK_DEF_EPILOGUE(psm3_getenv);

/*
 * Parsing int parameters
 * 0 -> ok, *val updated
 * -1 -> empty string
 * -2 -> parse error
 */
int psm3_parse_str_int(const char *string, int *val)
{
	char *ep;
	long ret;

	psmi_assert(val != NULL);
	if (! string || ! *string)
		return -1;
	/* Avoid base 8 (octal) on purpose, so don't pass in 0 for radix */
	ret = strtol(string, &ep, 10);
	if (! _CONSUMED_ALL(ep)) {
		ret = strtol(string, &ep, 16);
		if (! _CONSUMED_ALL(ep))
			return -2;
	}
	*val = ret;
	return 0;
}

/*
 * Parsing uint parameters
 * 0 -> ok, *val updated
 * -1 -> empty string
 * -2 -> parse error
 */
int psm3_parse_str_uint(const char *string, unsigned int *val)
{
	char *ep;
	unsigned long ret;

	psmi_assert(val != NULL);
	if (! string || ! *string)
		return -1;
	/* Avoid base 8 (octal) on purpose, so don't pass in 0 for radix */
	ret = strtoul(string, &ep, 10);
	if (! _CONSUMED_ALL(ep)) {
		ret = strtoul(string, &ep, 16);
		if (! _CONSUMED_ALL(ep))
			return -2;
	}
	*val = ret;
	return 0;
}

/*
 * Parsing long parameters
 * -1 -> empty string
 * -2 -> parse error
 */
long psm3_parse_str_long(const char *string)
{
	char *ep;
	long ret;

	if (! string || ! *string)
		return -1;
	/* Avoid base 8 (octal) on purpose, so don't pass in 0 for radix */
	ret = strtol(string, &ep, 10);
	if (! _CONSUMED_ALL(ep)) {
		ret = strtol(string, &ep, 16);
		if (! _CONSUMED_ALL(ep))
			return -2;
	}
	return ret;
}

/*
 * Parsing yesno parameters
 * allows: yes/no, true/false, on/off, 1/0
 * -1 -> empty string
 * -2 -> parse error
 */
int psm3_parse_str_yesno(const char *string)
{
	if (! string || ! *string)
		return -1;
	else if (string[0] == 'Y' || string[0] == 'y'
				|| string[0] == 'T' || string[0] == 't'
				|| ((string[0] == 'O' || string[0] == 'o')
					&& (string[1] == 'n' || string[1] == 'N')))
		return 1;
	else if (string[0] == 'N' || string[0] == 'n'
				|| string[0] == 'F' || string[0] == 'f'
				|| ((string[0] == 'O' || string[0] == 'o')
					&& (string[1] == 'f' || string[1] == 'F')))
		return 0;
	else {
		char *ep;
		unsigned long temp;
		temp = strtoul(string, &ep, 0);
		if (!_CONSUMED_ALL(ep)) {
			return -2;
		} else if (temp != 0) {
			return 1;
		} else {
			return 0;
		}
	}
}

/* parse env of the form 'val' or 'val:' or 'val:pattern'
 * for PSM3_VERBOSE_ENV, PSM3_TRACEMASK, PSM3_FI and PSM3_IDENITFY
 * 0 - parsed and matches current process, *val set to parsed val
 * 0 - parsed and doesn't match current process, *val set to def
 * -1 - nothing provided, *val set to def
 * -2 - syntax error, *val set to def
 */
int psm3_parse_val_pattern(const char *env, unsigned def, unsigned *val)
{
	int ret = 0;

	psmi_assert(val != NULL);
	if (!env || ! *env) {
		*val = def;
		ret = -1;
	} else {
		char *e = psmi_strdup(NULL, env);
		char *ep;
		char *p;

		psmi_assert_always(e != NULL);
		if (e == NULL)	// for klocwork
			goto done;
		p = strchr(e, ':');
		if (p)
			*p = '\0';
		*val = (int)strtoul(e, &ep, 0);
		if (! _CONSUMED_ALL(ep)) {
			*val = def;
			ret = -2;
		} else if (p) {
			if (! *(p+1)) { // val: -> val:*:rank0
				if (psm3_get_myrank() != 0)
					*val = def;
			} else if (0 != fnmatch(p+1, psm3_get_mylabel(),  0
#ifdef FNM_EXTMATCH
										| FNM_EXTMATCH
#endif
					))
					*val = def;
		}
		psmi_free(e);
	}
done:
	return ret;
}

/*
 * Parsing int parameters set in string tuples.
 * Output array int *vals should be able to store 'ntup' elements
 * and should be initialized by caller with default values for each field.
 * Values are only overwritten if they are parsed.
 * Tuples are always separated by colons ':'
 * Empty parameters are left unchanged in vals[]
 * It's valid for less than ntup values to be supplied, any unsupplied
 * fields are not updated in vals[]
 * Returns:
 * 	0 - parsed with no errors, vals[] updated
 * 	-1 - empty or NULL string, vals[] unchanged
 * 	-2 -  syntax error in one of more of the parameters
 * 			parameters with syntax errors are unchanged, others without
 * 			syntax errors are updated in vals[]
 */
int psm3_parse_str_tuples(const char *string, int ntup, int *vals)
{
	char *b = (char *)string;
	char *e = b;
	int tup_i = 0;
	int ret = 0;

	psmi_assert(vals != NULL);
	if (! string || ! *string)
		return -1;

	char *buf = psmi_strdup(NULL, string);
	psmi_assert_always(buf != NULL);
	if (! buf)	// for klocwork
		return 0;

	while (*e && tup_i < ntup) {
		b = e;
		while (*e && *e != ':')
			e++;
		if (e > b) {	/* something to parse */
			char *ep;
			int len = e - b;
			long int l;
			strncpy(buf, b, len);
			buf[len] = '\0';
			l = strtol(buf, &ep, 0);
			if (ep != buf) {	/* successful conversion */
				vals[tup_i] = (int)l;
			} else {
				ret = -2;
			}
		}
		if (*e == ':')
			e++;	/* skip delimiter */
		tup_i++;
	}
	if (*e)	// too many tuples
		ret = -2;
	psmi_free(buf);
	return ret;
}

#if defined(PSM_VERBS) || defined(PSM_SOCKETS)
// return forced speed in mbps or 0 if not forced
unsigned long psm3_parse_force_speed()
{
	union psmi_envvar_val envval;
	static int have_value = 0;
	static unsigned long saved;

	// only parse once so doesn't appear in PSM3_VERBOSE_ENV multiple times
	if (have_value)
		return saved;

	psm3_getenv("PSM3_FORCE_SPEED", "Override for device link speed file in /sys/class.  Specified in mbps. Default is 0 [no override]",
			PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_ULONG_FLAGS,
			(union psmi_envvar_val)0 /* Disabled by default */,
			&envval);
	saved = envval.e_ulong;
	have_value = 1;
	return saved;
}
#endif /* defined(PSM_VERBS) || defined(PSM_SOCKETS) */
