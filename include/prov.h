/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#ifndef _PROV_H_
#define _PROV_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

/* dl providers ctor and dtors are called when loaded and unloaded */
#define EXT_INI \
	__attribute__((visibility ("default"))) \
	__attribute__((alias(fi_prov_ini)))

#define EXT_FINI \
	__attribute__((visibility ("default"))) \
	__attribute__((alias(fi_prov_fini)))

/* ctor and dtor function signatures */
#define INI_SIG(name) void name(void)
#define FINI_SIG(name) void name(void)

/* for each provider defines for three scenarios:
 * dl: externally visible ctor and dtor, aliased to fi_prov_*
 * built-in: ctor and dtor function defs, don't export symbols
 * not built: no-op calls for ctor/dtor
*/

#define VERBS_IF fi_verbs_ini
#define VERBS_DF fi_verbs_fini

#if (HAVE_VERBS) && (HAVE_VERBS_DL)
#  define VERBS_INI EXT_INI INI_SIG(VERBS_IF)
#  define VERBS_FINI EXT_FINI FINI_SIG(VERBS_DF)
#  define VERBS_C
#  define VERBS_D
#elif (HAVE_VERBS)
#  define VERBS_INI INI_SIG(VERBS_IF)
#  define VERBS_FINI INI_SIG(VERBS_DF)
#  define VERBS_C VERBS_IF();
#  define VERBS_D VERBS_DF();
VERBS_INI ;
VERBS_FINI ;
#else
#  define VERBS_C
#  define VERBS_D
#endif

#define PSM_IF fi_psm_ini
#define PSM_DF fi_psm_fini

#if (HAVE_PSM) && (HAVE_PSM_DL)
#  define PSM_INI EXT_INI INI_SIG(PSM_IF)
#  define PSM_FINI EXT_FINI FINI_SIG(PSM_DF)
#  define PSM_C
#  define PSM_D
#elif (HAVE_PSM)
#  define PSM_INI INI_SIG(PSM_IF)
#  define PSM_FINI FINI_SIG(PSM_DF)
#  define PSM_C PSM_IF();
#  define PSM_D PSM_DF();
PSM_INI ;
PSM_FINI ;
#else
#  define PSM_C
#  define PSM_D
#endif

#define SOCKETS_IF fi_sockets_ini
#define SOCKETS_DF fi_sockets_fini

#if (HAVE_SOCKETS) && (HAVE_SOCKETS_DL)
#  define SOCKETS_INI EXT_INI INI_SIG(SOCKETS_IF)
#  define SOCKETS_FINI EXT_FINI FINI_SIG(SOCKETS_DF)
#  define SOCKETS_C
#  define SOCKETS_D
#elif (HAVE_SOCKETS)
#  define SOCKETS_INI INI_SIG(SOCKETS_IF)
#  define SOCKETS_FINI FINI_SIG(SOCKETS_DF)
#  define SOCKETS_C SOCKETS_IF();
#  define SOCKETS_D SOCKETS_DF();
SOCKETS_INI ;
SOCKETS_FINI ;
#else
#  define SOCKETS_C
#  define SOCKETS_D
#endif

#define USNIC_IF fi_usnic_ini
#define USNIC_DF fi_usnic_fini

#if (HAVE_USNIC) && (HAVE_USNIC_DL)
#  define USNIC_INI EXT_INI INI_SIG(USNIC_IF)
#  define USNIC_FINI EXT_FINI FINI_SIG(USNIC_DF)
#  define USNIC_C
#  define USNIC_D
#elif (HAVE_USNIC)
#  define USNIC_INI INI_SIG(USNIC_IF)
#  define USNIC_FINI FINI_SIG(USNIC_DF)
#  define USNIC_C USNIC_IF();
#  define USNIC_D USNIC_DF();
USNIC_INI ;
USNIC_FINI ;
#else
#  define USNIC_C
#  define USNIC_D
#endif

#endif /* _PROV_H_ */
