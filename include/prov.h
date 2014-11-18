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

#define EXT_INI __attribute__((visibility ("default"))) void fi_prov_ini(void)
#define EXT_FINI __attribute__((visibility ("default"))) void fi_prov_fini(void)

#define INI_SIG(name) void name(void)

#if (HAVE_VERBS) && (HAVE_VERBS_DL)
#  define VERBS_INI EXT_INI
#  define VERBS_FINI EXT_FINI
#elif (HAVE_VERBS)
#  define VERBS_C fi_verbs_ini
#  define VERBS_D fi_verbs_fini
#  define VERBS_INI INI_SIG(VERBS_C)
#  define VERBS_FINI INI_SIG(VERBS_D)
VERBS_INI ;
VERBS_FINI ;
#else
#  define VERBS_INI
#  define VERBS_FINI
#endif

#if (HAVE_PSM) && (HAVE_PSM_DL)
#  define PSM_INI EXT_INI
#  define PSM_FINI EXT_FINI
#elif (HAVE_PSM)
#  define PSM_C fi_psm_ini
#  define PSM_D fi_psm_fini
#  define PSM_INI void fi_psm_ini(void)
#  define PSM_FINI void fi_psm_fini(void)
PSM_INI ;
PSM_FINI ;
#else
#  define VERBS_INI
#  define VERBS_FINI
#endif

#if (HAVE_SOCKETS) && (HAVE_SOCKETS_DL)
#  define SOCKETS_INI EXT_INI
#  define SOCKETS_FINI EXT_FINI
#elif (HAVE_SOCKETS)
#  define SOCKETS_C fi_sockets_ini
#  define SOCKETS_D fi_sockets_fini
#  define SOCKETS_INI void fi_sockets_ini(void)
#  define SOCKETS_FINI void fi_sockets_fini(void)
SOCKETS_INI ;
SOCKETS_FINI ;
#else
#  define SOCKETS_INI
#  define SOCKETS_FINI
#endif

#if (HAVE_USNIC) && (HAVE_USNIC_DL)
#  define USNIC_INI EXT_INI
#  define USNIC_FINI EXT_FINI
#elif (HAVE_USNIC)
#  define USNIC_C fi_usnic_ini
#  define USNIC_D fi_usnic_fini
#  define USNIC_INI void fi_usnic_ini(void)
#  define USNIC_FINI void fi_usnic_fini(void)
USNIC_INI ;
USNIC_FINI ;
#else
#  define USNIC_INI
#  define USNIC_FINI
#endif

#endif /* _PROV_H_ */
