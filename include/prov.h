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

#include <rdma/fi_prov.h>

/* for each provider defines for three scenarios:
 * dl: externally visible ctor and dtor, aliased to fi_prov_*
 * built-in: ctor and dtor function defs, don't export symbols
 * not built: no-op calls for ctor/dtor
*/

#if (HAVE_VERBS) && (HAVE_VERBS_DL)
#  define VERBS_INI EXT_INI
#  define VERBS_FINI EXT_FINI
#  define VERBS_INIT NULL
#  define VERBS_DEINIT NULL
#elif (HAVE_VERBS)
#  define VERBS_INI INI_SIG(fi_verbs_ini)
#  define VERBS_FINI FINI_SIG(fi_verbs_fini)
#  define VERBS_INIT fi_verbs_ini()
#  define VERBS_DEINIT fi_verbs_fini()
VERBS_INI ;
VERBS_FINI ;
#else
#  define VERBS_INIT NULL
#  define VERBS_DEINIT NULL
#endif

#if (HAVE_PSM) && (HAVE_PSM_DL)
#  define PSM_INI EXT_INI
#  define PSM_FINI EXT_FINI
#  define PSM_INIT NULL
#  define PSM_DEINIT NULL
#elif (HAVE_PSM)
#  define PSM_INI INI_SIG(fi_psm_ini)
#  define PSM_FINI FINI_SIG(fi_psm_fini)
#  define PSM_INIT fi_psm_ini()
#  define PSM_DEINIT fi_psm_fini()
PSM_INI ;
PSM_FINI ;
#else
#  define PSM_INIT NULL
#  define PSM_DEINIT NULL
#endif

#if (HAVE_SOCKETS) && (HAVE_SOCKETS_DL)
#  define SOCKETS_INI EXT_INI
#  define SOCKETS_FINI EXT_FINI
#  define SOCKETS_INIT NULL
#  define SOCKETS_DEINIT NULL
#elif (HAVE_SOCKETS)
#  define SOCKETS_INI INI_SIG(fi_sockets_ini)
#  define SOCKETS_FINI FINI_SIG(fi_sockets_fini)
#  define SOCKETS_INIT fi_sockets_ini()
#  define SOCKETS_DEINIT fi_sockets_fini()
SOCKETS_INI ;
SOCKETS_FINI ;
#else
#  define SOCKETS_INIT NULL
#  define SOCKETS_DEINIT NULL
#endif

#if (HAVE_USNIC) && (HAVE_USNIC_DL)
#  define USNIC_INI EXT_INI
#  define USNIC_FINI EXT_FINI
#  define USNIC_INIT NULL
#  define USNIC_DEINIT NULL
#elif (HAVE_USNIC)
#  define USNIC_INI INI_SIG(fi_usnic_ini)
#  define USNIC_FINI FINI_SIG(fi_usnic_fini)
#  define USNIC_INIT fi_usnic_ini()
#  define USNIC_DEINIT fi_usnic_fini()
USNIC_INI ;
USNIC_FINI ;
#else
#  define USNIC_INIT NULL
#  define USNIC_DEINIT NULL
#endif

#endif /* _PROV_H_ */
