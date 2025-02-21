/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#ifndef _EFA_SHARED_H
#define _EFA_SHARED_H

#define EFA_FABRIC_NAME	       "efa"
#define EFA_DIRECT_FABRIC_NAME "efa-direct"

#define EFA_INFO_TYPE_IS_RDM(_info)                                        \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM) && \
	 !strcasecmp(_info->fabric_attr->name, EFA_FABRIC_NAME))

#define EFA_INFO_TYPE_IS_DIRECT(_info)                                     \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM) && \
	 !strcasecmp(_info->fabric_attr->name, EFA_DIRECT_FABRIC_NAME))

#endif /* _EFA_SHARED_H */
