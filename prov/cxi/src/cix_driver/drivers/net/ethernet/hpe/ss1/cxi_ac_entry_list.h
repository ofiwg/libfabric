// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

#ifndef _CXI_AC_ENTRY_LIST_H_
#define _CXI_AC_ENTRY_LIST_H_

/* Access Control Entries */

#define AC_ENTRY_GFP_OPTS            (GFP_KERNEL)

#define AC_ENTRY_ID_MIN              (1)
#define AC_ENTRY_ID_MAX              (INT_MAX)
#define AC_ENTRY_ID_LIMITS           (XA_LIMIT(AC_ENTRY_ID_MIN, \
					       AC_ENTRY_ID_MAX))
#define AC_ENTRY_ID_XARRAY_FLAGS     (XA_FLAGS_ALLOC1)
#define AC_ENTRY_UID_XARRAY_FLAGS    (0)
#define AC_ENTRY_GID_XARRAY_FLAGS    (0)

#endif /* _CXI_AC_ENTRY_LIST_H_ */
