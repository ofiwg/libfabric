/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_AUTH_H_
#define _CXIP_AUTH_H_

/* Function declarations */
int cxip_check_auth_key_info(struct fi_info *info);

int cxip_gen_auth_key(struct fi_info *info, struct cxi_auth_key *key);

#endif /* _CXIP_AUTH_H_ */
