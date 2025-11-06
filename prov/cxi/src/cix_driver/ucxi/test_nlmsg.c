// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Cray Inc. All rights reserved */

/* Sample to read messages from the CXI netlink socket */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <err.h>
#include <netlink/genl/genl.h>
#include <netlink/genl/ctrl.h>

#include "uapi/ethernet/cxi-abi.h"

/* Same as kernel version, but not const */
static struct nla_policy cxierr_genl_policy[CXIERR_GENL_ATTR_MAX] = {
	[CXIERR_GENL_ATTR_DEV_NUM] = { .type = NLA_U32 },
	[CXIERR_GENL_ATTR_CSR_FLG] = { .type = NLA_U32 },
	[CXIERR_GENL_ATTR_BIT] = { .type = NLA_U8 },
};

static int msg_cb(struct nl_msg *msg, void *arg)
{
	struct nlattr *attr[CXIERR_GENL_ATTR_MAX];

	genlmsg_parse(nlmsg_hdr(msg), 0, attr,
		      CXIERR_GENL_ATTR_MAX, cxierr_genl_policy);

	printf("CSR error message: device=%d, csr=%08x, bit=%d\n",
	       nla_get_u32(attr[CXIERR_GENL_ATTR_DEV_NUM]),
	       nla_get_u32(attr[CXIERR_GENL_ATTR_CSR_FLG]),
	       nla_get_u8(attr[CXIERR_GENL_ATTR_BIT]));

	return NL_OK;
}

int main(void)
{
	struct nl_sock *sock = NULL;
	struct nl_cb *cb = NULL;
	int rc;
	int fam_id;
	int grp_id;

	sock = nl_socket_alloc();
	if (sock == NULL)
		errx(1, "nl_socket_alloc error\n");

	nl_socket_disable_seq_check(sock);
	nl_socket_disable_auto_ack(sock);

	if (genl_connect(sock))
		errx(1, "genl_connect error\n");

	fam_id = genl_ctrl_resolve(sock, CXIERR_GENL_FAMILY_NAME);
	if (fam_id < 0)
		errx(1, "genl_ctrl_resolve error\n");

	grp_id = genl_ctrl_resolve_grp(sock, CXIERR_GENL_FAMILY_NAME,
				       CXIERR_GENL_MCAST_GROUP_NAME);
	if (grp_id < 0)
		errx(1, "genl_ctrl_resolve_grp error\n");

	if (nl_socket_add_membership(sock, grp_id))
		errx(1, "nl_socket_add_membership error\n");

	cb = nl_cb_alloc(NL_CB_DEFAULT);
	nl_cb_set(cb, NL_CB_VALID, NL_CB_CUSTOM, msg_cb, NULL);

	do {
		rc = nl_recvmsgs(sock, cb);
	} while (!rc);

	nl_cb_put(cb);
	nl_socket_free(sock);

	return EXIT_SUCCESS;
}
