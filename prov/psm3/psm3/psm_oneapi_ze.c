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
#ifdef PSM_ONEAPI
#include <dirent.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <sys/ioctl.h>
#include <linux/sockios.h>
#include <sys/poll.h>
#include "psm_user.h"
#include "psm_mq_internal.h"
#include "ptl_am/psm_am_internal.h"
#include "psmi_wrappers.h"

int psm3_ze_dev_fds[MAX_ZE_DEVICES];
int psm3_num_ze_dev_fds;

const char* psmi_oneapi_ze_result_to_string(const ze_result_t result) {
#define ZE_RESULT_CASE(RES) case ZE_RESULT_##RES: return STRINGIFY(RES)

	switch (result) {
	ZE_RESULT_CASE(SUCCESS);
	ZE_RESULT_CASE(NOT_READY);
	ZE_RESULT_CASE(ERROR_UNINITIALIZED);
	ZE_RESULT_CASE(ERROR_DEVICE_LOST);
	ZE_RESULT_CASE(ERROR_INVALID_ARGUMENT);
	ZE_RESULT_CASE(ERROR_OUT_OF_HOST_MEMORY);
	ZE_RESULT_CASE(ERROR_OUT_OF_DEVICE_MEMORY);
	ZE_RESULT_CASE(ERROR_MODULE_BUILD_FAILURE);
	ZE_RESULT_CASE(ERROR_INSUFFICIENT_PERMISSIONS);
	ZE_RESULT_CASE(ERROR_NOT_AVAILABLE);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_VERSION);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_FEATURE);
	ZE_RESULT_CASE(ERROR_INVALID_NULL_HANDLE);
	ZE_RESULT_CASE(ERROR_HANDLE_OBJECT_IN_USE);
	ZE_RESULT_CASE(ERROR_INVALID_NULL_POINTER);
	ZE_RESULT_CASE(ERROR_INVALID_SIZE);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_SIZE);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_ALIGNMENT);
	ZE_RESULT_CASE(ERROR_INVALID_SYNCHRONIZATION_OBJECT);
	ZE_RESULT_CASE(ERROR_INVALID_ENUMERATION);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_ENUMERATION);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_IMAGE_FORMAT);
	ZE_RESULT_CASE(ERROR_INVALID_NATIVE_BINARY);
	ZE_RESULT_CASE(ERROR_INVALID_GLOBAL_NAME);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_NAME);
	ZE_RESULT_CASE(ERROR_INVALID_FUNCTION_NAME);
	ZE_RESULT_CASE(ERROR_INVALID_GROUP_SIZE_DIMENSION);
	ZE_RESULT_CASE(ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
	ZE_RESULT_CASE(ERROR_INVALID_COMMAND_LIST_TYPE);
	ZE_RESULT_CASE(ERROR_OVERLAPPING_REGIONS);
	ZE_RESULT_CASE(ERROR_UNKNOWN);
	default:
		return "Unknown error";
	}

#undef ZE_RESULT_CASE
}

void psmi_oneapi_ze_memcpy(void *dstptr, const void *srcptr, size_t size)
{
	struct ze_dev_ctxt *ctxt;

	ctxt = psmi_oneapi_dev_ctxt_get(dstptr);
	if (!ctxt) {
		ctxt = psmi_oneapi_dev_ctxt_get(srcptr);
		if (!ctxt) {
			_HFI_ERROR("dst %p src %p not GPU buf for copying\n",
				   dstptr, srcptr);
			return;
		}
	}
	PSMI_ONEAPI_ZE_CALL(zeCommandListReset, ctxt->cl);
	PSMI_ONEAPI_ZE_CALL(zeCommandListAppendMemoryCopy, ctxt->cl, dstptr, srcptr, size, NULL, 0, NULL);
	PSMI_ONEAPI_ZE_CALL(zeCommandListClose, ctxt->cl);
	PSMI_ONEAPI_ZE_CALL(zeCommandQueueExecuteCommandLists, ctxt->cq, 1, &ctxt->cl, NULL);
	PSMI_ONEAPI_ZE_CALL(zeCommandQueueSynchronize, ctxt->cq, UINT32_MAX);
}

/*
 * psmi_ze_init_fds - initialize the file descriptors (ze_dev_fds) 
 *
 * The file descriptors are used in intra-node communication to pass to peers
 * via socket with sendmsg/recvmsg SCM_RIGHTS message type.
 *
 */

int psm3_ze_init_fds(void)
{
	const char *dev_dir = "/dev/dri/by-path/";
	const char *suffix = "-render";
	DIR *dir;
	struct dirent *ent = NULL;
	char dev_name[NAME_MAX];
	int i = 0, ret;

	dir = opendir(dev_dir);
	if (dir == NULL)
		return PSM2_INTERNAL_ERR;

	while ((ent = readdir(dir)) != NULL) {
		if (ent->d_name[0] == '.' ||
		    strstr(ent->d_name, suffix) == NULL)
			continue;

		memset(dev_name, 0, sizeof(dev_name));
		ret = snprintf(dev_name, NAME_MAX, "%s%s", dev_dir, ent->d_name);
		if (ret < 0 || ret >= NAME_MAX)
			goto err;

		psm3_ze_dev_fds[i] = open(dev_name, O_RDWR);
		if (psm3_ze_dev_fds[i] == -1)
			goto err;
		i++;
		psm3_num_ze_dev_fds++;
	}
	(void) closedir(dir);
	return PSM2_OK;

err:
	(void) closedir(dir);
	_HFI_INFO("Failed to open device %s\n", dev_name);
	return PSM2_INTERNAL_ERR;
}

/*
 * psmi_ze_get_dev_fds - fetch device file descriptors
 *
 * Returns a pointer to ze_dev_fds while putting the number
 * of fds into the in/out nfds parameter
 *
 */

int *psm3_ze_get_dev_fds(int *nfds)
{
	*nfds = psm3_num_ze_dev_fds;
	return psm3_ze_dev_fds;
}
/*
 * psmi_ze_get_num_dev_fds() - return number of device file descriptors
 *
 * Returns the number of ze_dev_fds
 *
 */

int psm3_ze_get_num_dev_fds(void)
{
	return psm3_num_ze_dev_fds;
}

/*
 * psmi_sendmsg_fds - send device file descriptors over socket w/ sendmsg
 *
 * Prepares message of type SCM_RIGHTS, copies file descriptors as payload,
 * and sends over socket via sendmsg
 *
 */

static int psmi_sendmsg_fds(int sock, int *fds, int nfds, psm2_epid_t epid)
{
	struct msghdr msg;
	struct cmsghdr *cmsg;
	struct iovec iov;
	int64_t peer_id = *(int64_t *)&epid;
	char *ctrl_buf;
	size_t ctrl_size;
	int ret;

	ctrl_size = sizeof(*fds) * nfds;
	ctrl_buf = (char *)psmi_calloc(NULL, UNDEFINED, 1, CMSG_SPACE(ctrl_size));
	if (!ctrl_buf)
		return ENOMEM;

	iov.iov_base = &peer_id;
	iov.iov_len = sizeof(peer_id);

	memset(&msg, 0, sizeof(msg));
	msg.msg_control = ctrl_buf;
	msg.msg_controllen = CMSG_SPACE(ctrl_size);

	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	cmsg = CMSG_FIRSTHDR(&msg);
	cmsg->cmsg_level = SOL_SOCKET;
	cmsg->cmsg_type = SCM_RIGHTS;
	cmsg->cmsg_len = CMSG_LEN(ctrl_size);
	memcpy(CMSG_DATA(cmsg), fds, ctrl_size);

	ret = sendmsg(sock, &msg, 0);
	if (ret <= 0) {
		ret = -EIO;
	}

	psmi_free(ctrl_buf);
	return ret;
}

/*
 * psmi_recvmsg_fds - receive device file descriptors from socket w/ recvmsg
 *
 * Prepares message buffer of type SCM_RIGHTS, receives message from socket
 * via recvmsg, and copies device file descriptors to in/out parameter.
 *
 */

static int psmi_recvmsg_fd(int sock, int *fds, int nfds, psm2_epid_t epid)
{
	struct msghdr msg;
	struct cmsghdr *cmsg;
	struct iovec iov;
	int64_t peer_id = *(int64_t *)&epid;
	char *ctrl_buf;
	size_t ctrl_size;
	int ret;

	ctrl_size = sizeof(*fds) * nfds;
	ctrl_buf = (char *)psmi_calloc(NULL, UNDEFINED, 1, CMSG_SPACE(ctrl_size));
	if (!ctrl_buf)
		return ENOMEM;

	iov.iov_base = &peer_id;
	iov.iov_len = sizeof(peer_id);

	memset(&msg, 0, sizeof(msg));
	msg.msg_control = ctrl_buf;
	msg.msg_controllen = CMSG_SPACE(ctrl_size);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	ret = recvmsg(sock, &msg, 0);
	if (ret ==  sizeof(peer_id)) {
		ret = 0;
	} else {
		ret = EIO;
		goto out;
	}

	psmi_assert(!(msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC)));
	cmsg = CMSG_FIRSTHDR(&msg);
	psmi_assert(cmsg && cmsg->cmsg_len == CMSG_LEN(ctrl_size) &&
	       cmsg->cmsg_level == SOL_SOCKET &&
	       cmsg->cmsg_type == SCM_RIGHTS && CMSG_DATA(cmsg));
	memcpy(fds, CMSG_DATA(cmsg), ctrl_size);
out:
	psmi_free(ctrl_buf);
	return ret;
}

/*
 * psm3_ze_init_ipc_socket - initialize ipc socket in ep
 *
 * Set up the ipc socket in the ep for listen mode. Name it
 * using our epid, and bind it.
 *
 */

psm2_error_t psm3_ze_init_ipc_socket(ptl_t *ptl_gen)
{
	struct ptl_am *ptl = (struct ptl_am *)ptl_gen;
	psm2_error_t err = PSM2_OK;
	int ret;
	struct sockaddr_un sockaddr = {0};
	socklen_t len = sizeof(sockaddr);
	char *dev_fds_sockname = NULL;

	if ((ptl->ep->ze_ipc_socket = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
		err =  PSM2_INTERNAL_ERR;
		goto fail;
	}

	sockaddr.sun_family = AF_UNIX;
	snprintf(sockaddr.sun_path, 108, "/dev/shm/psm3_shm.ze_sock2.%ld.%s", (long int) getuid(),
		psm3_epid_fmt_internal(ptl->epid, 0));
	dev_fds_sockname = psmi_strdup(NULL, sockaddr.sun_path);
	if (dev_fds_sockname == NULL) {
		ret = -PSM2_NO_MEMORY;
		goto fail;
	}

	ptl->ep->listen_sockname = dev_fds_sockname;

	if ((ret = bind(ptl->ep->ze_ipc_socket, (struct sockaddr *) &sockaddr, len)) < 0) {
		close(ptl->ep->ze_ipc_socket);
		goto fail;
	}

	if ((ret = listen(ptl->ep->ze_ipc_socket, 256)) < 0) {
		ret = -EIO;
		goto fail;
	}

fail:
	return err;
}

/*
 * psm3_receive_ze_dev_fds - receive the dev fds on the listen socket
 *
 * Set up the listen socket to be polled for POLLIN. When the event is
 * received, accept for the new socket and then read the peer epid,
 * and locate the epaddr for it. Then receive the dev fds to be stored
 * in the am_epaddr.
 *
 */

psm2_error_t psm3_receive_ze_dev_fds(ptl_t *ptl_gen)
{
	struct ptl_am *ptl = (struct ptl_am *)ptl_gen;
	psm2_error_t err = PSM2_OK;
	struct pollfd fdset;
	struct sockaddr_un sockaddr = {0};
	int poll_result;
	int nfds;
	socklen_t len = sizeof(sockaddr);
	int newsock;
	int nread;
	psm2_epid_t epid;
	psm2_epaddr_t epaddr;
	am_epaddr_t *am_epaddr;

	if (psm3_ze_get_num_dev_fds() == 0)
		psm3_ze_init_fds();
	nfds = psm3_ze_get_num_dev_fds();

	sockaddr.sun_family = AF_UNIX;
	snprintf(sockaddr.sun_path, 108, "/dev/shm/psm3_shm.ze_sock2.%ld.%s", (long int) getuid(),
		psm3_epid_fmt_internal(ptl->epid, 0));

	fdset.fd = ptl->ep->ze_ipc_socket;
	fdset.events = POLLIN;

	poll_result = poll(&fdset, 1, 0);
	if (poll_result > 0) {
		newsock = accept(ptl->ep->ze_ipc_socket, (struct sockaddr *)&sockaddr, &len);
		if (newsock < 0) {
			_HFI_ERROR("accept to ipc fds socket failed: %s\n", strerror(errno));
			err =  PSM2_INTERNAL_ERR;
			goto fail;
		} else {
			if ((nread = recv(newsock, &epid, sizeof(epid), 0)) < 0) {
				err =  PSM2_INTERNAL_ERR;
				close(newsock);
				goto fail;
			}
			if ((epaddr = psm3_epid_lookup(ptl->ep, epid)) == NULL) {
				_HFI_ERROR("Lookup of epid %s failed, unable to receive ipc dev fds from peer\n", psm3_epid_fmt_internal(epid, 0));
				err =  PSM2_INTERNAL_ERR;
				goto fail;
			}
			am_epaddr = (am_epaddr_t *)epaddr;
			am_epaddr->num_peer_fds = nfds;
			psmi_recvmsg_fd(newsock, am_epaddr->peer_fds, nfds, ptl->epid);
			close(newsock);
		}
	} else {
		err =  PSM2_INTERNAL_ERR;
	}

fail:
	return err;
}

/*
 * psm3_send_dev_fds - send the dev fds to the peer's listen socket
 *
 * Check the connected state and proceed accordingly:
 * - ZE_SOCK_NOT_CONNECTED
 *     We have not done anything yet, so connect and send our epid,
 *     followed by the dev fds. Set state to ZE_SOCK_DEV_FDS_SENT
 * - ZE_SOCK_DEV_FDS_SENT
 *     The dev fds have been sent. Issue ioctl to see if the output
 *     queue has been emptied indicating that the peer has read the data.
 *     If so, set state to ZE_SOCK_DEV_FDS_SENT_AND_RECD.
 * - ZE_SOCK_DEV_FDS_SENT_AND_RECD
 *     We are done, just return.
 *
 */

psm2_error_t psm3_send_dev_fds(ptl_t *ptl_gen, psm2_epaddr_t epaddr)
{
	am_epaddr_t *am_epaddr = (am_epaddr_t *)epaddr;
	struct ptl_am *ptl = (struct ptl_am *)ptl_gen;
	struct sockaddr_un sockaddr = {0};;
	socklen_t len = sizeof(sockaddr);
	psm2_epid_t peer_epid = epaddr->epid;
	psm2_error_t err = PSM2_OK;
	int nwritten;
	int *fds, nfds;
	int pending;

	if (psm3_ze_get_num_dev_fds() == 0)
		psm3_ze_init_fds();
	fds = psm3_ze_get_dev_fds(&nfds);

	switch (am_epaddr->sock_connected_state) {

		case ZE_SOCK_DEV_FDS_SENT_AND_RECD:
			return err;
		case ZE_SOCK_DEV_FDS_SENT:
			ioctl(am_epaddr->sock, SIOCOUTQ, &pending);
			if (pending == 0)
				am_epaddr->sock_connected_state = ZE_SOCK_DEV_FDS_SENT_AND_RECD;
			break;
		case ZE_SOCK_NOT_CONNECTED:
			if ((am_epaddr->sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
				goto fail;
			}

			sockaddr.sun_family = AF_UNIX;
			snprintf(sockaddr.sun_path, 108, "/dev/shm/psm3_shm.ze_sock2.%ld.%s",
				(long int) getuid(), psm3_epid_fmt_internal(peer_epid, 0));

			if (connect(am_epaddr->sock, (struct sockaddr *) &sockaddr, len) < 0) {
				_HFI_ERROR("connect to ipc fds socket failed: %s\n", strerror(errno));
			} else {
				nwritten = send(am_epaddr->sock, &ptl->epid, sizeof(ptl->epid), 0);
				if (nwritten < 0) {
					err = -EIO;
					goto fail;
				}
		
				if (psmi_sendmsg_fds(am_epaddr->sock, fds, nfds, peer_epid) <= 0) {
					err = -EIO;
					goto fail;
				}
				am_epaddr->sock_connected_state = ZE_SOCK_DEV_FDS_SENT;
			}
			break;
		default:
			err =  PSM2_INTERNAL_ERR;
	}

fail:
	close(am_epaddr->sock);
	return err;
}

/*
 * psm3_check_dev_fds_exchanged - check that dev fds have been exchanged
 * with peer
 *
 * Loop through the epaddrs in am_ep. For each:
 *   - If connect state is not ZE_SOCK_DEV_FDS_SENT_AND_RECD, peer has not
 *     received our data, so call psm3_send_dev_fds, then return NO PROGRESS
 *   - if number of peer fds is zero, we have not received peer's data,
 *     so call psm3_receive_ze_dev_fds, then return NO PROGRESS
 *
 */

psm2_error_t psm3_check_dev_fds_exchanged(ptl_t *ptl_gen)
{
	struct ptl_am *ptl = (struct ptl_am *)ptl_gen;
	psm2_error_t err = PSM2_OK;
	am_epaddr_t *am_epaddr;
	int i;

	for (i = 0; i <= ptl->max_ep_idx; i++) {
		if (psm3_epid_zero_internal(ptl->am_ep[i].epid)) {
			continue;
		} else {
			am_epaddr = (am_epaddr_t *)ptl->am_ep[i].epaddr;
			if (am_epaddr->sock_connected_state != ZE_SOCK_DEV_FDS_SENT_AND_RECD) {
				psm3_send_dev_fds(ptl_gen, ptl->am_ep[i].epaddr);
				err = PSM2_OK_NO_PROGRESS;
			}
			if (am_epaddr->num_peer_fds == 0) {
				psm3_receive_ze_dev_fds(ptl_gen);
				err = PSM2_OK_NO_PROGRESS;
			}
		}
	}

	return err;
}

psm2_error_t psm3_sock_detach(ptl_t *ptl_gen)
{
	struct ptl_am *ptl = (struct ptl_am *)ptl_gen;

	unlink(ptl->ep->listen_sockname);
	psmi_free(ptl->ep->listen_sockname);
	return PSM2_OK;
}

#endif // PSM_ONEAPI
