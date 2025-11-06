// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/* Cassini SRIOV and VFs handler
 *
 * AF_VSOCK sockets are used to pass messages and responses from the VF to the
 * PF, both when the VF is attached to a guest (using the appropriate hypervisor
 * vsock transport), and when the VF is attached to the host (using the
 * vsock_loopback module).
 *
 * The VF index of an incoming vsock connection is identified by probing the
 * PF-to-VF interrupt of each unaccounted-for VF and waiting for a response.
 * After this initial handshake, communication is always initiated by the VF,
 * and an acknowledgment from the PF is always expected. Messages are prefixed
 * with an integer result code (only used by the response from the PF) and
 * message length.
 */

#include <linux/pci.h>
#include <linux/types.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/net.h>
#include <linux/vm_sockets.h>
#include <net/sock.h>

#include "cass_core.h"
#include "cxi_core.h"

/* TODO: make port configurable? (0x17db is arbitrary, taken from C1 PCI vendor ID) */
#define CXI_SRIOV_VSOCK_PORT 0x17db

/* Magic numbers used in VF-PF handshake */
#define CXI_SRIOV_MAGIC1 0x17db0000
#define CXI_SRIOV_MAGIC2 0x12345678

/* PF-side timeout for vsock. Kept short so that listener-thread loop still runs. */
#define CXI_SRIOV_PF_TIMEOUT (HZ / 4)

/* VF-side timeout - longer than PF timeout, to allow time for PF to respond to requests */
#define CXI_SRIOV_VF_TIMEOUT (HZ * 10)

static int write_message_to_vsock(struct socket *sock, const void *msg, size_t msg_len, int msg_rc)
{
	struct vf_pf_msg_hdr hdr = {
		.len = msg_len,
		.rc = msg_rc
	};
	struct msghdr msghdr = {};
	struct kvec vec[] = {
		{
			.iov_base = &hdr,
			.iov_len = sizeof(hdr),
		},
		{
			.iov_base = (void *)msg,
			.iov_len = msg_len
		}
	};

	return kernel_sendmsg(sock, &msghdr, vec, 2, sizeof(hdr) + msg_len);
}

static int read_message_from_vsock(struct socket *sock, void *msg, size_t *msg_len, int *msg_rc)
{
	struct vf_pf_msg_hdr hdr;
	struct msghdr msghdr = {};
	struct kvec hdrvec = {
		.iov_base = &hdr,
		.iov_len = sizeof(hdr),
	};
	struct kvec msgvec = {
		.iov_base = msg,
		.iov_len = *msg_len,
	};
	int rc;

	rc = kernel_recvmsg(sock, &msghdr, &hdrvec, 1, sizeof(hdr), 0);
	if (rc < 0)
		return rc;
	else if (rc == 0) {
		/* Connection closed by the other end */
		return 0;
	} else if (rc < sizeof(hdr)) {
		/* Not enough data received for header */
		return -EINVAL;
	}

	if (hdr.len > MAX_VFMSG_SIZE || hdr.len > *msg_len)
		return -EINVAL;

	*msg_len = hdr.len;
	if (msg_rc)
		*msg_rc = hdr.rc;

	rc = kernel_recvmsg(sock, &msghdr, &msgvec, 1, hdr.len, 0);
	if (rc >= 0 && rc < hdr.len)
		return -EINVAL;

	return rc;
}

/* Handler thread for incoming messages from the VF driver to the PF. 1 instance
 * per active VF.
 */
static int pf_vf_msghandler(void *data)
{
	struct cass_vf *vf = (struct cass_vf *)data;
	struct cass_dev *hw = vf->hw;
	int rc, msg_rc;
	size_t request_len = MAX_VFMSG_SIZE;
	size_t reply_len;

	vf->sock->sk->sk_rcvtimeo = CXI_SRIOV_PF_TIMEOUT;

	cxidev_dbg(&hw->cdev, "vf %d: started message handler", vf->vf_idx);

	while (!kthread_should_stop()) {
		rc = read_message_from_vsock(vf->sock, vf->request,
					     &request_len, NULL);
		if (rc == -EAGAIN) {
			continue;
		} else if (rc == -EINTR) {
			/* Expected when thread is asked to terminate */
			continue;
		} else if (rc < 0) {
			cxidev_err(&hw->cdev, "vf %d: error reading request: %d",
				   vf->vf_idx, rc);
			break;
		} else if (rc == 0) {
			cxidev_err(&hw->cdev, "vf %d: connection closed by VF",
				   vf->vf_idx);
			break;
		}

		cxidev_dbg(&hw->cdev, "vf %d: got %ld byte message", vf->vf_idx,
			   request_len);

		mutex_lock(&hw->msg_relay_lock);
		if (hw->msg_relay) {
			reply_len = MAX_VFMSG_SIZE;
			msg_rc = hw->msg_relay(hw->msg_relay_data, vf->vf_idx,
						vf->request, request_len,
						vf->reply, &reply_len);
		}
		mutex_unlock(&hw->msg_relay_lock);

		if (reply_len > MAX_VFMSG_SIZE) {
			reply_len = 0;
			msg_rc = -E2BIG;
		}

		cxidev_dbg(&hw->cdev, "vf %d: responding with %ld bytes, rc=%d",
				vf->vf_idx, reply_len, msg_rc);
		rc = write_message_to_vsock(vf->sock, vf->reply, reply_len,
					    msg_rc);
		if (rc < 0) {
			cxidev_err(&hw->cdev, "vf %d: error sending response: %d",
					vf->vf_idx, rc);
			break;
		}
	}

	/* TODO: Clean up any resources left behind by VF */

	if (rc > 0)
		rc = 0;

	cxidev_dbg(&hw->cdev, "vf %d: handler exiting, rc=%d", vf->vf_idx, rc);

	kernel_sock_shutdown(vf->sock, SHUT_RDWR);
	sock_release(vf->sock);
	vf->sock = NULL;

	return rc;
}

/* Probe inactive VFs from range_min to range_max inclusive */
static unsigned int pf_probe_vfs(struct cass_dev *hw, struct socket *sock,
				 int range_min, int range_max)
{
	int i, rc;
	unsigned int magic;
	size_t msg_len = sizeof(magic);
	union c_pi_ipd_cfg_pf_vf_irq irqs = {
		.irq = 0,
	};

	for (i = range_min; i <= range_max; i++)
		if (!hw->vfs[i].sock) /* Only probe VFs that aren't already active */
			irqs.irq |= 1ULL << i;
	if (!irqs.irq)
		return 0;

	magic = CXI_SRIOV_MAGIC1;
	rc = write_message_to_vsock(sock, &magic, sizeof(magic), 0);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "could not send magic to incoming VF: %d", rc);
		return rc;
	}

	cass_write(hw, C_PI_IPD_CFG_PF_VF_IRQ, &irqs,
		   sizeof(union c_pi_ipd_cfg_pf_vf_irq));

	msg_len = sizeof(magic);
	rc = read_message_from_vsock(sock, &magic, &msg_len, NULL);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "could not read magic from incoming VF: %d", rc);
		return rc;
	}

	return magic;
}

/* Identify which VF an incoming connection belongs to, by probing the PF-to-VF
 * interrupt of VFs that do not have an active connection.
 */
static int pf_vf_handshake(struct cass_dev *hw, struct socket *sock)
{
	int range_min = 0;
	int range_max = hw->num_vfs - 1;
	int range_mid, rc, seq, vf_idx;
	unsigned int magic;

	seq = 0; /* Handshake sequence number */
	do {
		magic = CXI_SRIOV_MAGIC1 + seq;
		range_mid = range_min + (range_max - range_min) / 2;
		if (pf_probe_vfs(hw, sock, range_min, range_mid) == magic) {
			seq += 1;
			range_max = range_mid;
		} else if (pf_probe_vfs(hw, sock, range_mid + 1, range_max) == magic) {
			seq += 1;
			range_min = range_mid + 1;
		} else {
			break;
		}
	} while (range_max != range_min);

	if (range_max != range_min) {
		cxidev_err(&hw->cdev, "vf search failed");
		return -ENOENT;
	}
	vf_idx = range_min;

	magic = CXI_SRIOV_MAGIC2;
	rc = write_message_to_vsock(sock, &magic, sizeof(magic), 0);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "could not send magic to incoming VF: %d", rc);
		return rc;
	}

	return vf_idx;
}

/* Listener thread for incoming connections from VFs to the PF. 1 instance only. */
static int pf_vf_listener(void *data)
{
	int rc, vf_idx;
	struct cass_dev *hw = (struct cass_dev *)data;
	struct cass_vf *vf;
	struct socket *incoming;
	struct sockaddr_vm peeraddr;
	const struct sockaddr_vm myaddr = {
		.svm_family = AF_VSOCK,
		.svm_port = CXI_SRIOV_VSOCK_PORT,
		.svm_cid = VMADDR_CID_ANY
	};

	cxidev_dbg(&hw->cdev, "started vf listener");

	rc = sock_create_kern(&init_net, PF_VSOCK, SOCK_STREAM, 0, &hw->vf_sock);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "vf listener socket create error: %d", rc);
		return rc;
	}

	rc = kernel_bind(hw->vf_sock, (struct sockaddr *)&myaddr, sizeof(myaddr));
	if (rc < 0) {
		cxidev_err(&hw->cdev, "vf listener socket bind error: %d", rc);
		goto release_sock;
	}

	hw->vf_sock->sk->sk_rcvtimeo = CXI_SRIOV_PF_TIMEOUT;
	rc = kernel_listen(hw->vf_sock, hw->num_vfs);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "vf listener socket listen error: %d", rc);
		goto release_sock;
	}

	while (!kthread_should_stop()) {
		rc = kernel_accept(hw->vf_sock, &incoming, 0);
		if (rc == -EAGAIN) {
			continue;
		} else if (rc == -EINTR) {
			/* Expected when listener thread is asked to terminate */
			continue;
		} else if (rc < 0) {
			cxidev_err(&hw->cdev, "vf listener socket accept error: %d", rc);
			break;
		}

		rc = kernel_getpeername(incoming, (struct sockaddr *) &peeraddr);
		if (rc < 0) {
			cxidev_err(&hw->cdev, "could not get peer addr for vf: %d", vf_idx);
			kernel_sock_shutdown(incoming, SHUT_RDWR);
			sock_release(incoming);
			continue;
		}

		vf_idx = pf_vf_handshake(hw, incoming);
		if (vf_idx < 0) {
			cxidev_err(&hw->cdev, "vf handshake from cid %d failed: %d",
				   peeraddr.svm_cid, vf_idx);
			kernel_sock_shutdown(incoming, SHUT_RDWR);
			sock_release(incoming);
			continue;
		}

		cxidev_dbg(&hw->cdev, "pf got connection from cid %d, vf %d",
			   peeraddr.svm_cid, vf_idx);

		vf = &hw->vfs[vf_idx];
		if (vf->sock) {
			cxidev_err(&hw->cdev, "vf %d already in use", vf_idx);
			kernel_sock_shutdown(incoming, SHUT_RDWR);
			sock_release(incoming);
			continue;
		}

		vf->vf_idx = vf_idx;
		vf->hw = hw;
		vf->sock = incoming;
		vf->task = kthread_run(pf_vf_msghandler, vf, "cxi_vf_%d", vf_idx);
		if (IS_ERR(vf->task)) {
			cxidev_err(&hw->cdev, "failed to start handler for vf %d",
				   vf_idx);
			kernel_sock_shutdown(incoming, SHUT_RDWR);
			sock_release(incoming);
			vf->sock = NULL;
			vf->task = NULL;
		}
	}

	cxidev_dbg(&hw->cdev, "vf listener exiting");

	for (vf_idx = 0; vf_idx < hw->num_vfs; vf_idx++)
		if (hw->vfs[vf_idx].sock)
			kthread_stop(hw->vfs[vf_idx].task);

	kernel_sock_shutdown(hw->vf_sock, SHUT_RDWR);
release_sock:
	sock_release(hw->vf_sock);
	hw->vf_sock = NULL;
	return rc;
}

static void disable_sriov(struct pci_dev *pdev)
{
	struct cass_dev *hw = pci_get_drvdata(pdev);

	pci_disable_sriov(pdev);

	if (hw->vf_listener) {
		kthread_stop(hw->vf_listener);
		hw->vf_listener = NULL;
	}

	hw->num_vfs = 0;
}

static int enable_sriov(struct pci_dev *pdev, int num_vfs)
{
	int rc;
	int sriov;
	u16 offset;
	u16 stride;
	union c_pi_cfg_pri_sriov pri_sriov = {};
	struct cass_dev *hw = pci_get_drvdata(pdev);

	rc = request_module("vsock_loopback");
	if (rc) {
		cxidev_err(&hw->cdev, "could not load vsock_loopback module");
		goto err_novf;
	}

	hw->num_vfs = num_vfs;

	if (!hw->vf_listener)
		hw->vf_listener = kthread_run(pf_vf_listener, hw, "cxi_vf_listener");
	if (IS_ERR(hw->vf_listener)) {
		cxidev_err(&hw->cdev, "could not start vf listener thread");
		rc = PTR_ERR(hw->vf_listener);
		hw->vf_listener = NULL;
		goto err_novf;
	}

	/* The VF Offset and Stride need to match the SR-IOV configuration. */
	sriov = pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_SRIOV);
	if (!sriov) {
		cxidev_err(&hw->cdev, "No extended capabilities found\n");
		rc = -ENODEV;
		goto err_kill_listener;
	}

	pci_read_config_word(pdev, sriov + PCI_SRIOV_VF_OFFSET, &offset);
	pci_read_config_word(pdev, sriov + PCI_SRIOV_VF_STRIDE, &stride);

	pri_sriov.vf_offset = offset;
	pri_sriov.vf_stride = stride;

	cass_write(hw, C_PI_CFG_PRI_SRIOV, &pri_sriov,
		   sizeof(union c_pi_cfg_pri_sriov));

	rc = pci_enable_sriov(pdev, num_vfs);
	if (rc) {
		cxidev_err(&hw->cdev, "SRIOV enable failed %d\n", rc);
		goto err_kill_listener;
	}

	return num_vfs;

err_kill_listener:
	kthread_stop(hw->vf_listener);
	hw->vf_listener = NULL;
err_novf:
	hw->num_vfs = 0;
	return rc;
}

int cass_sriov_configure(struct pci_dev *pdev, int num_vfs)
{
	if (num_vfs < 0)
		return -EINVAL;

	if (num_vfs == 0) {
		disable_sriov(pdev);
		return 0;
	}

	return enable_sriov(pdev, num_vfs);
}

static irqreturn_t pf_to_vf_int_cb(int irq, void *context)
{
	struct cass_dev *hw = context;

	complete(&hw->pf_to_vf_comp);
	return IRQ_HANDLED;
}

int cass_vf_init(struct cass_dev *hw)
{
	int rc, seq;
	unsigned int magic;
	size_t msg_len = sizeof(magic);
	bool init_done = false;
	struct sockaddr_vm addr = {
		.svm_family = AF_VSOCK,
		.svm_port = CXI_SRIOV_VSOCK_PORT,
		.svm_cid = VMADDR_CID_HOST,
	};

	if (!hw->with_vf_support)
		return 0;

	init_completion(&hw->pf_to_vf_comp);

	scnprintf(hw->pf_vf_int_name, sizeof(hw->pf_vf_int_name),
		  "%s_from_pf", hw->cdev.name);
	hw->pf_vf_vec = pci_irq_vector(hw->cdev.pdev, 0);
	rc = request_irq(hw->pf_vf_vec, pf_to_vf_int_cb, 0, hw->pf_vf_int_name, hw);
	if (rc)
		return rc;

	rc = sock_create_kern(&init_net, PF_VSOCK, SOCK_STREAM, 0, &hw->vf_sock);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "vf socket create failed: %d", rc);
		goto free_irq;
	}

	hw->vf_sock->sk->sk_rcvtimeo = CXI_SRIOV_VF_TIMEOUT;

	rc = kernel_connect(hw->vf_sock, (struct sockaddr *) &addr, sizeof(addr), 0);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "vf socket connect failed: %d", rc);
		goto sock_release;
	}

	/* Handshake with PF driver starts here. */
	seq = 0;
	while (!init_done) {
		msg_len = sizeof(magic);
		rc = read_message_from_vsock(hw->vf_sock, &magic, &msg_len, NULL);
		if (rc < 0) {
			cxidev_err(&hw->cdev, "error receiving magic from PF: %d", rc);
			init_done = true;
			break;
		} else if (rc == 0) {
			cxidev_err(&hw->cdev, "PF closed connection during handshake");
			rc = -ENOTCONN;
			init_done = true;
			break;
		}
		switch (magic) {
		case CXI_SRIOV_MAGIC1:
			rc = wait_for_completion_timeout(&hw->pf_to_vf_comp,
							 CXI_SRIOV_VF_TIMEOUT);
			if (rc == 0) {
				cxidev_err(&hw->cdev, "timed out waiting for irq from pf");
				rc = -ETIMEDOUT;
				init_done = true;
				break;
			}
			magic = CXI_SRIOV_MAGIC1 + seq;
			msg_len = sizeof(magic);
			rc = write_message_to_vsock(hw->vf_sock, &magic, msg_len, 0);
			if (rc < 0) {
				cxidev_err(&hw->cdev, "error sending magic to PF: %d", rc);
				init_done = true;
				break;
			}
			seq += 1;
			break;
		case CXI_SRIOV_MAGIC2:
			init_done = true;
			break;
		default:
			cxidev_err(&hw->cdev, "got unexpected magic from PF: %x", magic);
			rc = -EINVAL;
			init_done = true;
			break;
		}
	}

	if (rc >= 0)
		return 0;

	kernel_sock_shutdown(hw->vf_sock, SHUT_RDWR);
sock_release:
	sock_release(hw->vf_sock);
	hw->vf_sock = NULL;
free_irq:
	free_irq(hw->pf_vf_vec, hw);
	return rc;
}

void cass_vf_fini(struct cass_dev *hw)
{
	if (!hw->with_vf_support)
		return;

	free_irq(hw->pf_vf_vec, hw);

	if (hw->vf_sock) {
		kernel_sock_shutdown(hw->vf_sock, SHUT_RDWR);
		sock_release(hw->vf_sock);
		hw->vf_sock = NULL;
	}
}

/**
 * cxi_send_msg_to_pf() - Send a message to PF and wait for the reply.
 *
 * The VF driver calls this function to send messages to the PF.
 *
 * @cdev: the device
 * @req: message data
 * @req_len: length of message
 * @rsp: buffer for response from PF
 * @rsp_len: length of response buffer (updated to reflect response length)
 */
int cxi_send_msg_to_pf(struct cxi_dev *cdev, const void *req, size_t req_len,
		       void *rsp, size_t *rsp_len)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int rc;
	int msg_rc;

	if (cdev->is_physfn)
		return -ENOTSUPP;

	if (req_len % 2 != 0 || req_len > MAX_VFMSG_SIZE)
		return -EINVAL;


	cxidev_dbg(&hw->cdev, "Sending %ld bytes to PF", req_len);

	rc = write_message_to_vsock(hw->vf_sock, req, req_len, 0);
	if (rc < 0) {
		cxidev_err(&hw->cdev, "Failed to send message to PF: %d\n", rc);
		return rc;
	}
	rc = read_message_from_vsock(hw->vf_sock, rsp, rsp_len, &msg_rc);
	if (rc == -EAGAIN) {
		cxidev_err(&hw->cdev, "PF didn't reply in time\n");
		return rc;
	} else if (rc < 0) {
		cxidev_err(&hw->cdev, "Failed to read response from PF: %d", rc);
		return rc;
	}

	cxidev_dbg(&hw->cdev, "Got %ld byte reply from PF", *rsp_len);

	return msg_rc;
}
EXPORT_SYMBOL(cxi_send_msg_to_pf);

/**
 * cxi_register_msg_relay() - Register a VF to PF message handler
 *
 * The user driver, when inserting a new PF device, is registering a
 * callback to receive messages from VFs.
 *
 * @cdev: the device
 * @msg_relay: the message handler
 * @msg_relay_data: opaque pointer to give when caller the handler
 */
int cxi_register_msg_relay(struct cxi_dev *cdev, cxi_msg_relay_t msg_relay,
			   void *msg_relay_data)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int rc;

	if (!cdev->is_physfn)
		return -EINVAL;

	mutex_lock(&hw->msg_relay_lock);

	if (hw->msg_relay) {
		rc = -EINVAL;
	} else {
		hw->msg_relay = msg_relay;
		hw->msg_relay_data = msg_relay_data;
		rc = 0;
	}

	mutex_unlock(&hw->msg_relay_lock);

	return rc;
}
EXPORT_SYMBOL(cxi_register_msg_relay);

int cxi_unregister_msg_relay(struct cxi_dev *cdev)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int rc;

	mutex_lock(&hw->msg_relay_lock);

	if (!hw->msg_relay) {
		rc = -EINVAL;
	} else {
		hw->msg_relay = NULL;
		hw->msg_relay_data = NULL;
		rc = 0;
	}

	mutex_unlock(&hw->msg_relay_lock);

	return rc;
}
EXPORT_SYMBOL(cxi_unregister_msg_relay);
