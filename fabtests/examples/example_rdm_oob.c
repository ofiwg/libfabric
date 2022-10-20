/*
 * Copyright (c) Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

//Run Server: example_rdm_oob
//Run client: example_rdm_oob <server>

#define BUF_SIZE 64
#define MR_KEY 0xC0DE

char *src_addr = NULL, *dst_addr = NULL;
char *oob_port = "9228";
int listen_sock, oob_sock;
struct fi_info *hints, *info;
struct fid_fabric *fabric = NULL;
struct fid_domain *domain = NULL;
struct fid_ep *ep = NULL;
struct fid_av *av = NULL;
struct fid_cq *cq = NULL;
struct fid_mr *mr = NULL;
void *desc;
char buf[BUF_SIZE];
static fi_addr_t fi_addr = FI_ADDR_UNSPEC;

static int sock_listen(char *node, char *service)
{
	struct addrinfo *ai, hints;
	int val, ret;

	memset(&hints, 0, sizeof hints);
	hints.ai_flags = AI_PASSIVE;

	ret = getaddrinfo(node, service, &hints, &ai);
	if (ret) {
		printf("getaddrinfo() %s\n", gai_strerror(ret));
		return ret;
	}

	listen_sock = socket(ai->ai_family, SOCK_STREAM, 0);
	if (listen_sock < 0) {
		printf("socket error");
		ret = listen_sock;
		goto out;
	}

	val = 1;
	ret = setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR,
			 (void *) &val, sizeof val);
	if (ret) {
		printf("setsockopt SO_REUSEADDR");
		goto out;
	}

	ret = bind(listen_sock, ai->ai_addr, ai->ai_addrlen);
	if (ret) {
		printf("bind");
		goto out;
	}

	ret = listen(listen_sock, 0);
	if (ret)
		printf("listen error");

out:
	if (ret && listen_sock >= 0)
		close(listen_sock);
	freeaddrinfo(ai);
	return ret;
}

static int sock_setup(int sock)
{
	int ret, op;
	long flags;

	op = 1;
	ret = setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
			  (void *) &op, sizeof(op));
	if (ret)
		return ret;

	flags = fcntl(sock, F_GETFL);
	if (flags < 0)
		return -errno;

	if (fcntl(sock, F_SETFL, flags))
		return -errno;

	return 0;
}

static int init_oob(void)
{
	struct addrinfo *ai = NULL;
	int ret;

	if (!dst_addr) {
		ret = sock_listen(src_addr, oob_port);
		if (ret)
			return ret;

		oob_sock = accept(listen_sock, NULL, 0);
		if (oob_sock < 0) {
			printf("accept error");
			ret = oob_sock;
			return ret;
		}

		close(listen_sock);
	} else {
		ret = getaddrinfo(dst_addr, oob_port, NULL, &ai);
		if (ret) {
			printf("getaddrinfo error");
			return ret;
		}

		oob_sock = socket(ai->ai_family, SOCK_STREAM, 0);
		if (oob_sock < 0) {
			printf("socket error");
			ret = oob_sock;
			goto free;
		}

		ret = connect(oob_sock, ai->ai_addr, ai->ai_addrlen);
		if (ret) {
			printf("connect error");
			close(oob_sock);
			goto free;
		}
		sleep(1);
	}

	ret = sock_setup(oob_sock);

free:
	if (ai)
		freeaddrinfo(ai);
	return ret;
}

static int sock_send(int fd, void *msg, size_t len)
{
	size_t sent;
	ssize_t ret, err = 0;

	for (sent = 0; sent < len; ) {
		ret = send(fd, ((char *) msg) + sent, len - sent, 0);
		if (ret > 0) {
			sent += ret;
		} else {
			err = -errno;
			break;
		}
	}

	return err ? err: 0;
}

static int sock_recv(int fd, void *msg, size_t len)
{
	size_t rcvd;
	ssize_t ret, err = 0;

	for (rcvd = 0; rcvd < len; ) {
		ret = recv(fd, ((char *) msg) + rcvd, len - rcvd, 0);
		if (ret > 0) {
			rcvd += ret;
		} else if (ret == 0) {
			err = -FI_ENOTCONN;
			break;
		} else {
			err = -errno;
			break;
		}
	}

	return err ? err: 0;
}

static int sync_progress(void)
{
	int ret, value = 0, result = -FI_EOTHER;

	if (dst_addr) {
		ret = send(oob_sock, &value, sizeof(value), 0);
		if (ret != sizeof(value))
			return -FI_EOTHER;

		do {
			ret = recv(oob_sock, &result, sizeof(result), MSG_DONTWAIT);
			if (ret == sizeof(result))
				break;

			ret = fi_cq_read(cq, NULL, 0);
			if (ret && ret != -FI_EAGAIN)
				return ret;
		} while (1);
	} else {
		do {
			ret = recv(oob_sock, &result, sizeof(result), MSG_DONTWAIT);
			if (ret == sizeof(result))
				break;

			ret = fi_cq_read(cq, NULL, 0);
			if (ret && ret != -FI_EAGAIN)
				return ret;
		} while (1);

		ret = send(oob_sock, &value, sizeof(value), 0);
		if (ret != sizeof(value))
			return -FI_EOTHER;
	}
	return 0;
}

/*
 * The server and client need to exchange their local addresses with each other
 * so that the entity performing read or write operation knows where to read
 * from or write to.
 */
static int exchange_addresses(void)
{
	char addr_buf[BUF_SIZE];
	int ret;
	size_t addrlen = BUF_SIZE;

	ret = fi_getname(&ep->fid, addr_buf, &addrlen);
	if (ret) {
		printf("fi_getname error %d\n", ret);
		return ret;
	}

	ret = sock_send(oob_sock, addr_buf, BUF_SIZE);
	if (ret) {
		printf("sock_send error %d\n", ret);
		return ret;
	}

	memset(addr_buf, 0, BUF_SIZE);
	ret = sock_recv(oob_sock, addr_buf, BUF_SIZE);
	if (ret) {
		printf("sock_recv error %d\n", ret);
		return ret;
	}

	ret = fi_av_insert(av, addr_buf, 1, &fi_addr, 0, NULL);
	if (ret != 1) {
		printf("av insert error\n");
		return -FI_ENOSYS;
	}

	return sync_progress();
}

/* Set anything in hints that the application needs */
static int set_hints(void)
{

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	/*
	 * Request FI_EP_RDM (reliable datagram) endpoint which will allow us
	 * to reliably send messages to peers without having to
	 * listen/connect/accept.
	 */
	hints->ep_attr->type = FI_EP_RDM;

	/*
	 * Request basic messaging capabilities from the provider (no tag
	 * matching, no RMA, no atomic operations)
	 */
	hints->caps = FI_MSG;

	/*
	 * Default to FI_DELIVERY_COMPLETE which will make sure completions do
	 * not get generated until our message arrives at the destination.
	 * Otherwise, the client might get a completion and exit before the
	 * server receives the message. This is to make the test simpler.
	 */
	hints->tx_attr->op_flags = FI_DELIVERY_COMPLETE;

	/* Set memory registration modes. */
	hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_LOCAL |
		FI_MR_PROV_KEY | FI_MR_ALLOCATED | FI_MR_VIRT_ADDR;

	/* Specifically request the tcp provider for the simple test */
	hints->fabric_attr->prov_name = "tcp";

	//Done setting hints

	return 0;
}

/*
 * Initializes all basic libfabric resources to allow for a server/client to
 * exchange a message
 */
static int initialize(void)
{
	struct fi_cq_attr cq_attr = {0};
	struct fi_av_attr av_attr = {0};
	int ret;

	/*
	 * The first libfabric call to happen for initialization is fi_getinfo
	 * which queries libfabric and returns any appropriate providers that
	 * fulfill the hints requirements. Any applicable providers will be
	 * returned as a list of fi_info structs (&info). Any info can be
	 * selected. In this test we select the first fi_info struct. Assuming
	 * all hints were set appropriately, the first fi_info should be most
	 * appropriate. The flag FI_SOURCE is set for the server to indicate
	 * that the address/port refer to source information. This is not set
	 * for the client because the fields refer to the server, not the
	 * caller (client).
	 */
	ret = fi_getinfo(FI_VERSION(1,9), NULL, NULL, 0,
			 hints, &info);
	if (ret) {
		printf("fi_getinfo error (%d)\n", ret);
		return ret;
	}

	/*
	 * Initialize our fabric. The fabric network represents a collection of
	 * hardware and software resources that access a single physical or
	 * virtual network. All network ports on a system that can communicate
	 * with each other through their attached networks belong to the same
	 * fabric.
	 */

	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	if (ret) {
		printf("fi_fabric error (%d)\n", ret);
		return ret;
	}

	/*
	 * Initialize our domain (associated with our fabric). A domain defines
	 * the boundary for associating different resources together.
	 */

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		printf("fi_domain error (%d)\n", ret);
		return ret;
	}

	/*
	 * Initialize our endpoint. Endpoints are transport level communication
	 * portals which are used to initiate and drive communication. There
	 * are three main types of endpoints:
	 * FI_EP_MSG - connected, reliable
	 * FI_EP_RDM - unconnected, reliable
	 * FI_EP_DGRAM - unconnected, unreliable
	 * The type of endpoint will be requested in hints/fi_getinfo.
	 * Different providers support different types of endpoints.
	 */

	ret = fi_endpoint(domain, info, &ep, NULL);
	if (ret) {
		printf("fi_endpoint error (%d)\n", ret);
		return ret;
	}

	/*
	 * Initialize our completion queue. Completion queues are used to
	 * report events associated with data transfers. In this example, we
	 * use one CQ that tracks sends and receives, but often times there
	 * will be separate CQs for sends and receives.
	 */

	cq_attr.size = 128;
	cq_attr.format = FI_CQ_FORMAT_MSG;
	ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
	if (ret) {
		printf("fi_cq_open error (%d)\n", ret);
		return ret;
	}

	/*
	 * Bind our CQ to our endpoint to track any sends and receives that
	 * come in or out on that endpoint. A CQ can be bound to multiple
	 * endpoints but one EP can only have one send CQ and one receive CQ
	 * (which can be the same CQ).
	 */

	ret = fi_ep_bind(ep, &cq->fid, FI_SEND | FI_RECV);
	if (ret) {
		printf("fi_ep_bind cq error (%d)\n", ret);
		return ret;
	}

	/*
	 * Initialize our address vector. Address vectors are used to map
	 * higher level addresses, which may be more natural for an application
	 * to use, into fabric specific addresses. An AV_TABLE av will map
	 * these addresses to indexed addresses, starting with fi_addr 0. These
	 * addresses are used in data transfer calls to specify which peer to
	 * send to/recv from. Address vectors are only used for FI_EP_RDM and
	 * FI_EP_DGRAM endpoints, allowing the application to avoid connection
	 * management. For FI_EP_MSG endpoints, the AV is replaced by the
	 * traditional listen/connect/accept steps.
	 */

	av_attr.type = FI_AV_TABLE;
	av_attr.count = 1;
	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		printf("fi_av_open error (%d)\n", ret);
		return ret;
	}

	/*
	 * Bind the AV to the EP. The EP can only send data to a peer in its
	 * AV.
	 */

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		printf("fi_ep_bind av error (%d)\n", ret);
		return ret;
	}

	/*
	 * Once we have all our resources initialized and ready to go, we can
	 * enable our EP in order to send/receive data.
	 */

	ret = fi_enable(ep);
	if (ret) {
		printf("fi_enable error (%d)\n", ret);
		return ret;
	}

	/* Register the memory region mr */
	ret = fi_mr_reg(domain, buf, BUF_SIZE, FI_SEND | FI_RECV | FI_WRITE | FI_READ |
			FI_REMOTE_WRITE | FI_REMOTE_READ, 0, MR_KEY, 0, &mr, NULL);
	if (ret) {
		printf("fi_mr_reg error (%d)\n", ret);
		return ret;
	}

	 /* Obtain local descriptor */
	desc = fi_mr_desc(mr);

	/* Bind the memory region mr with the endpoint and enable mr */
	if (info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		ret = fi_mr_bind(mr, &ep->fid, 0);
		if (ret) {
			printf("fi_mr_bind error (%d)\n", ret);
			return ret;
		}

		ret = fi_mr_enable(mr);
		if (ret) {
			printf("fi_mr_enable error (%d)\n", ret);
			return ret;
		}
	}

	ret = exchange_addresses();
	if (ret)
		return ret;

	return 0;
}

/* All libfabric resources are cleaned up using the same fi_close(fid) call.
 * Resources must be closed in a specific order to allow references between
 * objects to be removed correctly. For example the endpoint must be closed
 * before the CQ or AV.
 */
static void cleanup(void)
{
	int ret;

	if (mr) {
		ret = fi_close(&mr->fid);
		if (ret)
			printf("warning: error closing EP (%d)\n", ret);
	}

	if (ep) {
		ret = fi_close(&ep->fid);
		if (ret)
			printf("warning: error closing EP (%d)\n", ret);
	}

	if (av) {
		ret = fi_close(&av->fid);
		if (ret)
			printf("warning: error closing AV (%d)\n", ret);
	}

	if (cq) {
		ret = fi_close(&cq->fid);
		if (ret)
			printf("warning: error closing CQ (%d)\n", ret);
	}

	if (domain) {
		ret = fi_close(&domain->fid);
		if (ret)
			printf("warning: error closing domain (%d)\n", ret);
	}

	if (fabric) {
		ret = fi_close(&fabric->fid);
		if (ret)
			printf("warning: error closing fabric (%d)\n", ret);
	}

	/* Free the space occupied by info struct*/
	if (info)
		fi_freeinfo(info);
}

/*
 * Post a receive buffer. This call does not ensure a message has been
 * received, just that a buffer has been passed to libfabric for the next
 * message the provider receives. Receives may be directed or undirected using
 * the address parameter. Here, we pass in the fi_addr but note that the server
 * has not inserted the client's address into its AV, so the address is still
 * FI_ADDR_UNSPEC, indicating that this buffer may receive incoming data from
 * any address. An application may set this to a real fi_addr if the buffer
 * should only receive data from a certain peer.
 * When posting a buffer, if the provider is not ready to process messages
 * (because of connection initialization for example), it may return
 * -FI_EAGAIN. This does not indicate an error, but rather that the application
 * should try again later. This is why we almost always wrap sends and receives
 * in a do/while. Some providers may need the application to drive progress in
 * order to get out of the -FI_EAGAIN loop. To drive progress, the application
 * needs to call fi_cq_read (not necessarily reading any completion entries).
 */
static int post_recv(void)
{
	int ret;

	do {
		ret = fi_recv(ep, buf, BUF_SIZE, NULL, fi_addr, NULL);
		if (ret && ret != -FI_EAGAIN) {
			printf("error posting recv buffer (%d\n", ret);
			return ret;
		}
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(cq, NULL, 0);
	} while (ret);

	return 0;
}

/*
 * Post a send buffer. This call does not ensure a message has been sent, just
 * that a buffer has been submitted to libfabric to be sent. Unlike a receive
 * buffer, a send needs a valid fi_addr as input to tell the provider where to
 * send the message. Similar to the receive buffer posting porcess, when
 * posting a send buffer, if the provider is not ready to process messages, it
 * may return -FI_EAGAIN. This does not indicate an error, but rather that the
 * application should try again later. Just like the receive, we drive progress
 * with fi_cq_read if this is the case.
 */
static int post_send(void)
{
	char *msg = "Hello, server! I am the client you've been waiting for!";
	int ret;

	//(void) snprintf(buf, BUF_SIZE, "%s", msg);

	do {
		ret = fi_send(ep, msg, strlen(msg), desc, fi_addr, NULL);
		if (ret && ret != -FI_EAGAIN) {
			printf("error posting send buffer (%d)\n", ret);
			return ret;
		}
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(cq, NULL, 0);
	} while (ret);

	return 0;
}

/*
 * Wait for the message to be sent/received using the CQ. fi_cq_read not only
 * drives progress but also returns any completed events to notify the
 * application that it can reuse the send/recv buffer. The returned completion
 * entry will have fields set to let the application know what operation
 * completed. Not all fields will be valid. The fields set will be indicated by
 * the cq format (when creating the CQ). In this example, we use
 * FI_CQ_FORMAT_MSG in order to use the flags field.
 */
static int spin_for_comp(void)
{
	struct fi_cq_err_entry comp;
	int ret;

	do {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret < 0 && ret != -FI_EAGAIN) {
			printf("error reading cq (%d)\n", ret);
			return ret;
		}
	} while (ret != 1);

	if (comp.flags & FI_RECV)
		printf("I received a message!\n");
	else if (comp.flags & FI_SEND)
		printf("My sent message got sent!\n");

	return 0;
}

static int run(void)
{
	int ret;

	if (dst_addr) {
		printf("Client: send to server %s\n", dst_addr);

		ret = post_send();
		if (ret)
			return ret;

		ret = spin_for_comp();
		if (ret)
			return ret;

	} else {
		printf("Server: post buffer and wait for message from client\n");

		ret = post_recv();
		if (ret)
			return ret;

		ret = spin_for_comp();
		if (ret)
			return ret;

		printf("This is the message I received: %s\n", buf);
	}

	return sync_progress();
}

int main(int argc, char **argv)
{
	int ret;

	/*
	 * Server run with no args, client has server's address as an
	 * argument.
	 */
	dst_addr = argv[optind];

	/* Init out-of-band addressing */
	ret = init_oob();
	if (ret)
		return ret;

	/*
	 * Hints are used to request support for specific features from a
	 * provider.
	 */
	ret = set_hints();
	if (ret) {
		printf ("Error settings hints.\n");
		goto out;
	}

	ret = initialize();
	if (ret)
		goto out;

	ret = run();
out:
	cleanup();
	return ret;
}
