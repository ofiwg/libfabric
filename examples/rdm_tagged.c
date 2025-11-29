#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <netinet/in.h>
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
#include <rdma/fi_tagged.h>

#define BUF_SIZE 64

#define TAG_1 1
#define TAG_2 2

char *src_addr = NULL, *dst_addr = NULL;
char *oob_port = "9228";
int listen_sock, oob_sock;
struct fi_info *hints, *info;
struct fid_fabric *fabric = NULL;
struct fid_domain *domain = NULL;
struct fid_ep *ep = NULL;
struct fid_av *av = NULL;
struct fid_cq *cq = NULL;
char buf_1[BUF_SIZE];
char buf_2[BUF_SIZE];
fi_addr_t fi_addr = FI_ADDR_UNSPEC;
uint64_t fi_tag = 0;

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

	return err;
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
	return err;
}

static int sync_progress(void)
{
	int ret, value = 0, result = -FI_EOTHER;

	if (dst_addr) {
		ret = send(oob_sock, &value, sizeof(value), 0);
		if (ret != sizeof(value))
			return -FI_EOTHER;

		do {
			ret = recv(oob_sock, &result, sizeof(result),
				   MSG_DONTWAIT);
			if (ret == sizeof(result))
				break;

			ret = fi_cq_read(cq, NULL, 0);
			if (ret && ret != -FI_EAGAIN)
				return ret;
		} while (1);
	} else {
		do {
			ret = recv(oob_sock, &result, sizeof(result),
				   MSG_DONTWAIT);
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


static int initialize(void)
{
	struct fi_cq_attr cq_attr = {0};
	struct fi_av_attr av_attr = {0};
	int ret;

	ret = fi_getinfo(FI_VERSION(1,9), NULL, NULL, 0,
			 hints, &info);
	if (ret) {
		printf("fi_getinfo error (%d)\n", ret);
		return ret;
	}

	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	if (ret) {
		printf("fi_fabric error (%d)\n", ret);
		return ret;
	}

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		printf("fi_domain error (%d)\n", ret);
		return ret;
	}

	ret = fi_endpoint(domain, info, &ep, NULL);
	if (ret) {
		printf("fi_endpoint error (%d)\n", ret);
		return ret;
	}

	cq_attr.size = 128;
	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cq_attr.wait_obj = 0;
	ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
	if (ret) {
		printf("fi_cq_open error (%d)\n", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &cq->fid, FI_SEND | FI_RECV);
	if (ret) {
		printf("fi_ep_bind cq error (%d)\n", ret);
		return ret;
	}

	av_attr.type = FI_AV_TABLE;
	av_attr.count = 1;
	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		printf("fi_av_open error (%d)\n", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		printf("fi_ep_bind av error (%d)\n", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		printf("fi_enable error (%d)\n", ret);
		return ret;
	}

	ret = exchange_addresses();
	if (ret)
		return ret;

	return 0;
}

static void cleanup(void)
{
	int ret;

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

	if (info)
		fi_freeinfo(info);
}

static int post_recv(void)
{
	int ret;

	fi_tag = TAG_2;
	do {
		ret = fi_trecv(ep, buf_2, BUF_SIZE, NULL, fi_addr,
			       fi_tag, 0, NULL);
		if (ret && ret != -FI_EAGAIN) {
			printf("error posting recv buffer (%d\n", ret);
			return ret;
		}
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(cq, NULL, 0);
	} while (ret);

	fi_tag = TAG_1;
	do {
		ret = fi_trecv(ep, buf_1, BUF_SIZE, NULL, fi_addr,
			       fi_tag, 0, NULL);
		if (ret && ret != -FI_EAGAIN) {
		    printf("error posting recv buffer (%d\n", ret);
		    return ret;
		}
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(cq, NULL, 0);
	} while (ret);

	return 0;
}

static int post_send(void)
{
	char *msg_1 = "Hello, server! I am the client sending TAG_1 message!\0";
	char *msg_2 = "Hello, server! I am the client sending TAG_2 message!\0";
	int ret;

	(void) snprintf(buf_1, BUF_SIZE, "%s", msg_1);
	(void) snprintf(buf_2, BUF_SIZE, "%s", msg_2);

	fi_tag = TAG_1;
	do {
		ret = fi_tsend(ep, buf_1, strlen(msg_1), NULL, fi_addr,
			       fi_tag, NULL);
		if (ret && ret != -FI_EAGAIN) {
			printf("error posting send buffer (%d)\n", ret);
			return ret;
		}
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(cq, NULL, 0);
	} while (ret);

	printf ("Client: I posted TAG_1 message\n");

	fi_tag = TAG_2;
	do {
		ret = fi_tsend(ep, buf_2, strlen(msg_2), NULL, fi_addr,
			       fi_tag, NULL);
		if (ret && ret != -FI_EAGAIN) {
			printf("error posting send buffer (%d)\n", ret);
			return ret;
		}
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(cq, NULL, 0);
	} while (ret);

	printf ("Client: I posted TAG_2 message\n");

	return 0;
}

static int wait_cq(void)
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
		printf("I received a message with this tag: %ld!\n", comp.tag);
	else if (comp.flags & FI_SEND)
		printf("My message got sent!\n");

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

		ret = wait_cq();
		if (ret)
			return ret;
	} else {
		printf("Server: post buffer and wait for message from client\n");

		ret = post_recv();
		if (ret)
			return ret;

		ret = wait_cq();
		if (ret)
			return ret;

		ret = wait_cq();
		if (ret)
			return ret;

		printf("This is the message I received: %s\n", buf_2);
		printf("This is the message I received: %s\n", buf_1);
	}

	return sync_progress();
}

int main(int argc, char **argv)
{
	int ret;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_TAGGED;
	hints->mode = FI_CONTEXT;
	hints->tx_attr->op_flags = FI_DELIVERY_COMPLETE;

	ret = init_oob();
	if (ret)
		return ret;

	ret = initialize();
	if (ret)
		goto out;

	ret = run();
out:
	cleanup();
	return ret;
}
