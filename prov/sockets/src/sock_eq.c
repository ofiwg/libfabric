/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "sock.h"

static int _sock_eq_read_out_fd(struct sock_eq *sock_eq)
{
	int read_done = 0;
	do{
		char byte;
		read_done = read(sock_eq->fd[SOCK_RD_FD], &byte, 1);
		if(read_done < 0){
			fprintf(stderr, "Error reading CQ FD\n");
			return read_done;
		}
	}while(read_done == 1);
	return 0;
}

ssize_t sock_eq_read(struct fid_eq *eq, uint32_t *event, void *buf, size_t len,
		     uint64_t flags)
{
	struct sock_eq *sock_eq;
	struct sock_eq_item *eq_entry;
	
	sock_eq = container_of(eq, struct sock_eq, eq);
	if(!sock_eq)
		return -FI_ENOENT;
	
	if(peek_item(sock_eq->error_list))
		return -FI_EAVAIL;
	
	if(FI_PEEK & flags)
		eq_entry = peek_item(sock_eq->completed_list);
	else{
		eq_entry = dequeue_item(sock_eq->completed_list);
		if(eq_entry){
			_sock_eq_read_out_fd(sock_eq);
		}
	}

	if(eq_entry){
		int copy_len = MIN(len, eq_entry->len);
		memcpy(buf, (char*)eq_entry + sizeof(struct sock_eq_item), copy_len);

		if(event)
			*event = eq_entry->type;

		if(!(FI_PEEK & flags))
			free(eq_entry);

		return copy_len;
	}
	return 0;
}

ssize_t sock_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *buf,
		     size_t len, uint64_t flags)
{
	struct sock_eq *sock_eq;
	struct sock_eq_item *eq_entry;
	
	sock_eq = container_of(eq, struct sock_eq, eq);
	if(!sock_eq)
		return -FI_ENOENT;
	
	if(peek_item(sock_eq->error_list))
		return -FI_EAVAIL;
	
	if(FI_PEEK & flags)
		eq_entry = peek_item(sock_eq->error_list);
	else{
		eq_entry = dequeue_item(sock_eq->error_list);
		if(eq_entry){
			_sock_eq_read_out_fd(sock_eq);
		}
	}

	if(eq_entry){
		int copy_len = MIN(len, eq_entry->len);
		memcpy(buf, (char*)eq_entry + sizeof(struct sock_eq_item), copy_len);

		if(!(FI_PEEK & flags))
			free(eq_entry);

		return copy_len;
	}
	return 0;
}

static ssize_t _sock_eq_write(struct sock_eq *sock_eq, 
			      int event, 
			      const void *buf, size_t len)
{
	int ret;
	char byte;
	struct sock_eq_item *eq_entry = calloc(1, len + sizeof(struct sock_eq_item));
	if(!eq_entry)
		return -FI_ENOMEM;

	if((ret = write(sock_eq->fd[SOCK_WR_FD], &byte, 1)) < 0){
		free(eq_entry);
		return -errno;
	}

	eq_entry->type = event;
	eq_entry->len = len;
	memcpy((char*)eq_entry + sizeof(struct sock_eq_item), buf, len);
	
	ret = enqueue_item(sock_eq->completed_list, eq_entry);
	return (ret == 0) ? len : ret;
}

ssize_t _sock_eq_report_error(struct sock_eq *sock_eq, const void *buf, size_t len)
{
	int ret;
	char byte;
	struct sock_eq_item *eq_entry = calloc(1, len + sizeof(struct sock_eq_item));
	if(!eq_entry)
		return -FI_ENOMEM;

	if((ret = write(sock_eq->fd[SOCK_WR_FD], &byte, 1)) < 0){
		free(eq_entry);
		return -errno;
	}
	
	eq_entry->len = len;
	memcpy((char*)eq_entry + sizeof(struct sock_eq_item), buf, len);
	
	ret = enqueue_item(sock_eq->error_list, eq_entry);
	return (ret == 0) ? len : ret;
}

static ssize_t sock_eq_write(struct fid_eq *eq, uint32_t event, 
		      const void *buf, size_t len, uint64_t flags)
{
	struct sock_eq *sock_eq;
	sock_eq = container_of(eq, struct sock_eq, eq);
	if(!sock_eq)
		return -FI_ENOENT;
	
	if(!(sock_eq->attr.flags & FI_WRITE))
		return -FI_EINVAL;
	
	return _sock_eq_write(sock_eq, event, buf, len);
}

ssize_t _sock_eq_report_event(struct sock_eq *sock_eq, int event, 
			      const void *buf, size_t len)
{
	return sock_eq_write(&sock_eq->eq, event, buf, len, 0);
}

ssize_t sock_eq_sread(struct fid_eq *eq, uint32_t *event, 
			  void *buf, size_t len, int timeout, uint64_t flags)
{
	int ret;
	fd_set rfds;
	struct timeval tv, *ptv;
	struct sock_eq *sock_eq;

	sock_eq = container_of(eq, struct sock_eq, eq);
	if(!sock_eq)
		return -FI_ENOENT;

	FD_ZERO(&rfds);
	FD_SET(sock_eq->fd[SOCK_RD_FD], &rfds);
	tv.tv_sec = timeout;
	tv.tv_usec = 0;
	ptv = (timeout >= 0) ? &tv : NULL;

	ret = select(1, &rfds, NULL, NULL, ptv);
	if (ret == -1)
		return ret;
	else if (ret == 0)
		return -FI_ETIMEDOUT;
	else
		return sock_eq_read(eq, event, buf, len, flags);
}

const char * sock_eq_strerror(struct fid_eq *eq, int prov_errno,
			      const void *err_data, void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

static struct fi_ops_eq sock_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = sock_eq_read,
	.readerr = sock_eq_readerr,
	.write = sock_eq_write,
	.sread = sock_eq_sread,
	.strerror = sock_eq_strerror,
};

int sock_eq_fi_close(struct fid *fid)
{
	struct sock_eq *sock_eq;
	sock_eq = container_of(fid, struct sock_eq, eq.fid);
	if(!sock_eq)
		return -FI_ENOENT;

	free_list(sock_eq->completed_list);
	free_list(sock_eq->error_list);

	close(sock_eq->fd[SOCK_RD_FD]);
	close(sock_eq->fd[SOCK_WR_FD]);

	free(sock_eq);
	return 0;
}

int sock_eq_fi_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return -FI_ENOSYS;
}

int sock_eq_fi_sync(struct fid *fid, uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}

int sock_eq_fi_control(struct fid *fid, int command, void *arg)
{
	return -FI_ENOSYS;
}

int sock_eq_fi_open(struct fid *fid, const char *name,
		    uint64_t flags, void **ops, void *context)
{
	return -FI_ENOSYS;
}

static struct fi_ops sock_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sock_eq_fi_close,
	.bind = fi_no_bind,
	.sync = fi_no_sync,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int _sock_eq_verify_attr(struct fi_eq_attr *attr)
{
	if(!attr)
		return 0;

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_FD:
		break;
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
		break;
	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static struct fi_eq_attr _sock_eq_def_attr ={
	.size = SOCK_EQ_DEF_LEN,
	.flags = 0,
	.wait_obj = FI_WAIT_FD,
	.signaling_vector = 0,
	.wait_set = NULL,
};

int sock_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context)
{
	int ret;
	long flags;
	struct sock_eq *sock_eq;

	ret = _sock_eq_verify_attr(attr);
	if (ret)
		return ret;
	
	sock_eq = (struct sock_eq *)calloc(1, sizeof(struct sock_eq));
	if(!sock_eq)
		return -FI_ENOMEM;

	sock_eq->sock_fab = container_of(fabric, struct sock_fabric, fab_fid);

	sock_eq->eq.fid.fclass = FI_CLASS_EQ;
	sock_eq->eq.fid.context = context;
	sock_eq->eq.fid.ops = &sock_eq_fi_ops;	
	sock_eq->eq.ops = &sock_eq_ops;	

	if(attr == NULL)
		memcpy(&sock_eq->attr, &_sock_eq_def_attr, 
		       sizeof(struct fi_cq_attr));
	else
		memcpy(&sock_eq->attr, attr, sizeof(struct fi_cq_attr));		

	ret = socketpair(AF_UNIX, SOCK_STREAM, 0, sock_eq->fd);
	if (ret){
		ret = -errno;
		goto err1;
	}
	
	fcntl(sock_eq->fd[SOCK_RD_FD], F_GETFL, &flags);
	ret = fcntl(sock_eq->fd[SOCK_RD_FD], F_SETFL, flags | O_NONBLOCK);
	if (ret) {
		ret = -errno;
		goto err1;
	}

	sock_eq->completed_list = new_list(sock_eq->attr.size);
	if(!sock_eq->completed_list)
		goto err1;

	sock_eq->error_list = new_list(sock_eq->attr.size);
	if(!sock_eq->error_list)
		goto err2;
	
	return 0;

err2:
	free_list(sock_eq->completed_list);
err1:
	free(sock_eq);
	return -FI_EAVAIL;
}
