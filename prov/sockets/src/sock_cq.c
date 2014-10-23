/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include <sys/socket.h>
#include <sys/types.h>

#include "list.h"
#include "sock.h"

static int _sock_cq_read_out_fd(struct sock_cq *sock_cq)
{
	int read_done = 0;
	do{
		char byte;
		read_done = read(sock_cq->fd[SOCK_RD_FD], &byte, 1);
		if(read_done < 0){
			fprintf(stderr, "Error reading CQ FD\n");
			return read_done;
		}
	}while(read_done == 1);
	return 0;
}

static ssize_t _sock_cq_entry_size(struct sock_cq *sock_cq)
{
	ssize_t size = 0;

	switch(sock_cq->attr.format){
	case FI_CQ_FORMAT_CONTEXT:
		size = sizeof(struct fi_cq_entry);
		break;

	case FI_CQ_FORMAT_MSG:
		size = sizeof(struct fi_cq_msg_entry);
		break;

	case FI_CQ_FORMAT_DATA:
		size = sizeof(struct fi_cq_data_entry);
		break;

	case FI_CQ_FORMAT_TAGGED:
		size = sizeof(struct fi_cq_tagged_entry);
		break;

	case FI_CQ_FORMAT_UNSPEC:
	default:
		size = -1;
		break;
	}
	return size;
}

static void _sock_cq_write_to_buf(struct sock_cq *sock_cq,
				  void *buf, struct sock_req_item *cq_entry)
{
	ssize_t size;
	switch(sock_cq->attr.format){

	case FI_CQ_FORMAT_CONTEXT:
	{
		struct fi_cq_entry entry;
		size = sizeof(struct fi_cq_entry);

		entry.op_context = cq_entry->context;
		memcpy(buf, &entry, size);
		break;
	}

	case FI_CQ_FORMAT_MSG:
	{
		struct fi_cq_msg_entry entry;
		size = sizeof(struct fi_cq_msg_entry);

		entry.op_context = cq_entry->context;
		entry.flags = cq_entry->flags;
		entry.len = cq_entry->total_len;
		memcpy(buf, &entry, size);
		break;
	}

	case FI_CQ_FORMAT_DATA:
	{
		struct fi_cq_data_entry entry;
		size = sizeof(struct fi_cq_data_entry);

		entry.op_context = cq_entry->context;
		entry.flags = cq_entry->flags;
		entry.len = cq_entry->total_len;
		if(cq_entry->comm_type == SOCK_COMM_TYPE_SEND ||
		   cq_entry->comm_type == SOCK_COMM_TYPE_SENDTO||
		   cq_entry->comm_type == SOCK_COMM_TYPE_SENDDATA||
		   cq_entry->comm_type == SOCK_COMM_TYPE_SENDDATATO)
			entry.buf = cq_entry->item.buf;
		else
			entry.buf = NULL;
		entry.data = cq_entry->data;
		memcpy(buf, &entry, size);
		break;
	}

	case FI_CQ_FORMAT_TAGGED:
	{
		struct fi_cq_tagged_entry entry;
		size = sizeof(struct fi_cq_tagged_entry);

		entry.op_context = cq_entry->context;
		entry.flags = cq_entry->flags;
		entry.len = cq_entry->total_len;
		if(cq_entry->comm_type == SOCK_COMM_TYPE_SEND ||
		   cq_entry->comm_type == SOCK_COMM_TYPE_SENDTO||
		   cq_entry->comm_type == SOCK_COMM_TYPE_SENDDATA||
		   cq_entry->comm_type == SOCK_COMM_TYPE_SENDDATATO)
			entry.buf = cq_entry->item.buf;
		else
			entry.buf = NULL;
		entry.data = cq_entry->data;
		entry.tag = cq_entry->tag;
		memcpy(buf, &entry, size);
		break;
	}
	default:
		fprintf(stderr, "Invalid CQ format!\n");
		break;
	}
}

static void _sock_cq_progress(struct sock_cq *sock_cq)
{
	list_element_t *curr = sock_cq->ep_list->head;
	while(curr){
		((struct sock_ep *)curr->data)->progress_fn((struct sock_ep *)curr->data, sock_cq);
		curr=curr->next;
	}
}

static ssize_t sock_cq_readfrom(struct fid_cq *cq, void *buf, size_t len,
				fi_addr_t *src_addr)
{
	int num_done = 0;
	struct sock_cq *sock_cq;
	ssize_t bytes_written = 0;
	struct sock_req_item *cq_entry;

	sock_cq = container_of(cq, struct sock_cq, cq_fid);
	if(!sock_cq)
		return -FI_ENOENT;

	if(peek_item(sock_cq->error_list))
		return -FI_EAVAIL;

	if(len < sock_cq->cq_entry_size)
		return -FI_ETOOSMALL;

	do{
		size_t entry_len;
		_sock_cq_progress(sock_cq);
		
		cq_entry = peek_item(sock_cq->completed_list);
		if(!cq_entry)
			return bytes_written;
		entry_len = cq_entry->req_type == SOCK_REQ_TYPE_USER ?
			cq_entry->total_len : sock_cq->cq_entry_size;
		if(len < entry_len)
			return bytes_written;
		cq_entry = dequeue_item(sock_cq->completed_list);
		_sock_cq_read_out_fd(sock_cq);

		if(cq_entry->req_type == SOCK_REQ_TYPE_USER){
			/* Handle user event */
			memcpy((char*)buf + bytes_written, cq_entry->item.buf,
			       cq_entry->total_len);
			bytes_written += cq_entry->total_len;
			len -= cq_entry->total_len;

			if(src_addr){
				fi_addr_t addr = FI_ADDR_UNSPEC;
				memcpy((char*)src_addr + num_done * sizeof(fi_addr_t), 
				       &addr, sizeof(fi_addr_t));
			}
			free(cq_entry->item.buf);
		}else{
			/* Handle completion event */
			_sock_cq_write_to_buf(sock_cq, 
					      (char *)buf + bytes_written, cq_entry);
			bytes_written += sock_cq->cq_entry_size;
			len -= sock_cq->cq_entry_size;
			
			if(src_addr){
				fi_addr_t addr;
				if(FI_SOURCE & cq_entry->ep->info.caps){
					addr = _sock_av_lookup(cq_entry->ep->av, &cq_entry->src_addr);
				}else{
					addr = FI_ADDR_UNSPEC;
				}
				memcpy((char*)src_addr + num_done * sizeof(fi_addr_t), 
				       &addr, sizeof(fi_addr_t));
			}
		}
		free(cq_entry);
		num_done++;
	}while(1);
	
	return bytes_written;
}

static ssize_t sock_cq_read(struct fid_cq *cq, void *buf, size_t len)
{
	return sock_cq_readfrom(cq, buf, len, NULL);
}

static ssize_t sock_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
			       size_t len, uint64_t flags)
{
	int num_done = 0;
	struct sock_cq *sock_cq;
	ssize_t bytes_written = 0;
	struct fi_cq_err_entry *err_entry;
	
	sock_cq = container_of(cq, struct sock_cq, cq_fid);
	if(!sock_cq)
		return -FI_ENOENT;
	
	if(len < sizeof(struct fi_cq_err_entry))
		return -FI_ETOOSMALL;

	while(len >= sizeof(struct fi_cq_err_entry)){
		
		err_entry = dequeue_item(sock_cq->error_list);
		_sock_cq_read_out_fd(sock_cq);

		if(err_entry){
			memcpy( (char *)buf + num_done * sizeof(struct fi_cq_err_entry), 
				err_entry, sizeof(struct fi_cq_err_entry));
			bytes_written += sizeof(struct fi_cq_err_entry);
			len -= sizeof(struct fi_cq_err_entry);
			
			free(err_entry);
			num_done++;
		}else{
			return bytes_written;
		}
	}
	
	return bytes_written;
}

static ssize_t sock_cq_write(struct fid_cq *cq, const void *buf, size_t len)
{
	struct sock_cq *sock_cq;
	struct sock_req_item *item;

	sock_cq = container_of(cq, struct sock_cq, cq_fid);
	if(!sock_cq)
		return -FI_ENOENT;

	if(!(sock_cq->attr.flags & FI_WRITE))
		return -FI_EINVAL;

	item = calloc(1, sizeof(struct sock_req_item));
	if(!item)
		return -FI_ENOMEM;
	
	item->req_type = SOCK_REQ_TYPE_USER;
	item->item.buf = malloc(len);
	if(!item->item.buf){
		free(item);
		return -FI_ENOMEM;
	}
	memcpy(item->item.buf, buf, len);

	if(0 != _sock_cq_report_completion(sock_cq, item)){
		free(item->item.buf);
		free(item);
		return -FI_EINVAL;
	}
	return len;
}

static ssize_t sock_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t len,
				    fi_addr_t *src_addr, const void *cond, int timeout)
{
	struct sock_cq *sock_cq;
	int wait_infinite;
	int64_t cq_threshold;
	double curr_time, end_time = 0.0;

	if(timeout>0){
		struct timeval now;
		gettimeofday(&now, NULL);
		end_time = (double)now.tv_sec * 1000000.0 + 
			(double)now.tv_usec + (double)timeout * 1000.0;
		wait_infinite = 0;
	}else{
		wait_infinite = 1;
	}

	sock_cq = container_of(cq, struct sock_cq, cq_fid);
	if(!sock_cq)
		return -FI_ENOENT;

	if (sock_cq->attr.wait_obj == FI_CQ_COND_THRESHOLD){
		cq_threshold = (int64_t)cond;
	}else{
		cq_threshold = 1;
	}

	do{
		if(peek_item(sock_cq->error_list))
			return -FI_EAVAIL;
		
		_sock_cq_progress(sock_cq);

		if(list_length(sock_cq->completed_list) >= cq_threshold)
			return sock_cq_readfrom(cq, buf, len, src_addr);
		
		if(!wait_infinite){
			struct timeval now;
			gettimeofday(&now, NULL);
			curr_time = (double)now.tv_sec * 1000000.0 + 
				(double)now.tv_usec;
			
			if(curr_time >= end_time)
				break;
		}
	}while(1);

	return sock_cq_readfrom(cq, buf, len, src_addr);
}

static ssize_t sock_cq_sread(struct fid_cq *cq, void *buf, size_t len,
			     const void *cond, int timeout)
{
	return sock_cq_sreadfrom(cq, buf, len, NULL, cond, timeout);
}

static const char * sock_cq_strerror(struct fid_cq *cq, int prov_errno,
				     const void *err_data, void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

static int sock_cq_close(struct fid *fid)
{
	struct sock_cq *cq;

	cq = container_of(fid, struct sock_cq, cq_fid.fid);
	if(!cq)
		return -FI_EINVAL;

	if (atomic_get(&cq->ref))
		return -FI_EBUSY;

	free_list(cq->ep_list);
	free_list(cq->completed_list);
	free_list(cq->error_list);

	close(cq->fd[SOCK_RD_FD]);
	close(cq->fd[SOCK_WR_FD]);

	free(cq);
	return 0;
}

static struct fi_ops_cq sock_cq_ops = {
	.read = sock_cq_read,
	.readfrom = sock_cq_readfrom,
	.readerr = sock_cq_readerr,
	.write = sock_cq_write,
	.sread = sock_cq_sread,
	.sreadfrom = sock_cq_sreadfrom,
	.strerror = sock_cq_strerror,
};

static struct fi_ops sock_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sock_cq_close,
};

static int sock_cq_verify_attr(struct fi_cq_attr *attr)
{
	if(!attr)
		return 0;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
	case FI_CQ_FORMAT_MSG:
	case FI_CQ_FORMAT_DATA:
	case FI_CQ_FORMAT_TAGGED:
		break;
	default:
		return -FI_ENOSYS;
	}

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

static struct fi_cq_attr _sock_cq_def_attr = {
	.size = SOCK_CQ_DEF_LEN,
	.flags = 0,
	.format = FI_CQ_FORMAT_CONTEXT,
	.wait_obj = FI_WAIT_FD,
	.signaling_vector = 0,
	.wait_cond = FI_CQ_COND_NONE,
	.wait_set = NULL,
};

int sock_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context)
{
	struct sock_domain *sock_dom;
	struct sock_cq *sock_cq;
	long flags = 0;
	int ret;

	sock_dom = container_of(domain, struct sock_domain, dom_fid);
	if(!sock_dom)
		return -FI_EINVAL;

	ret = sock_cq_verify_attr(attr);
	if (ret)
		return ret;

	sock_cq = calloc(1, sizeof(*sock_cq));
	if (!sock_cq)
		return -FI_ENOMEM;

	atomic_init(&sock_cq->ref);
	sock_cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	sock_cq->cq_fid.fid.context = context;
	sock_cq->cq_fid.fid.ops = &sock_cq_fi_ops;
	sock_cq->cq_fid.ops = &sock_cq_ops;
	atomic_inc(&sock_dom->ref);

	if(attr == NULL)
		memcpy(&sock_cq->attr, &_sock_cq_def_attr, 
		       sizeof(struct fi_cq_attr));
	else
		memcpy(&sock_cq->attr, attr, sizeof(struct fi_cq_attr));

	sock_cq->domain = sock_dom;
	sock_cq->cq_entry_size = _sock_cq_entry_size(sock_cq);
	
	sock_cq->ep_list = new_list(128);
	sock_cq->completed_list = new_list(sock_cq->attr.size);
	sock_cq->error_list = new_list(sock_cq->attr.size);
	if(!sock_cq->ep_list || !sock_cq->completed_list || 
	   !sock_cq->error_list){
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = socketpair(AF_UNIX, SOCK_STREAM, 0, sock_cq->fd);
	if(ret){
		ret = -errno;
		goto err;
	}

	fcntl(sock_cq->fd[SOCK_RD_FD], F_GETFL, &flags);
	ret = fcntl(sock_cq->fd[SOCK_RD_FD], F_SETFL, flags | O_NONBLOCK);
	if (ret) {
		ret = -errno;
		goto err;
	}
	
	*cq = &sock_cq->cq_fid;
	return 0;

err:
	free(sock_cq);
	return ret;
}

int _sock_cq_report_completion(struct sock_cq *sock_cq, 
			      struct sock_req_item *item)
{
	char byte;
	write(sock_cq->fd[SOCK_WR_FD], &byte, 1);
	return enqueue_item(sock_cq->completed_list, item);
}

int _sock_cq_report_error(struct sock_cq *sock_cq, 
			  struct fi_cq_err_entry *error)
{
	char byte;
	write(sock_cq->fd[SOCK_WR_FD], &byte, 1);
	return enqueue_item(sock_cq->error_list, error);
}
