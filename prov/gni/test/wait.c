/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
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

#include "gnix.h"
#include "gnix_wait.h"
#include <fi_list.h>
#include <string.h>

#include <criterion/criterion.h>

static struct fid_fabric *fab;
static struct fi_info *hints;
static struct fi_info *fi;
static struct gnix_fid_wait *wait_priv;
static struct fi_wait_attr wait_attr;
static struct fid_wait *wait_set;

void wait_setup(void)
{
	int ret = 0;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->mode = ~0;

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");
}

void wait_teardown(void)
{
	int ret = 0;

	ret = fi_close(&wait_set->fid);
	cr_assert(!ret, "failure in closing wait set.");

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

void setup_wait_type(enum fi_wait_obj wait_obj)
{
	int ret;

	wait_setup();
	wait_attr.wait_obj = wait_obj;

	ret = fi_wait_open(fab, &wait_attr, &wait_set);
	cr_assert(!ret, "fi_wait_open");

	wait_priv = container_of(wait_set, struct gnix_fid_wait, wait);
}

void unspec_setup(void)
{
	setup_wait_type(FI_WAIT_UNSPEC);
}

void fd_setup(void)
{
	setup_wait_type(FI_WAIT_FD);
}

void mutex_cond_setup(void)
{
	setup_wait_type(FI_WAIT_MUTEX_COND);
}

Test(wait_creation, unspec, .init = unspec_setup, .fini = wait_teardown)
{
	cr_expect_eq(wait_priv->type, FI_WAIT_FD);
	cr_expect_eq(wait_priv->type, wait_attr.wait_obj);
	cr_expect_eq(&wait_priv->fabric->fab_fid, fab);
	cr_expect_eq(wait_priv->cond_type, FI_CQ_COND_NONE);
}

Test(wait_creation, fd, .init = fd_setup, .fini = wait_teardown)
{
	cr_expect_eq(wait_priv->type, FI_WAIT_FD);
	cr_expect_eq(wait_priv->type, wait_attr.wait_obj);
	cr_expect_eq(&wait_priv->fabric->fab_fid, fab);
	cr_expect_eq(wait_priv->cond_type, FI_CQ_COND_NONE);
}

Test(wait_creation, mutex_cond, .init = mutex_cond_setup, .fini = wait_teardown)
{
	cr_expect_eq(wait_priv->type, FI_WAIT_MUTEX_COND);
	cr_expect_eq(wait_priv->type, wait_attr.wait_obj);
	cr_expect_eq(&wait_priv->fabric->fab_fid, fab);
	cr_expect_eq(wait_priv->cond_type, FI_CQ_COND_NONE);
}

Test(wait_control, unspec, .init = unspec_setup, .fini = wait_teardown)
{
	int fd;
	int ret;

	ret = fi_control(&wait_priv->wait.fid, FI_GETWAIT, &fd);
	cr_expect_eq(FI_SUCCESS, ret, "fi_control failed.");

	cr_expect_eq(wait_priv->fd[WAIT_READ], fd);
}

Test(wait_control, fd, .init = fd_setup, .fini = wait_teardown)
{
	int fd;
	int ret;

	ret = fi_control(&wait_priv->wait.fid, FI_GETWAIT, &fd);
	cr_expect_eq(FI_SUCCESS, ret, "fi_control failed.");

	cr_expect_eq(wait_priv->fd[WAIT_READ], fd);
}

Test(wait_control, mutex_cond, .init = mutex_cond_setup, .fini = wait_teardown)
{
	int ret;
	struct fi_mutex_cond mutex_cond;

	ret = fi_control(&wait_priv->wait.fid, FI_GETWAIT, &mutex_cond);
	cr_expect_eq(FI_SUCCESS, ret, "fi_control failed.");

	ret = memcmp(&wait_priv->mutex, mutex_cond.mutex,
		     sizeof(*mutex_cond.mutex));
	cr_expect_eq(0, ret, "mutex compare failed.");

	ret = memcmp(&wait_priv->cond, mutex_cond.cond,
		     sizeof(*mutex_cond.cond));
	cr_expect_eq(0, ret, "cond compare failed.");
}

Test(wait_set, add, .init = fd_setup)
{
	int ret;
	struct gnix_wait_entry *entry;

	struct fid temp_wait = {
		.fclass = FI_CLASS_CQ
	};

	cr_expect(slist_empty(&wait_priv->set),
		  "wait set is not initially empty.");
	ret = _gnix_wait_set_add(&wait_priv->wait, &temp_wait);

	cr_expect_eq(FI_SUCCESS, ret, "gnix_wait_set_add failed.");

	cr_expect(!slist_empty(&wait_priv->set),
		  "wait set is empty after add.");

	entry = container_of(wait_priv->set.head, struct gnix_wait_entry,
			     entry);

	ret = memcmp(entry->wait_obj, &temp_wait, sizeof(temp_wait));
	cr_expect_eq(0, ret, "wait objects are not equal.");

	ret = fi_close(&wait_set->fid);
	cr_expect_eq(-FI_EBUSY, ret);

	ret = _gnix_wait_set_remove(&wait_priv->wait, &temp_wait);

	cr_expect_eq(FI_SUCCESS, ret, "gnix_wait_set_remove failed.");

	ret = fi_close(&wait_set->fid);
	cr_expect_eq(FI_SUCCESS, ret, "fi_close on wait set failed.");

	ret = fi_close(&fab->fid);
	cr_expect_eq(FI_SUCCESS, ret, "failure in closing fabric.");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

Test(wait_set, empty_remove, .init = fd_setup)
{
	int ret;

	struct fid temp_wait = {
		.fclass = FI_CLASS_CQ
	};

	cr_expect(slist_empty(&wait_priv->set));
	ret = _gnix_wait_set_remove(&wait_priv->wait, &temp_wait);
	cr_expect_eq(-FI_EINVAL, ret);
	cr_expect(slist_empty(&wait_priv->set));

	ret = fi_close(&wait_set->fid);
	cr_expect_eq(FI_SUCCESS, ret, "fi_close on wait set failed.");

	ret = fi_close(&fab->fid);
	cr_expect_eq(FI_SUCCESS, ret, "fi_close on fabric failed.");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

Test(wait_verify, invalid_type, .init = wait_setup)
{
	int ret;

	wait_attr.wait_obj = FI_WAIT_SET;

	ret = fi_wait_open(fab, &wait_attr, &wait_set);
	cr_expect_eq(-FI_EINVAL, ret,
		     "Requesting incorrect type FI_WAIT_SET succeeded.");

	ret = fi_wait_open(fab, NULL, &wait_set);
	cr_expect_eq(-FI_EINVAL, ret,
		     "Requesting verification with NULL attr succeeded.");

	wait_attr.flags = 1;
	ret = fi_wait_open(fab, &wait_attr, &wait_set);
	cr_expect_eq(-FI_EINVAL, ret,
		     "Requesting verifications with flags set succeeded.");
}
