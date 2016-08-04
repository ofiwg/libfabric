/*
 * Copyright (c) 2016 Cray Inc. All rights reserved.
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

#ifdef HAVE_KDREG

#include "gnix_mr_notifier.h"

static inline int
notifier_verify_stuff(struct gnix_mr_notifier *mrn) {
	/* Can someone confirm that these values are POSIX so we can
	 * be less pedantic? */
	if (mrn->fd == STDIN_FILENO ||
	    mrn->fd == STDOUT_FILENO ||
	    mrn->fd == STDERR_FILENO ||
	    mrn->fd < 0) {
		// Be quiet here
		return -FI_EBADF;
	}

	if (mrn->cntr == NULL) {
		// Be quiet here
		return -FI_ENODATA;
	}

	return FI_SUCCESS;
}

int
_gnix_notifier_init(struct gnix_mr_notifier *mrn)
{
	if (mrn == NULL) {
		GNIX_INFO(FI_LOG_MR, "mr notifier NULL\n");
		return -FI_EINVAL;
	}

	mrn->fd = 0;
	mrn->cntr = NULL;
	fastlock_init(&mrn->lock);

	return FI_SUCCESS;
}

int
_gnix_notifier_open(struct gnix_mr_notifier *mrn)
{
	int ret = FI_SUCCESS;
	int kdreg_fd, ret_errno;
        kdreg_get_user_delta_args_t get_user_delta_args;

	if ((mrn->fd != 0) || (mrn->cntr != NULL)) {
		GNIX_INFO(FI_LOG_MR, "mr notifier already open\n");
		return -FI_EBUSY;
	}

	fastlock_acquire(&mrn->lock);

	kdreg_fd = open(KDREG_DEV, O_RDWR | O_NONBLOCK);
	if (kdreg_fd < 0) {
		ret_errno = errno;
		if (ret_errno != FI_EBUSY) {
			GNIX_INFO(FI_LOG_MR, "kdreg device open failed: %s\n",
				  strerror(ret_errno));
		}
		// Not all of these map to fi_errno values
		ret = -ret_errno;
		goto err_exit;
	}

	(void) memset(&get_user_delta_args,0,sizeof(get_user_delta_args));
	if (ioctl(kdreg_fd, KDREG_IOC_GET_USER_DELTA,
		  &get_user_delta_args) < 0) {
		ret_errno = errno;
		GNIX_INFO(FI_LOG_MR, "kdreg get_user_delta failed: %s\n",
			  strerror(ret_errno));
		close(kdreg_fd);
		// Not all of these map to fi_errno values
		ret = -ret_errno;
		goto err_exit;
	}

	if (get_user_delta_args.user_delta == NULL) {
		GNIX_INFO(FI_LOG_MR, "kdreg get_user_delta is NULL\n");
		ret = -FI_ENODATA;
		goto err_exit;
	}

	mrn->fd = kdreg_fd;
	mrn->cntr = (kdreg_user_delta_t *) get_user_delta_args.user_delta;

err_exit:
	fastlock_release(&mrn->lock);

	return ret;
}

int
_gnix_notifier_close(struct gnix_mr_notifier *mrn)
{
	int ret = FI_SUCCESS;
	int ret_errno;

	ret = notifier_verify_stuff(mrn);

	if (ret == 0) {
		fastlock_acquire(&mrn->lock);

		if (close(mrn->fd) != 0) {
			ret_errno = errno;
			GNIX_INFO(FI_LOG_MR, "error closing kdreg device: %s\n",
				  strerror(ret_errno));
			// Not all of these map to fi_errno values
			ret = -ret_errno;
			goto err_exit;
		}

		mrn->cntr = NULL;
	err_exit:
		fastlock_release(&mrn->lock);
	}

	return ret;
}

static inline int
kdreg_write(struct gnix_mr_notifier *mrn, void *buf, size_t len) {
	int ret;

	ret = write(mrn->fd, buf, len);
	if ((ret < 0) || (ret != len)) {
		// Not all of these map to fi_errno values
		ret = -errno;
		GNIX_INFO(FI_LOG_MR, "kdreg_write failed: %s\n",
			  strerror(errno));
		return ret;
	}

	return FI_SUCCESS;
}

int
_gnix_notifier_monitor(struct gnix_mr_notifier *mrn,
		    void *addr, uint64_t len, uint64_t cookie)
{
	int ret;
	struct registration_monitor rm;

	ret = notifier_verify_stuff(mrn);

	if (ret == 0) {
		GNIX_DEBUG(FI_LOG_MR, "monitoring %p (len=%lu) cookie=%lu\n",
			   addr, len, cookie);

		memset(&rm, 0, sizeof(rm));
		rm.type = REGISTRATION_MONITOR;
		rm.u.mon.addr = (uint64_t) addr;
		rm.u.mon.len = len;
		rm.u.mon.user_cookie = cookie;

		ret = kdreg_write(mrn, &rm, sizeof(rm));
	}

	return ret;
}

int
_gnix_notifier_unmonitor(struct gnix_mr_notifier *mrn, uint64_t cookie)
{
	int ret;
	struct registration_monitor rm;

	ret = notifier_verify_stuff(mrn);
	if (ret == 0) {
		GNIX_DEBUG(FI_LOG_MR, "unmonitoring cookie=%lu\n", cookie);

		memset(&rm, 0, sizeof(rm));

		rm.type = REGISTRATION_UNMONITOR;
		rm.u.unmon.user_cookie = cookie;

		ret = kdreg_write(mrn, &rm, sizeof(rm));
	}

	return ret;
}

int
_gnix_notifier_get_event(struct gnix_mr_notifier *mrn, void* buf, size_t len)
{
	int ret, ret_errno;

	if ((mrn == NULL) || (buf == NULL) || (len <= 0)) {
		GNIX_INFO(FI_LOG_MR,
			  "Invalid argument to _gnix_notifier_get_event\n");
		return -FI_EINVAL;
	}

	if (*(mrn->cntr) > 0) {
		GNIX_DEBUG(FI_LOG_MR, "reading kdreg event\n");
		ret = read(mrn->fd, buf, len);
		if (ret >= 0) {
			return ret;
		} else {
			ret_errno = errno;
			if (ret_errno != EAGAIN) {
				GNIX_WARN(FI_LOG_MR,
					  "kdreg event read failed: %s\n",
					  strerror(ret_errno));
			}
			// Not all of these map to fi_errno values
			return -ret_errno;
		}
	} else {
		GNIX_DEBUG(FI_LOG_MR, "nothing to read from kdreg :(\n");
		return -FI_EAGAIN;
	}
}

#endif /* HAVE_KDREG */
