/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2006 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017-2018 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <glob.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <dirent.h>
#include <errno.h>

#include "efa_ib.h"

#ifndef PCI_VENDOR_ID_AMAZON
#define PCI_VENDOR_ID_AMAZON 0x1d0f
#endif /* PCI_VENDOR_ID_AMAZON */

#ifndef PCI_DEV_ID_EFA_VF
#define PCI_DEV_ID_EFA_VF 0xefa0
#endif

#define HCA(v, d) { .vendor = PCI_VENDOR_ID_##v, .device = d }

struct {
	unsigned vendor;
	unsigned device;
} hca_table[] = {
	HCA(AMAZON, PCI_DEV_ID_EFA_VF),
};

HIDDEN int abi_ver;

struct ibv_sysfs_dev {
	char		        sysfs_name[IBV_SYSFS_NAME_MAX];
	char		        ibdev_name[IBV_SYSFS_NAME_MAX];
	char		        sysfs_path[IBV_SYSFS_PATH_MAX];
	char		        ibdev_path[IBV_SYSFS_PATH_MAX];
	struct ibv_sysfs_dev   *next;
	int			abi_ver;
	int			have_driver;
};

char *get_sysfs_path(void)
{
	char *env = NULL;
	char *sysfs_path = NULL;
	int len;

	/*
	 * Only follow use path passed in through the calling user's
	 * environment if we're not running SUID.
	 */
	if (getuid() == geteuid())
		env = getenv("SYSFS_PATH");

	if (env) {
		sysfs_path = strndup(env, IBV_SYSFS_PATH_MAX);
		len = strlen(sysfs_path);
		while (len > 0 && sysfs_path[len - 1] == '/') {
			--len;
			sysfs_path[len] = '\0';
		}
	} else {
		sysfs_path = strndup("/sys", IBV_SYSFS_PATH_MAX);
	}

	return sysfs_path;
}

/* Return true if the snprintf succeeded, false if there was truncation or
 * error.
 */
static inline bool __good_snprintf(size_t len, int rc)
{
	return (rc < len && rc >= 0);
}

#define check_snprintf(buf, len, fmt, ...)                                     \
	__good_snprintf(len, snprintf(buf, len, fmt, ##__VA_ARGS__))

static int efa_find_sysfs_devs(struct ibv_sysfs_dev **sysfs_dev_list)
{
	char class_path[IBV_SYSFS_PATH_MAX];
	DIR *class_dir;
	struct dirent *dent;
	struct ibv_sysfs_dev *sysfs_dev = NULL;
	char *sysfs_path;
	char value[8];
	int ret = 0;

	sysfs_path = get_sysfs_path();
	if (!sysfs_path)
		return -ENOMEM;
	if (!check_snprintf(class_path, sizeof(class_path),
			    "%s/class/infiniband_verbs", sysfs_path)) {
		ret = -ENOMEM;
		goto sysfs_path_free;
	}

	class_dir = opendir(class_path);
	if (!class_dir) {
		EFA_DBG(FI_LOG_CORE, "Opendir error: %d (%s)\n", errno,
			strerror(errno));
		ret = errno;
		goto sysfs_path_free;
	}

	*sysfs_dev_list = NULL;
	while ((dent = readdir(class_dir))) {
		struct stat buf;

		if (dent->d_name[0] == '.')
			continue;

		if (!sysfs_dev)
			sysfs_dev = malloc(sizeof(*sysfs_dev));
		if (!sysfs_dev) {
			ret = -ENOMEM;
			goto class_dir_close;
		}

		if (!check_snprintf(sysfs_dev->sysfs_path, sizeof(sysfs_dev->sysfs_path),
				    "%s/%s", class_path, dent->d_name))
			continue;

		if (stat(sysfs_dev->sysfs_path, &buf)) {
			EFA_INFO(FI_LOG_FABRIC, "couldn't stat '%s'.\n",
				 sysfs_dev->sysfs_path);
			continue;
		}

		if (!S_ISDIR(buf.st_mode))
			continue;

		if (!check_snprintf(sysfs_dev->sysfs_name, sizeof(sysfs_dev->sysfs_name),
				    "%s", dent->d_name))
			continue;

		if (fi_read_file(sysfs_dev->sysfs_path, "ibdev",
				 sysfs_dev->ibdev_name,
				 sizeof(sysfs_dev->ibdev_name)) < 0) {
			EFA_INFO(FI_LOG_FABRIC, "No ibdev class attr for '%s'.\n",
				 dent->d_name);
			continue;
		}

		sysfs_dev->ibdev_name[sizeof(sysfs_dev->ibdev_name) - 1] = '\0';

		if (strncmp(sysfs_dev->ibdev_name, "efa_", 4) != 0)
			continue;

		if (!check_snprintf(sysfs_dev->ibdev_path,
				    sizeof(sysfs_dev->ibdev_path),
				    "%s/class/infiniband/%s", sysfs_path,
				    sysfs_dev->ibdev_name))
			continue;

		sysfs_dev->next        = *sysfs_dev_list;
		sysfs_dev->have_driver = 0;
		if (fi_read_file(sysfs_dev->sysfs_path, "abi_version",
				 value, sizeof(value)) > 0)
			sysfs_dev->abi_ver = strtol(value, NULL, 10);
		else
			sysfs_dev->abi_ver = 0;

		*sysfs_dev_list = sysfs_dev;
		sysfs_dev      = NULL;
	}

	if (sysfs_dev)
		free(sysfs_dev);

class_dir_close:
	closedir(class_dir);
sysfs_path_free:
	free(sysfs_path);
	return ret;
}

static struct verbs_device *driver_init(const char *uverbs_sys_path, int abi_version)
{
	char value[8];
	struct efa_device *dev;
	unsigned vendor, device;
	int i;

	if (fi_read_file(uverbs_sys_path, "device/vendor", value,
			 sizeof(value)) < 0)
		return NULL;
	vendor = strtol(value, NULL, 16);

	if (fi_read_file(uverbs_sys_path, "device/device", value,
			 sizeof(value)) < 0)
		return NULL;
	device = strtol(value, NULL, 16);

	for (i = 0; i < ARRAY_SIZE(hca_table); ++i)
		if (vendor == hca_table[i].vendor &&
		    device == hca_table[i].device)
			goto found;

	return NULL;

found:
	dev = calloc(1, sizeof(*dev));
	if (!dev) {
		EFA_WARN(FI_LOG_FABRIC, "Couldn't allocate device for %s\n",
			 uverbs_sys_path);
		return NULL;
	}

	dev->page_size = sysconf(_SC_PAGESIZE);
	dev->abi_version = abi_version;

	return &dev->verbs_dev;
}

static struct ibv_device *device_init(struct ibv_sysfs_dev *sysfs_dev)
{
	struct verbs_device *vdev;
	struct ibv_device *dev;

	vdev = driver_init(sysfs_dev->sysfs_path, sysfs_dev->abi_ver);
	if (!vdev)
		return NULL;

	dev = &vdev->device;

	strcpy(dev->dev_name,   sysfs_dev->sysfs_name);
	strcpy(dev->dev_path,   sysfs_dev->sysfs_path);
	strcpy(dev->name,       sysfs_dev->ibdev_name);
	strcpy(dev->ibdev_path, sysfs_dev->ibdev_path);

	return dev;
}

static int check_abi_version(const char *path)
{
	char value[8];

	if (fi_read_file(path, "class/infiniband_verbs/abi_version",
			 value, sizeof(value)) < 0) {
		return -ENOSYS;
	}

	abi_ver = strtol(value, NULL, 10);

	if (abi_ver < IB_USER_VERBS_MIN_ABI_VERSION ||
	    abi_ver > IB_USER_VERBS_MAX_ABI_VERSION) {
		EFA_WARN(FI_LOG_FABRIC, "Kernel ABI version %d doesn't match library version %d.\n",
			 abi_ver, IB_USER_VERBS_MAX_ABI_VERSION);
		return -ENOSYS;
	}

	return 0;
}

static void check_memlock_limit(void)
{
	struct rlimit rlim;

	if (!geteuid())
		return;

	if (getrlimit(RLIMIT_MEMLOCK, &rlim)) {
		EFA_INFO(FI_LOG_FABRIC, "getrlimit(RLIMIT_MEMLOCK) failed.\n");
		return;
	}

	if (rlim.rlim_cur <= 32768)
		EFA_INFO(FI_LOG_FABRIC,
			 "RLIMIT_MEMLOCK is %lu bytes. This will severely limit memory registrations.\n",
			 rlim.rlim_cur);
}

static void add_device(struct ibv_device *dev,
		       struct ibv_device ***dev_list,
		       int *num_devices,
		       int *list_size)
{
	struct ibv_device **new_list;

	if (*list_size <= *num_devices) {
		*list_size = *list_size ? *list_size * 2 : 1;
		new_list = realloc(*dev_list, *list_size * sizeof(*new_list));
		if (!new_list)
			return;
		*dev_list = new_list;
	}

	(*dev_list)[(*num_devices)++] = dev;
}

HIDDEN int efa_ib_init(struct ibv_device ***list)
{
	struct ibv_sysfs_dev *sysfs_dev_list;
	struct ibv_sysfs_dev *sysfs_dev;
	struct ibv_sysfs_dev *next_dev;
	struct ibv_device *device;
	int num_devices = 0;
	int list_size = 0;
	char *sysfs_path;
	int ret;

	*list = NULL;

	sysfs_path = get_sysfs_path();
	if (!sysfs_path)
		return -ENOSYS;

	ret = check_abi_version(sysfs_path);
	if (ret)
		goto err_free_path;

	check_memlock_limit();

	ret = efa_find_sysfs_devs(&sysfs_dev_list);
	if (ret)
		goto err_free_path;

	sysfs_dev = sysfs_dev_list;
	while (sysfs_dev) {
		device = device_init(sysfs_dev);
		if (device) {
			add_device(device, list, &num_devices, &list_size);
			sysfs_dev->have_driver = 1;
		}
		sysfs_dev = sysfs_dev->next;
	}

	sysfs_dev = sysfs_dev_list;
	while (sysfs_dev) {
		next_dev = sysfs_dev->next;
		free(sysfs_dev);
		sysfs_dev = next_dev;
	}

	free(sysfs_path);

	return num_devices;

err_free_path:
	free(sysfs_path);
	return ret;
}
