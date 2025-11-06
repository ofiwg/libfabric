/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* User space-CXI device interaction */

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>
#include <stdio.h>
#include <glob.h>
#include <linux/limits.h>
#include <libgen.h>

#include "libcxi_priv.h"

#define CXIL_DEV_BASE "/sys/class/cxi_user/cxi"
#define CXIL_DEV_FMT CXIL_DEV_BASE "%u"
#define CXIL_DEVINFO_FNAME(fname) CXIL_DEV_FMT "/device/" fname
#define STRINGIFY(s) STR(s)
#define STR(s) #s

int read_sysfs_var(char *fname, char *var_fmt, void *var)
{
	FILE *file;
	int rc;

	file = fopen(fname, "r");
	if (!file)
		return -errno;

	rc = fscanf(file, var_fmt, var);
	if (rc != 1)
		rc = -EINVAL;
	else
		rc = 0;

	fclose(file);

	return rc;
}

static int read_sysfs_string(char *fname, char *out_buf, size_t buf_len)
{
	FILE *file;
	char *line = NULL;
	size_t len = 0;
	int rc;

	if (!out_buf || !buf_len)
		return -EINVAL;

	file = fopen(fname, "r");
	if (!file)
		return -errno;

	rc = getline(&line, &len, file);

	fclose(file);

	if (rc == -1)
		return -ENODATA;

	/* Remove \n */
	if (rc > 1 && line[rc - 1] == '\n')
		line[rc - 1] = 0;

	strncpy(out_buf, line, buf_len - 1);
	out_buf[buf_len - 1] = '\0';

	free(line);

	return 0;
}

int cxil_query_devinfo(uint32_t dev_id, struct cxil_dev *dev)
{
	struct cxi_dev_info_use dev_info_use = {};
	struct cxil_devinfo *info;
	int rc;
	char fname[PATH_MAX];
	char pname[PATH_MAX];
	char cassini_version[50];
	char *rpath;
	char *bname;
	unsigned int domain;
	unsigned int bus;
	unsigned int device;
	unsigned int function;

	info = &dev->info;
	memset(info, 0, sizeof(*info));

	rc = snprintf(fname, PATH_MAX, CXIL_DEV_FMT, dev_id);
	if (rc < 0)
		return rc;

	rc = access(fname, R_OK);
	if (rc)
		return -ENOENT;

	/* Set device index */
	info->dev_id = dev_id;

	/* Set device name */
	rc = snprintf(info->device_name, CXIL_DEVNAME_MAX, "cxi%u", dev_id);
	if (rc < 0)
		return rc;

	/* Parse driver name */
	rc = snprintf(fname, PATH_MAX, CXIL_DEVINFO_FNAME("driver"),
		      dev_id);
	if (rc < 0)
		return -ENOMEM;

	rpath = realpath(fname, pname);
	if (rpath != pname)
		return -errno;

	bname = basename(rpath);

	rc = sscanf(bname, "%" STRINGIFY(CXIL_DRVNAME_MAX) "s",
		    info->driver_name);
	if (rc != 1)
		return rc;

	/* Parse PCI topology */
	rc = snprintf(fname, PATH_MAX, CXIL_DEV_FMT "/device", dev_id);
	if (rc < 0)
		return -ENOMEM;

	rpath = realpath(fname, pname);
	if (rpath != pname)
		return -errno;

	bname = basename(rpath);

	rc = sscanf(bname, "%x:%x:%x.%x", &domain, &bus, &device, &function);
	if (rc != 4)
		return -EINVAL;

	info->pci_domain = domain;
	info->pci_bus = bus;
	info->pci_device = device;
	info->pci_function = function;

	/* Check whether the device is a physical or virtual function */
	rc = snprintf(fname, PATH_MAX, CXIL_DEVINFO_FNAME("physfn"), dev_id);
	if (rc < 0)
		return -ENOMEM;

	if (access(fname, F_OK) == 0)
		info->is_vf = true;

	rc = cxil_get_dev_info(dev, &dev_info_use);
	/* If there is an error fallback to sysfs read */
	if (rc) {
		/* Parse NID */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/nid"), dev_id);
		if (rc < 0)
			return rc;

		rc = read_sysfs_var(fname, "%x", &info->nid);
		if (rc)
			return rc;

		/* Parse PID Bits */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/pid_bits"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->pid_bits);
		if (rc)
			return rc;

		/* Parse PID Count */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/pid_count"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->pid_count);
		if (rc)
			return rc;

		/* Parse PID Granule */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/pid_granule"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->pid_granule);
		if (rc)
			return rc;

		/* Parse min_free_shift */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/min_free_shift"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->min_free_shift);
		if (rc)
			return rc;

		/* Parse rdzv_get_idx */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/rdzv_get_idx"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->rdzv_get_idx);
		if (rc)
			return rc;

		/* Parse PCI vendor ID */
		rc = snprintf(fname, PATH_MAX, CXIL_DEVINFO_FNAME("vendor"),
		      dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%x", &info->vendor_id);
		if (rc)
			return rc;

		/* Parse PCI device ID */
		rc = snprintf(fname, PATH_MAX, CXIL_DEVINFO_FNAME("device"),
		      dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%x", &info->device_id);
		if (rc)
			return rc;

		/* Cassini version */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/cassini_version"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%s", cassini_version);
		if (rc)
			return rc;

		if (!strcmp(cassini_version, "1.0"))
			info->cassini_version = CASSINI_1_0;
		else if (!strcmp(cassini_version, "1.1"))
			info->cassini_version = CASSINI_1_1;
		else if (!strcmp(cassini_version, "2.0"))
			info->cassini_version = CASSINI_2_0;
		else
			return -EINVAL;

		/* System info */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/system_type_identifier"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		/* Not all drivers support system_type_identifier. */
		rc = read_sysfs_var(fname, "%u", &info->system_type_identifier);
		if (rc) {
			if ((info->cassini_version & CASSINI_2) == CASSINI_2)
				info->system_type_identifier = CASSINI_2_ONLY;
			else
				info->system_type_identifier = CASSINI_1_ONLY;
		}

		/* Parse device revision */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/rev"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->device_rev);
		if (rc)
			return rc;

		/* Parse device prototype version */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/proto"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->device_proto);
		if (rc)
			return rc;

		/* Parse device platform */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/platform"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->device_platform);
		if (rc)
			return rc;

		/* Parse uC NIC index */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/uc_nic"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%i", &info->uc_nic);
		if (rc)
			return rc;

		/* Parse PCT EQ */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/pct_eq"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%i", &info->pct_eq);
		if (rc)
			return rc;

			/* Parse MTU */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/hpc_mtu"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%u", &info->link_mtu);
		if (rc)
			return rc;

		/* Parse link speed */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/speed"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%lu", &info->link_speed);
		if (rc)
			return rc;

		/* Parse link state */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/link"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hhu", &info->link_state);
		if (rc)
			return rc;

		/* Get device resource limits*/
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_ptes"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_ptes);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_txqs"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_txqs);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_tgqs"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_tgqs);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_eqs"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_eqs);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_cts"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_cts);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_acs"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_acs);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_tles"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_tles);
		if (rc)
			return rc;

		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/num_les"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		rc = read_sysfs_var(fname, "%hu", &info->num_les);
		if (rc)
			return rc;

		/* Get NIC board type */
		rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("fru/description"), dev_id);
		if (rc < 0)
			return -ENOMEM;

		/* FRU info is not available if the VPD SEEP is not programmed.
		 * This is not fatal.
		 */
		rc = read_sysfs_string(fname, info->fru_description,
			       CXIL_FRUDESC_MAX);
		if (rc)
			strcpy(info->fru_description, "Not Available");
	} else {
		dev->info.nid = dev_info_use.nid;
		dev->info.pid_bits = dev_info_use.pid_bits;
		dev->info.pid_count = dev_info_use.pid_count;
		dev->info.pid_granule = dev_info_use.pid_granule;
		dev->info.min_free_shift = dev_info_use.min_free_shift;
		dev->info.rdzv_get_idx = dev_info_use.rdzv_get_idx;
		dev->info.vendor_id = dev_info_use.vendor_id;
		dev->info.device_id = dev_info_use.device_id;
		dev->info.device_rev = dev_info_use.device_rev;
		dev->info.device_proto = dev_info_use.device_proto;
		dev->info.device_platform = dev_info_use.device_platform;
		dev->info.pct_eq = dev_info_use.pct_eq;
		dev->info.uc_nic = dev_info_use.uc_nic;

		dev->info.link_mtu = dev_info_use.link_mtu;
		dev->info.link_speed = dev_info_use.link_speed;
		dev->info.link_state = dev_info_use.link_state;
		dev->info.num_ptes = dev_info_use.num_ptes;
		dev->info.num_txqs = dev_info_use.num_txqs;
		dev->info.num_tgqs = dev_info_use.num_tgqs;
		dev->info.num_eqs = dev_info_use.num_eqs;
		dev->info.num_cts = dev_info_use.num_cts;
		dev->info.num_acs = dev_info_use.num_acs;
		dev->info.num_tles = dev_info_use.num_tles;
		dev->info.num_les = dev_info_use.num_les;

		dev->info.cassini_version = dev_info_use.cassini_version;
		dev->info.system_type_identifier = dev_info_use.system_type_identifier;

		strcpy(dev->info.fru_description, dev_info_use.fru_description);
	}

	return 0;
}

CXIL_API int cxil_get_device_list(struct cxil_device_list **dev_list)
{
	glob_t globbuf;
	int rc;
	struct cxil_device_list *list;
	struct cxil_dev_priv *new_dev;
	char *dev_name;
	unsigned int count;
	size_t list_size;
	int i;

	rc = glob(CXIL_DEV_BASE "*", 0, NULL, &globbuf);
	if (rc == 0) {
		count = globbuf.gl_pathc;
	} else if (rc == GLOB_NOMATCH) {
		count = 0;
	} else {
		/* Failed for some reason */
		return -EINVAL;
	}

	list_size = sizeof(struct cxil_device_list) +
		count * sizeof(list->info[0]);
	list = calloc(1, list_size);
	if (list == NULL) {
		rc = -ENOMEM;
		goto free_glob;
	}

	for (i = 0; i < count; i++) {
		const char *num = &globbuf.gl_pathv[i][strlen(CXIL_DEV_BASE)];

		if (*num) {
			new_dev = calloc(1, sizeof(*new_dev));
			if (new_dev == NULL) {
				rc = -ENOMEM;
				goto free_glob;
			}

			rc = asprintf(&dev_name, "/dev/cxi%u", atoi(num));
			if (rc == -1) {
				rc = errno ? -errno : -ENOMEM;
				goto err_free_dev;
			}

			new_dev->fd = open(dev_name, O_RDWR | O_CLOEXEC);
			if (new_dev->fd == -1) {
				rc = -errno;
				goto err_free_dev_name;
			}

			rc = cxil_query_devinfo(atoi(num), &new_dev->dev);
			if (rc)
				goto err_close_fd;

			memcpy(&list->info[list->count], &new_dev->dev.info,
			       sizeof(struct cxil_devinfo));
			list->count++;

err_close_fd:
			close(new_dev->fd);
err_free_dev_name:
			free(dev_name);
err_free_dev:
			free(new_dev);
		}
	}

	*dev_list = list;

	rc = 0;

free_glob:
	globfree(&globbuf);

	return rc;
}

CXIL_API void cxil_free_device_list(struct cxil_device_list *dev_list)
{
	free(dev_list);
}

CXIL_API int cxil_get_amo_remap_to_pcie_fadd(struct cxil_dev *dev,
					     int *amo_remap_to_pcie_fadd)
{
	int amo_remap;
	int rc;
	char fname[PATH_MAX];

	if (!dev || !amo_remap_to_pcie_fadd)
		return -EINVAL;

	rc = snprintf(fname, PATH_MAX,
		      CXIL_DEVINFO_FNAME("properties/amo_remap_to_pcie_fadd"),
		      dev->info.dev_id);
	if (rc < 0)
		return -ENOMEM;

	rc = read_sysfs_var(fname, "%d", &amo_remap);
	if (rc)
		return rc;

	*amo_remap_to_pcie_fadd = amo_remap;
	return 0;
}
