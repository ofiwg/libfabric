// SPDX-License-Identifier: GPL-2.0
/* Copyright 2022 Hewlett Packard Enterprise Development LP */

#include "cass_core.h"

#define PROP_ATTR_RO(_name) \
	static struct kobj_attribute dev_attr_##_name = __ATTR_RO(_name)

#define PROP_ATTR_RW(_name) \
	static struct kobj_attribute dev_attr_##_name = __ATTR_RW(_name)

static ssize_t nic_addr_show(struct kobject *kobj, struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "0x%x\n", hw->cdev.prop.nic_addr);
}
PROP_ATTR_RO(nic_addr);

static ssize_t nid_show(struct kobject *kobj, struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "0x%x\n", hw->cdev.prop.nid);
}

static ssize_t nid_store(struct kobject *kobj, struct kobj_attribute *attr,
			 const char *buf, size_t count)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   properties_kobj);
	u32 nid;

	if (kstrtoint(buf, 0, &nid) < 0)
		return -EINVAL;

	cxi_set_nid(&hw->cdev, nid);

	return count;
}
PROP_ATTR_RW(nid);

static ssize_t pid_bits_show(struct kobject *kobj, struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.pid_bits);
}
PROP_ATTR_RO(pid_bits);

static ssize_t system_type_identifier_show(struct kobject *kobj, struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n", hw->cdev.system_type_identifier);
}
PROP_ATTR_RO(system_type_identifier);

static ssize_t pid_count_show(struct kobject *kobj,
			      struct kobj_attribute *kattr,
			      char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.pid_count);
}
PROP_ATTR_RO(pid_count);

static ssize_t pid_granule_show(struct kobject *kobj,
				struct kobj_attribute *kattr,
				char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.pid_granule);
}
PROP_ATTR_RO(pid_granule);

static ssize_t min_free_shift_show(struct kobject *kobj,
				   struct kobj_attribute *kattr,
				   char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.min_free_shift);
}
PROP_ATTR_RO(min_free_shift);

static ssize_t rdzv_get_idx_show(struct kobject *kobj,
				 struct kobj_attribute *kattr,
				 char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rdzv_get_idx);
}
PROP_ATTR_RO(rdzv_get_idx);

static ssize_t num_ptes_show(struct kobject *kobj,
			     struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.ptes.max);
}
PROP_ATTR_RO(num_ptes);

static ssize_t num_txqs_show(struct kobject *kobj,
			     struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.txqs.max);
}
PROP_ATTR_RO(num_txqs);

static ssize_t num_tgqs_show(struct kobject *kobj,
			     struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.tgqs.max);
}
PROP_ATTR_RO(num_tgqs);

static ssize_t num_eqs_show(struct kobject *kobj,
			    struct kobj_attribute *kattr,
			    char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.eqs.max);
}
PROP_ATTR_RO(num_eqs);

static ssize_t num_cts_show(struct kobject *kobj,
			    struct kobj_attribute *kattr,
			    char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.cts.max);
}
PROP_ATTR_RO(num_cts);

static ssize_t num_acs_show(struct kobject *kobj,
			    struct kobj_attribute *kattr,
			    char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.acs.max);
}
PROP_ATTR_RO(num_acs);

static ssize_t num_tles_show(struct kobject *kobj,
			     struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.tles.max);
}
PROP_ATTR_RO(num_tles);

static ssize_t num_les_show(struct kobject *kobj,
			    struct kobj_attribute *kattr,
			    char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rsrcs.les.max);
}
PROP_ATTR_RO(num_les);

static ssize_t rgids_avail_show(struct kobject *kobj,
			      struct kobj_attribute *kattr, char *buf)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n",
			 C_NUM_RGIDS - refcount_read(&hw->rgids_refcount));
}
PROP_ATTR_RO(rgids_avail);

static ssize_t hpc_mtu_show(struct kobject *kobj, struct kobj_attribute *kattr,
			    char *buf)
{
	return scnprintf(buf, PAGE_SIZE, "%u\n", C_MAX_HPC_MTU);
}
PROP_ATTR_RO(hpc_mtu);

static ssize_t speed_show(struct kobject *kobj, struct kobj_attribute *kattr,
			  char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);
	int speed;

	if (hw->port)
		cass_sbl_link_mode_to_speed(hw->port->lattr.bl.link_mode, &speed);
	else
		speed = 0;

	return scnprintf(buf, PAGE_SIZE, "%d\n", speed);
}
PROP_ATTR_RO(speed);

static ssize_t link_show(struct kobject *kobj, struct kobj_attribute *kattr,
			 char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);
	int state = hw->phy.state == CASS_PHY_RUNNING;

	return scnprintf(buf, PAGE_SIZE, "%d\n", state);
}
PROP_ATTR_RO(link);

static ssize_t rev_show(struct kobject *kobj, struct kobj_attribute *kattr,
			char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n", hw->cdev.prop.device_rev);
}
PROP_ATTR_RO(rev);

static ssize_t proto_show(struct kobject *kobj, struct kobj_attribute *kattr,
			  char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n", hw->cdev.prop.device_proto);
}
PROP_ATTR_RO(proto);

static ssize_t platform_show(struct kobject *kobj, struct kobj_attribute *kattr,
			     char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n",
			 hw->cdev.prop.device_platform);
}
PROP_ATTR_RO(platform);

static ssize_t cassini_version_show(struct kobject *kobj,
				    struct kobj_attribute *kattr, char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);
	const char *str;

	switch (hw->cdev.prop.cassini_version) {
	case CASSINI_1_0:
		str = "1.0";
		break;
	case CASSINI_1_1:
		str = "1.1";
		break;
	case CASSINI_2_0:
		str = "2.0";
		break;
	default:
		str = "unknown";
		break;
	}

	return scnprintf(buf, PAGE_SIZE, "%s\n", str);
}
PROP_ATTR_RO(cassini_version);

static ssize_t uc_nic_show(struct kobject *kobj, struct kobj_attribute *kattr,
			   char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n", hw->uc_nic);
}
PROP_ATTR_RO(uc_nic);

static ssize_t pct_eq_show(struct kobject *kobj, struct kobj_attribute *kattr,
			   char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%u\n", hw->pct_eq_n);
}
PROP_ATTR_RO(pct_eq);

static ssize_t driver_rev_show(struct kobject *kobj,
			       struct kobj_attribute *kattr, char *buf)
{
	return scnprintf(buf, PAGE_SIZE, "%s\n", CXI_COMMIT);
}
PROP_ATTR_RO(driver_rev);

/* This function is similar to the kernel's current_link_speed_show() and
 * ideally would be merged into it
 */
static ssize_t current_esm_link_speed_show(struct kobject *kobj,
					   struct kobj_attribute *kattr,
					   char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);
	const char *speed;

	if (!hw->esm_active)
		return scnprintf(buf, PAGE_SIZE, "Disabled\n");

	switch (hw->pcie_link_speed) {
	case CASS_SPEED_2_5GT:
		speed = "2.5 GT/s";
		break;
	case CASS_SPEED_5_0GT:
		speed = "5.0 GT/s";
		break;
	case CASS_SPEED_8_0GT:
		speed = "8.0 GT/s";
		break;
	case CASS_SPEED_16_0GT:
		speed = "16.0 GT/s";
		break;
	case CASS_SPEED_20_0GT:
		speed = "20.0 GT/s";
		break;
	case CASS_SPEED_25_0GT:
		speed = "25.0 GT/s";
		break;
	case CASS_SPEED_32_0GT:
		speed = "32.0 GT/s";
		break;
	default:
		speed = "Unknown speed";
		break;
	}

	return scnprintf(buf, PAGE_SIZE, "%s\n", speed);
}
PROP_ATTR_RO(current_esm_link_speed);

static ssize_t rdzv_get_en_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   properties_kobj);

	return snprintf(buf, PAGE_SIZE, "%d\n", hw->cdev.prop.rdzv_get_en);
}

static ssize_t rdzv_get_en_store(struct kobject *kobj,
				 struct kobj_attribute *attr,
				 const char *buf, size_t count)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   properties_kobj);

	mutex_lock(&hw->get_ctrl_mutex);
	if (kstrtobool(buf, &hw->cdev.prop.rdzv_get_en) < 0) {
		mutex_unlock(&hw->get_ctrl_mutex);

		return -EINVAL;
	}
	cass_pte_set_get_ctrl(hw);
	mutex_unlock(&hw->get_ctrl_mutex);

	return count;
}
PROP_ATTR_RW(rdzv_get_en);

static ssize_t amo_remap_to_pcie_fadd_show(struct kobject *kobj,
					   struct kobj_attribute *attr,
					   char *buf)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   properties_kobj);

	return snprintf(buf, PAGE_SIZE, "%d\n",
			hw->cdev.prop.amo_remap_to_pcie_fadd);
}

static ssize_t amo_remap_to_pcie_fadd_store(struct kobject *kobj,
					    struct kobj_attribute *attr,
					    const char *buf, size_t count)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   properties_kobj);
	int amo_remap_to_pcie_fadd;
	int ret;

	ret = kstrtoint(buf, 10, &amo_remap_to_pcie_fadd);
	if (ret)
		return ret;

	mutex_lock(&hw->amo_remap_to_pcie_fadd_mutex);
	ret = cass_ixe_set_amo_remap_to_pcie_fadd(hw, amo_remap_to_pcie_fadd);
	mutex_unlock(&hw->amo_remap_to_pcie_fadd_mutex);

	if (ret)
		return ret;

	return count;
}
PROP_ATTR_RW(amo_remap_to_pcie_fadd);

static ssize_t pcie_corr_err_show(struct kobject *kobj,
				  struct kobj_attribute *kattr, char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%llu\n", hw->pcie_mon.corr_err);
}
PROP_ATTR_RO(pcie_corr_err);

static ssize_t pcie_uncorr_err_show(struct kobject *kobj,
				  struct kobj_attribute *kattr, char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, properties_kobj);

	return scnprintf(buf, PAGE_SIZE, "%llu\n", hw->pcie_mon.uncorr_err);
}
PROP_ATTR_RO(pcie_uncorr_err);

static struct attribute *properties_attrs[] = {
	&dev_attr_nic_addr.attr,
	&dev_attr_nid.attr,
	&dev_attr_pid_bits.attr,
	&dev_attr_pid_count.attr,
	&dev_attr_pid_granule.attr,
	&dev_attr_min_free_shift.attr,
	&dev_attr_rdzv_get_idx.attr,
	&dev_attr_hpc_mtu.attr,
	&dev_attr_speed.attr,
	&dev_attr_link.attr,
	&dev_attr_rev.attr,
	&dev_attr_proto.attr,
	&dev_attr_platform.attr,
	&dev_attr_system_type_identifier.attr,
	&dev_attr_cassini_version.attr,
	&dev_attr_uc_nic.attr,
	&dev_attr_num_ptes.attr,
	&dev_attr_num_txqs.attr,
	&dev_attr_num_tgqs.attr,
	&dev_attr_num_eqs.attr,
	&dev_attr_num_cts.attr,
	&dev_attr_num_acs.attr,
	&dev_attr_num_tles.attr,
	&dev_attr_num_les.attr,
	&dev_attr_rgids_avail.attr,
	&dev_attr_pct_eq.attr,
	&dev_attr_driver_rev.attr,
	&dev_attr_current_esm_link_speed.attr,
	&dev_attr_rdzv_get_en.attr,
	&dev_attr_pcie_corr_err.attr,
	&dev_attr_pcie_uncorr_err.attr,
	&dev_attr_amo_remap_to_pcie_fadd.attr,
	NULL,
};
ATTRIBUTE_GROUPS(properties);

static struct kobj_type properties_info = {
	.sysfs_ops      = &kobj_sysfs_ops,
	.default_groups = properties_groups,
};

/* FRU attributes from the uC. The FRU information is made available
 * in hw->fru_info[]. Create each entry with the index in that table.
 */
#define FRU_ATTR(_name, _idx)						\
	static ssize_t _name##_show(struct kobject *kobj,		\
				    struct kobj_attribute *kattr,	\
				    char *buf)				\
	{								\
		struct cass_dev *hw =					\
			container_of(kobj, struct cass_dev, fru_kobj);	\
		if (!hw->fru_info_valid)				\
			uc_cmd_get_fru(hw);				\
		if (hw->fru_info[_idx])					\
			return scnprintf(buf, PAGE_SIZE, "%s\n",	\
					 hw->fru_info[_idx]);		\
		else							\
			return 0;					\
	}								\
	static struct kobj_attribute dev_attr_##_name = __ATTR_RO(_name)

FRU_ATTR(chassis_type, PLDM_FRU_FIELD_CHASSIS_TYPE);
FRU_ATTR(model, PLDM_FRU_FIELD_MODEL);
FRU_ATTR(part_number, PLDM_FRU_FIELD_PART_NUMBER);
FRU_ATTR(serial_number, PLDM_FRU_FIELD_SERIAL_NUMBER);
FRU_ATTR(manufacturer, PLDM_FRU_FIELD_MANUFACTURER);
FRU_ATTR(manufacture_date, PLDM_FRU_FIELD_MANUFACTURE_DATE);
FRU_ATTR(vendor, PLDM_FRU_FIELD_VENDOR);
FRU_ATTR(name, PLDM_FRU_FIELD_NAME);
FRU_ATTR(sku, PLDM_FRU_FIELD_SKU);
FRU_ATTR(version, PLDM_FRU_FIELD_VERSION);
FRU_ATTR(asset_tag, PLDM_FRU_FIELD_ASSET_TAG);
FRU_ATTR(description, PLDM_FRU_FIELD_DESCRIPTION);
FRU_ATTR(engineering_change_level,
	 PLDM_FRU_FIELD_ENGINEERING_CHANGE_LEVEL);
FRU_ATTR(other, PLDM_FRU_FIELD_OTHER);
FRU_ATTR(vendor_iana, PLDM_FRU_FIELD_VENDOR_IANA);

static struct attribute *fru_attrs[] = {
	&dev_attr_chassis_type.attr,
	&dev_attr_model.attr,
	&dev_attr_part_number.attr,
	&dev_attr_serial_number.attr,
	&dev_attr_manufacturer.attr,
	&dev_attr_manufacture_date.attr,
	&dev_attr_vendor.attr,
	&dev_attr_name.attr,
	&dev_attr_sku.attr,
	&dev_attr_version.attr,
	&dev_attr_asset_tag.attr,
	&dev_attr_description.attr,
	&dev_attr_engineering_change_level.attr,
	&dev_attr_other.attr,
	&dev_attr_vendor_iana.attr,

	NULL,
};
ATTRIBUTE_GROUPS(fru);

static struct kobj_type fru_info = {
	.sysfs_ops      = &kobj_sysfs_ops,
	.default_groups = fru_groups,
};

#define LINK_RESTART_TIME(_idx)					\
	static ssize_t time_##_idx##_show(struct kobject *kobj,	\
		struct kobj_attribute *kattr, char *buf)	\
	{							\
		struct cass_dev *hw =				\
			container_of(kobj, struct cass_dev,	\
			link_restarts_kobj);			\
								\
		return scnprintf(buf, PAGE_SIZE, "%lld\n",	\
			hw->port->link_restart_time_buf[_idx]);	\
	}							\
	PROP_ATTR_RO(time_##_idx)

LINK_RESTART_TIME(0);
LINK_RESTART_TIME(1);
LINK_RESTART_TIME(2);
LINK_RESTART_TIME(3);
LINK_RESTART_TIME(4);
LINK_RESTART_TIME(5);
LINK_RESTART_TIME(6);
LINK_RESTART_TIME(7);
LINK_RESTART_TIME(8);
LINK_RESTART_TIME(9);
/* NOTE: don't forget to add more buffer entries */

static struct attribute *link_restarts_attrs[] = {
	&dev_attr_time_0.attr,
	&dev_attr_time_1.attr,
	&dev_attr_time_2.attr,
	&dev_attr_time_3.attr,
	&dev_attr_time_4.attr,
	&dev_attr_time_5.attr,
	&dev_attr_time_6.attr,
	&dev_attr_time_7.attr,
	&dev_attr_time_8.attr,
	&dev_attr_time_9.attr,
	NULL,
};
ATTRIBUTE_GROUPS(link_restarts);

static struct kobj_type link_restarts_info = {
	.sysfs_ops      = &kobj_sysfs_ops,
	.default_groups = link_restarts_groups,
};

#define SENSOR_ATTR_RO(_name, _fmt, _field)				\
	static ssize_t sensor_##_name##_show(struct kobject *kobj,	\
					     struct kobj_attribute *kattr, \
					     char *buf)			\
	{								\
		struct pldm_sensor *sensor =				\
			container_of(kobj, struct pldm_sensor, kobj);	\
									\
		return scnprintf(buf, PAGE_SIZE, _fmt "\n",		\
				 sensor->_field);			\
	}								\
									\
	static struct kobj_attribute dev_attr_sensor_##_name = {	\
		.attr   = { .name = __stringify(_name), .mode = 0444 },	\
		.show   = sensor_##_name##_show,			\
	}

SENSOR_ATTR_RO(name, "%s", name);
SENSOR_ATTR_RO(base_unit, "%u", num.base_unit);
SENSOR_ATTR_RO(unit_modifier, "%d", num.unit_modifier);
SENSOR_ATTR_RO(rate_unit, "%u", num.rate_unit);

#define SENSOR_SSD_ATTR_RO(_name, _field)				\
	static ssize_t sensor_##_name##_show(struct kobject *kobj,	\
					     struct kobj_attribute *kattr, \
					     char *buf)			\
	{								\
		struct pldm_sensor *sensor =				\
			container_of(kobj, struct pldm_sensor, kobj);	\
									\
		return scnprintf(buf, PAGE_SIZE, "%llu\n",		\
				 get_pldm_value(sensor, _field));	\
	}								\
									\
	static struct kobj_attribute dev_attr_sensor_##_name = {	\
		.attr   = { .name = __stringify(_name), .mode = 0444 },	\
		.show   = sensor_##_name##_show,			\
	}

SENSOR_SSD_ATTR_RO(range_field_supported, range_field_support);
SENSOR_SSD_ATTR_RO(nominal_value, nominal_value);
SENSOR_SSD_ATTR_RO(warning_high, warning_high);
SENSOR_SSD_ATTR_RO(warning_low, warning_low);
SENSOR_SSD_ATTR_RO(critical_high, critical_high);
SENSOR_SSD_ATTR_RO(critical_low, critical_low);
SENSOR_SSD_ATTR_RO(fatal_high, fatal_high);
SENSOR_SSD_ATTR_RO(fatal_low, fatal_low);

#define SENSOR_VAL_ATTR_RO(_name, _fmt)					\
	static ssize_t sensor_##_name##_show(struct kobject *kobj,	\
					     struct kobj_attribute *kattr, \
					     char *buf)			\
	{								\
		struct pldm_sensor *sensor =				\
			container_of(kobj, struct pldm_sensor, kobj);	\
		struct pldm_sensor_reading result;			\
		int rc;							\
									\
		rc = update_sensor(sensor, &result);			\
		if (rc)							\
			return rc;					\
									\
		return scnprintf(buf, PAGE_SIZE, _fmt "\n", result._name); \
	}								\
									\
	static struct kobj_attribute dev_attr_sensor_##_name = {	\
		.attr   = { .name = __stringify(_name), .mode = 0400 },	\
		.show   = sensor_##_name##_show,			\
	}

SENSOR_VAL_ATTR_RO(operational_state, "%u");
SENSOR_VAL_ATTR_RO(present_state, "%u");
SENSOR_VAL_ATTR_RO(previous_state, "%u");
SENSOR_VAL_ATTR_RO(present_reading, "%lld");

static struct attribute *sensor_attrs[] = {
	&dev_attr_sensor_name.attr,
	&dev_attr_sensor_base_unit.attr,
	&dev_attr_sensor_unit_modifier.attr,
	&dev_attr_sensor_rate_unit.attr,
	&dev_attr_sensor_nominal_value.attr,
	&dev_attr_sensor_range_field_supported.attr,
	&dev_attr_sensor_warning_high.attr,
	&dev_attr_sensor_warning_low.attr,
	&dev_attr_sensor_critical_high.attr,
	&dev_attr_sensor_critical_low.attr,
	&dev_attr_sensor_fatal_high.attr,
	&dev_attr_sensor_fatal_low.attr,
	&dev_attr_sensor_operational_state.attr,
	&dev_attr_sensor_present_state.attr,
	&dev_attr_sensor_previous_state.attr,
	&dev_attr_sensor_present_reading.attr,
	NULL,
};
ATTRIBUTE_GROUPS(sensor);

static struct kobj_type sensor_settings = {
	.sysfs_ops      = &kobj_sysfs_ops,
	.default_groups = sensor_groups,
};

static int create_sensors(struct cass_dev *hw)
{
	struct pldm_sensor *sensor;
	int id;
	int bad_id;
	int rc;

	hw->pldm_sensors_kobj =
		kobject_create_and_add("sensors", &hw->cdev.pdev->dev.kobj);
	if (!hw->pldm_sensors_kobj)
		return -ENOMEM;

	idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
		rc = kobject_init_and_add(&sensor->kobj, &sensor_settings,
					  hw->pldm_sensors_kobj,
					  "%u", id);
		if (rc)
			goto err;
	}

	return 0;

err:
	bad_id = id;
	idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
		kobject_put(&sensor->kobj);
		if (id == bad_id)
			break;
	}

	kobject_put(hw->pldm_sensors_kobj);

	return rc;
}

static void delete_sensors(struct cass_dev *hw)
{
	struct pldm_sensor *sensor;
	int id;

	idr_for_each_entry(&hw->pldm_sensors, sensor, id)
		kobject_put(&sensor->kobj);

	kobject_put(hw->pldm_sensors_kobj);
}

static ssize_t ac_filter_disabled_show(struct device *dev,
				      struct device_attribute *attr, char *buf)
{
	struct cass_dev *hw = dev_get_drvdata(dev);

	return snprintf(buf, PAGE_SIZE, "%d\n", hw->ac_filter_disabled);
}

static ssize_t ac_filter_disabled_store(struct device *dev,
				       struct device_attribute *attr,
				       const char *buf,
				       size_t count)
{
	struct cass_dev *hw = dev_get_drvdata(dev);

	if (kstrtobool(buf, &hw->ac_filter_disabled) < 0)
		return -EINVAL;

	return count;
}

static DEVICE_ATTR_RW(ac_filter_disabled);

static struct attribute *cxi_dev_settings[] = {
	&dev_attr_ac_filter_disabled.attr,
	NULL,
};

static const struct attribute_group cxi_dev_settings_group = {
	.name = "settings",
	.attrs = cxi_dev_settings,
};

int create_sysfs_properties(struct cass_dev *hw)
{
	int rc;

	if (!hw->cdev.is_physfn)
		return 0;

	rc = kobject_init_and_add(&hw->properties_kobj, &properties_info,
				  &hw->cdev.pdev->dev.kobj,
				  "properties");
	if (rc)
		goto put_properties;

	rc = kobject_init_and_add(&hw->link_restarts_kobj, &link_restarts_info,
				  &hw->cdev.pdev->dev.kobj,
				  "link_restarts");
	if (rc)
		goto put_link_restarts;

	if (!hw->cdev.is_physfn)
		return 0;

	rc = kobject_init_and_add(&hw->fru_kobj, &fru_info,
				  &hw->cdev.pdev->dev.kobj,
				  "fru");
	if (rc)
		goto put_fru;

	rc = sysfs_create_group(&hw->cdev.pdev->dev.kobj,
				&cxi_dev_settings_group);
	if (rc)
		goto put_fru;

	rc = cass_create_tc_sysfs(hw);
	if (rc)
		goto remove_sysfs_group;

	rc = create_sensors(hw);
	if (rc)
		goto remove_tc;

	return 0;

remove_tc:
	cass_destroy_tc_sysfs(hw);
remove_sysfs_group:
	sysfs_remove_group(&hw->cdev.pdev->dev.kobj, &cxi_dev_settings_group);
put_fru:
	kobject_put(&hw->fru_kobj);
put_link_restarts:
	kobject_put(&hw->link_restarts_kobj);
put_properties:
	kobject_put(&hw->properties_kobj);

	return rc;
}

void destroy_sysfs_properties(struct cass_dev *hw)
{
	if (!hw->cdev.is_physfn)
		return;

	delete_sensors(hw);
	cass_destroy_tc_sysfs(hw);
	sysfs_remove_group(&hw->cdev.pdev->dev.kobj, &cxi_dev_settings_group);
	kobject_put(&hw->fru_kobj);
	kobject_put(&hw->link_restarts_kobj);
	kobject_put(&hw->properties_kobj);
}
