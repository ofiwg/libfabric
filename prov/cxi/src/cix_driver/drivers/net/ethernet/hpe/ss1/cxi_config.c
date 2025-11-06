/* SPDX-License-Identifier: GPL-2.0 */

/* Copyright 2025 Hewlett Packard Enterprise Development LP */

/*
 * Configuration and Management of rgroup/rx profile/tx profile using
 * configfs Filesystem Interface
 */

#include "cxi_config.h"

static enum cxi_resource_type cxi_resource_str_to_type(const char *name)
{
	int i;

	for (i = CXI_RESOURCE_PTLTE; i < CXI_RESOURCE_MAX; i++) {
		if (strcmp(name, cxi_resource_type_strs[i]) == 0)
			return i;
	}

	return CXI_RESOURCE_MAX;
}

static struct config_group *get_parent_config_group(struct config_group *child_group)
{
	struct config_group *parent_config_group;

	if (!child_group->cg_item.ci_parent)
		return NULL;

	parent_config_group = container_of(child_group->cg_item.ci_parent,
					   struct config_group,
					   cg_item);

	return parent_config_group;
}

static const int resource_max[] = {
	C_NUM_PTLTES,
	C_NUM_TRANSMIT_CQS,
	C_NUM_TARGET_CQS,
	EQS_AVAIL,
	CTS_AVAIL,
	PE_LE_MAX,
	PE_LE_MAX,
	PE_LE_MAX,
	PE_LE_MAX,
	0,
	ACS_AVAIL,
	CXI_RESOURCE_MAX
};

static struct cxi_cfg_rgroup_ac_entry_sub *to_cxi_cfg_rgroup_ac_entry_sub(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rgroup_ac_entry_sub, group);
}

static ssize_t cxi_cfg_rgroup_ac_entry_sub_type_show(struct config_item *item,
						     char *page)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;
	int rc;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;

	ac_entry_sub = to_cxi_cfg_rgroup_ac_entry_sub(item);

	rc = cxi_rgroup_get_ac_entry_data(ac_entry_sub->rgroup,
					  ac_entry_sub->id,
					  &ac_type,
					  &ac_data);
	if (rc)
		return rc;

	return sprintf(page, "%d\n", ac_type);
}

static ssize_t cxi_cfg_rgroup_ac_entry_sub_type_store(struct config_item *item,
						      const char *page,
						      size_t count)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;
	int rc;
	enum cxi_ac_type ac_type_val;
	union cxi_ac_data ac_data_t;
	unsigned int ac_entry_id;

	ac_entry_sub = to_cxi_cfg_rgroup_ac_entry_sub(item);
	if (!ac_entry_sub->rgroup) {
		pr_debug("Unable to get rgroup reference\n");
		return -EINVAL;
	}

	rc = kstrtouint(page, 0, (unsigned int *)&ac_type_val);
	if (rc)
		return rc;

	if (ac_type_val == CXI_AC_UID)
		ac_data_t.uid = ac_entry_sub->ac_data.uid;
	else if (ac_type_val == CXI_AC_GID)
		ac_data_t.gid = ac_entry_sub->ac_data.gid;
	else if (ac_type_val == CXI_AC_OPEN)
		pr_debug("Setting type to CXI_AC_OPEN\n");
	else
		return -EINVAL;

	rc = cxi_rgroup_add_ac_entry(ac_entry_sub->rgroup,
				     ac_type_val,
				     &ac_data_t,
				     &ac_entry_id);
	if (rc < 0) {
		pr_debug("Unable to add ac entry to resource group: err val %d\n", rc);
		return rc;
	}

	ac_entry_sub->ac_type = ac_type_val;
	ac_entry_sub->id = ac_entry_id;

	return count;
}

static ssize_t cxi_cfg_rgroup_ac_entry_sub_data_show(struct config_item *item,
						     char *page)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	int rc;

	ac_entry_sub = to_cxi_cfg_rgroup_ac_entry_sub(item);

	rc = cxi_rgroup_get_ac_entry_data(ac_entry_sub->rgroup,
					  ac_entry_sub->id,
					  &ac_type,
					  &ac_data);
	if (rc)
		return rc;

	if (ac_type == CXI_AC_UID)
		return sprintf(page, "%d\n", ac_data.uid);
	else if (ac_type == CXI_AC_GID)
		return sprintf(page, "%d\n", ac_data.gid);
	else if (ac_type == CXI_AC_OPEN)
		return sprintf(page, "%d\n", CXI_AC_OPEN);

	return -EINVAL;
}

static ssize_t cxi_cfg_rgroup_ac_entry_sub_data_store(struct config_item *item,
						      const char *page,
						      size_t count)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;
	int rc;
	unsigned int ac_data_val;

	ac_entry_sub = to_cxi_cfg_rgroup_ac_entry_sub(item);

	rc = kstrtouint(page, 0, &ac_data_val);
	if (rc)
		return rc;

	if (ac_entry_sub->ac_type == CXI_AC_UID) {
		ac_entry_sub->ac_data.uid = ac_data_val;
		return count;
	} else if (ac_entry_sub->ac_type == CXI_AC_GID) {
		ac_entry_sub->ac_data.gid = ac_data_val;
		return count;
	}

	return -EINVAL;
}

static void cxi_cfg_rgroup_ac_entry_sub_free(struct config_item *item)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;

	ac_entry_sub = container_of(item, struct cxi_cfg_rgroup_ac_entry_sub,
				    group.cg_item);
	kfree(ac_entry_sub);
}

CONFIGFS_ATTR(cxi_cfg_rgroup_ac_entry_sub_, type);
CONFIGFS_ATTR(cxi_cfg_rgroup_ac_entry_sub_, data);

static struct configfs_attribute *cxi_cfg_rgroup_ac_entry_sub_attrs[] = {
	&cxi_cfg_rgroup_ac_entry_sub_attr_type,
	&cxi_cfg_rgroup_ac_entry_sub_attr_data,
	NULL,
};

static struct configfs_item_operations cxi_cfg_rgroup_ac_entry_sub_item_ops = {
	.release     = cxi_cfg_rgroup_ac_entry_sub_free,
};

static const struct config_item_type cxi_cfg_rgroup_ac_entry_sub_type = {
	.ct_attrs       = cxi_cfg_rgroup_ac_entry_sub_attrs,
	.ct_item_ops    = &cxi_cfg_rgroup_ac_entry_sub_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup_ac_entry_sub *alloc_dev_rgroup_ac_entry_sub(const char *name)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_dir_sub;

	ac_entry_dir_sub = kzalloc(sizeof(*ac_entry_dir_sub),
				   GFP_KERNEL);
	if (!ac_entry_dir_sub)
		return NULL;

	strscpy(ac_entry_dir_sub->name, name, sizeof(ac_entry_dir_sub->name));

	config_group_init_type_name(&ac_entry_dir_sub->group,
				    ac_entry_dir_sub->name,
				    &cxi_cfg_rgroup_ac_entry_sub_type);

	return ac_entry_dir_sub;
}

static struct
config_group *cxi_cfg_rgroup_ac_entry_make_group(struct config_group *group,
						 const char *name)
{
	struct cxi_cfg_rgroup_ac_entry *ac_entry;
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;

	if (strlen(name) > CONFIGFS_DIR_LEN - 1)
		return ERR_PTR(-EINVAL);

	ac_entry = container_of(group, struct cxi_cfg_rgroup_ac_entry, group);

	ac_entry_sub = alloc_dev_rgroup_ac_entry_sub(name);
	if (!ac_entry_sub) {
		pr_debug("Failed to alloc memory for configfs ac entry");
		return ERR_PTR(-ENOMEM);
	}

	ac_entry_sub->rgroup = ac_entry->rgroup;
	ac_entry_sub->ac_type = CXI_AC_UID;
	ac_entry_sub->ac_data.uid = 0;

	return &ac_entry_sub->group;
}

static void
cxi_cfg_rgroup_ac_entry_drop_item(struct config_group *group,
				  struct config_item *item)
{
	struct cxi_cfg_rgroup_ac_entry_sub *ac_entry_sub;
	unsigned int ac_entry_id;
	int rc;

	ac_entry_sub = container_of(item, struct cxi_cfg_rgroup_ac_entry_sub, group.cg_item);

	if (ac_entry_sub->ac_type == CXI_AC_UID) {
		rc = cxi_rgroup_get_ac_entry_id_by_data(ac_entry_sub->rgroup,
							CXI_AC_UID,
							&ac_entry_sub->ac_data,
							&ac_entry_id);

		pr_debug("Getting an entry info rc = %d\n", rc);
	} else if (ac_entry_sub->ac_type == CXI_AC_GID) {
		rc = cxi_rgroup_get_ac_entry_id_by_data(ac_entry_sub->rgroup,
							CXI_AC_GID,
							&ac_entry_sub->ac_data,
							&ac_entry_id);

		pr_debug("Getting an entry info rc = %d\n", rc);
	} else {
		rc = cxi_rgroup_get_ac_entry_id_by_data(ac_entry_sub->rgroup,
							CXI_AC_OPEN,
							NULL,
							&ac_entry_id);
		pr_debug("Getting an entry info rc = %d\n", rc);
	}

	cxi_rgroup_delete_ac_entry(ac_entry_sub->rgroup,
				   ac_entry_id);
	config_item_put(item);
}

static void cxi_cfg_rgroup_ac_entry_free(struct config_item *item)
{
	struct cxi_cfg_rgroup_ac_entry *ac_entry;

	ac_entry = container_of(item, struct cxi_cfg_rgroup_ac_entry,
				group.cg_item);

	kfree(ac_entry);
}

static struct configfs_group_operations cxi_cfg_rgroup_ac_entry_ops = {
	.make_group     = cxi_cfg_rgroup_ac_entry_make_group,
	.drop_item      = cxi_cfg_rgroup_ac_entry_drop_item
};

static struct configfs_item_operations cxi_cfg_rgroup_ac_entry_item_ops = {
	.release     = cxi_cfg_rgroup_ac_entry_free,
};

static const struct config_item_type cxi_cfg_rgroup_ac_entry_type = {
	.ct_group_ops   = &cxi_cfg_rgroup_ac_entry_ops,
	.ct_item_ops    = &cxi_cfg_rgroup_ac_entry_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup_res_type *to_cxi_cfg_rgroup_res_type(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rgroup_res_type, group);
}

static ssize_t cxi_cfg_rgroup_res_type_max_show(struct config_item *item,
						char *page)
{
	struct cxi_cfg_rgroup_res_type *rgroup_res_type;
	struct cxi_resource_entry *entry;
	int rc;

	rgroup_res_type = to_cxi_cfg_rgroup_res_type(item);
	rc = cxi_rgroup_get_resource_entry(rgroup_res_type->rgroup,
					   cxi_resource_str_to_type(rgroup_res_type->name),
					   &entry);
	if (rc != 0) {
		pr_debug("Error: Resource entry not yet created for the resource type\n");
		pr_debug("Error: Set the max file first and then the reserved file to create the resource type\n");
		return -EINVAL;
	}

	return sprintf(page, "%ld\n", entry->limits.max);
}

static ssize_t cxi_cfg_rgroup_res_type_max_store(struct config_item *item,
						 const char *page,
						 size_t count)
{
	struct cxi_cfg_rgroup_res_type *rgroup_res_type;
	struct cxi_resource_entry *entry;
	int rc;

	rgroup_res_type = to_cxi_cfg_rgroup_res_type(item);

	rc = cxi_rgroup_get_resource_entry(rgroup_res_type->rgroup,
					   cxi_resource_str_to_type(rgroup_res_type->name),
					   &entry);
	if (rc == 0) {
		pr_debug("Error: Given resource type has already been added\n");
		return -EINVAL;
	}

	rc = kstrtoint(page, 0, &rgroup_res_type->max);
	if (rc)
		return rc;

	pr_debug("resource max value set to: %d\n", rgroup_res_type->max);

	return count;
}

static ssize_t cxi_cfg_rgroup_res_type_reserved_show(struct config_item *item,
						     char *page)
{
	int rc;
	struct cxi_cfg_rgroup_res_type *rgroup_res_type;
	struct cxi_resource_entry *entry;

	rgroup_res_type = to_cxi_cfg_rgroup_res_type(item);

	rc = cxi_rgroup_get_resource_entry(rgroup_res_type->rgroup,
					   cxi_resource_str_to_type(rgroup_res_type->name),
					   &entry);
	if (rc != 0) {
		pr_debug("Error: Resource entry not found for the resource type\n");
		return -EINVAL;
	}

	return sprintf(page, "%ld\n", entry->limits.reserved);
}

static ssize_t cxi_cfg_rgroup_res_type_reserved_store(struct config_item *item,
						      const char *page,
						      size_t count)
{
	struct cxi_cfg_rgroup_res_type *rgroup_res_type;
	int rc;
	enum cxi_resource_type e_type;
	struct cxi_resource_limits limits = {};

	rgroup_res_type = to_cxi_cfg_rgroup_res_type(item);

	rc = kstrtoint(page, 0, &rgroup_res_type->reserved);
	if (rc)
		return rc;

	e_type = cxi_resource_str_to_type(rgroup_res_type->name);

	limits.max = rgroup_res_type->max;
	limits.reserved = rgroup_res_type->reserved;

	rc = cxi_rgroup_add_resource(rgroup_res_type->rgroup, e_type, &limits);
	if (rc)
		return rc;

	return count;
}

static ssize_t cxi_cfg_rgroup_res_type_in_use_show(struct config_item *item,
						   char *page)
{
	struct cxi_cfg_rgroup_res_type *rgroup_res_type;
	struct cxi_resource_entry *entry;
	int rc;

	rgroup_res_type = to_cxi_cfg_rgroup_res_type(item);

	rc = cxi_rgroup_get_resource_entry(rgroup_res_type->rgroup,
					   cxi_resource_str_to_type(rgroup_res_type->name),
					   &entry);
	if (rc != 0) {
		pr_debug("Error: Given resource type has not yet been added\n");
		return -EINVAL;
	}

	return sprintf(page, "%ld\n", entry->limits.in_use);
}

static void cxi_cfg_rgroup_res_type_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rgroup_res_type, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_rgroup_res_type_, reserved);
CONFIGFS_ATTR(cxi_cfg_rgroup_res_type_, max);
CONFIGFS_ATTR_RO(cxi_cfg_rgroup_res_type_, in_use);

static struct configfs_attribute *cxi_cfg_rgroup_res_type_attrs[] = {
	&cxi_cfg_rgroup_res_type_attr_reserved,
	&cxi_cfg_rgroup_res_type_attr_max,
	&cxi_cfg_rgroup_res_type_attr_in_use,
	NULL
};

static struct configfs_item_operations cxi_cfg_rgroup_res_type_item_ops = {
	.release     = cxi_cfg_rgroup_res_type_free,
};

static const struct config_item_type cxi_cfg_rgroup_res_type_type = {
	.ct_attrs       = cxi_cfg_rgroup_res_type_attrs,
	.ct_item_ops    = &cxi_cfg_rgroup_res_type_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup_res_type *alloc_dev_rgroup_res_type(const char *name)
{
	struct cxi_cfg_rgroup_res_type *dev_dir;

	dev_dir = kzalloc(sizeof(*dev_dir), GFP_KERNEL);
	if (!dev_dir)
		return NULL;

	strscpy(dev_dir->name, name, sizeof(dev_dir->name));

	config_group_init_type_name(&dev_dir->group,
				    dev_dir->name,
				    &cxi_cfg_rgroup_res_type_type);

	return dev_dir;
}

static struct
config_group *cxi_cfg_rgroup_res_make_group(struct config_group *group,
					    const char *name)
{
	struct cxi_cfg_rgroup_res_type *res_type;
	struct cxi_cfg_rgroup_res *res;
	enum cxi_resource_type e_type;

	e_type = cxi_resource_str_to_type(name);
	if (e_type == CXI_RESOURCE_MAX) {
		pr_debug("Unknown resource name %s\n", name);
		return ERR_PTR(-EINVAL);
	}

	res_type = alloc_dev_rgroup_res_type(name);
	if (!res_type) {
		pr_debug("Failed to alloc memory for configfs resource types");
		return ERR_PTR(-ENOMEM);
	}

	res = container_of(group, struct cxi_cfg_rgroup_res, group);
	if (!res) {
		pr_debug("No parent resource group found\n");
		kfree(res_type);
		return ERR_PTR(-EINVAL);
	}

	res_type->max = resource_max[e_type];
	res_type->reserved = 0;
	res_type->rgroup = res->rgroup;

	return &res_type->group;
}

static void
cxi_cfg_rgroup_res_drop_item(struct config_group *group,
			     struct config_item *item)
{
	struct config_group *parent_group;
	int rc;
	struct cxi_cfg_rgroup *rgp;
	struct cxi_cfg_rgroup_res_type *res_type;

	parent_group = get_parent_config_group(group);

	rgp = container_of(parent_group, struct cxi_cfg_rgroup, group);

	res_type = container_of(to_config_group(item),
				struct cxi_cfg_rgroup_res_type,
				group);

	/*
	 * For accidental deletion of a configfs resource type directory which is
	 * enabled. In the future a different approach will be taken to handle
	 * accidental deletion by user
	 */
	if (cxi_rgroup_is_enabled(rgp->rgroup))
		cxi_rgroup_disable(rgp->rgroup);

	rc = cxi_rgroup_delete_resource(res_type->rgroup,
					cxi_resource_str_to_type(item->ci_name));

	pr_debug("Resource group resource deletion rc = %d\n", rc);
	config_item_put(item);
}

static void cxi_cfg_rgroup_res_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rgroup_res, group.cg_item));
}

static struct configfs_group_operations cxi_cfg_rgroup_res_ops = {
	.make_group     = cxi_cfg_rgroup_res_make_group,
	.drop_item      = cxi_cfg_rgroup_res_drop_item,
};

static struct configfs_item_operations cxi_cfg_rgroup_res_item_ops = {
	.release     = cxi_cfg_rgroup_res_free,
};

static const struct config_item_type cxi_cfg_rgroup_res_type = {
	.ct_group_ops   = &cxi_cfg_rgroup_res_ops,
	.ct_item_ops    = &cxi_cfg_rgroup_res_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup_attr *to_cxi_cfg_rgroup_attr(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rgroup_attr, group);
}

static ssize_t cxi_cfg_rgroup_attr_name_show(struct config_item *item,
					     char *page)
{
	struct cxi_cfg_rgroup_attr *rgroup_attr;

	rgroup_attr = to_cxi_cfg_rgroup_attr(item);

	return sprintf(page, "%s\n", cxi_rgroup_name(rgroup_attr->rgroup));
}

static ssize_t cxi_cfg_rgroup_attr_cntr_pool_id_show(struct config_item *item,
						     char *page)
{
	struct cxi_cfg_rgroup_attr *rgroup_attr;

	rgroup_attr = to_cxi_cfg_rgroup_attr(item);

	return sprintf(page, "%d\n", rgroup_attr->c_p_id);
}

static ssize_t cxi_cfg_rgroup_attr_system_service_show(struct config_item *item,
						       char *page)
{
	struct cxi_cfg_rgroup_attr *rgroup_attr;

	rgroup_attr = to_cxi_cfg_rgroup_attr(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 cxi_rgroup_system_service(rgroup_attr->rgroup) ? "true" : "false");
}

static ssize_t cxi_cfg_rgroup_attr_lnis_per_rgid_show(struct config_item *item,
						      char *page)
{
	struct cxi_cfg_rgroup_attr *rgroup_attr;

	rgroup_attr = to_cxi_cfg_rgroup_attr(item);

	return sprintf(page, "%d\n", cxi_rgroup_lnis_per_rgid(rgroup_attr->rgroup));
}

static void cxi_cfg_rgroup_attr_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rgroup_attr, group.cg_item));
}

CONFIGFS_ATTR_RO(cxi_cfg_rgroup_attr_, name);
CONFIGFS_ATTR_RO(cxi_cfg_rgroup_attr_, cntr_pool_id);
CONFIGFS_ATTR_RO(cxi_cfg_rgroup_attr_, system_service);
CONFIGFS_ATTR_RO(cxi_cfg_rgroup_attr_, lnis_per_rgid);

static struct configfs_attribute *cxi_cfg_rgroup_attr_attrs[] = {
	&cxi_cfg_rgroup_attr_attr_name,
	&cxi_cfg_rgroup_attr_attr_cntr_pool_id,
	&cxi_cfg_rgroup_attr_attr_system_service,
	&cxi_cfg_rgroup_attr_attr_lnis_per_rgid,
	NULL,
};

static struct configfs_item_operations cxi_cfg_rgroup_attr_item_ops = {
	.release     = cxi_cfg_rgroup_attr_free,
};

static const struct config_item_type cxi_cfg_rgroup_attr_type = {
	.ct_attrs       = cxi_cfg_rgroup_attr_attrs,
	.ct_item_ops    = &cxi_cfg_rgroup_attr_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup_state *to_cxi_cfg_rgroup_state(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rgroup_state, group);
}

static ssize_t cxi_cfg_rgroup_state_enabled_show(struct config_item *item,
						 char *page)
{
	struct cxi_cfg_rgroup_state *rgroup_state;

	rgroup_state = to_cxi_cfg_rgroup_state(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 cxi_rgroup_is_enabled(rgroup_state->rgroup) ? "true" : "false");
}

static ssize_t cxi_cfg_rgroup_state_enabled_store(struct config_item *item,
						  const char *page,
						  size_t count)
{
	struct cxi_cfg_rgroup_state *rgroup_state;
	bool b_val;

	if (kstrtobool(page, &b_val))
		return -EINVAL;

	rgroup_state = to_cxi_cfg_rgroup_state(item);

	if (b_val) {
		cxi_rgroup_enable(rgroup_state->rgroup);
	} else {
		cxi_rgroup_disable(rgroup_state->rgroup);
	}

	return count;
}

static ssize_t cxi_cfg_rgroup_state_refcount_show(struct config_item *item,
						  char *page)
{
	struct cxi_cfg_rgroup_state *rgroup_state;

	rgroup_state = to_cxi_cfg_rgroup_state(item);

	return sprintf(page, "%d\n", cxi_rgroup_refcount(rgroup_state->rgroup));
}

static void cxi_cfg_rgroup_state_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rgroup_state, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_rgroup_state_, enabled);
CONFIGFS_ATTR_RO(cxi_cfg_rgroup_state_, refcount);

static struct configfs_attribute *cxi_cfg_rgroup_state_attrs[] = {
	&cxi_cfg_rgroup_state_attr_enabled,
	&cxi_cfg_rgroup_state_attr_refcount,
	NULL
};

static struct configfs_item_operations cxi_cfg_rgroup_state_item_ops = {
	.release     = cxi_cfg_rgroup_state_free,
};

static const struct config_item_type cxi_cfg_rgroup_state_type = {
	.ct_attrs       = cxi_cfg_rgroup_state_attrs,
	.ct_item_ops    = &cxi_cfg_rgroup_state_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup_ac_entry *alloc_dev_rgroup_ac_entry(const char *name)
{
	struct cxi_cfg_rgroup_ac_entry *ac_entry_dir;

	ac_entry_dir = kzalloc(sizeof(*ac_entry_dir),
			       GFP_KERNEL);
	if (!ac_entry_dir)
		return NULL;

	strscpy(ac_entry_dir->name, name, sizeof(ac_entry_dir->name));

	config_group_init_type_name(&ac_entry_dir->group,
				    ac_entry_dir->name,
				    &cxi_cfg_rgroup_ac_entry_type);

	return ac_entry_dir;
}

static struct cxi_cfg_rgroup_res *alloc_dev_rgroup_res(const char *name)
{
	struct cxi_cfg_rgroup_res *p_res;

	p_res = kzalloc(sizeof(*p_res), GFP_KERNEL);
	if (!p_res)
		return NULL;

	strscpy(p_res->name, name, sizeof(p_res->name));

	config_group_init_type_name(&p_res->group,
				    p_res->name,
				    &cxi_cfg_rgroup_res_type);

	return p_res;
}

static struct cxi_cfg_rgroup_state *alloc_dev_rgroup_state(const char *name)
{
	struct cxi_cfg_rgroup_state *rgp_st;

	rgp_st = kzalloc(sizeof(*rgp_st), GFP_KERNEL);
	if (!rgp_st)
		return NULL;

	strscpy(rgp_st->name, name, sizeof(rgp_st->name));

	config_group_init_type_name(&rgp_st->group,
				    rgp_st->name,
				    &cxi_cfg_rgroup_state_type);

	return rgp_st;
}

static struct cxi_cfg_rgroup_attr *alloc_dev_rgroup_attr(const char *name)
{
	struct cxi_cfg_rgroup_attr *p_attr;

	p_attr = kzalloc(sizeof(*p_attr), GFP_KERNEL);
	if (!p_attr)
		return NULL;

	strscpy(p_attr->name, name, sizeof(p_attr->name));

	config_group_init_type_name(&p_attr->group,
				    p_attr->name,
				    &cxi_cfg_rgroup_attr_type);

	return p_attr;
}

static struct
config_group *cxi_cfg_rgroup_make_group(struct config_group *group,
					const char *name)
{
	struct cxi_cfg_rgroup *rgp = container_of(group,
						  struct cxi_cfg_rgroup,
						  group);
	struct cxi_cfg_rgroup_state *rgp_state;
	struct cxi_cfg_rgroup_attr *rgp_attr;
	struct cxi_cfg_rgroup_ac_entry *ac_entry;
	struct cxi_cfg_rgroup_res *rgp_res;

	if (strncmp(name, "resources", sizeof("resources") - 1) == 0) {
		rgp_res = alloc_dev_rgroup_res("resources");
		if (!rgp_res) {
			pr_debug("unable to alloc rgroup resources");
			return ERR_PTR(-EINVAL);
		}
		rgp_res->rgroup = rgp->rgroup;
		rgp->rgp_res = rgp_res;
		rgp->rgp_res->cdev = rgp->cdev;

		return &rgp->rgp_res->group;
	}

	if (strncmp(name, "attr", sizeof("attr") - 1) == 0) {
		rgp_attr = alloc_dev_rgroup_attr("attr");
		if (!rgp_attr) {
			pr_debug("unable to alloc rgroup attr");
			return ERR_PTR(-EINVAL);
		}

		rgp_attr->rgroup = rgp->rgroup;
		rgp->rgp_attr = rgp_attr;
		rgp->rgp_attr->cdev = rgp->cdev;

		return &rgp->rgp_attr->group;
	}

	if (strncmp(name, "state", sizeof("state") - 1) == 0) {
		rgp_state = alloc_dev_rgroup_state("state");
		if (!rgp_state) {
			pr_debug("unable to alloc rgroup state");
			return ERR_PTR(-EINVAL);
		}
		rgp_state->rgroup = rgp->rgroup;
		rgp->rgp_state = rgp_state;
		rgp->rgp_state->cdev = rgp->cdev;

		return &rgp->rgp_state->group;
	}

	if (strncmp(name, "ac-entry", sizeof("ac-entry") - 1) == 0) {
		ac_entry = alloc_dev_rgroup_ac_entry("ac-entry");
		if (!ac_entry) {
			pr_debug("unable to alloc rgroup ac-entry");
			return ERR_PTR(-EINVAL);
		}
		ac_entry->rgroup = rgp->rgroup;
		rgp->ac_entry = ac_entry;

		return &rgp->ac_entry->group;
	}

	return ERR_PTR(-EINVAL);
}

static void
cxi_cfg_rgroup_drop_item(struct config_group *group,
			 struct config_item *item)
{
	config_item_put(item);
}

static void cxi_cfg_rgroup_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rgroup, group.cg_item));
}

static struct cxi_cfg_rgroup *to_cxi_cfg_rgroup(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rgroup, group);
}

static ssize_t cxi_cfg_rgroup_id_show(struct config_item *item,
				      char *page)
{
	struct cxi_cfg_rgroup *rgp = to_cxi_cfg_rgroup(item);

	return scnprintf(page, PAGE_SIZE, "%d\n", rgp->rgroup->id);
}

CONFIGFS_ATTR_RO(cxi_cfg_rgroup_, id);

static struct configfs_attribute *cxi_cfg_rgroup_attrs[] = {
	&cxi_cfg_rgroup_attr_id,
	NULL,
};

static struct configfs_group_operations cxi_cfg_rgroup_ops = {
	.make_group    = cxi_cfg_rgroup_make_group,
	.drop_item     = cxi_cfg_rgroup_drop_item
};

static struct configfs_item_operations cxi_cfg_rgroup_item_ops = {
	.release     = cxi_cfg_rgroup_free,
};

static const struct config_item_type cxi_cfg_rgroup_type = {
	.ct_group_ops   = &cxi_cfg_rgroup_ops,
	.ct_attrs       = cxi_cfg_rgroup_attrs,
	.ct_item_ops    = &cxi_cfg_rgroup_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rgroup *alloc_dev_rgroup(const char *name)
{
	struct cxi_cfg_rgroup *rgp;

	rgp = kzalloc(sizeof(*rgp), GFP_KERNEL);
	if (!rgp)
		return NULL;

	strscpy(rgp->name, name, sizeof(rgp->name));

	config_group_init_type_name(&rgp->group,
				    rgp->name,
				    &cxi_cfg_rgroup_type);

	return rgp;
}

static struct
config_group *cxi_cfg_rgroup_dir_make_group(struct config_group *group,
					    const char *name)
{
	struct cxi_cfg_rgroup_dir *dev_rgp;
	struct cxi_cfg_rgroup *rgp;
	struct cxi_rgroup *rgroup;
	struct cxi_rgroup_attr attr = {
		.cntr_pool_id = 0,
		.name = {'\0'},
		.system_service = false,
		.lnis_per_rgid = CXI_DEFAULT_LNIS_PER_RGID
	};

	if (strlen(name) > CONFIGFS_DIR_LEN - 1)
		return ERR_PTR(-EINVAL);

	dev_rgp = container_of(group, struct cxi_cfg_rgroup_dir, group);
	rgp = alloc_dev_rgroup(name);
	if (!rgp) {
		pr_debug("Failed to alloc memory for configfs rgroup dir");
		return ERR_PTR(-ENOMEM);
	}

	rgp->cdev = dev_rgp->cdev;
	strscpy(attr.name, name, sizeof(attr.name));
	rgroup = cxi_dev_alloc_rgroup(dev_rgp->cdev,
				      &attr);
	if (IS_ERR(rgroup)) {
		kfree(rgp);
		return ERR_PTR(PTR_ERR(rgroup));
	}
	rgp->rgroup = rgroup;

	return &rgp->group;
}

static void
cxi_cfg_rgroup_dir_drop_item(struct config_group *group,
			     struct config_item *item)
{
	struct cxi_cfg_rgroup *rgp;

	rgp = container_of(item, struct cxi_cfg_rgroup, group.cg_item);

	/*
	 * For accidental deletion of resource group directory
	 * by configfs user. In the future we will employ a
	 * different approach to handle accidental deletion.
	 */
	cxi_rgroup_dec_refcount(rgp->rgroup);

	config_item_put(item);
}

static void cxi_cfg_rgroup_dir_free(struct config_item *item)
{
	struct cxi_cfg_rgroup_dir *rgp_dir;

	rgp_dir = container_of(item, struct cxi_cfg_rgroup_dir,
			       group.cg_item);
	kfree(rgp_dir);
}

static struct configfs_group_operations cxi_cfg_rgroup_dir_ops = {
	.make_group     = cxi_cfg_rgroup_dir_make_group,
	.drop_item      = cxi_cfg_rgroup_dir_drop_item,
};

static struct configfs_item_operations cxi_cfg_rgroup_dir_item_ops = {
	.release     = cxi_cfg_rgroup_dir_free,
};

static const struct config_item_type cxi_cfg_rgroup_dir_type = {
	.ct_group_ops   = &cxi_cfg_rgroup_dir_ops,
	.ct_item_ops    = &cxi_cfg_rgroup_dir_item_ops,
	.ct_owner       = THIS_MODULE,
};

static void cxi_cfg_group_free(struct config_item *item)
{
	struct cxi_cfg_group_item *grp_dir;

	grp_dir = container_of(item, struct cxi_cfg_group_item,
			       group.cg_item);
	kfree(grp_dir);
}

static struct configfs_item_operations cxi_cfg_group_item_ops = {
	.release     = cxi_cfg_group_free,
};

static const struct config_item_type cxi_cfg_group_type = {
	.ct_item_ops    = &cxi_cfg_group_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct configfs_subsystem cxi_cfg_group_subsys = {
	.su_group = {
		.cg_item = {
			.ci_namebuf = "cxi",
			.ci_type = &cxi_cfg_group_type,
		},
	},
};

static struct cxi_cfg_group_item *alloc_dev_subdir(const char *name)
{
	struct cxi_cfg_group_item *dev_dir;

	dev_dir = kzalloc(sizeof(*dev_dir), GFP_KERNEL);
	if (!dev_dir)
		return NULL;

	strscpy(dev_dir->name, name, sizeof(dev_dir->name));

	config_group_init_type_name(&dev_dir->group,
				    dev_dir->name,
				    &cxi_cfg_group_type);

	return dev_dir;
}

static struct cxi_cfg_rgroup_dir *alloc_dev_attr1(const char *name)
{
	struct cxi_cfg_rgroup_dir *dev_rgp;

	dev_rgp = kzalloc(sizeof(*dev_rgp), GFP_KERNEL);
	if (!dev_rgp)
		return NULL;

	strscpy(dev_rgp->name, name, sizeof(dev_rgp->name));

	config_group_init_type_name(&dev_rgp->group,
				    dev_rgp->name,
				    &cxi_cfg_rgroup_dir_type);

	return dev_rgp;
}

static struct cxi_cfg_tx_prof_ac_entry_sub *to_cxi_cfg_tx_prof_ac_entry_sub(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_tx_prof_ac_entry_sub, group);
}

static ssize_t cxi_cfg_tx_prof_ac_entry_sub_type_show(struct config_item *item,
						      char *page)
{
	struct cxi_cfg_tx_prof_ac_entry_sub *tx_prof_ac_entry_sub;
	int rc;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;

	tx_prof_ac_entry_sub = to_cxi_cfg_tx_prof_ac_entry_sub(item);

	rc = cxi_tx_profile_get_ac_entry_data(tx_prof_ac_entry_sub->tx_profile,
					      tx_prof_ac_entry_sub->id,
					      &ac_type,
					      &ac_data);
	if (rc)
		return rc;

	return sprintf(page, "%d\n", ac_type);
}

static ssize_t cxi_cfg_tx_prof_ac_entry_sub_type_store(struct config_item *item,
						       const char *page,
						       size_t count)
{
	struct cxi_cfg_tx_prof_ac_entry_sub *ac_entry_sub;
	int rc;
	enum cxi_ac_type ac_type_val;
	uid_t uid;
	gid_t gid;
	unsigned int ac_entry_id;

	ac_entry_sub = to_cxi_cfg_tx_prof_ac_entry_sub(item);
	if (!ac_entry_sub->tx_profile) {
		pr_debug("Unable to get tx profile reference\n");
		return -EINVAL;
	}

	rc = kstrtouint(page, 0, (unsigned int *)&ac_type_val);
	if (rc)
		return rc;

	if (ac_type_val == CXI_AC_UID)
		uid = ac_entry_sub->ac_data.uid;
	else if (ac_type_val == CXI_AC_GID)
		gid = ac_entry_sub->ac_data.gid;
	else if (ac_type_val == CXI_AC_OPEN)
		pr_debug("Setting type to CXI_AC_OPEN\n");
	else
		return -EINVAL;

	rc = cxi_tx_profile_add_ac_entry(ac_entry_sub->tx_profile,
					 ac_type_val,
					 uid,
					 gid,
					 &ac_entry_id);
	if (rc < 0) {
		pr_debug("Unable to add ac entry to tx profile: err val %d\n", rc);
		return rc;
	}

	ac_entry_sub->id = ac_entry_id;
	ac_entry_sub->ac_type = ac_type_val;

	return count;
}

static ssize_t cxi_cfg_tx_prof_ac_entry_sub_data_show(struct config_item *item,
						      char *page)
{
	struct cxi_cfg_tx_prof_ac_entry_sub *tx_prof_ac_entry_sub;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	int rc;

	tx_prof_ac_entry_sub = to_cxi_cfg_tx_prof_ac_entry_sub(item);

	rc = cxi_tx_profile_get_ac_entry_data(tx_prof_ac_entry_sub->tx_profile,
					      tx_prof_ac_entry_sub->id,
					      &ac_type,
					      &ac_data);
	if (rc)
		return rc;

	if (ac_type == CXI_AC_UID)
		return sprintf(page, "%d\n", ac_data.uid);
	else if (ac_type == CXI_AC_GID)
		return sprintf(page, "%d\n", ac_data.gid);
	else if (ac_type == CXI_AC_OPEN)
		return sprintf(page, "%d\n", CXI_AC_OPEN);

	return -EINVAL;
}

static ssize_t cxi_cfg_tx_prof_ac_entry_sub_data_store(struct config_item *item,
						       const char *page,
						       size_t count)
{
	struct cxi_cfg_tx_prof_ac_entry_sub *ac_entry_sub;
	int rc;
	unsigned int ac_data_val;

	ac_entry_sub = to_cxi_cfg_tx_prof_ac_entry_sub(item);

	rc = kstrtouint(page, 0, &ac_data_val);
	if (rc)
		return rc;

	if (ac_entry_sub->ac_type == CXI_AC_UID) {
		ac_entry_sub->ac_data.uid = ac_data_val;
		return count;
	} else if (ac_entry_sub->ac_type == CXI_AC_GID) {
		ac_entry_sub->ac_data.gid = ac_data_val;
		return count;
	}

	return -EINVAL;
}

static void cxi_cfg_tx_prof_ac_entry_sub_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_tx_prof_ac_entry_sub, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_tx_prof_ac_entry_sub_, type);
CONFIGFS_ATTR(cxi_cfg_tx_prof_ac_entry_sub_, data);

static struct configfs_attribute *cxi_cfg_tx_prof_ac_entry_sub_attrs[] = {
	&cxi_cfg_tx_prof_ac_entry_sub_attr_type,
	&cxi_cfg_tx_prof_ac_entry_sub_attr_data,
	NULL,
};

static struct configfs_item_operations cxi_cfg_tx_prof_ac_entry_sub_item_ops = {
	.release     = cxi_cfg_tx_prof_ac_entry_sub_free,
};

static const struct config_item_type cxi_cfg_tx_prof_ac_entry_sub_type = {
	.ct_attrs       = cxi_cfg_tx_prof_ac_entry_sub_attrs,
	.ct_item_ops    = &cxi_cfg_tx_prof_ac_entry_sub_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof_ac_entry_sub *alloc_dev_tx_prof_ac_entry_sub(const char *name)
{
	struct cxi_cfg_tx_prof_ac_entry_sub *tx_prof_ac_entry_sub;

	tx_prof_ac_entry_sub = kzalloc(sizeof(*tx_prof_ac_entry_sub), GFP_KERNEL);
	if (!tx_prof_ac_entry_sub)
		return ERR_PTR(-ENOMEM);

	strscpy(tx_prof_ac_entry_sub->name, name, sizeof(tx_prof_ac_entry_sub->name));

	config_group_init_type_name(&tx_prof_ac_entry_sub->group,
				    tx_prof_ac_entry_sub->name,
				    &cxi_cfg_tx_prof_ac_entry_sub_type);

	return tx_prof_ac_entry_sub;
}

static struct
config_group *cxi_cfg_tx_prof_ac_entry_make_group(struct config_group *group,
						  const char *name)
{
	struct cxi_cfg_tx_prof_ac_entry *tx_prof_ac_entry;
	struct cxi_cfg_tx_prof_ac_entry_sub *tx_prof_ac_entry_sub;

	if (strlen(name) > CONFIGFS_DIR_LEN - 1)
		return ERR_PTR(-EINVAL);

	tx_prof_ac_entry = container_of(group, struct cxi_cfg_tx_prof_ac_entry, group);

	tx_prof_ac_entry_sub = alloc_dev_tx_prof_ac_entry_sub(name);
	if (!tx_prof_ac_entry_sub) {
		pr_debug("Memory allocation failed for ac entry group\n");
		return ERR_PTR(-ENOMEM);
	}

	tx_prof_ac_entry_sub->ac_type = CXI_AC_UID;
	tx_prof_ac_entry_sub->ac_data.uid = 0;
	tx_prof_ac_entry_sub->tx_profile = tx_prof_ac_entry->tx_profile;

	return &tx_prof_ac_entry_sub->group;
}

static void
cxi_cfg_tx_prof_ac_entry_drop_item(struct config_group *group,
				   struct config_item *item)
{
	struct cxi_cfg_tx_prof_ac_entry_sub *tx_prof_ac_entry_sub;
	int rc;
	unsigned int ac_entry_id;

	tx_prof_ac_entry_sub = container_of(item, struct cxi_cfg_tx_prof_ac_entry_sub, group.cg_item);

	if (tx_prof_ac_entry_sub->ac_type == CXI_AC_UID) {
		rc = cxi_tx_profile_get_ac_entry_id_by_data(tx_prof_ac_entry_sub->tx_profile,
							    CXI_AC_UID,
							    &tx_prof_ac_entry_sub->ac_data,
							    &ac_entry_id);

		pr_debug("Getting an entry info type uid rc = %d\n", rc);
	} else if (tx_prof_ac_entry_sub->ac_type == CXI_AC_GID) {
		rc = cxi_tx_profile_get_ac_entry_id_by_data(tx_prof_ac_entry_sub->tx_profile,
							    CXI_AC_GID,
							    &tx_prof_ac_entry_sub->ac_data,
							    &ac_entry_id);

		pr_debug("Getting an entry info type gid rc = %d\n", rc);
	} else {
		rc = cxi_tx_profile_get_ac_entry_id_by_data(tx_prof_ac_entry_sub->tx_profile,
							    CXI_AC_OPEN,
							    NULL,
							    &ac_entry_id);
		pr_debug("Getting an entry info type open rc = %d\n", rc);
	}

	cxi_tx_profile_remove_ac_entry(tx_prof_ac_entry_sub->tx_profile,
				       ac_entry_id);
	config_item_put(item);
}

static void cxi_cfg_tx_prof_ac_entry_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_tx_prof_ac_entry, group.cg_item));
}

static struct configfs_group_operations cxi_cfg_tx_prof_ac_entry_ops = {
	.make_group     = cxi_cfg_tx_prof_ac_entry_make_group,
	.drop_item      = cxi_cfg_tx_prof_ac_entry_drop_item,
};

static struct configfs_item_operations cxi_cfg_tx_prof_ac_entry_item_ops = {
	.release     = cxi_cfg_tx_prof_ac_entry_free,
};

static const struct config_item_type cxi_cfg_tx_prof_ac_entry_type = {
	.ct_group_ops   = &cxi_cfg_tx_prof_ac_entry_ops,
	.ct_item_ops    = &cxi_cfg_tx_prof_ac_entry_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof_state *to_cxi_cfg_tx_prof_state(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_tx_prof_state, group);
}

static ssize_t cxi_cfg_tx_prof_state_enabled_show(struct config_item *item,
						  char *page)
{
	struct cxi_cfg_tx_prof_state *tx_prof_state;
	struct cxi_tx_attr tx_attr;
	struct cxi_rxtx_profile_state state;
	int rc;

	tx_prof_state = to_cxi_cfg_tx_prof_state(item);

	rc = cxi_tx_profile_get_info(tx_prof_state->cdev,
				     tx_prof_state->tx_profile,
				     &tx_attr,
				     &state);
	if (rc < 0) {
		pr_debug("Unable to get tx-profile info rc = %d\n", rc);
		return rc;
	}

	return scnprintf(page, PAGE_SIZE, "%s\n", state.enable ? "true" : "false");
}

static ssize_t cxi_cfg_tx_prof_state_enabled_store(struct config_item *item,
						   const char *page,
						   size_t count)
{
	struct cxi_cfg_tx_prof_state *tx_prof_state;
	int rc;
	bool b_val;

	if (kstrtobool(page, &b_val))
		return -EINVAL;

	tx_prof_state = to_cxi_cfg_tx_prof_state(item);

	if (b_val) {
		rc = cxi_tx_profile_enable(tx_prof_state->cdev,
					   tx_prof_state->tx_profile);
		if (rc)
			return rc;
	} else {
		cxi_tx_profile_disable(tx_prof_state->cdev,
				       tx_prof_state->tx_profile);
	}

	return count;
}

static ssize_t cxi_cfg_tx_prof_state_refcount_show(struct config_item *item,
						   char *page)
{
	struct cxi_cfg_tx_prof_state *tx_prof_state;
	struct cxi_tx_attr tx_attr;
	struct cxi_rxtx_profile_state state;
	int rc;

	tx_prof_state = to_cxi_cfg_tx_prof_state(item);

	rc = cxi_tx_profile_get_info(tx_prof_state->cdev,
				     tx_prof_state->tx_profile,
				     &tx_attr,
				     &state);
	if (rc < 0) {
		pr_debug("Unable to get rx-profile info rc = %d\n", rc);
		return rc;
	}

	return sprintf(page, "%d\n", refcount_read(&state.refcount));
}

static void cxi_cfg_tx_prof_state_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_tx_prof_state, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_tx_prof_state_, enabled);
CONFIGFS_ATTR_RO(cxi_cfg_tx_prof_state_, refcount);

static struct configfs_attribute *cxi_cfg_tx_prof_state_attrs[] = {
	&cxi_cfg_tx_prof_state_attr_enabled,
	&cxi_cfg_tx_prof_state_attr_refcount,
	NULL
};

static struct configfs_item_operations cxi_cfg_tx_prof_state_item_ops = {
	.release     = cxi_cfg_tx_prof_state_free,
};

static const struct config_item_type cxi_cfg_tx_prof_state_type = {
	.ct_attrs       = cxi_cfg_tx_prof_state_attrs,
	.ct_item_ops    = &cxi_cfg_tx_prof_state_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof_state *alloc_dev_tx_prof_state(const char *name)
{
	struct cxi_cfg_tx_prof_state *tx_prof_state;

	tx_prof_state = kzalloc(sizeof(*tx_prof_state), GFP_KERNEL);
	if (!tx_prof_state)
		return ERR_PTR(-ENOMEM);

	strscpy(tx_prof_state->name, name, sizeof(tx_prof_state->name));

	config_group_init_type_name(&tx_prof_state->group,
				    tx_prof_state->name,
				    &cxi_cfg_tx_prof_state_type);
	return tx_prof_state;
}

static struct cxi_cfg_tx_prof_attr *to_cxi_cfg_tx_prof_attr(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_tx_prof_attr, group);
}

static ssize_t cxi_cfg_tx_prof_attr_name_show(struct config_item *item,
					      char *page)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 tx_prof_attr->tx_profile->profile_common.vni_attr.name);
}

static ssize_t cxi_cfg_tx_prof_attr_match_show(struct config_item *item,
					       char *page)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	struct cxi_tx_attr tx_attr;
	struct cxi_rxtx_profile_state state;
	int rc;

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);

	rc = cxi_tx_profile_get_info(tx_prof_attr->cdev,
				     tx_prof_attr->tx_profile,
				     &tx_attr,
				     &state);
	if (rc < 0)
		return rc;

	return scnprintf(page, PAGE_SIZE, "%d\n", tx_attr.vni_attr.match);
}

static ssize_t cxi_cfg_tx_prof_attr_match_store(struct config_item *item,
						const char *page,
						size_t count)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	int rc;
	struct cxi_tx_attr tx_attr = {};

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);

	rc = kstrtou16(page, 0, &tx_prof_attr->match);
	if (rc)
		return rc;

	tx_attr.vni_attr.match = tx_prof_attr->match;
	tx_attr.vni_attr.ignore = tx_prof_attr->ignore;

	strscpy(tx_attr.vni_attr.name,
		tx_prof_attr->tx_profile->profile_common.vni_attr.name,
		sizeof(tx_attr.vni_attr.name));

	rc = cxi_dev_set_tx_profile_attr(tx_prof_attr->cdev,
					 tx_prof_attr->tx_profile,
					 &tx_attr);
	if (rc)
		return rc;

	return count;
}

static ssize_t cxi_cfg_tx_prof_attr_ignore_show(struct config_item *item,
						char *page)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	int rc;
	struct cxi_tx_attr tx_attr;
	struct cxi_rxtx_profile_state state;

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);

	rc = cxi_tx_profile_get_info(tx_prof_attr->cdev,
				     tx_prof_attr->tx_profile,
				     &tx_attr,
				     &state);
	if (rc < 0) {
		pr_debug("Unable to get rx-profile info rc = %d\n", rc);
		return rc;
	}

	return scnprintf(page, PAGE_SIZE, "%d\n", tx_attr.vni_attr.ignore);
}

static ssize_t cxi_cfg_tx_prof_attr_ignore_store(struct config_item *item,
						 const char *page,
						 size_t count)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	int rc;

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);
	rc = kstrtou16(page, 0, &tx_prof_attr->ignore);
	if (rc)
		return rc;

	return count;
}

static ssize_t cxi_cfg_tx_prof_attr_exclusive_cp_show(struct config_item *item,
						      char *page)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 cxi_tx_profile_exclusive_cp(tx_prof_attr->tx_profile) ? "true" : "false");
}

static ssize_t cxi_cfg_tx_prof_attr_exclusive_cp_store(struct config_item *item,
						       const char *page,
						       size_t count)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	bool cp_val;
	int rc;

	if (kstrtobool(page, &cp_val))
		return -EINVAL;

	tx_prof_attr = to_cxi_cfg_tx_prof_attr(item);

	rc = cxi_tx_profile_set_exclusive_cp(tx_prof_attr->tx_profile,
					     cp_val);
	if (rc)
		return rc;

	return count;
}

static void cxi_cfg_tx_prof_attr_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_tx_prof_attr, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_tx_prof_attr_, ignore);
CONFIGFS_ATTR(cxi_cfg_tx_prof_attr_, match);
CONFIGFS_ATTR(cxi_cfg_tx_prof_attr_, exclusive_cp);
CONFIGFS_ATTR_RO(cxi_cfg_tx_prof_attr_, name);

static struct configfs_attribute *cxi_cfg_tx_prof_attr_attrs[] = {
	&cxi_cfg_tx_prof_attr_attr_name,
	&cxi_cfg_tx_prof_attr_attr_match,
	&cxi_cfg_tx_prof_attr_attr_ignore,
	&cxi_cfg_tx_prof_attr_attr_exclusive_cp,
	NULL
};

static struct configfs_item_operations cxi_cfg_tx_prof_attr_item_ops = {
	.release     = cxi_cfg_tx_prof_attr_free,
};

static const struct config_item_type cxi_cfg_tx_prof_attr_type = {
	.ct_attrs       = cxi_cfg_tx_prof_attr_attrs,
	.ct_item_ops    = &cxi_cfg_tx_prof_attr_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof_tc *to_cxi_cfg_tx_prof_tc(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_tx_prof_tc, group);
}

static void set_tc_bool_type(unsigned int tc_type,
			     struct cxi_cfg_tx_prof_tc *tx_prof_tc,
			     bool tc_bool)
{
	switch (tc_type) {
	case CXI_TC_BEST_EFFORT:
		tx_prof_tc->best_effort = tc_bool;
		break;
	case CXI_TC_DEDICATED_ACCESS:
		tx_prof_tc->dedicated = tc_bool;
		break;
	case CXI_TC_LOW_LATENCY:
		tx_prof_tc->low_latency = tc_bool;
		break;
	case CXI_TC_BULK_DATA:
		tx_prof_tc->bulk_data = tc_bool;
		break;
	case CXI_TC_ETH:
		tx_prof_tc->eth = tc_bool;
		break;
	}

	cxi_tx_profile_set_tc(tx_prof_tc->tx_profile, tc_type, tc_bool);
}

#define cxi_cfg_tx_prof_set_tc_common(item, tc_type, page)			\
	do {									\
		struct cxi_cfg_tx_prof_tc *tx_prof_tc;				\
		bool b_val;							\
		if (kstrtobool(page, &b_val))					\
			return -EINVAL;						\
		tx_prof_tc = to_cxi_cfg_tx_prof_tc((item));			\
		set_tc_bool_type((tc_type), tx_prof_tc, b_val);			\
	} while (0)

static ssize_t cxi_cfg_tx_prof_tc_best_effort_show(struct config_item *item,
						   char *page)
{
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof_tc = to_cxi_cfg_tx_prof_tc(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 tx_prof_tc->best_effort ? "true" : "false");
}

static ssize_t cxi_cfg_tx_prof_tc_best_effort_store(struct config_item *item,
						    const char *page,
						    size_t count)
{
	cxi_cfg_tx_prof_set_tc_common(item, CXI_TC_BEST_EFFORT, page);

	return count;
}

static ssize_t cxi_cfg_tx_prof_tc_dedicated_show(struct config_item *item,
						 char *page)
{
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof_tc = to_cxi_cfg_tx_prof_tc(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 tx_prof_tc->dedicated ? "true" : "false");
}

static ssize_t cxi_cfg_tx_prof_tc_dedicated_store(struct config_item *item,
						  const char *page,
						  size_t count)
{
	cxi_cfg_tx_prof_set_tc_common(item, CXI_TC_DEDICATED_ACCESS, page);

	return count;
}

static ssize_t cxi_cfg_tx_prof_tc_low_latency_show(struct config_item *item,
						   char *page)
{
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof_tc = to_cxi_cfg_tx_prof_tc(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 tx_prof_tc->low_latency ? "true" : "false");
}

static ssize_t cxi_cfg_tx_prof_tc_low_latency_store(struct config_item *item,
						    const char *page,
						    size_t count)
{
	cxi_cfg_tx_prof_set_tc_common(item, CXI_TC_LOW_LATENCY, page);

	return count;
}

static ssize_t cxi_cfg_tx_prof_tc_bulk_data_show(struct config_item *item,
						 char *page)
{
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof_tc = to_cxi_cfg_tx_prof_tc(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 tx_prof_tc->bulk_data ? "true" : "false");
}

static ssize_t cxi_cfg_tx_prof_tc_bulk_data_store(struct config_item *item,
						  const char *page,
						  size_t count)
{
	cxi_cfg_tx_prof_set_tc_common(item, CXI_TC_BULK_DATA, page);

	return count;
}

static ssize_t cxi_cfg_tx_prof_tc_eth_show(struct config_item *item,
					   char *page)
{
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof_tc = to_cxi_cfg_tx_prof_tc(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 tx_prof_tc->eth ? "true" : "false");
}

static void cxi_cfg_tx_prof_tc_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_tx_prof_tc, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_tx_prof_tc_,  best_effort);
CONFIGFS_ATTR(cxi_cfg_tx_prof_tc_,  dedicated);
CONFIGFS_ATTR(cxi_cfg_tx_prof_tc_,  low_latency);
CONFIGFS_ATTR(cxi_cfg_tx_prof_tc_,  bulk_data);
CONFIGFS_ATTR_RO(cxi_cfg_tx_prof_tc_,  eth);

static struct configfs_attribute *cxi_cfg_tx_prof_tc_attrs[] = {
	&cxi_cfg_tx_prof_tc_attr_best_effort,
	&cxi_cfg_tx_prof_tc_attr_dedicated,
	&cxi_cfg_tx_prof_tc_attr_low_latency,
	&cxi_cfg_tx_prof_tc_attr_bulk_data,
	&cxi_cfg_tx_prof_tc_attr_eth,
	NULL
};

static struct configfs_item_operations cxi_cfg_tx_prof_tc_item_ops = {
	.release     = cxi_cfg_tx_prof_tc_free,
};

static const struct config_item_type cxi_cfg_tx_prof_tc_type = {
	.ct_attrs       = cxi_cfg_tx_prof_tc_attrs,
	.ct_item_ops    = &cxi_cfg_tx_prof_tc_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof_tc *alloc_dev_tx_prof_tc(const char *name)
{
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof_tc = kzalloc(sizeof(*tx_prof_tc), GFP_KERNEL);
	if (!tx_prof_tc)
		return NULL;

	strscpy(tx_prof_tc->name, name, sizeof(tx_prof_tc->name));

	config_group_init_type_name(&tx_prof_tc->group,
				    tx_prof_tc->name,
				    &cxi_cfg_tx_prof_tc_type);

	return tx_prof_tc;
}

static struct cxi_cfg_tx_prof_attr *alloc_dev_tx_prof_attr(const char *name)
{
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;

	tx_prof_attr = kzalloc(sizeof(*tx_prof_attr), GFP_KERNEL);
	if (!tx_prof_attr)
		return NULL;

	strscpy(tx_prof_attr->name, name, sizeof(tx_prof_attr->name));

	config_group_init_type_name(&tx_prof_attr->group,
				    tx_prof_attr->name,
				    &cxi_cfg_tx_prof_attr_type);

	return tx_prof_attr;
}

static struct cxi_cfg_tx_prof_ac_entry *alloc_dev_tx_prof_ac_entry(const char *name)
{
	struct cxi_cfg_tx_prof_ac_entry *tx_prof_ac_entry;

	tx_prof_ac_entry = kzalloc(sizeof(*tx_prof_ac_entry),
				   GFP_KERNEL);
	if (!tx_prof_ac_entry)
		return NULL;

	strscpy(tx_prof_ac_entry->name,
		name,
		sizeof(tx_prof_ac_entry->name));

	config_group_init_type_name(&tx_prof_ac_entry->group,
				    tx_prof_ac_entry->name,
				    &cxi_cfg_tx_prof_ac_entry_type);

	return tx_prof_ac_entry;
}

static struct
config_group *cxi_cfg_tx_prof_make_group(struct config_group *group,
					 const char *name)
{
	struct cxi_cfg_tx_prof *tx_prof;
	struct cxi_cfg_tx_prof_state *tx_prof_state;
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	struct cxi_cfg_tx_prof_ac_entry *tx_prof_ac_entry;
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;

	tx_prof = container_of(group,
			       struct cxi_cfg_tx_prof,
			       group);

	if (strncmp(name, "attr", sizeof("attr") - 1) == 0) {
		tx_prof_attr = alloc_dev_tx_prof_attr("attr");
		if (!tx_prof_attr)
			return ERR_PTR(-ENOMEM);
		tx_prof->tx_prof_attr = tx_prof_attr;
		tx_prof->tx_prof_attr->cdev = tx_prof->cdev;
		tx_prof_attr->tx_profile = tx_prof->tx_profile;
		tx_prof->tx_prof_attr->cdev = tx_prof->cdev;

		return &tx_prof->tx_prof_attr->group;
	}

	if (strncmp(name, "state", sizeof("state") - 1) == 0) {
		tx_prof_state = alloc_dev_tx_prof_state("state");
		if (!tx_prof_state)
			return ERR_PTR(-ENOMEM);
		tx_prof->tx_prof_state = tx_prof_state;
		tx_prof_state->tx_profile = tx_prof->tx_profile;
		tx_prof->tx_prof_state->cdev = tx_prof->cdev;

		return &tx_prof->tx_prof_state->group;
	}

	if (strncmp(name, "ac-entry", sizeof("ac-entry") - 1) == 0) {
		tx_prof_ac_entry = alloc_dev_tx_prof_ac_entry("ac-entry");
		if (!tx_prof_ac_entry)
			return ERR_PTR(-ENOMEM);
		tx_prof->tx_prof_ac_entry = tx_prof_ac_entry;
		tx_prof_ac_entry->tx_profile = tx_prof->tx_profile;
		tx_prof_ac_entry->cdev = tx_prof->cdev;

		return &tx_prof->tx_prof_ac_entry->group;
	}

	if (strncmp(name, "tc", sizeof("tc") - 1) == 0) {
		tx_prof_tc = alloc_dev_tx_prof_tc("tc");
		if (!tx_prof_tc)
			return ERR_PTR(-ENOMEM);
		tx_prof->tx_prof_tc = tx_prof_tc;
		tx_prof_tc->tx_profile = tx_prof->tx_profile;
		tx_prof_tc->cdev = tx_prof->cdev;

		return &tx_prof->tx_prof_tc->group;
	}

	return ERR_PTR(-EINVAL);
}

static void
cxi_cfg_tx_prof_drop_item(struct config_group *group,
			  struct config_item *item)
{
	config_item_put(item);
}

static void cxi_cfg_tx_prof_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_tx_prof, group.cg_item));
}

static struct cxi_cfg_tx_prof *to_cxi_cfg_tx_prof(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_tx_prof, group);
}

static ssize_t cxi_cfg_tx_prof_id_show(struct config_item *item,
				       char *page)
{
	struct cxi_cfg_tx_prof *tx_prof = to_cxi_cfg_tx_prof(item);

	return scnprintf(page, PAGE_SIZE, "%d\n", tx_prof->id);
}

CONFIGFS_ATTR_RO(cxi_cfg_tx_prof_, id);

static struct configfs_attribute *cxi_cfg_tx_prof_attrs[] = {
	&cxi_cfg_tx_prof_attr_id,
	NULL,
};

static struct configfs_group_operations cxi_cfg_tx_prof_ops = {
	.make_group = cxi_cfg_tx_prof_make_group,
	.drop_item  = cxi_cfg_tx_prof_drop_item,
};

static struct configfs_item_operations cxi_cfg_tx_prof_item_ops = {
	.release     = cxi_cfg_tx_prof_free,
};

static const struct config_item_type cxi_cfg_tx_prof_type = {
	.ct_group_ops   = &cxi_cfg_tx_prof_ops,
	.ct_attrs       = cxi_cfg_tx_prof_attrs,
	.ct_item_ops    = &cxi_cfg_tx_prof_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof *alloc_dev_tx_prof(const char *name)
{
	struct cxi_cfg_tx_prof *dev_tx_prof;

	dev_tx_prof = kzalloc(sizeof(struct cxi_cfg_tx_prof), GFP_KERNEL);
	if (!dev_tx_prof)
		return NULL;

	strscpy(dev_tx_prof->name, name, sizeof(dev_tx_prof->name));

	config_group_init_type_name(&dev_tx_prof->group,
				    dev_tx_prof->name,
				    &cxi_cfg_tx_prof_type);

	return dev_tx_prof;
}

static struct
config_group *cxi_cfg_tx_prof_dir_make_group(struct config_group *group,
					     const char *name)
{
	struct cxi_cfg_tx_prof_dir *tx_prof_dir;
	struct cxi_cfg_tx_prof *tx_prof;
	struct cxi_tx_profile *tx_prof_ptr;
	struct cxi_tx_attr tx_attr = {};

	if (strlen(name) > CONFIGFS_DIR_LEN - 1)
		return ERR_PTR(-EINVAL);

	tx_prof_dir = container_of(group, struct cxi_cfg_tx_prof_dir, group);
	tx_prof = alloc_dev_tx_prof(name);
	if (!tx_prof) {
		pr_debug("Failed to alloc memory\n");
		return ERR_PTR(-ENOMEM);
	}

	tx_prof->cdev = tx_prof_dir->cdev;
	strscpy(tx_attr.vni_attr.name, name, sizeof(tx_attr.vni_attr.name));

	tx_prof_ptr = cxi_dev_alloc_tx_profile(tx_prof->cdev,
					       &tx_attr);
	if (IS_ERR(tx_prof_ptr)) {
		kfree(tx_prof);
		pr_debug("Failed to alloc rx profile\n");
		return ERR_PTR(PTR_ERR(tx_prof_ptr));
	}

	tx_prof->tx_profile = tx_prof_ptr;
	tx_prof->id = tx_prof_ptr->profile_common.id;

	return &tx_prof->group;
}

static void
cxi_cfg_tx_prof_dir_drop_item(struct config_group *group,
			      struct config_item *item)
{
	struct cxi_cfg_tx_prof *tx_prof;

	tx_prof = container_of(item, struct cxi_cfg_tx_prof, group.cg_item);

	/*
	 * For accidental deletion of resource group directory
	 * by configfs user. In the future we will employ a
	 * different approach to handle accidental deletion.
	 */
	cxi_tx_profile_dec_refcount(tx_prof->cdev,
				    tx_prof->tx_profile,
				    true);
	config_item_put(item);
}

static void cxi_cfg_tx_prof_dir_free(struct config_item *item)
{
	struct cxi_cfg_tx_prof_dir *tx_prof_dir;

	tx_prof_dir = container_of(item, struct cxi_cfg_tx_prof_dir,
				   group.cg_item);
	kfree(tx_prof_dir);
}

static struct configfs_group_operations cxi_cfg_tx_prof_dir_ops = {
	.make_group     = cxi_cfg_tx_prof_dir_make_group,
	.drop_item      = cxi_cfg_tx_prof_dir_drop_item,
};

static struct configfs_item_operations cxi_cfg_tx_prof_dir_item_ops = {
	.release     = cxi_cfg_tx_prof_dir_free,
};

static const struct config_item_type cxi_cfg_tx_prof_dir_type = {
	.ct_group_ops   = &cxi_cfg_tx_prof_dir_ops,
	.ct_item_ops    = &cxi_cfg_tx_prof_dir_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_tx_prof_dir *alloc_dev_tx_prof_dir(const char *name)
{
	struct cxi_cfg_tx_prof_dir *dev_tx_prof_dir;

	dev_tx_prof_dir = kzalloc(sizeof(*dev_tx_prof_dir), GFP_KERNEL);
	if (!dev_tx_prof_dir)
		return NULL;

	strscpy(dev_tx_prof_dir->name, name, sizeof(dev_tx_prof_dir->name));

	config_group_init_type_name(&dev_tx_prof_dir->group,
				    dev_tx_prof_dir->name,
				    &cxi_cfg_tx_prof_dir_type);

	return dev_tx_prof_dir;
}

static struct cxi_cfg_rx_prof_ac_entry_sub *to_cxi_cfg_rx_prof_ac_entry_sub(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rx_prof_ac_entry_sub, group);
}

static ssize_t cxi_cfg_rx_prof_ac_entry_sub_type_show(struct config_item *item,
						      char *page)
{
	struct cxi_cfg_rx_prof_ac_entry_sub *rx_prof_ac_entry_sub;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	int rc;

	rx_prof_ac_entry_sub = to_cxi_cfg_rx_prof_ac_entry_sub(item);

	rc = cxi_rx_profile_get_ac_entry_data(rx_prof_ac_entry_sub->rx_profile,
					      rx_prof_ac_entry_sub->id,
					      &ac_type,
					      &ac_data);
	if (rc)
		return rc;

	return sprintf(page, "%d\n", ac_type);
}

static ssize_t cxi_cfg_rx_prof_ac_entry_sub_type_store(struct config_item *item,
						       const char *page,
						       size_t count)
{
	struct cxi_cfg_rx_prof_ac_entry_sub *ac_entry_sub;
	int rc;
	enum cxi_ac_type ac_type_val;
	uid_t uid;
	gid_t gid;
	unsigned int ac_entry_id;

	ac_entry_sub = to_cxi_cfg_rx_prof_ac_entry_sub(item);
	if (!ac_entry_sub->rx_profile) {
		pr_debug("Unable to get rx profile reference\n");
		return -EINVAL;
	}

	rc = kstrtouint(page, 0, (unsigned int *)&ac_type_val);
	if (rc)
		return rc;

	if (ac_type_val == CXI_AC_UID)
		uid = ac_entry_sub->ac_data.uid;
	else if (ac_type_val == CXI_AC_GID)
		gid = ac_entry_sub->ac_data.gid;
	else if (ac_type_val == CXI_AC_OPEN)
		pr_debug("Setting type to CXI_AC_OPEN\n");
	else
		return -EINVAL;

	rc = cxi_rx_profile_add_ac_entry(ac_entry_sub->rx_profile,
					 ac_type_val,
					 uid,
					 gid,
					 &ac_entry_id);
	if (rc < 0) {
		pr_debug("Unable to add ac entry to rx profile: err val %d\n", rc);
		return rc;
	}

	ac_entry_sub->id = ac_entry_id;
	ac_entry_sub->ac_type = ac_type_val;

	return count;
}

static ssize_t cxi_cfg_rx_prof_ac_entry_sub_data_show(struct config_item *item,
						      char *page)
{
	struct cxi_cfg_rx_prof_ac_entry_sub *rx_prof_ac_entry_sub;
	int rc;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;

	rx_prof_ac_entry_sub = to_cxi_cfg_rx_prof_ac_entry_sub(item);

	rc = cxi_rx_profile_get_ac_entry_data(rx_prof_ac_entry_sub->rx_profile,
					      rx_prof_ac_entry_sub->id,
					      &ac_type,
					      &ac_data);
	if (rc)
		return rc;

	if (ac_type == CXI_AC_UID)
		return sprintf(page, "%d\n", ac_data.uid);
	else if (ac_type == CXI_AC_GID)
		return sprintf(page, "%d\n", ac_data.gid);
	else if (ac_type == CXI_AC_OPEN)
		return sprintf(page, "%d\n", CXI_AC_OPEN);

	return -EINVAL;
}

static ssize_t cxi_cfg_rx_prof_ac_entry_sub_data_store(struct config_item *item,
						       const char *page,
						       size_t count)
{
	struct cxi_cfg_rx_prof_ac_entry_sub *ac_entry_sub;
	int rc;
	unsigned int ac_data_val;

	ac_entry_sub = to_cxi_cfg_rx_prof_ac_entry_sub(item);

	rc = kstrtouint(page, 0, &ac_data_val);
	if (rc)
		return rc;

	if (ac_entry_sub->ac_type == CXI_AC_UID) {
		ac_entry_sub->ac_data.uid = ac_data_val;
		return count;
	} else if (ac_entry_sub->ac_type == CXI_AC_GID) {
		ac_entry_sub->ac_data.gid = ac_data_val;
		return count;
	}

	return -EINVAL;
}

static void cxi_cfg_rx_prof_ac_entry_sub_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rx_prof_ac_entry_sub, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_rx_prof_ac_entry_sub_, type);
CONFIGFS_ATTR(cxi_cfg_rx_prof_ac_entry_sub_, data);

static struct configfs_attribute *cxi_cfg_rx_prof_ac_entry_sub_attrs[] = {
	&cxi_cfg_rx_prof_ac_entry_sub_attr_type,
	&cxi_cfg_rx_prof_ac_entry_sub_attr_data,
	NULL,
};

static struct configfs_item_operations cxi_cfg_rx_prof_ac_entry_sub_item_ops = {
	.release     = cxi_cfg_rx_prof_ac_entry_sub_free,
};

static const struct config_item_type cxi_cfg_rx_prof_ac_entry_sub_type = {
	.ct_attrs       = cxi_cfg_rx_prof_ac_entry_sub_attrs,
	.ct_item_ops    = &cxi_cfg_rx_prof_ac_entry_sub_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rx_prof_ac_entry_sub *alloc_dev_rx_prof_ac_entry_sub(const char *name)
{
	struct cxi_cfg_rx_prof_ac_entry_sub *rx_prof_ac_entry_sub;

	rx_prof_ac_entry_sub = kzalloc(sizeof(*rx_prof_ac_entry_sub), GFP_KERNEL);
	if (!rx_prof_ac_entry_sub)
		return ERR_PTR(-ENOMEM);

	strscpy(rx_prof_ac_entry_sub->name, name, sizeof(rx_prof_ac_entry_sub->name));

	config_group_init_type_name(&rx_prof_ac_entry_sub->group,
				    rx_prof_ac_entry_sub->name,
				    &cxi_cfg_rx_prof_ac_entry_sub_type);

	return rx_prof_ac_entry_sub;
}

static struct
config_group *cxi_cfg_rx_prof_ac_entry_make_group(struct config_group *group,
						  const char *name)
{
	struct cxi_cfg_rx_prof_ac_entry *rx_prof_ac_entry;
	struct cxi_cfg_rx_prof_ac_entry_sub *rx_prof_ac_entry_sub;

	if (strlen(name) > CONFIGFS_DIR_LEN - 1)
		return ERR_PTR(-EINVAL);

	rx_prof_ac_entry_sub = alloc_dev_rx_prof_ac_entry_sub(name);
	if (!rx_prof_ac_entry_sub) {
		pr_debug("Memory allocation failed for ac entry group\n");
		return ERR_PTR(-ENOMEM);
	}

	rx_prof_ac_entry_sub->ac_type = CXI_AC_UID;
	rx_prof_ac_entry_sub->ac_data.uid = 0;

	rx_prof_ac_entry = container_of(group, struct cxi_cfg_rx_prof_ac_entry, group);
	rx_prof_ac_entry_sub->rx_profile = rx_prof_ac_entry->rx_profile;

	return &rx_prof_ac_entry_sub->group;
}

static void
cxi_cfg_rx_prof_ac_entry_drop_item(struct config_group *group,
				   struct config_item *item)
{
	struct cxi_cfg_rx_prof_ac_entry_sub *rx_prof_ac_entry_sub;

	rx_prof_ac_entry_sub = container_of(item, struct cxi_cfg_rx_prof_ac_entry_sub, group.cg_item);
	cxi_rx_profile_remove_ac_entry(rx_prof_ac_entry_sub->rx_profile,
				       rx_prof_ac_entry_sub->id);

	config_item_put(item);
}

static void cxi_cfg_rx_prof_ac_entry_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rx_prof_ac_entry, group.cg_item));
}

static struct configfs_group_operations cxi_cfg_rx_prof_ac_entry_ops = {
	.make_group     = cxi_cfg_rx_prof_ac_entry_make_group,
	.drop_item      = cxi_cfg_rx_prof_ac_entry_drop_item,
};

static struct configfs_item_operations cxi_cfg_rx_prof_ac_entry_item_ops = {
	.release     = cxi_cfg_rx_prof_ac_entry_free,
};

static const struct config_item_type cxi_cfg_rx_prof_ac_entry_type = {
	.ct_group_ops   = &cxi_cfg_rx_prof_ac_entry_ops,
	.ct_item_ops    = &cxi_cfg_rx_prof_ac_entry_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rx_prof_state *to_cxi_cfg_rx_prof_state(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rx_prof_state, group);
}

static ssize_t cxi_cfg_rx_prof_state_enabled_show(struct config_item *item,
						  char *page)
{
	struct cxi_cfg_rx_prof_state *rx_prof_state;
	struct cxi_rx_attr rx_attr;
	struct cxi_rxtx_profile_state state;
	int rc;

	rx_prof_state = to_cxi_cfg_rx_prof_state(item);

	rc = cxi_rx_profile_get_info(rx_prof_state->cdev,
				     rx_prof_state->rx_profile,
				     &rx_attr,
				     &state);
	if (rc < 0) {
		pr_debug("Unable to get rx-profile info rc = %d\n", rc);
		return rc;
	}

	return scnprintf(page, PAGE_SIZE, "%s\n", state.enable ? "true" : "false");
}

static ssize_t cxi_cfg_rx_prof_state_enabled_store(struct config_item *item,
						   const char *page,
						   size_t count)
{
	struct cxi_cfg_rx_prof_state *rx_prof_state;
	int rc;
	bool b_val;

	if (kstrtobool(page, &b_val))
		return -EINVAL;

	rx_prof_state = to_cxi_cfg_rx_prof_state(item);

	if (b_val) {
		rc = cxi_rx_profile_enable(rx_prof_state->cdev,
					   rx_prof_state->rx_profile);
		if (rc)
			return rc;
	} else {
		cxi_rx_profile_disable(rx_prof_state->cdev,
				       rx_prof_state->rx_profile);
	}

	return count;
}

static ssize_t cxi_cfg_rx_prof_state_refcount_show(struct config_item *item,
						   char *page)
{
	struct cxi_cfg_rx_prof_state *rx_prof_state;
	struct cxi_rx_attr rx_attr;
	struct cxi_rxtx_profile_state state;
	int rc;

	rx_prof_state = to_cxi_cfg_rx_prof_state(item);

	rc = cxi_rx_profile_get_info(rx_prof_state->cdev,
				     rx_prof_state->rx_profile,
				     &rx_attr,
				     &state);
	if (rc < 0) {
		pr_debug("Unable to get rx-profile info rc = %d\n", rc);
		return rc;
	}

	return sprintf(page, "%d\n", refcount_read(&state.refcount));
}

static void cxi_cfg_rx_prof_state_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rx_prof_state, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_rx_prof_state_, enabled);
CONFIGFS_ATTR_RO(cxi_cfg_rx_prof_state_, refcount);

static struct configfs_attribute *cxi_cfg_rx_prof_state_attrs[] = {
	&cxi_cfg_rx_prof_state_attr_enabled,
	&cxi_cfg_rx_prof_state_attr_refcount,
	NULL
};

static struct configfs_item_operations cxi_cfg_rx_prof_state_item_ops = {
	.release     = cxi_cfg_rx_prof_state_free,
};

static const struct config_item_type cxi_cfg_rx_prof_state_type = {
	.ct_attrs       = cxi_cfg_rx_prof_state_attrs,
	.ct_item_ops    = &cxi_cfg_rx_prof_state_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rx_prof_state *alloc_dev_rx_prof_state(const char *name)
{
	struct cxi_cfg_rx_prof_state *rx_prof_state;

	rx_prof_state = kzalloc(sizeof(*rx_prof_state), GFP_KERNEL);
	if (!rx_prof_state)
		return ERR_PTR(-ENOMEM);

	strscpy(rx_prof_state->name, name, sizeof(rx_prof_state->name));

	config_group_init_type_name(&rx_prof_state->group,
				    rx_prof_state->name,
				    &cxi_cfg_rx_prof_state_type);

	return rx_prof_state;
}

static struct cxi_cfg_rx_prof_attr *to_cxi_cfg_rx_prof_attr(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rx_prof_attr, group);
}

static ssize_t cxi_cfg_rx_prof_attr_name_show(struct config_item *item,
					      char *page)
{
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;

	rx_prof_attr = to_cxi_cfg_rx_prof_attr(item);

	return scnprintf(page, PAGE_SIZE, "%s\n",
			 rx_prof_attr->rx_profile->profile_common.vni_attr.name);
}

static ssize_t cxi_cfg_rx_prof_attr_match_show(struct config_item *item,
					       char *page)
{
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;
	struct cxi_rx_attr rx_attr;
	struct cxi_rxtx_profile_state state;
	int rc;

	rx_prof_attr = to_cxi_cfg_rx_prof_attr(item);

	rc = cxi_rx_profile_get_info(rx_prof_attr->cdev,
				     rx_prof_attr->rx_profile,
				     &rx_attr,
				     &state);
	if (rc < 0)
		return rc;

	return scnprintf(page, PAGE_SIZE, "%d\n", rx_attr.vni_attr.match);
}

static ssize_t cxi_cfg_rx_prof_attr_match_store(struct config_item *item,
						const char *page,
						size_t count)
{
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;
	int rc;
	struct cxi_rx_attr rx_attr = {};

	rx_prof_attr = to_cxi_cfg_rx_prof_attr(item);
	rc = kstrtou16(page, 0, &rx_prof_attr->match);
	if (rc)
		return rc;

	rx_attr.vni_attr.match = rx_prof_attr->match;
	rx_attr.vni_attr.ignore = rx_prof_attr->ignore;

	strscpy(rx_attr.vni_attr.name,
		rx_prof_attr->rx_profile->profile_common.vni_attr.name,
		sizeof(rx_attr.vni_attr.name));

	rc = cxi_dev_set_rx_profile_attr(rx_prof_attr->cdev,
					 rx_prof_attr->rx_profile,
					 &rx_attr);
	if (rc)
		return rc;

	return count;
}

static ssize_t cxi_cfg_rx_prof_attr_ignore_show(struct config_item *item,
						char *page)
{
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;
	int rc;
	struct cxi_rx_attr rx_attr;
	struct cxi_rxtx_profile_state state;

	rx_prof_attr = to_cxi_cfg_rx_prof_attr(item);

	rc = cxi_rx_profile_get_info(rx_prof_attr->cdev,
				     rx_prof_attr->rx_profile,
				     &rx_attr,
				     &state);
	if (rc < 0) {
		pr_err("Unable to get rx-profile info rc = %d\n", rc);
		return rc;
	}

	return scnprintf(page, PAGE_SIZE, "%d\n", rx_attr.vni_attr.ignore);
}

static ssize_t cxi_cfg_rx_prof_attr_ignore_store(struct config_item *item,
						 const char *page,
						 size_t count)
{
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;
	int rc;

	rx_prof_attr = to_cxi_cfg_rx_prof_attr(item);
	rc = kstrtou16(page, 0, &rx_prof_attr->ignore);
	if (rc)
		return rc;

	return count;
}

static void cxi_cfg_rx_prof_attr_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rx_prof_attr, group.cg_item));
}

CONFIGFS_ATTR(cxi_cfg_rx_prof_attr_, ignore);
CONFIGFS_ATTR(cxi_cfg_rx_prof_attr_, match);
CONFIGFS_ATTR_RO(cxi_cfg_rx_prof_attr_, name);

static struct configfs_attribute *cxi_cfg_rx_prof_attr_attrs[] = {
	&cxi_cfg_rx_prof_attr_attr_name,
	&cxi_cfg_rx_prof_attr_attr_match,
	&cxi_cfg_rx_prof_attr_attr_ignore,
	NULL
};

static struct configfs_item_operations cxi_cfg_rx_prof_attr_item_ops = {
	.release     = cxi_cfg_rx_prof_attr_free,
};

static const struct config_item_type cxi_cfg_rx_prof_attr_type = {
	.ct_attrs       = cxi_cfg_rx_prof_attr_attrs,
	.ct_item_ops    = &cxi_cfg_rx_prof_attr_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rx_prof_attr *alloc_dev_rx_prof_attr(const char *name)
{
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;

	rx_prof_attr = kzalloc(sizeof(*rx_prof_attr), GFP_KERNEL);
	if (!rx_prof_attr)
		return NULL;

	strscpy(rx_prof_attr->name, name, sizeof(rx_prof_attr->name));

	config_group_init_type_name(&rx_prof_attr->group,
				    rx_prof_attr->name,
				    &cxi_cfg_rx_prof_attr_type);

	return rx_prof_attr;
}

static struct cxi_cfg_rx_prof_ac_entry *alloc_dev_rx_prof_ac_entry(const char *name)
{
	struct cxi_cfg_rx_prof_ac_entry *rx_prof_ac_entry;

	rx_prof_ac_entry = kzalloc(sizeof(*rx_prof_ac_entry),
				   GFP_KERNEL);
	if (!rx_prof_ac_entry)
		return NULL;

	strscpy(rx_prof_ac_entry->name,
		name,
		sizeof(rx_prof_ac_entry->name));

	config_group_init_type_name(&rx_prof_ac_entry->group,
				    rx_prof_ac_entry->name,
				    &cxi_cfg_rx_prof_ac_entry_type);

	return rx_prof_ac_entry;
}

static struct
config_group *cxi_cfg_rx_prof_make_group(struct config_group *group,
					 const char *name)
{
	struct cxi_cfg_rx_prof *rx_prof;
	struct cxi_cfg_rx_prof_state *rx_prof_state;
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;
	struct cxi_cfg_rx_prof_ac_entry *rx_prof_ac_entry;

	rx_prof = container_of(group, struct cxi_cfg_rx_prof, group);

	if (strncmp(name, "attr", sizeof("attr") - 1) == 0) {
		rx_prof_attr = alloc_dev_rx_prof_attr("attr");
		if (!rx_prof_attr)
			return ERR_PTR(-ENOMEM);

		rx_prof->rx_prof_attr = rx_prof_attr;
		rx_prof->rx_prof_attr->cdev = rx_prof->cdev;
		rx_prof_attr->rx_profile = rx_prof->rx_profile;
		rx_prof->rx_prof_attr->cdev = rx_prof->cdev;

		return &rx_prof->rx_prof_attr->group;
	}

	if (strncmp(name, "state", sizeof("state") - 1) == 0) {
		rx_prof_state = alloc_dev_rx_prof_state("state");
		if (!rx_prof_state)
			return ERR_PTR(-ENOMEM);

		rx_prof->rx_prof_state = rx_prof_state;
		rx_prof_state->rx_profile = rx_prof->rx_profile;
		rx_prof->rx_prof_state->cdev = rx_prof->cdev;

		return &rx_prof->rx_prof_state->group;
	}

	if (strncmp(name, "ac-entry", sizeof("ac-entry") - 1) == 0) {
		rx_prof_ac_entry = alloc_dev_rx_prof_ac_entry("ac-entry");
		if (!rx_prof_ac_entry)
			return ERR_PTR(-ENOMEM);

		rx_prof->rx_prof_ac_entry = rx_prof_ac_entry;
		rx_prof_ac_entry->rx_profile = rx_prof->rx_profile;
		rx_prof_ac_entry->cdev = rx_prof->cdev;

		return &rx_prof->rx_prof_ac_entry->group;
	}

	return ERR_PTR(-EINVAL);
}

static void
cxi_cfg_rx_prof_drop_item(struct config_group *group,
			  struct config_item *item)
{
	config_item_put(item);
}

static void cxi_cfg_rx_prof_free(struct config_item *item)
{
	kfree(container_of(item, struct cxi_cfg_rx_prof, group.cg_item));
}

static struct cxi_cfg_rx_prof *to_cxi_cfg_rx_prof(struct config_item *item)
{
	return container_of(to_config_group(item), struct cxi_cfg_rx_prof, group);
}

static ssize_t cxi_cfg_rx_prof_id_show(struct config_item *item,
				       char *page)
{
	struct cxi_cfg_rx_prof *rx_prof = to_cxi_cfg_rx_prof(item);

	return scnprintf(page, PAGE_SIZE, "%d\n", rx_prof->id);
}

CONFIGFS_ATTR_RO(cxi_cfg_rx_prof_, id);

static struct configfs_attribute *cxi_cfg_rx_prof_attrs[] = {
	&cxi_cfg_rx_prof_attr_id,
	NULL,
};

static struct configfs_group_operations cxi_cfg_rx_prof_ops = {
	.make_group = cxi_cfg_rx_prof_make_group,
	.drop_item  = cxi_cfg_rx_prof_drop_item,
};

static struct configfs_item_operations cxi_cfg_rx_prof_item_ops = {
	.release     = cxi_cfg_rx_prof_free,
};

static const struct config_item_type cxi_cfg_rx_prof_type = {
	.ct_group_ops   = &cxi_cfg_rx_prof_ops,
	.ct_attrs       = cxi_cfg_rx_prof_attrs,
	.ct_item_ops    = &cxi_cfg_rx_prof_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rx_prof *alloc_dev_rx_prof(const char *name)
{
	struct cxi_cfg_rx_prof *dev_rx_prof;

	dev_rx_prof = kzalloc(sizeof(struct cxi_cfg_rx_prof), GFP_KERNEL);
	if (!dev_rx_prof)
		return NULL;

	strscpy(dev_rx_prof->name, name, sizeof(dev_rx_prof->name));

	config_group_init_type_name(&dev_rx_prof->group,
				    dev_rx_prof->name,
				    &cxi_cfg_rx_prof_type);

	return dev_rx_prof;
}

static struct
config_group *cxi_cfg_rx_prof_dir_make_group(struct config_group *group,
					     const char *name)
{
	struct cxi_cfg_rx_prof_dir *rx_prof_dir;
	struct cxi_cfg_rx_prof *rx_prof;
	struct cxi_rx_profile *rx_prof_ptr;
	struct cxi_rx_attr rx_attr = {};

	if (strlen(name) > CONFIGFS_DIR_LEN - 1)
		return ERR_PTR(-EINVAL);

	rx_prof_dir = container_of(group, struct cxi_cfg_rx_prof_dir, group);
	rx_prof = alloc_dev_rx_prof(name);
	if (!rx_prof) {
		pr_debug("Failed to alloc memory\n");
		return ERR_PTR(-ENOMEM);
	}

	rx_prof->cdev = rx_prof_dir->cdev;

	strscpy(rx_attr.vni_attr.name, name, sizeof(rx_attr.vni_attr.name));

	rx_prof_ptr = cxi_dev_alloc_rx_profile(rx_prof->cdev,
					       &rx_attr);
	if (IS_ERR(rx_prof_ptr)) {
		kfree(rx_prof);
		pr_debug("Failed to alloc rx profile\n");
		return ERR_PTR(PTR_ERR(rx_prof_ptr));
	}

	rx_prof->rx_profile = rx_prof_ptr;
	rx_prof->id = rx_prof_ptr->profile_common.id;

	return &rx_prof->group;
}

static void
cxi_cfg_rx_prof_dir_drop_item(struct config_group *group,
			      struct config_item *item)
{
	struct cxi_cfg_rx_prof *rx_prof;

	rx_prof = container_of(item, struct cxi_cfg_rx_prof, group.cg_item);

	/*
	 * For accidental deletion of resource group directory
	 * by configfs user. In the future we will employ a
	 * different approach to handle accidental deletion.
	 */
	cxi_rx_profile_dec_refcount(rx_prof->cdev,
				    rx_prof->rx_profile);

	config_item_put(item);
}

static void cxi_cfg_rx_prof_dir_free(struct config_item *item)
{
	struct cxi_cfg_rx_prof_dir *rx_prof_dir;

	rx_prof_dir = container_of(item, struct cxi_cfg_rx_prof_dir,
				   group.cg_item);
	kfree(rx_prof_dir);
}

static struct configfs_group_operations cxi_cfg_rx_prof_dir_ops = {
	.make_group     = cxi_cfg_rx_prof_dir_make_group,
	.drop_item      = cxi_cfg_rx_prof_dir_drop_item,
};

static struct configfs_item_operations cxi_cfg_rx_prof_dir_item_ops = {
	.release     = cxi_cfg_rx_prof_dir_free,
};

static const struct config_item_type cxi_cfg_rx_prof_dir_type = {
	.ct_group_ops   = &cxi_cfg_rx_prof_dir_ops,
	.ct_item_ops    = &cxi_cfg_rx_prof_dir_item_ops,
	.ct_owner       = THIS_MODULE,
};

static struct cxi_cfg_rx_prof_dir *alloc_dev_rx_prof_dir(const char *name)
{
	struct cxi_cfg_rx_prof_dir *dev_rx_prof_dir;

	dev_rx_prof_dir = kzalloc(sizeof(*dev_rx_prof_dir), GFP_KERNEL);
	if (!dev_rx_prof_dir)
		return NULL;

	strscpy(dev_rx_prof_dir->name, name, sizeof(dev_rx_prof_dir->name));

	config_group_init_type_name(&dev_rx_prof_dir->group,
				    dev_rx_prof_dir->name,
				    &cxi_cfg_rx_prof_dir_type);

	return dev_rx_prof_dir;
}

/*
 * TODO: kmemleak shows leak if configfs_register_group fails
 * Memory was allocated by config_group_init_type_name()
 */
int cxi_configfs_device_init(struct cass_dev *hw)
{
	struct config_group *parent_group;
	struct cxi_cfg_group_item *cxi_cfg_configfs;
	struct cxi_cfg_rgroup_dir *rgp_dir;
	struct cxi_cfg_rx_prof_dir *rx_prof_dir;
	struct cxi_cfg_tx_prof_dir *tx_prof_dir;
	int rc;

	parent_group = &cxi_cfg_group_subsys.su_group;

	cxi_cfg_configfs = alloc_dev_subdir(hw->cdev.name);
	if (!cxi_cfg_configfs)
		return -ENOMEM;

	hw->cfg_dev_dir = cxi_cfg_configfs;

	rc = configfs_register_group(parent_group,
				     &cxi_cfg_configfs->group);
	if (rc)
		goto dev_grp_reg_err;

	rgp_dir = alloc_dev_attr1("rgroup");
	if (!rgp_dir) {
		rc = -ENOMEM;
		goto rgp_dir_alloc_err;
	}

	rc = configfs_register_group(&cxi_cfg_configfs->group,
				     &rgp_dir->group);
	if (rc)
		goto rgp_dir_reg_err;

	cxi_cfg_configfs->rgp_dir = rgp_dir;
	rgp_dir->cdev = &hw->cdev;

	rx_prof_dir = alloc_dev_rx_prof_dir("rx-profile");
	if (!rx_prof_dir) {
		rc = -ENOMEM;
		goto rx_prof_dir_alloc_err;
	}

	rc = configfs_register_group(&cxi_cfg_configfs->group,
				     &rx_prof_dir->group);
	if (rc)
		goto rx_prof_dir_reg_err;

	cxi_cfg_configfs->rx_prof_dir = rx_prof_dir;
	rx_prof_dir->cdev = &hw->cdev;

	tx_prof_dir = alloc_dev_tx_prof_dir("tx-profile");
	if (!tx_prof_dir) {
		rc = -ENOMEM;
		goto tx_prof_dir_alloc_err;
	}

	rc = configfs_register_group(&cxi_cfg_configfs->group,
				     &tx_prof_dir->group);
	if (rc)
		goto tx_prof_dir_reg_err;

	cxi_cfg_configfs->tx_prof_dir = tx_prof_dir;
	tx_prof_dir->cdev = &hw->cdev;

	return 0;

tx_prof_dir_reg_err:
	kfree(tx_prof_dir);
tx_prof_dir_alloc_err:
	configfs_unregister_group(&rx_prof_dir->group);
rx_prof_dir_reg_err:
	kfree(rx_prof_dir);
rx_prof_dir_alloc_err:
	configfs_unregister_group(&rgp_dir->group);
rgp_dir_reg_err:
	kfree(rgp_dir);
rgp_dir_alloc_err:
	configfs_unregister_group(&cxi_cfg_configfs->group);
dev_grp_reg_err:
	kfree(cxi_cfg_configfs);

	return rc;
}

int cxi_configfs_subsys_init(void)
{
	struct configfs_subsystem *subsys;
	int rc;

	subsys = &cxi_cfg_group_subsys;

	config_group_init(&subsys->su_group);
	mutex_init(&subsys->su_mutex);

	rc = configfs_register_subsystem(subsys);
	if (rc)
		return rc;

	return 0;
}

static void unregister_all_sub_config_groups_recur(struct config_group *parent_group)
{
	struct config_item *child_item;
	struct config_item *t;
	struct config_group *child_group;

	list_for_each_entry_safe(child_item, t, &parent_group->cg_children, ci_entry) {
		child_group = container_of(child_item,
					   struct config_group,
					   cg_item);
		unregister_all_sub_config_groups_recur(child_group);
		configfs_unregister_group(child_group);
		config_item_put(&child_group->cg_item);
	}
}

void cxi_configfs_cleanup(struct cass_dev *hw)
{
	unregister_all_sub_config_groups_recur(&hw->cfg_dev_dir->group);
	configfs_unregister_group(&hw->cfg_dev_dir->group);
}

void cxi_configfs_exit(void)
{
	configfs_unregister_subsystem(&cxi_cfg_group_subsys);
}
