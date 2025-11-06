/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2022 Hewlett Packard Enterprise Development LP
 */

/* CXI Service Utility */

#define _GNU_SOURCE
#include <err.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <yaml.h>

#include <libcxi.h>

static const char *name = "cxi_service";
static const char *version = "0.5.0";

static const char * const yaml_event_strs[] = {
	[YAML_NO_EVENT] = "NO_EVENT",
	[YAML_STREAM_START_EVENT] = "STREAM_START_EVENT",
	[YAML_STREAM_END_EVENT] = "STREAM_END_EVENT",
	[YAML_DOCUMENT_START_EVENT] = "DOCUMENT_START_EVENT",
	[YAML_DOCUMENT_END_EVENT] = "DOCUMENT_END_EVENT",
	[YAML_ALIAS_EVENT] = "ALIAS_EVENT",
	[YAML_SCALAR_EVENT] = "SCALAR_EVENT",
	[YAML_SEQUENCE_START_EVENT] = "SEQUENCE_START_EVENT",
	[YAML_SEQUENCE_END_EVENT] = "SEQUENCE_END_EVENT",
	[YAML_MAPPING_START_EVENT] = "MAPPING_START_EVENT",
	[YAML_MAPPING_END_EVENT] = "MAPPING_END_EVENT",
};

enum cmd_type {
	CMD_LIST = 1,
	CMD_DELETE,
	CMD_ENABLE,
	CMD_DISABLE,
	CMD_CREATE,
	CMD_UPDATE,
};

struct util_opts {
	int dev_id;
	int svc_id;
	bool list;
	bool verbose;
	bool dry_run;
	int enable;
	int lnis_per_rgid;
	struct cxil_dev *dev;
	enum cmd_type cmd;
	char *yaml_file;
};

/* Our example parser states. */
enum state_value {
	START,
	ACCEPT_KEY,
	ACCEPT_SEQUENCE,
	ACCEPT_SEQUENCE_MAPS,
	ACCEPT_VALUE,
	STOP,
};

struct parser_state {
	enum state_value state;
	int member_idx;
	int vni_idx;
	int tc_idx;
	char *key;
	struct cxi_svc_desc *desc;
	enum cxi_rsrc_type rsrc_type;
	enum cxi_svc_member_type member_type;
	bool exclusive_cp;
	uint16_t vni_min;
	uint16_t vni_max;
};

void usage(void)
{
	fprintf(stderr,
		"cxi_service - CXI Service Utility\n\n"
		"Usage: cxi_service <COMMAND> [options]\n"
		" -d --device=DEV       CXI device. Default is cxi0\n"
		" -h --help             Show this help\n"
		" -s --svc-id           Only apply commands to this specific svc_id\n"
		" -V --version          Print the version and exit\n"
		" -v --verbose          Increase verbosity\n"
		" -y --yaml-file        Path to yaml file to use with 'create' and 'update' commands\n\n"
		"Commands:\n"
		" create                Create a service\n"
		" delete                Delete the service specified by the -s flag\n"
		" disable               Disable the service specified by the -s flag\n"
		" enable                Enable the service specified by the -s flag\n"
		" list                  List all services for a device\n");
}

static void list_members(struct cxi_svc_desc *desc)
{
	int i;

	printf("   ---> Valid Members :");
	if (!desc->restricted_members) {
		printf(" All uids/gids");
	} else {
		for (i = 0; i < CXI_SVC_MAX_MEMBERS; i++) {
			if (desc->members[i].type == CXI_SVC_MEMBER_UID)
				printf(" uid=%u",
				       desc->members[i].svc_member.uid);
			else if (desc->members[i].type == CXI_SVC_MEMBER_GID)
				printf(" gid=%u",
				       desc->members[i].svc_member.gid);
		}
	}
	printf("\n");
}

static void list_vnis(struct cxi_svc_desc *desc, struct util_opts *opts)
{
	int i;
	int rc;
	uint16_t vni_min;
	uint16_t vni_max;
	bool exclusive_cp;

	printf("   VNIs               :");
	if (!desc->restricted_vnis) {
		rc = cxil_svc_get_vni_range(opts->dev,
					    desc->svc_id,
					    &vni_min,
					    &vni_max);
		if (rc) {
			printf("\n");
			return;
		}

		printf(" %d-%d", vni_min, vni_max);
		printf("\n");

		rc = cxil_svc_get_exclusive_cp(opts->dev,
					       desc->svc_id,
					       &exclusive_cp);
		if (rc)
			return;

		printf("   Exclusive CP       : %s\n",
		       exclusive_cp ? "Yes" : "No");
	} else {
		for (i = 0; i < desc->num_vld_vnis; i++)
			printf(" %d", desc->vnis[i]);
		printf("\n");
	}
}

static void list_tcs(struct cxi_svc_desc *desc)
{
	int i;

	printf("   ---> Valid TCs     :");
	if (!desc->restricted_tcs) {
		printf(" All");
	} else {
		for (i = 0; i < CXI_TC_MAX; i++) {
			if (desc->tcs[i])
				printf(" %s", cxi_tc_strs[i]);
		}
	}
	printf("\n");
}

static void list_resource_limits(struct cxi_svc_desc *desc,
				 struct cxi_rsrc_use *rsrc_use)
{
	struct cxi_rsrc_limits *limits = &desc->limits;

	if (!desc->resource_limits) {
		printf("   ---> Max           : Max\n"
		       "   ---> Reserved      : None\n"
		       "          -----------\n"
		       "          |  %-5s |\n"
		       "          -----------\n"
		       "     ACs  |  %-5u  |\n"
		       "     CTs  |  %-5u  |\n"
		       "     EQs  |  %-5u  |\n"
		       "     LEs  |  %-5u  |\n"
		       "     PTEs |  %-5u  |\n"
		       "     TGQs |  %-5u  |\n"
		       "     TXQs |  %-5u  |\n"
		       "     TLEs |  %-5u  |\n"
		       "          -----------\n",
		       "In Use",
		       rsrc_use->in_use[CXI_RSRC_TYPE_AC],
		       rsrc_use->in_use[CXI_RSRC_TYPE_CT],
		       rsrc_use->in_use[CXI_RSRC_TYPE_EQ],
		       rsrc_use->in_use[CXI_RSRC_TYPE_LE],
		       rsrc_use->in_use[CXI_RSRC_TYPE_PTE],
		       rsrc_use->in_use[CXI_RSRC_TYPE_TGQ],
		       rsrc_use->in_use[CXI_RSRC_TYPE_TXQ],
		       rsrc_use->in_use[CXI_RSRC_TYPE_TLE]);
		return;
	}

	printf("          ---------------------------------\n"
	       "          |  %-5s  |  %-5s |  %-5s |\n"
	       "          ---------------------------------\n"
	       "     ACs  |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     CTs  |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     EQs  |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     LEs  |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     PTEs |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     TGQs |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     TXQs |  %-5u  |   %-5u   |  %-5u  |\n"
	       "     TLEs |  %-5u  |   %-5u   |  %-5u  |\n"
	       "          ---------------------------------\n",
	       "Max",
	       "Reserved",
	       "In Use",
	       limits->type[CXI_RSRC_TYPE_AC].max,
	       limits->type[CXI_RSRC_TYPE_AC].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_AC],
	       limits->type[CXI_RSRC_TYPE_CT].max,
	       limits->type[CXI_RSRC_TYPE_CT].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_CT],
	       limits->type[CXI_RSRC_TYPE_EQ].max,
	       limits->type[CXI_RSRC_TYPE_EQ].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_EQ],
	       limits->type[CXI_RSRC_TYPE_LE].max,
	       limits->type[CXI_RSRC_TYPE_LE].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_LE],
	       limits->type[CXI_RSRC_TYPE_PTE].max,
	       limits->type[CXI_RSRC_TYPE_PTE].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_PTE],
	       limits->type[CXI_RSRC_TYPE_TGQ].max,
	       limits->type[CXI_RSRC_TYPE_TGQ].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_TGQ],
	       limits->type[CXI_RSRC_TYPE_TXQ].max,
	       limits->type[CXI_RSRC_TYPE_TXQ].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_TXQ],
	       limits->type[CXI_RSRC_TYPE_TLE].max,
	       limits->type[CXI_RSRC_TYPE_TLE].res,
	       rsrc_use->in_use[CXI_RSRC_TYPE_TLE]);
}

static void print_dev_resources(struct util_opts *opts)
{
	struct cxil_dev *dev = opts->dev;
	struct cxil_devinfo *info = &dev->info;

	printf("%s\n"
	       "----\n"
	       " Total Device Resources\n"
	       " ----------------------\n"
	       " ACs:  %u\n"
	       " CTs:  %u\n"
	       " EQs:  %u\n"
	       " LEs:  %u\n"
	       " PTEs: %u\n"
	       " TGQs: %u\n"
	       " TXQs: %u\n"
	       " TLEs: %u\n",
	       dev->info.device_name,
	       info->num_acs, info->num_cts,
	       info->num_eqs, info->num_les, info->num_ptes,
	       info->num_tgqs, info->num_txqs, info->num_tles);
}

static void print_descriptor(struct cxi_svc_desc *desc,
			     struct cxi_rsrc_use *rsrc_use,
			     struct util_opts *opts)
{
	int lpr = cxil_get_svc_lpr(opts->dev, desc->svc_id);

	if (lpr < 0)
		errx(1, "Couldn't get lnis_per_rgid for descriptor: %d",
		     desc->svc_id);

	printf(" --------------------------\n");
	printf(" ID: %u%s\n", desc->svc_id,
	       (desc->svc_id == CXI_DEFAULT_SVC_ID) ? " (DEFAULT)" : "");
	printf("   LNIs/RGID          : %u\n", lpr);
	printf("   Enabled            : %s\n",
	       desc->enable ? "Yes" : "No");
	printf("   System Service     : %s\n",
	       desc->is_system_svc ? "Yes" : "No");

	printf("   Restricted Members : %s\n",
	       desc->restricted_members ? "Yes" : "No");
	if (opts->verbose)
		list_members(desc);
	if (opts->verbose)
		list_vnis(desc, opts);

	printf("   Restricted TCs     : %s\n",
	       desc->restricted_tcs ? "Yes" : "No");
	if (opts->verbose)
		list_tcs(desc);

	printf("   Resource Limits    : %s\n",
	       desc->resource_limits ? "Yes" : "No");

	if (opts->verbose && rsrc_use)
		list_resource_limits(desc, rsrc_use);
}

static void list_services(struct util_opts *opts)
{
	int i, j, rc;
	struct cxil_svc_list *svc_list = NULL;
	struct cxil_svc_rsrc_list *rsrc_list = NULL;
	struct cxi_svc_desc *desc;
	struct cxi_rsrc_use *rsrc_use;

again:
	/* Get full list of descriptors */
	rc = cxil_get_svc_list(opts->dev, &svc_list);
	if (rc)
		errx(1, "Failed to get list of services: %s", strerror(-rc));

	/* Get resrouce usage for each descriptor */
	rc = cxil_get_svc_rsrc_list(opts->dev, &rsrc_list);
	if (rc)
		errx(1, "Failed to get service resource usage list: %s",
		     strerror(-rc));

	if (svc_list->count != rsrc_list->count) {
		cxil_free_svc_list(svc_list);
		cxil_free_svc_rsrc_list(rsrc_list);
		goto again;
	}

	/* Print total device resources */
	print_dev_resources(opts);

	/* Print each descriptor */
	for (i = 0; i < svc_list->count; i++) {
		rsrc_use = NULL;
		desc = &svc_list->descs[i];
		for (j = 0; j < rsrc_list->count; j++) {
			if (desc->svc_id == rsrc_list->rsrcs[j].svc_id) {
				rsrc_use = &rsrc_list->rsrcs[j];
				break;
			}
		}
		if (!rsrc_use)
			errx(1, "Couldn't find resource usage for descriptor: %d",
			     desc->svc_id);
		print_descriptor(desc, rsrc_use, opts);
	}

	cxil_free_svc_list(svc_list);
	cxil_free_svc_rsrc_list(rsrc_list);
}

/*
 * Consume yaml events generated by the libyaml parser to
 * import our data into raw c data structures.
 *
 * The expected sequence of events is roughly:
 *
 *    stream ::= STREAM-START document* STREAM-END
 *    document ::= DOCUMENT-START section* DOCUMENT-END
 *    section ::= MAPPING-START (key list) MAPPING-END
 *    list ::= SEQUENCE-START values* SEQUENCE-END     // Optional
 *    values ::= MAPPING-START (key value)* MAPPING-END
 *    key = SCALAR
 *    value = SCALAR
 *
 */
int consume_event(struct parser_state *s, yaml_event_t *event,
		  struct util_opts *opts)
{
	char *val;
	int i;
	bool vld_tc;

	if (opts->verbose)
		printf("%s\n", yaml_event_strs[event->type]);
	switch (s->state) {
	case START:
		switch (event->type) {
		case YAML_MAPPING_START_EVENT:
			if (opts->verbose)
				printf("Mapping start from start\n");
			s->state = ACCEPT_KEY;
			break;
		case YAML_SCALAR_EVENT:
			errx(1, "Unexpected event while processing yaml\n");
		case YAML_SEQUENCE_START_EVENT:
			errx(1, "Unexpected event while processing yaml\n");
		case YAML_DOCUMENT_END_EVENT:
			s->state = STOP;
			break;
		default:
			break;
		}
		break;
	case ACCEPT_KEY:
		/* Accept a Key for a key/value pair.
		 * Might be a top level section, or within a map or sequence.
		 */
		switch (event->type) {
		case YAML_SCALAR_EVENT:
			if (opts->verbose)
				printf("Key=%s\n", event->data.scalar.value);
			if ((strcmp((char *)event->data.scalar.value, "resource_limits") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "restricted_members") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "restricted_vnis") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "restricted_tcs") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "exclusive_cp") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "name") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "max") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "res") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "type") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "id") == 0) ||
			    (strcmp((char *)event->data.scalar.value, "tc") == 0)) {
				s->state = ACCEPT_VALUE;
				s->key = strdup((char *)event->data.scalar.value);
			} else if ((strcmp((char *)event->data.scalar.value, "limits") == 0) ||
				   (strcmp((char *)event->data.scalar.value, "members") == 0)) {
				s->state = ACCEPT_SEQUENCE;
			} else if ((strcmp((char *)event->data.scalar.value, "vni") == 0)) {
				if (s->vni_min || s->vni_max)
					errx(1, "vni range already specified\n");
				s->state = ACCEPT_VALUE;
				s->key = strdup((char *)event->data.scalar.value);
				s->desc->restricted_vnis = 1;
			} else if ((strcmp((char *)event->data.scalar.value, "vni_min") == 0) ||
				   (strcmp((char *)event->data.scalar.value, "vni_max") == 0)) {
				if (s->desc->vnis[0])
					errx(1, "vni list already specified\n");
				s->state = ACCEPT_VALUE;
				s->key = strdup((char *)event->data.scalar.value);
				s->desc->restricted_vnis = 0;
			} else if ((strcmp((char *)event->data.scalar.value, "vnis") == 0) ||
				   (strcmp((char *)event->data.scalar.value, "tcs") == 0)) {
				s->state = ACCEPT_KEY;
			} else {
				errx(1, "Unexpected event while processing yaml\n");
			}
			break;
		case YAML_MAPPING_START_EVENT:
		case YAML_MAPPING_END_EVENT:
			/* Finished processing a map (key/val pair) */
			s->state = ACCEPT_KEY;
			if (opts->verbose)
				printf("ACCEPT_KEY->ACCEPT_KEY\n");
			break;
		case YAML_SEQUENCE_END_EVENT:
			/* Finished processing a sequence (limits for instance */
			s->state = ACCEPT_KEY;
			if (opts->verbose)
				printf("ACCEPT_KEY->ACCEPT_KEY\n");
			break;
		case YAML_DOCUMENT_END_EVENT:
			s->state = STOP;
			break;
		default:
			errx(1, "Unexpected event while processing yaml\n");
			break;
		}
		break;
	case ACCEPT_SEQUENCE:
		/* Intermediate case for processing resource list */
		/* TODO delete or make more generic */
		switch (event->type) {
		case YAML_SEQUENCE_START_EVENT:
			s->state = ACCEPT_SEQUENCE_MAPS;
			if (opts->verbose)
				printf("ACCEPT_SEQUENCE->ACCEPT_SEQUENCE_MAPS\n");
			break;
		default:
			errx(1, "Unexpected event while processing yaml");
		}
		break;
	case ACCEPT_SEQUENCE_MAPS:
		/* Another intermediate state that we could possibly delete TODO */
		switch (event->type) {
		case YAML_MAPPING_START_EVENT:
			s->state = ACCEPT_KEY;
			if (opts->verbose)
				printf("ACCEPT_SEQUENCE_MAPS->ACCEPT_KEY\n");
			break;
		default:
			errx(1, "Unexpected event while processing yaml\n");
		}
		break;
	case ACCEPT_VALUE:
		/* Get value from stored key */
		switch (event->type) {
		case YAML_SCALAR_EVENT:
			s->state = ACCEPT_KEY;
			val = (char *)event->data.scalar.value;
			if (opts->verbose)
				printf("val=%s\n", val);
			if (strcmp(s->key, "resource_limits") == 0) {
				i = atoi(val);
				if (i < 0 || i > 1)
					errx(1, "Invalid value for 'resource_limits': %s", val);
				s->desc->resource_limits = atoi(val);
			} else if (strcmp(s->key, "restricted_members") == 0) {
				i = atoi(val);
				if (i < 0 || i > 1)
					errx(1, "Invalid value for 'restricted_members': %s", val);
				s->desc->restricted_members = atoi(val);
			} else if (strcmp(s->key, "restricted_vnis") == 0) {
				i = atoi(val);
				if (i < 0 || i > 1)
					errx(1, "Invalid value for 'restricted_vnis': %s", val);
				s->desc->restricted_vnis = atoi(val);
			} else if (strcmp(s->key, "restricted_tcs") == 0) {
				i = atoi(val);
				if (i < 0 || i > 1)
					errx(1, "Invalid value for 'restricted_tcs': %s", val);
				s->desc->restricted_tcs = atoi(val);
			} else if (strcmp(s->key, "exclusive_cp") == 0) {
				i = atoi(val);
				if (i < 0 || i > 1)
					errx(1, "Invalid value for 'exclusive_cp': %s", val);
				s->exclusive_cp = atoi(val);
			} else if (strcmp(s->key, "vni_min") == 0) {
				s->vni_min = atoi(val);
			} else if (strcmp(s->key, "vni_max") == 0) {
				s->vni_max = atoi(val);
			} else if (strcmp(s->key, "name") == 0) {
				for (i = 0; i < CXI_RSRC_TYPE_MAX; i++) {
					if (strcmp(cxi_rsrc_type_strs[i],
						   val) == 0) {
						s->rsrc_type = i;
					}
				}
			} else if (strcmp(s->key, "max") == 0) {
				s->desc->limits.type[s->rsrc_type].max = atoi(val);
			} else if (strcmp(s->key, "res") == 0) {
				s->desc->limits.type[s->rsrc_type].res = atoi(val);
			} else if (strcmp(s->key, "type") == 0) {
				if (s->member_idx >= CXI_SVC_MAX_MEMBERS)
					errx(1, "Too many Service 'Members' provided\n");
				if (strcmp(val, "uid") == 0)
					s->desc->members[s->member_idx].type = CXI_SVC_MEMBER_UID;
				else if (strcmp(val, "gid") == 0)
					s->desc->members[s->member_idx].type = CXI_SVC_MEMBER_GID;
				else
					errx(1, "Invalid input for Service Member 'type'\n");
			} else if (strcmp(s->key, "id") == 0) {
				s->desc->members[s->member_idx].svc_member.gid =
					(uid_t)(atoi(val));
				s->member_idx++;
			} else if (strcmp(s->key, "vni") == 0) {
				if (s->vni_idx >= CXI_SVC_MAX_VNIS)
					errx(1, "Too many VNIs provided\n");
				s->desc->vnis[s->vni_idx] = atoi(val);
				s->desc->num_vld_vnis++;
				s->vni_idx++;
			} else if (strcmp(s->key, "tc") == 0) {
				/* TODO temp work around. Eth shouldn't be allowed */
				if (s->tc_idx >= CXI_TC_MAX - 1)
					errx(1, "Too many TCs provided\n");
				vld_tc = false;
				for (i = 0; i < CXI_TC_MAX - 1; i++) {
					if (strcmp(cxi_tc_strs[i], val) == 0 ||
					    (i == CXI_TC_BULK_DATA &&
					     strcmp("BULK DATA", val) == 0)) {
						s->desc->tcs[i] = true;
						vld_tc = true;
						break;
					}
				}
				if (!vld_tc)
					errx(1, "Invalid TC provided: %s\n", val);
				s->tc_idx++;
			} else {
				errx(1, "Ignoring unknown key: %s\n", s->key);
			}
			free(s->key);
			break;
		default:
			errx(1, "Unexpected event while processing yaml\n");
		}
		break;
	case STOP:
	default:
		break;
	}
	return 1;
}

static void desc_from_yaml(struct cxi_svc_desc *desc,
			   struct parser_state *state,
			   struct util_opts *opts)
{
	int rc;
	int done = 0;
	int count = 0;
	FILE *file;
	yaml_parser_t parser;
	yaml_event_t event;

	state->state = START;
	state->desc = desc;

	file = fopen(opts->yaml_file, "r");
	if (!file)
		errx(1, "Couldn't open Service YAML File: %s\n", opts->yaml_file);

	rc = yaml_parser_initialize(&parser);
	if (rc != 1)
		errx(1, "Couldn't initialize yaml parser\n");

	yaml_parser_set_input_file(&parser, file);

	while (!done) {
		if (!yaml_parser_parse(&parser, &event))
			errx(1, "Error parsing yaml: %s\n", parser.problem);
		if (!consume_event(state, &event, opts))
			errx(1, "Error consuming event\n");
		done = (event.type == YAML_STREAM_END_EVENT);
		yaml_event_delete(&event);
		count++;
	}
}

static void create_service(struct cxi_svc_desc *desc,
			   struct util_opts *opts)
{
	int rc;
	int ret;
	int svc_id;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_rsrc_use rsrc_use = {};
	struct parser_state p_state = {};

	desc_from_yaml(desc, &p_state, opts);
	svc_id = cxil_alloc_svc(opts->dev, desc, &fail_info);
	if (svc_id < 0) {
		/* TODO provide more detailed info from fail_info */
		errx(1, "Failed to create service: %s\n", strerror(-svc_id));
	}

	if (!desc->restricted_vnis) {
		rc = cxil_svc_set_vni_range(opts->dev,
					    svc_id,
					    p_state.vni_min,
					    p_state.vni_max);
		if (rc) {
			ret = cxil_destroy_svc(opts->dev, svc_id);
			if (ret)
				errx(1, "Failed to destroy service with ID %d: %s\n",
				     svc_id, strerror(-ret));

			errx(1, "Failed to set vni range: %d-%d %s\n",
			     p_state.vni_min, p_state.vni_max, strerror(-rc));
		}

		if (p_state.exclusive_cp) {
			rc = cxil_svc_enable(opts->dev, svc_id, false);
			if (rc)
				errx(1, "Failed to disable service: %s\n",
					strerror(-rc));

			rc = cxil_svc_set_exclusive_cp(opts->dev, svc_id,
						       p_state.exclusive_cp);
			if (rc)
				errx(1, "Failed to set exclusive cp: %s\n",
					strerror(-rc));

			rc = cxil_svc_enable(opts->dev, svc_id, true);
			if (rc)
				errx(1, "Failed to enable service: %s\n",
					strerror(-rc));
		}
	}

	desc->svc_id = svc_id;
	opts->verbose = true;
	print_descriptor(desc, &rsrc_use, opts);
}

static void enable_service(struct cxi_svc_desc *desc,
			   struct util_opts *opts, bool enable)
{
	int rc;

	rc = cxil_svc_enable(opts->dev, opts->svc_id, enable);
	if (rc)
		errx(1, "Could not %s service %d: %s",
		     enable ? "enable" : "disable", opts->svc_id, strerror(-rc));
}

int main(int argc, char *argv[])
{
	int rc;
	long tmp;
	char *endptr;
	struct cxi_svc_desc desc = {};
	struct cxi_rsrc_use rsrc_use = {};
	struct util_opts opts = {};
	struct option long_options[] = { { "device", required_argument, 0, 'd' },
					 { "dry_run", no_argument, 0, 'D' },
					 { "dry-run", no_argument, 0, 'D' },
					 { "help", no_argument, 0, 'h' },
					 { "svc_id", required_argument, 0, 's'},
					 { "svc-id", required_argument, 0, 's'},
					 { "verbose", no_argument, 0, 'v' },
					 { "version", no_argument, 0, 'V' },
					 { "yaml_file", required_argument, 0, 'y'},
					 { "yaml-file", required_argument, 0, 'y'},
					 { NULL, 0, 0, 0 } };

	while (1) {
		int option_index = 0;
		int c = getopt_long(argc, argv, "hvd:l:s:y:V", long_options,
				    &option_index);

		if (c == -1)
			break;

		switch (c) {
		case 'h':
			usage();
			return 0;
		case 'V':
			printf("%s version: %s\n", name, version);
			exit(0);
		case 'd':
			if (strlen(optarg) < 4 || strncmp(optarg, "cxi", 3))
				errx(1, "Invalid device name: %s", optarg);
			optarg += 3;

			errno = 0;
			endptr = NULL;
			tmp = strtol(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0 ||
			    tmp < 0 || tmp > INT_MAX)
				errx(1, "Invalid device name: cxi%s", optarg);
			opts.dev_id = tmp;
			break;
		case 'l':
			errno = 0;
			endptr = NULL;
			tmp = strtol(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0 || endptr == optarg ||
			    tmp < 1 || tmp > INT_MAX)
				errx(1, "Invalid svc_id: %s", optarg);
			opts.lnis_per_rgid = tmp;
			break;
		case 'D':
			opts.dry_run = true;
			break;
		case 's':
			errno = 0;
			endptr = NULL;
			tmp = strtol(optarg, &endptr, 10);
			if (errno != 0 || *endptr != 0 || endptr == optarg ||
			    tmp < 1 || tmp > INT_MAX)
				errx(1, "Invalid svc_id: %s", optarg);
			opts.svc_id = tmp;
			break;
		case 'v':
			opts.verbose = true;
			break;
		case 'y':
			opts.yaml_file = optarg;
			break;
		default:
			usage();
			return 1;
		}
	}

	while (optind < argc) {
		if (opts.cmd)
			errx(1, "Only one command may be specified");
		if (!strcmp(argv[optind], "list")) {
			opts.cmd = CMD_LIST;
		} else if (!strcmp(argv[optind], "delete")) {
			opts.cmd = CMD_DELETE;
		} else if (!strcmp(argv[optind], "enable")) {
			opts.cmd = CMD_ENABLE;
			opts.enable = 1;
		} else if (!strcmp(argv[optind], "disable")) {
			opts.cmd = CMD_DISABLE;
			opts.enable = 0;
		} else if (!strcmp(argv[optind], "update")) {
			errx(1, "update command is deprecated");
		} else if (!strcmp(argv[optind], "create")) {
			opts.cmd = CMD_CREATE;
		} else {
			errx(1, "Unexpected argument: %s", argv[optind]);
		}
		/* Future commands will have subopts which will increment
		 * opt_ind more than once
		 */
		optind++;
	}

	/* Open Device */
	rc = cxil_open_device(opts.dev_id, &opts.dev);
	if (rc != 0)
		errx(1, "Failed to open device cxi%u: %s",
		     opts.dev_id, strerror(-rc));

	/* Check if options are compatible */
	if (opts.cmd == CMD_DELETE && !opts.svc_id)
		errx(1, "Delete command requires -s / --svc_id");
	if (opts.cmd == CMD_ENABLE && !opts.svc_id)
		errx(1, "Enable command requires -s / --svc_id");
	if (opts.cmd == CMD_DISABLE && !opts.svc_id)
		errx(1, "Disable command requires -s / --svc_id");
	if (opts.cmd == CMD_CREATE) {
		if (!opts.yaml_file)
			errx(1, "Create command requires -y / --yaml_file");
		if (opts.svc_id)
			errx(1, "Create command incompatible with -s / --svc_id");
	}

	/* Get single svc_descriptor and rsrc_use if applicable */
	if (opts.svc_id) {
		rc = cxil_get_svc(opts.dev, opts.svc_id, &desc);
		if (rc)
			errx(1, "Failed to get service_id %d: %s",
			     opts.svc_id, strerror(-rc));
		rc = cxil_get_svc_rsrc_use(opts.dev, opts.svc_id, &rsrc_use);
		if (rc)
			errx(1, "Failed to get resource usage for service_id %d: %s",
			     opts.svc_id, strerror(-rc));
	}

	/* Run one of the commands */
	switch (opts.cmd) {
	case CMD_LIST:
		if (opts.svc_id)
			print_descriptor(&desc, &rsrc_use, &opts);
		else
			list_services(&opts);
		break;
	case CMD_DELETE:
		/* Kernel will also prevent this */
		if (opts.svc_id == CXI_DEFAULT_SVC_ID)
			errx(1, "Default service cannot be deleted.");
		rc = cxil_destroy_svc(opts.dev, opts.svc_id);
		if (rc)
			errx(1, "Could not delete service %d: %s",
					opts.svc_id, strerror(-rc));
		printf("Successfully deleted service: %d\n", opts.svc_id);
		break;
	case CMD_ENABLE:
		enable_service(&desc, &opts, true);
		printf("Successfully enabled service: %d\n", opts.svc_id);
		break;
	case CMD_DISABLE:
		enable_service(&desc, &opts, false);
		printf("Successfully disabled service: %d\n", opts.svc_id);
		break;
	case CMD_CREATE:
		create_service(&desc, &opts);
		printf("\nSuccessfully created service: %d\n", desc.svc_id);
		break;
	default:
		break;
	}

	/* Close Device */
	cxil_close_device(opts.dev);
}
