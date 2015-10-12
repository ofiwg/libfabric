/*
 * Copyright (c) 2015 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <string.h>

#include "fabtest.h"
#include "jsmn.h"


#define FT_CAP_MSG	FI_MSG | FI_SEND | FI_RECV
#define FT_CAP_TAGGED	FI_TAGGED | FI_SEND | FI_RECV
#define FT_CAP_RMA	FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE
#define FT_CAP_ATOMIC	FI_ATOMICS | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE

#define FT_MODE_ALL	/*FI_CONTEXT |*/ FI_LOCAL_MR /*| FI_MSG_PREFIX*/
#define FT_MODE_NONE	~0ULL

struct key_t {
	char *str;
	size_t offset;
	enum { VAL_STRING, VAL_NUM } val_type;
	int val_size;
};

static struct ft_set test_sets_default[] = {
	{
		.prov_name = "sockets",
		.test_type = {
			FT_TEST_LATENCY,
			FT_TEST_BANDWIDTH
		},
		.class_function = {
			FT_FUNC_SEND,
			FT_FUNC_SENDV,
			FT_FUNC_SENDMSG
		},
		.ep_type = {
			FI_EP_MSG,
			FI_EP_DGRAM,
			FI_EP_RDM
		},
		.av_type = {
			FI_AV_TABLE,
			FI_AV_MAP
		},
		.comp_type = {
			FT_COMP_QUEUE
		},
		.mode = {
			FT_MODE_ALL
		},
		.caps = {
			FT_CAP_MSG,
			FT_CAP_TAGGED,
//			FT_CAP_RMA,
//			FT_CAP_ATOMIC
		},
		.test_flags = FT_FLAG_QUICKTEST
	},
	{
		.prov_name = "verbs",
		.test_type = {
			FT_TEST_LATENCY,
			FT_TEST_BANDWIDTH
		},
		.class_function = {
			FT_FUNC_SEND,
			FT_FUNC_SENDV,
			FT_FUNC_SENDMSG
		},
		.ep_type = {
			FI_EP_MSG,
		},
		.comp_type = {
			FT_COMP_QUEUE
		},
		.mode = {
			FT_MODE_ALL
		},
		.caps = {
			FT_CAP_MSG,
		},
		.test_flags = FT_FLAG_QUICKTEST
	},
};

static struct ft_series test_series;

size_t sm_size_array[] = {
		1 << 0,
		1 << 1,
		1 << 2, (1 << 2) + (1 << 1),
		1 << 3, (1 << 3) + (1 << 2),
		1 << 4, (1 << 4) + (1 << 3),
		1 << 5, (1 << 5) + (1 << 4),
		1 << 6, (1 << 6) + (1 << 5),
		1 << 7, (1 << 7) + (1 << 6),
		1 << 8
};
const unsigned int sm_size_cnt = (sizeof sm_size_array / sizeof sm_size_array[0]);

size_t med_size_array[] = {
		1 <<  4,
		1 <<  5,
		1 <<  6,
		1 <<  7, (1 <<  7) + (1 <<  6),
		1 <<  8, (1 <<  8) + (1 <<  7),
		1 <<  9, (1 <<  9) + (1 <<  8),
		1 << 10, (1 << 10) + (1 <<  9),
		1 << 11, (1 << 11) + (1 << 10),
		1 << 12, (1 << 12) + (1 << 11),
		1 << 13, (1 << 13) + (1 << 12),
		1 << 14
};
const unsigned int med_size_cnt = (sizeof med_size_array / sizeof med_size_array[0]);

size_t lg_size_array[] = {
		1 <<  4,
		1 <<  5,
		1 <<  6,
		1 <<  7, (1 <<  7) + (1 <<  6),
		1 <<  8, (1 <<  8) + (1 <<  7),
		1 <<  9, (1 <<  9) + (1 <<  8),
		1 << 10, (1 << 10) + (1 <<  9),
		1 << 11, (1 << 11) + (1 << 10),
		1 << 12, (1 << 12) + (1 << 11),
		1 << 13, (1 << 13) + (1 << 12),
		1 << 14, (1 << 14) + (1 << 13),
		1 << 15, (1 << 15) + (1 << 14),
		1 << 16, (1 << 16) + (1 << 15),
		1 << 17, (1 << 17) + (1 << 16),
		1 << 18, (1 << 18) + (1 << 17),
		1 << 19, (1 << 19) + (1 << 18),
		1 << 20, (1 << 20) + (1 << 19),
		1 << 21, (1 << 21) + (1 << 20),
		1 << 22, (1 << 22) + (1 << 21),
};
const unsigned int lg_size_cnt = (sizeof lg_size_array / sizeof lg_size_array[0]);

static struct key_t keys[] = {
	{
		.str = "node",
		.offset = offsetof(struct ft_set, node),
		.val_type = VAL_STRING,
	},
	{
		.str = "service",
		.offset = offsetof(struct ft_set, service),
		.val_type = VAL_STRING,
	},
	{
		.str = "prov_name",
		.offset = offsetof(struct ft_set, prov_name),
		.val_type = VAL_STRING,
	},
	{
		.str = "test_type",
		.offset = offsetof(struct ft_set, test_type),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->test_type) / FT_MAX_TEST,
	},
	{
		.str = "class_function",
		.offset = offsetof(struct ft_set, class_function),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->class_function) / FT_MAX_FUNCTIONS,
	},
	{
		.str = "ep_type",
		.offset = offsetof(struct ft_set, ep_type),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->ep_type) / FT_MAX_EP_TYPES,
	},
	{
		.str = "av_type",
		.offset = offsetof(struct ft_set, av_type),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->av_type) / FT_MAX_AV_TYPES,
	},
	{
		.str = "comp_type",
		.offset = offsetof(struct ft_set, comp_type),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->comp_type) / FT_MAX_COMP,
	},
	{
		.str = "eq_wait_obj",
		.offset = offsetof(struct ft_set, eq_wait_obj),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->eq_wait_obj) / FT_MAX_WAIT_OBJ,
	},
	{
		.str = "cq_wait_obj",
		.offset = offsetof(struct ft_set, cq_wait_obj),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->cq_wait_obj) / FT_MAX_WAIT_OBJ,
	},
	{
		.str = "mode",
		.offset = offsetof(struct ft_set, mode),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->mode) / FT_MAX_PROV_MODES,
	},
	{
		.str = "caps",
		.offset = offsetof(struct ft_set, caps),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->caps) / FT_MAX_CAPS,
	},
	{
		.str = "test_flags",
		.offset = offsetof(struct ft_set, test_flags),
		.val_type = VAL_NUM,
		.val_size = sizeof(((struct ft_set *)0)->test_flags),
	},
};

static int ft_parse_num(char *str, struct key_t *key, void *buf)
{
	if (!strncmp(key->str, "test_type", strlen("test_type"))) {
		TEST_ENUM_SET_N_RETURN(str, FT_TEST_LATENCY, enum ft_test_type, buf);
		TEST_ENUM_SET_N_RETURN(str, FT_TEST_BANDWIDTH, enum ft_test_type, buf);
		FT_ERR("Unknown test_type\n");
	} else if (!strncmp(key->str, "class_function", strlen("class_function"))) {
		TEST_ENUM_SET_N_RETURN(str, FT_FUNC_SENDMSG, enum ft_class_function, buf);
		TEST_ENUM_SET_N_RETURN(str, FT_FUNC_SENDV, enum ft_class_function, buf);
		TEST_ENUM_SET_N_RETURN(str, FT_FUNC_SEND, enum ft_class_function, buf);
		FT_ERR("Unknown class_function\n");
	} else if (!strncmp(key->str, "ep_type", strlen("ep_type"))) {
		TEST_ENUM_SET_N_RETURN(str, FI_EP_MSG, enum fi_ep_type, buf);
		TEST_ENUM_SET_N_RETURN(str, FI_EP_DGRAM, enum fi_ep_type, buf);
		TEST_ENUM_SET_N_RETURN(str, FI_EP_RDM, enum fi_ep_type, buf);
		FT_ERR("Unknown ep_type\n");
	} else if (!strncmp(key->str, "av_type", strlen("av_type"))) {
		TEST_ENUM_SET_N_RETURN(str, FI_AV_MAP, enum fi_av_type, buf);
		TEST_ENUM_SET_N_RETURN(str, FI_AV_TABLE, enum fi_av_type, buf);
		FT_ERR("Unknown av_type\n");
	} else if (!strncmp(key->str, "caps", strlen("caps"))) {
		TEST_SET_N_RETURN(str, "FT_CAP_MSG", FT_CAP_MSG, uint64_t, buf);
		TEST_SET_N_RETURN(str, "FT_CAP_TAGGED", FT_CAP_TAGGED, uint64_t, buf);
		TEST_SET_N_RETURN(str, "FT_CAP_RMA", FT_CAP_RMA, uint64_t, buf);
		TEST_SET_N_RETURN(str, "FT_CAP_ATOMIC", FT_CAP_ATOMIC, uint64_t, buf);
		FT_ERR("Unknown caps\n");
	} else if (!strncmp(key->str, "eq_wait_obj", strlen("eq_wait_obj")) ||
		!strncmp(key->str, "cq_wait_obj", strlen("cq_wait_obj"))) {
		TEST_ENUM_SET_N_RETURN(str, FI_WAIT_NONE, enum fi_wait_obj, buf);
		TEST_ENUM_SET_N_RETURN(str, FI_WAIT_UNSPEC, enum fi_wait_obj, buf);
		TEST_ENUM_SET_N_RETURN(str, FI_WAIT_FD, enum fi_wait_obj, buf);
		TEST_ENUM_SET_N_RETURN(str, FI_WAIT_MUTEX_COND, enum fi_wait_obj, buf);
		FT_ERR("Unknown (eq/cq)_wait_obj\n");
	} else {
		TEST_ENUM_SET_N_RETURN(str, FT_COMP_QUEUE, enum ft_comp_type, buf);
		TEST_SET_N_RETURN(str, "FT_MODE_ALL", FT_MODE_ALL, uint64_t, buf);
		TEST_SET_N_RETURN(str, "FT_FLAG_QUICKTEST", FT_FLAG_QUICKTEST, uint64_t, buf);
		FT_ERR("Unknown comp_type/mode/test_flags\n");
	}

	return -1;
}

static int ft_parse_key_val(char *config, jsmntok_t *token, char *test_set)
{
	int i, parsed = 0;
	jsmntok_t *key_token = token;
	jsmntok_t *val_token = token + 1;
	struct key_t *key = NULL;
	int size = 0;

	for (i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
		if (!strncmp(config + key_token->start, keys[i].str, strlen(keys[i].str))) {
			key = &keys[i];
			parsed++;
			break;
		}
	}

	if (!key) {
		FT_ERR("Unknown key\n");
		return -1;
	}

	if (val_token->type == JSMN_STRING) {
		size = 1;
	} else if (val_token->type == JSMN_ARRAY) {
		size = val_token->size;
		val_token++;
		parsed++;
	} else {
		FT_ERR("[jsmn] Unknown token type\n");
		return -1;
	}

	for (i = 0; i < size; i++) {
		switch(key->val_type) {
		case VAL_STRING:
			memcpy(test_set + key->offset + key->val_size * i,
					config + val_token[i].start,
					val_token[i].end - val_token[i].start);
			break;
		case VAL_NUM:
			if (ft_parse_num(config + val_token[i].start, key,
					test_set + key->offset + key->val_size * i) < 0)
				return -1;
			break;
		default:
			FT_ERR("Invalid key->val_type\n");
			return -1;
		}
		parsed++;
	}

	return parsed;
}

static int ft_parse_config(char *config, int size,
		struct ft_set **test_sets_out, int *nsets)
{
	struct ft_set *test_sets;
	jsmn_parser parser;
	jsmntok_t *tokens;
	int num_tokens, num_tokens_parsed;
	int i, ret, ts_count, ts_index;

	jsmn_init(&parser);
	num_tokens = jsmn_parse(&parser, config, size, NULL, 0);
	if (num_tokens <= 0)
		return 1;

	tokens = malloc(sizeof(jsmntok_t) * num_tokens);
	if (!tokens)
		return 1;

	/* jsmn parser returns a list of JSON tokens (jsmntok_t)
	 * e.g. JSMN_OBJECT
	 * 	JSMN_STRING : <key>
	 * 	JSMN_STRING : <value>
	 * 	JSMN_STRING : <key>
	 * 	JSMN_ARRAY  : <value: array with 2 elements>
	 * 	JSMN_STRING
	 * 	JSMN_STRING
	 * 	JSMN_STRING : <key>
	 * 	JSMN_STRING : <value>
	 * In our case, JSMN_OBJECT would represent a ft_set structure. The rest 
	 * of the tokens would be treated as key-value pairs. The first JSMN_STRING 
	 * would represent a key and the next would represent a value. A value
	 * can also be an array. jsmntok_t.size would represent the length of
	 * the array.
	 */
	jsmn_init(&parser);
	ret = jsmn_parse(&parser, config, size, tokens, num_tokens);
	if (ret < 0) {
		switch (ret) {
		case JSMN_ERROR_INVAL:
			FT_ERR("[jsmn] bad token, JSON string is corrupted!\n");
			break;
		case JSMN_ERROR_NOMEM:
			FT_ERR("[jsmn] not enough tokens, JSON string is too large!\n");
			break;
		case JSMN_ERROR_PART:
			FT_ERR("[jsmn] JSON string is too short, expecting more JSON data!\n");
			break;
		default:
			FT_ERR("[jsmn] Unknown error!\n");
			break;
		}
		goto err1;
	}

	if (ret != num_tokens) {
		FT_ERR("[jsmn] Expected # of tokens: %d, Got: %d\n", num_tokens, ret);
		goto err1;
	}

	for (i = 0, ts_count = 0; i < num_tokens; i++) {
		if (tokens[i].type == JSMN_OBJECT)
			ts_count++;
	}

	test_sets = calloc(ts_count, sizeof(struct ft_set));

	for (i = 0, ts_index = -1; i < num_tokens;) {
		switch (tokens[i].type) {
		case JSMN_OBJECT:
			ts_index++;
			i++;
			break;
		case JSMN_STRING:
			num_tokens_parsed = ft_parse_key_val(config, &tokens[i],
					(char *)(test_sets + ts_index));
		        if (num_tokens_parsed <= 0)	{
				FT_ERR("Error parsing config!\n");
				goto err2;
			}
			i += num_tokens_parsed;
			break;
		default:
			FT_ERR("[jsmn] Unknown token!\n");
			goto err2;
		}
	}

	*test_sets_out = test_sets;
	*nsets = ts_count;

	free(tokens);
	return 0;
err2:
	free(test_sets);
err1:
	free(tokens);
	return 1;
}

struct ft_series *fts_load(char *filename)
{
	int nsets = 0;
	char *config;
	FILE *fp;

	if (filename) {
		int size;
		struct ft_set *test_sets = NULL;

		fp = fopen(filename, "rb");
		if (!fp) {
			FT_ERR("Unable to open file\n");
			return NULL;
		}

		fseek(fp, 0, SEEK_END);
		size = ftell(fp);
		if (size < 0) {
			FT_ERR("ftell error");
			goto err1;
		}
		fseek(fp, 0, SEEK_SET);

		config = malloc(size + 1);
		if (!config) {
			FT_ERR("Unable to allocate memory\n");
			goto err1;
		}

		if (fread(config, size, 1, fp) != 1) {
			FT_ERR("Error reading config file\n");
			goto err2;
		}

		config[size] = 0;

		if (ft_parse_config(config, size, &test_sets, &nsets)) {
			FT_ERR("Unable to parse file\n");
			goto err2;
		}

		test_series.sets = test_sets;
		test_series.nsets = nsets;
		free(config);
		fclose(fp);
	} else {
		printf("No config file given. Using default tests.\n");
		test_series.sets = test_sets_default;
		test_series.nsets = sizeof(test_sets_default) / sizeof(test_sets_default[0]);;
	}

	for (fts_start(&test_series, 0); !fts_end(&test_series, 0);
	     fts_next(&test_series))
		test_series.test_count++;
	fts_start(&test_series, 0);

	printf("Test configurations loaded: %d\n", test_series.test_count);
	return &test_series;

err2:
	free(config);
err1:
	fclose(fp);
	return NULL;
}

void fts_close(struct ft_series *series)
{
	if (series->sets != test_sets_default)
		free(series->sets);
}

void fts_start(struct ft_series *series, int index)
{
	series->cur_set = 0;
	series->cur_type = 0;
	series->cur_ep = 0;
	series->cur_av = 0;
	series->cur_comp = 0;
	series->cur_eq_wait_obj = 0;
	series->cur_cq_wait_obj = 0;
	series->cur_mode = 0;
	series->cur_caps = 0;

	series->test_index = 1;
	if (index > 1) {
		for (; !fts_end(series, index - 1); fts_next(series))
			;
	}
}

void fts_next(struct ft_series *series)
{
	struct ft_set *set;

	if (fts_end(series, 0))
		return;

	series->test_index++;
	set = &series->sets[series->cur_set];

	if (set->caps[++series->cur_caps])
		return;
	series->cur_caps = 0;

	if (set->mode[++series->cur_mode])
		return;
	series->cur_mode = 0;

	if (set->class_function[++series->cur_func])
		return;
	series->cur_func = 0;

	if (set->comp_type[++series->cur_comp])
		return;
	series->cur_comp = 0;

	if (set->eq_wait_obj[++series->cur_eq_wait_obj])
		return;
	series->cur_eq_wait_obj = 0;

	if (set->cq_wait_obj[++series->cur_cq_wait_obj])
		return;
	series->cur_cq_wait_obj = 0;

	if (set->ep_type[series->cur_ep] == FI_EP_RDM ||
	    set->ep_type[series->cur_ep] == FI_EP_DGRAM) {
		if (set->av_type[++series->cur_av])
			return;
	}
	series->cur_av = 0;

	if (set->ep_type[++series->cur_ep])
		return;
	series->cur_ep = 0;

	if (set->test_type[++series->cur_type])
		return;
	series->cur_type = 0;

	series->cur_set++;
}

int fts_end(struct ft_series *series, int index)
{
	return (series->cur_set >= series->nsets) ||
		((index > 0) && (series->test_index > index));
}

void fts_cur_info(struct ft_series *series, struct ft_info *info)
{
	static struct ft_set *set;

	memset(info, 0, sizeof *info);
	if (series->cur_set >= series->nsets)
		return;

	set = &series->sets[series->cur_set];
	info->test_type = set->test_type[series->cur_type];
	info->test_index = series->test_index;
	info->class_function = set->class_function[series->cur_func];
	info->test_flags = set->test_flags;
	info->caps = set->caps[series->cur_caps];
	info->mode = (set->mode[series->cur_mode] == FT_MODE_NONE) ?
			0 : set->mode[series->cur_mode];
	info->ep_type = set->ep_type[series->cur_ep];
	info->av_type = set->av_type[series->cur_av];
	info->comp_type = set->comp_type[series->cur_comp];
	info->eq_wait_obj = set->eq_wait_obj[series->cur_eq_wait_obj];
	info->cq_wait_obj = set->cq_wait_obj[series->cur_cq_wait_obj];

	memcpy(info->node, set->node[0] ? set->node : opts.dst_addr, FI_NAME_MAX);
	memcpy(info->service, set->service[0] ? set->service : opts.dst_port, FI_NAME_MAX);
	memcpy(info->prov_name, set->prov_name, FI_NAME_MAX);
}
