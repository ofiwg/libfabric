/*
 * Copyright (c) 2022 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>

#include "shared.h"
#include "jsmn.h"
#include "rpc.h"

static struct rpc_ctrl *ctrls;
static int ctrl_cnt;
#define MAX_RPC_CLIENTS 128
static struct rpc_client clients[MAX_RPC_CLIENTS];

static int run_child(void)
{
	int i, ret;

	printf("(%d-?) running\n", myid);
	ret = ft_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_init_fabric", ret);
		return ret;
	}

	ret = fi_av_insert(av, fi->dest_addr, 1, &server_addr, 0, NULL);
	if (ret != 1) {
		ret = -FI_EINTR;
		FT_PRINTERR("fi_av_insert", ret);
		goto free;
	}

	ret = rpc_op_exec(op_hello, NULL);
	if (ret)
		goto free;

	for (i = 0; i < ctrl_cnt && !ret; i++) {
		printf("(%d-%d) rpc op %s\n", myid, id_at_server,
		       rpc_op_str(ctrls[i].op));
		ret = rpc_op_exec(ctrls[i].op, &ctrls[i]);
	}

free:
	ft_free_res();
	return ret;
}

static bool get_uint64_val(const char *js, jsmntok_t *t, uint64_t *val)
{
	if (t->type != JSMN_PRIMITIVE)
		return false;
	return (sscanf(&js[t->start], "%lu", val) == 1);
}

static bool get_op_enum(const char *js, jsmntok_t *t, uint32_t *op)
{
	const char *str;
	size_t len;

	if (t->type != JSMN_STRING)
		return false;

	str = &js[t->start];
	len = t->end - t->start;

	if (FT_TOKEN_CHECK(str, len, "msg_req")) {
		*op = op_msg_req;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "msg_inject_req")) {
		*op = op_msg_inject_req;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "msg_resp")) {
		*op = op_msg_resp;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "tag_req")) {
		*op = op_tag_req;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "tag_resp")) {
		*op = op_tag_resp;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "read_req")) {
		*op = op_read_req;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "read_resp")) {
		*op = op_read_resp;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "write_req")) {
		*op = op_write_req;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "write_resp")) {
		*op = op_write_resp;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "sleep")) {
		*op = op_sleep;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "noop")) {
		*op = op_noop;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "goodbye")) {
		*op = op_goodbye;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "hello")) {
		*op = op_hello;
		return true;
	} else if (FT_TOKEN_CHECK(str, len, "exit")) {
		*op = op_exit;
		return true;
	}

	return false;
}

/* add_ctrl extracts a rpc_ctrl struct from information in jts[idx], a
 *          JSMN_OBJECT, and its child tokens.
 * Returns true if a valid rpc_ctrl is extracted.
 *         false otherwise.
 */
static bool add_ctrl(const char *js, int njts, jsmntok_t *jts,
		     struct rpc_ctrl *ctrl, int *idx)
{
	int oidx = *idx;
	int osize = jts[*idx].size;
	jsmntok_t *t;
	const char *ks;
	size_t len;
	int i;

	ft_assert(jts[*idx].type == JSMN_OBJECT);

	init_rpc_ctrl(ctrl);
	/* i is indexing # of key:value pairs in JSMN_OBJECT */
	for (i = 0; i < osize && *idx < njts; i++) {
		(*idx)++; /* advance to next token, expecting key token */
		t = &jts[*idx];
		if (t->type != JSMN_STRING || t->size != 1)
			goto err_out;

		ks = &js[t->start];
		len = t->end - t->start;
		if (FT_TOKEN_CHECK(ks, len, "op")) {
			(*idx)++;
			t = &jts[*idx];
			if (!get_op_enum(js, t, &ctrl->op))
				goto err_out;
		} else if (FT_TOKEN_CHECK(ks, len, "size")) {
			(*idx)++;
			t = &jts[*idx];
			if (!get_uint64_val(js, t, &ctrl->size))
				goto err_out;
		} else if (FT_TOKEN_CHECK(ks, len, "offset")) {
			(*idx)++;
			t = &jts[*idx];
			if (!get_uint64_val(js, t, &ctrl->offset))
				goto err_out;
		} else if (FT_TOKEN_CHECK(ks, len, "ms")) {
			(*idx)++;
			t = &jts[*idx];
			if (!get_uint64_val(js, t, &ctrl->size))
				goto err_out;
		} else {
			goto err_out;
		}
	}

	/* op is rquired for rpc_ctrl to be valid */
	if (ctrl->op == op_last)
		goto err_out;
	return true;

err_out:
	printf("Invalid JSON entry: %.*s\n",
		jts[oidx].end - jts[oidx].start,
		&js[jts[oidx].start]);
	init_rpc_ctrl(ctrl);
	return false;
}

/* read and parse control file */
static int init_ctrls(const char *ctrlfile)
{
	FILE *ctrl_f;
	struct stat sb;
	char *js;	/* control file loaded in string */
	jsmn_parser jp;
	int njts;	/* # of JSON tokens in the control file */
	jsmntok_t *jts;
	int nobj;	/* # of JSON objects = possible rpc_ctrl entries */
	int start, i;
	int ret = 0;

	ctrl_f = fopen(ctrlfile, "r");
	if (!ctrl_f) {
		FT_PRINTERR("fopen", -errno);
		return -errno;
	}

	if (stat(ctrlfile, &sb)) {
		FT_PRINTERR("stat", -errno);
		return -errno;
	}

	js = malloc(sb.st_size + 1);
	if (!js) {
		ret = -FI_ENOMEM;
		goto no_mem_out;
	}

	if (fread(js, sb.st_size, 1, ctrl_f) != 1) {
		ret = -FI_EINVAL;
		goto read_err_out;
	}
	js[sb.st_size] = 0;

	/* get # of tokens, allcoate memory and parse JSON */
	jsmn_init(&jp);
	njts = jsmn_parse(&jp, js, sb.st_size, NULL, 0);
	if (njts < 0) {
		ret = -FI_EINVAL;
		goto read_err_out;
	}

	jts = malloc(sizeof(jsmntok_t) * njts);
	if (!jts) {
		ret = -FI_ENOMEM;
		goto read_err_out;
	}

	jsmn_init(&jp);
	if (jsmn_parse(&jp, js, sb.st_size, jts, njts) != njts) {
		ret = -FI_EINVAL;
		goto parse_err_out;
	}

	/* find the first JSON array bypassing comments at the top */
	for (start = 0; start < njts && jts[start].type != JSMN_ARRAY; start++)
		;
	if (start == njts) {
		ret = -FI_EINVAL;
		goto parse_err_out;
	}

	/* count # of JSMN_OBJECT which is # of potential rpc_ctrl entries */
	for (i = start, nobj = 0; i < njts; i++)
		if  (jts[i].type == JSMN_OBJECT)
			nobj++;

	if (nobj <= 0) {
		ret = -FI_EINVAL;
		goto parse_err_out;
	}

	ctrls = malloc(sizeof(struct rpc_ctrl) * nobj);
	if (!ctrls) {
		ret = -FI_ENOMEM;
		goto parse_err_out;
	}

	/* extract rpc_ctrl structs from tokens */
	for (ctrl_cnt = 0; start < njts; start++) {
		if (jts[start].type != JSMN_OBJECT)
			continue;

		if (add_ctrl(js, njts, jts, &ctrls[ctrl_cnt], &start))
			ctrl_cnt++;
	}

	free(jts);
	free(js);
	fclose(ctrl_f);

	if (ctrl_cnt <= 0) {
		free(ctrls);
		ctrls = NULL;
		return -FI_EINVAL;
	}
	return 0;

parse_err_out:
	free(jts);
read_err_out:
	free(js);
no_mem_out:
	fclose(ctrl_f);
	return ret;
}

static void free_ctrls(void)
{
	free(ctrls);
	ctrls = NULL;
}

static int run_parent(const char *ctrlfile)
{
	pid_t pid;
	int i, ret;

	if (!ctrlfile)
		return -FI_ENOENT;

	printf("Starting rpc client(s)\n");
	ret = init_ctrls(ctrlfile);
	if (ret)
		return ret;

	for (i = 0; i < opts.iterations; i++) {
		/* If there's only 1 client, run it from this process.  This
		 * greatly helps with debugging.
		 */
		if (opts.num_connections == 1) {
			ret = run_child();
			if (ret)
				goto free;
			continue;
		}

		for (myid = 0; myid < (uint32_t) opts.num_connections; myid++) {
			if (clients[myid].pid) {
				pid = waitpid(clients[myid].pid, NULL, 0);
				if (pid < 0)
					FT_PRINTERR("waitpid", -errno);
				clients[myid].pid = 0;
			}

			ret = fork();
			if (!ret)
				return run_child();
			if (ret < 0) {
				ret = -errno;
				goto free;
			}

			clients[myid].pid = ret;
		}
	}

	for (myid = 0; myid < (uint32_t) opts.num_connections; myid++) {
		if (clients[myid].pid) {
			pid = waitpid(clients[myid].pid, NULL, 0);
			if (pid < 0)
				FT_PRINTERR("waitpid", -errno);
			clients[myid].pid = 0;
		}
	}

free:
	free_ctrls();
	return ret;
}

#define SERVER_THREAD_COUNT 32U

int main(int argc, char **argv)
{
	char *ctrlfile = NULL;
	int op, ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SKIP_MSG_ALLOC | FT_OPT_SKIP_ADDR_EXCH;
	opts.mr_mode = FI_MR_PROV_KEY | FI_MR_ALLOCATED | FI_MR_ENDPOINT |
		       FI_MR_VIRT_ADDR | FI_MR_LOCAL | FI_MR_HMEM;
	opts.iterations = 1;
	opts.num_connections = 16;
	opts.comp_method = FT_COMP_WAIT_FD;
	opts.av_size = MAX_RPC_CLIENTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "u:h" CS_OPTS INFO_OPTS,
				  long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parsecsopts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'u':
			ctrlfile = optarg;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "An RDM endpoint error stress test.");
			ft_longopts_usage();
			FT_PRINT_OPTS_USAGE("-u <test_config.json>",
				"specify test control file at client");
			fprintf(stderr, "\nExample execution:\n");
			fprintf(stderr, "  server: %s -p tcp -s 127.0.0.1\n", argv[0]);
			fprintf(stderr, "  client: %s -p tcp -u "
				"fabtests/test_configs/rdm_stress/stress.json "
				"127.0.0.1\n", argv[0]);
			return EXIT_FAILURE;
		}
	}

	if (timeout >= 0)
		rpc_timeout = timeout * 1000;
	if (optind < argc)
		opts.dst_addr = argv[optind];

	/* limit num_connections to MAX_RPC_CLIENTS */
	opts.num_connections = MIN(opts.num_connections, MAX_RPC_CLIENTS);

	hints->caps = FI_MSG | FI_TAGGED | FI_RMA;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->av_type = FI_AV_TABLE;
	hints->ep_attr->type = FI_EP_RDM;
	hints->tx_attr->inject_size = sizeof(struct rpc_hello_msg);

	if (opts.dst_addr)
		ret = run_parent(ctrlfile);
	else
		ret = rpc_run_server(SERVER_THREAD_COUNT);

	return ft_exit_code(ret);
}
