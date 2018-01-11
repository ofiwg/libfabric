/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#include "mlx.h"

int mlx_fabric_close(struct fid *fid)
{
	int status;

	if (mlx_descriptor.use_ns)
		ofi_ns_stop_server (&mlx_descriptor.name_serv);

	status = ofi_fabric_close(
			container_of(fid, struct util_fabric, fabric_fid.fid));
	return status;
}

static struct fi_ops mlx_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mlx_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_fabric mlx_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = mlx_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = fi_no_trywait,
};

int mlx_ns_service_cmp(void *svc1, void *svc2)
{
	int service1 = *(int *)svc1, service2 = *(int *)svc2;
	if (service1 == FI_MLX_ANY_SERVICE ||
	    service2 == FI_MLX_ANY_SERVICE)
		return 0;
	return (service1 < service2) ?
		-1 : (service1 > service2);
}

int mlx_ns_is_service_wildcard(void *svc)
{
	return (*(int *)svc == FI_MLX_ANY_SERVICE);
}

#define MLX_IGNORED_LO_ADDR "127.0.0.1"
static char* mlx_local_host_resolve()
{
	int status;
	struct ifaddrs *ifaddr, *ifa;
	char host[NI_MAXHOST];
	char *iface = NULL;
	char *result = NULL;

	status = fi_param_get( &mlx_prov, "mlx_ns_iface",
		&iface);
	if (!status) {
		iface = NULL;
	}

	if (-1 == getifaddrs(&ifaddr)) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"Unable to resolve local host address");
		return NULL;
	}

	for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
		/*Ignore not IPv$ ifaces*/
		if ((ifa->ifa_addr == NULL) ||
				(ifa->ifa_addr->sa_family != AF_INET)) {
			continue;
		}

		if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
				host, NI_MAXHOST,
				NULL, 0, NI_NUMERICHOST) != 0) {
			host[0] = '\0';
			continue;
		}

		/*Skip loopback device*/
		if (strncmp(host, MLX_IGNORED_LO_ADDR,
				strlen(MLX_IGNORED_LO_ADDR))==0) {
			host[0] = '\0';
			continue;
		}

		/* If iface name is specified */
		if (iface && strcmp(iface, ifa->ifa_name)!=0) {
			host[0] = '\0';
			continue;
		}

		result = strdup(host);
		break;
	}
	if (result == NULL) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"No IPv4-compatible interface was found. (match mask:%s)",
			iface?iface:"*");
	}
	freeifaddrs(ifaddr);
	return result;
}

int mlx_ns_start ()
{
	if (!mlx_descriptor.localhost)
		mlx_descriptor.localhost = mlx_local_host_resolve();

	if (!mlx_descriptor.localhost) {
		FI_INFO(&mlx_prov, FI_LOG_CORE,
			"Unable to resolve local host address:\n"
			"\t - unable to start NS\n"
			"\t - Please try MLX-address format");
		return -FI_EINVAL;
	}

	mlx_descriptor.name_serv.hostname = mlx_descriptor.localhost;
	mlx_descriptor.name_serv.port = (int) mlx_descriptor.ns_port;
	mlx_descriptor.name_serv.name_len = FI_MLX_MAX_NAME_LEN;
	mlx_descriptor.name_serv.service_len = sizeof(short);
	mlx_descriptor.name_serv.service_cmp = mlx_ns_service_cmp;
	mlx_descriptor.name_serv.is_service_wildcard = mlx_ns_is_service_wildcard;

	ofi_ns_init(&mlx_descriptor.name_serv);
	ofi_ns_start_server(&mlx_descriptor.name_serv);

	return FI_SUCCESS;
}

int mlx_fabric_open(
		struct fi_fabric_attr *attr,
		struct fid_fabric **fabric,
		void *context)
{
	struct mlx_fabric *fabric_priv;
	int status;

	FI_INFO( &mlx_prov, FI_LOG_CORE, "\n" );

	if (strcmp(attr->name, FI_MLX_FABRIC_NAME))
		return -FI_ENODATA;

	fabric_priv = calloc(1, sizeof(struct mlx_fabric));
	if (!fabric_priv) {
		return -FI_ENOMEM;
	}

	status = ofi_fabric_init(&mlx_prov, &mlx_fabric_attrs, attr,
			 &(fabric_priv->u_fabric), context);
	if (status) {
		FI_INFO( &mlx_prov, FI_LOG_CORE,
			"Error in ofi_fabric_init: %d\n", status);
		free(fabric_priv);
		return status;
	}

	fabric_priv->u_fabric.fabric_fid.fid.ops = &mlx_fabric_fi_ops;
	fabric_priv->u_fabric.fabric_fid.ops = &mlx_fabric_ops;
	*fabric = &(fabric_priv->u_fabric.fabric_fid);

	if (mlx_descriptor.use_ns) {
		if(mlx_ns_start() != FI_SUCCESS) {
			free(fabric_priv);
			return status;
		}
	}

	return FI_SUCCESS;
}
