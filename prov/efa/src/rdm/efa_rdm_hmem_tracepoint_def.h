/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/* HMEM copy tracepoints - included by main tracepoint definition file */

#include <rdma/fi_domain.h>

/* HMEM copy tracepoints */
LTTNG_UST_TRACEPOINT_ENUM(efa_rdm, hmem_iface,
    LTTNG_UST_TP_ENUM_VALUES(
        lttng_ust_field_enum_value("FI_HMEM_SYSTEM", FI_HMEM_SYSTEM)
        lttng_ust_field_enum_value("FI_HMEM_CUDA", FI_HMEM_CUDA)
        lttng_ust_field_enum_value("FI_HMEM_ROCR", FI_HMEM_ROCR)
        lttng_ust_field_enum_value("FI_HMEM_NEURON", FI_HMEM_NEURON)
        lttng_ust_field_enum_value("FI_HMEM_SYNAPSEAI", FI_HMEM_SYNAPSEAI)
    )
)

#define HMEM_COPY_COMMON_ARGS \
	int, iface, \
	void *, dest, \
	const void *, src, \
	size_t, size

#define HMEM_COPY_ARGS \
	HMEM_COPY_COMMON_ARGS, \
	uint64_t, device

#define HMEM_DEV_REG_COPY_ARGS \
	HMEM_COPY_COMMON_ARGS, \
	uint64_t, hmem_data

#define HMEM_COMMON_FIELDS \
	lttng_ust_field_enum(efa_rdm, hmem_iface, int, iface, iface) \
	lttng_ust_field_integer_hex(void *, dest, dest) \
	lttng_ust_field_integer_hex(void *, src, src) \
	lttng_ust_field_integer(size_t, size, size)

#define HMEM_COPY_FIELDS \
	HMEM_COMMON_FIELDS \
	lttng_ust_field_integer(uint64_t, device, device)

#define HMEM_DEV_REG_COPY_FIELDS \
	HMEM_COMMON_FIELDS \
	lttng_ust_field_integer_hex(uint64_t, handle, hmem_data)

LTTNG_UST_TRACEPOINT_EVENT_CLASS(EFA_RDM_TP_PROV, hmem_copy,
	LTTNG_UST_TP_ARGS(HMEM_COPY_ARGS),
	LTTNG_UST_TP_FIELDS(HMEM_COPY_FIELDS))

LTTNG_UST_TRACEPOINT_EVENT_CLASS(EFA_RDM_TP_PROV, hmem_dev_reg_copy,
	LTTNG_UST_TP_ARGS(HMEM_DEV_REG_COPY_ARGS),
	LTTNG_UST_TP_FIELDS(HMEM_DEV_REG_COPY_FIELDS))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(EFA_RDM_TP_PROV, hmem_copy, EFA_RDM_TP_PROV,
	copy_to_hmem,
	LTTNG_UST_TP_ARGS(HMEM_COPY_ARGS))
LTTNG_UST_TRACEPOINT_LOGLEVEL(EFA_RDM_TP_PROV, copy_to_hmem, LTTNG_UST_TRACEPOINT_LOGLEVEL_INFO)

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(EFA_RDM_TP_PROV, hmem_copy, EFA_RDM_TP_PROV,
	copy_from_hmem,
	LTTNG_UST_TP_ARGS(HMEM_COPY_ARGS))
LTTNG_UST_TRACEPOINT_LOGLEVEL(EFA_RDM_TP_PROV, copy_from_hmem, LTTNG_UST_TRACEPOINT_LOGLEVEL_INFO)

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(EFA_RDM_TP_PROV, hmem_dev_reg_copy, EFA_RDM_TP_PROV,
	dev_reg_copy_to_hmem,
	LTTNG_UST_TP_ARGS(HMEM_DEV_REG_COPY_ARGS))
LTTNG_UST_TRACEPOINT_LOGLEVEL(EFA_RDM_TP_PROV, dev_reg_copy_to_hmem, LTTNG_UST_TRACEPOINT_LOGLEVEL_INFO)

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(EFA_RDM_TP_PROV, hmem_dev_reg_copy, EFA_RDM_TP_PROV,
	dev_reg_copy_from_hmem,
	LTTNG_UST_TP_ARGS(HMEM_DEV_REG_COPY_ARGS))
LTTNG_UST_TRACEPOINT_LOGLEVEL(EFA_RDM_TP_PROV, dev_reg_copy_from_hmem, LTTNG_UST_TRACEPOINT_LOGLEVEL_INFO)