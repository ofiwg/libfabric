/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/* Cassini Kernel Trace Event Definition */

#undef TRACE_SYSTEM
#define TRACE_SYSTEM cass

#if !defined(_CASS_TRACE_H) || defined(TRACE_HEADER_MULTI_READ)
#define _CASS_TRACE_H

#include <linux/tracepoint.h>

TRACE_EVENT(
	cass_err,
	TP_PROTO(unsigned int csr_flag, unsigned int bit,
		 unsigned long seconds, unsigned long nanoseconds,
		 const struct flg_err_info *flg_info,
		 struct err_info *err_info,
		 u64 *cntrs_val),
	TP_ARGS(csr_flag, bit, seconds, nanoseconds, flg_info,
		err_info, cntrs_val),
	TP_STRUCT__entry(
		__field(unsigned int, csr_flag)
		__field(unsigned int, bit)
		__field(unsigned long, seconds)
		__field(unsigned long, nanoseconds)
		__field(enum c_error_class, ec)
		__dynamic_array(u64, err_info_data_0, err_info[0].count)
		__dynamic_array(u64, err_info_data_1, err_info[1].count)
		__dynamic_array(u64, err_info_data_2, err_info[2].count)
		__array(u64, cntrs_val, C_MAX_CSR_CNTRS)
	),
	TP_fast_assign(
		__entry->csr_flag = csr_flag;
		__entry->bit = bit;
		__entry->seconds = seconds;
		__entry->nanoseconds = nanoseconds;
		__entry->ec = flg_info->ec;
		memcpy(__get_dynamic_array(err_info_data_0), err_info[0].data,
		       err_info[0].count * sizeof(u64));
		memcpy(__get_dynamic_array(err_info_data_1), err_info[1].data,
		       err_info[1].count * sizeof(u64));
		memcpy(__get_dynamic_array(err_info_data_2), err_info[2].data,
		       err_info[2].count * sizeof(u64));
		memcpy(__entry->cntrs_val, cntrs_val,
		       C_MAX_CSR_CNTRS * sizeof(u64));
	),
	TP_printk("csr_flag: %u, bit: %u, ec: %u",
		  __entry->csr_flag, __entry->bit, __entry->ec)
);

#endif /* _CASS_TRACE_H */

#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH .
#undef TRACE_INCLUDE_FILE
#define TRACE_INCLUDE_FILE cass_trace

#include <trace/define_trace.h>
