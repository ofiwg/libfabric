#ifndef _RXR_TP_H
#define _RXR_TP_H

#include <config.h>

#if HAVE_LTTNG

#include "rxr_tp_def.h"

#include <lttng/tracef.h>
#include <lttng/tracelog.h>

#define rxr_tracepoint(...) \
	lttng_ust_tracepoint(EFA_RDM_TP_PROV, __VA_ARGS__)

/*
 * Simple printf()-style tracepoints
 * Tracing events will be labeled `lttng_ust_tracef:*`
 */
#define rxr_tracef	lttng_ust_tracef

/* tracelog() is similar to tracef(), but with a log level param */
#define rxr_tracelog	lttng_ust_tracelog

#else

#define rxr_tracepoint(...)	while (0) {}
#define rxr_tracef(...)	while (0) {}
#define rxr_tracelog(...)	while (0) {}

#endif /* HAVE_LTTNG */

#endif /* _RXR_TP_H */
