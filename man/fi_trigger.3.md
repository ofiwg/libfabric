---
layout: page
title: fi_trigger(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_trigger - Triggered operations

# SYNOPSIS

```c
#include <rdma/fi_trigger.h>
```

# DESCRIPTION

Triggered operations allow an application to queue a data transfer
request that is deferred until a specified condition is met.  A typical
use is to send a message only after receiving all input data.

A triggered operation may be requested by specifying the FI_TRIGGER
flag as part of the operation.  Alternatively, an endpoint alias may
be created and configured with the FI_TRIGGER flag.  Such an endpoint
is referred to as a trigger-able endpoint.  All data transfer
operations on a trigger-able endpoint are deferred.

Any data transfer operation is potentially trigger-able, subject to
provider constraints.  Trigger-able endpoints are initialized such that
only those interfaces supported by the provider which are trigger-able
are available.

Triggered operations require that applications use struct
fi_triggered_context as their per operation context parameter.  The
use of struct fi_triggered_context replaces struct fi_context, if
required by the provider.  Although struct fi_triggered_context is not
opaque to the application, the contents of the structure may be
modified by the provider.  This structure has similar requirements as
struct fi_context.  It must be allocated by the application and remain
valid until the corresponding operation completes or is successfully
canceled.

Struct fi_triggered_context is used to specify the condition that must
be met before the triggered data transfer is initiated.  If the
condition is met when the request is made, then the data transfer may
be initiated immediately.  The format of struct fi_triggered_context
is described below.

```c
struct fi_triggered_context {
	enum fi_trigger_event   event_type;   /* trigger type */
	union {
		struct fi_trigger_threshold	threshold;
		void                *internal[3]; /* reserved */
	} trigger;
};
```

The triggered context indicates the type of event assigned to the
trigger, along with a union of trigger details that is based on the
event type.

## TRIGGER EVENTS

The following trigger events are defined.

*FI_TRIGGER_THRESHOLD*
: This indicates that the data transfer operation will be deferred
  until an event counter crosses an application specified threshold
  value.  The threshold is specified using struct
  fi_trigger_threshold:

  ```c
  struct fi_trigger_threshold {
	struct fid_cntr *cntr; /* event counter to check */
	size_t threshold;      /* threshold value */
  };
  ```

  Threshold operations are triggered in the order of the threshold
  values.  This is true even if the counter increments by a value
  greater than 1.  If two triggered operations have the same threshold,
  they will be triggered in the order in which they were submitted to
  the endpoint.

# EXPERIMENTAL TRIGGERED OPERATIONS

The following feature and description are enhancements to triggered
operation support, but should be considered experimental.  Until the
experimental tag is removed, the interfaces and data structures defined
below may change between library versions.

Experimental triggered operations allow an application to queue operations
to a deferred work queue that is associated with the domain.  Note that
the deferred work queue is a conceptual construct, rather than an
implementation requirement.  Deferred work requests consist of two
components: an event or condition that must first be met, along with an
operation to perform.

Because deferred work requests are posted directly to the domain, they
can support a broader set of conditions and operations.  Deferred
work requests are submitted using struct fi_deferred_work.  That structure,
along with any event and operation structures used to describe the work
must remain valid until the operation completes or is canceled.

```c
struct fi_deferred_work {
	struct fi_context     context;

	enum fi_trigger_event event_type;
	enum fi_trigger_op    op_type;

	union {
		struct fi_trigger_threshold *threshold;
		struct fi_trigger_completion *completion;
	} event;

	union {
		struct fi_op_msg            *msg;
		struct fi_op_tagged         *tagged;
		struct fi_op_rma            *rma;
		struct fi_op_atomic         *atomic;
		struct fi_op_fetch_atomic   *fetch_atomic;
		struct fi_op_compare_atomic *compare_atomic;
		struct fi_op_cntr           *cntr;
	} op;
};

```

Once a work request has been posted to the deferred work queue, it will
remain on the queue until its condition has been met.  It is the
responsibility of the application to detect and handle situations that
occur which could result in the condition not being met.  For example,
if a work request is dependent upon the successful completion of a data
transfer operation, which fails, then the application must cancel the
work request.  A work request with its condition met at the time of posting
will automatically have its operation initiated.

If the work request will generate a completion, the completion event
will return the context field of struct fi_deferred_work as the context
for the operation.

To submit a deferred work request, applications should use the domain's
fi_control function with command FI_QUEUE_WORK and struct fi_deferred_work
as the fi_control arg parameter.  To cancel a deferred work request, use
fi_control with command FI_CANCEL_WORK and the corresponding struct
fi_deferred_work to cancel.  The fi_control command FI_FLUSH_WORK will
cancel all queued work requests.

Deferred work requests are not acted upon by the provider until the
associated event has occurred; although, certain validation checks
may still occur when a request is submitted.  Referenced data buffers are
not read or otherwise accessed.  But the provider may validate fabric
objects, such as endpoints and counters, and that input parameters fall
within supported ranges.  If a specific request is not supported by the
provider, it will fail the operation with -FI_ENOSYS.

## EXPERIMENTAL TRIGGER EVENTS

The following trigger events are defined.

*FI_TRIGGER_THRESHOLD*
: See the above definition.

*FI_TRIGGER_COMPLETION*
: This indicates that the data transfer operation will be deferred
  until the specified operation has completed.  The operation is specified
  using struct fi_trigger_completion:

  ```c
  struct fi_trigger_completion {
	struct fi_context *context;
  };
  ```

  The context must refer to the fi_context of the posted operation.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_alias`(3)](fi_alias.3.html),
[`fi_cntr`(3)](fi_cntr.3.html)
