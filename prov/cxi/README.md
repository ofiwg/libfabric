Overall Code Structure
======================

In general, libfabric implements its providers using a file-like system. Each
object in libfabric is represented by a fabric identifier (fid), and this
identifier generally supports an open(), a close(), and other fabric operations.
The open() generally creates the object and returns the opaque fid, and close()
destroys it.

The provider implements these functions by supplying the callback routines for
the various file-like operations. All direct libfabric calls supply the fid of
the corresponding libfabric object to the provider implementation.

The provider object structure that represents a libfabric object generally
contains the libfabric file identifier as one of its structure fields, and it
acquires the object pointer from the file identifier using the container_of()
function, specifying the provider object type and the name of the file
identifier field within the object. Note that this makes acquiring the object
context dependent: you can apply container_of() to any file identifier for any
type of object, but picking the wrong object type for the file identifier will
give you a garbage object.

cxip_X functions and objects belong to the Cassini libfabric provider. Many of
them are libfabric-only: they implement libfabric provider infrastructure
support, and do not interact with the libcxi library or Cassini hardware at all.
Other functions provide network operation support, and will maintain libcxi
objects and call libcxi functions. The libcxi functions and objects are named
cxil_X. The cxil_X objects are tailored to Cassini hardware support.

For performance, some Cassini features, such as the command queue memory, will
be exposed directly to the cxip_X (libfabric) layer, and these need a working
knowledge of the Cassini hardware operation as well as hardware-specific include
files.

Acronyms
========

-  VA   = Virtual Address (memory)
-  IOVA = I/O Virtual Address (memory)
-  PTE  = Portals Table Entry
-  LE   = Portals Table Entry List Entry
-  LNI  = (libcxi) Logical Network Interface
-  MR   = (libfabric) Memory Region
-  EQ   = (libfabric) Event Queue
-  CQ   = (libfabric) Completion Queue
-  CC   = (libfabric) Completion Counter
-  EP   = (libfabric) End Point
-  SEP  = (libfabric) Scalable End Point
-  AV   = (libfabric) Address Vector (network)

Code Namespace Conventions
==========================

- struct dlist_entry x_entry : 'x' suggests the type of dlist to which this
  dlist hook will be attached.

- struct dlist_entry x_list : 'x' suggests the type of objects that will
  be attached to this dlist.

Typical (simple) setup
======================

<pre>
  hints = fi_allocinfo(); // zeroed structure
  ...modify hints...
  fi_getinfo();		// use hints to produce fi_info structure
  fi_fabric();		// create a fabric object using fi_info
  fi_domain();		// create a domain object within fabric
  fi_endpoint();	// create one or more endpoint objects within domain
  fi_cq_open();		// create zero or more TX CQs
  fi_cq_open();		// create zero or more RX CQs
  fi_ep_bind();		// bind CQs to endpoint
  fi_av_open();		// create one or more AVs
  fi_ep_bind();		// bind AV to endpoint
  fi_av_insert();	// insert node address into AV
  fi_enable();		// enable the EP
</pre>

Thus (simple case):

<pre>
  fabric
  +-> EQ
  +-> domain
      +-> EQ
      +-> endpoint
          +-> EQ
	  +-> AV
	      +-> EQ
          +-> TX context
	      +-> CQs/CCs
	  +-> RX context
	      +-> CQs/CCs
</pre>

Supports:

- fabric supports multiple domains
- domain has one parent fabric, supports multiple EPs
- EP has one parent domain, supports one AV, multiple TX/RX contexts


Sharing:

- domain cannot be shared among multiple fabrics
- EP cannot be shared among multiple domains
- AV can be shared among multiple EPs
- TX can be shared among multiple EPs
- RX can be shared among multiple EPs


Portals Tables
==============

Each Cassini hardware device is address by a "Dev" (device) value. The Dev
address allows the Cassini device to be addressed on the network.

Each Cassini device supports multiple distinct namespaces of soft endpoints, and
each endpoint within a namespace is referenced by a distinct 17-bit value. Each
active soft endpoint is represented by a Cassini hardware Portal Table Entry
(PTE).

The namespace is identified by a VNI value. The VNI is supplied as a parameter
from outside libfabric, typically by the Workload Manager (WLM), and the kernel
drivers enforce the proper use of different VNI identifiers by the appropriate
processes. In much the same way that Virtual Addresses Tables provide separate
address spaces for the limited physical memory used by different processes, the
VNI namespaces provide separate namespaces for the limited Cassini PTEs and
other resources used by different jobs and services under the WLM.

Each VNI namespace is broken into "granules" by the implementation-specific
"granule size." Cassini supports a small set of power-of-two granule sizes. This
implementation uses 8 bits, or 256 entries as the granule size, leaving 9 bits
available to specify the index for the 512 granules within the full 17-bit index
space. The 9-bit granule index is call the Process Identifier (pid). The 8-bit
index within the granule is called the Process Identifier Index (pid_idx).

The (dev, vni, pid) triple identifies a single "granule" in a particular VNI
namespace for a specific Cassini Chip, and this "granule" is called a libcxi
Domain. A single libcxi Domain is mapped to a single libfabric Endpoint (EP).
Thus:

* libfabric EP <=> libcxi Domain <=> Cassini VNI granule

NOTE: A libcxi Domain has nothing to do with a libfabric domain.

An application -- a WLM job or service -- can thus have up to 512 distinct
libfabric EPs that all communicate with the same remote Cassini chip. The use of
these 512 EPs is up to the application or applications sharing a VNI: they are
typically used by different processes running on different CPUs (hence, "PID ==
Process ID").

Each EP has 256 different pid_idx values that it can use to communicate with on
the remote Cassini chip, and each of these is mapped to a libfabric provider
Portal Table Entry (PTE), which is mapped to a libcxi PTE, which is mapped to a
Cassini hardware PTE. This is a "receive port" on the target device.

All EP initiator operations specify (or imply) a pid_idx value, which targets
exactly one of these PTEs on the remote Cassini device.

The libfabric provider defines fixed logical functions for these PTEs, as
follows:

| PID_IDX | Usage  |
| :-----: | :----- |
| 0       | RX 0  |
| 1       | RX 1  |
| ...     | ...   |
| 15      | RX 15 |
| 16      | MR key 0 |
| 17      | MR key 1 |
| ...     | ...      |
| 254     | MR key 238 |
| 255     | Rendezvous Send |

Libfabric Standard EP RX connections use pid_idx=0. Libfabric Scalable EP RX
connections can use pid_idx=0 through pid_idx=15.

Libfabric MRs (Memory Regions) use pid_idx=16 through pid_idx=254 for RDMA
operations.

Rendezvous Send uses pid_idx=255.

In general, a PTE consists of collection of List Entries (LE), which are target
memory addresses pre-established by the remote node.


The AV System
=============

The AV system is a global cache of libfabric target addresses. It will "resolve"
potentially more natural addressing schemes into usable libfabric addresses.

The AV system contains a MAP option that allows addresses to be shared among
different processes on the same processor.


Memory Regions
==============

Memory regions allow the target to set up multiple memory locations to be used
as RDMA targets for initiator writes, with a pre-defined application convention
that "data of type A" will be present at pid_idx=16, while "data of type B" will
be present at pid_idx=17, or a separate messaging protocol used to set up these
different memory regions. Once the convention is established, the initiator can
read or write that remote memory through the network with no impact on the
target CPU performance (other than memory bus cycle contention).


RX Contexts
===========

The RX Context is a PTE wrapper object. Its core elements are:

- rx_pte  : a pointer to the PTE support structure
- rx_cmdq : a libcxi command queue for controlling Cassini
- comp    : a completion event handler that collects events from Cassini


TX Contexts
===========

The TX Context is a Cassini command queue wrapper object. Its core elements are:

- tx_cmdq : a libcxi command queue for controlling Cassini
- comp    : a completion event handler that collects events from Cassini


Endpoints
=========

A libfabric Endpoint (EP) is a wrapper object containing one or more TX/RX
contexts.

A Standard Endpoint will contain one TX context and one RX context. The TX
context provides a means for sending data to other Cassini devices. The RX
context provides a means for receiving data from other Cassini devices.

A Scalable Endpoint will contain more than one TX context, or RX context, or
both. These are two main uses for these:

- Different RX VNI endpoint addresses can be used to separate LEs intended for
different purposes into separate RX queues. For instance, there could be large
messages and small messages. These could be differentiated in a single RX queue
by using different tags, but that incurs the penalty of tag matching. By
implementing two separate RX queues, one with large buffers, and the other with
small buffers, tagging would not be necessary.

- Different TX and RX VNI endpoint addresses can use different options, such as
enabling/disabling flow control.


Domains
=======

A libfabric Domain (entirely different from a libcxi Domain) is a collection of
endpoints (EPs) that share a set of common attributes and properties, such as
authorization keys for MRs.


Fabrics
=======

A Fabric is a collection of FI network addresses that interoperate. Ethernet and
Cassini networks do not interoperate, and are two separate fabrics.


Additional References
=====================

* https://connect.us.cray.com/confluence/display/NET/Cassini+Software+Stack+Concept
* https://connect.us.cray.com/confluence/display/NET/OFI+Provider+Design
