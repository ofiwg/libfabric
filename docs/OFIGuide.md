---
layout: page
title: High Performance Network Programming with OFI
tagline: Libfabric Programmer's Guide
---
{% include JB/setup %}


Review of Sockets Communication
	Connected (TCP) communication
	Connectionless (UDP) communication
	Advantages
	Disadvantages
High-Performance Networking
	Avoiding memory copies
		Network buffers
		Resource management
	Asynchronous operations
		Interrupts and signals
		Event queues
	Direct hardware access
		Kernel bypass
		Direct data placement
Designing Interfaces for Performance
	Call setup costs
	Branches and loops
	Command formatting
	Memory footprint
		Addressing
		Communication resources
		Network Buffering
			Shared Rx queues
			Multi-receive buffers
	Optimal hardware allocation
		Shared queues
		Multiple queues
	Progress model considerations
	Multi-threading synchronization
	Ordering
		Message
		Completion
		Data
OFI Architecture
	Framework versus Provider
	Control services
	Communication services
	Completion services
	Data transfer services
	Memory registration
Object Model
Communication Model
	Connected communications
	Connectionless communications
Data Transfers
	Endpoints
		Shared Contexts
			Rx
			Tx
		Scalable Endpoints
	Message transfers
	Tagged messages
	RMA
	Atomic operations
Fabric Interfaces
	fi_info / fi_getinfo
		Capabilities
		Mode bits
		Addressing
	Fabric
		Attributes
		Accessing
	Domains
		Attributes
		Opening
		Memory registration
	Endpoints
		Active
			Enabling
		Passive
		Scalable
		Resource Bindings
		EP Attributes
		Rx Attributes
		Tx Attributes
	Completions
		CQs
			Attributes
			Reading completions
			Retrieving errors
		Counters
			Checking value
			Error reporting
	Address Vectors
		Types
		Address formats
		Insertion methods
		Sharing with other processes
	Wait and Poll Sets
		Blocking on events
			TryWait
			Wait
		Efficiently checking multiple queues
Putting It All Together
	MSG EP pingpong
	RDM EP pingpong