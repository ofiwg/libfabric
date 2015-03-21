---
layout: page
title: fabtests(7)
tagline: Fabtests Programmer's Manual
---

## NAME

Fabtests

##SYNOPSIS

Fabtests is a set of examples for fabric providers that demonstrates various features of Libfabric- high-performance fabric software library.

## OVERVIEW
  
Libfabric defines a sets of interface that fabric providers can support. The purpose of Fabtests examples is to demonstrate some of the major features. The goal is to familiarize the user with different functionalities Libfabric offers and how to use them.

The tests are divided into four categories:

### Simple

These tess are a mix of very basic tests and major functionalies of Libfabric. All of the tests except info.c are designed to run as client-server processes. A server is satrted first and then a client process connects to the server and performs various operations.

	/simple/cq_data.c: A client-server example that tranfers CQ data
	/simple/dgram.c: A basic DGRAM client-server example
	/simple/dgram_waitset.c: A basic DGRAM client-server example that uses waitset
	/simple/info.c: An example (non client-server) that prints fabric interface information obtained by fi_getinfo call
	/simple/msg.c: A basic MSG client-server example
	/simple/msg_pingpong.c: A ping-pong client-server example using MSG endpoints
	/simple/msg_rma.c: A ping pong client-server example using RMA operations between MSG endpoints
	/simple/poll.c: A basic RDM client-server example that uses poll
	/simple/rdm.c: A basic RDM client-server example
	/simple/rdm_atomic.c: An RDM ping pong client-server using atomic operations
	/simple/rdm_cntr_pingpong.c: An RDM ping pong client-server using counters
	/simple/rdm_inject_pingpong.c: An RDM ping pong client-server example using inject
	/simple/rdm_multi_recv.c: An RDM ping pong client-server example using multi recv buffer
	/simple/rdm_pingpong.c: A ping pong client-server example using RDM endpoints
	/simple/rdm_rma.c: A ping pong client-server example using RMA operations
	/simple/rdm_rma_simple.c: A simple RDM client-sever RMA example
	/simple/rdm_shared_ctx.c: An RDM client-server example that uses shared context
	/simple/rdm_tagged_pingpong.c: A ping pong client-server example using tagged messages
	/simple/rdm_tagged_search.c: An RDM client-server example that uses tagged search
	/simple/scalable_ep.c: An RDM client-server example with scalable endpoints
	/simple/ud_pingpong.c: A ping-pong client-server example using DGRAM endpoints

### Unit
	 /unit/eq_test.c
	 /unit/dom_test.c
	 /unit/av_test.c
	 /unit/size_left_test.c

### Ported
	 /ported/librdmacm/cmatose.c
	 /ported/libibverbs/rc_pingpong.c

### Complex
	 Under development

## HOW TO RUN TESTS
(1) You need to build libfabric before running fabtests.

	export LD_LIBRARY_PATH=/path/to/libfabric/install:$LD_LIBRARY_PATH

(2) Go to the the root directory of fabtests

	./autogen.sh
	./configure --prefix=/path/to/fabtests/install
	make && make install

(3) All the test executables are prefixed with "fi_". For example, in order to run simple/rdm_pingpong test, the test executable will be simple/fi_rdm_pingpong.

All of the tests except info.c in the simple folder has the following usage model and minimum options:

	./simple/fi_<testname> [OPTIONS]        start server
	./simple/fi_<testname> <host>     	connect to server

	Options:
		-f <provider>   specific provider name eg sockets, verbs, psm
  		-h 		display this help output

Other available options for the non-basic tests are:

	-n <domain>   domain name
   	-b <src_port> non default source port number
	-p <dst_port> non default destination port number
	-s <address>  source address
	-I <number>   number of iterations
	-S <size>     specific transfer size or 'all'
	-m            machine readable output
	-i            print hints structure and exit
	-v            display versions and exit

To run basic tests: simple/dgram.c, simple/msg.c, simple/rdm.c, simple/rdm_rma_simple.c

	run server: ./simple/<basic_test_name> -f <provider_name>
		e.g.	./simple/fi_dgram -f sockest
	run client: ./simple<basic_test_name> <server addr> -f <provider_name>
		e.g.	./simple/fi_dgram 192.168.0.123 -f sockets

To run non-basic tests:

	run server: ./simple/<non_basic_test_name> -f <provider_name> -s <source addr>
		e.g.	./simple/fi_msg_rma -f sockest -s 192.168.0.123
	run client: ./simple<non_basic_test_name> <server addr> -f <provider_name>
		e.g.	./simple/fi_msg_rma 192.168.0.123 -f sockets

To run tests with different options and log level:

	run server: FI_LOG_LEVEL=3 ./simple/fi_rdm_atomic -f psm -s 192.168.0.123 -I 1000 -S 1024
	run client: FI_LOG_LEVEL=3 ./simple/fi_rdm_atomic 192.168.0.123 -f psm -I 1000 -S 1024

This will run the RDM example simple/fi_rdm_atomic.c with

	- log level that includes error, debug and info
	- PSM provider
	- 1000 iterations
	- 1024 bytes message size
