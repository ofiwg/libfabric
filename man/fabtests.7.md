---
layout: page
title: fabtests(7)
tagline: Fabtests Programmer's Manual
---

# NAME

Fabtests

# SYNOPSIS

Fabtests is a set of examples for fabric providers that demonstrates various features of libfabric- high-performance fabric software library.

# OVERVIEW
  
Libfabric defines sets of interface that fabric providers can support. The purpose of Fabtests examples is to demonstrate some of the major features. The goal is to familiarize users with different functionalities libfabric offers and how to use them.

The tests are divided into four categories:

## Simple

These tests are a mix of very basic tests and major features of libfabric. All of the tests except info.c are designed to run as client-server processes. A server is started first and then a client process connects to the server and performs various operations.

	/simple/fi_cq_data: A client-server example that tranfers CQ data
	/simple/fi_dgram: A basic DGRAM client-server example
	/simple/fi_dgram_waitset: A basic DGRAM client-server example that uses waitset
	/simple/fi_info: An example (non client-server) that prints fabric interface information obtained by fi_getinfo call
	/simple/fi_msg: A basic MSG client-server example
	/simple/fi_msg_pingpong: A ping-pong client-server example using MSG endpoints
	/simple/fi_msg_rma: A ping pong client-server example using RMA operations between MSG endpoints
	/simple/fi_poll: A basic RDM client-server example that uses poll
	/simple/fi_rdm: A basic RDM client-server example
	/simple/fi_rdm_atomic: An RDM ping pong client-server using atomic operations
	/simple/fi_rdm_cntr_pingpong: An RDM ping pong client-server using counters
	/simple/fi_rdm_inject_pingpong: An RDM ping pong client-server example using inject
	/simple/fi_rdm_multi_recv: An RDM ping pong client-server example using multi recv buffer
	/simple/fi_rdm_pingpong: A ping pong client-server example using RDM endpoints
	/simple/fi_rdm_rma: A ping pong client-server example using RMA operations
	/simple/fi_rdm_rma_simple: A simple RDM client-sever RMA example
	/simple/fi_rdm_shared_ctx: An RDM client-server example that uses shared context
	/simple/fi_rdm_tagged_pingpong: A ping pong client-server example using tagged messages
	/simple/fi_rdm_tagged_search: An RDM client-server example that uses tagged search
	/simple/fi_scalable_ep: An RDM client-server example with scalable endpoints
	/simple/fi_ud_pingpong: A ping-pong client-server example using DGRAM endpoints

## Unit
	 /unit/fi_eq_test
	 /unit/fi_dom_test
	 /unit/fi_av_test
	 /unit/fi_size_left_test

## Ported
	 /ported/librdmacm/fi_cmatose
	 /ported/libibverbs/fi_rc_pingpong

## Complex
	 Under development

# HOW TO RUN TESTS
(1) You need to build libfabric before running fabtests.

	export LD_LIBRARY_PATH=/path/to/libfabric/install:$LD_LIBRARY_PATH

(2) Follow README to build fabtests.

(3) All the test executables are prefixed with "fi_". For example, in order to run simple/rdm_pingpong.c, the test executable will be simple/fi_rdm_pingpong.

All the tests except fi_info in the simple folder have the following usage model and minimum options:

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

To run basic tests: simple/fi_dgram, simple/fi_msg, simple/fi_rdm, simple/fi_rdm_rma_simple

	run server: ./simple/<basic_test_name> -f <provider_name>
		e.g.	./simple/fi_dgram -f sockest
	run client: ./simple<basic_test_name> <server addr> -f <provider_name>
		e.g.	./simple/fi_dgram 192.168.0.123 -f sockets

To run non-basic tests:

	run server: ./simple/<non_basic_test_name> -f <provider_name> -s <source addr>
		e.g.	./simple/fi_msg_rma -f sockest -s 192.168.0.123
	run client: ./simple<non_basic_test_name> <server addr> -f <provider_name>
		e.g.	./simple/fi_msg_rma 192.168.0.123 -f sockets

To run tests with different options:

	run server: ./simple/fi_rdm_atomic -f psm -s 192.168.0.123 -I 1000 -S 1024
	run client: ./simple/fi_rdm_atomic 192.168.0.123 -f psm -I 1000 -S 1024

This will run the RDM example simple/fi_rdm_atomic with

	- PSM provider
	- 1000 iterations
	- 1024 bytes message size
