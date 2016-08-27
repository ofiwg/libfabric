---
layout: page
title: fi_pingpong(1)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}


# NAME

fi_pingpong  \- Quick and simple pingpong test for libfabric


# SYNOPSYS
```
 fi_pingpong [OPTIONS]						start server
 fi_pingpong [OPTIONS] <server address>		connect to server
```


# DESCRIPTION

fi_pingpong is a generic pingpong test for the core feature of the libfabric library: transmitting data between two processes. fi_pingpong also displays aggregated statistics after each test run, and can additionally verify data integrity upon receipt.

By default, the datagram (FI_EP_DGRAM) endpoint is used for the test, unless otherwise specified via -e.

# HOW TO RUN TESTS

Two copies of the program must be launched: first, one copy must be launched as the server. Second, another copy is launched with the address of the server.

As a client-server test, each have the following usage model:

## Start the server
```
server$ fi_pingpong
```

## Start the client
```
client$ fi_pingpong <server endpoint address>
```


# OPTIONS

The client's command line options must match those used on the server. If they do not match, the client and server may not be able to communicate properly.

## Nodes addressing

*-B \<src_port\>*
: The non-default source port number of the endpoint.

*-P \<dest_port\>*
: The non-default destination port number of the endpoint.

## Fabric

*-p \<provider_name\>*
: The name of the underlying fabric provider (e.g., sockets, psm, usnic, etc.). If a provider is not specified via the -f switch, the test will pick one from the list of available providers (as returned by fi_getinfo(3)).

*-p \<endpoint\>* where endpoint = (dgram|rdm|msg)
: The type of endpoint to be used for data messaging between the two processes.

*-d \<domain\>*
: The name of the specific domain to be used.

## Messaging

*-I \<iter\>*
: The number of iterations of the test will run.

*-S \<msg_size\>*
: The specific size of the message in bytes the test will use or 'all' to run all the default sizes.

## Utils

*-c*
: Activate data integrity checks at the receiver (note: this may have performance impact).

*-v*
: Activate output debugging (warning: highly verbose)

*-h*
: Displays help output for the pingpong test.


# USAGE EXAMPLES

## A simple example

### Server: `fi_pingpong -p <provider_name>`
`server$ fi_pingpong -p sockets`

### Client: `fi_pingpong -p <provider_name> <server_addr>`
`client$ fi_pingpong -p sockets 192.168.0.123`

## An example with various options

### Server:
`server$ fi_pingpong -p usnic -I 1000 -S 1024`

### Client:
`client$ fi_pingpong -p usnic -I 1000 -S 1024 192.168.0.123`


Specifically, this will run a pingpong test with:

	- usNIC provider
	- 1000 iterations
	- 1024 bytes message size
	- server node as 192.168.0.123

## A longer test

### Server:
`server$ fi_pingpong -p usnic -I 10000 -S all`

### Client:
`client$ fi_pingpong -p usnic -I 10000 -S all 192.168.0.123`


# DEFAULTS

There is no default provider; if a provider is not specified via the `-p` switch, the test will pick one from the list of available providers (as returned by `fi_getinfo`(3)).

If no endpoint type is specified, 'dgram' is used.

The default tested sizes are:  64, 256, 1024, 4096.

If no server address is specified, the server address is determined by the selected provider. With the current implementation of libfabric, it means that the picked address will be the first address in the list of available addresses matching the selected provider.


# OUTPUT

Each test generates data messages which are accounted for. Specifically, the displayed statistics at the end are :

 - *bytes*          : number of bytes per message sent
 - *#sent*          : number of messages (ping) sent from the client to the server
 - *#ack*           : number of replies (pong) of the server received by the client
 - *total*          : amount of memory exchanged between the processes
 - *time*           : duration of this single test
 - *MB/sec*         : throughput computed from *total* and *time*
 - *usec/xfer*      : average time for transfering a message outbound (ping or pong) in microseconds
 - *Mxfers/sec*     : average amount of transfers of message outbound per second


# SEE ALSO

[`fi_info`(1)](info.1.html),
[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html)
