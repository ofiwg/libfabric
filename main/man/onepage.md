---
layout: page
title: All-in-one Man Page
tagline: Libfabric Programmer's Manual
---

{% include JB/setup %}

# NAME

fi_guide - libfabric programmer's guide

# OVERVIEW

libfabric is a communication library framework designed to meet the
performance and scalability requirements of high-performance computing
(HPC) applications. libfabric defines communication interfaces that
enable a tight semantic map between applications and underlying network
services. Specifically, libfabric software interfaces have been
co-designed with network hardware providers and application developers,
with a focus on the needs of HPC users.

This guide describes the libfabric architecture and interfaces. Due to
the length of the guide, it has been broken into multiple pages. These
sections are:

*Introduction [`fi_intro`(7)](fi_intro.7.html)*
:   This section provides insight into the motivation for the libfabric
    design and underlying networking features that are being exposed
    through the API.

*Architecture [`fi_arch`(7)](fi_arch.7.html)*
:   This describes the exposed architecture of libfabric, including the
    object-model and their related operations

*Setup [`fi_setup`(7)](fi_setup.7.html)*
:   This provides basic bootstrapping and setup for using the libfabric
    API.

{% include JB/setup %}

# NAME

fi_intro - libfabric introduction

# OVERVIEW

This introduction is part of the libfabric's programmer's guide. See
[`fi_guide`(7)](fi_guide.7.html). This section provides insight into the
motivation for the libfabric design and underlying networking features
that are being exposed through the API.

# Review of Sockets Communication

The sockets API is a widely used networking API. This guide assumes that
a reader has a working knowledge of programming to sockets. It makes
reference to socket based communications throughout in an effort to help
explain libfabric concepts and how they relate or differ from the socket
API. To be clear, the intent of this guide is not to criticize the
socket API, but reference sockets as a starting point in order to
explain certain network features or limitations. The following sections
provide a high-level overview of socket semantics for reference.

## Connected (TCP) Communication

The most widely used type of socket is SOCK_STREAM. This sort of socket
usually runs over TCP/IP, and as a result is often referred to as a
'TCP' socket. TCP sockets are connection-oriented, requiring an explicit
connection setup before data transfers can occur. A single TCP socket
can only transfer data to a single peer socket. Communicating with
multiple peers requires the use of one socket per peer.

Applications using TCP sockets are typically labeled as either a client
or server. Server applications listen for connection requests, and
accept them when they occur. Clients, on the other hand, initiate
connections to the server. In socket API terms, a server calls listen(),
and the client calls connect(). After a connection has been established,
data transfers between a client and server are similar. The following
code segments highlight the general flow for a sample client and server.
Error handling and some subtleties of the socket API are omitted for
brevity.

    /* Example server code flow to initiate listen */
    struct addrinfo *ai, hints;
    int listen_fd;

    memset(&hints, 0, sizeof hints);
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    getaddrinfo(NULL, "7471", &hints, &ai);

    listen_fd = socket(ai->ai_family, SOCK_STREAM, 0);
    bind(listen_fd, ai->ai_addr, ai->ai_addrlen);
    freeaddrinfo(ai);

    fcntl(listen_fd, F_SETFL, O_NONBLOCK);
    listen(listen_fd, 128);

In this example, the server will listen for connection requests on port
7471 across all addresses in the system. The call to getaddrinfo() is
used to form the local socket address. The node parameter is set to
NULL, which result in a wild card IP address being returned. The port is
hard-coded to 7471. The AI_PASSIVE flag signifies that the address will
be used by the listening side of the connection. That is, the address
information should be relative to the local node.

This example will work with both IPv4 and IPv6. The getaddrinfo() call
abstracts the address format away from the server, improving its
portability. Using the data returned by getaddrinfo(), the server
allocates a socket of type SOCK_STREAM, and binds the socket to port
7471.

In practice, most enterprise-level applications make use of non-blocking
sockets. This is needed for a single application thread to manage
multiple socket connections. The fcntl() command sets the listening
socket to non-blocking mode. This will affect how the server processes
connection requests (shown below). Finally, the server starts listening
for connection requests by calling listen. Until listen is called,
connection requests that arrive at the server will be rejected by the
operating system.

    /* Example client code flow to start connection */
    struct addrinfo *ai, hints;
    int client_fd;

    memset(&hints, 0, sizeof hints);
    hints.ai_socktype = SOCK_STREAM;
    getaddrinfo("10.31.20.04", "7471", &hints, &ai);

    client_fd = socket(ai->ai_family, SOCK_STREAM, 0);
    fcntl(client_fd, F_SETFL, O_NONBLOCK);

    connect(client_fd, ai->ai_addr, ai->ai_addrlen);
    freeaddrinfo(ai);

Similar to the server, the client makes use of getaddrinfo(). Since the
AI_PASSIVE flag is not specified, the given address is treated as that
of the destination. The client expects to reach the server at IP address
10.31.20.04, port 7471. For this example the address is hard-coded into
the client. More typically, the address will be given to the client via
the command line, through a configuration file, or from a service. Often
the port number will be well-known, and the client will find the server
by name, with DNS (domain name service) providing the name to address
resolution. Fortunately, the getaddrinfo call can be used to convert
host names into IP addresses.

Whether the client is given the server's network address directly or a
name which must be translated into the network address, the mechanism
used to provide this information to the client varies widely. A simple
mechanism that is commonly used is for users to provide the server's
address using a command line option. The problem of telling applications
where its peers are located increases significantly for applications
that communicate with hundreds to millions of peer processes, often
requiring a separate, dedicated application to solve. For a typical
client-server socket application, this is not an issue, so we will defer
more discussion until later.

Using the getaddrinfo() results, the client opens a socket, configures
it for non-blocking mode, and initiates the connection request. At this
point, the network stack has sent a request to the server to establish
the connection. Because the socket has been set to non-blocking, the
connect call will return immediately and not wait for the connection to
be established. As a result any attempt to send data at this point will
likely fail.

    /* Example server code flow to accept a connection */
    struct pollfd fds;
    int server_fd;

    fds.fd = listen_fd;
    fds.events = POLLIN;

    poll(&fds, -1);

    server_fd = accept(listen_fd, NULL, 0);
    fcntl(server_fd, F_SETFL, O_NONBLOCK);

Applications that use non-blocking sockets use select(), poll(), or an
equivalent such as epoll() to receive notification of when a socket is
ready to send or receive data. In this case, the server wishes to know
when the listening socket has a connection request to process. It adds
the listening socket to a poll set, then waits until a connection
request arrives (i.e. POLLIN is true). The poll() call blocks until
POLLIN is set on the socket. POLLIN indicates that the socket has data
to accept. Since this is a listening socket, the data is a connection
request. The server accepts the request by calling accept(). That
returns a new socket to the server, which is ready for data transfers.
Finally, the server sets the new socket to non-blocking mode.

    /* Example client code flow to establish a connection */
    struct pollfd fds;
    int err;
    socklen_t len;

    fds.fd = client_fd;
    fds.events = POLLOUT;

    poll(&fds, -1);

    len = sizeof err;
    getsockopt(client_fd, SOL_SOCKET, SO_ERROR, &err, &len);

The client is notified that its connection request has completed when
its connecting socket is 'ready to send data' (i.e. POLLOUT is true).
The poll() call blocks until POLLOUT is set on the socket, indicating
the connection attempt is done. Note that the connection request may
have completed with an error, and the client still needs to check if the
connection attempt was successful. That is not conveyed to the
application by the poll() call. The getsockopt() call is used to
retrieve the result of the connection attempt. If err in this example is
set to 0, then the connection attempt succeeded. The socket is now ready
to send and receive data.

After a connection has been established, the process of sending or
receiving data is the same for both the client and server. The examples
below differ only by name of the socket variable used by the client or
server application.

    /* Example of client sending data to server */
    struct pollfd fds;
    size_t offset, size, ret;
    char buf[4096];

    fds.fd = client_fd;
    fds.events = POLLOUT;

    size = sizeof(buf);
    for (offset = 0; offset < size; ) {
        poll(&fds, -1);

        ret = send(client_fd, buf + offset, size - offset, 0);
        offset += ret;
    }

Network communication involves buffering of data at both the sending and
receiving sides of the connection. TCP uses a credit based scheme to
manage flow control to ensure that there is sufficient buffer space at
the receive side of a connection to accept incoming data. This flow
control is hidden from the application by the socket API. As a result,
stream based sockets may not transfer all the data that the application
requests to send as part of a single operation.

In this example, the client maintains an offset into the buffer that it
wishes to send. As data is accepted by the network, the offset
increases. The client then waits until the network is ready to accept
more data before attempting another transfer. The poll() operation
supports this. When the client socket is ready for data, it sets POLLOUT
to true. This indicates that send will transfer some additional amount
of data. The client issues a send() request for the remaining amount of
buffer that it wishes to transfer. If send() transfers less data than
requested, the client updates the offset, waits for the network to
become ready, then tries again.

    /* Example of server receiving data from client */
    struct pollfd fds;
    size_t offset, size, ret;
    char buf[4096];

    fds.fd = server_fd;
    fds.events = POLLIN;

    size = sizeof(buf);
    for (offset = 0; offset < size; ) {
        poll(&fds, -1);

        ret = recv(client_fd, buf + offset, size - offset, 0);
        offset += ret;
    }

The flow for receiving data is similar to that used to send it. Because
of the streaming nature of the socket, there is no guarantee that the
receiver will obtain all of the available data as part of a single call.
The server instead must wait until the socket is ready to receive data
(POLLIN), before calling receive to obtain what data is available. In
this example, the server knows to expect exactly 4 KB of data from the
client. More generally, a client and server will exchange communication
protocol headers at the start of all messages, and the header will
include the size of the message.

It is worth noting that the previous two examples are written so that
they are simple to understand. They are poorly constructed when
considering performance. In both cases, the application always precedes
a data transfer call (send or recv) with poll(). The impact is even if
the network is ready to transfer data or has data queued for receiving,
the application will always experience the latency and processing
overhead of poll(). A better approach is to call send() or recv() prior
to entering the for() loops, and only enter the loops if needed.

## Connection-less (UDP) Communication

As mentioned, TCP sockets are connection-oriented. They may be used to
communicate between exactly 2 processes. For parallel applications that
need to communicate with thousands peer processes, the overhead of
managing this many simultaneous sockets can be significant, to the point
where the application performance may decrease as more processes are
added.

To support communicating with a large number of peers, or for
applications that do not need the overhead of reliable communication,
sockets offers another commonly used socket option, SOCK_DGRAM.
Datagrams are unreliable, connectionless messages. The most common type
of SOCK_DGRAM socket runs over UDP/IP. As a result, datagram sockets are
often referred to as UDP sockets.

UDP sockets use the same socket API as that described above for TCP
sockets; however, the communication behavior differs. First, an
application using UDP sockets does not need to connect to a peer prior
to sending it a message. The destination address is specified as part of
the send operation. A second major difference is that the message is not
guaranteed to arrive at the peer. Network congestion in switches,
routers, or the remote NIC can discard the message, and no attempt will
be made to resend the message. The sender will not be notified that the
message either arrived or was dropped. Another difference between TCP
and UDP sockets is the maximum size of the transfer that is allowed. UDP
sockets limit messages to at most 64k, though in practice, applications
use a much smaller size, usually aligned to the network MTU size (for
example, 1500 bytes).

Most use of UDP sockets replace the socket send() / recv() calls with
sendto() and recvfrom().

    /* Example send to peer at given IP address and UDP port */
    struct addrinfo *ai, hints;

    memset(&hints, 0, sizeof hints);
    hints.ai_socktype = SOCK_DGRAM;
    getaddrinfo("10.31.20.04", "7471", &hints, &ai);

    ret = sendto(client_fd, buf, size, 0, ai->ai_addr, ai->ai_addrlen);

In the above example, we use getadddrinfo() to convert the given IP
address and UDP port number into a sockaddr. That is passed into the
sendto() call in order to specify the destination of the message. Note
the similarities between this flow and the TCP socket flow. The
recvfrom() call allows us to receive the address of the sender of a
message. Note that unlike streaming sockets, the entire message is
accepted by the network on success. All contents of the buf parameter,
specified by the size parameter, have been queued by the network layer.

Although not shown, the application could call poll() or an equivalent
prior to calling sendto() to ensure that the socket is ready to accept
new data. Similarly, poll() may be used prior to calling recvfrom() to
check if there is data ready to be read from the socket.

    /* Example receive a message from a peer */
    struct sockaddr_in addr;
    socklen_t addrlen;

    addrlen = sizeof(addr);
    ret = recvfrom(client_fd, buf, size, 0, &addr, &addrlen);

This example will receive any incoming message from any peer. The
address of the peer will be provided in the addr parameter. In this
case, we only provide enough space to record and IPv4 address (limited
by our use of struct sockaddr_in). Supporting an IPv6 address would
simply require passing in a larger address buffer (mapped to struct
sockaddr_in6 for example).

## Advantages

The socket API has two significant advantages. First, it is available on
a wide variety of operating systems and platforms, and works over the
vast majority of available networking hardware. It can even work for
communication between processes on the same system without any network
hardware. It is easily the de-facto networking API. This by itself makes
it appealing to use.

The second key advantage is that it is relatively easy to program to.
The importance of this should not be overlooked. Networking APIs that
offer access to higher performing features, but are difficult to program
to correctly or well, often result in lower application performance.
This is not unlike coding an application in a higher-level language such
as C or C++, versus assembly. Although writing directly to assembly
language offers the *promise* of being better performing, for the vast
majority of developers, their applications will perform better if
written in C or C++, and using an optimized compiler. Applications
should have a clear need for high-performance networking before
selecting an alternative API to sockets.

## Disadvantages

When considering the problems with the socket API as it pertains to
high-performance networking, we limit our discussion to the two most
common sockets types: streaming (TCP) and datagram (UDP).

Most applications require that network data be sent reliably. This
invariably means using a connection-oriented TCP socket. TCP sockets
transfer data as a stream of bytes. However, many applications operate
on messages. The result is that applications often insert headers that
are simply used to convert application message to / from a byte stream.
These headers consume additional network bandwidth and processing. The
streaming nature of the interface also results in the application using
loops as shown in the examples above to send and receive larger
messages. The complexity of those loops can be significant if the
application is managing sockets to hundreds or thousands of peers.

Another issue highlighted by the above examples deals with the
asynchronous nature of network traffic. When using a reliable transport,
it is not enough to place an application's data onto the network. If the
network is busy, it could drop the packet, or the data could become
corrupted during a transfer. The data must be kept until it has been
acknowledged by the peer, so that it can be resent if needed. The socket
API is defined such that the application owns the contents of its memory
buffers after a socket call returns.

As an example, if we examine the socket send() call, once send() returns
the application is free to modify its buffer. The network implementation
has a couple of options. One option is for the send call to place the
data directly onto the network. The call must then block before
returning to the user until the peer acknowledges that it received the
data, at which point send() can return. The obvious problem with this
approach is that the application is blocked in the send() call until the
network stack at the peer can process the data and generate an
acknowledgment. This can be a significant amount of time where the
application is blocked and unable to process other work, such as
responding to messages from other clients. Such an approach is not
feasible.

A better option is for the send() call to copy the application's data
into an internal buffer. The data transfer is then issued out of that
buffer, which allows retrying the operation in case of a failure. The
send() call in this case is not blocked, but all data that passes
through the network will result in a memory copy to a local buffer, even
in the absence of any errors.

Allowing immediate re-use of a data buffer is a feature of the socket
API that keeps it simple and easy to program to. However, such a feature
can potentially have a negative impact on network performance. For
network or memory limited applications, or even applications concerned
about power consumption, an alternative API may be attractive.

A slightly more hidden problem occurs in the socket APIs designed for
UDP sockets. This problem is an inefficiency in the implementation as a
result of the API design being designed for ease of use. In order for
the application to send data to a peer, it needs to provide the IP
address and UDP port number of the peer. That involves passing a
sockaddr structure to the sendto() and recvfrom() calls. However, IP
addresses are a higher- level network layer address. In order to
transfer data between systems, low-level link layer addresses are
needed, for example Ethernet addresses. The network layer must map IP
addresses to Ethernet addresses on every send operation. When scaled to
thousands of peers, that overhead on every send call can be significant.

Finally, because the socket API is often considered in conjunction with
TCP and UDP protocols, it is intentionally detached from the underlying
network hardware implementation, including NICs, switches, and routers.
Access to available network features is therefore constrained by what
the API can support.

It is worth noting here, that some operating systems support enhanced
APIs that may be used to interact with TCP and UDP sockets. For example,
Linux supports an interface known as io_uring, and Windows has an
asynchronous socket API. Those APIs can help alleviate some of the
problems described above. However, an application will still be
restricted by the features that the TCP an UDP protocols provide.

# High-Performance Networking

By analyzing the socket API in the context of high-performance
networking, we can start to see some features that are desirable for a
network API.

## Avoiding Memory Copies

The socket API implementation usually results in data copies occurring
at both the sender and the receiver. This is a trade-off between keeping
the interface easy to use, versus providing reliability. Ideally, all
memory copies would be avoided when transferring data over the network.
There are techniques and APIs that can be used to avoid memory copies,
but in practice, the cost of avoiding a copy can often be more than the
copy itself, in particular for small transfers (measured in bytes,
versus kilobytes or more).

To avoid a memory copy at the sender, we need to place the application
data directly onto the network. If we also want to avoid blocking the
sending application, we need some way for the network layer to
communicate with the application when the buffer is safe to re-use. This
would allow the original buffer to be re-used in case the data needs to
be re-transmitted. This leads us to crafting a network interface that
behaves asynchronously. The application will need to issue a request,
then receive some sort of notification when the request has completed.

Avoiding a memory copy at the receiver is more challenging. When data
arrives from the network, it needs to land into an available memory
buffer, or it will be dropped, resulting in the sender re-transmitting
the data. If we use socket recv() semantics, the only way to avoid a
copy at the receiver is for the recv() to be called before the send().
Recv() would then need to block until the data has arrived. Not only
does this block the receiver, it is impractical to use outside of an
application with a simple request-reply protocol.

Instead, what is needed is a way for the receiving application to
provide one or more buffers to the network for received data to land.
The network then needs to notify the application when data is available.
This sort of mechanism works well if the receiver does not care where in
its memory space the data is located; it only needs to be able to
process the incoming message.

As an alternative, it is possible to reverse this flow, and have the
network layer hand its buffer to the application. The application would
then be responsible for returning the buffer to the network layer when
it is done with its processing. While this approach can avoid memory
copies, it suffers from a few drawbacks. First, the network layer does
not know what size of messages to expect, which can lead to inefficient
memory use. Second, many would consider this a more difficult
programming model to use. And finally, the network buffers would need to
be mapped into the application process' memory space, which negatively
impacts performance.

In addition to processing messages, some applications want to receive
data and store it in a specific location in memory. For example, a
database may want to merge received data records into an existing table.
In such cases, even if data arriving from the network goes directly into
an application's receive buffers, it may still need to be copied into
its final location. It would be ideal if the network supported placing
data that arrives from the network into a specific memory buffer, with
the buffer determined based on the contents of the data.

### Network Buffers

Based on the problems described above, we can start to see that avoiding
memory copies depends upon the ownership of the memory buffers used for
network traffic. With socket based transports, the network buffers are
owned and managed by the networking stack. This is usually handled by
the operating system kernel. However, this results in the data
'bouncing' between the application buffers and the network buffers. By
putting the application in control of managing the network buffers, we
can avoid this overhead. The cost for doing so is additional complexity
in the application.

Note that even though we want the application to own the network
buffers, we would still like to avoid the situation where the
application implements a complex network protocol. The trade-off is that
the app provides the data buffers to the network stack, but the network
stack continues to handle things like flow control, reliability, and
segmentation and reassembly.

### Resource Management

We define resource management to mean properly allocating network
resources in order to avoid overrunning data buffers or queues. Flow
control is a common aspect of resource management. Without proper flow
control, a sender can overrun a slow or busy receiver. This can result
in dropped packets, re-transmissions, and increased network congestion.
Significant research and development has gone into implementing flow
control algorithms. Because of its complexity, it is not something that
an application developer should need to deal with. That said, there are
some applications where flow control simply falls out of the network
protocol. For example, a request-reply protocol naturally has flow
control built in.

For our purposes, we expand the definition of resource management beyond
flow control. Flow control typically only deals with available network
buffering at a peer. We also want to be concerned about having available
space in outbound data transfer queues. That is, as we issue commands to
the local NIC to send data, that those commands can be queued at the
NIC. When we consider reliability, this means tracking outstanding
requests until they have been acknowledged. Resource management will
need to ensure that we do not overflow that request queue.

Additionally, supporting asynchronous operations (described in detail
below) will introduce potential new queues. Those queues also must not
overflow.

## Asynchronous Operations

Arguably, the key feature of achieving high-performance is supporting
asynchronous operations, or the ability to overlap different
communication and communication with computation. The socket API
supports asynchronous transfers with its non-blocking mode. However,
because the API itself operates synchronously, the result is additional
data copies. For an API to be asynchronous, an application needs to be
able to submit work, then later receive some sort of notification that
the work is done. In order to avoid extra memory copies, the application
must agree not to modify its data buffers until the operation completes.

There are two main ways to notify an application that it is safe to
re-use its data buffers. One mechanism is for the network layer to
invoke some sort of callback or send a signal to the application that
the request is done. Some asynchronous APIs use this mechanism. The
drawback of this approach is that signals interrupt an application's
processing. This can negatively impact the CPU caches, plus requires
interrupt processing. Additionally, it is often difficult to develop an
application that can handle processing a signal that can occur at
anytime.

An alternative mechanism for supporting asynchronous operations is to
write events into some sort of completion queue when an operation
completes. This provides a way to indicate to an application when a data
transfer has completed, plus gives the application control over when and
how to process completed requests. For example, it can process requests
in batches to improve code locality and performance.

### Interrupts and Signals

Interrupts are a natural extension to supporting asynchronous
operations. However, when dealing with an asynchronous API, they can
negatively impact performance. Interrupts, even when directed to a
kernel agent, can interfere with application processing.

If an application has an asynchronous interface with completed
operations written into a completion queue, it is often sufficient for
the application to simply check the queue for events. As long as the
application has other work to perform, there is no need for it to block.
This alleviates the need for interrupt generation. A NIC merely needs to
write an entry into the completion queue and update a tail pointer to
signal that a request is done.

If we follow this argument, then it can be beneficial to give the
application control over when interrupts should occur and when to write
events to some sort of wait object. By having the application notify the
network layer that it will wait until a completion occurs, we can better
manage the number and type of interrupts that are generated.

### Event Queues

As outlined above, there are performance advantages to having an API
that reports completions or provides other types of notification using
an event queue. A very simple type of event queue merely tracks
completed operations. As data is received or a send completes, an entry
is written into the event queue.

## Direct Hardware Access

When discussing the network layer, most software implementations refer
to kernel modules responsible for implementing the necessary transport
and network protocols. However, if we want network latency to approach
sub-microsecond speeds, then we need to remove as much software between
the application and its access to the hardware as possible. One way to
do this is for the application to have direct access to the network
interface controller's command queues. Similarly, the NIC requires
direct access to the application's data buffers and control structures,
such as the above mentioned completion queues.

Note that when we speak about an application having direct access to
network hardware, we're referring to the application process. Naturally,
an application developer is highly unlikely to code for a specific
hardware NIC. That work would be left to some sort of network library
specifically targeting the NIC. The actual network layer, which
implements the network transport, could be part of the network library
or offloaded onto the NIC's hardware or firmware.

### Kernel Bypass

Kernel bypass is a feature that allows the application to avoid calling
into the kernel for data transfer operations. This is possible when it
has direct access to the NIC hardware. Complete kernel bypass is
impractical because of security concerns and resource management
constraints. However, it is possible to avoid kernel calls for what are
called 'fast-path' operations, such as send or receive.

For security and stability reasons, operating system kernels cannot rely
on data that comes from user space applications. As a result, even a
simple kernel call often requires acquiring and releasing locks, coupled
with data verification checks. If we can limit the effects of a poorly
written or malicious application to its own process space, we can avoid
the overhead that comes with kernel validation without impacting system
stability.

### Direct Data Placement

Direct data placement means avoiding data copies when sending and
receiving data, plus placing received data into the correct memory
buffer where needed. On a broader scale, it is part of having direct
hardware access, with the application and NIC communicating directly
with shared memory buffers and queues.

Direct data placement is often thought of by those familiar with RDMA -
remote direct memory access. RDMA is a technique that allows reading and
writing memory that belongs to a peer process that is running on a node
across the network. Advanced RDMA hardware is capable of accessing the
target memory buffers without involving the execution of the peer
process. RDMA relies on offloading the network transport onto the NIC in
order to avoid interrupting the target process.

The main advantages of supporting direct data placement is avoiding
memory copies and minimizing processing overhead.

# Designing Interfaces for Performance

We want to design a network interface that can meet the requirements
outlined above. Moreover, we also want to take into account the
performance of the interface itself. It is often not obvious how an
interface can adversely affect performance, versus performance being a
result of the underlying implementation. The following sections describe
how interface choices can impact performance. Of course, when we begin
defining the actual APIs that an application will use, we will need to
trade off raw performance for ease of use where it makes sense.

When considering performance goals for an API, we need to take into
account the target application use cases. For the purposes of this
discussion, we want to consider applications that communicate with
thousands to millions of peer processes. Data transfers will include
millions of small messages per second per peer, and large transfers that
may be up to gigabytes of data. At such extreme scales, even small
optimizations are measurable, in terms of both performance and power. If
we have a million peers sending a millions messages per second,
eliminating even a single instruction from the code path quickly
multiplies to saving billions of instructions per second from the
overall execution, when viewing the operation of the entire application.

We once again refer to the socket API as part of this discussion in
order to illustrate how an API can affect performance.

    /* Notable socket function prototypes */
    /* "control" functions */
    int socket(int domain, int type, int protocol);
    int bind(int socket, const struct sockaddr *addr, socklen_t addrlen);
    int listen(int socket, int backlog);
    int accept(int socket, struct sockaddr *addr, socklen_t *addrlen);
    int connect(int socket, const struct sockaddr *addr, socklen_t addrlen);
    int shutdown(int socket, int how);
    int close(int socket);

    /* "fast path" data operations - send only (receive calls not shown) */
    ssize_t send(int socket, const void *buf, size_t len, int flags);
    ssize_t sendto(int socket, const void *buf, size_t len, int flags,
        const struct sockaddr *dest_addr, socklen_t addrlen);
    ssize_t sendmsg(int socket, const struct msghdr *msg, int flags);
    ssize_t write(int socket, const void *buf, size_t count);
    ssize_t writev(int socket, const struct iovec *iov, int iovcnt);

    /* "indirect" data operations */
    int poll(struct pollfd *fds, nfds_t nfds, int timeout);
    int select(int nfds, fd_set *readfds, fd_set *writefds,
        fd_set *exceptfds, struct timeval *timeout);

Examining this list, there are a couple of features to note. First,
there are multiple calls that can be used to send data, as well as
multiple calls that can be used to wait for a non-blocking socket to
become ready. This will be discussed in more detail further on. Second,
the operations have been split into different groups (terminology is
ours). Control operations are those functions that an application seldom
invokes during execution. They often only occur as part of
initialization.

Data operations, on the other hand, may be called hundreds to millions
of times during an application's lifetime. They deal directly or
indirectly with transferring or receiving data over the network. Data
operations can be split into two groups. Fast path calls interact with
the network stack to immediately send or receive data. In order to
achieve high bandwidth and low latency, those operations need to be as
fast as possible. Non-fast path operations that still deal with data
transfers are those calls, that while still frequently called by the
application, are not as performance critical. For example, the select()
and poll() calls are used to block an application thread until a socket
becomes ready. Because those calls suspend the thread execution,
performance is a lesser concern. (Performance of those operations is
still of a concern, but the cost of executing the operating system
scheduler often swamps any but the most substantial performance gains.)

## Call Setup Costs

The amount of work that an application needs to perform before issuing a
data transfer operation can affect performance, especially message
rates. Obviously, the more parameters an application must push on the
stack to call a function increases its instruction count. However,
replacing stack variables with a single data structure does not help to
reduce the setup costs.

Suppose that an application wishes to send a single data buffer of a
given size to a peer. If we examine the socket API, the best fit for
such an operation is the write() call. That call takes only those values
which are necessary to perform the data transfer. The send() call is a
close second, and send() is a more natural function name for network
communication, but send() requires one extra argument over write().
Other functions are even worse in terms of setup costs. The sendmsg()
function, for example, requires that the application format a data
structure, the address of which is passed into the call. This requires
significantly more instructions from the application if done for every
data transfer.

Even though all other send functions can be replaced by sendmsg(), it is
useful to have multiple ways for the application to issue send requests.
Not only are the other calls easier to read and use (which lower
software maintenance costs), but they can also improve performance.

## Branches and Loops

When designing an API, developers rarely consider how the API impacts
the underlying implementation. However, the selection of API parameters
can require that the underlying implementation add branches or use
control loops. Consider the difference between the write() and writev()
calls. The latter passes in an array of I/O vectors, which may be
processed using a loop such as this:

    /* Sample implementation for processing an array */
    for (i = 0; i < iovcnt; i++) {
        ...
    }

In order to process the iovec array, the natural software construct
would be to use a loop to iterate over the entries. Loops result in
additional processing. Typically, a loop requires initializing a loop
control variable (e.g. i = 0), adds ALU operations (e.g. i++), and a
comparison (e.g. i \< iovcnt). This overhead is necessary to handle an
arbitrary number of iovec entries. If the common case is that the
application wants to send a single data buffer, write() is a better
option.

In addition to control loops, an API can result in the implementation
needing branches. Branches can change the execution flow of a program,
impacting processor pipe-lining techniques. Processor branch prediction
helps alleviate this issue. However, while branch prediction can be
correct nearly 100% of the time while running a micro-benchmark, such as
a network bandwidth or latency test, with more realistic network
traffic, the impact can become measurable.

We can easily see how an API can introduce branches into the code flow
if we examine the send() call. Send() takes an extra flags parameter
over the write() call. This allows the application to modify the
behavior of send(). From the viewpoint of implementing send(), the flags
parameter must be checked. In the best case, this adds one additional
check (flags are non-zero). In the worst case, every valid flag may need
a separate check, resulting in potentially dozens of checks.

Overall, the sockets API is well designed considering these performance
implications. It provides complex calls where they are needed, with
simpler functions available that can avoid some of the overhead inherent
in other calls.

## Command Formatting

The ultimate objective of invoking a network function is to transfer or
receive data from the network. In this section, we're dropping to the
very bottom of the software stack to the component responsible for
directly accessing the hardware. This is usually referred to as the
network driver, and its implementation is often tied to a specific piece
of hardware, or a series of NICs by a single hardware vendor.

In order to signal a NIC that it should read a memory buffer and copy
that data onto the network, the software driver usually needs to write
some sort of command to the NIC. To limit hardware complexity and cost,
a NIC may only support a couple of command formats. This differs from
the software interfaces that we've been discussing, where we can have
different APIs of varying complexity in order to reduce overhead. There
can be significant costs associated with formatting the command and
posting it to the hardware.

With a standard NIC, the command is formatted by a kernel driver. That
driver sits at the bottom of the network stack servicing requests from
multiple applications. It must typically format each command only after
a request has passed through the network stack.

With devices that are directly accessible by a single application, there
are opportunities to use pre-formatted command structures. The more of
the command that can be initialized prior to the application submitting
a network request, the more streamlined the process, and the better the
performance.

As an example, a NIC needs to have the destination address as part of a
send operation. If an application is sending to a single peer, that
information can be cached and be part of a pre-formatted network header.
This is only possible if the NIC driver knows that the destination will
not change between sends. The closer that the driver can be to the
application, the greater the chance for optimization. An optimal
approach is for the driver to be part of a library that executes
entirely within the application process space.

## Memory Footprint

Memory footprint concerns are most notable among high-performance
computing (HPC) applications that communicate with thousands of peers.
Excessive memory consumption impacts application scalability, limiting
the number of peers that can operate in parallel to solve problems.
There is often a trade-off between minimizing the memory footprint
needed for network communication, application performance, and ease of
use of the network interface.

As we discussed with the socket API semantics, part of the ease of using
sockets comes from the network layering copying the user's buffer into
an internal buffer belonging to the network stack. The amount of
internal buffering that's made available to the application directly
correlates with the bandwidth that an application can achieve. In
general, larger internal buffering increases network performance, with a
cost of increasing the memory footprint consumed by the application.
This memory footprint exists independent of the amount of memory
allocated directly by the application. Eliminating network buffering not
only helps with performance, but also scalability, by reducing the
memory footprint needed to support the application.

While network memory buffering increases as an application scales, it
can often be configured to a fixed size. The amount of buffering needed
is dependent on the number of active communication streams being used at
any one time. That number is often significantly lower than the total
number of peers that an application may need to communicate with. The
amount of memory required to *address* the peers, however, usually has a
linear relationship with the total number of peers.

With the socket API, each peer is identified using a struct sockaddr. If
we consider a UDP based socket application using IPv4 addresses, a peer
is identified by the following address.

    /* IPv4 socket address - with typedefs removed */
    struct sockaddr_in {
        uint16_t sin_family; /* AF_INET */
        uint16_t sin_port;
        struct {
            uint32_t sin_addr;
        } in_addr;
    };

In total, the application requires 8-bytes of addressing for each peer.
If the app communicates with a million peers, that explodes to roughly 8
MB of memory space that is consumed just to maintain the address list.
If IPv6 addressing is needed, then the requirement increases by a factor
of 4.

Luckily, there are some tricks that can be used to help reduce the
addressing memory footprint, though doing so will introduce more
instructions into code path to access the network stack. For instance,
we can notice that all addresses in the above example have the same
sin_family value (AF_INET). There's no need to store that for each
address. This potentially shrinks each address from 8 bytes to 6. (We
may be left with unaligned data, but that's a trade-off to reducing the
memory consumption). Depending on how the addresses are assigned,
further reduction may be possible. For example, if the application uses
the same set of port addresses at each node, then we can eliminate
storing the port, and instead calculate it from some base value. This
type of trick can be applied to the IP portion of the address if the app
is lucky enough to run across sequential IP addresses.

The main issue with this sort of address reduction is that it is
difficult to achieve. It requires that each application check for and
handle address compression, exposing the application to the addressing
format used by the networking stack. It should be kept in mind that
TCP/IP and UDP/IP addresses are logical addresses, not physical. When
running over Ethernet, the addresses that appear at the link layer are
MAC addresses, not IP addresses. The IP to MAC address association is
managed by the network software. We would like to provide addressing
that is simple for an application to use, but at the same time can
provide a minimal memory footprint.

## Communication Resources

We need to take a brief detour in the discussion in order to delve
deeper into the network problem and solution space. Instead of
continuing to think of a socket as a single entity, with both send and
receive capabilities, we want to consider its components separately. A
network socket can be viewed as three basic constructs: a transport
level address, a send or transmit queue, and a receive queue. Because
our discussion will begin to pivot away from pure socket semantics, we
will refer to our network 'socket' as an endpoint.

In order to reduce an application's memory footprint, we need to
consider features that fall outside of the socket API. So far, much of
the discussion has been around sending data to a peer. We now want to
focus on the best mechanisms for receiving data.

With sockets, when an app has data to receive (indicated, for example,
by a POLLIN event), we call recv(). The network stack copies the receive
data into its buffer and returns. If we want to avoid the data copy on
the receive side, we need a way for the application to post its buffers
to the network stack *before* data arrives.

Arguably, a natural way of extending the socket API to support this
feature is to have each call to recv() simply post the buffer to the
network layer. As data is received, the receive buffers are removed in
the order that they were posted. Data is copied into the posted buffer
and returned to the user. It would be noted that the size of the posted
receive buffer may be larger (or smaller) than the amount of data
received. If the available buffer space is larger, hypothetically, the
network layer could wait a short amount of time to see if more data
arrives. If nothing more arrives, the receive completes with the buffer
returned to the application.

This raises an issue regarding how to handle buffering on the receive
side. So far, with sockets we've mostly considered a streaming protocol.
However, many applications deal with messages which end up being layered
over the data stream. If they send an 8 KB message, they want the
receiver to receive an 8 KB message. Message boundaries need to be
maintained.

If an application sends and receives a fixed sized message, buffer
allocation becomes trivial. The app can post X number of buffers each of
an optimal size. However, if there is a wide mix in message sizes,
difficulties arise. It is not uncommon for an app to have 80% of its
messages be a couple hundred of bytes or less, but 80% of the total data
that it sends to be in large transfers that are, say, a megabyte or
more. Pre-posting receive buffers in such a situation is challenging.

A commonly used technique used to handle this situation is to implement
one application level protocol for smaller messages, and use a separate
protocol for transfers that are larger than some given threshold. This
would allow an application to post a bunch of smaller messages, say 4
KB, to receive data. For transfers that are larger than 4 KB, a
different communication protocol is used, possibly over a different
socket or endpoint.

### Shared Receive Queues

If an application pre-posts receive buffers to a network queue, it needs
to balance the size of each buffer posted, the number of buffers that
are posted to each queue, and the number of queues that are in use. With
a socket like approach, each socket would maintain an independent
receive queue where data is placed. If an application is using 1000
endpoints and posts 100 buffers, each 4 KB, that results in 400 MB of
memory space being consumed to receive data. (We can start to realize
that by eliminating memory copies, one of the trade offs is increased
memory consumption.) While 400 MB seems like a lot of memory, there is
less than half a megabyte allocated to a single receive queue. At
today's networking speeds, that amount of space can be consumed within
milliseconds. The result is that if only a few endpoints are in use, the
application will experience long delays where flow control will kick in
and back the transfers off.

There are a couple of observations that we can make here. The first is
that in order to achieve high scalability, we need to move away from a
connection-oriented protocol, such as streaming sockets. Secondly, we
need to reduce the number of receive queues that an application uses.

A shared receive queue is a network queue that can receive data for many
different endpoints at once. With shared receive queues, we no longer
associate a receive queue with a specific transport address. Instead
network data will target a specific endpoint address. As data arrives,
the endpoint will remove an entry from the shared receive queue, place
the data into the application's posted buffer, and return it to the
user. Shared receive queues can greatly reduce the amount of buffer
space needed by an applications. In the previous example, if a shared
receive queue were used, the app could post 10 times the number of
buffers (1000 total), yet still consume 100 times less memory (4 MB
total). This is far more scalable. The drawback is that the application
must now be aware of receive queues and shared receive queues, rather
than considering the network only at the level of a socket.

### Multi-Receive Buffers

Shared receive queues greatly improve application scalability; however,
it still results in some inefficiencies as defined so far. We've only
considered the case of posting a series of fixed sized memory buffers to
the receive queue. As mentioned, determining the size of each buffer is
challenging. Transfers larger than the fixed size require using some
other protocol in order to complete. If transfers are typically much
smaller than the fixed size, then the extra buffer space goes unused.

Again referring to our example, if the application posts 1000 buffers,
then it can only receive 1000 messages before the queue is emptied. At
data rates measured in millions of messages per second, this will
introduce stalls in the data stream. An obvious solution is to increase
the number of buffers posted. The problem is dealing with variable sized
messages, including some which are only a couple hundred bytes in
length. For example, if the average message size in our case is 256
bytes or less, then even though we've allocated 4 MB of buffer space, we
only make use of 6% of that space. The rest is wasted in order to handle
messages which may only occasionally be up to 4 KB.

A second optimization that we can make is to fill up each posted receive
buffer as messages arrive. So, instead of a 4 KB buffer being removed
from use as soon as a single 256 byte message arrives, it can instead
receive up to 16, 256 byte, messages. We refer to such a feature as
'multi-receive' buffers.

With multi-receive buffers, instead of posting a bunch of smaller
buffers, we instead post a single larger buffer, say the entire 4 MB, at
once. As data is received, it is placed into the posted buffer. Unlike
TCP streams, we still maintain message boundaries. The advantages here
are twofold. Not only is memory used more efficiently, allowing us to
receive more smaller messages at once and larger messages overall, but
we reduce the number of function calls that the application must make to
maintain its supply of available receive buffers.

When combined with shared receive queues, multi-receive buffers help
support optimal receive side buffering and processing. The main drawback
to supporting multi-receive buffers are that the application will not
necessarily know up front how many messages may be associated with a
single posted memory buffer. This is rarely a problem for applications.

## Optimal Hardware Allocation

As part of scalability considerations, we not only need to consider the
processing and memory resources of the host system, but also the
allocation and use of the NIC hardware. We've referred to network
endpoints as combination of transport addressing, transmit queues, and
receive queues. The latter two queues are often implemented as hardware
command queues. Command queues are used to signal the NIC to perform
some sort of work. A transmit queue indicates that the NIC should
transfer data. A transmit command often contains information such as the
address of the buffer to transmit, the length of the buffer, and
destination addressing data. The actual format and data contents vary
based on the hardware implementation.

NICs have limited resources. Only the most scalable, high-performance
applications likely need to be concerned with utilizing NIC hardware
optimally. However, such applications are an important and specific
focus of libfabric. Managing NIC resources is often handled by a
resource manager application, which is responsible for allocating
systems to competing applications, among other activities.

Supporting applications that wish to make optimal use of hardware
requires that hardware related abstractions be exposed to the
application. Such abstractions cannot require a specific hardware
implementation, and care must be taken to ensure that the resulting API
is still usable by developers unfamiliar with dealing with such low
level details. Exposing concepts such as shared receive queues is an
example of giving an application more control over how hardware
resources are used.

### Sharing Command Queues

By exposing the transmit and receive queues to the application, we open
the possibility for the application that makes use of multiple endpoints
to determine how those queues might be shared. We talked about the
benefits of sharing a receive queue among endpoints. The benefits of
sharing transmit queues are not as obvious.

An application that uses more addressable endpoints than there are
transmit queues will need to share transmit queues among the endpoints.
By controlling which endpoint uses which transmit queue, the application
can prioritize traffic. A transmit queue can also be configured to
optimize for a specific type of data transfer, such as large transfers
only.

From the perspective of a software API, sharing transmit or receive
queues implies exposing those constructs to the application, and
allowing them to be associated with different endpoint addresses.

### Multiple Queues

The opposite of a shared command queue are endpoints that have multiple
queues. An application that can take advantage of multiple transmit or
receive queues can increase parallel handling of messages without
synchronization constraints. Being able to use multiple command queues
through a single endpoint has advantages over using multiple endpoints.
Multiple endpoints require separate addresses, which increases memory
use. A single endpoint with multiple queues can continue to expose a
single address, while taking full advantage of available NIC resources.

## Progress Model Considerations

One aspect of the sockets programming interface that developers often
don't consider is the location of the protocol implementation. This is
usually managed by the operating system kernel. The network stack is
responsible for handling flow control messages, timing out transfers,
re-transmitting unacknowledged transfers, processing received data, and
sending acknowledgments. This processing requires that the network stack
consume CPU cycles. Portions of that processing can be done within the
context of the application thread, but much must be handled by kernel
threads dedicated to network processing.

By moving the network processing directly into the application process,
we need to be concerned with how network communication makes forward
progress. For example, how and when are acknowledgments sent? How are
timeouts and message re-transmissions handled? The progress model
defines this behavior, and it depends on how much of the network
processing has been offloaded onto the NIC.

More generally, progress is the ability of the underlying network
implementation to complete processing of an asynchronous request. In
many cases, the processing of an asynchronous request requires the use
of the host processor. For performance reasons, it may be undesirable
for the provider to allocate a thread for this purpose, which will
compete with the application thread(s). We can avoid thread context
switches if the application thread can be used to make forward progress
on requests -- check for acknowledgments, retry timed out operations,
etc. Doing so requires that the application periodically call into the
network stack.

## Ordering

Network ordering is a complex subject. With TCP sockets, data is sent
and received in the same order. Buffers are re-usable by the application
immediately upon returning from a function call. As a result, ordering
is simple to understand and use. UDP sockets complicate things slightly.
With UDP sockets, messages may be received out of order from how they
were sent. In practice, this often doesn't occur, particularly, if the
application only communicates over a local area network, such as
Ethernet.

With our evolving network API, there are situations where exposing
different order semantics can improve performance. These details will be
discussed further below.

### Messages

UDP sockets allow messages to arrive out of order because each message
is routed from the sender to the receiver independently. This allows
packets to take different network paths, to avoid congestion or take
advantage of multiple network links for improved bandwidth. We would
like to take advantage of the same features in those cases where the
application doesn't care in which order messages arrive.

Unlike UDP sockets, however, our definition of message ordering is more
subtle. UDP messages are small, MTU sized packets. In our case, messages
may be gigabytes in size. We define message ordering to indicate whether
the start of each message is processed in order or out of order. This is
related to, but separate from the order of how the message payload is
received.

An example will help clarify this distinction. Suppose that an
application has posted two messages to its receive queue. The first
receive points to a 4 KB buffer. The second receive points to a 64 KB
buffer. The sender will transmit a 4 KB message followed by a 64 KB
message. If messages are processed in order, then the 4 KB send will
match with the 4 KB received, and the 64 KB send will match with the 64
KB receive. However, if messages can be processed out of order, then the
sends and receives can mismatch, resulting in the 64 KB send being
truncated.

In this example, we're not concerned with what order the data is
received in. The 64 KB send could be broken in 64 1-KB transfers that
take different routes to the destination. So, bytes 2k-3k could be
received before bytes 1k-2k. Message ordering is not concerned with
ordering *within* a message, only *between* messages. With ordered
messages, the messages themselves need to be processed in order.

The more relaxed message ordering can be the more optimizations that the
network stack can use to transfer the data. However, the application
must be aware of message ordering semantics, and be able to select the
desired semantic for its needs. For the purposes of this section,
messages refers to transport level operations, which includes RDMA and
similar operations (some of which have not yet been discussed).

### Data

Data ordering refers to the receiving and placement of data both *within
and between* messages. Data ordering is most important to messages that
can update the same target memory buffer. For example, imagine an
application that writes a series of database records directly into a
peer memory location. Data ordering, combined with message ordering,
ensures that the data from the second write updates memory after the
first write completes. The result is that the memory location will
contain the records carried in the second write.

Enforcing data ordering between messages requires that the messages
themselves be ordered. Data ordering can also apply within a single
message, though this level of ordering is usually less important to
applications. Intra-message data ordering indicates that the data for a
single message is received in order. Some applications use this feature
to 'spin' reading the last byte of a receive buffer. Once the byte
changes, the application knows that the operation has completed and all
earlier data has been received. (Note that while such behavior is
interesting for benchmark purposes, using such a feature in this way is
strongly discouraged. It is not portable between networks or platforms.)

### Completions

Completion ordering refers to the sequence that asynchronous operations
report their completion to the application. Typically, unreliable data
transfer will naturally complete in the order that they are submitted to
a transmit queue. Each operation is transmitted to the network, with the
completion occurring immediately after. For reliable data transfers, an
operation cannot complete until it has been acknowledged by the peer.
Since ack packets can be lost or possibly take different paths through
the network, operations can be marked as completed out of order. Out of
order acks is more likely if messages can be processed out of order.

Asynchronous interfaces require that the application track their
outstanding requests. Handling out of order completions can increase
application complexity, but it does allow for optimizing network
utilization.

# lifabric Architecture

Libfabric is well architected to support the previously discussed
features. For further information on the libfabric architecture, see the
next programmer's guide section: [`fi_arch`(7)](fi_arch.7.html).

{% include JB/setup %}

# NAME

fi_arch - libfabric architecture

# OVERVIEW

Libfabric APIs define application facing communication semantics without
mandating the underlying implementation or wire protocols. It is
architected such that applications can have direct access to network
hardware without operating system intervention, but does not mandate
such an implementation. The APIs have been defined specifically to allow
multiple implementations.

The following diagram highlights the general architecture of the
interfaces exposed by libfabric.

                     Applications and Middleware
           [MPI]   [SHMEM]   [PGAS]   [Storage]   [Other]

    --------------------- libfabric API ---------------------

    /  Core  \ + /Communication\ + /  Data  \ + <Completion>
    \Services/   \    Setup    /   \Transfer/

    ----------------- libfabric Provider API ----------------

                        libfabric providers
       [TCP]   [UDP]   [Verbs]    [EFA]    [SHM]   [Other]

    ---------------------------------------------------------

         Low-level network hardware and software interfaces

Details of each libfabric component is described below.

## Core Services

libfabric can be divided between the libfabric core and providers. The
core defines defines the APIs that applications use and implements what
is referred to as discovery services. Discovery services are responsible
for identifying what hardware is available in the system, platform
features, operating system features, associated communication and
computational libraries, and so forth. Providers are optimized
implementations of the libfabric API. One of the goals of the libfabric
core is to link upper level applications and middleware with specific
providers best suited for their needs.

From the viewpoint of an application, the core libfabric services are
accessed primarily by the fi_getinfo() API. See
[`fi_getinfo`(3)](fi_getinfo.3.html).

## Providers

Unlike many libraries, the libfabric core does not implement most of the
APIs called by its users. Instead, that responsibility instead falls to
what libfabric calls providers. The bulk of the libfabric API is
implemented by each provider. When an application calls a libfabric API,
that function call is routed directly into a specific provider. This is
done using function pointers associated with specific libfabric defined
objects. The object model is describe in more detail below.

The benefit of this approach is that each provider can optimize the
libfabric defined communication semantics according to their available
network hardware, operating system, platform, and network protocol
features.

In general, each provider focuses on supporting the libfabric API over a
specific lower-level communication API or NIC. See
[`fi_provider`(7)](fi_provider.7.html) for a discussion on the different
types of providers available and the provider architecture.

## Communication Setup

At a high-level, communication via libfabric may be either connection-
oriented or connectionless. This is similar to choosing between using
TCP or UDP sockets, though libfabric supports reliable-connectionless
communication semantics. Communication between two processes occurs over
a construct known as an endpoint. Conceptually, an endpoint is
equivalent to a socket in the socket API world.

Specific APIs and libfabric objects are designed to manage and setup
communication between nodes. It includes calls for connection management
(CM), as well as functionality used to address connection-less
endpoints.

The CM APIs are modeled after APIs used to connect TCP sockets:
connect(), bind(), listen(), and accept(). A main difference is that
libfabric calls are designed to always operate asynchronously. CM APIs
are discussed in [`fi_cm`(3)](fi_cm.3.html).

For performance and scalability reasons discussed in the
[`fi_intro`(7)](fi_intro.7.html) page, connection-less endpoints use a
unique model to setup communication. These are based on a concept
referred to as address vectors, where the term vector means table or
array. Address vectors are discussed in detail later, but target
applications needing to talk with potentially thousands to millions of
peers.

## Data Transfer Services

libfabric provides several data transfer semantics to meet different
application requirements. There are five basic sets of data transfer
APIs: messages, tagged messages, RMA, atomics, and collectives.

*Messages*
:   Message APIs expose the ability to send and receive data with
    message boundaries being maintained. Message transfers act as FIFOs,
    with sent messages matched with receive buffers in the order that
    messages are received at the target. The message APIs are modeled
    after socket APIs, such as send(). sendto(), sendmsg(), recv(),
    recvmsg(), etc. For more information see
    [`fi_msg`(3)](fi_msg.3.html).

*Tagged Messages*
:   Tagged messages are similar to messages APIs, with the exception of
    how messages are matched at the receiver. Tagged messages maintain
    message boundaries, same as the message API. The tag matching APIs
    differ from the message APIs in that received messages are directed
    into buffers based on small steering tags that are specified and
    carried in the sent message. All message buffers, posted to send or
    receive data, are associated with a tag value. Sent messages are
    matched with buffers at the receiver that have the same tag. For
    more information, see [`fi_tagged`(3)](fi_tagged.3.html).

*RMA*
:   RMA stands for remote memory access. RMA transfers allow an
    application to write data directly into a specific memory location
    in a target process or to read memory from a specific address at the
    target process and return the data into a local buffer. RMA is also
    known as RDMA (remote direct memory access); however, RDMA
    originally defined a specific transport implementation of RMA. For
    more information, see [`fi_rma`(3)](fi_rma.3.html).

*Atomics*
:   Atomic operations add arithmetic operations to RMA transfers.
    Atomics permit direct access and manipulation of memory on the
    target process. libfabric defines a wide range of arithmetic
    operations that may be invoked as part of a data transfer operation.
    For more information, see [`fi_atomic`(3)](fi_atomic.3.html).

*Collectives*
:   The above data transfer APIs perform point-to-point communication.
    Data transfers occur between exactly one initiator and one target.
    Collective operations are coordinated atomic operations among an
    arbitrarily large number of peers. For more information, see
    [`fi_collective`(3)](fi_collective.3.html).

## Memory Registration

One of the objective of libfabric is to allow network hardware direct
access to application data buffers. This is accomplished through an
operation known as memory registration.

In order for a NIC to read or write application memory directly, it must
access the physical memory pages that back the application's address
space. Modern operating systems employ page files that swap out virtual
pages from one process with the virtual pages from another. As a result,
a physical memory page may map to different virtual addresses depending
on when it is accessed. Furthermore, when a virtual page is swapped in,
it may be mapped to a new physical page. If a NIC attempts to read or
write application memory without being linked into the virtual address
manager, it could access the wrong data, possibly corrupting an
application's memory. Memory registration can be used to avoid this
situation from occurring. For example, registered pages can be marked
such that the operating system locks the virtual to physical mapping,
avoiding any possibility of the virtual page being paged out or
remapped.

Memory registration is also the security mechanism used to grant a
remote peer access to local memory buffers. Registered memory regions
associate memory buffers with permissions granted for access by fabric
resources. A memory buffer must be registered before it can be used as
the target of an RMA or atomic data transfer. Memory registration
provides a simple protection mechanism. (Advanced scalable networks
employ other mechanisms, which are considered out of scope for the
purposes of this discussion.) After a memory buffer has been registered,
that registration request (buffer's address, buffer length, and access
permission) is given a registration key. Peers that issue RMA or atomic
operations against that memory buffer must provide this key as part of
their operation. This helps protects against unintentional accesses to
the region.

## Completion Services

libfabric data transfers operate asynchronously. Completion services are
used to report the results of submitted data transfer operations.
Completions may be reported using the cleverly named completions queues,
which provide details about the operation that completed. Or,
completions may be reported using completion counters that simply return
the number of operations that have completed.

Completion services are designed with high-performance, low-latency in
mind. The calls map directly into the providers, and data structures are
defined to minimize memory writes and cache impact. Completion services
do not have corresponding socket APIs. However, for Windows developers,
they are similar to IO completion ports.

# Object Model

libfabric follows an object-oriented design model. Although the
interfaces are written in C, the structures and implementation have a
C++ feel to them. The following diagram shows a high-level view of
notable libfabric objects and object dependencies.

    / Passive \ ---> <Fabric> <--- /Event\
    \Endpoints/         ^          \Queue/
                        |
      /Address\ ---> <Domain> <--- /Completion\
      \Vector /       ^  ^         \  Queue   /
                      |  |
          /Memory\ ---    --- / Active \
          \Region/            \Endpoint/

*Fabric*
:   A fabric represents a collection of hardware and software resources
    that access a single physical or virtual network. For example, a
    fabric may be a single network subnet or cluster. All network ports
    on a system that can communicate with each other through the fabric
    belong to the same fabric. A fabric shares network addresses and can
    span multiple providers. Fabrics are the top level object from which
    other objects are allocated.

*Domain*
:   A domain represents a logical connection into a fabric. In the
    simplest case, a domain may correspond to a physical or virtual NIC;
    however a domain could include multiple NICs (in the case of a
    multi-rail provider), or no NIC at all (in the case of shared
    memory). A domain defines the boundary within which other resources
    may be associated. Active endpoints and completion queues must be
    part of the same domain in order to be related to each other.

*Passive Endpoint*
:   Passive endpoints are used by connection-oriented protocols to
    listen for incoming connection requests. Passive endpoints often map
    to software constructs and may span multiple domains. They are best
    represented by a listening socket.

*Event Queues*
:   Event queues (EQs) are used to collect and report the completion of
    asynchronous operations and events. Event queues handle *control*
    events, that is, operations which are not directly associated with
    data transfer operations. The reason for separating control events
    from data transfer events is for performance reasons. Event queues
    are often implemented entirely in software using operating system
    constructs. Control events usually occur during an application's
    initialization phase, or at a rate that's several orders of
    magnitude smaller than data transfer events. Event queues are most
    commonly used by connection-oriented protocols for notification of
    connection request or established events.

*Active Endpoint*
:   Active endpoints are data transfer communication portals. They are
    conceptually similar to a TCP or UDP socket. Active endpoints are
    used to perform data transfers. Active endpoints implement the
    network protocol.

*Completion Queue*
:   Completion queues (CQs) are high-performance queues used to report
    the completion of data transfer operations. Unlike event queues,
    completion queues are often fully or partially implemented in
    hardware. Completion queue interfaces are designed to minimize
    software overhead.

*Memory Region*
:   Memory regions describe application's local memory buffers. In order
    for fabric resources to access application memory, the application
    must first grant permission to the fabric provider by constructing a
    memory region. Memory regions are required for specific types of
    data transfer operations, such as RMA and atomic operations.

*Address Vectors*
:   Address vectors are used by connection-less endpoints. They map
    higher level addresses, such as IP addresses or hostnames, which may
    be more natural for an application to use, into fabric specific
    addresses. The use of address vectors allows providers to reduce the
    amount of memory required to maintain large address look-up tables,
    and eliminate expensive address resolution and look-up methods
    during data transfer operations.

# Communication Model

Endpoints represent communication portals, and all data transfer
operations are initiated on endpoints. libfabric defines the conceptual
model for how endpoints are exposed to applications. It supports three
main communication endpoint types. The endpoint names are borrowed from
socket API naming.

*FI_EP_MSG*
:   Reliable-connected

*FI_EP_DGRAM*
:   Unreliable datagram

*FI_EP_RDM*
:   Reliable-unconnected

Communication setup is based on whether the endpoint is connected or
unconnected. Reliability is a feature of the endpoint's data transfer
protocol.

## Connected Communications

The following diagram highlights the general usage behind
connection-oriented communication. Connected communication is based on
the flow used to connect TCP sockets, with improved asynchronous
support.

             1 listen()              2 connect()
                 |                      |
             /Passive \  <---(3)--- / Active \
             \Endpoint/             \Endpoint/
             /                               \
            / (4 CONNREQ)                     \
    /Event\                                     /Event\
    \Queue/                                     \Queue/
                                               /
             5 accept()         (8 CONNECTED) /
                 |                           /
             / Active \  ------(6)--------->
             \Endpoint/  <-----(7)----------
             /
            / (9 CONNECTED)
    /Event\
    \Queue/

Connections require the use of both passive and active endpoints. In
order to establish a connection, an application must first create a
passive endpoint and associate it with an event queue. The event queue
will be used to report the connection management events. The application
then calls listen on the passive endpoint. A single passive endpoint can
be used to form multiple connections.

The connecting peer allocates an active endpoint, which is also
associated with an event queue. Connect is called on the active
endpoint, which results in sending a connection request (CONNREQ)
message to the passive endpoint. The CONNREQ event is inserted into the
passive endpoint's event queue, where the listening application can
process it.

Upon processing the CONNREQ, the listening application will allocate an
active endpoint to use with the connection. The active endpoint is bound
with an event queue. Although the diagram shows the use of a separate
event queue, the active endpoint may use the same event queue as used by
the passive endpoint. Accept is called on the active endpoint to finish
forming the connection. It should be noted that the OFI accept call is
different than the accept call used by sockets. The differences result
from OFI supporting process direct I/O.

libfabric does not define the connection establishment protocol, but
does support a traditional three-way handshake used by many
technologies. After calling accept, a response is sent to the connecting
active endpoint. That response generates a CONNECTED event on the remote
event queue. If a three-way handshake is used, the remote endpoint will
generate an acknowledgment message that will generate a CONNECTED event
for the accepting endpoint. Regardless of the connection protocol, both
the active and passive sides of the connection will receive a CONNECTED
event that signals that the connection has been established.

## Connectionless Communications

Connectionless communication allows data transfers between active
endpoints without going through a connection setup process. The diagram
below shows the basic components needed to setup connection-less
communication. Connectionless communication setup differs from UDP
sockets in that it requires that the remote addresses be stored with
libfabric.

      1 insert_addr()              2 send()
             |                        |
         /Address\ <--3 lookup--> / Active \
         \Vector /                \Endpoint/

libfabric requires the addresses of peer endpoints be inserted into a
local addressing table, or address vector, before data transfers can be
initiated against the remote endpoint. Address vectors abstract fabric
specific addressing requirements and avoid long queuing delays on data
transfers when address resolution is needed. For example, IP addresses
may need to be resolved into Ethernet MAC addresses. Address vectors
allow this resolution to occur during application initialization time.
libfabric does not define how an address vector be implemented, only its
conceptual model.

All connection-less endpoints that transfer data must be associated with
an address vector.

# Endpoints

At a low-level, endpoints are usually associated with a transmit
context, or queue, and a receive context, or queue. Although the terms
transmit and receive queues are easier to understand, libfabric uses the
terminology context, since queue like behavior of acting as a FIFO
(first-in, first-out) is not guaranteed. Transmit and receive contexts
may be implemented using hardware queues mapped directly into the
process's address space. An endpoint may be configured only to transmit
or receive data. Data transfer requests are converted by the underlying
provider into commands that are inserted into hardware transmit and/or
receive contexts.

Endpoints are also associated with completion queues. Completion queues
are used to report the completion of asynchronous data transfer
operations.

## Shared Contexts

An advanced usage model allows for sharing resources among multiple
endpoints. The most common form of sharing is having multiple connected
endpoints make use of a single receive context. This can reduce receive
side buffering requirements, allowing the number of connected endpoints
that an application can manage to scale to larger numbers.

# Data Transfers

Obviously, a primary goal of network communication is to transfer data
between processes running on different systems. In a similar way that
the socket API defines different data transfer semantics for TCP versus
UDP sockets, that is, streaming versus datagram messages, libfabric
defines different types of data transfers. However, unlike sockets,
libfabric allows different semantics over a single endpoint, even when
communicating with the same peer.

libfabric uses separate API sets for the different data transfer
semantics; although, there are strong similarities between the API sets.
The differences are the result of the parameters needed to invoke each
type of data transfer.

## Message transfers

Message transfers are most similar to UDP datagram transfers, except
that transfers may be sent and received reliably. Message transfers may
also be gigabytes in size, depending on the provider implementation. The
sender requests that data be transferred as a single transport operation
to a peer. Even if the data is referenced using an I/O vector, it is
treated as a single logical unit or message. The data is placed into a
waiting receive buffer at the peer, with the receive buffer usually
chosen using FIFO ordering. Note that even though receive buffers are
selected using FIFO ordering, the received messages may complete out of
order. This can occur as a result of data between and within messages
taking different paths through the network, handling lost or
retransmitted packets, etc.

Message transfers are usually invoked using API calls that contain the
string "send" or "recv". As a result they may be referred to simply as
sent or received messages.

Message transfers involve the target process posting memory buffers to
the receive (Rx) context of its endpoint. When a message arrives from
the network, a receive buffer is removed from the Rx context, and the
data is copied from the network into the receive buffer. Messages are
matched with posted receives in the order that they are received. Note
that this may differ from the order that messages are sent, depending on
the transmit side's ordering semantics.

Conceptually, on the transmit side, messages are posted to a transmit
(Tx) context. The network processes messages from the Tx context,
packetizing the data into outbound messages. Although many
implementations process the Tx context in order (i.e. the Tx context is
a true queue), ordering guarantees specified through the libfabric API
determine the actual processing order. As a general rule, the more
relaxed an application is on its message and data ordering, the more
optimizations the networking software and hardware can leverage,
providing better performance.

## Tagged messages

Tagged messages are similar to message transfers except that the
messages carry one additional piece of information, a message tag. Tags
are application defined values that are part of the message transfer
protocol and are used to route packets at the receiver. At a high level,
they are roughly similar to message ids. The difference is that tag
values are set by the application, may be any value, and duplicate tag
values are allowed.

Each sent message carries a single tag value, which is used to select a
receive buffer into which the data is copied. On the receiving side,
message buffers are also marked with a tag. Messages that arrive from
the network search through the posted receive messages until a matching
tag is found.

Tags are often used to identify virtual communication groups or roles.
In practice, message tags are typically divided into fields. For
example, the upper 16 bits of the tag may indicate a virtual group, with
the lower 16 bits identifying the message purpose. The tag message
interface in libfabric is designed around this usage model. Each sent
message carries exactly one tag value, specified through the API. At the
receiver, buffers are associated with both a tag value and a mask. The
mask is used as part of the buffer matching process. The mask is applied
against the received tag value carried in the sent message prior to
checking the tag against the receive buffer. For example, the mask may
indicate to ignore the lower 16-bits of a tag. If the resulting values
match, then the tags are said to match. The received data is then placed
into the matched buffer.

For performance reasons, the mask is specified as 'ignore' bits.
Although this is backwards from how many developers think of a mask
(where the bits that are valid would be set to 1), the definition ends
up mapping well with applications. The actual operation performed when
matching tags is:

    send_tag | ignore == recv_tag | ignore

    /* this is equivalent to:
     * send_tag & ~ignore == recv_tag & ~ignore
     */

Tagged messages are equivalent of message transfers if a single tag
value is used. But tagged messages require that the receiver perform a
matching operation at the target, which can impact performance versus
untagged messages.

## RMA

RMA operations are architected such that they can require no processing
by the CPU at the RMA target. NICs which offload transport functionality
can perform RMA operations without impacting host processing. RMA write
operations transmit data from the initiator to the target. The memory
location where the data should be written is carried within the
transport message itself, with verification checks at the target to
prevent invalid access.

RMA read operations fetch data from the target system and transfer it
back to the initiator of the request, where it is placed into memory.
This too can be done without involving the host processor at the target
system when the NIC supports transport offloading.

The advantage of RMA operations is that they decouple the processing of
the peers. Data can be placed or fetched whenever the initiator is ready
without necessarily impacting the peer process.

Because RMA operations allow a peer to directly access the memory of a
process, additional protection mechanisms are used to prevent
unintentional or unwanted access. RMA memory that is updated by a write
operation or is fetched by a read operation must be registered for
access with the correct permissions specified.

## Atomic operations

Atomic transfers are used to read and update data located in remote
memory regions in an atomic fashion. Conceptually, they are similar to
local atomic operations of a similar nature (e.g. atomic increment,
compare and swap, etc.). The benefit of atomic operations is they enable
offloading basic arithmetic capabilities onto a NIC. Unlike other data
transfer operations, which merely need to transfer bytes of data,
atomics require knowledge of the format of the data being accessed.

A single atomic function operates across an array of data, applying an
atomic operation to each entry. The atomicity of an operation is limited
to a single data type or entry, however, not across the entire array.
libfabric defines a wide variety of atomic operations across all common
data types. However support for a given operation is dependent on the
provider implementation.

## Collective operations

In general, collective operations can be thought of as coordinated
atomic operations between a set of peer endpoints, almost like a
multicast atomic request. A single collective operation can result in
data being collected from multiple peers, combined using a set of atomic
primitives, and the results distributed to all peers. A collective
operation is a group communication exchange. It involves multiple peers
exchanging data with other peers participating in the collective call.
Collective operations require close coordination by all participating
members, and collective calls can strain the fabric, as well as local
and remote data buffers.

Collective operations are an area of heavy research, with dedicated
libraries focused almost exclusively on implementing collective
operations efficiently. Such libraries are a specific target of
libfabric. The main object of the libfabric collection APIs is to expose
network acceleration features for implementing collectives to
higher-level libraries and applications. It is recommended that
applications needing collective communication target higher-level
libraries, such as MPI, instead of using libfabric collective APIs for
that purpose.

{% include JB/setup %}

# NAME

fi_setup - libfabric setup and initialization

# OVERVIEW

A full description of the libfabric API is documented in the relevant
man pages. This section provides an introduction to select interfaces,
including how they may be used. It does not attempt to capture all
subtleties or use cases, nor describe all possible data structures or
fields. However, it is useful for new developers trying to kick-start
using libfabric.

# fi_getinfo()

The fi_getinfo() call is one of the first calls that applications
invoke. It is designed to be easy to use for simple applications, but
extensible enough to configure a network for optimal performance. It
serves several purposes. First, it abstracts away network implementation
and addressing details. Second, it allows an application to specify
which features they require of the network. Last, it provides a
mechanism for a provider to report how an application can use the
network in order to achieve the best performance. fi_getinfo() is
loosely based on the getaddrinfo() call.

    /* API prototypes */
    struct fi_info *fi_allocinfo(void);

    int fi_getinfo(int version, const char *node, const char *service,
        uint64_t flags, struct fi_info *hints, struct fi_info **info);

    /* Sample initialization code flow */
    struct fi_info *hints, *info;

    hints = fi_allocinfo();

    /* hints will point to a cleared fi_info structure
     * Initialize hints here to request specific network capabilities
     */

    fi_getinfo(FI_VERSION(1, 16), NULL, NULL, 0, hints, &info);
    fi_freeinfo(hints);

    /* Use the returned info structure to allocate fabric resources */

The hints parameter is the key for requesting fabric services. The
fi_info structure contains several data fields, plus pointers to a wide
variety of attributes. The fi_allocinfo() call simplifies the creation
of an fi_info structure and is strongly recommended for use. In this
example, the application is merely attempting to get a list of what
providers are available in the system and the features that they
support. Note that the API is designed to be extensible. Versioning
information is provided as part of the fi_getinfo() call. The version is
used by libfabric to determine what API features the application is
aware of. In this case, the application indicates that it can properly
handle any feature that was defined for the 1.16 release (or earlier).

Applications should *always* hard code the version that they are written
for into the fi_getinfo() call. This ensures that newer versions of
libfabric will provide backwards compatibility with that used by the
application. Newer versions of libfabric must support applications that
were compiled against an older version of the library. It must also
support applications written against header files from an older library
version, but re-compiled against newer header files. Among other things,
the version parameter allows libfabric to determine if an application is
aware of new fields that may have been added to structures, or if the
data in those fields may be uninitialized.

Typically, an application will initialize the hints parameter to list
the features that it will use.

    /* Taking a peek at the contents of fi_info */
    struct fi_info {
        struct fi_info *next;
        uint64_t caps;
        uint64_t mode;
        uint32_t addr_format;
        size_t src_addrlen;
        size_t dest_addrlen;
        void *src_addr;
        void *dest_addr;
        fid_t handle;
        struct fi_tx_attr *tx_attr;
        struct fi_rx_attr *rx_attr;
        struct fi_ep_attr *ep_attr;
        struct fi_domain_attr *domain_attr;
        struct fi_fabric_attr *fabric_attr;
        struct fid_nic *nic;
    };

The fi_info structure references several different attributes, which
correspond to the different libfabric objects that an application
allocates. For basic applications, modifying or accessing most attribute
fields are unnecessary. Many applications will only need to deal with a
few fields of fi_info, most notably the endpoint type, capability (caps)
bits, and mode bits. These are defined in more detail below.

On success, the fi_getinfo() function returns a linked list of fi_info
structures. Each entry in the list will meet the conditions specified
through the hints parameter. The returned entries may come from
different network providers, or may differ in the returned attributes.
For example, if hints does not specify a particular endpoint type, there
may be an entry for each of the three endpoint types. As a general rule,
libfabric attempts to return the list of fi_info structures in order
from most desirable to least. High-performance network providers are
listed before more generic providers.

## Capabilities (fi_info::caps)

The fi_info caps field is used to specify the features and services that
the application requires of the network. This field is a bit-mask of
desired capabilities. There are capability bits for each of the data
transfer services previously mentioned: FI_MSG, FI_TAGGED, FI_RMA,
FI_ATOMIC, and FI_COLLECTIVE. Applications should set each bit for each
set of operations that it will use. These bits are often the only caps
bits set by an application.

Capabilities are grouped into three general categories: primary,
secondary, and primary modifiers. Primary capabilities must explicitly
be requested by an application, and a provider must enable support for
only those primary capabilities which were selected. This is required
for both performance and security reasons. Primary modifiers are used to
limit a primary capability, such as restricting an endpoint to being
send-only.

Secondary capabilities may optionally be requested by an application. If
requested, a provider must support a capability if it is asked for or
fail the fi_getinfo request. A provider may optionally report
non-requested secondary capabilities if doing so would not compromise
performance or security. That is, a provider may grant an application a
secondary capability, regardless of whether the application requested
it. The most commonly accessed secondary capability bits indicate if
provider communication is restricted to the local node (for example, the
shared memory provider only supports local communication) and/or remote
nodes (which can be the case for NICs that lack loopback support). Other
secondary capability bits mostly deal with features targeting
highly-scalable applications, but may not be commonly supported across
multiple providers.

Because different providers support different sets of capabilities,
applications that desire optimal network performance may need to code
for a capability being either present or absent. When present, such
capabilities can offer a scalability or performance boost. When absent,
an application may prefer to adjust its protocol or implementation to
work around the network limitations. Although providers can often
emulate features, doing so can impact overall performance, including the
performance of data transfers that otherwise appear unrelated to the
feature in use. For example, if a provider needs to insert protocol
headers into the message stream in order to implement a given
capability, the insertion of that header could negatively impact the
performance of all transfers. By exposing such limitations to the
application, the application developer has better control over how to
best emulate the feature or work around its absence.

It is recommended that applications code for only those capabilities
required to achieve the best performance. If a capability would have
little to no effect on overall performance, developers should avoid
using such features as part of an initial implementation. This will
allow the application to work well across the widest variety of
hardware. Application optimizations can then add support for less common
features. To see which features are supported by which providers, see
the libfabric [Provider Feature
Maxtrix](https://github.com/ofiwg/libfabric/wiki/Provider-Feature-Matrix)
for the relevant release.

## Mode Bits (fi_info::mode)

Where capability bits represent features desired by applications, mode
bits correspond to behavior needed by the provider. That is, capability
bits are top down requests, whereas mode bits are bottom up
restrictions. Mode bits are set by the provider to request that the
application use the API in a specific way in order to achieve optimal
performance. Mode bits often imply that the additional work to implement
certain communication semantics needed by the application will be less
if implemented by the applicaiton than forcing that same implementation
down into the provider. Mode bits arise as a result of hardware
implementation restrictions.

An application developer decides which mode bits they want to or can
easily support as part of their development process. Each mode bit
describes a particular behavior that the application must follow to use
various interfaces. Applications set the mode bits that they support
when calling fi_getinfo(). If a provider requires a mode bit that isn't
set, that provider will be skipped by fi_getinfo(). If a provider does
not need a mode bit that is set, it will respond to the fi_getinfo()
call, with the mode bit cleared. This indicates that the application
does not need to perform the action required by the mode bit.

One of common mode bit needed by providers is FI_CONTEXT (and
FI_CONTEXT2). This mode bit requires that applications pass in a
libfabric defined data structure (struct fi_context) into any data
transfer function. That structure must remain valid and unused by the
application until the data transfer operation completes. The purpose
behind this mode bit is that the struct fi_context provides "scratch"
space that the provider can use to track the request. For example, it
may need to insert the request into a linked list while it is pending,
or track the number of times that an outbound transfer has been retried.
Since many applications already track outstanding operations with their
own data structure, by embedding the struct fi_context into that same
structure, overall performance can be improved. This avoids the provider
needing to allocate and free internal structures for each request.

Continuing with this example, if an application does not already track
outstanding requests, then it would leave the FI_CONTEXT mode bit unset.
This would indicate that the provider needs to get and release its own
structure for tracking purposes. In this case, the costs would
essentially be the same whether it were done by the application or
provider.

For the broadest support of different network technologies, applications
should attempt to support as many mode bits as feasible. It is
recommended that providers support applications that cannot support any
mode bits, with as small an impact as possible. However, implementation
of mode bit avoidance in the provider can still impact performance, even
when the mode bit is disabled. As a result, some providers may always
require specific mode bits be set.

# FIDs (fid_t)

FID stands for fabric identifier. It is the base object type assigned to
all libfabric API objects. All fabric resources are represented by a fid
structure, and all fid's are derived from a base fid type. In
object-oriented terms, a fid would be the parent class. The contents of
a fid are visible to the application.

    /* Base FID definition */
    enum {
        FI_CLASS_UNSPEC,
        FI_CLASS_FABRIC,
        FI_CLASS_DOMAIN,
        ...
    };

    struct fi_ops {
        size_t size;
        int (*close)(struct fid *fid);
        ...
    };

    /* All fabric interface descriptors must start with this structure */
    struct fid {
        size_t fclass;
        void *context;
        struct fi_ops *ops;
    };

The fid structure is designed as a trade-off between minimizing memory
footprint versus software overhead. Each fid is identified as a specific
object class, which helps with debugging. Examples are given above
(e.g. FI_CLASS_FABRIC). The context field is an application defined data
value, assigned to an object during its creation. The use of the context
field is application specific, but it is meant to be read by
applications. Applications often set context to a corresponding
structure that it's allocated. The context field is the only field that
applications are recommended to access directly. Access to other fields
should be done using defined function calls (for example, the close()
operation).

The ops field points to a set of function pointers. The fi_ops structure
defines the operations that apply to that class. The size field in the
fi_ops structure is used for extensibility, and allows the fi_ops
structure to grow in a backward compatible manner as new operations are
added. The fid deliberately points to the fi_ops structure, rather than
embedding the operations directly. This allows multiple fids to point to
the same set of ops, which minimizes the memory footprint of each fid.
(Internally, providers usually set ops to a static data structure, with
the fid structure dynamically allocated.)

Although it's possible for applications to access function pointers
directly, it is strongly recommended that the static inline functions
defined in the man pages be used instead. This is required by
applications that may be built using the FABRIC_DIRECT library feature.
(FABRIC_DIRECT is a compile time option that allows for highly optimized
builds by tightly coupling an application with a specific provider.)

Other OFI classes are derived from this structure, adding their own set
of operations.

    /* Example of deriving a new class for a fabric object */
    struct fi_ops_fabric {
        size_t size;
        int (*domain)(struct fid_fabric *fabric, struct fi_info *info,
            struct fid_domain **dom, void *context);
        ...
    };

    struct fid_fabric {
        struct fid fid;
        struct fi_ops_fabric *ops;
    };

Other fid classes follow a similar pattern as that shown for fid_fabric.
The base fid structure is followed by zero or more pointers to operation
sets.

# Fabric (fid_fabric)

The top-level object that applications open is the fabric identifier.
The fabric can mostly be viewed as a container object by applications,
though it does identify which provider(s) applications use.

Opening a fabric is usually a straightforward call after calling
fi_getinfo().

    int fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);

The fabric attributes can be directly accessed from struct fi_info. The
newly opened fabric is returned through the 'fabric' parameter. The
'context' parameter appears in many operations. It is a user-specified
value that is associated with the fabric. It may be used to point to an
application specific structure and is retrievable from struct
fid_fabric.

## Attributes (fi_fabric_attr)

The fabric attributes are straightforward.

    struct fi_fabric_attr {
        struct fid_fabric *fabric;
        char *name;
        char *prov_name;
        uint32_t prov_version;
        uint32_t api_version;
    };

The only field that applications are likely to use directly is the
prov_name. This is a string value that can be used by hints to select a
specific provider for use. On most systems, there will be multiple
providers available. Only one is likely to represent the
high-performance network attached to the system. Others are generic
providers that may be available on any system, such as the TCP socket
and UDP providers.

The fabric field is used to help applications manage opened fabric
resources. If an application has already opened a fabric that can
support the returned fi_info structure, this will be set to that fabric.

# Domains (fid_domain)

Domains frequently map to a specific local network interface adapter. A
domain may either refer to the entire NIC, a port on a multi-port NIC, a
virtual device exposed by a NIC, multiple NICs being used in a
multi-rail fashion, and so forth. Although it's convenient to think of a
domain as referring to a NIC, such an association isn't expected by
libfabric. From the viewpoint of the application, a domain identifies a
set of resources that may be used together.

Similar to a fabric, opening a domain is straightforward after calling
fi_getinfo().

    int fi_domain(struct fid_fabric *fabric, struct fi_info *info,
        struct fid_domain **domain, void *context);

The fi_info structure returned from fi_getinfo() can be passed directly
to fi_domain() to open a new domain.

## Attributes (fi_domain_attr)

One of the goals of a domain is to define the relationship between data
transfer services (endpoints) and completion services (completion queues
and counters). Many of the domain attributes describe that relationship
and its impact to the application.

    struct fi_domain_attr {
        struct fid_domain *domain;
        char *name;
        enum fi_threading threading;
        enum fi_progress progress;
        enum fi_resource_mgmt resource_mgmt;
        enum fi_av_type av_type;
        enum fi_mr_mode mr_mode;
        size_t mr_key_size;
        size_t cq_data_size;
        size_t cq_cnt;
        size_t ep_cnt;
        size_t tx_ctx_cnt;
        size_t rx_ctx_cnt;
        ...

Full details of the domain attributes and their meaning are in the
fi_domain man page. Information on select attributes and their impact to
the application are described below.

## Threading (fi_threading)

libfabric defines a unique threading model. The libfabric design is
heavily influenced by object-oriented programming concepts. A
multi-threaded application must determine how libfabric objects
(domains, endpoints, completion queues, etc.) will be allocated among
its threads, or if any thread can access any object. For example, an
application may spawn a new thread to handle each new connected
endpoint. The domain threading field provides a mechanism for an
application to identify which objects may be accessed simultaneously by
different threads. This in turn allows a provider to optimize or, in
some cases, eliminate internal synchronization and locking around those
objects.

Threading defines where providers could optimize synchronization
primitives. However, providers may still implement more serialization
than is needed by the application. (This is usually a result of keeping
the provider implementation simpler).

It is recommended that applications target either FI_THREAD_SAFE (full
thread safety implemented by the provider) or FI_THREAD_DOMAIN (objects
associated with a single domain will only be accessed by a single
thread).

## Progress (fi_progress)

Progress models are a result of using the host processor in order to
perform some portion of the transport protocol. In order to simplify
development, libfabric defines two progress models: automatic or manual.
It does not attempt to identify which specific interface features may be
offloaded, or what operations require additional processing by the
application's thread.

Automatic progress means that an operation initiated by the application
will eventually complete, even if the application makes no further calls
into the libfabric API. The operation is either offloaded entirely onto
hardware, the provider uses an internal thread, or the operating system
kernel may perform the task. The use of automatic progress may increase
system overhead and latency in the latter two cases. For control
operations, such as connection setup, this is usually acceptable.
However, the impact to data transfers may be measurable, especially if
internal threads are required to provide automatic progress.

The manual progress model can avoid this overhead for providers that do
not offload all transport features into hardware. With manual progress
the provider implementation will handle transport operations as part of
specific libfabric functions. For example, a call to fi_cq_read() which
retrieves an array completed operations may also be responsible for
sending ack messages to notify peers that a message has been received.
Since reading the completion queue is part of the normal operation of an
application, there is minimal impact to the application and additional
threads are avoided.

Applications need to take care when using manual progress, particularly
if they link into libfabric multiple times through different code paths
or library dependencies. If application threads are used to drive
progress, such as responding to received data with ACKs, then it is
critical that the application thread call into libfabric in a timely
manner.

## Memory Registration (fid_mr)

RMA, atomic, and collective operations can read and write memory that is
owned by a peer process, and neither require the involvement of the
target processor. Because the memory can be modified over the network,
an application must opt into exposing its memory to peers. This is
handled by the memory registration process. Registered memory regions
associate memory buffers with permissions granted for access by fabric
resources. A memory buffer must be registered before it can be used as
the target of a remote RMA, atomic, or collective data transfer.
Additionally, a fabric provider may require that data buffers be
registered before being used even in the case of local transfers. The
latter is necessary to ensure that the virtual to physical page mappings
do not change while network hardware is performing the transfer.

In order to handle diverse hardware requirements, there are a set of
mr_mode bits associated with memory registration. The mr_mode bits
behave similar to fi_info mode bits. Applications indicate which types
of restrictions they can support, and the providers clear those bits
which aren't needed.

For hardware that requires memory registration, managing registration is
critical to achieving good performance and scalability. The act of
registering memory is costly and should be avoided on a per data
transfer basis. libfabric has extensive internal support for managing
memory registration, hiding registration from user application, caching
registration to reduce per transfer overhead, and detecting when cached
registrations are no longer valid. It is recommended that applications
that are not natively designed to account for registering memory to make
use of libfabric's registration cache. This can be done by simply not
setting the relevant mr_mode bits.

### Memory Region APIs

The following APIs highlight how to allocate and access a registered
memory region. Note that this is not a complete list of memory region
(MR) calls, and for full details on each API, readers should refer
directly to the fi_mr man page.

    int fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
        uint64_t access, uint64_t offset, uint64_t requested_key, uint64_t flags,
        struct fid_mr **mr, void *context);

    void * fi_mr_desc(struct fid_mr *mr);
    uint64_t fi_mr_key(struct fid_mr *mr);

By default, memory regions are associated with a domain. A MR is
accessible by any endpoint that is opened on that domain. A region
starts at the address specified by 'buf', and is 'len' bytes long. The
'access' parameter are permission flags that are OR'ed together. The
permissions indicate which type of operations may be invoked against the
region (e.g. FI_READ, FI_WRITE, FI_REMOTE_READ, FI_REMOTE_WRITE). The
'buf' parameter typically references allocated virtual memory.

A MR is associated with local and remote protection keys. The local key
is referred to as a memory descriptor and may be retrieved by calling
fi_mr_desc(). This call is only needed if the FI_MR_LOCAL mr_mode bit
has been set. The memory descriptor is passed directly into data
transfer operations, for example:

    /* fi_mr_desc() example using fi_send() */
    fi_send(ep, buf, len, fi_mr_desc(mr), 0, NULL);

The remote key, or simply MR key, is used by the peer when targeting the
MR with an RMA or atomic operation. In many cases, the key will need to
be sent in a separate message to the initiating peer. libfabric API uses
a 64-bit key where one is used. The actual key size used by a provider
is part of its domain attributes Support for larger key sizes, as
required by some providers, is conveyed through an mr_mode bit, and
requires the use of extended MR API calls that map the larger size to a
64-bit value.

# Endpoints

Endpoints are transport level communication portals. Opening an endpoint
is trivial after calling fi_getinfo().

## Active (fid_ep)

Active endpoints may be connection-oriented or connection-less. They are
considered active as they may be used to perform data transfers. All
data transfer interfaces -- messages (fi_msg), tagged messages
(fi_tagged), RMA (fi_rma), atomics (fi_atomic), and collectives
(fi_collective) -- are associated with active endpoints. Though an
individual endpoint may not be enabled to use all data transfers. In
standard configurations, an active endpoint has one transmit and one
receive queue. In general, operations that generate traffic on the
fabric are posted to the transmit queue. This includes all RMA and
atomic operations, along with sent messages and sent tagged messages.
Operations that post buffers for receiving incoming data are submitted
to the receive queue.

Active endpoints are created in the disabled state. The endpoint must
first be configured prior to it being enabled. Endpoints must transition
into an enabled state before accepting data transfer operations,
including posting of receive buffers. The fi_enable() call is used to
transition an active endpoint into an enabled state. The fi_connect()
and fi_accept() calls will also transition an endpoint into the enabled
state, if it is not already enabled.

    int fi_endpoint(struct fid_domain *domain, struct fi_info *info,
        struct fid_ep **ep, void *context);

### Enabling (fi_enable)

In order to transition an endpoint into an enabled state, it must be
bound to one or more fabric resources. This includes binding the
endpoint to a completion queue and event queue. Unconnected endpoints
must also be bound to an address vector.

    /* Example to enable an unconnected endpoint */

    /* Allocate an address vector and associated it with the endpoint */
    fi_av_open(domain, &av_attr, &av, NULL);
    fi_ep_bind(ep, &av->fid, 0);

    /* Allocate and associate completion queues with the endpoint */
    fi_cq_open(domain, &cq_attr, &cq, NULL);
    fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);

    fi_enable(ep);

In the above example, we allocate an AV and CQ. The attributes for the
AV and CQ are omitted (additional discussion below). Those are then
associated with the endpoint through the fi_ep_bind() call. After all
necessary resources have been assigned to the endpoint, we enable it.
Enabling the endpoint indicates to the provider that it should allocate
any hardware and software resources and complete the initialization for
the endpoint. (If the endpoint is not bound to all necessary resources,
the fi_enable() call will fail.)

The fi_enable() call is always called for unconnected endpoints.
Connected endpoints may be able to skip calling fi_enable(), since
fi_connect() and fi_accept() will enable the endpoint automatically.
However, applications may still call fi_enable() prior to calling
fi_connect() or fi_accept(). Doing so allows the application to post
receive buffers to the endpoint, which ensures that they are available
to receive data in the case the peer endpoint sends messages immediately
after it establishes the connection.

## Passive (fid_pep)

Passive endpoints are used to listen for incoming connection requests.
Passive endpoints are of type FI_EP_MSG, and may not perform any data
transfers. An application wishing to create a passive endpoint typically
calls fi_getinfo() using the FI_SOURCE flag, often only specifying a
'service' address. The service address corresponds to a TCP port number.

Passive endpoints are associated with event queues. Event queues report
connection requests from peers. Unlike active endpoints, passive
endpoints are not associated with a domain. This allows an application
to listen for connection requests across multiple domains, though still
restricted to a single provider.

    /* Example passive endpoint listen */
    fi_passive_ep(fabric, info, &pep, NULL);

    fi_eq_open(fabric, &eq_attr, &eq, NULL);
    fi_pep_bind(pep, &eq->fid, 0);

    fi_listen(pep);

A passive endpoint must be bound to an event queue before calling
listen. This ensures that connection requests can be reported to the
application. To accept new connections, the application waits for a
request, allocates a new active endpoint for it, and accepts the
request.

    /* Example accepting a new connection */

    /* Wait for a CONNREQ event */
    fi_eq_sread(eq, &event, &cm_entry, sizeof cm_entry, -1, 0);
    assert(event == FI_CONNREQ);

    /* Allocate a new endpoint for the connection */
    if (!cm_entry.info->domain_attr->domain)
        fi_domain(fabric, cm_entry.info, &domain, NULL);
    fi_endpoint(domain, cm_entry.info, &ep, NULL);

    fi_ep_bind(ep, &eq->fid, 0);
    fi_cq_open(domain, &cq_attr, &cq, NULL);
    fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);

    fi_enable(ep);
    fi_recv(ep, rx_buf, len, NULL, 0, NULL);

    fi_accept(ep, NULL, 0);
    fi_eq_sread(eq, &event, &cm_entry, sizeof cm_entry, -1, 0);
    assert(event == FI_CONNECTED);

The connection request event (FI_CONNREQ) includes information about the
type of endpoint to allocate, including default attributes to use. If a
domain has not already been opened for the endpoint, one must be opened.
Then the endpoint and related resources can be allocated. Unlike the
unconnected endpoint example above, a connected endpoint does not have
an AV, but does need to be bound to an event queue. In this case, we use
the same EQ as the listening endpoint. Once the other EP resources
(e.g. CQ) have been allocated and bound, the EP can be enabled.

To accept the connection, the application calls fi_accept(). Note that
because of thread synchronization issues, it is possible for the active
endpoint to receive data even before fi_accept() can return. The posting
of receive buffers prior to calling fi_accept() handles this condition,
which avoids network flow control issues occurring immediately after
connecting.

The fi_eq_sread() calls are blocking (synchronous) read calls to the
event queue. These calls wait until an event occurs, which in this case
are connection request and establishment events.

## EP Attributes (fi_ep_attr)

The properties of an endpoint are specified using endpoint attributes.
These are attributes for the endpoint as a whole. There are additional
attributes specifically related to the transmit and receive contexts
underpinning the endpoint (details below).

    struct fi_ep_attr {
        enum fi_ep_type type;
        uint32_t        protocol;
        uint32_t        protocol_version;
        size_t          max_msg_size;
        ...
    };

A full description of each field is available in the fi_endpoint man
page, with selected details listed below.

### Endpoint Type (fi_ep_type)

This indicates the type of endpoint: reliable datagram (FI_EP_RDM),
reliable-connected (FI_EP_MSG), or unreliable datagram (FI_EP_DGRAM).
Nearly all applications will want to specify the endpoint type as a hint
passed into fi_getinfo, as most applications will only be coded to
support a single endpoint type.

### Maximum Message Size (max_msg_size)

This size is the maximum size for any data transfer operation that goes
over the endpoint. For unreliable datagram endpoints, this is often the
MTU of the underlying network. For reliable endpoints, this value is
often a restriction of the underlying transport protocol. A common
minimum maximum message size is 2GB, though some providers support an
arbitrarily large size. Applications that require transfers larger than
the maximum reported size are required to break up a single, large
transfer into multiple operations.

Providers expose their hardware or network limits to the applications,
rather than segmenting large transfers internally, in order to minimize
completion overhead. For example, for a provider to support large
message segmentation internally, it would need to emulate all completion
mechanisms (queues and counters) in software, even if transfers that are
larger than the transport supported maximum were never used.

### Message Order Size (max_order_xxx_size)

These fields specify data ordering. They define the delivery order of
transport data into target memory for RMA and atomic operations. Data
ordering requires message ordering. If message ordering is not
specified, these fields do not apply.

For example, suppose that an application issues two RMA write operations
to the same target memory location. (The application may be writing a
time stamp value every time a local condition is met, for instance).
Message ordering indicates that the first write as initiated by the
sender is the first write processed by the receiver. Data ordering
indicates whether the *data* from the first write updates memory before
the second write updates memory.

The max_order_xxx_size fields indicate how large a message may be while
still achieving data ordering. If a field is 0, then no data ordering is
guaranteed. If a field is the same as the max_msg_size, then data order
is guaranteed for all messages.

Providers may support data ordering up to max_msg_size for back to back
operations that are the same. For example, an RMA write followed by an
RMA write may have data ordering regardless of the size of the data
transfer (max_order_waw_size = max_msg_size). Mixed operations, such as
a read followed by a write, are often restricted. This is because RMA
read operations may require acknowledgments from the *initiator*, which
impacts the re-transmission protocol.

For example, consider an RMA read followed by a write. The target will
process the read request, retrieve the data, and send a reply. While
that is occurring, a write is received that wants to update the same
memory location accessed by the read. If the target processes the write,
it will overwrite the memory used by the read. If the read response is
lost, and the read is retried, the target will be unable to re-send the
data. To handle this, the target either needs to: defer handling the
write until it receives an acknowledgment for the read response, buffer
the read response so it can be re-transmitted, or indicate that data
ordering is not guaranteed.

Because the read or write operation may be gigabytes in size, deferring
the write may add significant latency, and buffering the read response
may be impractical. The max_order_xxx_size fields indicate how large
back to back operations may be with ordering still maintained. In many
cases, read after write and write and read ordering may be significantly
limited, but still usable for implementing specific algorithms, such as
a global locking mechanism.

## Rx/Tx Context Attributes (fi_rx_attr / fi_tx_attr)

The endpoint attributes define the overall abilities for the endpoint;
however, attributes that apply specifically to receive or transmit
contexts are defined by struct fi_rx_attr and fi_tx_attr, respectively:

    struct fi_rx_attr {
        uint64_t caps;
        uint64_t mode;
        uint64_t op_flags;
        uint64_t msg_order;
        ...
    };

    struct fi_tx_attr {
        uint64_t caps;
        uint64_t mode;
        uint64_t op_flags;
        uint64_t msg_order;
        size_t inject_size;
        ...
    };

Rx/Tx context capabilities must be a subset of the endpoint
capabilities. For many applications, the default attributes returned by
the provider will be sufficient, with the application only needing to
specify endpoint attributes.

Both context attributes include an op_flags field. This field is used by
applications to specify the default operation flags to use with any
call. For example, by setting the transmit context's op_flags to
FI_INJECT, the application has indicated to the provider that all
transmit operations should assume 'inject' behavior is desired. I.e. the
buffer provided to the call must be returned to the application upon
return from the function. The op_flags applies to all operations that do
not provide flags as part of the call (e.g. fi_sendmsg). One use of
op_flags is to specify the default completion semantic desired
(discussed next) by the application. By setting the default op_flags at
initialization time, we can avoid passing the flags as arguments into
some data transfer calls, avoid parsing the flags, and can prepare
submitted commands ahead of time.

It should be noted that some attributes are dependent upon the peer
endpoint having supporting attributes in order to achieve correct
application behavior. For example, message order must be the compatible
between the initiator's transmit attributes and the target's receive
attributes. Any mismatch may result in incorrect behavior that could be
difficult to debug.

# Completions

Data transfer operations complete asynchronously. Libfabric defines two
mechanism by which an application can be notified that an operation has
completed: completion queues and counters. Regardless of which mechanism
is used to notify the application that an operation is done, developers
must be aware of what a completion indicates.

In all cases, a completion indicates that it is safe to reuse the
buffer(s) associated with the data transfer. This completion mode is
referred to as *inject* complete and corresponds to the operational
flags FI_INJECT_COMPLETE. However, a completion may also guarantee
stronger semantics.

Although libfabric does not define an implementation, a provider can
meet the requirement for inject complete by copying the application's
buffer into a network buffer before generating the completion. Even if
the transmit operation is lost and must be retried, the provider can
resend the original data from the copied location. For large transfers,
a provider may not mark a request as inject complete until the data has
been acknowledged by the target. Applications, however, should only
infer that it is safe to reuse their data buffer for an inject complete
operation.

Transmit complete is a completion mode that provides slightly stronger
guarantees to the application. The meaning of transmit complete depends
on whether the endpoint is reliable or unreliable. For an unreliable
endpoint (FI_EP_DGRAM), a transmit completion indicates that the request
has been delivered to the network. That is, the message has been
delivered at least as far as hardware queues on the local NIC. For
reliable endpoints, a transmit complete occurs when the request has
reached the target endpoint. Typically, this indicates that the target
has acked the request. Transmit complete maps to the operation flag
FI_TRANSMIT_COMPLETE.

A third completion mode is defined to provide guarantees beyond transmit
complete. With transmit complete, an application knows that the message
is no longer dependent on the local NIC or network (e.g. switches).
However, the data may be buffered at the remote NIC and has not
necessarily been written to the target memory. As a result, data sent in
the request may not be visible to all processes. The third completion
mode is delivery complete.

Delivery complete indicates that the results of the operation are
available to all processes on the fabric. The distinction between
transmit and delivery complete is subtle, but important. It often deals
with *when* the target endpoint generates an acknowledgment to a
message. For providers that offload transport protocol to the NIC,
support for transmit complete is common. Delivery complete guarantees
are more easily met by providers that implement portions of their
protocol on the host processor. Delivery complete corresponds to the
FI_DELIVERY_COMPLETE operation flag.

Applications can request a default completion mode when opening an
endpoint by setting one of the above mentioned complete flags as an
op_flags for the context's attributes. However, it is usually
recommended that application use the provider's default flags for best
performance, and amend its protocol to achieve its completion semantics.
For example, many applications will perform a 'finalize' or 'commit'
procedure as part of their operation, which synchronizes the processing
of all peers and guarantees that all previously sent data has been
received.

A full discussion of completion semantics is given in the fi_cq man
page.

## CQs (fid_cq)

Completion queues often map directly to provider hardware mechanisms,
and libfabric is designed around minimizing the software impact of
accessing those mechanisms. Unlike other objects discussed so far
(fabrics, domains, endpoints), completion queues are not part of the
fi_info structure or involved with the fi_getinfo() call.

All active endpoints must be bound with one or more completion queues.
This is true even if completions will be suppressed by the application
(e.g. using the FI_SELECTIVE_COMPLETION flag). Completion queues are
needed to report operations that complete in error and help drive
progress in the case of manual progress.

CQs are allocated separately from endpoints and are associated with
endpoints through the fi_ep_bind() function.

## CQ Format (fi_cq_format)

In order to minimize the amount of data that a provider must report, the
type of completion data written back to the application is select-able.
This limits the number of bytes the provider writes to memory, and
allows necessary completion data to fit into a compact structure. Each
CQ format maps to a specific completion structure. Developers should
analyze each structure, select the smallest structure that contains all
of the data it requires, and specify the corresponding enum value as the
CQ format.

For example, if an application only needs to know which request
completed, along with the size of a received message, it can select the
following:

    cq_attr->format = FI_CQ_FORMAT_MSG;

    struct fi_cq_msg_entry {
        void      *op_context;
        uint64_t  flags;
        size_t    len;
    };

Once the format has been selected, the underlying provider will assume
that read operations against the CQ will pass in an array of the
corresponding structure. The CQ data formats are designed such that a
structure that reports more information can be cast to one that reports
less.

## Reading Completions (fi_cq_read)

Completions may be read from a CQ by using one of the non-blocking
calls, fi_cq_read / fi_cq_readfrom, or one of the blocking calls,
fi_cq_sread / fi_cq_sreadfrom. Regardless of which call is used,
applications pass in an array of completion structures based on the
selected CQ format. The CQ interfaces are optimized for batch completion
processing, allowing the application to retrieve multiple completions
from a single read call. The difference between the read and readfrom
calls is that readfrom returns source addressing data, if available. The
readfrom derivative of the calls is only useful for unconnected
endpoints, and only if the corresponding endpoint has been configured
with the FI_SOURCE capability.

FI_SOURCE requires that the provider use the source address available in
the raw completion data, such as the packet's source address, to
retrieve a matching entry in the endpoint's address vector. Applications
that carry some sort of source identifier as part of their data packets
can avoid the overhead associated with using FI_SOURCE.

### Retrieving Errors

Because the selected completion structure is insufficient to report all
data necessary to debug or handle an operation that completes in error,
failed operations are reported using a separate fi_cq_readerr()
function. This call takes as input a CQ error entry structure, which
allows the provider to report more information regarding the reason for
the failure.

    /* read error prototype */
    fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf, uint64_t flags);

    /* error data structure */
    struct fi_cq_err_entry {
        void      *op_context;
        uint64_t  flags;
        size_t    len;
        void      *buf;
        uint64_t  data;
        uint64_t  tag;
        size_t    olen;
        int       err;
        int       prov_errno;
        void      *err_data;
        size_t    err_data_size;
    };

    /* Sample error handling */
    struct fi_cq_msg_entry entry;
    struct fi_cq_err_entry err_entry;
    char err_data[256];
    int ret;

    err_entry.err_data = err_data;
    err_entry.err_data_size = 256;

    ret = fi_cq_read(cq, &entry, 1);
    if (ret == -FI_EAVAIL)
        ret = fi_cq_readerr(cq, &err_entry, 0);

As illustrated, if an error entry has been inserted into the completion
queue, then attempting to read the CQ will result in the read call
returning -FI_EAVAIL (error available). This indicates that the
application must use the fi_cq_readerr() call to remove the failed
operation's completion information before other completions can be
reaped from the CQ.

A fabric error code regarding the failure is reported as the err field
in the fi_cq_err_entry structure. A provider specific error code is also
available through the prov_errno field. This field can be decoded into a
displayable string using the fi_cq_strerror() routine. The err_data
field is provider specific data that assists the provider in decoding
the reason for the failure.

# Address Vectors (fid_av)

A primary goal of address vectors is to allow applications to
communicate with thousands to millions of peers while minimizing the
amount of data needed to store peer addressing information. It pushes
fabric specific addressing details away from the application to the
provider. This allows the provider to optimize how it converts addresses
into routing data, and enables data compression techniques that may be
difficult for an application to achieve without being aware of low-level
fabric addressing details. For example, providers may be able to
algorithmically calculate addressing components, rather than storing the
data locally. Additionally, providers can communicate with resource
management entities or fabric manager agents to obtain quality of
service or other information about the fabric, in order to improve
network utilization.

An equally important objective is ensuring that the resulting
interfaces, particularly data transfer operations, are fast and easy to
use. Conceptually, an address vector converts an endpoint address into
an fi_addr_t. The fi_addr_t (fabric interface address datatype) is a
64-bit value that is used in all 'fast-path' operations -- data
transfers and completions.

Address vectors are associated with domain objects. This allows
providers to implement portions of an address vector, such as quality of
service mappings, in hardware.

## AV Type (fi_av_type)

There are two types of address vectors. The type refers to the format of
the returned fi_addr_t values for addresses that are inserted into the
AV. With type FI_AV_TABLE, returned addresses are simple indices, and
developers may think of the AV as an array of addresses. Each address
that is inserted into the AV is mapped to the index of the next free
array slot. The advantage of FI_AV_TABLE is that applications can refer
to peers using a simple index, eliminating an application's need to
store any addressing data. I.e. the application can generate the
fi_addr_t values themselves. This type maps well to applications, such
as MPI, where a peer is referenced by rank.

The second type is FI_AV_MAP. This type does not define any specific
format for the fi_addr_t value. Applications that use type map are
required to provide the correct fi_addr_t for a given peer when issuing
a data transfer operation. The advantage of FI_AV_MAP is that a provider
can use the fi_addr_t to encode the target's address, which avoids
retrieving the data from memory. As a simple example, consider a fabric
that uses TCP/IPv4 based addressing. An fi_addr_t is large enough to
contain the address, which allows a provider to copy the data from the
fi_addr_t directly into an outgoing packet.

## Sharing AVs Between Processes

Large scale parallel programs typically run with multiple processes
allocated on each node. Because these processes communicate with the
same set of peers, the addressing data needed by each process is the
same. Libfabric defines a mechanism by which processes running on the
same node may share their address vectors. This allows a system to
maintain a single copy of addressing data, rather than one copy per
process.

Although libfabric does not require any implementation for how an
address vector is shared, the interfaces map well to using shared
memory. Address vectors which will be shared are given an application
specific name. How an application selects a name that avoid conflicts
with unrelated processes, or how it communicates the name with peer
processes is outside the scope of libfabric.

In addition to having a name, a shared AV also has a base map address --
map_addr. Use of map_addr is only important for address vectors that are
of type FI_AV_MAP, and allows applications to share fi_addr_t values.
From the viewpoint of the application, the map_addr is the base value
for all fi_addr_t values. A common use for map_addr is for the process
that creates the initial address vector to request a value from the
provider, exchange the returned map_addr with its peers, and for the
peers to open the shared AV using the same map_addr. This allows the
fi_addr_t values to be stored in shared memory that is accessible by all
peers.

# Using Native Wait Objects: TryWait

There is an important difference between using libfabric completion
objects, versus sockets, that may not be obvious from the discussions so
far. With sockets, the object that is signaled is the same object that
abstracts the queues, namely the file descriptor. When data is received
on a socket, that data is placed in a queue associated directly with the
fd. Reading from the fd retrieves that data. If an application wishes to
block until data arrives on a socket, it calls select() or poll() on the
fd. The fd is signaled when a message is received, which releases the
blocked thread, allowing it to read the fd.

By associating the wait object with the underlying data queue,
applications are exposed to an interface that is easy to use and race
free. If data is available to read from the socket at the time select()
or poll() is called, those calls simply return that the fd is readable.

There are a couple of significant disadvantages to this approach, which
have been discussed previously, but from different perspectives. The
first is that every socket must be associated with its own fd. There is
no way to share a wait object among multiple sockets. (This is a main
reason for the development of epoll semantics). The second is that the
queue is maintained in the kernel, so that the select() and poll() calls
can check them.

Libfabric allows for the separation of the wait object from the data
queues. For applications that use libfabric interfaces to wait for
events, such as fi_cq_sread, this separation is mostly hidden from the
application. The exception is that applications may receive a signal,
but no events are retrieved when a queue is read. This separation allows
the queues to reside in the application's memory space, while wait
objects may still use kernel components. A reason for the latter is that
wait objects may be signaled as part of system interrupt processing,
which would go through a kernel driver.

Applications that want to use native wait objects (e.g. file
descriptors) directly in operating system calls must perform an
additional step in their processing. In order to handle race conditions
that can occur between inserting an event into a completion or event
object and signaling the corresponding wait object, libfabric defines an
'fi_trywait()' function. The fi_trywait implementation is responsible
for handling potential race conditions which could result in an
application either losing events or hanging. The following example
demonstrates the use of fi_trywait().

    /* Get the native wait object -- an fd in this case */
    fi_control(&cq->fid, FI_GETWAIT, (void *) &fd);
    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    while (1) {
        ret = fi_trywait(fabric, &cq->fid, 1);
        if (ret == FI_SUCCESS) {
            /* It’s safe to block on the fd */
            select(fd + 1, &fds, NULL, &fds, &timeout);
        } else if (ret == -FI_EAGAIN) {
            /* Read and process all completions from the CQ */
            do {
                ret = fi_cq_read(cq, &comp, 1);
            } while (ret > 0);
        } else {
            /* something really bad happened */
        }
    }

In this example, the application has allocated a CQ with an fd as its
wait object. It calls select() on the fd. Before calling select(), the
application must call fi_trywait() successfully (return code of
FI_SUCCESS). Success indicates that a blocking operation can now be
invoked on the native wait object without fear of the application
hanging or events being lost. If fi_trywait() returns --FI_EAGAIN, it
usually indicates that there are queued events to process.

# Environment Variables

Environment variables are used by providers to configure internal
options for optimal performance or memory consumption. Libfabric
provides an interface for querying which environment variables are
usable, along with an application to display the information to a
command window. Although environment variables are usually configured by
an administrator, an application can query for variables
programmatically.

    /* APIs to query for supported environment variables */
    enum fi_param_type {
        FI_PARAM_STRING,
        FI_PARAM_INT,
        FI_PARAM_BOOL,
        FI_PARAM_SIZE_T,
    };

    struct fi_param {
        /* The name of the environment variable */
        const char *name;
        /* What type of value it stores */
        enum fi_param_type type;
        /* A description of how the variable is used */
        const char *help_string;
        /* The current value of the variable */
        const char *value;
    };

    int fi_getparams(struct fi_param **params, int *count);
    void fi_freeparams(struct fi_param *params);

The modification of environment variables is typically a tuning activity
done on larger clusters. However there are a few values that are useful
for developers. These can be seen by executing the fi_info command.

    $ fi_info -e
    # FI_LOG_LEVEL: String
    # Specify logging level: warn, trace, info, debug (default: warn)

    # FI_LOG_PROV: String
    # Specify specific provider to log (default: all)

    # FI_PROVIDER: String
    # Only use specified provider (default: all available)

The fi_info application, which ships with libfabric, can be used to list
all environment variables for all providers. The '-e' option will list
all variables, and the '-g' option can be used to filter the output to
only those variables with a matching substring. Variables are documented
directly in code with the description available as the help_string
output.

The FI_LOG_LEVEL can be used to increase the debug output from libfabric
and the providers. Note that in the release build of libfabric, debug
output from data path operations (transmit, receive, and completion
processing) may not be available. The FI_PROVIDER variable can be used
to enable or disable specific providers. This is useful to ensure that a
given provider will be used.

{% include JB/setup %}

# NAME

fabric - Fabric Interface Library

# SYNOPSIS

``` c
#include <rdma/fabric.h>
```

Libfabric is a high-performance fabric software library designed to
provide low-latency interfaces to fabric hardware. For an in-depth
discussion of the motivation and design see
[`fi_guide`(7)](fi_guide.7.html).

# OVERVIEW

Libfabric provides 'process direct I/O' to application software
communicating across fabric software and hardware. Process direct I/O,
historically referred to as RDMA, allows an application to directly
access network resources without operating system interventions. Data
transfers can occur directly to and from application memory.

There are two components to the libfabric software:

*Fabric Providers*
:   Conceptually, a fabric provider may be viewed as a local hardware
    NIC driver, though a provider is not limited by this definition. The
    first component of libfabric is a general purpose framework that is
    capable of handling different types of fabric hardware. All fabric
    hardware devices and their software drivers are required to support
    this framework. Devices and the drivers that plug into the libfabric
    framework are referred to as fabric providers, or simply providers.
    Provider details may be found in
    [`fi_provider`(7)](fi_provider.7.html).

*Fabric Interfaces*
:   The second component is a set of communication operations. Libfabric
    defines several sets of communication functions that providers can
    support. It is not required that providers implement all the
    interfaces that are defined; however, providers clearly indicate
    which interfaces they do support.

# FABRIC INTERFACES

The fabric interfaces are designed such that they are cohesive and not
simply a union of disjoint interfaces. The interfaces are logically
divided into two groups: control interfaces and communication
operations. The control interfaces are a common set of operations that
provide access to local communication resources, such as address vectors
and event queues. The communication operations expose particular models
of communication and fabric functionality, such as message queues,
remote memory access, and atomic operations. Communication operations
are associated with fabric endpoints.

Applications will typically use the control interfaces to discover local
capabilities and allocate necessary resources. They will then allocate
and configure a communication endpoint to send and receive data, or
perform other types of data transfers, with remote endpoints.

# CONTROL INTERFACES

The control interfaces APIs provide applications access to network
resources. This involves listing all the interfaces available, obtaining
the capabilities of the interfaces and opening a provider.

*fi_getinfo - Fabric Information*
:   The fi_getinfo call is the base call used to discover and request
    fabric services offered by the system. Applications can use this
    call to indicate the type of communication that they desire. The
    results from fi_getinfo, fi_info, are used to reserve and configure
    fabric resources.

fi_getinfo returns a list of fi_info structures. Each structure
references a single fabric provider, indicating the interfaces that the
provider supports, along with a named set of resources. A fabric
provider may include multiple fi_info structures in the returned list.

*fi_fabric - Fabric Domain*
:   A fabric domain represents a collection of hardware and software
    resources that access a single physical or virtual network. All
    network ports on a system that can communicate with each other
    through the fabric belong to the same fabric domain. A fabric domain
    shares network addresses and can span multiple providers. libfabric
    supports systems connected to multiple fabrics.

*fi_domain - Access Domains*
:   An access domain represents a single logical connection into a
    fabric. It may map to a single physical or virtual NIC or a port. An
    access domain defines the boundary across which fabric resources may
    be associated. Each access domain belongs to a single fabric domain.

*fi_endpoint - Fabric Endpoint*
:   A fabric endpoint is a communication portal. An endpoint may be
    either active or passive. Passive endpoints are used to listen for
    connection requests. Active endpoints can perform data transfers.
    Endpoints are configured with specific communication capabilities
    and data transfer interfaces.

*fi_eq - Event Queue*
:   Event queues, are used to collect and report the completion of
    asynchronous operations and events. Event queues report events that
    are not directly associated with data transfer operations.

*fi_cq - Completion Queue*
:   Completion queues are high-performance event queues used to report
    the completion of data transfer operations.

*fi_cntr - Event Counters*
:   Event counters are used to report the number of completed
    asynchronous operations. Event counters are considered light-weight,
    in that a completion simply increments a counter, rather than
    placing an entry into an event queue.

*fi_mr - Memory Region*
:   Memory regions describe application local memory buffers. In order
    for fabric resources to access application memory, the application
    must first grant permission to the fabric provider by constructing a
    memory region. Memory regions are required for specific types of
    data transfer operations, such as RMA transfers (see below).

*fi_av - Address Vector*
:   Address vectors are used to map higher level addresses, such as IP
    addresses, which may be more natural for an application to use, into
    fabric specific addresses. The use of address vectors allows
    providers to reduce the amount of memory required to maintain large
    address look-up tables, and eliminate expensive address resolution
    and look-up methods during data transfer operations.

# DATA TRANSFER INTERFACES

Fabric endpoints are associated with multiple data transfer interfaces.
Each interface set is designed to support a specific style of
communication, with an endpoint allowing the different interfaces to be
used in conjunction. The following data transfer interfaces are defined
by libfabric.

*fi_msg - Message Queue*
:   Message queues expose a simple, message-based FIFO queue interface
    to the application. Message data transfers allow applications to
    send and receive data with message boundaries being maintained.

*fi_tagged - Tagged Message Queues*
:   Tagged message lists expose send/receive data transfer operations
    built on the concept of tagged messaging. The tagged message queue
    is conceptually similar to standard message queues, but with the
    addition of 64-bit tags for each message. Sent messages are matched
    with receive buffers that are tagged with a similar value.

*fi_rma - Remote Memory Access*
:   RMA transfers are one-sided operations that read or write data
    directly to a remote memory region. Other than defining the
    appropriate memory region, RMA operations do not require interaction
    at the target side for the data transfer to complete.

*fi_atomic - Atomic*
:   Atomic operations can perform one of several operations on a remote
    memory region. Atomic operations include well-known functionality,
    such as atomic-add and compare-and-swap, plus several other
    pre-defined calls. Unlike other data transfer interfaces, atomic
    operations are aware of the data formatting at the target memory
    region.

# LOGGING INTERFACE

Logging can be controlled using the FI_LOG_LEVEL, FI_LOG_PROV, and
FI_LOG_SUBSYS environment variables.

*FI_LOG_LEVEL*
:   FI_LOG_LEVEL controls the amount of logging data that is output. The
    following log levels are defined.

\- *Warn*
:   Warn is the least verbose setting and is intended for reporting
    errors or warnings.

\- *Trace*
:   Trace is more verbose and is meant to include non-detailed output
    helpful to tracing program execution.

\- *Info*
:   Info is high traffic and meant for detailed output.

\- *Debug*
:   Debug is high traffic and is likely to impact application
    performance. Debug output is only available if the library has been
    compiled with debugging enabled.

*FI_LOG_PROV*
:   The FI_LOG_PROV environment variable enables or disables logging
    from specific providers. Providers can be enabled by listing them in
    a comma separated fashion. If the list begins with the '\^' symbol,
    then the list will be negated. By default all providers are enabled.

Example: To enable logging from the psm3 and sockets provider:
FI_LOG_PROV="psm3,sockets"

Example: To enable logging from providers other than psm3:
FI_LOG_PROV="\^psm3"

*FI_LOG_SUBSYS*
:   The FI_LOG_SUBSYS environment variable enables or disables logging
    at the subsystem level. The syntax for enabling or disabling
    subsystems is similar to that used for FI_LOG_PROV. The following
    subsystems are defined.

\- *core*
:   Provides output related to the core framework and its management of
    providers.

\- *fabric*
:   Provides output specific to interactions associated with the fabric
    object.

\- *domain*
:   Provides output specific to interactions associated with the domain
    object.

\- *ep_ctrl*
:   Provides output specific to endpoint non-data transfer operations,
    such as CM operations.

\- *ep_data*
:   Provides output specific to endpoint data transfer operations.

\- *av*
:   Provides output specific to address vector operations.

\- *cq*
:   Provides output specific to completion queue operations.

\- *eq*
:   Provides output specific to event queue operations.

\- *mr*
:   Provides output specific to memory registration.

# PROVIDER INSTALLATION AND SELECTION

The libfabric build scripts will install all providers that are
supported by the installation system. Providers that are missing build
prerequisites will be disabled. Installed providers will dynamically
check for necessary hardware on library initialization and respond
appropriately to application queries.

Users can enable or disable available providers through build
configuration options. See 'configure --help' for details. In general, a
specific provider can be controlled using the configure option
'--enable-`<provider_name>`{=html}'. For example, '--enable-udp' (or
'--enable-udp=yes') will add the udp provider to the build. To disable
the provider, '--enable-udp=no' can be used. To build the provider as a
stand-alone dynamically loadable library (i.e. DL provider),
'--enable-udp=dl' can be used.

Providers can also be enable or disabled at run time using the
FI_PROVIDER environment variable. The FI_PROVIDER variable is set to a
comma separated list of providers to include. If the list begins with
the '\^' symbol, then the list will be negated.

Example: To enable the udp and tcp providers only, set:
`FI_PROVIDER="udp,tcp"`

When libfabric is installed, DL providers are put under the *default
provider path*, which is determined by how libfabric is built and
installed. Usually the default provider path is
`<libfabric-install-dir>/lib/libfabric` or
`<libfabric-install-dir>/lib64/libfabric`. By default, libfabric tries
to find DL providers in the following order:

1.  Use 'dlopen' to load provider libraries named `lib<prov_name>-fi.so`
    for all providers enabled at build time. The search path of 'ld.so'
    is used to locate the files. This step is skipped if libfabric is
    configured with the option '--enable-restricted-dl'.

2.  Try to load every file under the default provider path as a DL
    provider.

The FI_PROVIDER_PATH variable can be used to change the location to
search for DL providers and how to resolve conflicts if multiple
providers with the same name are found. Setting FI_PROVIDER_PATH to any
value, even if empty, would cause step 1 be skipped, and may change the
search directory used in step 2.

In the simplest form, the FI_PROVIDER_PATH variable is set to a colon
separated list of directories. These directories replace the default
provider path used in step 2. For example:

    FI_PROVIDER_PATH=/opt/libfabric:/opt/libfabric2

By default, if multiple providers (including the built-in providers)
with the same name are found, the first one with the highest version is
active and all the others are hidden. This can be changed by setting the
FI_PROVIDER_PATH variable to start with '@', which force the first one
to be active regardless of the version. For example:

    FI_PROVIDER_PATH=@/opt/libfabric:/opt/libfabric2

The FI_PROVIDER_PATH variable can also specify preferred providers by
supplying full paths to libraries instead of directories to search
under. A preferred provider takes precedence over other providers with
the same name. The specification of a preferred provider must be
prefixed with '+'. For example:

    FI_PROVIDER_PATH=+/opt/libfabric2/libtcp-fi.so:/opt/libfabric:+/opt/libfabric2/libudp-fi.so

If FI_PROVIDER_PATH is set, but no directory is supplied, the default
provider path is used. Some examples:

    FI_PROVIDER_PATH=
    FI_PROVIDER_PATH=@
    FI_PROVIDER_PATH=+/opt/libfabric/libtcp-fi.so
    FI_PROVIDER_PATH=@+/opt/libfabric/libtcp-fi.so

The fi_info utility, which is included as part of the libfabric package,
can be used to retrieve information about which providers are available
in the system. Additionally, it can retrieve a list of all environment
variables that may be used to configure libfabric and each provider. See
[`fi_info`(1)](fi_info.1.html) for more details.

# ENVIRONMENT VARIABLE CONTROLS

Core features of libfabric and its providers may be configured by an
administrator through the use of environment variables. Man pages will
usually describe the most commonly accessed variables, such as those
mentioned above. However, libfabric defines interfaces for publishing
and obtaining environment variables. These are targeted for providers,
but allow applications and users to obtain the full list of variables
that may be set, along with a brief description of their use.

A full list of variables available may be obtained by running the
fi_info application, with the -e or --env command line option.

# NOTES

## System Calls

Because libfabric is designed to provide applications direct access to
fabric hardware, there are limits on how libfabric resources may be used
in conjunction with system calls. These limitations are notable for
developers who may be familiar programming to the sockets interface.
Although limits are provider specific, the following restrictions apply
to many providers and should be adhered to by applications desiring
portability across providers.

*fork*
:   Fabric resources are not guaranteed to be available by child
    processes. This includes objects, such as endpoints and completion
    queues, as well as application controlled data buffers which have
    been assigned to the network. For example, data buffers that have
    been registered with a fabric domain may not be available in a child
    process because of copy on write restrictions.

## CUDA deadlock

In some cases, calls to `cudaMemcpy()` within libfabric may result in a
deadlock. This typically occurs when a CUDA kernel blocks until a
`cudaMemcpy` on the host completes. Applications which can cause such
behavior can restrict Libfabric's ability to invoke CUDA API operations
with the endpoint option `FI_OPT_CUDA_API_PERMITTED`. See
[`fi_endpoint`(3)](fi_endpoint.3.html) for more details.

Another mechanism which can be used to avoid deadlock is Nvidia's
GDRCopy. Using GDRCopy requires an external library and kernel module
available at https://github.com/NVIDIA/gdrcopy. Libfabric must be
configured with GDRCopy support using the `--with-gdrcopy` option, and
be run with `FI_HMEM_CUDA_USE_GDRCOPY=1`. This may not be supported by
all providers.

# ABI CHANGES

libfabric releases maintain compatibility with older releases, so that
compiled applications can continue to work as-is, and previously written
applications will compile against newer versions of the library without
needing source code changes. The changes below describe ABI updates that
have occurred and which libfabric release corresponds to the changes.

Note that because most functions called by applications actually call
static inline functions, which in turn reference function pointers in
order to call directly into providers, libfabric only exports a handful
of functions directly. ABI changes are limited to those functions, most
notably the fi_getinfo call and its returned attribute structures.

The ABI version is independent from the libfabric release version.

## ABI 1.0

The initial libfabric release (1.0.0) also corresponds to ABI version
1.0. The 1.0 ABI was unchanged for libfabric major.minor versions 1.0,
1.1, 1.2, 1.3, and 1.4.

## ABI 1.1

A number of external data structures were appended starting with
libfabric version 1.5. These changes included adding the fields to the
following data structures. The 1.1 ABI was exported by libfabric
versions 1.5 and 1.6.

*fi_fabric_attr*
:   Added api_version

*fi_domain_attr*
:   Added cntr_cnt, mr_iov_limit, caps, mode, auth_key, auth_key_size,
    max_err_data, and mr_cnt fields. The mr_mode field was also changed
    from an enum to an integer flag field.

*fi_ep_attr*
:   Added auth_key_size and auth_key fields.

## ABI 1.2

The 1.2 ABI version was exported by libfabric versions 1.7 and 1.8, and
expanded the following structure.

*fi_info*
:   The fi_info structure was expanded to reference a new fabric object,
    fid_nic. When available, the fid_nic references a new set of
    attributes related to network hardware details.

## ABI 1.3

The 1.3 ABI version was exported by libfabric versions 1.9, 1.10, and
1.11. Added new fields to the following attributes:

*fi_domain_attr*
:   Added tclass

*fi_tx_attr*
:   Added tclass

## ABI 1.4

The 1.4 ABI version was exported by libfabric 1.12. Added fi_tostr_r, a
thread-safe (re-entrant) version of fi_tostr.

## ABI 1.5

ABI version starting with libfabric 1.13. Added new fi_open API call.

## ABI 1.6

ABI version starting with libfabric 1.14. Added fi_log_ready for
providers.

## ABI 1.7

ABI version starting with libfabric 1.20. Added new fields to the
following attributes:

*fi_domain_attr*
:   Added max_ep_auth_key

## ABI 1.8

ABI version starting with libfabric 2.0. Added new fi_fabric2 API call.
Added new fields to the following attributes:

*fi_domain_attr*
:   Added max_group_id

# SEE ALSO

[`fi_info`(1)](fi_info.1.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_av`(3)](fi_av.3.html),
[`fi_eq`(3)](fi_eq.3.html), [`fi_cq`(3)](fi_cq.3.html),
[`fi_cntr`(3)](fi_cntr.3.html), [`fi_mr`(3)](fi_mr.3.html)

{% include JB/setup %}

# NAME

fi_direct - Direct fabric provider access

# SYNOPSIS

``` c
-DFABRIC_DIRECT

#define FABRIC_DIRECT
```

Fabric direct provides a mechanism for applications to compile against a
specific fabric providers without going through the libfabric framework
or function vector tables. This allows for extreme optimization via
function inlining at the cost of supporting multiple providers or
different versions of the same provider.

# DESCRIPTION

The use of fabric direct is intended only for applications that require
the absolute minimum software latency, and are willing to re-compile for
specific fabric hardware. Providers that support fabric direct implement
their own versions of the static inline calls which are define in the
libfabric header files, define selected enum values, and provide defines
for compile-time optimizations. Applications can then code against the
standard libfabric calls, but link directly against the provider calls
by defining FABRIC_DIRECT as part of their build.

In general, the use of fabric direct does not require application source
code changes, and, instead, is limited to the build process.

Providers supporting fabric direct must install 'direct' versions of all
libfabric header files. For convenience, the libfabric sources contain
sample header files that may be modified by a provider. The 'direct'
header file names have 'fi_direct' as their prefix: fi_direct.h,
fi_direct_endpoint.h, etc.

Direct providers are prohibited from overriding or modifying existing
data structures. However, provider specific extensions are still
available. In addition to provider direct function calls to provider
code, a fabric direct provider may define zero of more of the following
capability definitions. Applications can check for these capabilities in
order to optimize code paths at compile time, versus relying on run-time
checks.

# CAPABILITY DEFINITIONS

In order that application code may be optimized during compile time,
direct providers must provide definitions for various capabilities and
modes, if those capabilities are supported. The following #define values
may be used by an application to test for provider support of supported
features.

*FI_DIRECT_CONTEXT*
:   The provider sets FI_CONTEXT or FI_CONTEXT2 for fi_info:mode. See
    fi_getinfo for additional details. When FI_DIRECT_CONTEXT is
    defined, applications should use struct fi_context in their
    definitions, even if FI_CONTEXT2 is set.

*FI_DIRECT_LOCAL_MR*
:   The provider sets FI_LOCAL_MR for fi_info:mode. See fi_getinfo for
    additional details.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html)

{% include JB/setup %}

# NAME

fi_provider - Fabric Interface Providers

# OVERVIEW

See [`fi_arch`(7)](fi_arch.7.html) for a brief description of how
providers fit into the libfabric architecture.

Conceptually, a fabric provider implements and maps the libfabric API
over lower-level network software and/or hardware. Most application
calls into the libfabric API go directly into a provider's
implementation of that API.

Libfabric providers are grouped into different type: core, utility,
hooking, and offload. These are describe in more detail below. The
following diagram illustrates the architecture between the provider
types.

    ---------------------------- libfabric API ---------------------------- 
      [core]   provider|<- [hooking provider]
    [services]   API   |  --- libfabric API --- 
                       |<- [utility provider]
                       |  ---------------- libfabric API ------------------ 
                       |<-  [core provider] <-peer API-> [offload provider]

All providers plug into libfabric using an exported provider API.
libfabric supports both internal providers, which ship with the library
for user convenience, as well as external providers. External provider
libraries must be in the library search path, end with the suffix "-fi",
and export the function fi_prov_ini().

Once registered with the libfabric core, a provider will be reported to
applications as part of the discovery process. Hooking and utility
providers will intercept libfabric calls from the application to perform
some task before calling through to the next provider. If there's no
need to intercept a specific API call, the application will call
directly to the core provider. Where possible provider to provider
communication is done using the libfabric APIs itself, including the use
of provider specific extensions to reduce call overhead.

libfabric defines a set of APIs that specifically target providers that
may be used as peers. These APIs are oddly enough called peer APIs. Peer
APIs are technically part of the external libfabric API, but are not
designed for direct use by applications and are not considered stable
for API backwards compatibility.

# Core Providers

Core providers are stand-alone providers that usually target a specific
class of networking devices. That is, a specific NIC, class of network
hardware, or lower-level software interface. The core providers are
usually what most application developers are concerned with. Core
providers may only support libfabric features and interfaces that map
efficiently to the underlying hardware or network protocols.

The following core providers are built into libfabric by default,
assuming all build pre-requisites are met. That is, necessary libraries
are installed, operating system support is available, etc. This list is
not exhaustive.

*CXI*
:   Provider for Cray's Slingshot network. See
    [`fi_cxi`(7)](fi_cxi.7.html) for more information.

*EFA*
:   A provider for the [Amazon EC2 Elastic Fabric Adapter
    (EFA)](https://aws.amazon.com/hpc/efa/), a custom-built OS bypass
    hardware interface for inter-instance communication on EC2. See
    [`fi_efa`(7)](fi_efa.7.html) for more information.

*LPP*
:   A provider runs on FabreX PCIe networks. See
    [`fi_lpp`(7)](fi_lpp.7.html) for more information.

*OPX*
:   Supports Omni-Path networking from Cornelis Networks. See
    [`fi_opx`(7)](fi_opx.7.html) for more information.

*PSM2*
:   Older provider for Omni-Path networks. See
    [`fi_psm2`(7)](fi_psm2.7.html) for more information.

*PSM3*
:   Provider for Ethernet networking from Intel. See
    [`fi_psm3`(7)](fi_psm3.7.html) for more information.

*SHM*
:   A provider for intra-node communication using shared memory. See
    [`fi_shm`(7)](fi_shm.7.html) for more information.

*TCP*
:   A provider which runs over the TCP/IP protocol and is available on
    multiple operating systems. This provider enables develop of
    libfabric applications on most platforms. See
    [`fi_tcp`(7)](fi_tcp.7.html) for more information.

*UCX*
:   A provider which runs over the UCX library which is currently
    supported by Infiniband fabrics from NVIDIA. See
    [`fi_ucx`(7)](fi_ucx.7.html) for more information.

*UDP*
:   A provider which runs over the UDP/IP protocol and is available on
    multiple operating systems. This provider enables develop of
    libfabric applications on most platforms. See
    [`fi_udp`(7)](fi_udp.7.html) for more information.

*Verbs*
:   This provider targets RDMA NICs for both Linux and Windows
    platforms. See [`fi_verbs`(7)](fi_verbs.7.html) for more
    information.

# Utility Providers

Utility providers are named with a starting prefix of "ofi\_". Utility
providers are distinct from core providers in that they are not
associated with specific classes of devices. They instead work with core
providers to expand their features and interact with core providers
through libfabric interfaces internally. Utility providers are used to
support a specific endpoint type over a simpler endpoint type.

Utility providers show up as part of the return's provider's name. See
[`fi_fabric`(3)](fi_fabric.3.html). Utility providers are enabled
automatically for core providers that do not support the feature set
requested by an application.

*RxM*
:   Implements RDM endpoint semantics over MSG endpoints. See
    [`fi_rxm`(7)](fi_rxm.7.html) for more information.

*RxD*
:   Implements RDM endpoint semantis over DGRAM endpoints. See
    [`fi_rxd`(7)](fi_rxd.7.html) for more information.

# Hooking Providers

Hooking providers are mostly used for debugging purposes. Since hooking
providers are built and included in release versions of libfabric, they
are always available and have no impact on performance unless enabled.
Hooking providers can layer over all other providers and intercept, or
hook, their calls in order to perform some dedicated task, such as
gathering performance data on call paths or providing debug output.

See [`fi_hook`(7)](fi_hook.7.html) for more information.

# Offload Providers

Offload providers start with the naming prefix "off\_". An offload
provider is meant to be paired with other core and/or utility providers.
An offload provider is intended to accelerate specific types of
communication, generally by taking advantage of network services that
have been offloaded into hardware, though actual hardware offload
support is not a requirement.

# LINKx (LNX) provider

The LNX provider is designed to link two or more providers, allowing
applications to seamlessly use multiple providers or NICs. This provider
uses the libfabric peer infrastructure to aid in the use of the
underlying providers. This version of the provider is able to link any
libfabric provider which supports the FI_PEER capability.

See [`fi_lnx`(7)](fi_lnx.7.html) for more information.

# SEE ALSO

[`fabric`(7)](fabric.7.html) [`fi_provider`(3)](fi_provider.3.html)

{% include JB/setup %}

# NAME

fi_atomic - Remote atomic functions

fi_atomic / fi_atomicv / fi_atomicmsg / fi_inject_atomic
:   Initiates an atomic operation to remote memory

fi_fetch_atomic / fi_fetch_atomicv / fi_fetch_atomicmsg
:   Initiates an atomic operation to remote memory, retrieving the
    initial value.

fi_compare_atomic / fi_compare_atomicv / fi_compare_atomicmsg
:   Initiates an atomic compare-operation to remote memory, retrieving
    the initial value.

fi_atomicvalid / fi_fetch_atomicvalid / fi_compare_atomicvalid /
fi_query_atomic : Indicates if a provider supports a specific atomic
operation

# SYNOPSIS

``` c
#include <rdma/fi_atomic.h>

ssize_t fi_atomic(struct fid_ep *ep, const void *buf,
    size_t count, void *desc, fi_addr_t dest_addr,
    uint64_t addr, uint64_t key,
    enum fi_datatype datatype, enum fi_op op, void *context);

ssize_t fi_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
    void **desc, size_t count, fi_addr_t dest_addr,
    uint64_t addr, uint64_t key,
    enum fi_datatype datatype, enum fi_op op, void *context);

ssize_t fi_atomicmsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
    uint64_t flags);

ssize_t fi_inject_atomic(struct fid_ep *ep, const void *buf,
    size_t count, fi_addr_t dest_addr,
    uint64_t addr, uint64_t key,
    enum fi_datatype datatype, enum fi_op op);

ssize_t fi_fetch_atomic(struct fid_ep *ep, const void *buf,
    size_t count, void *desc, void *result, void *result_desc,
    fi_addr_t dest_addr, uint64_t addr, uint64_t key,
    enum fi_datatype datatype, enum fi_op op, void *context);

ssize_t fi_fetch_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
    void **desc, size_t count, struct fi_ioc *resultv,
    void **result_desc, size_t result_count, fi_addr_t dest_addr,
    uint64_t addr, uint64_t key, enum fi_datatype datatype,
    enum fi_op op, void *context);

ssize_t fi_fetch_atomicmsg(struct fid_ep *ep,
    const struct fi_msg_atomic *msg, struct fi_ioc *resultv,
    void **result_desc, size_t result_count, uint64_t flags);

ssize_t fi_compare_atomic(struct fid_ep *ep, const void *buf,
    size_t count, void *desc, const void *compare,
    void *compare_desc, void *result, void *result_desc,
    fi_addr_t dest_addr, uint64_t addr, uint64_t key,
    enum fi_datatype datatype, enum fi_op op, void *context);

size_t fi_compare_atomicv(struct fid_ep *ep, const struct fi_ioc *iov,
       void **desc, size_t count, const struct fi_ioc *comparev,
       void **compare_desc, size_t compare_count, struct fi_ioc *resultv,
       void **result_desc, size_t result_count, fi_addr_t dest_addr,
       uint64_t addr, uint64_t key, enum fi_datatype datatype,
       enum fi_op op, void *context);

ssize_t fi_compare_atomicmsg(struct fid_ep *ep,
    const struct fi_msg_atomic *msg, const struct fi_ioc *comparev,
    void **compare_desc, size_t compare_count,
    struct fi_ioc *resultv, void **result_desc, size_t result_count,
    uint64_t flags);

int fi_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
    enum fi_op op, size_t *count);

int fi_fetch_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
    enum fi_op op, size_t *count);

int fi_compare_atomicvalid(struct fid_ep *ep, enum fi_datatype datatype,
    enum fi_op op, size_t *count);

int fi_query_atomic(struct fid_domain *domain,
    enum fi_datatype datatype, enum fi_op op,
    struct fi_atomic_attr *attr, uint64_t flags);
```

# ARGUMENTS

*ep*
:   Fabric endpoint on which to initiate atomic operation.

*buf*
:   Local data buffer that specifies first operand of atomic operation

*iov / comparev / resultv*
:   Vectored data buffer(s).

*count / compare_count / result_count*
:   Count of vectored data entries. The number of elements referenced,
    where each element is the indicated datatype.

*addr*
:   Address of remote memory to access.

*key*
:   Protection key associated with the remote memory.

*datatype*
:   Datatype associated with atomic operands

*op*
:   Atomic operation to perform

*compare*
:   Local compare buffer, containing comparison data.

*result*
:   Local data buffer to store initial value of remote buffer

*desc / compare_desc / result_desc*
:   Data descriptor associated with the local data buffer, local compare
    buffer, and local result buffer, respectively. See
    [`fi_mr`(3)](fi_mr.3.html).

*dest_addr*
:   Destination address for connectionless atomic operations. Ignored
    for connected endpoints.

*msg*
:   Message descriptor for atomic operations

*flags*
:   Additional flags to apply for the atomic operation

*context*
:   User specified pointer to associate with the operation. This
    parameter is ignored if the operation will not generate a successful
    completion, unless an op flag specifies the context parameter be
    used for required input.

# DESCRIPTION

Atomic transfers are used to read and update data located in remote
memory regions in an atomic fashion. Conceptually, they are similar to
local atomic operations of a similar nature (e.g. atomic increment,
compare and swap, etc.). Updates to remote data involve one of several
operations on the data, and act on specific types of data, as listed
below. As such, atomic transfers have knowledge of the format of the
data being accessed. A single atomic function may operate across an
array of data applying an atomic operation to each entry, but the
atomicity of an operation is limited to a single datatype or entry.

## Atomic Data Types

Atomic functions may operate on one of the following identified data
types. A given atomic function may support any datatype, subject to
provider implementation constraints.

*FI_INT8*
:   Signed 8-bit integer.

*FI_UINT8*
:   Unsigned 8-bit integer.

*FI_INT16*
:   Signed 16-bit integer.

*FI_UINT16*
:   Unsigned 16-bit integer.

*FI_INT32*
:   Signed 32-bit integer.

*FI_UINT32*
:   Unsigned 32-bit integer.

*FI_INT64*
:   Signed 64-bit integer.

*FI_UINT64*
:   Unsigned 64-bit integer.

*FI_INT128*
:   Signed 128-bit integer.

*FI_UINT128*
:   Unsigned 128-bit integer.

*FI_FLOAT*
:   A single-precision floating point value (IEEE 754).

*FI_DOUBLE*
:   A double-precision floating point value (IEEE 754).

*FI_FLOAT_COMPLEX*
:   An ordered pair of single-precision floating point values (IEEE
    754), with the first value representing the real portion of a
    complex number and the second representing the imaginary portion.

*FI_DOUBLE_COMPLEX*
:   An ordered pair of double-precision floating point values (IEEE
    754), with the first value representing the real portion of a
    complex number and the second representing the imaginary portion.

*FI_LONG_DOUBLE*
:   A double-extended precision floating point value (IEEE 754). Note
    that the size of a long double and number of bits used for precision
    is compiler, platform, and/or provider specific. Developers that use
    long double should ensure that libfabric is built using a long
    double format that is compatible with their application, and that
    format is supported by the provider. The mechanism used for this
    validation is currently beyond the scope of the libfabric API.

*FI_LONG_DOUBLE_COMPLEX*
:   An ordered pair of double-extended precision floating point values
    (IEEE 754), with the first value representing the real portion of a
    complex number and the second representing the imaginary portion.

*FI_FLOAT16*
:   16-bit half precision floating point value (IEEE 754-2008).

*FI_BFLOAT16*
:   16-bit brain floating point value (IEEE 754-2008).

*FI_FLOAT8_E4M3*
:   8-bit floating point value with 4-bit exponent and 3-bit mantissa.

*FI_FLOAT8_E5M2*
:   8-bit floating point value with 5-bit exponent and 2-bit mantissa.

## Atomic Operations

The following atomic operations are defined. An atomic operation often
acts against a target value in the remote memory buffer and source value
provided with the atomic function. It may also carry source data to
replace the target value in compare and swap operations. A conceptual
description of each operation is provided.

*FI_MIN*
:   Minimum

``` c
if (buf[i] < addr[i])
    addr[i] = buf[i]
```

*FI_MAX*
:   Maximum

``` c
if (buf[i] > addr[i])
    addr[i] = buf[i]
```

*FI_SUM*
:   Sum

``` c
addr[i] = addr[i] + buf[i]
```

*FI_PROD*
:   Product

``` c
addr[i] = addr[i] * buf[i]
```

*FI_LOR*
:   Logical OR

``` c
addr[i] = (addr[i] || buf[i])
```

*FI_LAND*
:   Logical AND

``` c
addr[i] = (addr[i] && buf[i])
```

*FI_BOR*
:   Bitwise OR

``` c
addr[i] = addr[i] | buf[i]
```

*FI_BAND*
:   Bitwise AND

``` c
addr[i] = addr[i] & buf[i]
```

*FI_LXOR*
:   Logical exclusive-OR (XOR)

``` c
addr[i] = ((addr[i] && !buf[i]) || (!addr[i] && buf[i]))
```

*FI_BXOR*
:   Bitwise exclusive-OR (XOR)

``` c
addr[i] = addr[i] ^ buf[i]
```

*FI_ATOMIC_READ*
:   Read data atomically

``` c
result[i] = addr[i]
```

*FI_ATOMIC_WRITE*
:   Write data atomically

``` c
addr[i] = buf[i]
```

*FI_CSWAP*
:   Compare values and if equal swap with data

``` c
if (compare[i] == addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_NE*
:   Compare values and if not equal swap with data

``` c
if (compare[i] != addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_LE*
:   Compare values and if less than or equal swap with data

``` c
if (compare[i] <= addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_LT*
:   Compare values and if less than swap with data

``` c
if (compare[i] < addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_GE*
:   Compare values and if greater than or equal swap with data

``` c
if (compare[i] >= addr[i])
    addr[i] = buf[i]
```

*FI_CSWAP_GT*
:   Compare values and if greater than swap with data

``` c
if (compare[i] > addr[i])
    addr[i] = buf[i]
```

*FI_MSWAP*
:   Swap masked bits with data

``` c
addr[i] = (buf[i] & compare[i]) | (addr[i] & ~compare[i])
```

*FI_DIFF*
:   Calculate the difference

``` c
addr[i] = addr[i] - buf[i]
```

## Base Atomic Functions

The base atomic functions -- fi_atomic, fi_atomicv, fi_atomicmsg -- are
used to transmit data to a remote node, where the specified atomic
operation is performed against the target data. The result of a base
atomic function is stored at the remote memory region. The main
difference between atomic functions are the number and type of
parameters that they accept as input. Otherwise, they perform the same
general function.

The call fi_atomic transfers the data contained in the user-specified
data buffer to a remote node. For connectionless endpoints, the
destination endpoint is specified through the dest_addr parameter.
Unless the endpoint has been configured differently, the data buffer
passed into fi_atomic must not be touched by the application until the
fi_atomic call completes asynchronously. The target buffer of a base
atomic operation must allow for remote read an/or write access, as
appropriate.

The fi_atomicv call adds support for a scatter-gather list to fi_atomic.
The fi_atomicv transfers the set of data buffers referenced by the ioc
parameter to the remote node for processing.

The fi_inject_atomic call is an optimized version of fi_atomic. The
fi_inject_atomic function behaves as if the FI_INJECT transfer flag were
set, and FI_COMPLETION were not. That is, the data buffer is available
for reuse immediately on returning from from fi_inject_atomic, and no
completion event will be generated for this atomic. The completion event
will be suppressed even if the endpoint has not been configured with
FI_SELECTIVE_COMPLETION. See the flags discussion below for more
details. The requested message size that can be used with
fi_inject_atomic is limited by inject_size.

The fi_atomicmsg call supports atomic functions over both connected and
connectionless endpoints, with the ability to control the atomic
operation per call through the use of flags. The fi_atomicmsg function
takes a struct fi_msg_atomic as input.

``` c
struct fi_msg_atomic {
    const struct fi_ioc *msg_iov; /* local scatter-gather array */
    void                **desc;   /* local access descriptors */
    size_t              iov_count;/* # elements in ioc */
    const void          *addr;    /* optional endpoint address */
    const struct fi_rma_ioc *rma_iov; /* remote SGL */
    size_t              rma_iov_count;/* # elements in remote SGL */
    enum fi_datatype    datatype; /* operand datatype */
    enum fi_op          op;       /* atomic operation */
    void                *context; /* user-defined context */
    uint64_t            data;     /* optional data */
};

struct fi_ioc {
    void        *addr;    /* local address */
    size_t      count;    /* # target operands */
};

struct fi_rma_ioc {
    uint64_t    addr;     /* target address */
    size_t      count;    /* # target operands */
    uint64_t    key;      /* access key */
};
```

The following list of atomic operations are usable with base atomic
operations: FI_MIN, FI_MAX, FI_SUM, FI_PROD, FI_LOR, FI_LAND, FI_BOR,
FI_BAND, FI_LXOR, FI_BXOR, and FI_ATOMIC_WRITE.

## Fetch-Atomic Functions

The fetch atomic functions -- fi_fetch_atomic, fi_fetch_atomicv, and
fi_fetch atomicmsg -- behave similar to the equivalent base atomic
function. The difference between the fetch and base atomic calls are the
fetch atomic routines return the initial value that was stored at the
target to the user. The initial value is read into the user provided
result buffer. The target buffer of fetch-atomic operations must be
enabled for remote read access.

The following list of atomic operations are usable with fetch atomic
operations: FI_MIN, FI_MAX, FI_SUM, FI_PROD, FI_LOR, FI_LAND, FI_BOR,
FI_BAND, FI_LXOR, FI_BXOR, FI_ATOMIC_READ, and FI_ATOMIC_WRITE.

For FI_ATOMIC_READ operations, the source buffer operand (e.g.
fi_fetch_atomic buf parameter) is ignored and may be NULL. The results
are written into the result buffer.

## Compare-Atomic Functions

The compare atomic functions -- fi_compare_atomic, fi_compare_atomicv,
and fi_compare atomicmsg -- are used for operations that require
comparing the target data against a value before performing a swap
operation. The compare atomic functions support: FI_CSWAP, FI_CSWAP_NE,
FI_CSWAP_LE, FI_CSWAP_LT, FI_CSWAP_GE, FI_CSWAP_GT, and FI_MSWAP.

## Atomic Valid Functions

The atomic valid functions -- fi_atomicvalid, fi_fetch_atomicvalid, and
fi_compare_atomicvalid --indicate which operations the local provider
supports. Needed operations not supported by the provider must be
emulated by the application. Each valid call corresponds to a set of
atomic functions. fi_atomicvalid checks whether a provider supports a
specific base atomic operation for a given datatype and operation.
fi_fetch_atomicvalid indicates if a provider supports a specific
fetch-atomic operation for a given datatype and operation. And
fi_compare_atomicvalid checks if a provider supports a specified
compare-atomic operation for a given datatype and operation.

If an operation is supported, an atomic valid call will return 0, along
with a count of atomic data units that a single function call will
operate on.

## Query Atomic Attributes

The fi_query_atomic call acts as an enhanced atomic valid operation (see
the atomic valid function definitions above). It is provided, in part,
for future extensibility. The query operation reports which atomic
operations are supported by the domain, for suitably configured
endpoints.

The behavior of fi_query_atomic is adjusted based on the flags
parameter. If flags is 0, then the operation reports the supported
atomic attributes for base atomic operations, similar to fi_atomicvalid
for endpoints. If flags has the FI_FETCH_ATOMIC bit set, the operation
behaves similar to fi_fetch_atomicvalid. Similarly, the flag bit
FI_COMPARE_ATOMIC results in query acting as fi_compare_atomicvalid. The
FI_FETCH_ATOMIC and FI_COMPARE_ATOMIC bits may not both be set.

If the FI_TAGGED bit is set, the provider will indicate if it supports
atomic operations to tagged receive buffers. The FI_TAGGED bit may be
used by itself, or in conjunction with the FI_FETCH_ATOMIC and
FI_COMPARE_ATOMIC flags.

The output of fi_query_atomic is struct fi_atomic_attr:

``` c
struct fi_atomic_attr {
    size_t count;
    size_t size;
};
```

The count attribute field is as defined for the atomic valid calls. The
size field indicates the size in bytes of the atomic datatype. The size
field is useful for datatypes that may differ in sizes based on the
platform or compiler, such FI_LONG_DOUBLE.

## Completions

Completed atomic operations are reported to the initiator of the request
through an associated completion queue or counter. Any user provided
context specified with the request will be returned as part of any
completion event written to a CQ. See fi_cq for completion event
details.

Any results returned to the initiator as part of an atomic operation
will be available prior to a completion event being generated. This will
be true even if the requested completion semantic provides a weaker
guarantee. That is, atomic fetch operations have FI_DELIVERY_COMPLETE
semantics. Completions generated for other types of atomic operations
indicate that it is safe to re-use the source data buffers.

Any updates to data at the target of an atomic operation will be visible
to agents (CPU processes, NICs, and other devices) on the target node
prior to one of the following occurring. If the atomic operation
generates a completion event or updates a completion counter at the
target endpoint, the results will be available prior to the completion
notification. After processing a completion for the atomic, if the
initiator submits a transfer between the same endpoints that generates a
completion at the target, the results will be available prior to the
subsequent transfer's event. Or, if a fenced data transfer from the
initiator follows the atomic request, the results will be available
prior to a completion at the target for the fenced transfer.

The correctness of atomic operations on a target memory region is
guaranteed only when performed by a single actor for a given window of
time. An actor is defined as a single libfabric domain on the target
(identified by the domain name, and not an open instance of that
domain), a coherent CPU complex, or other device (e.g. GPU) capable of
performing atomic operations on the target memory. The results of atomic
operations performed by multiple actors simultaneously are undefined.
For example, issuing CPU based atomic operations to a target region
concurrently being updated by NIC based atomics may leave the region's
data in an unknown state. The results of a first actor's atomic
operations must be visible to a second actor prior to the second actor
issuing its own atomics.

# FLAGS

The fi_atomicmsg, fi_fetch_atomicmsg, and fi_compare_atomicmsg calls
allow the user to specify flags which can change the default data
transfer operation. Flags specified with atomic message operations
override most flags previously configured with the endpoint, except
where noted (see fi_control). The following list of flags are usable
with atomic message calls.

*FI_COMPLETION*
:   Indicates that a completion entry should be generated for the
    specified operation. The endpoint must be bound to a completion
    queue with FI_SELECTIVE_COMPLETION that corresponds to the specified
    operation, or this flag is ignored.

*FI_MORE*
:   Indicates that the user has additional requests that will
    immediately be posted after the current call returns. Use of this
    flag may improve performance by enabling the provider to optimize
    its access to the fabric hardware.

*FI_INJECT*
:   Indicates that the control of constant data buffers should be
    returned to the user immediately after the call returns, even if the
    operation is handled asynchronously. This may require that the
    underlying provider implementation copy the data into a local buffer
    and transfer out of that buffer. Constant data buffers refers to any
    data buffer or iovec used by the atomic APIs that are marked as
    'const'. Non-constant or output buffers are unaffected by this flag
    and may be accessed by the provider at anytime until the operation
    has completed. This flag can only be used with messages smaller than
    inject_size.

*FI_FENCE*
:   Applies to transmits. Indicates that the requested operation, also
    known as the fenced operation, and any operation posted after the
    fenced operation will be deferred until all previous operations
    targeting the same peer endpoint have completed. Operations posted
    after the fencing will see and/or replace the results of any
    operations initiated prior to the fenced operation.

The ordering of operations starting at the posting of the fenced
operation (inclusive) to the posting of a subsequent fenced operation
(exclusive) is controlled by the endpoint's ordering semantics.

*FI_TAGGED*
:   Specifies that the target of the atomic operation is a tagged
    receive buffer instead of an RMA buffer. When a tagged buffer is the
    target memory region, the addr parameter is used as a 0-based byte
    offset into the tagged buffer, with the key parameter specifying the
    tag.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
:   See [`fi_msg`(3)](fi_msg.3.html) for a detailed description of
    handling FI_EAGAIN.

*-FI_EOPNOTSUPP*
:   The requested atomic operation is not supported on this endpoint.

*-FI_EMSGSIZE*
:   The number of atomic operations in a single request exceeds that
    supported by the underlying provider.

# NOTES

Atomic operations operate on an array of values of a specific data type.
Atomicity is only guaranteed for each data type operation, not across
the entire array. The following pseudo-code demonstrates this operation
for 64-bit unsigned atomic write. ATOMIC_WRITE_U64 is a platform
dependent macro that atomically writes 8 bytes to an aligned memory
location.

``` c
fi_atomic(ep, buf, count, NULL, dest_addr, addr, key,
      FI_UINT64, FI_ATOMIC_WRITE, context)
{
    for (i = 1; i < count; i ++)
        ATOMIC_WRITE_U64(((uint64_t *) addr)[i],
                 ((uint64_t *) buf)[i]);
}
```

The number of array elements to operate on is specified through a count
parameter. This must be between 1 and the maximum returned through the
relevant valid operation, inclusive. The requested operation and data
type must also be valid for the given provider.

The ordering of atomic operations carried as part of different request
messages is subject to the message and data ordering definitions
assigned to the transmitting and receiving endpoints. Both message and
data ordering are required if the results of two atomic operations to
the same memory buffers are to reflect the second operation acting on
the results of the first. See [`fi_endpoint`(3)](fi_endpoint.3.html) for
further details and message size restrictions.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cq`(3)](fi_cq.3.html),
[`fi_rma`(3)](fi_rma.3.html)

{% include JB/setup %}

# NAME

fi_av - Address vector operations

fi_av_open / fi_close
:   Open or close an address vector

fi_av_bind
:   Associate an address vector with an event queue. This function is
    deprecated and should not be used.

fi_av_insert / fi_av_insertsvc / fi_av_remove
:   Insert/remove an address into/from the address vector.

fi_av_lookup
:   Retrieve an address stored in the address vector.

fi_av_straddr
:   Convert an address into a printable string.

fi_av_insert_auth_key
:   Insert an authorization key into the address vector.

fi_av_lookup_auth_key
:   Retrieve an authorization key stored in the address vector.

fi_av_set_user_id
:   Set the user-defined fi_addr_t for an inserted fi_addr_t.

# SYNOPSIS

``` c
#include <rdma/fi_domain.h>

int fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
    struct fid_av **av, void *context);

int fi_close(struct fid *av);

int fi_av_bind(struct fid_av *av, struct fid *eq, uint64_t flags);

int fi_av_insert(struct fid_av *av, void *addr, size_t count,
    fi_addr_t *fi_addr, uint64_t flags, void *context);

int fi_av_insertsvc(struct fid_av *av, const char *node,
    const char *service, fi_addr_t *fi_addr, uint64_t flags,
    void *context);

int fi_av_insertsym(struct fid_av *av, const char *node,
    size_t nodecnt, const char *service, size_t svccnt,
    fi_addr_t *fi_addr, uint64_t flags, void *context);

int fi_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
    uint64_t flags);

int fi_av_lookup(struct fid_av *av, fi_addr_t fi_addr,
    void *addr, size_t *addrlen);

fi_addr_t fi_rx_addr(fi_addr_t fi_addr, int rx_index,
      int rx_ctx_bits);

fi_addr_t fi_group_addr(fi_addr_t fi_addr, uint32_t group_id);

const char * fi_av_straddr(struct fid_av *av, const void *addr,
      char *buf, size_t *len);

int fi_av_insert_auth_key(struct fid_av *av, const void *auth_key,
      size_t auth_key_size, fi_addr_t *fi_addr, uint64_t flags);

int fi_av_lookup_auth_key(struct fid_av *av, fi_addr_t addr,
      void *auth_key, size_t *auth_key_size);

int fi_av_set_user_id(struct fid_av *av, fi_addr_t fi_addr,
      fi_addr_t user_id, uint64_t flags);
```

# ARGUMENTS

*domain*
:   Resource domain

*av*
:   Address vector

*eq*
:   Event queue

*attr*
:   Address vector attributes

*context*
:   User specified context associated with the address vector or insert
    operation.

*addr*
:   Buffer containing one or more addresses to insert into address
    vector.

*addrlen*
:   On input, specifies size of addr buffer. On output, stores number of
    bytes written to addr buffer.

*fi_addr*
:   For insert, a reference to an array where returned fabric addresses
    will be written. For remove, one or more fabric addresses to remove.
    If FI_AV_USER_ID is requested, also used as input into insert calls
    to assign the user ID with the added address.

*count*
:   Number of addresses to insert/remove from an AV.

*flags*
:   Additional flags to apply to the operation.

*auth_key*
:   Buffer containing authorization key to be inserted into the address
    vector.

*auth_key_size*
:   On input, specifies size of auth_key buffer. On output, stores
    number of bytes written to auth_key buffer.

*user_id*
:   For address vectors configured with FI_AV_USER_ID, this defines the
    user ID to be associated with a specific fi_addr_t.

# DESCRIPTION

Address vectors are used to map higher-level addresses, which may be
more natural for an application to use, into fabric specific addresses.
For example, an endpoint may be associated with a struct sockaddr_in
address, indicating the endpoint is reachable using a TCP port number
over an IPv4 address. This may hold even if the endpoint communicates
using a proprietary network protocol. The purpose of the AV is to
associate a higher-level address with a simpler, more efficient value
that can be used by the libfabric API in a fabric agnostic way. The
mapped address is of type fi_addr_t and is returned through an AV
insertion call.

The process of mapping an address is fabric and provider specific, but
may involve lengthy address resolution and fabric management protocols.
AV operations are synchronous by default (the asynchrouous option has
been deprecated, see below). See the NOTES section for AV restrictions
on duplicate addresses.

**Deprecated**: AV operations may be set to operate asynchronously by
specifying the FI_EVENT flag to `fi_av_open`. When requesting
asynchronous operation, the application must first bind an event queue
to the AV before inserting addresses.

## fi_av_open

fi_av_open allocates or opens an address vector. The properties and
behavior of the address vector are defined by `struct fi_av_attr`.

``` c
struct fi_av_attr {
    enum fi_av_type  type;        /* type of AV */
    int              rx_ctx_bits; /* address bits to identify rx ctx */
    size_t           count;       /* # entries for AV */
    size_t           ep_per_node; /* # endpoints per fabric address */
    const char       *name;       /* system name of AV */
    void             *map_addr;   /* base mmap address */
    uint64_t         flags;       /* operation flags */
};
```

*type*
:   An AV type corresponds to a conceptual implementation of an address
    vector. The type specifies how an application views data stored in
    the AV, including how it may be accessed. Valid values are:

\- *FI_AV_MAP* (deprecated)
:   Addresses which are inserted into an AV are mapped to a native
    fabric address for use by the application. The use of FI_AV_MAP
    requires that an application store the returned fi_addr_t value that
    is associated with each inserted address. The advantage of using
    FI_AV_MAP is that the returned fi_addr_t value may contain encoded
    address data, which is immediately available when processing data
    transfer requests. This can eliminate or reduce the number of memory
    lookups needed when initiating a transfer. The disadvantage of
    FI_AV_MAP is the increase in memory usage needed to store the
    returned addresses. Addresses are stored in the AV using a provider
    specific mechanism, including, but not limited to a tree, hash
    table, or maintained on the heap. This option is deprecated, and
    providers are encouraged to align the behavior of FI_AV_MAP with
    FI_AV_TABLE.

\- *FI_AV_TABLE*
:   Addresses which are inserted into an AV of type FI_AV_TABLE are
    accessible using a simple index. Conceptually, the AV may be treated
    as an array of addresses, though the provider may implement the AV
    using a variety of mechanisms. When FI_AV_TABLE is used, the
    returned fi_addr_t is an index, with the index for an inserted
    address the same as its insertion order into the table. The index of
    the first address inserted into an FI_AV_TABLE will be 0, and
    successive insertions will be given sequential indices. Sequential
    indices will be assigned across insertion calls on the same AV.
    Because the fi_addr_t values returned from an insertion call are
    deterministic, applications may not need to provide the fi_addr_t
    output parameters to insertion calls. The exception is when the
    fi_addr_t parameters are also used as input for supplying
    authentication keys or user defined IDs.

\- *FI_AV_UNSPEC*
:   Provider will choose its preferred AV type. The AV type used will be
    returned through the type field in fi_av_attr.

*Receive Context Bits (rx_ctx_bits)*
:   The receive context bits field is only for use with scalable
    endpoints. It indicates the number of bits reserved in a returned
    fi_addr_t, which will be used to identify a specific target receive
    context. See fi_rx_addr() and fi_endpoint(3) for additional details
    on receive contexts. The requested number of bits should be selected
    such that 2 \^ rx_ctx_bits \>= rx_ctx_cnt for the endpoint.

*count*
:   Indicates the expected number of addresses that will be inserted
    into the AV. The provider uses this to optimize resource
    allocations.

*ep_per_node*
:   This field indicates the number of endpoints that will be associated
    with a specific fabric, or network, address. If the number of
    endpoints per node is unknown, this value should be set to 0. The
    provider uses this value to optimize resource allocations. For
    example, distributed, parallel applications may set this to the
    number of processes allocated per node, times the number of
    endpoints each process will open.

*name*
:   An optional system name associated with the address vector to create
    or open. Address vectors may be shared across multiple processes
    which access the same named domain on the same node. The name field
    allows the underlying provider to identify a shared AV.

If the name field is non-NULL and the AV is not opened for read-only
access, a named AV will be created, if it does not already exist.

*map_addr*
:   The map_addr determines the base fi_addr_t address that a provider
    should use when sharing an AV of type FI_AV_MAP between processes.
    Processes that provide the same value for map_addr to a shared AV
    may use the same fi_addr_t values returned from an fi_av_insert
    call.

The map_addr may be used by the provider to mmap memory allocated for a
shared AV between processes; however, the provider is not required to
use the map_addr in this fashion. The only requirement is that an
fi_addr_t returned as part of an fi_av_insert call on one process is
usable on another process which opens an AV of the same name at the same
map_addr value. The relationship between the map_addr and any returned
fi_addr_t is not defined.

If name is non-NULL and map_addr is 0, then the map_addr used by the
provider will be returned through the attribute structure. The map_addr
field is ignored if name is NULL.

*flags*
:   The following flags may be used when opening an AV.

\- *FI_EVENT* (deprecated)
:   When the flag FI_EVENT is specified, all insert operations on this
    AV will occur asynchronously. There will be one EQ error entry
    generated for each failed address insertion, followed by one
    non-error event indicating that the insertion operation has
    completed. There will always be one non-error completion event for
    each insert operation, even if all addresses fail. The context field
    in all completions will be the context specified to the insert call,
    and the data field in the final completion entry will report the
    number of addresses successfully inserted. If an error occurs during
    the asynchronous insertion, an error completion entry is returned
    (see [`fi_eq`(3)](fi_eq.3.html) for a discussion of the
    fi_eq_err_entry error completion struct). The context field of the
    error completion will be the context that was specified in the
    insert call; the data field will contain the index of the failed
    address. There will be one error completion returned for each
    address that fails to insert into the AV.

If an AV is opened with FI_EVENT, any insertions attempted before an EQ
is bound to the AV will fail with -FI_ENOEQ.

Error completions for failed insertions will contain the index of the
failed address in the index field of the error completion entry.

Note that the order of delivery of insert completions may not match the
order in which the calls to fi_av_insert were made. The only guarantee
is that all error completions for a given call to fi_av_insert will
precede the single associated non-error completion.

-   

    *FI_READ*
    :   Opens an AV for read-only access. An AV opened for read-only
        access must be named (name attribute specified), and the AV must
        exist.

-   

    *FI_SYMMETRIC*
    :   Indicates that each node will be associated with the same number
        of endpoints, the same transport addresses will be allocated on
        each node, and the transport addresses will be sequential. This
        feature targets distributed applications on large fabrics and
        allows for highly-optimized storage of remote endpoint
        addressing.

-   

    *FI_AV_USER_ID*
    :   Indicates that the user will be associating user-defined IDs
        with a address vector via fi_av_set_user_id. If the domain has
        been configured with FI_AV_AUTH_KEY or the user requires
        FI_AV_USER_ID support, using the FI_AV_USER_ID flag per
        fi_av_insert / fi_av_insertsvc / fi_av_remove is not supported.
        fi_av_set_user_id must be used.

## fi_close

The fi_close call is used to release all resources associated with an
address vector. Note that any events queued on an event queue
referencing the AV are left untouched. It is recommended that callers
retrieve all events associated with the AV before closing it.

When closing the address vector, there must be no opened endpoints
associated with the AV. If resources are still associated with the AV
when attempting to close, the call will return -FI_EBUSY.

## fi_av_bind (deprecated)

Associates an event queue with the AV. If an AV has been opened with
`FI_EVENT`, then an event queue must be bound to the AV before any
insertion calls are attempted. Any calls to insert addresses before an
event queue has been bound will fail with `-FI_ENOEQ`. Flags are
reserved for future use and must be 0.

## fi_av_insert

The fi_av_insert call inserts zero or more addresses into an AV. The
number of addresses is specified through the count parameter. The addr
parameter references an array of addresses to insert into the AV.
Addresses inserted into an address vector must be in the same format as
specified in the addr_format field of the fi_info struct provided when
opening the corresponding domain. When using the `FI_ADDR_STR` format,
the `addr` parameter should reference an array of strings (char \*\*).

**Deprecated**: For AV's of type FI_AV_MAP, once inserted addresses have
been mapped, the mapped values are written into the buffer referenced by
fi_addr. The fi_addr buffer must remain valid until the AV insertion has
completed and an event has been generated to an associated event queue.
The value of the returned fi_addr should be considered opaque by the
application for AVs of type FI_AV_MAP. The returned value may point to
an internal structure or a provider specific encoding of low-level
addressing data, for example. In the latter case, use of FI_AV_MAP may
be able to avoid memory references during data transfer operations.

For AV's of type FI_AV_TABLE, addresses are placed into the table in
order. An address is inserted at the lowest index that corresponds to an
unused table location, with indices starting at 0. That is, the first
address inserted may be referenced at index 0, the second at index 1,
and so forth. When addresses are inserted into an AV table, the assigned
fi_addr values will be simple indices corresponding to the entry into
the table where the address was inserted. Index values accumulate across
successive insert calls in the order the calls are made, not necessarily
in the order the insertions complete.

Because insertions occur at a pre-determined index, the fi_addr
parameter may be NULL. If fi_addr is non-NULL, it must reference an
array of fi_addr_t, and the buffer must remain valid until the insertion
operation completes. Note that if fi_addr is NULL and synchronous
operation is requested without using FI_SYNC_ERR flag, individual
insertion failures cannot be reported and the application must use other
calls, such as `fi_av_lookup` to learn which specific addresses failed
to insert.

If the address vector is configured with authorization keys, the fi_addr
parameter may be used as input to define the authorization keys
associated with the endpoint addresses being inserted. This is done by
setting the fi_addr to an authorization key fi_addr_t generated from
`fi_av_insert_auth_key` and setting the FI_AUTH_KEY flag. If the
FI_AUTH_KEY flag is not set, addresses being inserted will not be
associated with any authorization keys. Whether or not an address can be
disassociated with an authorization key is provider specific. If a
provider cannot support this disassociation, an error will be returned.
Upon successful insert with FI_AUTH_KEY flag, the returned fi_addr_t's
will map to endpoint address against the specified authorization keys.
These fi_addr_t's can be used as the target for local data transfer
operations.

If the endpoint supports `FI_DIRECTED_RECV` or
`FI_TAGGED_DIRECTED_RECV`, these fi_addr_t's can be used to restrict
receive buffers to a specific endpoint address and authorization key.

For address vectors configured with FI_AV_USER_ID, all subsequent target
events corresponding to the address being inserted will return
FI_ADDR_NOTAVAIL until the user defines a user ID for this fi_addr_t.
This is done by using fi_av_set_user_id.

*flags*
:   The following flag may be passed to AV insertion calls:
    fi_av_insert, fi_av_insertsvc, or fi_av_insertsym.

\- *FI_MORE*
:   In order to allow optimized address insertion, the application may
    specify the FI_MORE flag to the insert call to give a hint to the
    provider that more insertion requests will follow, allowing the
    provider to aggregate insertion requests if desired. An application
    may make any number of insertion calls with FI_MORE set, provided
    that they are followed by an insertion call without FI_MORE. This
    signifies to the provider that the insertion list is complete.
    Providers are free to ignore FI_MORE.

\- *FI_SYNC_ERR*
:   This flag applies to synchronous insertions only, and is used to
    retrieve error details of failed insertions. If set, the context
    parameter of insertion calls references an array of integers, with
    context set to address of the first element of the array. The
    resulting status of attempting to insert each address will be
    written to the corresponding array location. Successful insertions
    will be updated to 0. Failures will contain a fabric errno code.

\- *FI_AV_USER_ID*
:   For address vectors configured without FI_AV_USER_ID specified, this
    flag associates a user-assigned identifier with each AV entry that
    is returned with any completion entry in place of the AV's address.
    If a provider does not support FI_AV_USER_ID with insert, requesting
    this flag during insert will result runtime failure.

Using the FI_AV_USER_ID flag per insert is invalid if the AV was opened
with the FI_AV_USER_ID or if the corresponding domain was configured
with FI_AV_AUTH_KEY.

With libfabric 1.20, users are encouraged to specify the FI_AV_USER_ID
when opening an AV and use fi_av_set_user_id.

-   

    *FI_AUTH_KEY*
    :   Denotes that the address being inserted should be associated
        with the passed in authorization key fi_addr_t.

    See the user ID section below.

-   

    *FI_FIREWALL_ADDR*
    :   This flag indicates that the address is behind a firewall and
        outgoing connections are not allowed. If there is not an
        existing connection and the provider is unable to circumvent the
        firewall, an FI_EFIREWALLADDR error should be expected. If
        multiple addresses are being inserted simultaneously, the flag
        applies to all of them. Additionally, it is possible that a
        connection is available at insertion time, but is later torn
        down. Future reconnects triggered by operations on the ep
        (fi_send, for example) may also fail with the same error.

## fi_av_insertsvc

The fi_av_insertsvc call behaves similar to fi_av_insert, but allows the
application to specify the node and service names, similar to the
fi_getinfo inputs, rather than an encoded address. The node and service
parameters are defined the same as fi_getinfo(3). Node should be a
string that corresponds to a hostname or network address. The service
string corresponds to a textual representation of a transport address.
Applications may also pass in an `FI_ADDR_STR` formatted address as the
node parameter. In such cases, the service parameter must be NULL. See
fi_getinfo.3 for details on using `FI_ADDR_STR`. Supported flags are the
same as for fi_av_insert.

## fi_av_insertsym

fi_av_insertsym performs a symmetric insert that inserts a sequential
range of nodes and/or service addresses into an AV. The svccnt parameter
indicates the number of transport (endpoint) addresses to insert into
the AV for each node address, with the service parameter specifying the
starting transport address. Inserted transport addresses will be of the
range {service, service + svccnt - 1}, inclusive. All service addresses
for a node will be inserted before the next node is inserted.

The nodecnt parameter indicates the number of node (network) addresses
to insert into the AV, with the node parameter specifying the starting
node address. Inserted node addresses will be of the range {node, node +
nodecnt - 1}, inclusive. If node is a non-numeric string, such as a
hostname, it must contain a numeric suffix if nodecnt \> 1.

As an example, if node = "10.1.1.1", nodecnt = 2, service = "5000", and
svccnt = 2, the following addresses will be inserted into the AV in the
order shown: 10.1.1.1:5000, 10.1.1.1:5001, 10.1.1.2:5000, 10.1.1.2:5001.
If node were replaced by the hostname "host10", the addresses would be:
host10:5000, host10:5001, host11:5000, host11:5001.

The total number of inserted addresses will be nodecnt x svccnt.

Supported flags are the same as for fi_av_insert.

## fi_av_remove

fi_av_remove removes a set of addresses from an address vector. The
corresponding fi_addr_t values are invalidated and may not be used in
data transfer calls. The behavior of operations in progress that
reference the removed addresses is undefined.

Note that removing an address may not disable receiving data from the
peer endpoint. fi_av_close will automatically cleanup any associated
resource.

If the address being removed came from `fi_av_insert_auth_key`, the
address will only be removed if all endpoints, which have been enabled
against the corresponding authorization key, have been closed. If all
endpoints are not closed, -FI_EBUSY will be returned. In addition, the
FI_AUTH_KEY flag must be set when removing an authorization key
fi_addr_t.

*flags*
:   The following flags may be used when removing an AV entry.

\- *FI_AUTH_KEY*
:   Denotes that the fi_addr_t being removed is an authorization key
    fi_addr_t.

## fi_av_lookup

This call returns the address stored in the address vector that
corresponds to the given fi_addr. The returned address is the same
format as those stored by the AV. On input, the addrlen parameter should
indicate the size of the addr buffer. If the actual address is larger
than what can fit into the buffer, it will be truncated. On output,
addrlen is set to the size of the buffer needed to store the address,
which may be larger than the input value.

## fi_rx_addr

This function is used to convert an endpoint address, returned by
fi_av_insert, into an address that specifies a target receive context.
The value for rx_ctx_bits must match that specified in the AV attributes
for the given address.

Connected endpoints that support multiple receive contexts, but are not
associated with address vectors should specify FI_ADDR_NOTAVAIL for the
fi_addr parameter.

## fi_av_straddr

The fi_av_straddr function converts the provided address into a
printable string. The specified address must be of the same format as
those stored by the AV, though the address itself is not required to
have been inserted. On input, the len parameter should specify the size
of the buffer referenced by buf. On output, addrlen is set to the size
of the buffer needed to store the address. This size may be larger than
the input len. If the provided buffer is too small, the results will be
truncated. fi_av_straddr returns a pointer to buf.

## fi_av_insert_auth_key

This function associates authorization keys with an address vector. This
requires the domain to be opened with `FI_AV_AUTH_KEY`. `FI_AV_AUTH_KEY`
enables endpoints and memory regions to be associated with authorization
keys from the address vector. This behavior enables a single endpoint or
memory region to be associated with multiple authorization keys.

When endpoints or memory regions are enabled, they are configured with
address vector authorization keys at that point in time. Later
authorization key insertions will not propagate to already enabled
endpoints and memory regions.

The `auth_key` and `auth_key_size` parameters are used to input the
authorization key into the address vector. The structure of the
authorization key is provider specific. If the `auth_key_size` does not
align with provider specific structure, -FI_EINVAL will be returned.

The output of `fi_av_insert_auth_key` is an authorization key fi_addr_t
handle representing all endpoint addresses against this specific
authorization key. For all operations, including address vector, memory
registration, and data transfers, which may accept an authorization key
fi_addr_t as input, the FI_AUTH_KEY flag must be specified. Otherwise,
the fi_addr_t will be treated as an fi_addr_t returned from the
`fi_av_insert` and related functions.

For endpoints enabled with FI_DIRECTED_RECV, authorization key
fi_addr_t's can be used to restrict incoming messages to only endpoint
addresses within the authorization key. This will require passing in the
FI_AUTH_KEY flag to `fi_recvmsg` and `fi_trecvmsg`.

For domains enabled with FI_DIRECTED_RECV, authorization key fi_addr_t's
can be used to restrict memory region access to only endpoint addresses
within the authorization key. This will require passing in the
FI_AUTH_KEY flag to `fi_mr_regattr`.

These authorization key fi_addr_t's can later be used an input for
endpoint address insertion functions to generate an fi_addr_t for a
specific endpoint address and authorization key. This will require
passing in the FI_AUTH_KEY flag to `fi_av_insert` and related functions.

For address vectors configured with FI_AV_USER_ID and endpoints with
FI_SOURCE_ERR, all subsequent FI_EADDRNOTAVAIL error events will return
FI_ADDR_NOTAVAIL until the user defines a user ID for this authorization
key fi_addr_t. This is done by using fi_av_set_user_id.

For address vectors configured without FI_AV_USER_ID and endpoints with
FI_SOURCE_ERR, all subsequent FI_EADDRNOTAVAIL error events will return
the authorization key fi_addr_t handle.

Flags are reserved for future use and must be 0.

## fi_av_lookup_auth_key

This functions returns the authorization key associated with a
fi_addr_t. Acceptable fi_addr_t's input are the output of
`fi_av_insert_auth_key` and AV address insertion functions. The returned
authorization key is in a provider specific format. On input, the
auth_key_size parameter should indicate the size of the auth_key buffer.
If the actual authorization key is larger than what can fit into the
buffer, it will be truncated. On output, auth_key_size is set to the
size of the buffer needed to store the authorization key, which may be
larger than the input value.

## fi_av_set_user_id

If the address vector has been opened with FI_AV_USER_ID, this function
defines the user ID for a specific fi_addr_t. By default, all
fi_addr_t's will be assigned the user ID FI_ADDR_NOTAVAIL.

*flags*
:   The following flag may be passed to AV set user id.

\- *FI_AUTH_KEY*
:   Denotes that the fi_addr fi_addr_t, for which the user ID is being
    set for, is an authorization key fi_addr_t.

# NOTES

An AV should only store a single instance of an address. Attempting to
insert a duplicate copy of the same address into an AV may result in
undefined behavior, depending on the provider implementation. Providers
are not required to check for duplicates, as doing so could incur
significant overhead to the insertion process. For portability,
applications may need to track which peer addresses have been inserted
into a given AV in order to avoid duplicate entries. However, providers
are required to support the removal, followed by the re-insertion of an
address. Only duplicate insertions are restricted.

# USER IDENTIFIERS FOR ADDRESSES

As described above, endpoint addresses authorization keys that are
inserted into an AV are mapped to an fi_addr_t value. The endpoint
address fi_addr_t is used in data transfer APIs to specify the
destination of an outbound transfer, in receive APIs to indicate the
source for an inbound transfer, and also in completion events to report
the source address of inbound transfers. The authorization key fi_addr_t
are used in receive and MR APIs to resource incoming operations to a
specific authorization key, and also in completion error events if the
endpoint is configured with FI_SOURCE_ERR. The FI_AV_USER_ID capability
bit and flag provide a mechanism by which the fi_addr_t value reported
by a completion success or error event is replaced with a user-specified
value instead. This is useful for applications that need to map the
source address to their own data structure.

Support for FI_AV_USER_ID is provider specific, as it may not be
feasible for a provider to implement this support without significant
overhead. For example, some providers may need to add a reverse lookup
mechanism. This feature may be unavailable if shared AVs are requested,
or negatively impact the per process memory footprint if implemented.
For providers that do not support FI_AV_USER_ID, users may be able to
trade off lookup processing with protocol overhead, by carrying source
identification within a message header.

For address vectors opened without FI_AV_USER_ID, user-specified
fi_addr_t values are provided as part of address insertion
(e.g. fi_av_insert) through the fi_addr parameter. The fi_addr parameter
acts as input/output in this case. When the FI_AV_USER_ID flag is passed
to any of the insert calls, the caller must specify an fi_addr_t
identifier value to associate with each address. The provider will
record that identifier and use it where required as part of any
completion event. Note that the output from the AV insertion call is
unchanged. The provider will return an fi_addr_t value that maps to each
address, and that value must be used for all data transfer operations.

For address vectors opened with FI_AV_USER_ID, fi_av_set_user_id is used
to defined the user-specified fi_addr_t.

# PEER GROUPS

Peer groups provide a direct mapping to HPC and AI communicator
constructs.

The addresses in an AV represent the full set of peers that a local
process may communicate with. A peer group conceptually represents a
subset of those peers. A peer group may be used to identify peers
working on a common task, which need their communication logically
separated from other traffic. Peer groups are not a security mechanism,
but instead help separate data. A given peer may belong to 0 or more
peer groups, with no limit placed on how many peers can belong to a
single peer group.

Peer groups are identified using an integer value, known as a group id.
Group id's are selected by the user and conveyed as part of an fi_addr_t
value. The management of a group id and it's relationship to addresses
inserted into an AV is directly controlled by the user. When enabled,
sent messages are marked as belonging to a specific peer group, and
posted receive buffers must have a matching group id to receive the
data.

Users are responsible for selecting a valid peer group id, subject to
the limitation negotiated using the domain attribute max_group_id. The
group id of an fi_addr_t may be set using the fi_group_addr() function.

## fi_group_addr

This function is used to set the group ID portion of an fi_addr_t.

# RETURN VALUES

Insertion calls, excluding `fi_av_insert_auth_key`, for an AV opened for
synchronous operation will return the number of addresses that were
successfully inserted. In the case of failure, the return value will be
less than the number of addresses that was specified.

**Deprecated**: Insertion calls, excluding `fi_av_insert_auth_key`, for
an AV opened for asynchronous operation (with FI_EVENT flag specified)
will return FI_SUCCESS if the operation was successfully initiated. In
the case of failure, a negative fabric errno will be returned. Providers
are allowed to abort insertion operations in the case of an error.
Addresses that are not inserted because they were aborted will fail with
an error code of FI_ECANCELED.

In both the synchronous and asynchronous modes of operation, the fi_addr
buffer associated with a failed or aborted insertion will be set to
FI_ADDR_NOTAVAIL.

All other calls return FI_SUCCESS on success, or a negative value
corresponding to fabric errno on error. Fabric errno values are defined
in `rdma/fi_errno.h`.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_eq`(3)](fi_eq.3.html)

{% include JB/setup %}

# NAME

fi_av_set - Address vector set operations

fi_av_set / fi_close
:   Open or close an address vector set

fi_av_set_union
:   Perform a set union operation on two AV sets

fi_av_set_intersect
:   Perform a set intersect operation on two AV sets

fi_av_set_diff
:   Perform a set difference operation on two AV sets

fi_av_set_insert
:   Add an address to an AV set

fi_av_set_remove
:   Remove an address from an AV set

fi_av_set_addr
:   Obtain a collective address for current addresses in an AV set

# SYNOPSIS

``` c
#include <rdma/fi_collective.h>

int fi_av_set(struct fid_av *av, struct fi_av_set_attr *attr,
      struct fid_av_set **set, void * context);

int fi_av_set_union(struct fid_av_set *dst, const struct fid_av_set *src);

int fi_av_set_intersect(struct fid_av_set *dst, const struct fid_av_set *src);

int fi_av_set_diff(struct fid_av_set *dst, const struct fid_av_set *src);

int fi_av_set_insert(struct fid_av_set *set, fi_addr_t addr);

int fi_av_set_remove(struct fid_av_set *set, fi_addr_t addr);

int fi_av_set_addr(struct fid_av_set *set, fi_addr_t *coll_addr);

int fi_close(struct fid *av_set);
```

# ARGUMENTS

*av*
:   Address vector

*set*
:   Address vector set

*dst*
:   Address vector set updated by set operation

*src*
:   Address vector set providing input to a set operation

*attr*
:   Address vector set attributes

*context*
:   User specified context associated with the address vector set

*flags*
:   Additional flags to apply to the operation.

*addr*
:   Destination address to insert to remove from AV set.

*coll_addr*
:   Address identifying collective group.

# DESCRIPTION

An address vector set (AV set) represents an ordered subset of addresses
of an address vector. AV sets are used to identify the participants in a
collective operation. Endpoints use the fi_join_collective() operation
to associate itself with an AV set. The join collective operation
provides an fi_addr that is used when communicating with a collective
group.

The creation and manipulation of an AV set is a local operation. No
fabric traffic is exchanged between peers. As a result, each peer is
responsible for creating matching AV sets as part of their collective
membership definition. See [`fi_collective`(3)](fi_collective.3.html)
for a discussion of membership models.

## fi_av_set

The fi_av_set call creates a new AV set. The initial properties of the
AV set are specified through the struct fi_av_set_attr parameter. This
structure is defined below, and allows including a subset of addresses
in the AV set as part of AV set creation. Addresses may be added or
removed from an AV set using the AV set interfaces defined below.

## fi_av_set_attr

{% highlight c %} struct fi_av_set_attr { size_t count; fi_addr_t
start_addr; fi_addr_t end_addr; uint64_t stride; size_t comm_key_size;
uint8_t \*comm_key; uint64_t flags; }; {% endhighlight %}

*count*
:   Indicates the expected the number of members that will be a part of
    the AV set. The provider uses this to optimize resource allocations.
    If count is 0, the provider will select a size based on available
    system configuration data or underlying limitations.

*start_addr / end_addr*
:   The starting and ending addresses, inclusive, to include as part of
    the AV set. The use of start and end address require that the
    associated AV have been created as type FI_AV_TABLE. Valid addresses
    in the AV which fall within the specified range and which meet other
    requirements (such as stride) will be added as initial members to
    the AV set. The start_addr and end_addr must be set to
    FI_ADDR_NOTAVAIL if creating an empty AV set, a communication key is
    being provided, or the AV is of type FI_AV_MAP.

The number of addresses between start_addr and end_addr must be less
than or equal to the specified count value.

*stride*
:   The number of entries between successive addresses included in the
    AV set. The AV set will include all addresses from start_addr +
    stride x i, for increasing, non-negative, integer values of i, up to
    end_addr. A stride of 1 indicates that all addresses between
    start_addr and end_addr should be added to the AV set. Stride should
    be set to 0 unless the start_addr and end_addr fields are valid.

*comm_key_size*
:   The length of the communication key in bytes. This field should be 0
    if a communication key is not available.

*comm_key*
:   If supported by the fabric, this represents a key associated with
    the AV set. The communication key is used by applications that
    directly manage collective membership through a fabric management
    agent or resource manager. The key is used to convey that results of
    the membership setup to the underlying provider. The use and format
    of a communication key is fabric provider specific.

*flags*
:   Flags may be used to configure the AV set, including restricting
    which collective operations the AV set needs to support. See the
    flags section for a list of flags that may be specified when
    creating the AV set.

## fi_av_set_union

The AV set union call adds all addresses in the source AV set that are
not in the destination AV set to the destination AV set. Where ordering
matters, the newly inserted addresses are placed at the end of the AV
set.

## fi_av_set_intersect

The AV set intersect call remove all addresses from the destination AV
set that are not also members of the source AV set. The order of the
addresses in the destination AV set is unchanged.

## fi_av_set_diff

The AV set difference call removes all address from the destination AV
set that are also members of the source AV set. The order of the
addresses in the destination AV set is unchanged.

## fi_av_set_insert

The AV set insert call appends the specified address to the end of the
AV set.

## fi_av_set_remove

The AV set remove call removes the specified address from the given AV
set. The order of the remaining addresses in the AV set is unchanged.

## fi_av_set_addr

Returns an address that may be used to communicate with all current
members of an AV set. This is a local operation only that does not
involve network communication. The returned address may be used as input
into fi_join_collective. Note that attempting to use the address
returned from fi_av_set_addr (e.g. passing it to fi_join_collective)
while simultaneously modifying the addresses stored in an AV set results
in undefined behavior.

## fi_close

Closes an AV set and releases all resources associated with it. Any
operations active at the time an AV set is closed will be aborted, with
the result of the collective undefined.

# FLAGS

The following flags may be specified as part of AV set creation.

*FI_UNIVERSE*
:   When set, then the AV set will be created containing all addresses
    stored in the corresponding AV.

*FI_BARRIER_SET*
:   If set, the AV set will be configured to support barrier operations.

*FI_BROADCAST_SET*
:   If set, the AV set will be configured to support broadcast
    operations.

*FI_ALLTOALL_SET*
:   If set, the AV set will be configured to support all to all
    operations.

*FI_ALLREDUCE_SET*
:   If set, the AV set will be configured to support all reduce
    operations.

*FI_ALLGATHER_SET*
:   If set, the AV set will be configured to support all gather
    operations.

*FI_REDUCE_SCATTER_SET*
:   If set, the AV set will be configured to support reduce scatter
    operations.

*FI_REDUCE_SET*
:   If set, the AV set will be configured to support reduce operations.

*FI_SCATTER_SET*
:   If set, the AV set will be configured to support scatter operations.

*FI_GATHER_SET*
:   If set, the AV set will be configured to support gather operations.

# NOTES

Developers who are familiar with MPI will find that AV sets are similar
to MPI groups, and may act as a direct mapping in some, but not all,
situations.

By default an AV set will be created to support all collective
operations supported by the underlying provider (see
fi_query_collective). Users may reduce resource requirements by
specifying only those collection operations needed by the AV set through
the use of creation flags: FI_BARRIER_SET, FI_BROADCAST_SET, etc. If no
such flags are specified, the AV set will be configured to support any
that are supported. It is an error for a user to request an unsupported
collective.

# RETURN VALUES

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

# SEE ALSO

[`fi_av`(3)](fi_av.3.html), [`fi_collective`(3)](fi_collective.3.html)

{% include JB/setup %}

# NAME

fi_cm - Connection management operations

fi_connect / fi_listen / fi_accept / fi_reject / fi_shutdown
:   Manage endpoint connection state.

fi_setname / fi_getname / fi_getpeer
:   Set local, or return local or peer endpoint address.

fi_join / fi_close / fi_mc_addr
:   Join, leave, or retrieve a multicast address.

# SYNOPSIS

``` c
#include <rdma/fi_cm.h>

int fi_connect(struct fid_ep *ep, const void *addr,
    const void *param, size_t paramlen);

int fi_listen(struct fid_pep *pep);

int fi_accept(struct fid_ep *ep, const void *param, size_t paramlen);

int fi_reject(struct fid_pep *pep, fid_t handle,
    const void *param, size_t paramlen);

int fi_shutdown(struct fid_ep *ep, uint64_t flags);

int fi_setname(fid_t fid, void *addr, size_t addrlen);

int fi_getname(fid_t fid, void *addr, size_t *addrlen);

int fi_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen);

int fi_join(struct fid_ep *ep, const void *addr, uint64_t flags,
    struct fid_mc **mc, void *context);

int fi_close(struct fid *mc);

fi_addr_t fi_mc_addr(struct fid_mc *mc);
```

# ARGUMENTS

*ep / pep*
:   Fabric endpoint on which to change connection state.

*fid* Active or passive endpoint to get/set address.

*addr*
:   Buffer to address. On a set call, the endpoint will be assigned the
    specified address. On a get, the local address will be copied into
    the buffer, up to the space provided. For connect, this parameter
    indicates the peer address to connect to. The address must be in the
    same format as that specified using fi_info: addr_format when the
    endpoint was created.

*addrlen*
:   On input, specifies size of addr buffer. On output, stores number of
    bytes written to addr buffer.

*param*
:   User-specified data exchanged as part of the connection exchange.

*paramlen*
:   Size of param buffer.

*info*
:   Fabric information associated with a connection request.

*mc*
:   Multicast group associated with an endpoint.

*flags*
:   Additional flags for controlling connection operation.

*context*
:   User context associated with the request.

# DESCRIPTION

Connection management functions are used to connect an
connection-oriented (FI_EP_MSG) endpoint to a listening peer.

## fi_listen

The fi_listen call indicates that the specified endpoint should be
transitioned into a passive connection state, allowing it to accept
incoming connection requests. Connection requests against a listening
endpoint are reported asynchronously to the user through a bound CM
event queue using the FI_CONNREQ event type. The number of outstanding
connection requests that can be queued at an endpoint is limited by the
listening endpoint's backlog parameter. The backlog is initialized based
on administrative configuration values, but may be adjusted through the
fi_control call.

## fi_connect

The fi_connect call initiates a connection request on a
connection-oriented endpoint to the destination address. fi_connect may
only be called on an endpoint once in its lifetime.

## fi_accept / fi_reject

The fi_accept and fi_reject calls are used on the passive (listening)
side of a connection to accept or reject a connection request,
respectively. To accept a connection, the listening application first
waits for a connection request event (FI_CONNREQ). After receiving such
an event, the application allocates a new endpoint to accept the
connection. This endpoint must be allocated using an fi_info structure
referencing the handle from this FI_CONNREQ event. fi_accept is then
invoked with the newly allocated endpoint. If the listening application
wishes to reject a connection request, it calls fi_reject with the
listening endpoint and a reference to the connection request.

A successfully accepted connection request will result in the active
(connecting) endpoint seeing an FI_CONNECTED event on its associated
event queue. A rejected or failed connection request will generate an
error event. The error entry will provide additional details describing
the reason for the failed attempt.

An FI_CONNECTED event will also be generated on the passive side for the
accepting endpoint once the connection has been properly established.
The fid of the FI_CONNECTED event will be that of the endpoint passed to
fi_accept as opposed to the listening passive endpoint. Outbound data
transfers cannot be initiated on a connection-oriented endpoint until an
FI_CONNECTED event has been generated. However, receive buffers may be
associated with an endpoint anytime.

## fi_shutdown

The fi_shutdown call is used to gracefully disconnect an endpoint from
its peer. The flags parameter is reserved and must be 0.

Outstanding operations posted to the endpoint when fi_shutdown is called
will be canceled or discarded. Notification of canceled operations will
be reported by the provider to the corresponding completion queue(s).
Discarded operations will silently be dropped, with no completions
generated. The choice of canceling, versus discarding operations, is
provider dependent. However, all canceled completions will be written
before fi_shutdown returns.

When called, fi_shutdown does not affect completions already written to
a completion queue. Any queued completions associated with asynchronous
operations posted to the endpoint may still be retrieved from the
corresponding completion queue(s) after an endpoint has been shutdown.

An FI_SHUTDOWN event will be generated for an endpoint when the remote
peer issues a disconnect using fi_shutdown or abruptly closes the
endpoint. Note that in the abrupt close case, an FI_SHUTDOWN event will
only be generated if the peer system is reachable and a service or
kernel agent on the peer system is able to notify the local endpoint
that the connection has been aborted.

## fi_close

Fi_close is used to disassociate an endpoint from a multicast group and
close all resources associated with the group. Fi_close must be called
on all multicast groups that an endpoint joins.

## fi_setname

The fi_setname call may be used to modify or assign the address of the
local endpoint. It is conceptually similar to the socket bind operation.
An endpoint may be assigned an address on its creation, through the
fi_info structure. The fi_setname call allows an endpoint to be created
without being associated with a specific service (e.g., port number)
and/or node (e.g., network) address, with the addressing assigned
dynamically. The format of the specified addressing data must match that
specified through the fi_info structure when the endpoint was created.

If no service address is specified and a service address has not yet
been assigned to the endpoint, then the provider will allocate a service
address and assign it to the endpoint. If a node or service address is
specified, then, upon successful completion of fi_setname, the endpoint
will be assigned the given addressing. If an address cannot be assigned,
or the endpoint address cannot be modified, an appropriate fabric error
number is returned.

## fi_getname / fi_getpeer

The fi_getname and fi_getpeer calls may be used to retrieve the local or
peer endpoint address, respectively. On input, the addrlen parameter
should indicate the size of the addr buffer. If the actual address is
larger than what can fit into the buffer, it will be truncated and
-FI_ETOOSMALL will be returned. On output, addrlen is set to the size of
the buffer needed to store the address, which may be larger than the
input value.

fi_getname is not guaranteed to return a valid source address until
after the specified endpoint has been enabled or has had an address
assigned. An endpoint may be enabled explicitly through fi_enable, or
implicitly, such as through fi_connect or fi_listen. An address may be
assigned using fi_setname. fi_getpeer is not guaranteed to return a
valid peer address until an endpoint has been completely connected -- an
FI_CONNECTED event has been generated.

## fi_join

This call attaches an endpoint to a multicast group. By default, the
endpoint will join the group based on the data transfer capabilities of
the endpoint. For example, if the endpoint has been configured to both
send and receive data, then the endpoint will be able to initiate and
receive transfers to and from the multicast group. The fi_join flags may
be used to restrict access to the multicast group, subject to endpoint
capability limitations.

Multicast join operations complete asynchronously. An endpoint must be
bound to an event queue prior to calling fi_join. The result of the join
operation will be reported to the EQ as an FI_JOIN_COMPLETE event.
Applications cannot issue multicast transfers until receiving
notification that the join operation has completed. Note that an
endpoint may begin receiving messages from the multicast group as soon
as the join completes, which can occur prior to the FI_JOIN_COMPLETE
event being generated.

Applications must call fi_close on the multicast group to disconnect the
endpoint from the group. After a join operation has completed, the
fi_mc_addr call may be used to retrieve the address associated with the
multicast group.

## fi_mc_addr

Returns the fi_addr_t address associated with a multicast group. This
address must be used when transmitting data to a multicast group and
paired with the FI_MULTICAST operation flag.

# FLAGS

Except in functions noted below, flags are reserved and must be 0.

*FI_SEND*
:   Applies to fi_join. This flag indicates that the endpoint should
    join the multicast group as a send only member. The endpoint must be
    configured for transmit operations to use this flag, or an error
    will occur.

*FI_RECV*
:   Applies to fi_join. This flag indicates that the endpoint should
    join the multicast group with receive permissions only. The endpoint
    must be configured for receive operations to use this flag, or an
    error will occur.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

# ERRORS

# NOTES

For connection-oriented endpoints, the buffer referenced by param will
be sent as part of the connection request or response, subject to the
constraints of the underlying connection protocol. Applications may use
fi_getopt with the FI_OPT_CM_DATA_SIZE endpoint option to determine the
size of application data that may be exchanged as part of a connection
request or response. The fi_connect, fi_accept, and fi_reject calls will
silently truncate any application data which cannot fit into underlying
protocol messages. User data exchanged as part of the connection process
is available as part of the fi_eq_cm_entry structure, for FI_CONNREQ and
FI_CONNECTED events, or as additional err_data to fi_eq_err_entry, in
the case of a rejected connection.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_eq`(3)](fi_eq.3.html)

{% include JB/setup %}

# NAME

fi_cntr - Completion and event counter operations

fi_cntr_open / fi_close
:   Allocate/free a counter

fi_cntr_read
:   Read the current value of a counter

fi_cntr_readerr
:   Reads the number of operations which have completed in error.

fi_cntr_add
:   Increment a counter by a specified value

fi_cntr_set
:   Set a counter to a specified value

fi_cntr_wait
:   Wait for a counter to be greater or equal to a threshold value

# SYNOPSIS

``` c
#include <rdma/fi_domain.h>

int fi_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
    struct fid_cntr **cntr, void *context);

int fi_close(struct fid *cntr);

uint64_t fi_cntr_read(struct fid_cntr *cntr);

uint64_t fi_cntr_readerr(struct fid_cntr *cntr);

int fi_cntr_add(struct fid_cntr *cntr, uint64_t value);

int fi_cntr_adderr(struct fid_cntr *cntr, uint64_t value);

int fi_cntr_set(struct fid_cntr *cntr, uint64_t value);

int fi_cntr_seterr(struct fid_cntr *cntr, uint64_t value);

int fi_cntr_wait(struct fid_cntr *cntr, uint64_t threshold,
    int timeout);
```

# ARGUMENTS

*domain*
:   Fabric domain

*cntr*
:   Fabric counter

*attr*
:   Counter attributes

*context*
:   User specified context associated with the counter

*value*
:   Value to increment or set counter

*threshold*
:   Value to compare counter against

*timeout*
:   Time in milliseconds to wait. A negative value indicates infinite
    timeout.

# DESCRIPTION

Counters record the number of requested operations that have completed.
Counters can provide a light-weight completion mechanism by allowing the
suppression of CQ completion entries. They are useful for applications
that only need to know the number of requests that have completed, and
not details about each request. For example, counters may be useful for
implementing credit based flow control or tracking the number of remote
processes that have responded to a request.

Counters typically only count successful completions. However, if an
operation completes in error, it may increment an associated error
value. That is, a counter actually stores two distinct values, with
error completions updating an error specific value.

Counters are updated following the completion event semantics defined in
[`fi_cq`(3)](fi_cq.3.html). The timing of the update is based on the
type of transfer and any specified operation flags.

## fi_cntr_open

fi_cntr_open allocates a new fabric counter. The properties and behavior
of the counter are defined by `struct fi_cntr_attr`.

``` c
struct fi_cntr_attr {
    enum fi_cntr_events  events;    /* type of events to count */
    enum fi_wait_obj     wait_obj;  /* requested wait object */
    struct fid_wait     *wait_set;  /* optional wait set, deprecated */
    uint64_t             flags;     /* operation flags */
};
```

*events*
:   A counter captures different types of events. The specific type
    which is to counted are one of the following:

\- *FI_CNTR_EVENTS_COMP*
:   The counter increments for every successful completion that occurs
    on an associated bound endpoint. The type of completions -- sends
    and/or receives -- which are counted may be restricted using control
    flags when binding the counter and the endpoint. Counters increment
    on all successful completions, separately from whether the operation
    generates an entry in an event queue.

\- *FI_CNTR_EVENTS_BYTES*
:   The counter is incremented by the number of user bytes, excluding
    any CQ data, transferred in a transport message upon reaching the
    specified completion semantic. For initiator side counters, the
    count reflects the size of the requested transfer and is updated
    after the message reaches the desired completion level
    (FI_INJECT_COMPLETE, FI_TRANSMIT_COMPLETE, etc.). For send and write
    operations, the count reflects the number of bytes transferred to
    the peer. For read operations, the count reflects the number of
    bytes returned in a read response. Operations which may both write
    and read data, such as atomics, behave as read operations at the
    initiator, but writes at the target. For target side counters, the
    count reflects the size of received user data and is incremented
    subject to target side completion semantics. In most cases, this
    indicates FI_DELIVERY_COMPLETE, but may differ when accessing device
    memory (HMEM). On error, the tranfer size is not applied to the
    error field, that field is increment by 1. The FI_COLLECTIVE
    transfer type is not supported.

*wait_obj*
:   Counters may be associated with a specific wait object. Wait objects
    allow applications to block until the wait object is signaled,
    indicating that a counter has reached a specific threshold. Users
    may use fi_control to retrieve the underlying wait object associated
    with a counter, in order to use it in other system calls. The
    following values may be used to specify the type of wait object
    associated with a counter: FI_WAIT_NONE, FI_WAIT_UNSPEC,
    FI_WAIT_SET, FI_WAIT_FD, FI_WAIT_MUTEX_COND (deprecated), and
    FI_WAIT_YIELD. The default is FI_WAIT_NONE.

\- *FI_WAIT_NONE*
:   Used to indicate that the user will not block (wait) for events on
    the counter.

\- *FI_WAIT_UNSPEC*
:   Specifies that the user will only wait on the counter using fabric
    interface calls, such as fi_cntr_wait. In this case, the underlying
    provider may select the most appropriate or highest performing wait
    object available, including custom wait mechanisms. Applications
    that select FI_WAIT_UNSPEC are not guaranteed to retrieve the
    underlying wait object.

\- *FI_WAIT_SET* (deprecated)
:   Indicates that the event counter should use a wait set object to
    wait for events. If specified, the wait_set field must reference an
    existing wait set object.

\- *FI_WAIT_FD*
:   Indicates that the counter should use a file descriptor as its wait
    mechanism. A file descriptor wait object must be usable in select,
    poll, and epoll routines. However, a provider may signal an FD wait
    object by marking it as readable, writable, or with an error.

\- *FI_WAIT_MUTEX_COND* (deprecated)
:   Specifies that the counter should use a pthread mutex and cond
    variable as a wait object.

\- *FI_WAIT_YIELD*
:   Indicates that the counter will wait without a wait object but
    instead yield on every wait. Allows usage of fi_cntr_wait through a
    spin.

*wait_set* (deprecated)
:   If wait_obj is FI_WAIT_SET, this field references a wait object to
    which the event counter should attach. When an event is added to the
    event counter, the corresponding wait set will be signaled if all
    necessary conditions are met. The use of a wait_set enables an
    optimized method of waiting for events across multiple event
    counters. This field is ignored if wait_obj is not FI_WAIT_SET.

*flags*
:   Flags are reserved for future use, and must be set to 0.

## fi_close

The fi_close call releases all resources associated with a counter. When
closing the counter, there must be no opened endpoints, transmit
contexts, receive contexts or memory regions associated with the
counter. If resources are still associated with the counter when
attempting to close, the call will return -FI_EBUSY.

## fi_cntr_control

The fi_cntr_control call is used to access provider or implementation
specific details of the counter. Access to the counter should be
serialized across all calls when fi_cntr_control is invoked, as it may
redirect the implementation of counter operations. The following control
commands are usable with a counter:

*FI_GETOPSFLAG (uint64_t \*)*
:   Returns the current default operational flags associated with the
    counter.

*FI_SETOPSFLAG (uint64_t \*)*
:   Modifies the current default operational flags associated with the
    counter.

*FI_GETWAIT (void \*\*)*
:   This command allows the user to retrieve the low-level wait object
    associated with the counter. The format of the wait-object is
    specified during counter creation, through the counter attributes.
    See fi_eq.3 for addition details using control with FI_GETWAIT.

## fi_cntr_read

The fi_cntr_read call returns the current value of the counter.

## fi_cntr_readerr

The read error call returns the number of operations that completed in
error and were unable to update the counter.

## fi_cntr_add

This adds the user-specified value to the counter.

## fi_cntr_adderr

This adds the user-specified value to the error value of the counter.

## fi_cntr_set

This sets the counter to the specified value.

## fi_cntr_seterr

This sets the error value of the counter to the specified value.

## fi_cntr_wait

This call may be used to wait until the counter reaches the specified
threshold, or until an error or timeout occurs. Upon successful return
from this call, the counter will be greater than or equal to the input
threshold value.

If an operation associated with the counter encounters an error, it will
increment the error value associated with the counter. Any change in a
counter's error value will unblock any thread inside fi_cntr_wait.

If the call returns due to timeout, -FI_ETIMEDOUT will be returned. The
error value associated with the counter remains unchanged.

It is invalid for applications to call this function if the counter has
been configured with a wait object of FI_WAIT_NONE or FI_WAIT_SET.

# RETURN VALUES

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned.

fi_cntr_read / fi_cntr_readerr
:   Returns the current value of the counter.

Fabric errno values are defined in `rdma/fi_errno.h`.

# NOTES

In order to support a variety of counter implementations, updates made
to counter values (e.g. fi_cntr_set or fi_cntr_add) may not be
immediately visible to counter read operations (i.e. fi_cntr_read or
fi_cntr_readerr). A small, but undefined, delay may occur between the
counter changing and the reported value being updated. However, a final
updated value will eventually be reflected in the read counter value.

Additionally, applications should ensure that the value of a counter is
stable and not subject to change prior to calling fi_cntr_set or
fi_cntr_seterr. Otherwise, the resulting value of the counter after
fi_cntr_set / fi_cntr_seterr is undefined, as updates to the counter may
be lost. A counter value is considered stable if all previous updates
using fi_cntr_set / fi_cntr_seterr and results of related operations are
reflected in the observed value of the counter.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_eq`(3)](fi_eq.3.html),
[`fi_poll`(3)](fi_poll.3.html)

{% include JB/setup %}

# NAME

fi_join_collective
:   Operation where a subset of peers join a new collective group.

fi_barrier / fi_barrier2
:   Collective operation that does not complete until all peers have
    entered the barrier call.

fi_broadcast
:   A single sender transmits data to all peers, including itself.

fi_alltoall
:   Each peer distributes a slice of its local data to all peers.

fi_allreduce
:   Collective operation where all peers broadcast an atomic operation
    to all other peers.

fi_allgather
:   Each peer sends a complete copy of its local data to all peers.

fi_reduce_scatter
:   Collective call where data is collected from all peers and merged
    (reduced). The results of the reduction is distributed back to the
    peers, with each peer receiving a slice of the results.

fi_reduce
:   Collective call where data is collected from all peers to a root
    peer and merged (reduced).

fi_scatter
:   A single sender distributes (scatters) a slice of its local data to
    all peers.

fi_gather
:   All peers send their data to a root peer.

fi_query_collective
:   Returns information about which collective operations are supported
    by a provider, and limitations on the collective.

# SYNOPSIS

``` c
#include <rdma/fi_collective.h>

int fi_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
    const struct fid_av_set *set,
    uint64_t flags, struct fid_mc **mc, void *context);

ssize_t fi_barrier(struct fid_ep *ep, fi_addr_t coll_addr,
    void *context);

ssize_t fi_barrier2(struct fid_ep *ep, fi_addr_t coll_addr,
    uint64_t flags, void *context);

ssize_t fi_broadcast(struct fid_ep *ep, void *buf, size_t count, void *desc,
    fi_addr_t coll_addr, fi_addr_t root_addr, enum fi_datatype datatype,
    uint64_t flags, void *context);

ssize_t fi_alltoall(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc,
    fi_addr_t coll_addr, enum fi_datatype datatype,
    uint64_t flags, void *context);

ssize_t fi_allreduce(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc,
    fi_addr_t coll_addr, enum fi_datatype datatype, enum fi_op op,
    uint64_t flags, void *context);

ssize_t fi_allgather(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc,
    fi_addr_t coll_addr, enum fi_datatype datatype,
    uint64_t flags, void *context);

ssize_t fi_reduce_scatter(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc,
    fi_addr_t coll_addr, enum fi_datatype datatype, enum fi_op op,
    uint64_t flags, void *context);

ssize_t fi_reduce(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc, fi_addr_t coll_addr,
    fi_addr_t root_addr, enum fi_datatype datatype, enum fi_op op,
    uint64_t flags, void *context);

ssize_t fi_scatter(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc, fi_addr_t coll_addr,
    fi_addr_t root_addr, enum fi_datatype datatype,
    uint64_t flags, void *context);

ssize_t fi_gather(struct fid_ep *ep, const void *buf, size_t count,
    void *desc, void *result, void *result_desc, fi_addr_t coll_addr,
    fi_addr_t root_addr, enum fi_datatype datatype,
    uint64_t flags, void *context);

int fi_query_collective(struct fid_domain *domain,
    fi_collective_op coll, struct fi_collective_attr *attr, uint64_t flags);
```

# ARGUMENTS

*ep*
:   Fabric endpoint on which to initiate collective operation.

*set*
:   Address vector set defining the collective membership.

*mc*
:   Multicast group associated with the collective.

*buf*
:   Local data buffer that specifies first operand of collective
    operation

*count*
:   The number of elements referenced, where each element is the
    indicated datatype.

*datatype*
:   Datatype associated with atomic operands

*op*
:   Atomic operation to perform

*result*
:   Local data buffer to store the result of the collective operation.

*desc / result_desc*
:   Data descriptor associated with the local data buffer and local
    result buffer, respectively.

*coll_addr*
:   Address referring to the collective group of endpoints.

*root_addr*
:   Single endpoint that is the source or destination of collective
    data.

*flags*
:   Additional flags to apply for the atomic operation

*context*
:   User specified pointer to associate with the operation. This
    parameter is ignored if the operation will not generate a successful
    completion, unless an op flag specifies the context parameter be
    used for required input.

# DESCRIPTION

In general collective operations can be thought of as coordinated atomic
operations between a set of peer endpoints. Readers should refer to the
[`fi_atomic`(3)](fi_atomic.3.html) man page for details on the atomic
operations and datatypes defined by libfabric.

A collective operation is a group communication exchange. It involves
multiple peers exchanging data with other peers participating in the
collective call. Collective operations require close coordination by all
participating members. All participants must invoke the same collective
call before any single member can complete its operation locally. As a
result, collective calls can strain the fabric, as well as local and
remote data buffers.

Libfabric collective interfaces target fabrics that support offloading
portions of the collective communication into network switches, NICs,
and other devices. However, no implementation requirement is placed on
the provider.

The first step in using a collective call is identifying the peer
endpoints that will participate. Collective membership follows one of
two models, both supported by libfabric. In the first model, the
application manages the membership. This usually means that the
application is performing a collective operation itself using point to
point communication to identify the members who will participate.
Additionally, the application may be interacting with a fabric resource
manager to reserve network resources needed to execute collective
operations. In this model, the application will inform libfabric that
the membership has already been established.

A separate model moves the membership management under libfabric and
directly into the provider. In this model, the application must identify
which peer addresses will be members. That information is conveyed to
the libfabric provider, which is then responsible for coordinating the
creation of the collective group. In the provider managed model, the
provider will usually perform the necessary collective operation to
establish the communication group and interact with any fabric
management agents.

In both models, the collective membership is communicated to the
provider by creating and configuring an address vector set (AV set). An
AV set represents an ordered subset of addresses in an address vector
(AV). Details on creating and configuring an AV set are available in
[`fi_av_set`(3)](fi_av_set.3.html).

Once an AV set has been programmed with the collective membership
information, an endpoint is joined to the set. This uses the
fi_join_collective operation and operates asynchronously. This differs
from how an endpoint is associated synchronously with an AV using the
fi_ep_bind() call. Upon completion of the fi_join_collective operation,
an fi_addr is provided that is used as the target address when invoking
a collective operation.

For developer convenience, a set of collective APIs are defined.
Collective APIs differ from message and RMA interfaces in that the
format of the data is known to the provider, and the collective may
perform an operation on that data. This aligns collective operations
closely with the atomic interfaces.

## Join Collective (fi_join_collective)

This call attaches an endpoint to a collective membership group.
Libfabric treats collective members as a multicast group, and the
fi_join_collective call attaches the endpoint to that multicast group.
By default, the endpoint will join the group based on the data transfer
capabilities of the endpoint. For example, if the endpoint has been
configured to both send and receive data, then the endpoint will be able
to initiate and receive transfers to and from the collective. The input
flags may be used to restrict access to the collective group, subject to
endpoint capability limitations.

Join collective operations complete asynchronously, and may involve
fabric transfers, dependent on the provider implementation. An endpoint
must be bound to an event queue prior to calling fi_join_collective. The
result of the join operation will be reported to the EQ as an
FI_JOIN_COMPLETE event. Applications cannot issue collective transfers
until receiving notification that the join operation has completed. Note
that an endpoint may begin receiving messages from the collective group
as soon as the join completes, which can occur prior to the
FI_JOIN_COMPLETE event being generated.

The join collective operation is itself a collective operation. All
participating peers must call fi_join_collective before any individual
peer will report that the join has completed. Application managed
collective memberships are an exception. With application managed
memberships, the fi_join_collective call may be completed locally
without fabric communication. For provider managed memberships, the join
collective call requires as input a coll_addr that refers to either an
address associated with an AV set (see fi_av_set_addr) or an existing
collective group (obtained through a previous call to
fi_join_collective). The fi_join_collective call will create a new
collective subgroup. If application managed memberships are used,
coll_addr should be set to FI_ADDR_UNAVAIL.

Applications must call fi_close on the collective group to disconnect
the endpoint from the group. After a join operation has completed, the
fi_mc_addr call may be used to retrieve the address associated with the
multicast group. See [`fi_cm`(3)](fi_cm.3.html) for additional details
on fi_mc_addr().

## Barrier (fi_barrier)

The fi_barrier operation provides a mechanism to synchronize peers.
Barrier does not result in any data being transferred at the application
level. A barrier does not complete locally until all peers have invoked
the barrier call. This signifies to the local application that work by
peers that completed prior to them calling barrier has finished.

## Barrier (fi_barrier2)

The fi_barrier2 operations is the same as fi_barrier, but with an extra
parameter to pass in operation flags.

## Broadcast (fi_broadcast)

fi_broadcast transfers an array of data from a single sender to all
other members of the collective group. The input buf parameter is
treated as the transmit buffer if the local rank is the root, otherwise
it is the receive buffer. The broadcast operation acts as an atomic
write or read to a data array. As a result, the format of the data in
buf is specified through the datatype parameter. Any non-void datatype
may be broadcast.

The following diagram shows an example of broadcast being used to
transfer an array of integers to a group of peers.

    [1]  [1]  [1]
    [5]  [5]  [5]
    [9]  [9]  [9]
     |____^    ^
     |_________|
     broadcast

## All to All (fi_alltoall)

The fi_alltoall collective involves distributing (or scattering)
different portions of an array of data to peers. It is best explained
using an example. Here three peers perform an all to all collective to
exchange different entries in an integer array.

    [1]   [2]   [3]
    [5]   [6]   [7]
    [9]  [10]  [11]
       \   |   /
       All to all
       /   |   \
    [1]   [5]   [9]
    [2]   [6]  [10]
    [3]   [7]  [11]

Each peer sends a piece of its data to the other peers.

All to all operations may be performed on any non-void datatype.
However, all to all does not perform an operation on the data itself, so
no operation is specified.

## All Reduce (fi_allreduce)

fi_allreduce can be described as all peers providing input into an
atomic operation, with the result copied back to each peer.
Conceptually, this can be viewed as each peer issuing a multicast atomic
operation to all other peers, fetching the results, and combining them.
The combining of the results is referred to as the reduction. The
fi_allreduce() operation takes as input an array of data and the
specified atomic operation to perform. The results of the reduction are
written into the result buffer.

Any non-void datatype may be specified. Valid atomic operations are
listed below in the fi_query_collective call. The following diagram
shows an example of an all reduce operation involving summing an array
of integers between three peers.

     [1]  [1]  [1]
     [5]  [5]  [5]
     [9]  [9]  [9]
       \   |   /
          sum
       /   |   \
     [3]  [3]  [3]
    [15] [15] [15]
    [27] [27] [27]
      All Reduce

## All Gather (fi_allgather)

Conceptually, all gather can be viewed as the opposite of the scatter
component from reduce-scatter. All gather collects data from all peers
into a single array, then copies that array back to each peer.

    [1]  [5]  [9]
      \   |   /
     All gather
      /   |   \
    [1]  [1]  [1]
    [5]  [5]  [5]
    [9]  [9]  [9]

All gather may be performed on any non-void datatype. However, all
gather does not perform an operation on the data itself, so no operation
is specified.

## Reduce-Scatter (fi_reduce_scatter)

The fi_reduce_scatter collective is similar to an fi_allreduce
operation, followed by all to all. With reduce scatter, all peers
provide input into an atomic operation, similar to all reduce. However,
rather than the full result being copied to each peer, each participant
receives only a slice of the result.

This is shown by the following example:

    [1]  [1]  [1]
    [5]  [5]  [5]
    [9]  [9]  [9]
      \   |   /
         sum (reduce)
          |
         [3]
        [15]
        [27]
          |
       scatter
      /   |   \
    [3] [15] [27]

The reduce scatter call supports the same datatype and atomic operation
as fi_allreduce.

## Reduce (fi_reduce)

The fi_reduce collective is the first half of an fi_allreduce operation.
With reduce, all peers provide input into an atomic operation, with the
the results collected by a single 'root' endpoint.

This is shown by the following example, with the leftmost peer
identified as the root:

    [1]  [1]  [1]
    [5]  [5]  [5]
    [9]  [9]  [9]
      \   |   /
         sum (reduce)
        /
     [3]
    [15]
    [27]

The reduce call supports the same datatype and atomic operation as
fi_allreduce.

## Scatter (fi_scatter)

The fi_scatter collective is the second half of an fi_reduce_scatter
operation. The data from a single 'root' endpoint is split and
distributed to all peers.

This is shown by the following example:

     [3]
    [15]
    [27]
        \
       scatter
      /   |   \
    [3] [15] [27]

The scatter operation is used to distribute results to the peers. No
atomic operation is performed on the data.

## Gather (fi_gather)

The fi_gather operation is used to collect (gather) the results from all
peers and store them at a 'root' peer.

This is shown by the following example, with the leftmost peer
identified as the root.

    [1]  [5]  [9]
      \   |   /
        gather
       /
    [1]
    [5]
    [9]

The gather operation does not perform any operation on the data itself.

## Query Collective Attributes (fi_query_collective)

The fi_query_collective call reports which collective operations are
supported by the underlying provider, for suitably configured endpoints.
Collective operations needed by an application that are not supported by
the provider must be implemented by the application. The query call
checks whether a provider supports a specific collective operation for a
given datatype and operation, if applicable.

The name of the collective, as well as the datatype and associated
operation, if applicable, and are provided as input into
fi_query_collective.

The coll parameter may reference one of these collectives: FI_BARRIER,
FI_BROADCAST, FI_ALLTOALL, FI_ALLREDUCE, FI_ALLGATHER,
FI_REDUCE_SCATTER, FI_REDUCE, FI_SCATTER, or FI_GATHER. Additional
details on the collective operation is specified through the struct
fi_collective_attr parameter. For collectives that act on data, the
operation and related data type must be specified through the given
attributes.

{% highlight c %} struct fi_collective_attr { enum fi_op op; enum
fi_datatype datatype; struct fi_atomic_attr datatype_attr; size_t
max_members; uint64_t mode; }; {% endhighlight %}

For a description of struct fi_atomic_attr, see
[`fi_atomic`(3)](fi_atomic.3.html).

*op*
:   On input, this specifies the atomic operation involved with the
    collective call. This should be set to one of the following values:
    FI_MIN, FI_MAX, FI_SUM, FI_PROD, FI_LOR, FI_LAND, FI_BOR, FI_BAND,
    FI_LXOR, FI_BXOR, FI_ATOMIC_READ, FI_ATOMIC_WRITE, of FI_NOOP. For
    collectives that do not exchange application data (fi_barrier), this
    should be set to FI_NOOP.

*datatype*
:   On onput, specifies the datatype of the data being modified by the
    collective. This should be set to one of the following values:
    FI_INT8, FI_UINT8, FI_INT16, FI_UINT16, FI_INT32, FI_UINT32,
    FI_INT64, FI_UINT64, FI_FLOAT, FI_DOUBLE, FI_FLOAT_COMPLEX,
    FI_DOUBLE_COMPLEX, FI_LONG_DOUBLE, FI_LONG_DOUBLE_COMPLEX, or
    FI_VOID. For collectives that do not exchange application data
    (fi_barrier), this should be set to FI_VOID.

*datatype_attr.count*
:   The maximum number of elements that may be used with the collective.

*datatype_attr.size*
:   The size of the datatype as supported by the provider. Applications
    should validate the size of datatypes that differ based on the
    platform, such as FI_LONG_DOUBLE.

*max_members*
:   The maximum number of peers that may participate in a collective
    operation.

*mode*
:   This field is reserved and should be 0.

If a collective operation is supported, the query call will return
FI_SUCCESS, along with attributes on the limits for using that
collective operation through the provider.

## Completions

Collective operations map to underlying fi_atomic operations. For a
discussion of atomic completion semantics, see
[`fi_atomic`(3)](fi_atomic.3.html). The completion, ordering, and
atomicity of collective operations match those defined for point to
point atomic operations.

# FLAGS

The following flags are defined for the specified operations.

*FI_SCATTER*
:   Applies to fi_query_collective. When set, requests attribute
    information on the reduce-scatter collective operation.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
:   See [`fi_msg`(3)](fi_msg.3.html) for a detailed description of
    handling FI_EAGAIN.

*-FI_EOPNOTSUPP*
:   The requested atomic operation is not supported on this endpoint.

*-FI_EMSGSIZE*
:   The number of collective operations in a single request exceeds that
    supported by the underlying provider.

# NOTES

Collective operations map to atomic operations. As such, they follow
most of the conventions and restrictions as peer to peer atomic
operations. This includes data atomicity, data alignment, and message
ordering semantics. See [`fi_atomic`(3)](fi_atomic.3.html) for
additional information on the datatypes and operations defined for
atomic and collective operations.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html), [`fi_av`(3)](fi_av.3.html),
[`fi_atomic`(3)](fi_atomic.3.html), [`fi_cm`(3)](fi_cm.3.html)

{% include JB/setup %}

# NAME

fi_control - Perform an operation on a fabric resource.

# SYNOPSIS

``` c
#include <rdma/fabric.h>

int fi_control(struct fid *fid, int command, void *arg);
int fi_alias(struct fid *fid, struct fid **alias_fid, uint64_t flags);
int fi_get_val(struct fid *fid, int name, void *val);
int fi_set_val(struct fid *fid, int name, void *val);
```

# ARGUMENTS

*fid*
:   Fabric resource

*command*
:   Operation to perform

*arg*
:   Optional argument to the command

# DESCRIPTION

The fi_control operation is used to perform one or more operations on a
fabric resource. Conceptually, fi_control is similar to the POSIX fcntl
routine. The exact behavior of using fi_control depends on the fabric
resource being operated on, the specified command, and any provided
arguments for the command. For specific details, see the fabric resource
specific help pages noted below.

fi_alias, fi_get_val, and fi_set_val are wrappers for fi_control with
commands FI_ALIAS, FI_GET_VAL, FI_SET_VAL, respectively. fi_alias
creates an alias of the specified fabric resource. fi_get_val reads the
value of the named parameter associated with the fabric resource, while
fi_set_val updates that value. Available parameter names depend on the
type of the fabric resource and the provider in use. Providers may
define provider specific names in the provider extension header files
('rdma/fi_ext\_\*.h'). Please refer to the provider man pages for
details.

# SEE ALSO

[`fi_endpoint`(3)](fi_endpoint.3.html), [`fi_cm`(3)](fi_cm.3.html),
[`fi_cntr`(3)](fi_cntr.3.html), [`fi_cq`(3)](fi_cq.3.html),
[`fi_eq`(3)](fi_eq.3.html),

{% include JB/setup %}

# NAME

fi_cq - Completion queue operations

fi_cq_open / fi_close
:   Open/close a completion queue

fi_control
:   Control CQ operation or attributes.

fi_cq_read / fi_cq_readfrom / fi_cq_readerr
:   Read a completion from a completion queue

fi_cq_sread / fi_cq_sreadfrom
:   A synchronous (blocking) read that waits until a specified condition
    has been met before reading a completion from a completion queue.

fi_cq_signal
:   Unblock any thread waiting in fi_cq_sread or fi_cq_sreadfrom.

fi_cq_strerror
:   Converts provider specific error information into a printable string

# SYNOPSIS

``` c
#include <rdma/fi_domain.h>

int fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
    struct fid_cq **cq, void *context);

int fi_close(struct fid *cq);

int fi_control(struct fid *cq, int command, void *arg);

ssize_t fi_cq_read(struct fid_cq *cq, void *buf, size_t count);

ssize_t fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
    fi_addr_t *src_addr);

ssize_t fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
    uint64_t flags);

ssize_t fi_cq_sread(struct fid_cq *cq, void *buf, size_t count,
    const void *cond, int timeout);

ssize_t fi_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
    fi_addr_t *src_addr, const void *cond, int timeout);

int fi_cq_signal(struct fid_cq *cq);

const char * fi_cq_strerror(struct fid_cq *cq, int prov_errno,
      const void *err_data, char *buf, size_t len);
```

# ARGUMENTS

*domain*
:   Open resource domain

*cq*
:   Completion queue

*attr*
:   Completion queue attributes

*context*
:   User specified context associated with the completion queue.

*buf*
:   For read calls, the data buffer to write completions into. For write
    calls, a completion to insert into the completion queue. For
    fi_cq_strerror, an optional buffer that receives printable error
    information.

*count*
:   Number of CQ entries.

*len*
:   Length of data buffer

*src_addr*
:   Source address of a completed receive operation

*flags*
:   Additional flags to apply to the operation

*command*
:   Command of control operation to perform on CQ.

*arg*
:   Optional control argument

*cond*
:   Condition that must be met before a completion is generated

*timeout*
:   Time in milliseconds to wait. A negative value indicates infinite
    timeout.

*prov_errno*
:   Provider specific error value

*err_data*
:   Provider specific error data related to a completion

# DESCRIPTION

Completion queues are used to report events associated with data
transfers. They are associated with message sends and receives, RMA,
atomic, tagged messages, and triggered events. Reported events are
usually associated with a fabric endpoint, but may also refer to memory
regions used as the target of an RMA or atomic operation.

## fi_cq_open

fi_cq_open allocates a new completion queue. Unlike event queues,
completion queues are associated with a resource domain and may be
offloaded entirely in provider hardware.

The properties and behavior of a completion queue are defined by
`struct fi_cq_attr`.

``` c
struct fi_cq_attr {
    size_t               size;      /* # entries for CQ */
    uint64_t             flags;     /* operation flags */
    enum fi_cq_format    format;    /* completion format */
    enum fi_wait_obj     wait_obj;  /* requested wait object */
    int                  signaling_vector; /* interrupt affinity */
    enum fi_cq_wait_cond wait_cond; /* wait condition format */
    struct fid_wait     *wait_set;  /* optional wait set, deprecated */
};
```

*size*
:   Specifies the minimum size of a completion queue. A value of 0
    indicates that the provider may choose a default value.

*flags*
:   Flags that control the configuration of the CQ.

\- *FI_AFFINITY*
:   Indicates that the signaling_vector field (see below) is valid.

*format*
:   Completion queues allow the application to select the amount of
    detail that it must store and report. The format attribute allows
    the application to select one of several completion formats,
    indicating the structure of the data that the completion queue
    should return when read. Supported formats and the structures that
    correspond to each are listed below. The meaning of the CQ entry
    fields are defined in the *Completion Fields* section.

\- *FI_CQ_FORMAT_UNSPEC*
:   If an unspecified format is requested, then the CQ will use a
    provider selected default format.

\- *FI_CQ_FORMAT_CONTEXT*
:   Provides only user specified context that was associated with the
    completion.

``` c
struct fi_cq_entry {
    void     *op_context; /* operation context */
};
```

-   

    *FI_CQ_FORMAT_MSG*
    :   Provides minimal data for processing completions, with expanded
        support for reporting information about received messages.

``` c
struct fi_cq_msg_entry {
    void     *op_context; /* operation context */
    uint64_t flags;       /* completion flags */
    size_t   len;         /* size of received data */
};
```

-   

    *FI_CQ_FORMAT_DATA*
    :   Provides data associated with a completion. Includes support for
        received message length, remote CQ data, and multi-receive
        buffers.

``` c
struct fi_cq_data_entry {
    void     *op_context; /* operation context */
    uint64_t flags;       /* completion flags */
    size_t   len;         /* size of received data */
    void     *buf;        /* receive data buffer */
    uint64_t data;        /* completion data */
};
```

-   

    *FI_CQ_FORMAT_TAGGED*
    :   Expands completion data to include support for the tagged
        message interfaces.

``` c
struct fi_cq_tagged_entry {
    void     *op_context; /* operation context */
    uint64_t flags;       /* completion flags */
    size_t   len;         /* size of received data */
    void     *buf;        /* receive data buffer */
    uint64_t data;        /* completion data */
    uint64_t tag;         /* received tag */
};
```

*wait_obj*
:   CQ's may be associated with a specific wait object. Wait objects
    allow applications to block until the wait object is signaled,
    indicating that a completion is available to be read. Users may use
    fi_control to retrieve the underlying wait object associated with a
    CQ, in order to use it in other system calls. The following values
    may be used to specify the type of wait object associated with a CQ:
    FI_WAIT_NONE, FI_WAIT_UNSPEC, FI_WAIT_SET, FI_WAIT_FD,
    FI_WAIT_MUTEX_COND (deprecated), and FI_WAIT_YIELD. The default is
    FI_WAIT_NONE.

\- *FI_WAIT_NONE*
:   Used to indicate that the user will not block (wait) for completions
    on the CQ. When FI_WAIT_NONE is specified, the application may not
    call fi_cq_sread or fi_cq_sreadfrom.

\- *FI_WAIT_UNSPEC*
:   Specifies that the user will only wait on the CQ using fabric
    interface calls, such as fi_cq_sread or fi_cq_sreadfrom. In this
    case, the underlying provider may select the most appropriate or
    highest performing wait object available, including custom wait
    mechanisms. Applications that select FI_WAIT_UNSPEC are not
    guaranteed to retrieve the underlying wait object.

\- *FI_WAIT_SET* (deprecated)
:   Indicates that the completion queue should use a wait set object to
    wait for completions. If specified, the wait_set field must
    reference an existing wait set object.

\- *FI_WAIT_FD*
:   Indicates that the CQ should use a file descriptor as its wait
    mechanism. A file descriptor wait object must be usable in select,
    poll, and epoll routines. However, a provider may signal an FD wait
    object by marking it as readable, writable, or with an error.

\- *FI_WAIT_MUTEX_COND* (deprecated)
:   Specifies that the CQ should use a pthread mutex and cond variable
    as a wait object.

\- *FI_WAIT_YIELD*
:   Indicates that the CQ will wait without a wait object but instead
    yield on every wait. Allows usage of fi_cq_sread and fi_cq_sreadfrom
    through a spin.

*signaling_vector*
:   If the FI_AFFINITY flag is set, this indicates the logical cpu
    number (0..max cpu - 1) that interrupts associated with the CQ
    should target. This field should be treated as a hint to the
    provider and may be ignored if the provider does not support
    interrupt affinity.

*wait_cond*
:   By default, when a completion is inserted into a CQ that supports
    blocking reads (fi_cq_sread/fi_cq_sreadfrom), the corresponding wait
    object is signaled. Users may specify a condition that must first be
    met before the wait is satisfied. This field indicates how the
    provider should interpret the cond field, which describes the
    condition needed to signal the wait object.

A wait condition should be treated as an optimization. Providers are not
required to meet the requirements of the condition before signaling the
wait object. Applications should not rely on the condition necessarily
being true when a blocking read call returns.

If wait_cond is set to FI_CQ_COND_NONE, then no additional conditions
are applied to the signaling of the CQ wait object, and the insertion of
any new entry will trigger the wait condition. If wait_cond is set to
FI_CQ_COND_THRESHOLD, then the cond field is interpreted as a size_t
threshold value. The threshold indicates the number of entries that are
to be queued before at the CQ before the wait is satisfied.

This field is ignored if wait_obj is set to FI_WAIT_NONE.

*wait_set* (deprecated)
:   If wait_obj is FI_WAIT_SET, this field references a wait object to
    which the completion queue should attach. When an event is inserted
    into the completion queue, the corresponding wait set will be
    signaled if all necessary conditions are met. The use of a wait_set
    enables an optimized method of waiting for events across multiple
    event and completion queues. This field is ignored if wait_obj is
    not FI_WAIT_SET.

## fi_close

The fi_close call releases all resources associated with a completion
queue. Any completions which remain on the CQ when it is closed are
lost.

When closing the CQ, there must be no opened endpoints, transmit
contexts, or receive contexts associated with the CQ. If resources are
still associated with the CQ when attempting to close, the call will
return -FI_EBUSY.

## fi_control

The fi_control call is used to access provider or implementation
specific details of the completion queue. Access to the CQ should be
serialized across all calls when fi_control is invoked, as it may
redirect the implementation of CQ operations. The following control
commands are usable with a CQ.

*FI_GETWAIT (void \*\*)*
:   This command allows the user to retrieve the low-level wait object
    associated with the CQ. The format of the wait-object is specified
    during CQ creation, through the CQ attributes. The fi_control arg
    parameter should be an address where a pointer to the returned wait
    object will be written. See fi_eq.3 for addition details using
    fi_control with FI_GETWAIT.

## fi_cq_read

The fi_cq_read operation performs a non-blocking read of completion data
from the CQ. The format of the completion event is determined using the
fi_cq_format option that was specified when the CQ was opened. Multiple
completions may be retrieved from a CQ in a single call. The maximum
number of entries to return is limited to the specified count parameter,
with the number of entries successfully read from the CQ returned by the
call. (See return values section below.) A count value of 0 may be used
to drive progress on associated endpoints when manual progress is
enabled.

CQs are optimized to report operations which have completed
successfully. Operations which fail are reported 'out of band'. Such
operations are retrieved using the fi_cq_readerr function. When an
operation that has completed with an unexpected error is encountered, it
is placed into a temporary error queue. Attempting to read from a CQ
while an item is in the error queue results in fi_cq_read failing with a
return code of -FI_EAVAIL. Applications may use this return code to
determine when to call fi_cq_readerr.

## fi_cq_readfrom

The fi_cq_readfrom call behaves identical to fi_cq_read, with the
exception that it allows the CQ to return source address information to
the user for any received data. Source address data is only available
for those endpoints configured with FI_SOURCE capability. If
fi_cq_readfrom is called on an endpoint for which source addressing data
is not available, the source address will be set to FI_ADDR_NOTAVAIL.
The number of input src_addr entries must be the same as the count
parameter.

Returned source addressing data is converted from the native address
used by the underlying fabric into an fi_addr_t, which may be used in
transmit operations. Under most circumstances, returning fi_addr_t
requires that the source address already have been inserted into the
address vector associated with the receiving endpoint. This is true for
address vectors of type FI_AV_TABLE. In select providers when FI_AV_MAP
is used, source addresses may be converted algorithmically into a usable
fi_addr_t, even though the source address has not been inserted into the
address vector. This is permitted by the API, as it allows the provider
to avoid address look-up as part of receive message processing. In no
case do providers insert addresses into an AV separate from an
application calling fi_av_insert or similar call.

For endpoints allocated using the FI_SOURCE_ERR capability, if the
source address cannot be converted into a valid fi_addr_t value,
fi_cq_readfrom will return -FI_EAVAIL, even if the data were received
successfully. The completion will then be reported through fi_cq_readerr
with error code -FI_EADDRNOTAVAIL. See fi_cq_readerr for details.

If FI_SOURCE is specified without FI_SOURCE_ERR, source addresses which
cannot be mapped to a usable fi_addr_t will be reported as
FI_ADDR_NOTAVAIL.

## fi_cq_sread / fi_cq_sreadfrom

The fi_cq_sread and fi_cq_sreadfrom calls are the blocking equivalent
operations to fi_cq_read and fi_cq_readfrom. Their behavior is similar
to the non-blocking calls, with the exception that the calls will not
return until either a completion has been read from the CQ or an error
or timeout occurs.

Threads blocking in this function will return to the caller if they are
signaled by some external source. This is true even if the timeout has
not occurred or was specified as infinite.

It is invalid for applications to call these functions if the CQ has
been configured with a wait object of FI_WAIT_NONE or FI_WAIT_SET.

## fi_cq_readerr

The read error function, fi_cq_readerr, retrieves information regarding
any asynchronous operation which has completed with an unexpected error.
fi_cq_readerr is a non-blocking call, returning immediately whether an
error completion was found or not.

Error information is reported to the user through
`struct fi_cq_err_entry`. The format of this structure is defined below.

``` c
struct fi_cq_err_entry {
    void     *op_context; /* operation context */
    uint64_t flags;       /* completion flags */
    size_t   len;         /* size of received data */
    void     *buf;        /* receive data buffer */
    uint64_t data;        /* completion data */
    uint64_t tag;         /* message tag */
    size_t   olen;        /* overflow length */
    int      err;         /* positive error code */
    int      prov_errno;  /* provider error code */
    void    *err_data;    /*  error data */
    size_t   err_data_size; /* size of err_data */
    fi_addr_t src_addr; /* error source address */
};
```

The general reason for the error is provided through the err field.
Provider specific error information may also be available through the
prov_errno and err_data fields. Users may call fi_cq_strerror to convert
provider specific error information into a printable string for
debugging purposes. See field details below for more information on the
use of err_data and err_data_size.

Note that error completions are generated for all operations, including
those for which a completion was not requested (e.g. an endpoint is
configured with FI_SELECTIVE_COMPLETION, but the request did not have
the FI_COMPLETION flag set). In such cases, providers will return as
much information as made available by the underlying software and
hardware about the failure, other fields will be set to NULL or 0. This
includes the op_context value, which may not have been provided or was
ignored on input as part of the transfer.

Notable completion error codes are given below.

*FI_EADDRNOTAVAIL*
:   This error code is used by CQs configured with FI_SOURCE_ERR to
    report completions for which a usable fi_addr_t source address could
    not be found. An error code of FI_EADDRNOTAVAIL indicates that the
    data transfer was successfully received and processed, with the
    fi_cq_err_entry fields containing information about the completion.
    The err_data field will be set to the source address data. The
    source address will be in the same format as specified through the
    fi_info addr_format field for the opened domain. This may be passed
    directly into an fi_av_insert call to add the source address to the
    address vector.

For API versions 1.20 and later, if the EP is configured with
FI_AV_AUTH_KEY, src_addr will be set to the fi_addr_t authorization key
handle or a user-define authorization key ID corresponding to the
incoming data transfer. Otherwise, the value will be set to
FI_ADDR_NOTAVAIL.

## fi_cq_signal

The fi_cq_signal call will unblock any thread waiting in fi_cq_sread or
fi_cq_sreadfrom. This may be used to wake-up a thread that is blocked
waiting to read a completion operation. The fi_cq_signal operation is
only available if the CQ was configured with a wait object.

# COMPLETION FIELDS

The CQ entry data structures share many of the same fields. The meanings
of these fields are the same for all CQ entry structure formats.

*op_context*
:   The operation context is the application specified context value
    that was provided with an asynchronous operation. The op_context
    field is valid for all completions that are associated with an
    asynchronous operation.

For completion events that are not associated with a posted operation,
this field will be set to NULL. This includes completions generated at
the target in response to RMA write operations that carry CQ data
(FI_REMOTE_WRITE \| FI_REMOTE_CQ_DATA flags set), when the FI_RX_CQ_DATA
mode bit is not required.

*flags*
:   This specifies flags associated with the completed operation. The
    *Completion Flags* section below lists valid flag values. Flags are
    set for all relevant completions.

*len*
:   This len field applies to completed receive operations
    (e.g. fi_recv, fi_trecv, etc.) and the completed write with remote
    cq data on the responder side (e.g. fi_write, with FI_REMOTE_CQ_DATA
    flag). It indicates the size of transferred *message* data --
    i.e. how many data bytes were placed into the associated
    receive/target buffer by a corresponding fi_send/fi_tsend/fi_write
    et al call. If an endpoint has been configured with the
    FI_MSG_PREFIX mode, the len also reflects the size of the prefix
    buffer.

*buf*
:   The buf field is only valid for completed receive operations, and
    only applies when the receive buffer was posted with the
    FI_MULTI_RECV flag. In this case, buf points to the starting
    location where the receive data was placed.

*data*
:   The data field is only valid if the FI_REMOTE_CQ_DATA completion
    flag is set, and only applies to receive completions. If
    FI_REMOTE_CQ_DATA is set, this field will contain the completion
    data provided by the peer as part of their transmit request. The
    completion data will be given in host byte order.

*tag*
:   A tag applies only to received messages that occur using the tagged
    interfaces. This field contains the tag that was included with the
    received message. The tag will be in host byte order.

*olen*
:   The olen field applies to received messages. It is used to indicate
    that a received message has overrun the available buffer space and
    has been truncated. The olen specifies the amount of data that did
    not fit into the available receive buffer and was discarded.

*err*
:   This err code is a positive fabric errno associated with a
    completion. The err value indicates the general reason for an error,
    if one occurred. See fi_errno.3 for a list of possible error codes.

*prov_errno*
:   On an error, prov_errno may contain a provider specific error code.
    The use of this field and its meaning is provider specific. It is
    intended to be used as a debugging aid. See fi_cq_strerror for
    additional details on converting this error value into a human
    readable string.

*err_data*
:   The err_data field is used to return provider specific information,
    if available, about the error. On input, err_data should reference a
    data buffer of size err_data_size. On output, the provider will fill
    in this buffer with any provider specific data which may help
    identify the cause of the error. The contents of the err_data field
    and its meaning is provider specific. It is intended to be used as a
    debugging aid. See fi_cq_strerror for additional details on
    converting this error data into a human readable string. See the
    compatibility note below on how this field is used for older
    libfabric releases.

*err_data_size*
:   On input, err_data_size indicates the size of the err_data buffer in
    bytes. On output, err_data_size will be set to the number of bytes
    copied to the err_data buffer. The err_data information is typically
    used with fi_cq_strerror to provide details about the type of error
    that occurred.

For compatibility purposes, the behavior of the err_data and
err_data_size fields is may be modified from that listed above. If
err_data_size is 0 on input, or the fabric was opened with release \<
1.5, then any buffer referenced by err_data will be ignored on input. In
this situation, on output err_data will be set to a data buffer owned by
the provider. The contents of the buffer will remain valid until a
subsequent read call against the CQ. Applications must serialize access
to the CQ when processing errors to ensure that the buffer referenced by
err_data does not change.

*src_addr*
:   Used to return source addressed related information for error
    events. How this field is used is error event specific.

# COMPLETION FLAGS

Completion flags provide additional details regarding the completed
operation. The following completion flags are defined.

*FI_SEND*
:   Indicates that the completion was for a send operation. This flag
    may be combined with an FI_MSG or FI_TAGGED flag.

*FI_RECV*
:   Indicates that the completion was for a receive operation. This flag
    may be combined with an FI_MSG or FI_TAGGED flag.

*FI_RMA*
:   Indicates that an RMA operation completed. This flag may be combined
    with an FI_READ, FI_WRITE, FI_REMOTE_READ, or FI_REMOTE_WRITE flag.

*FI_ATOMIC*
:   Indicates that an atomic operation completed. This flag may be
    combined with an FI_READ, FI_WRITE, FI_REMOTE_READ, or
    FI_REMOTE_WRITE flag.

*FI_MSG*
:   Indicates that a message-based operation completed. This flag may be
    combined with an FI_SEND or FI_RECV flag.

*FI_TAGGED*
:   Indicates that a tagged message operation completed. This flag may
    be combined with an FI_SEND or FI_RECV flag.

*FI_MULTICAST*
:   Indicates that a multicast operation completed. This flag may be
    combined with FI_MSG and relevant flags. This flag is only
    guaranteed to be valid for received messages if the endpoint has
    been configured with FI_SOURCE.

*FI_READ*
:   Indicates that a locally initiated RMA or atomic read operation has
    completed. This flag may be combined with an FI_RMA or FI_ATOMIC
    flag.

*FI_WRITE*
:   Indicates that a locally initiated RMA or atomic write operation has
    completed. This flag may be combined with an FI_RMA or FI_ATOMIC
    flag.

*FI_REMOTE_READ*
:   Indicates that a remotely initiated RMA or atomic read operation has
    completed. This flag may be combined with an FI_RMA or FI_ATOMIC
    flag.

*FI_REMOTE_WRITE*
:   Indicates that a remotely initiated RMA or atomic write operation
    has completed. This flag may be combined with an FI_RMA or FI_ATOMIC
    flag.

*FI_REMOTE_CQ_DATA*
:   This indicates that remote CQ data is available as part of the
    completion.

*FI_MULTI_RECV*
:   This flag applies to receive buffers that were posted with the
    FI_MULTI_RECV flag set. This completion flag indicates that the
    original receive buffer referenced by the completion has been
    consumed and was released by the provider. Providers may set this
    flag on the last message that is received into the multi- recv
    buffer, or may generate a separate completion that indicates that
    the buffer has been released.

Applications can distinguish between these two cases by examining the
completion entry flags field. If additional flags, such as FI_RECV, are
set, the completion is associated with a received message. In this case,
the buf field will reference the location where the received message was
placed into the multi-recv buffer. Other fields in the completion entry
will be determined based on the received message. If other flag bits are
zero, the provider is reporting that the multi-recv buffer has been
released, and the completion entry is not associated with a received
message.

# COMPLETION EVENT SEMANTICS

Libfabric defines several completion 'levels', identified using
operational flags. Each flag indicates the soonest that a completion
event may be generated by a provider, and the assumptions that an
application may make upon processing a completion. The operational flags
are defined below, along with an example of how a provider might
implement the semantic. Note that only meeting the semantic is required
of the provider and not the implementation. Providers may implement
stronger completion semantics than necessary for a given operation, but
only the behavior defined by the completion level is guaranteed.

To help understand the conceptual differences in completion levels,
consider mailing a letter. Placing the letter into the local mailbox for
pick-up is similar to 'inject complete'. Having the letter picked up and
dropped off at the destination mailbox is equivalent to 'transmit
complete'. The 'delivery complete' semantic is a stronger guarantee,
with a person at the destination signing for the letter. However, the
person who signed for the letter is not necessarily the intended
recipient. The 'match complete' option is similar to delivery complete,
but requires the intended recipient to sign for the letter.

The 'commit complete' level has different semantics than the previously
mentioned levels. Commit complete would be closer to the letter arriving
at the destination and being placed into a fire proof safe.

The operational flags for the described completion levels are defined
below.

*FI_INJECT_COMPLETE*
:   Indicates that a completion should be generated when the source
    buffer(s) may be reused. A completion guarantees that the buffers
    will not be read from again and the application may reclaim them. No
    other guarantees are made with respect to the state of the
    operation.

Example: A provider may generate this completion event after copying the
source buffer into a network buffer, either in host memory or on the
NIC. An inject completion does not indicate that the data has been
transmitted onto the network, and a local error could occur after the
completion event has been generated that could prevent it from being
transmitted.

Inject complete allows for the fastest completion reporting (and, hence,
buffer reuse), but provides the weakest guarantees against network
errors.

Note: This flag is used to control when a completion entry is inserted
into a completion queue. It does not apply to operations that do not
generate a completion queue entry, such as the fi_inject operation, and
is not subject to the inject_size message limit restriction.

*FI_TRANSMIT_COMPLETE*
:   Indicates that a completion should be generated when the transmit
    operation has completed relative to the local provider. The exact
    behavior is dependent on the endpoint type.

For reliable endpoints:

Indicates that a completion should be generated when the operation has
been delivered to the peer endpoint. A completion guarantees that the
operation is no longer dependent on the fabric or local resources. The
state of the operation at the peer endpoint is not defined.

Example: A provider may generate a transmit complete event upon
receiving an ack from the peer endpoint. The state of the message at the
peer is unknown and may be buffered in the target NIC at the time the
ack has been generated.

For unreliable endpoints:

Indicates that a completion should be generated when the operation has
been delivered to the fabric. A completion guarantees that the operation
is no longer dependent on local resources. The state of the operation
within the fabric is not defined.

*FI_DELIVERY_COMPLETE*
:   Indicates that a completion should not be generated until an
    operation has been processed by the destination endpoint(s). A
    completion guarantees that the result of the operation is available;
    however, additional steps may need to be taken at the destination to
    retrieve the results. For example, an application may need to
    provide a receive buffers in order to retrieve messages that were
    buffered by the provider.

Delivery complete indicates that the message has been processed by the
peer. If an application buffer was ready to receive the results of the
message when it arrived, then delivery complete indicates that the data
was placed into the application's buffer.

This completion mode applies only to reliable endpoints. For operations
that return data to the initiator, such as RMA read or atomic-fetch, the
source endpoint is also considered a destination endpoint. This is the
default completion mode for such operations.

*FI_MATCH_COMPLETE*
:   Indicates that a completion should be generated only after the
    operation has been matched with an application specified buffer.
    Operations using this completion semantic are dependent on the
    application at the target claiming the message or results. As a
    result, match complete may involve additional provider level
    acknowledgements or lengthy delays. However, this completion model
    enables peer applications to synchronize their execution. Many
    providers may not support this semantic.

*FI_COMMIT_COMPLETE*
:   Indicates that a completion should not be generated (locally or at
    the peer) until the result of an operation have been made
    persistent. A completion guarantees that the result is both
    available and durable, in the case of power failure.

This completion mode applies only to operations that target persistent
memory regions over reliable endpoints. This completion mode is
experimental.

*FI_FENCE*
:   This is not a completion level, but plays a role in the completion
    ordering between operations that would not normally be ordered. An
    operation that is marked with the FI_FENCE flag and all operations
    posted after the fenced operation are deferred until all previous
    operations targeting the same peer endpoint have completed.
    Additionally, the completion of the fenced operation indicates that
    prior operations have met the same completion level as the fenced
    operation. For example, if an operation is posted as
    FI_DELIVERY_COMPLETE \| FI_FENCE, then its completion indicates
    prior operations have met the semantic required for
    FI_DELIVERY_COMPLETE. This is true even if the prior operation was
    posted with a lower completion level, such as FI_TRANSMIT_COMPLETE
    or FI_INJECT_COMPLETE.

Note that a completion generated for an operation posted prior to the
fenced operation only guarantees that the completion level that was
originally requested has been met. It is the completion of the fenced
operation that guarantees that the additional semantics have been met.

The above completion semantics are defined with respect to the initiator
of the operation. The different semantics are useful for describing when
the initiator may re-use a data buffer, and guarantees what state a
transfer must reach prior to a completion being generated. This allows
applications to determine appropriate error handling in case of
communication failures.

# TARGET COMPLETION SEMANTICS

The completion semantic at the target is used to determine when data at
the target is visible to the peer application. Visibility indicates that
a memory read to the same address that was the target of a data transfer
will return the results of the transfer. The target of a transfer can be
identified by the initiator, as may be the case for RMA and atomic
operations, or determined by the target, for example by providing a
matching receive buffer. Global visibility indicates that the results
are available regardless of where the memory read originates. For
example, the read could come from a process running on a host CPU, it
may be accessed by subsequent data transfer over the fabric, or read
from a peer device such as a GPU.

In terms of completion semantics, visibility usually indicates that the
transfer meets the FI_DELIVERY_COMPLETE requirements from the
perspective of the target. The target completion semantic may be, but is
not necessarily, linked with the completion semantic specified by the
initiator of the transfer.

Often, target processes do not explicitly state a desired completion
semantic and instead rely on the default semantic. The default behavior
is based on several factors, including:

-   whether a completion even is generated at the target
-   the type of transfer involved (e.g. msg vs RMA)
-   endpoint data and message ordering guarantees
-   properties of the targeted memory buffer
-   the initiator's specified completion semantic

Broadly, target completion semantics are grouped based on whether or not
the transfer generates a completion event at the target. This includes
writing a CQ entry or updating a completion counter. In common use
cases, transfers that use a message interface (FI_MSG or FI_TAGGED)
typically generate target events, while transfers involving an RMA
interface (FI_RMA or FI_ATOMIC) often do not. There are exceptions to
both these cases, depending on endpoint to CQ and counter bindings and
operational flags. For example, RMA writes that carry remote CQ data
will generate a completion event at the target, and are frequently used
to convey visibility to the target application. The general guidelines
for target side semantics are described below, followed by exceptions
that modify that behavior.

By default, completions generated at the target indicate that the
transferred data is immediately available to be read from the target
buffer. That is, the target sees FI_DELIVERY_COMPLETE (or better)
semantics, even if the initiator requested lower semantics. For
applications using only data buffers allocated from host memory, this is
often sufficient.

For operations that do not generate a completion event at the target,
the visibility of the data at the target may need to be inferred based
on subsequent operations that do generate target completions. Absent a
target completion, when a completion of an operation is written at the
initiator, the visibility semantic of the operation at the target aligns
with the initiator completion semantic. For instance, if an RMA
operation completes at the initiator as either FI_INJECT_COMPLETE or
FI_TRANSMIT_COMPLETE, the data visibility at the target is not
guaranteed.

One or more of the following mechanisms can be used by the target
process to guarantee that the results of a data transfer that did not
generate a completion at the target is now visible. This list is not
inclusive of all options, but defines common uses. In the descriptions
below, the first transfer does not result in a completion event at the
target, but is eventually followed by a transfer which does.

-   If the endpoint guarantees message ordering between two transfers,
    the target completion of a second transfer will indicate that the
    data from the first transfer is available. For example, if the
    endpoint supports send after write ordering (FI_ORDER_SAW), then a
    receive completion corresponding to the send will indicate that the
    write data is available. This holds independent of the initiator's
    completion semantic for either the write or send. When ordering is
    guaranteed, the second transfer can be queued with the provider
    immediately after queuing the first.

-   If the endpoint does not guarantee message ordering, the initiator
    must take additional steps to ensure visibility. If initiator
    requests FI_DELIVERY_COMPLETE semantics for the first operation, the
    initiator can wait for the operation to complete locally. Once the
    completion has been read, the target completion of a second transfer
    will indicate that the first transfer's data is visible.

-   Alternatively, if message ordering is not guaranteed by the
    endpoint, the initiator can use the FI_FENCE and
    FI_DELIVERY_COMPLETE flags on the second data transfer to force the
    first transfers to meet the FI_DELIVERY_COMPLETE semantics. If the
    second transfer generates a completion at the target, that will
    indicate that the data is visible. Otherwise, a target completion
    for any transfer after the fenced operation will indicate that the
    data is visible.

The above semantics apply for transfers targeting traditional host
memory buffers. However, the behavior may differ when device memory
and/or persistent memory is involved (FI_HMEM and FI_PMEM capability
bits). When heterogenous memory is involved, the concept of memory
domains come into play. Memory domains identify the physical separation
of memory, which may or may not be accessible through the same virtual
address space. See the [`fi_mr`(3)](fi_mr.3.html) man page for further
details on memory domains.

Completion ordering and data visibility are only well-defined for
transfers that target the same memory domain. Applications need to be
aware of ordering and visibility differences when transfers target
different memory domains. Additionally, applications also need to be
concerned with the memory domain that completions themselves are written
and if it differs from the memory domain targeted by a transfer. In some
situations, either the provider or application may need to call device
specific APIs to synchronize or flush device memory caches in order to
achieve the desired data visibility.

When heterogenous memory is in use, the default target completion
semantic for transfers that generate a completion at the target is still
FI_DELIVERY_COMPLETE, however, applications should be aware that there
may be a negative impact on overall performance for providers to meet
this requirement.

For example, a target process may be using a GPU to accelerate
computations. A memory region mapping to memory on the GPU may be
exposed to peers as either an RMA target or posted locally as a receive
buffer. In this case, the application is concerned with two memory
domains -- system and GPU memory. Completions are written to system
memory.

Continuing the example, a peer process sends a tagged message. That
message is matched with the receive buffer located in GPU memory. The
NIC copies the data from the network into the receive buffer and writes
an entry into the completion queue. Note that both memory domains were
accessed as part of this transfer. The message data was directed to the
GPU memory, but the completion went to host memory. Because separate
memory domains may not be synchronized with each other, it is possible
for the host CPU to see and process the completion entry before the
transfer to the GPU memory is visible to either the host GPU or even
software running on the GPU. From the perspective of the *provider*,
visibility of the completion does not imply visibility of data written
to the GPU's memory domain.

The default completion semantic at the target *application* for message
operations is FI_DELIVERY_COMPLETE. An anticipated provider
implementation in this situation is for the provider software running on
the host CPU to intercept the CQ entry, detect that the data landed in
heterogenous memory, and perform the necessary device synchronization or
flush operation before reporting the completion up to the application.
This ensures that the data is visible to CPU *and* GPU software prior to
the application processing the completion.

In addition to the cost of provider software intercepting completions
and checking if a transfer targeted heterogenous memory, device
synchronization itself may impact performance. As a result, applications
can request a lower completion semantic when posting receives. That
indicates to the provider that the application will be responsible for
handling any device specific flush operations that might be needed. See
[`fi_msg`(3)](fi_msg.3.html) FLAGS.

For data transfers that do not generate a completion at the target, such
as RMA or atomics, it is the responsibility of the application to ensure
that all target buffers meet the necessary visibility requirements of
the application. The previously mentioned bulleted methods for notifying
the target that the data is visible may not be sufficient, as the
provider software at the target could lack the context needed to ensure
visibility. This implies that the application may need to call device
synchronization/flush APIs directly.

For example, a peer application could perform several RMA writes that
target GPU memory buffers. If the provider offloads RMA operations into
the NIC, the provider software at the target will be unaware that the
RMA operations have occurred. If the peer sends a message to the target
application that indicates that the RMA operations are done, the
application must ensure that the RMA data is visible to the host CPU or
GPU prior to executing code that accesses the data. The target
completion of having received the sent message is not sufficient, even
if send-after-write ordering is supported.

Most target heterogenous memory completion semantics map to
FI_TRANSMIT_COMPLETE or FI_DELIVERY_COMPLETE. Persistent memory (FI_PMEM
capability), however, is often used with FI_COMMIT_COMPLETE semantics.
Heterogenous completion concepts still apply.

For transfers flagged by the initiator with FI_COMMIT_COMPLETE, a
completion at the target indicates that the results are visible and
durable. For transfers targeting persistent memory, but using a
different completion semantic at the initiator, the visibility at the
target is similar to that described above. Durability is only associated
with transfers marked with FI_COMMIT_COMPLETE.

For transfers targeting persistent memory that request
FI_DELIVERY_COMPLETE, then a completion, at either the initiator or
target, indicates that the data is visible. Visibility at the target can
be conveyed using one of the above describe mechanism -- generating a
target completion, sending a message from the initiator, etc. Similarly,
if the initiator requested FI_TRANSMIT_COMPLETE, then additional steps
are needed to ensure visibility at the target. For example, the transfer
can generate a completion at the target, which would indicate
visibility, but not durability. The initiator can also follow the
transfer with another operation that forces visibility, such as using
FI_FENCE in conjunction with FI_DELIVERY_COMPLETE.

# NOTES

A completion queue must be bound to at least one enabled endpoint before
any operation such as fi_cq_read, fi_cq_readfrom, fi_cq_sread,
fi_cq_sreadfrom etc. can be called on it.

If a completion queue has been overrun, it will be placed into an
'overrun' state. Read operations will continue to return any valid,
non-corrupted completions, if available. After all valid completions
have been retrieved, any attempt to read the CQ will result in it
returning an FI_EOVERRUN error event. Overrun completion queues are
considered fatal and may not be used to report additional completions
once the overrun occurs.

# RETURN VALUES

## fi_cq_open / fi_cq_signal

: Returns 0 on success. On error, returns a negative fabric errno.

## fi_cq_read / fi_cq_readfrom

: On success, returns the number of completions retrieved from the
completion queue. On error, returns a negative fabric errno, with these
two errors explicitly identified: If no completions are available to
read from the CQ, returns -FI_EAGAIN. If the topmost completion is for a
failed transfer (an error entry), returns -FI_EAVAIL.

## fi_cq_sread / fi_cq_sreadfrom

: On success, returns the number of completions retrieved from the
completion queue. On error, returns a negative fabric errno, with these
two errors explicitly identified: If the timeout expires or the calling
thread is signaled and no data is available to be read from the
completion queue, returns -FI_EAGAIN. If the topmost completion is for a
failed transfer (an error entry), returns -FI_EAVAIL.

## fi_cq_readerr

: On success, returns the positive value 1 (number of error entries
returned). On error, returns a negative fabric errno, with this error
explicitly identified: If no error completions are available to read
from the CQ, returns -FI_EAGAIN.

## fi_cq_strerror

: Returns a character string interpretation of the provider specific
error returned with a completion.

Fabric errno values are defined in `rdma/fi_errno.h`.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_eq`(3)](fi_eq.3.html),
[`fi_cntr`(3)](fi_cntr.3.html), [`fi_poll`(3)](fi_poll.3.html)

{% include JB/setup %}

# NAME

fi_domain - Open a fabric access domain

# SYNOPSIS

``` c
#include <rdma/fabric.h>

#include <rdma/fi_domain.h>

int fi_domain(struct fid_fabric *fabric, struct fi_info *info,
    struct fid_domain **domain, void *context);

int fi_domain2(struct fid_fabric *fabric, struct fi_info *info,
    struct fid_domain **domain, uint64_t flags, void *context);

int fi_close(struct fid *domain);

int fi_domain_bind(struct fid_domain *domain, struct fid *eq,
    uint64_t flags);

int fi_open_ops(struct fid *domain, const char *name, uint64_t flags,
    void **ops, void *context);

int fi_set_ops(struct fid *domain, const char *name, uint64_t flags,
    void *ops, void *context);
```

# ARGUMENTS

*fabric*
:   Fabric domain

*info*
:   Fabric information, including domain capabilities and attributes.
    The struct fi_info must have been obtained using either fi_getinfo()
    or fi_dupinfo().

*domain*
:   An opened access domain.

*context*
:   User specified context associated with the domain. This context is
    returned as part of any asynchronous event associated with the
    domain.

*eq*
:   Event queue for asynchronous operations initiated on the domain.

*name*
:   Name associated with an interface.

*ops*
:   Fabric interface operations.

# DESCRIPTION

An access domain typically refers to a physical or virtual NIC or
hardware port; however, a domain may span across multiple hardware
components for fail-over or data striping purposes. A domain defines the
boundary for associating different resources together. Fabric resources
belonging to the same domain may share resources.

## fi_domain

Opens a fabric access domain, also referred to as a resource domain.
Fabric domains are identified by a name. The properties of the opened
domain are specified using the info parameter.

## fi_domain2

Similar to fi_domain, but accepts an extra parameter *flags*. Mainly
used for opening peer domain. See [`fi_peer`(3)](fi_peer.3.html).

## fi_open_ops

fi_open_ops is used to open provider specific interfaces. Provider
interfaces may be used to access low-level resources and operations that
are specific to the opened resource domain. The details of domain
interfaces are outside the scope of this documentation.

## fi_set_ops

fi_set_ops assigns callbacks that a provider should invoke in place of
performing selected tasks. This allows users to modify or control a
provider's default behavior. Conceptually, it allows the user to hook
specific functions used by a provider and replace it with their own.

The operations being modified are identified using a well-known
character string, passed as the name parameter. The format of the ops
parameter is dependent upon the name value. The ops parameter will
reference a structure containing the callbacks and other fields needed
by the provider to invoke the user's functions.

If a provider accepts the override, it will return FI_SUCCESS. If the
override is unknown or not supported, the provider will return
-FI_ENOSYS. Overrides should be set prior to allocating resources on the
domain.

The following fi_set_ops operations and corresponding callback
structures are defined.

**FI_SET_OPS_HMEM_OVERRIDE -- Heterogeneous Memory Overrides**

HMEM override allows users to override HMEM related operations a
provider may perform. Currently, the scope of the HMEM override is to
allow a user to define the memory movement functions a provider should
use when accessing a user buffer. The user-defined memory movement
functions need to account for all the different HMEM iface types a
provider may encounter.

All objects allocated against a domain will inherit this override.

The following is the HMEM override operation name and structure.

``` c
#define FI_SET_OPS_HMEM_OVERRIDE "hmem_override_ops"

struct fi_hmem_override_ops {
    size_t  size;

    ssize_t (*copy_from_hmem_iov)(void *dest, size_t size,
        enum fi_hmem_iface iface, uint64_t device, const struct iovec *hmem_iov,
        size_t hmem_iov_count, uint64_t hmem_iov_offset);

    ssize_t (*copy_to_hmem_iov)(enum fi_hmem_iface iface, uint64_t device,
    const struct iovec *hmem_iov, size_t hmem_iov_count,
        uint64_t hmem_iov_offset, const void *src, size_t size);
};
```

All fields in struct fi_hmem_override_ops must be set (non-null) to a
valid value.

*size*
:   This should be set to the sizeof(struct fi_hmem_override_ops). The
    size field is used for forward and backward compatibility purposes.

*copy_from_hmem_iov*
:   Copy data from the device/hmem to host memory. This function should
    return a negative fi_errno on error, or the number of bytes copied
    on success.

*copy_to_hmem_iov*
:   Copy data from host memory to the device/hmem. This function should
    return a negative fi_errno on error, or the number of bytes copied
    on success.

## fi_domain_bind

Associates an event queue with the domain. An event queue bound to a
domain will be the default EQ associated with asynchronous control
events that occur on the domain or active endpoints allocated on a
domain. This includes CM events. Endpoints may direct their control
events to alternate EQs by binding directly with the EQ.

**Deprecated**: Binding an event queue to a domain with the FI_REG_MR
flag indicates that the provider should perform all memory registration
operations asynchronously, with the completion reported through the
event queue. If an event queue is not bound to the domain with the
FI_REG_MR flag, then memory registration requests complete
synchronously.

## fi_close

The fi_close call is used to release all resources associated with a
domain or interface. All objects associated with the opened domain must
be released prior to calling fi_close, otherwise the call will return
-FI_EBUSY.

# DOMAIN ATTRIBUTES

The `fi_domain_attr` structure defines the set of attributes associated
with a domain.

``` c
struct fi_domain_attr {
    struct fid_domain     *domain;
    char                  *name;
    enum fi_threading     threading;
    enum fi_progress      progress;
    enum fi_resource_mgmt resource_mgmt;
    enum fi_av_type       av_type;
    int                   mr_mode;
    size_t                mr_key_size;
    size_t                cq_data_size;
    size_t                cq_cnt;
    size_t                ep_cnt;
    size_t                tx_ctx_cnt;
    size_t                rx_ctx_cnt;
    size_t                max_ep_tx_ctx;
    size_t                max_ep_rx_ctx;
    size_t                max_ep_stx_ctx;
    size_t                max_ep_srx_ctx;
    size_t                cntr_cnt;
    size_t                mr_iov_limit;
    uint64_t              caps;
    uint64_t              mode;
    uint8_t               *auth_key;
    size_t                auth_key_size;
    size_t                max_err_data;
    size_t                mr_cnt;
    uint32_t              tclass;
    size_t                max_ep_auth_key;
    uint32_t              max_group_id;
};
```

## domain

On input to fi_getinfo, a user may set this to an opened domain instance
to restrict output to the given domain. On output from fi_getinfo, if no
domain was specified, but the user has an opened instance of the named
domain, this will reference the first opened instance. If no instance
has been opened, this field will be NULL.

The domain instance returned by fi_getinfo should only be considered
valid if the application does not close any domain instances from
another thread while fi_getinfo is being processed.

## Name

The name of the access domain.

## Multi-threading Support (threading)

The threading model specifies the level of serialization required of an
application when using the libfabric data transfer interfaces. Control
interfaces are always considered thread safe unless the control progress
model is FI_PROGRESS_CONTROL_UNIFIED. A thread safe control interface
allows multiple threads to progress the control interface, and
(depending on threading model selected) one or more threads to progress
the data interfaces at the same time. Applications which can guarantee
serialization in their access of provider allocated resources and
interfaces enable a provider to eliminate lower-level locks.

*FI_THREAD_COMPLETION*
:   The completion threading model is best suited for multi-threaded
    applications using scalable endpoints which desire lockless
    operation. Applications must serialize access to all objects that
    are associated by a common completion mechanism (for example,
    transmit and receive contexts bound to the same CQ or counter). It
    is recommended that providers which support scalable endpoints
    support this threading model.

Applications wanting to leverage FI_THREAD_COMPLETION should dedicate
transmit contexts, receive contexts, completion queues, and counters to
individual threads.

*FI_THREAD_DOMAIN*
:   The domain threading model is best suited for single-threaded
    applications and multi-threaded applications using standard
    endpoints which desire lockless operation. Applications must
    serialize access to all objects under the same domain. This includes
    endpoints, transmit and receive contexts, completion queues and
    counters, and registered memory regions.

*FI_THREAD_ENDPOINT* (deprecated)
:   The endpoint threading model is similar to FI_THREAD_FID, but with
    the added restriction that serialization is required when accessing
    the same endpoint, even if multiple transmit and receive contexts
    are used.

*FI_THREAD_FID* (deprecated)
:   A fabric descriptor (FID) serialization model requires applications
    to serialize access to individual fabric resources associated with
    data transfer operations and completions. For endpoint access,
    serialization is only required when accessing the same endpoint data
    flow. Multiple threads may initiate transfers on different transmit
    contexts or the same endpoint without serializing, and no
    serialization is required between the submission of data transmit
    requests and data receive operations.

*FI_THREAD_SAFE*
:   A thread safe serialization model allows a multi-threaded
    application to access any allocated resources through any interface
    without restriction. All providers are required to support
    FI_THREAD_SAFE.

*FI_THREAD_UNSPEC*
:   This value indicates that no threading model has been defined. It
    may be used on input hints to the fi_getinfo call. When specified,
    providers will return a threading model that allows for the greatest
    level of parallelism.

## Progress Models (progress)

Progress is the ability of the underlying implementation to complete
processing of an asynchronous request. In many cases, the processing of
an asynchronous request requires the use of the host processor. For
example, a received message may need to be matched with the correct
buffer, or a timed out request may need to be retransmitted. For
performance reasons, it may be undesirable for the provider to allocate
a thread for this purpose, which will compete with the application
threads.

Control progress indicates the method that the provider uses to make
progress on asynchronous control operations. Control operations are
functions which do not directly involve the transfer of application data
between endpoints. They include address vector, memory registration, and
connection management routines.

Data progress indicates the method that the provider uses to make
progress on data transfer operations. This includes message queue, RMA,
tagged messaging, and atomic operations, along with their completion
processing.

The progress field defines the behavior of both control and data
operations. For applications that require compilation portability
between the version 1 and version 2 libfabric series, the progress field
may be referenced as data_progress.

Progress frequently requires action being taken at both the transmitting
and receiving sides of an operation. This is often a requirement for
reliable transfers, as a result of retry and acknowledgement processing.

To balance between performance and ease of use, the following progress
models are defined.

*FI_PROGRESS_AUTO*
:   This progress model indicates that the provider will make forward
    progress on an asynchronous operation without further intervention
    by the application. When FI_PROGRESS_AUTO is provided as output to
    fi_getinfo in the absence of any progress hints, it often indicates
    that the desired functionality is implemented by the provider
    hardware or is a standard service of the operating system.

It is recommended that providers support FI_PROGRESS_AUTO. However, if a
provider does not natively support automatic progress, forcing the use
of FI_PROGRESS_AUTO may result in threads being allocated below the
fabric interfaces.

Note that prior versions of the library required providers to support
FI_PROGRESS_AUTO. However, in some cases progress threads cannot be
blocked when communication is idle, which results in threads spinning in
progress functions. As a result, those providers only supported
FI_PROGRESS_MANUAL.

*FI_PROGRESS_MANUAL*
:   This progress model indicates that the provider requires the use of
    an application thread to complete an asynchronous request. When
    manual progress is set, the provider will attempt to advance an
    asynchronous operation forward when the application attempts to wait
    on or read an event queue, completion queue, or counter where the
    completed operation will be reported. Progress also occurs when the
    application processes a poll or wait set that has been associated
    with the event or completion queue.

Only wait operations defined by the fabric interface will result in an
operation progressing. Operating system or external wait functions, such
as select, poll, or pthread routines, cannot.

Manual progress requirements not only apply to endpoints that initiate
transmit operations, but also to endpoints that may be the target of
such operations. This holds true even if the target endpoint will not
generate completion events for the operations. For example, an endpoint
that acts purely as the target of RMA or atomic operations that uses
manual progress may still need application assistance to process
received operations.

*FI_PROGRESS_CONTROL_UNIFIED*
:   This progress model indicates that the user will synchronize
    progressing the data and control operations themselves (i.e. this
    allows the control interface to NOT be thread safe). It implies
    manual progress, and when combined with
    threading=FI_THREAD_DOMAIN/FI_THREAD_COMPLETION allows Libfabric to
    remove all locking in the critical data progress path.

*FI_PROGRESS_UNSPEC*
:   This value indicates that no progress model has been defined. It may
    be used on input hints to the fi_getinfo call.

## Resource Management (resource_mgmt)

Resource management (RM) is provider and protocol support to protect
against overrunning local and remote resources. This includes local and
remote transmit contexts, receive contexts, completion queues, and
source and target data buffers.

When enabled, applications are given some level of protection against
overrunning provider queues and local and remote data buffers. Such
support may be built directly into the hardware and/or network protocol,
but may also require that checks be enabled in the provider software. By
disabling resource management, an application assumes all responsibility
for preventing queue and buffer overruns, but doing so may allow a
provider to eliminate internal synchronization calls, such as atomic
variables or locks.

It should be noted that even if resource management is disabled, the
provider implementation and protocol may still provide some level of
protection against overruns. However, such protection is not guaranteed.
The following values for resource management are defined.

*FI_RM_DISABLED*
:   The provider is free to select an implementation and protocol that
    does not protect against resource overruns. The application is
    responsible for resource protection.

*FI_RM_ENABLED*
:   Resource management is enabled for this provider domain.

*FI_RM_UNSPEC*
:   This value indicates that no resource management model has been
    defined. It may be used on input hints to the fi_getinfo call.

The behavior of the various resource management options depends on
whether the endpoint is reliable or unreliable, as well as provider and
protocol specific implementation details, as shown in the following
table. The table assumes that all peers enable or disable RM the same.

  ----------------------------------------------------------------------------------------
    Resource    DGRAM EP-no     DGRAM     MSG EP-no RM MSG EP-with   RDM EP-no     RDM
                     RM       EP-with RM                    RM          RM      EP-with RM
  ------------- ------------ ------------ ------------ ------------ ----------- ----------
     Tx Ctx      undefined      EAGAIN     undefined      EAGAIN     undefined    EAGAIN
                   error                     error                     error    

     Rx Ctx      undefined      EAGAIN     undefined      EAGAIN     undefined    EAGAIN
                   error                     error                     error    

      Tx CQ      undefined      EAGAIN     undefined      EAGAIN     undefined    EAGAIN
                   error                     error                     error    

      Rx CQ      undefined      EAGAIN     undefined      EAGAIN     undefined    EAGAIN
                   error                     error                     error    

    Target EP     dropped      dropped      transmit     retried     transmit    retried
                                             error                     error    

  No Rx Buffer    dropped      dropped      transmit     retried     transmit    retried
                                             error                     error    

     Rx Buf     truncate or  truncate or  truncate or  truncate or  truncate or  truncate
     Overrun        drop         drop        error        error        error     or error

  Unmatched RMA     not          not        transmit     transmit    transmit    transmit
                 applicable   applicable     error        error        error      error

   RMA Overrun      not          not        transmit     transmit    transmit    transmit
                 applicable   applicable     error        error        error      error

   Unreachable    dropped      dropped        not          not       transmit    transmit
       EP                                  applicable   applicable     error      error
  ----------------------------------------------------------------------------------------

The resource column indicates the resource being accessed by a data
transfer operation.

*Tx Ctx / Rx Ctx*
:   Refers to the transmit/receive contexts when a data transfer
    operation is submitted. When RM is enabled, attempting to submit a
    request will fail if the context is full. If RM is disabled, an
    undefined error (provider specific) will occur. Such errors should
    be considered fatal to the context, and applications must take steps
    to avoid queue overruns.

*Tx CQ / Rx CQ*
:   Refers to the completion queue associated with the Tx or Rx context
    when a local operation completes. When RM is disabled, applications
    must take care to ensure that completion queues do not get overrun.
    When an overrun occurs, an undefined, but fatal, error will occur
    affecting all endpoints associated with the CQ. Overruns can be
    avoided by sizing the CQs appropriately or by deferring the posting
    of a data transfer operation unless CQ space is available to store
    its completion. When RM is enabled, providers may use different
    mechanisms to prevent CQ overruns. This includes failing (returning
    -FI_EAGAIN) the posting of operations that could result in CQ
    overruns, or internally retrying requests (which will be hidden from
    the application). See notes at the end of this section regarding CQ
    resource management restrictions.

*Target EP / No Rx Buffer*
:   Target EP refers to resources associated with the endpoint that is
    the target of a transmit operation. This includes the target
    endpoint's receive queue, posted receive buffers (no Rx buffers),
    the receive side completion queue, and other related packet
    processing queues. The defined behavior is that seen by the
    initiator of a request. For FI_EP_DGRAM endpoints, if the target EP
    queues are unable to accept incoming messages, received messages
    will be dropped. For reliable endpoints, if RM is disabled, the
    transmit operation will complete in error. A provider may choose to
    return an error completion with the error code FI_ENORX for that
    transmit operation so that it can be retried. If RM is enabled, the
    provider will internally retry the operation.

*Rx Buffer Overrun*
:   This refers to buffers posted to receive incoming tagged or untagged
    messages, with the behavior defined from the viewpoint of the
    sender. The behavior for handling received messages that are larger
    than the buffers provided by the application is provider specific.
    Providers may either truncate the message and report a successful
    completion, or fail the operation. For datagram endpoints, failed
    sends will result in the message being dropped. For reliable
    endpoints, send operations may complete successfully, yet be
    truncated at the receive side. This can occur when the target side
    buffers received data until an application buffer is made available.
    The completion status may also be dependent upon the completion
    model selected byt the application (e.g. FI_DELIVERY_COMPLETE versus
    FI_TRANSMIT_COMPLETE).

*Unmatched RMA / RMA Overrun*
:   Unmatched RMA and RMA overruns deal with the processing of RMA and
    atomic operations. Unlike send operations, RMA operations that
    attempt to access a memory address that is either not registered for
    such operations, or attempt to access outside of the target memory
    region will fail, resulting in a transmit error.

*Unreachable EP*
:   Unreachable endpoint is a connectionless specific scenario where
    transmit operations are issued to unreachable target endpoints. Such
    scenarios include no-route-to-host or down target NIC. For
    FI_EP_DGRAM endpoints, transmit operations targeting an unreachable
    endpoint will have operation dropped. For FI_EP_RDM, target
    operations targeting an unreachable endpoint will result in a
    transmit error.

When a resource management error occurs on an a connected endpoint, the
endpoint will transition into a disabled state and the connection torn
down. A disabled endpoint will drop any queued or inflight operations.

The behavior of resource management errors on connectionless endpoints
depends on the type of error. If RM is disabled and one of the following
errors occur, the endpoint will be disabled: Tx Ctx, Rx Ctx, Tx CQ, or
Rx CQ. For other errors (Target EP, No Rx Buffer, etc.), the operation
may fail, but the endpoint will remain enabled. A disabled endpoint will
drop or fail any queued or inflight operations. In addition, a disabled
endpoint must be re-enabled before it will accept new data transfer
operations.

There is one notable restriction on the protections offered by resource
management. This occurs when resource management is enabled on an
endpoint that has been bound to completion queue(s) using the
FI_SELECTIVE_COMPLETION flag. Operations posted to such an endpoint may
specify that a successful completion should not generate a entry on the
corresponding completion queue. (I.e. the operation leaves the
FI_COMPLETION flag unset). In such situations, the provider is not
required to reserve an entry in the completion queue to handle the case
where the operation fails and does generate a CQ entry, which would
effectively require tracking the operation to completion. Applications
concerned with avoiding CQ overruns in the occurrence of errors must
ensure that there is sufficient space in the CQ to report failed
operations. This can typically be achieved by sizing the CQ to at least
the same size as the endpoint queue(s) that are bound to it.

## AV Type (av_type)

Specifies the type of address vectors that are usable with this domain.
For additional details on AV type, see [`fi_av`(3)](fi_av.3.html). The
following values may be specified.

*FI_AV_MAP* (deprecated)
:   Only address vectors of type AV map are requested or supported.

*FI_AV_TABLE*
:   Only address vectors of type AV index are requested or supported.

*FI_AV_UNSPEC*
:   Any address vector format is requested and supported.

Address vectors are only used by connectionless endpoints. Applications
that require the use of a specific type of address vector should set the
domain attribute av_type to the necessary value when calling fi_getinfo.
The value FI_AV_UNSPEC may be used to indicate that the provider can
support either address vector format. In this case, a provider may
return FI_AV_UNSPEC to indicate that either format is supportable, or
may return another AV type to indicate the optimal AV type supported by
this domain.

## Memory Registration Mode (mr_mode)

Defines memory registration specific mode bits used with this domain.
Full details on MR mode options are available in
[`fi_mr`(3)](fi_mr.3.html). The following values may be specified.

*FI_MR_ALLOCATED*
:   Indicates that memory registration occurs on allocated data buffers,
    and physical pages must back all virtual addresses being registered.

*FI_MR_COLLECTIVE*
:   Requires data buffers passed to collective operations be explicitly
    registered for collective operations using the FI_COLLECTIVE flag.

*FI_MR_ENDPOINT*
:   Memory registration occurs at the endpoint level, rather than
    domain.

*FI_MR_LOCAL*
:   The provider is optimized around having applications register memory
    for locally accessed data buffers. Data buffers used in send and
    receive operations and as the source buffer for RMA and atomic
    operations must be registered by the application for access domains
    opened with this capability.

*FI_MR_MMU_NOTIFY*
:   Indicates that the application is responsible for notifying the
    provider when the page tables referencing a registered memory region
    may have been updated.

*FI_MR_PROV_KEY*
:   Memory registration keys are selected and returned by the provider.

*FI_MR_RAW*
:   The provider requires additional setup as part of their memory
    registration process. This mode is required by providers that use a
    memory key that is larger than 64-bits.

*FI_MR_RMA_EVENT*
:   Indicates that the memory regions associated with completion
    counters must be explicitly enabled after being bound to any
    counter.

*FI_MR_UNSPEC* (deprecated)
:   Defined for compatibility -- library versions 1.4 and earlier.
    Setting mr_mode to 0 indicates that FI_MR_BASIC or FI_MR_SCALABLE
    are requested and supported.

*FI_MR_VIRT_ADDR*
:   Registered memory regions are referenced by peers using the virtual
    address of the registered memory region, rather than a 0-based
    offset.

*FI_MR_BASIC* (deprecated)
:   Defined for compatibility -- library versions 1.4 and earlier. Only
    basic memory registration operations are requested or supported.
    This mode is equivalent to the FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, and
    FI_MR_PROV_KEY flags being set in later library versions. This flag
    may not be used in conjunction with other mr_mode bits.

*FI_MR_SCALABLE* (deprecated)
:   Defined for compatibility -- library versions 1.4 and earlier. Only
    scalable memory registration operations are requested or supported.
    Scalable registration uses offset based addressing, with application
    selectable memory keys. For library versions 1.5 and later, this is
    the default if no mr_mode bits are set. This flag may not be used in
    conjunction with other mr_mode bits.

Buffers used in data transfer operations may require notifying the
provider of their use before a data transfer can occur. The mr_mode
field indicates the type of memory registration that is required, and
when registration is necessary. Applications that require the use of a
specific registration mode should set the domain attribute mr_mode to
the necessary value when calling fi_getinfo. The value FI_MR_UNSPEC may
be used to indicate support for any registration mode.

## MR Key Size (mr_key_size)

Size of the memory region remote access key, in bytes. Applications that
request their own MR key must select a value within the range specified
by this value. Key sizes larger than 8 bytes require using the
FI_RAW_KEY mode bit.

## CQ Data Size (cq_data_size)

Applications may include a small message with a data transfer that is
placed directly into a remote completion queue as part of a completion
event. This is referred to as remote CQ data (sometimes referred to as
immediate data). This field indicates the number of bytes that the
provider supports for remote CQ data. If supported (non-zero value is
returned), the minimum size of remote CQ data must be at least 4-bytes.

## Completion Queue Count (cq_cnt)

The optimal number of completion queues supported by the domain,
relative to any specified or default CQ attributes. The cq_cnt value may
be a fixed value of the maximum number of CQs supported by the
underlying hardware, or may be a dynamic value, based on the default
attributes of an allocated CQ, such as the CQ size and data format.

## Endpoint Count (ep_cnt)

The total number of endpoints supported by the domain, relative to any
specified or default endpoint attributes. The ep_cnt value may be a
fixed value of the maximum number of endpoints supported by the
underlying hardware, or may be a dynamic value, based on the default
attributes of an allocated endpoint, such as the endpoint capabilities
and size. The endpoint count is the number of addressable endpoints
supported by the provider. Providers return capability limits based on
configured hardware maximum capabilities. Providers cannot predict all
possible system limitations without posteriori knowledge acquired during
runtime that will further limit these hardware maximums
(e.g. application memory consumption, FD usage, etc.).

## Transmit Context Count (tx_ctx_cnt)

The number of outbound command queues optimally supported by the
provider. For a low-level provider, this represents the number of
command queues to the hardware and/or the number of parallel transmit
engines effectively supported by the hardware and caches. Applications
which allocate more transmit contexts than this value will end up
sharing underlying resources. By default, there is a single transmit
context associated with each endpoint, but in an advanced usage model,
an endpoint may be configured with multiple transmit contexts.

## Receive Context Count (rx_ctx_cnt)

The number of inbound processing queues optimally supported by the
provider. For a low-level provider, this represents the number hardware
queues that can be effectively utilized for processing incoming packets.
Applications which allocate more receive contexts than this value will
end up sharing underlying resources. By default, a single receive
context is associated with each endpoint, but in an advanced usage
model, an endpoint may be configured with multiple receive contexts.

## Maximum Endpoint Transmit Context (max_ep_tx_ctx)

The maximum number of transmit contexts that may be associated with an
endpoint.

## Maximum Endpoint Receive Context (max_ep_rx_ctx)

The maximum number of receive contexts that may be associated with an
endpoint.

## Maximum Sharing of Transmit Context (max_ep_stx_ctx)

The maximum number of endpoints that may be associated with a shared
transmit context.

## Maximum Sharing of Receive Context (max_ep_srx_ctx)

The maximum number of endpoints that may be associated with a shared
receive context.

## Counter Count (cntr_cnt)

The optimal number of completion counters supported by the domain. The
cq_cnt value may be a fixed value of the maximum number of counters
supported by the underlying hardware, or may be a dynamic value, based
on the default attributes of the domain.

## MR IOV Limit (mr_iov_limit)

This is the maximum number of IO vectors (scatter-gather elements) that
a single memory registration operation may reference.

## Capabilities (caps)

Domain level capabilities. Domain capabilities indicate domain level
features that are supported by the provider.

The following are support primary capabilities: *FI_DIRECTED_RECV* :
When the domain is configured with FI_DIRECTED_RECV and FI_AV_AUTH_KEY,
memory regions can be limited to specific authorization keys.

*FI_AV_USER_ID*
:   Indicates that the domain supports the ability to open address
    vectors with the FI_AV_USER_ID flag. If this domain capability is
    not set, address vectors cannot be opened with FI_AV_USER_ID. Note
    that FI_AV_USER_ID can still be supported through the AV insert
    calls without this domain capability set. See
    [`fi_av`(3)](fi_av.3.html).

*FI_PEER*
:   Specifies that the domain must support importing resources to be
    used in the the peer API flow. The domain must support importing
    owner_ops when opening a CQ, counter, and shared receive queue.

The following are supported secondary capabilities:

*FI_LOCAL_COMM*
:   At a conceptual level, this field indicates that the underlying
    device supports loopback communication. More specifically, this
    field indicates that an endpoint may communicate with other
    endpoints that are allocated from the same underlying named domain.
    If this field is not set, an application may need to use an
    alternate domain or mechanism (e.g. shared memory) to communicate
    with peers that execute on the same node.

*FI_REMOTE_COMM*
:   This field indicates that the underlying provider supports
    communication with nodes that are reachable over the network. If
    this field is not set, then the provider only supports communication
    between processes that execute on the same node -- a shared memory
    provider, for example.

*FI_SHARED_AV*
:   Indicates that the domain supports the ability to share address
    vectors among multiple processes using the named address vector
    feature.

See [`fi_getinfo`(3)](fi_getinfo.3.html) for a discussion on primary
versus secondary capabilities.

## Default authorization key (auth_key)

The default authorization key to associate with endpoint and memory
registrations created within the domain. This field is ignored unless
the fabric is opened with API version 1.5 or greater.

If domain auth_key_size is set to the value FI_AV_AUTH_KEY, auth_key
must be NULL.

## Default authorization key length (auth_key_size)

The length in bytes of the default authorization key for the domain. If
set to 0, then no authorization key will be associated with endpoints
and memory registrations created within the domain unless specified in
the endpoint or memory registration attributes. This field is ignored
unless the fabric is opened with API version 1.5 or greater.

If the size is set to the value FI_AV_AUTH_KEY, all endpoints and memory
regions will be configured to use authorization keys associated with the
AV. Providers which support authorization keys and connectionless
endpoint must support this option.

## Max Error Data Size (max_err_data)

The maximum amount of error data, in bytes, that may be returned as part
of a completion or event queue error. This value corresponds to the
err_data_size field in struct fi_cq_err_entry and struct
fi_eq_err_entry.

## Memory Regions Count (mr_cnt)

The optimal number of memory regions supported by the domain, or
endpoint if the mr_mode FI_MR_ENDPOINT bit has been set. The mr_cnt
value may be a fixed value of the maximum number of MRs supported by the
underlying hardware, or may be a dynamic value, based on the default
attributes of the domain, such as the supported memory registration
modes. Applications can set the mr_cnt on input to fi_getinfo, in order
to indicate their memory registration requirements. Doing so may allow
the provider to optimize any memory registration cache or lookup tables.

## Traffic Class (tclass)

This specifies the default traffic class that will be associated any
endpoints created within the domain. See
[`fi_endpoint`(3)](fi_endpoint.3.html) for additional information.

## Max Authorization Keys per Endpoint (max_ep_auth_key)

The maximum number of authorization keys which can be supported per
connectionless endpoint.

## Maximum Peer Group Id (max_group_id)

The maximum value that a peer group may be assigned, inclusive. Valid
peer group id's must be between 0 and max_group_id. See
[`fi_av`(3)](fi_av.3.html) for additional information on peer groups and
their use. Users may request support for peer groups by setting this to
a non-zero value. Providers that cannot meet the requested max_group_id
will fail fi_getinfo(). On output, providers may return a value higher
than that requested by the application.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

# NOTES

Users should call fi_close to release all resources allocated to the
fabric domain.

The following fabric resources are associated with domains: active
endpoints, memory regions, completion event queues, and address vectors.

Domain attributes reflect the limitations and capabilities of the
underlying hardware and/or software provider. They do not reflect system
limitations, such as the number of physical pages that an application
may pin or number of file descriptors that the application may open. As
a result, the reported maximums may not be achievable, even on a lightly
loaded systems, without an administrator configuring system resources
appropriately for the installed provider(s).

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html), [`fi_av`(3)](fi_av.3.html),
[`fi_eq`(3)](fi_eq.3.html), [`fi_mr`(3)](fi_mr.3.html)
[`fi_peer`(3)](fi_peer.3.html)

{% include JB/setup %}

# NAME

fi_endpoint - Fabric endpoint operations

fi_endpoint / fi_endpoint2 / fi_scalable_ep / fi_passive_ep / fi_close
:   Allocate or close an endpoint.

fi_ep_bind
:   Associate an endpoint with hardware resources, such as event queues,
    completion queues, counters, address vectors, or shared
    transmit/receive contexts.

fi_scalable_ep_bind
:   Associate a scalable endpoint with an address vector

fi_pep_bind
:   Associate a passive endpoint with an event queue

fi_enable
:   Transitions an active endpoint into an enabled state.

fi_cancel
:   Cancel a pending asynchronous data transfer

fi_ep_alias
:   Create an alias to the endpoint

fi_control
:   Control endpoint operation.

fi_getopt / fi_setopt
:   Get or set endpoint options.

fi_rx_context / fi_tx_context / fi_srx_context / fi_stx_context
:   Open a transmit or receive context.

fi_tc_dscp_set / fi_tc_dscp_get
:   Convert between a DSCP value and a network traffic class

fi_rx_size_left / fi_tx_size_left (DEPRECATED)
:   Query the lower bound on how many RX/TX operations may be posted
    without an operation returning -FI_EAGAIN. This functions have been
    deprecated and will be removed in a future version of the library.

# SYNOPSIS

``` c
#include <rdma/fabric.h>

#include <rdma/fi_endpoint.h>

int fi_endpoint(struct fid_domain *domain, struct fi_info *info,
    struct fid_ep **ep, void *context);

int fi_endpoint2(struct fid_domain *domain, struct fi_info *info,
    struct fid_ep **ep, uint64_t flags, void *context);

int fi_scalable_ep(struct fid_domain *domain, struct fi_info *info,
    struct fid_ep **sep, void *context);

int fi_passive_ep(struct fi_fabric *fabric, struct fi_info *info,
    struct fid_pep **pep, void *context);

int fi_tx_context(struct fid_ep *sep, int index,
    struct fi_tx_attr *attr, struct fid_ep **tx_ep,
    void *context);

int fi_rx_context(struct fid_ep *sep, int index,
    struct fi_rx_attr *attr, struct fid_ep **rx_ep,
    void *context);

int fi_stx_context(struct fid_domain *domain,
    struct fi_tx_attr *attr, struct fid_stx **stx,
    void *context);

int fi_srx_context(struct fid_domain *domain,
    struct fi_rx_attr *attr, struct fid_ep **rx_ep,
    void *context);

int fi_close(struct fid *ep);

int fi_ep_bind(struct fid_ep *ep, struct fid *fid, uint64_t flags);

int fi_scalable_ep_bind(struct fid_ep *sep, struct fid *fid, uint64_t flags);

int fi_pep_bind(struct fid_pep *pep, struct fid *fid, uint64_t flags);

int fi_enable(struct fid_ep *ep);

int fi_cancel(struct fid_ep *ep, void *context);

int fi_ep_alias(struct fid_ep *ep, struct fid_ep **alias_ep, uint64_t flags);

int fi_control(struct fid *ep, int command, void *arg);

int fi_getopt(struct fid *ep, int level, int optname,
    void *optval, size_t *optlen);

int fi_setopt(struct fid *ep, int level, int optname,
    const void *optval, size_t optlen);

uint32_t fi_tc_dscp_set(uint8_t dscp);

uint8_t fi_tc_dscp_get(uint32_t tclass);

DEPRECATED ssize_t fi_rx_size_left(struct fid_ep *ep);

DEPRECATED ssize_t fi_tx_size_left(struct fid_ep *ep);
```

# ARGUMENTS

*fid*
:   On creation, specifies a fabric or access domain. On bind,
    identifies the event queue, completion queue, counter, or address
    vector to bind to the endpoint. In other cases, it's a fabric
    identifier of an associated resource.

*info*
:   Details about the fabric interface endpoint to be opened. The struct
    fi_info must have been obtained using either fi_getinfo() or
    fi_dupinfo().

*ep*
:   A fabric endpoint.

*sep*
:   A scalable fabric endpoint.

*pep*
:   A passive fabric endpoint.

*context*
:   Context associated with the endpoint or asynchronous operation.

*index*
:   Index to retrieve a specific transmit/receive context.

*attr*
:   Transmit or receive context attributes.

*flags*
:   Additional flags to apply to the operation.

*command*
:   Command of control operation to perform on endpoint.

*arg*
:   Optional control argument.

*level*
:   Protocol level at which the desired option resides.

*optname*
:   The protocol option to read or set.

*optval*
:   The option value that was read or to set.

*optlen*
:   The size of the optval buffer.

# DESCRIPTION

Endpoints are transport level communication portals. There are two types
of endpoints: active and passive. Passive endpoints belong to a fabric
domain and are most often used to listen for incoming connection
requests. However, a passive endpoint may be used to reserve a fabric
address that can be granted to an active endpoint. Active endpoints
belong to access domains and can perform data transfers.

Active endpoints may be connection-oriented or connectionless, and may
provide data reliability. The data transfer interfaces -- messages
(fi_msg), tagged messages (fi_tagged), RMA (fi_rma), and atomics
(fi_atomic) -- are associated with active endpoints. In basic
configurations, an active endpoint has transmit and receive queues. In
general, operations that generate traffic on the fabric are posted to
the transmit queue. This includes all RMA and atomic operations, along
with sent messages and sent tagged messages. Operations that post
buffers for receiving incoming data are submitted to the receive queue.

Active endpoints are created in the disabled state. They must transition
into an enabled state before accepting data transfer operations,
including posting of receive buffers. The fi_enable call is used to
transition an active endpoint into an enabled state. The fi_connect and
fi_accept calls will also transition an endpoint into the enabled state,
if it is not already active.

In order to transition an endpoint into an enabled state, it must be
bound to one or more fabric resources. An endpoint that will generate
asynchronous completions, either through data transfer operations or
communication establishment events, must be bound to the appropriate
completion queues or event queues, respectively, before being enabled.
Additionally, endpoints that use manual progress must be associated with
relevant completion queues or event queues in order to drive progress.
For endpoints that are only used as the target of RMA or atomic
operations, this means binding the endpoint to a completion queue
associated with receive processing. Connectionless endpoints must be
bound to an address vector.

Once an endpoint has been activated, it may be associated with an
address vector. Receive buffers may be posted to it and calls may be
made to connection establishment routines. Connectionless endpoints may
also perform data transfers.

The behavior of an endpoint may be adjusted by setting its control data
and protocol options. This allows the underlying provider to redirect
function calls to implementations optimized to meet the desired
application behavior.

If an endpoint experiences a critical error, it will transition back
into a disabled state. Critical errors are reported through the event
queue associated with the EP. In certain cases, a disabled endpoint may
be re-enabled. The ability to transition back into an enabled state is
provider specific and depends on the type of error that the endpoint
experienced. When an endpoint is disabled as a result of a critical
error, all pending operations are discarded.

## fi_endpoint / fi_passive_ep / fi_scalable_ep

fi_endpoint allocates a new active endpoint. fi_passive_ep allocates a
new passive endpoint. fi_scalable_ep allocates a scalable endpoint. The
properties and behavior of the endpoint are defined based on the
provided struct fi_info. See fi_getinfo for additional details on
fi_info. fi_info flags that control the operation of an endpoint are
defined below. See section SCALABLE ENDPOINTS.

If an active endpoint is allocated in order to accept a connection
request, the fi_info parameter must be the same as the fi_info structure
provided with the connection request (FI_CONNREQ) event.

An active endpoint may acquire the properties of a passive endpoint by
setting the fi_info handle field to the passive endpoint fabric
descriptor. This is useful for applications that need to reserve the
fabric address of an endpoint prior to knowing if the endpoint will be
used on the active or passive side of a connection. For example, this
feature is useful for simulating socket semantics. Once an active
endpoint acquires the properties of a passive endpoint, the passive
endpoint is no longer bound to any fabric resources and must no longer
be used. The user is expected to close the passive endpoint after
opening the active endpoint in order to free up any lingering resources
that had been used.

## fi_endpoint2

Similar to fi_endpoint, buf accepts an extra parameter *flags*. Mainly
used for opening endpoints that use peer transfer feature. See
[`fi_peer`(3)](fi_peer.3.html)

## fi_close

Closes an endpoint and release all resources associated with it.

When closing a scalable endpoint, there must be no opened transmit
contexts, or receive contexts associated with the scalable endpoint. If
resources are still associated with the scalable endpoint when
attempting to close, the call will return -FI_EBUSY.

Outstanding operations posted to the endpoint when fi_close is called
will be discarded. Discarded operations will silently be dropped, with
no completions reported. Memory buffers that the user associated with
the endpoint by calling libfabric's transmission interfaces must not be
released, deregistered, or reused until either a completion is generated
or fi_close on the respective endpoint returns. Additionally, a provider
may discard previously completed operations from the associated
completion queue(s). The behavior to discard completed operations is
provider specific.

## fi_ep_bind

fi_ep_bind is used to associate an endpoint with other allocated
resources, such as completion queues, counters, address vectors, event
queues, shared contexts, and memory regions. The type of objects that
must be bound with an endpoint depend on the endpoint type and its
configuration.

Passive endpoints must be bound with an EQ that supports connection
management events. Connectionless endpoints must be bound to a single
address vector. If an endpoint is using a shared transmit and/or receive
context, the shared contexts must be bound to the endpoint. CQs,
counters, AV, and shared contexts must be bound to endpoints before they
are enabled either explicitly or implicitly.

An endpoint must be bound with CQs capable of reporting completions for
any asynchronous operation initiated on the endpoint. For example, if
the endpoint supports any outbound transfers (sends, RMA, atomics,
etc.), then it must be bound to a completion queue that can report
transmit completions. This is true even if the endpoint is configured to
suppress successful completions, in order that operations that complete
in error may be reported to the user.

An active endpoint may direct asynchronous completions to different CQs,
based on the type of operation. This is specified using fi_ep_bind
flags. The following flags may be OR'ed together when binding an
endpoint to a completion domain CQ.

*FI_RECV*
:   Directs the notification of inbound data transfers to the specified
    completion queue. This includes received messages. This binding
    automatically includes FI_REMOTE_WRITE, if applicable to the
    endpoint.

*FI_SELECTIVE_COMPLETION*
:   By default, data transfer operations write CQ completion entries
    into the associated completion queue after they have successfully
    completed. Applications can use this bind flag to selectively enable
    when completions are generated. If FI_SELECTIVE_COMPLETION is
    specified, data transfer operations will not generate CQ entries for
    *successful* completions unless FI_COMPLETION is set as an
    operational flag for the given operation. Operations that fail
    asynchronously will still generate completions, even if a completion
    is not requested. FI_SELECTIVE_COMPLETION must be OR'ed with
    FI_TRANSMIT and/or FI_RECV flags.

When FI_SELECTIVE_COMPLETION is set, the user must determine when a
request that does NOT have FI_COMPLETION set has completed indirectly,
usually based on the completion of a subsequent operation or by using
completion counters. Use of this flag may improve performance by
allowing the provider to avoid writing a CQ completion entry for every
operation.

See Notes section below for additional information on how this flag
interacts with the FI_CONTEXT and FI_CONTEXT2 mode bits.

*FI_TRANSMIT*
:   Directs the completion of outbound data transfer requests to the
    specified completion queue. This includes send message, RMA, and
    atomic operations.

An endpoint may optionally be bound to a completion counter. Associating
an endpoint with a counter is in addition to binding the EP with a CQ.
When binding an endpoint to a counter, the following flags may be
specified.

*FI_READ*
:   Increments the specified counter whenever an RMA read, atomic fetch,
    or atomic compare operation initiated from the endpoint has
    completed successfully or in error.

*FI_RECV*
:   Increments the specified counter whenever a message is received over
    the endpoint. Received messages include both tagged and normal
    message operations.

*FI_REMOTE_READ*
:   Increments the specified counter whenever an RMA read, atomic fetch,
    or atomic compare operation is initiated from a remote endpoint that
    targets the given endpoint. Use of this flag requires that the
    endpoint be created using FI_RMA_EVENT.

*FI_REMOTE_WRITE*
:   Increments the specified counter whenever an RMA write or base
    atomic operation is initiated from a remote endpoint that targets
    the given endpoint. Use of this flag requires that the endpoint be
    created using FI_RMA_EVENT.

*FI_SEND*
:   Increments the specified counter whenever a message transfer
    initiated over the endpoint has completed successfully or in error.
    Sent messages include both tagged and normal message operations.

*FI_WRITE*
:   Increments the specified counter whenever an RMA write or base
    atomic operation initiated from the endpoint has completed
    successfully or in error.

An endpoint may only be bound to a single CQ or counter for a given type
of operation. For example, a EP may not bind to two counters both using
FI_WRITE. Furthermore, providers may limit CQ and counter bindings to
endpoints of the same endpoint type (DGRAM, MSG, RDM, etc.).

## fi_scalable_ep_bind

fi_scalable_ep_bind is used to associate a scalable endpoint with an
address vector. See section on SCALABLE ENDPOINTS. A scalable endpoint
has a single transport level address and can support multiple transmit
and receive contexts. The transmit and receive contexts share the
transport-level address. Address vectors that are bound to scalable
endpoints are implicitly bound to any transmit or receive contexts
created using the scalable endpoint.

## fi_enable

This call transitions the endpoint into an enabled state. An endpoint
must be enabled before it may be used to perform data transfers.
Enabling an endpoint typically results in hardware resources being
assigned to it. Endpoints making use of completion queues, counters,
event queues, and/or address vectors must be bound to them before being
enabled.

Calling connect or accept on an endpoint will implicitly enable an
endpoint if it has not already been enabled.

fi_enable may also be used to re-enable an endpoint that has been
disabled as a result of experiencing a critical error. Applications
should check the return value from fi_enable to see if a disabled
endpoint has successfully be re-enabled.

## fi_cancel

fi_cancel attempts to cancel an outstanding asynchronous operation.
Canceling an operation causes the fabric provider to search for the
operation and, if it is still pending, complete it as having been
canceled. An error queue entry will be available in the associated error
queue with error code FI_ECANCELED. On the other hand, if the operation
completed before the call to fi_cancel, then the completion status of
that operation will be available in the associated completion queue. No
specific entry related to fi_cancel itself will be posted.

Cancel uses the context parameter associated with an operation to
identify the request to cancel. Operations posted without a valid
context parameter -- either no context parameter is specified or the
context value was ignored by the provider -- cannot be canceled. If
multiple outstanding operations match the context parameter, only one
will be canceled. In this case, the operation which is canceled is
provider specific. The cancel operation is asynchronous, but will
complete within a bounded period of time.

## fi_ep_alias

This call creates an alias to the specified endpoint. Conceptually, an
endpoint alias provides an alternate software path from the application
to the underlying provider hardware. An alias EP differs from its parent
endpoint only by its default data transfer flags. For example, an alias
EP may be configured to use a different completion mode. By default, an
alias EP inherits the same data transfer flags as the parent endpoint.
An application can use fi_control to modify the alias EP operational
flags.

When allocating an alias, an application may configure either the
transmit or receive operational flags. This avoids needing a separate
call to fi_control to set those flags. The flags passed to fi_ep_alias
must include FI_TRANSMIT or FI_RECV (not both) with other operational
flags OR'ed in. This will override the transmit or receive flags,
respectively, for operations posted through the alias endpoint. All
allocated aliases must be closed for the underlying endpoint to be
released.

## fi_control

The control operation is used to adjust the default behavior of an
endpoint. It allows the underlying provider to redirect function calls
to implementations optimized to meet the desired application behavior.
As a result, calls to fi_ep_control must be serialized against all other
calls to an endpoint.

The base operation of an endpoint is selected during creation using
struct fi_info. The following control commands and arguments may be
assigned to an endpoint.

*FI_BACKLOG - int \*value*
:   This option only applies to passive endpoints. It is used to set the
    connection request backlog for listening endpoints.

*FI_GETOPSFLAG -- uint64_t \*flags*
:   Used to retrieve the current value of flags associated with the data
    transfer operations initiated on the endpoint. The control argument
    must include FI_TRANSMIT or FI_RECV (not both) flags to indicate the
    type of data transfer flags to be returned. See below for a list of
    control flags.

*FI_GETWAIT -- void \*\**
:   This command allows the user to retrieve the file descriptor
    associated with a socket endpoint. The fi_control arg parameter
    should be an address where a pointer to the returned file descriptor
    will be written. See fi_eq.3 for addition details using fi_control
    with FI_GETWAIT. The file descriptor may be used for notification
    that the endpoint is ready to send or receive data.

*FI_SETOPSFLAG -- uint64_t \*flags*
:   Used to change the data transfer operation flags associated with an
    endpoint. The control argument must include FI_TRANSMIT or FI_RECV
    (not both) to indicate the type of data transfer that the flags
    should apply to, with other flags OR'ed in. The given flags will
    override the previous transmit and receive attributes that were set
    when the endpoint was created. Valid control flags are defined
    below.

## fi_getopt / fi_setopt

Endpoint protocol operations may be retrieved using fi_getopt or set
using fi_setopt. Applications specify the level that a desired option
exists, identify the option, and provide input/output buffers to get or
set the option. fi_setopt provides an application a way to adjust
low-level protocol and implementation specific details of an endpoint,
and must be called before the endpoint is enabled (fi_enable).

The following option levels and option names and parameters are defined.

*FI_OPT_ENDPOINT*

-   

    *FI_OPT_CM_DATA_SIZE - size_t*
    :   Defines the size of available space in CM messages for
        user-defined data. This value limits the amount of data that
        applications can exchange between peer endpoints using the
        fi_connect, fi_accept, and fi_reject operations. The size
        returned is dependent upon the properties of the endpoint,
        except in the case of passive endpoints, in which the size
        reflects the maximum size of the data that may be present as
        part of a connection request event. This option is read only.

-   

    *FI_OPT_MIN_MULTI_RECV - size_t*
    :   Defines the minimum receive buffer space available below which
        the receive buffer is released by the provider (see
        FI_MULTI_RECV). Modifying this value is only guaranteed to set
        the minimum buffer space needed on receives posted after the
        value has been changed. It is recommended that applications that
        want to override the default MIN_MULTI_RECV value set this
        option before enabling the corresponding endpoint.

-   

    *FI_OPT_FI_HMEM_P2P - int*
    :   Defines how the provider should handle peer to peer FI_HMEM
        transfers for this endpoint. By default, the provider will chose
        whether to use peer to peer support based on the type of
        transfer (FI_HMEM_P2P_ENABLED). Valid values defined in
        fi_endpoint.h are:
        -   FI_HMEM_P2P_ENABLED: Peer to peer support may be used by the
            provider to handle FI_HMEM transfers, and which transfers
            are initiated using peer to peer is subject to the provider
            implementation.
        -   FI_HMEM_P2P_REQUIRED: Peer to peer support must be used for
            transfers, transfers that cannot be performed using p2p will
            be reported as failing.
        -   FI_HMEM_P2P_PREFERRED: Peer to peer support should be used
            by the provider for all transfers if available, but the
            provider may choose to copy the data to initiate the
            transfer if peer to peer support is unavailable.
        -   FI_HMEM_P2P_DISABLED: Peer to peer support should not be
            used.

    fi_setopt() will return -FI_EOPNOTSUPP if the mode requested cannot
    be supported by the provider. The FI_HMEM_DISABLE_P2P environment
    variable discussed in [`fi_mr`(3)](fi_mr.3.html) takes precedence
    over this setopt option.

-   

    *FI_OPT_CUDA_API_PERMITTED - bool*
    :   This option only applies to the fi_setopt call. It is used to
        control endpoint's behavior in making calls to CUDA API. By
        default, an endpoint is permitted to call CUDA API. If user wish
        to prohibit an endpoint from making such calls, user can achieve
        that by set this option to false. If an endpoint's support of
        CUDA memory relies on making calls to CUDA API, it will return
        -FI_EOPNOTSUPP for the call to fi_setopt. If either CUDA library
        or CUDA device is not available, endpoint will return
        -FI_EINVAL. All providers that support FI_HMEM capability
        implement this option.

-   

    *FI_OPT_SHARED_MEMORY_PERMITTED - bool*
    :   This option only applies to the fi_setopt call. This option
        controls the use of shared memory for intra-node communication.
        Setting it to true will allow the use of shared memory. When set
        to false, shared memory will not be used and the implementation
        of intra-node communication is provider dependent.

-   

    *FI_OPT_MAX_MSG_SIZE - size_t*
    :   Define the maximum message size that can be transferred by the
        endpoint in a single untagged message. The size is limited by
        the endpoint's configuration and the provider's capabilities,
        and must be less than or equal to `ep_attr->max_msg_size`.
        Providers that don't support this option will return
        -FI_ENOPROTOOPT. In that case, `ep_attr->max_msg_size` should be
        used.

-   

    *FI_OPT_MAX_TAGGED_SIZE - size_t*
    :   Define the maximum message size that can be transferred by the
        endpoint in a single tagged message. The size is limited by the
        endpoint's configuration and the provider's capabilities, and
        must be less than or equal to `ep_attr->max_msg_size`. Providers
        that don't support this option will return -FI_ENOPROTOOPT. In
        that case, `ep_attr->max_msg_size` should be used.

-   

    *FI_OPT_MAX_RMA_SIZE - size_t*
    :   Define the maximum message size that can be transferred by the
        endpoint via a single RMA operation. The size is limited by the
        endpoint's configuration and the provider's capabilities, and
        must be less than or equal to `ep_attr->max_msg_size`. Providers
        that don't support this option will return -FI_ENOPROTOOPT. In
        that case, `ep_attr->max_msg_size` should be used.

-   

    *FI_OPT_MAX_ATOMIC_SIZE - size_t*
    :   Define the maximum data size that can be transferred by the
        endpoint via a single atomic operation. The size is limited by
        the endpoint's configuration and the provider's capabilities,
        and must be less than or equal to `ep_attr->max_msg_size`.
        Providers that don't support this option will return
        -FI_ENOPROTOOPT. In that case, `ep_attr->max_msg_size` should be
        used.

-   

    *FI_OPT_INJECT_MSG_SIZE - size_t*
    :   Define the maximum message size that can be injected by the
        endpoint in a single untagged message. The size is limited by
        the endpoint's configuration and the provider's capabilities,
        and must be less than or equal to `tx_attr->inject_size`.
        Providers that don't support this option will return
        -FI_ENOPROTOOPT. In that case, `tx_attr->inject_size` should be
        used.

-   

    *FI_OPT_INJECT_TAGGED_SIZE - size_t*
    :   Define the maximum message size that can be injected by the
        endpoint in a single tagged message. The size is limited by the
        endpoint's configuration and the provider's capabilities, and
        must be less than or equal to `tx_attr->inject_size`. Providers
        that don't support this option will return -FI_ENOPROTOOPT. In
        that case, `tx_attr->inject_size` should be used.

-   

    *FI_OPT_INJECT_RMA_SIZE - size_t*
    :   Define the maximum data size that can be injected by the
        endpoint in a single RMA operation. The size is limited by the
        endpoint's configuration and the provider's capabilities, and
        must be less than or equal to `tx_attr->inject_size`. Providers
        that don't support this option will return -FI_ENOPROTOOPT. In
        that case, `tx_attr->inject_size` should be used.

-   

    *FI_OPT_INJECT_ATOMIC_SIZE - size_t*
    :   Define the maximum data size that can be injected by the
        endpoint in a single atomic operation. The size is limited by
        the endpoint's configuration and the provider's capabilities,
        and must be less than or equal to `tx_attr->inject_size`.
        Providers that don't support this option will return
        -FI_ENOPROTOOPT. In that case, `tx_attr->inject_size` should be
        used.

## fi_tc_dscp_set

This call converts a DSCP defined value into a libfabric traffic class
value. It should be used when assigning a DSCP value when setting the
tclass field in either domain or endpoint attributes

## fi_tc_dscp_get

This call returns the DSCP value associated with the tclass field for
the domain or endpoint attributes.

## fi_rx_size_left (DEPRECATED)

This function has been deprecated and will be removed in a future
version of the library. It may not be supported by all providers.

The fi_rx_size_left call returns a lower bound on the number of receive
operations that may be posted to the given endpoint without that
operation returning -FI_EAGAIN. Depending on the specific details of the
subsequently posted receive operations (e.g., number of iov entries,
which receive function is called, etc.), it may be possible to post more
receive operations than originally indicated by fi_rx_size_left.

## fi_tx_size_left (DEPRECATED)

This function has been deprecated and will be removed in a future
version of the library. It may not be supported by all providers.

The fi_tx_size_left call returns a lower bound on the number of transmit
operations that may be posted to the given endpoint without that
operation returning -FI_EAGAIN. Depending on the specific details of the
subsequently posted transmit operations (e.g., number of iov entries,
which transmit function is called, etc.), it may be possible to post
more transmit operations than originally indicated by fi_tx_size_left.

# ENDPOINT ATTRIBUTES

The fi_ep_attr structure defines the set of attributes associated with
an endpoint. Endpoint attributes may be further refined using the
transmit and receive context attributes as shown below.

{% highlight c %} struct fi_ep_attr { enum fi_ep_type type; uint32_t
protocol; uint32_t protocol_version; size_t max_msg_size; size_t
msg_prefix_size; size_t max_order_raw_size; size_t max_order_war_size;
size_t max_order_waw_size; uint64_t mem_tag_format; size_t tx_ctx_cnt;
size_t rx_ctx_cnt; size_t auth_key_size; uint8_t \*auth_key; }; {%
endhighlight %}

## type - Endpoint Type

If specified, indicates the type of fabric interface communication
desired. Supported types are:

*FI_EP_DGRAM*
:   Supports a connectionless, unreliable datagram communication.
    Message boundaries are maintained, but the maximum message size may
    be limited to the fabric MTU. Flow control is not guaranteed.

*FI_EP_MSG*
:   Provides a reliable, connection-oriented data transfer service with
    flow control that maintains message boundaries.

*FI_EP_RDM*
:   Reliable datagram message. Provides a reliable, connectionless data
    transfer service with flow control that maintains message
    boundaries.

*FI_EP_UNSPEC*
:   The type of endpoint is not specified. This is usually provided as
    input, with other attributes of the endpoint or the provider
    selecting the type.

## Protocol

Specifies the low-level end to end protocol employed by the provider. A
matching protocol must be used by communicating endpoints to ensure
interoperability. The following protocol values are defined. Provider
specific protocols are also allowed. Provider specific protocols will be
indicated by having the upper bit of the protocol value set to one.

*FI_PROTO_EFA*
:   Proprietary protocol on Elastic Fabric Adapter fabric. It supports
    both DGRAM and RDM endpoints.

*FI_PROTO_IB_RDM*
:   Reliable-datagram protocol implemented over InfiniBand
    reliable-connected queue pairs.

*FI_PROTO_IB_UD*
:   The protocol runs over Infiniband unreliable datagram queue pairs.

*FI_PROTO_IWARP*
:   The protocol runs over the Internet wide area RDMA protocol
    transport.

*FI_PROTO_IWARP_RDM*
:   Reliable-datagram protocol implemented over iWarp reliable-connected
    queue pairs.

*FI_PROTO_NETWORKDIRECT*
:   Protocol runs over Microsoft NetworkDirect service provider
    interface. This adds reliable-datagram semantics over the
    NetworkDirect connection- oriented endpoint semantics.

*FI_PROTO_PSMX2*
:   The protocol is based on an Intel proprietary protocol known as
    PSM2, performance scaled messaging version 2. PSMX2 is an extended
    version of the PSM2 protocol to support the libfabric interfaces.

*FI_PROTO_PSMX3*
:   The protocol is Intel's protocol known as PSM3, performance scaled
    messaging version 3. PSMX3 is implemented over RoCEv2 and verbs.

*FI_PROTO_RDMA_CM_IB_RC*
:   The protocol runs over Infiniband reliable-connected queue pairs,
    using the RDMA CM protocol for connection establishment.

*FI_PROTO_RXD*
:   Reliable-datagram protocol implemented over datagram endpoints. RXD
    is a libfabric utility component that adds RDM endpoint semantics
    over DGRAM endpoint semantics.

*FI_PROTO_RXM*
:   Reliable-datagram protocol implemented over message endpoints. RXM
    is a libfabric utility component that adds RDM endpoint semantics
    over MSG endpoint semantics.

*FI_PROTO_SOCK_TCP*
:   The protocol is layered over TCP packets.

*FI_PROTO_UDP*
:   The protocol sends and receives UDP datagrams. For example, an
    endpoint using *FI_PROTO_UDP* will be able to communicate with a
    remote peer that is using Berkeley *SOCK_DGRAM* sockets using
    *IPPROTO_UDP*.

*FI_PROTO_SHM*
:   Protocol for intra-node communication using shared memory segments
    used by the shm provider

*FI_PROTO_SM2*
:   Protocol for intra-node communication using shared memory segments
    used by the sm2 provider

*FI_PROTO_CXI*
:   Reliable-datagram protocol optimized for HPC applications used by
    cxi provider.

*FI_PROTO_CXI_RNR*
:   A version of the FI_PROTO_CXI protocol that implements an RNR
    protocol which can be used when messaging is primarily expected and
    FI_ORDER_SAS ordering is not required.

*FI_PROTO_UNSPEC*
:   The protocol is not specified. This is usually provided as input,
    with other attributes of the socket or the provider selecting the
    actual protocol.

## protocol_version - Protocol Version

Identifies which version of the protocol is employed by the provider.
The protocol version allows providers to extend an existing protocol, by
adding support for additional features or functionality for example, in
a backward compatible manner. Providers that support different versions
of the same protocol should inter-operate, but only when using the
capabilities defined for the lesser version.

## max_msg_size - Max Message Size

Defines the maximum size for an application data transfer as a single
operation.

## msg_prefix_size - Message Prefix Size

Specifies the size of any required message prefix buffer space. This
field will be 0 unless the FI_MSG_PREFIX mode is enabled. If
msg_prefix_size is \> 0 the specified value will be a multiple of
8-bytes.

## Max RMA Ordered Size

The maximum ordered size specifies the delivery order of transport data
into target memory for RMA and atomic operations. Data ordering is
separate, but dependent on message ordering (defined below). Data
ordering is unspecified where message order is not defined.

Data ordering refers to the access of the same target memory by
subsequent operations. When back to back RMA read or write operations
access the same registered memory location, data ordering indicates
whether the second operation reads or writes the target memory after the
first operation has completed. For example, will an RMA read that
follows an RMA write read back the data that was written? Similarly,
will an RMA write that follows an RMA read update the target buffer
after the read has transferred the original data? Data ordering answers
these questions, even in the presence of errors, such as the need to
resend data because of lost or corrupted network traffic.

RMA ordering applies between two operations, and not within a single
data transfer. Therefore, ordering is defined per byte-addressable
memory location. I.e. ordering specifies whether location X is accessed
by the second operation after the first operation. Nothing is implied
about the completion of the first operation before the second operation
is initiated. For example, if the first operation updates locations X
and Y, but the second operation only accesses location X, there are no
guarantees defined relative to location Y and the second operation.

In order to support large data transfers being broken into multiple
packets and sent using multiple paths through the fabric, data ordering
may be limited to transfers of a specific size or less. Providers
specify when data ordering is maintained through the following values.
Note that even if data ordering is not maintained, message ordering may
be.

*max_order_raw_size*
:   Read after write size. If set, an RMA or atomic read operation
    issued after an RMA or atomic write operation, both of which are
    smaller than the size, will be ordered. Where the target memory
    locations overlap, the RMA or atomic read operation will see the
    results of the previous RMA or atomic write.

*max_order_war_size*
:   Write after read size. If set, an RMA or atomic write operation
    issued after an RMA or atomic read operation, both of which are
    smaller than the size, will be ordered. The RMA or atomic read
    operation will see the initial value of the target memory location
    before a subsequent RMA or atomic write updates the value.

*max_order_waw_size*
:   Write after write size. If set, an RMA or atomic write operation
    issued after an RMA or atomic write operation, both of which are
    smaller than the size, will be ordered. The target memory location
    will reflect the results of the second RMA or atomic write.

An order size value of 0 indicates that ordering is not guaranteed. A
value of -1 guarantees ordering for any data size.

## mem_tag_format - Memory Tag Format

The memory tag format field is used to convey information on the use of
the tag and ignore parameters in the fi_tagged API calls, as well as
matching criteria. This information is used by the provider to optimize
tag matching support, including alignment with wire protocols. The
following tag formats are defined:

*FI_TAG_BITS*

:   If specified on input to fi_getinfo, this indicates that tags
    contain up to 64-bits of data, and the receiver must apply
    ignore_bits to tags when matching receive buffers with sends. The
    output of fi_getinfo will set 0 or more upper bits of mem_tag_format
    to 0 to indicate those tag bits which are ignored or reserved by the
    provider. Applications must check the number of upper bits which are
    0 and set them to 0 on all tag and ignore bits.

The value of FI_TAG_BITS is 0, making this the default behavior if the
hints are left uninialized after being allocated by fi_allocinfo(). This
format provides the most flexibility to applications, but limits
provider optimization options. FI_TAG_BITS aligns with the behavior
defined for libfabric versions 1.x.

*FI_TAG_MPI*

:   FI_TAG_MPI is a constrained usage of FI_TAG_BITS. When selected,
    applications treat the tag as fields of data, rather than bits, with
    the ability to wildcard each field. The MPI tag format specifically
    targets MPI based implementations and applications. An MPI formatted
    tag consists of 2 fields: a message tag and a payload identier. The
    message tag is a 32-bit searchable tag. Matching on a message tag
    requires searching through a list of posted buffers at the receiver,
    which we refer to as a searchable tag. The integer tag in MPI
    point-to-point messages can map directly to the libfabric message
    tag field.

The second field is an identifier that corresponds to the operation or
data being carried in the message payload. For example, this field may
be used to identify the type of collective operation associated with a
message payload. Note that only the size and behavior for the MPI tag
formats are defined. Described use of the fields are only suggestions.

Applications that use the MPI format should initialize their tags using
the fi_tag_mpi() function. Ignore bits should be specified as
FI_MPI_IGNORE_TAG, FI_MPI_IGNORE_PAYLOAD, or their bitwise OR'ing.

*FI_TAG_CCL*

:   The FI_TAG_CCL format further restricts the FI_TAG_MPI format. When
    used, only a single tag field may be set, which must match exactly
    at the target. The field may not be wild carded. The CCL tag format
    targets collective communication libraries and applications. The CCL
    format consists of a single field: a payload identifier. The
    identifier corresponds to the operation or data being carried in the
    message payload. For example, this field may be used to identify
    whether a message is for point-to-point communication or part of a
    collective operation, and in the latter case, the type of collective
    operation.

The CCL tag format does not require searching for matching receive
buffers, only directing the message to the correct virtual message queue
based on to the payload identifier.

Applications that use the CCL format pass in the payload identifier
directly as the tag and set ignore bits to 0.

*FI_TAG_MAX_FORMAT*
:   If the value of mem_tag_format is \>= FI_TAG_MAX_FORMAT, the tag
    format is treated as a set of bit fields. The behavior is
    functionally the same as FI_TAG_BITS. The following description is
    for backwards compatibility and describes how the provider may
    interpret the mem_tag_format field if the value is \>=
    FI_TAG_MAX_FORMAT.

The memory tag format may be used to divide the bit array into separate
fields. The mem_tag_format optionally begins with a series of bits set
to 0, to signify bits which are ignored by the provider. Following the
initial prefix of ignored bits, the array will consist of alternating
groups of bits set to all 1's or all 0's. Each group of bits corresponds
to a tagged field. The implication of defining a tagged field is that
when a mask is applied to the tagged bit array, all bits belonging to a
single field will either be set to 1 or 0, collectively.

For example, a mem_tag_format of 0x30FF indicates support for 14 tagged
bits, separated into 3 fields. The first field consists of 2-bits, the
second field 4-bits, and the final field 8-bits. Valid masks for such a
tagged field would be a bitwise OR'ing of zero or more of the following
values: 0x3000, 0x0F00, and 0x00FF. The provider may not validate the
mask provided by the application for performance reasons.

By identifying fields within a tag, a provider may be able to optimize
their search routines. An application which requests tag fields must
provide tag masks that either set all mask bits corresponding to a field
to all 0 or all 1. When negotiating tag fields, an application can
request a specific number of fields of a given size. A provider must
return a tag format that supports the requested number of fields, with
each field being at least the size requested, or fail the request. A
provider may increase the size of the fields. When reporting completions
(see FI_CQ_FORMAT_TAGGED), it is not guaranteed that the provider would
clear out any unsupported tag bits in the tag field of the completion
entry.

It is recommended that field sizes be ordered from smallest to largest.
A generic, unstructured tag and mask can be achieved by requesting a bit
array consisting of alternating 1's and 0's.

## tx_ctx_cnt - Transmit Context Count

Number of transmit contexts to associate with the endpoint. If not
specified (0), 1 context will be assigned if the endpoint supports
outbound transfers. Transmit contexts are independent transmit queues
that may be separately configured. Each transmit context may be bound to
a separate CQ, and no ordering is defined between contexts.
Additionally, no synchronization is needed when accessing contexts in
parallel.

If the count is set to the value FI_SHARED_CONTEXT, the endpoint will be
configured to use a shared transmit context, if supported by the
provider. Providers that do not support shared transmit contexts will
fail the request.

See the scalable endpoint and shared contexts sections for additional
details.

## rx_ctx_cnt - Receive Context Count

Number of receive contexts to associate with the endpoint. If not
specified, 1 context will be assigned if the endpoint supports inbound
transfers. Receive contexts are independent processing queues that may
be separately configured. Each receive context may be bound to a
separate CQ, and no ordering is defined between contexts. Additionally,
no synchronization is needed when accessing contexts in parallel.

If the count is set to the value FI_SHARED_CONTEXT, the endpoint will be
configured to use a shared receive context, if supported by the
provider. Providers that do not support shared receive contexts will
fail the request.

See the scalable endpoint and shared contexts sections for additional
details.

## auth_key_size - Authorization Key Length

The length of the authorization key in bytes. This field will be 0 if
authorization keys are not available or used. This field is ignored
unless the fabric is opened with API version 1.5 or greater.

If the domain is opened with FI_AV_AUTH_KEY, auth_key_size must be 0.

## auth_key - Authorization Key

If supported by the fabric, an authorization key (a.k.a. job key) to
associate with the endpoint. An authorization key is used to limit
communication between endpoints. Only peer endpoints that are programmed
to use the same authorization key may communicate. Authorization keys
are often used to implement job keys, to ensure that processes running
in different jobs do not accidentally cross traffic. The domain
authorization key will be used if auth_key_size is set to 0. This field
is ignored unless the fabric is opened with API version 1.5 or greater.

If the domain is opened with FI_AV_AUTH_KEY, auth_key is must be NULL.

# TRANSMIT CONTEXT ATTRIBUTES

Attributes specific to the transmit capabilities of an endpoint are
specified using struct fi_tx_attr.

{% highlight c %} struct fi_tx_attr { uint64_t caps; uint64_t mode;
uint64_t op_flags; uint64_t msg_order; uint64_t comp_order; size_t
inject_size; size_t size; size_t iov_limit; size_t rma_iov_limit;
uint32_t tclass; }; {% endhighlight %}

## caps - Capabilities

The requested capabilities of the context. The capabilities must be a
subset of those requested of the associated endpoint in fi_info-\>caps.
See the CAPABILITIES section of fi_getinfo(3) for capability details. If
the caps field is 0 on input to fi_getinfo(3), the applicable capability
bits from the fi_info structure will be used.

The following capabilities apply to the transmit attributes: FI_MSG,
FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_SEND, FI_HMEM,
FI_TRIGGER, FI_FENCE, FI_MULTICAST, FI_RMA_PMEM, FI_NAMED_RX_CTX,
FI_COLLECTIVE, and FI_XPU.

Many applications will be able to ignore this field and rely solely on
the fi_info::caps field. Use of this field provides fine grained control
over the transmit capabilities associated with an endpoint. It is useful
when handling scalable endpoints, with multiple transmit contexts, for
example, and allows configuring a specific transmit context with fewer
capabilities than that supported by the endpoint or other transmit
contexts.

## mode

The operational mode bits of the context. The mode bits will be a subset
of those associated with the endpoint. See the MODE section of
fi_getinfo(3) for details. A mode value of 0 will be ignored on input to
fi_getinfo(3), with the mode value of the fi_info structure used
instead. On return from fi_getinfo(3), the mode will be set only to
those constraints specific to transmit operations.

## op_flags - Default transmit operation flags

Flags that control the operation of operations submitted against the
context. Applicable flags are listed in the Operation Flags section.

## msg_order - Message Ordering

Message ordering refers to the order in which transport layer headers
(as viewed by the application) are identified and processed. Relaxed
message order enables data transfers to be sent and received out of
order, which may improve performance by utilizing multiple paths through
the fabric from the initiating endpoint to a target endpoint. Message
order applies only between a single source and destination endpoint
pair. Ordering between different target endpoints is not defined.

Message order is determined using a set of ordering bits. Each set bit
indicates that ordering is maintained between data transfers of the
specified type. Message order is defined for \[read \| write \| send\]
operations submitted by an application after \[read \| write \| send\]
operations. Value 0 indicates that no ordering is specified. Value 0 may
be used as input in order to obtain the default message order supported
by the provider.

Message ordering only applies to the end to end transmission of
transport headers. Message ordering is necessary, but does not
guarantee, the order in which message data is sent or received by the
transport layer. Message ordering requires matching ordering semantics
on the receiving side of a data transfer operation in order to guarantee
that ordering is met.

*FI_ORDER_NONE* (deprecated)
:   This is an alias for value 0. It is deprecated and should not be
    used.

*FI_ORDER_ATOMIC_RAR*
:   Atomic read after read. If set, atomic fetch operations are
    transmitted in the order submitted relative to other atomic fetch
    operations. If not set, atomic fetches may be transmitted out of
    order from their submission.

*FI_ORDER_ATOMIC_RAW*
:   Atomic read after write. If set, atomic fetch operations are
    transmitted in the order submitted relative to atomic update
    operations. If not set, atomic fetches may be transmitted ahead of
    atomic updates.

*FI_ORDER_ATOMIC_WAR*
:   RMA write after read. If set, atomic update operations are
    transmitted in the order submitted relative to atomic fetch
    operations. If not set, atomic updates may be transmitted ahead of
    atomic fetches.

*FI_ORDER_ATOMIC_WAW*
:   RMA write after write. If set, atomic update operations are
    transmitted in the order submitted relative to other atomic update
    operations. If not atomic updates may be transmitted out of order
    from their submission.

*FI_ORDER_RAR*
:   Read after read. If set, RMA and atomic read operations are
    transmitted in the order submitted relative to other RMA and atomic
    read operations. If not set, RMA and atomic reads may be transmitted
    out of order from their submission.

*FI_ORDER_RAS*
:   Read after send. If set, RMA and atomic read operations are
    transmitted in the order submitted relative to message send
    operations, including tagged sends. If not set, RMA and atomic reads
    may be transmitted ahead of sends.

*FI_ORDER_RAW*
:   Read after write. If set, RMA and atomic read operations are
    transmitted in the order submitted relative to RMA and atomic write
    operations. If not set, RMA and atomic reads may be transmitted
    ahead of RMA and atomic writes.

*FI_ORDER_RMA_RAR*
:   RMA read after read. If set, RMA read operations are transmitted in
    the order submitted relative to other RMA read operations. If not
    set, RMA reads may be transmitted out of order from their
    submission.

*FI_ORDER_RMA_RAW*
:   RMA read after write. If set, RMA read operations are transmitted in
    the order submitted relative to RMA write operations. If not set,
    RMA reads may be transmitted ahead of RMA writes.

*FI_ORDER_RMA_WAR*
:   RMA write after read. If set, RMA write operations are transmitted
    in the order submitted relative to RMA read operations. If not set,
    RMA writes may be transmitted ahead of RMA reads.

*FI_ORDER_RMA_WAW*
:   RMA write after write. If set, RMA write operations are transmitted
    in the order submitted relative to other RMA write operations. If
    not set, RMA writes may be transmitted out of order from their
    submission.

*FI_ORDER_SAR*
:   Send after read. If set, message send operations, including tagged
    sends, are transmitted in order submitted relative to RMA and atomic
    read operations. If not set, message sends may be transmitted ahead
    of RMA and atomic reads.

*FI_ORDER_SAS*
:   Send after send. If set, message send operations, including tagged
    sends, are transmitted in the order submitted relative to other
    message send. If not set, message sends may be transmitted out of
    order from their submission.

*FI_ORDER_SAW*
:   Send after write. If set, message send operations, including tagged
    sends, are transmitted in order submitted relative to RMA and atomic
    write operations. If not set, message sends may be transmitted ahead
    of RMA and atomic writes.

*FI_ORDER_WAR*
:   Write after read. If set, RMA and atomic write operations are
    transmitted in the order submitted relative to RMA and atomic read
    operations. If not set, RMA and atomic writes may be transmitted
    ahead of RMA and atomic reads.

*FI_ORDER_WAS*
:   Write after send. If set, RMA and atomic write operations are
    transmitted in the order submitted relative to message send
    operations, including tagged sends. If not set, RMA and atomic
    writes may be transmitted ahead of sends.

*FI_ORDER_WAW*
:   Write after write. If set, RMA and atomic write operations are
    transmitted in the order submitted relative to other RMA and atomic
    write operations. If not set, RMA and atomic writes may be
    transmitted out of order from their submission.

## comp_order - Completion Ordering

This field is provided for version 1 compatibility and should be set to
0.

**Deprecated**

Completion ordering refers to the order in which completed requests are
written into the completion queue. Supported completion order values
are:

*FI_ORDER_NONE* (deprecated)
:   No ordering is defined for completed operations. Requests submitted
    to the transmit context may complete in any order.

*FI_ORDER_STRICT* (deprecated)
:   Requests complete in the order in which they are submitted to the
    transmit context.

## inject_size

The requested inject operation size (see the FI_INJECT flag) that the
context will support. This is the maximum size data transfer that can be
associated with an inject operation (such as fi_inject) or may be used
with the FI_INJECT data transfer flag.

## size

The size of the transmit context. The mapping of the size value to
resources is provider specific, but it is directly related to the number
of command entries allocated for the endpoint. A smaller size value
consumes fewer hardware and software resources, while a larger size
allows queuing more transmit requests.

While the size attribute guides the size of underlying endpoint transmit
queue, there is not necessarily a one-to-one mapping between a transmit
operation and a queue entry. A single transmit operation may consume
multiple queue entries; for example, one per scatter-gather entry.
Additionally, the size field is intended to guide the allocation of the
endpoint's transmit context. Specifically, for connectionless endpoints,
there may be lower-level queues use to track communication on a per peer
basis. The sizes of any lower-level queues may only be significantly
smaller than the endpoint's transmit size, in order to reduce resource
utilization.

## iov_limit

This is the maximum number of IO vectors (scatter-gather elements) that
a single posted operation may reference.

## rma_iov_limit

This is the maximum number of RMA IO vectors (scatter-gather elements)
that an RMA or atomic operation may reference. The rma_iov_limit
corresponds to the rma_iov_count values in RMA and atomic operations.
See struct fi_msg_rma and struct fi_msg_atomic in fi_rma.3 and
fi_atomic.3, for additional details. This limit applies to both the
number of RMA IO vectors that may be specified when initiating an
operation from the local endpoint, as well as the maximum number of IO
vectors that may be carried in a single request from a remote endpoint.

## Traffic Class (tclass)

Traffic classes can be a differentiated services code point (DSCP)
value, one of the following defined labels, or a provider-specific
definition. If tclass is unset or set to FI_TC_UNSPEC, the endpoint will
use the default traffic class associated with the domain.

*FI_TC_BEST_EFFORT*
:   This is the default in the absence of any other local or fabric
    configuration. This class carries the traffic for a number of
    applications executing concurrently over the same network
    infrastructure. Even though it is shared, network capacity and
    resource allocation are distributed fairly across the applications.

*FI_TC_BULK_DATA*
:   This class is intended for large data transfers associated with I/O
    and is present to separate sustained I/O transfers from other
    application inter-process communications.

*FI_TC_DEDICATED_ACCESS*
:   This class operates at the highest priority, except the management
    class. It carries a high bandwidth allocation, minimum latency
    targets, and the highest scheduling and arbitration priority.

*FI_TC_LOW_LATENCY*
:   This class supports low latency, low jitter data patterns typically
    caused by transactional data exchanges, barrier synchronizations,
    and collective operations that are typical of HPC applications. This
    class often requires maximum tolerable latencies that data transfers
    must achieve for correct or performance operations. Fulfillment of
    such requests in this class will typically require accompanying
    bandwidth and message size limitations so as not to consume
    excessive bandwidth at high priority.

*FI_TC_NETWORK_CTRL*
:   This class is intended for traffic directly related to fabric
    (network) management, which is critical to the correct operation of
    the network. Its use is typically restricted to privileged network
    management applications.

*FI_TC_SCAVENGER*
:   This class is used for data that is desired but does not have strict
    delivery requirements, such as in-band network or application level
    monitoring data. Use of this class indicates that the traffic is
    considered lower priority and should not interfere with higher
    priority workflows.

*fi_tc_dscp_set / fi_tc_dscp_get*
:   DSCP values are supported via the DSCP get and set functions. The
    definitions for DSCP values are outside the scope of libfabric. See
    the fi_tc_dscp_set and fi_tc_dscp_get function definitions for
    details on their use.

# RECEIVE CONTEXT ATTRIBUTES

Attributes specific to the receive capabilities of an endpoint are
specified using struct fi_rx_attr.

{% highlight c %} struct fi_rx_attr { uint64_t caps; uint64_t mode;
uint64_t op_flags; uint64_t msg_order; uint64_t comp_order; size_t size;
size_t iov_limit; }; {% endhighlight %}

## caps - Capabilities

The requested capabilities of the context. The capabilities must be a
subset of those requested of the associated endpoint in fi_info-\>caps.
See the CAPABILITIES section of fi_getinfo(3) for capability details. If
the caps field is 0 on input to fi_getinfo(3), the applicable capability
bits from the fi_info structure will be used.

The following capabilities apply to the receive attributes: FI_MSG,
FI_RMA, FI_TAGGED, FI_ATOMIC, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_RECV,
FI_HMEM, FI_TRIGGER, FI_RMA_PMEM, FI_DIRECTED_RECV,
FI_TAGGED_DIRECTED_RECV, FI_TAGGED_MULTI_RECV, FI_MULTI_RECV, FI_SOURCE,
FI_RMA_EVENT, FI_SOURCE_ERR, FI_COLLECTIVE, and FI_XPU.

Many applications will be able to ignore this field and rely solely on
the fi_info::caps field. Use of this field provides fine grained control
over the receive capabilities associated with an endpoint. It is useful
when handling scalable endpoints, with multiple receive contexts, for
example, and allows configuring a specific receive context with fewer
capabilities than that supported by the endpoint or other receive
contexts.

## mode

The operational mode bits of the context. The mode bits will be a subset
of those associated with the endpoint. See the MODE section of
fi_getinfo(3) for details. A mode value of 0 will be ignored on input to
fi_getinfo(3), with the mode value of the fi_info structure used
instead. On return from fi_getinfo(3), the mode will be set only to
those constraints specific to receive operations.

## op_flags - Default receive operation flags

Flags that control the operation of operations submitted against the
context. Applicable flags are listed in the Operation Flags section.

## msg_order - Message Ordering

For a description of message ordering, see the msg_order field in the
*Transmit Context Attribute* section. Receive context message ordering
defines the order in which received transport message headers are
processed when received by an endpoint. When ordering is set, it
indicates that message headers will be processed in order, based on how
the transmit side has identified the messages. Typically, this means
that messages will be handled in order based on a message level sequence
number.

The following ordering flags, as defined for transmit ordering, also
apply to the processing of received operations: FI_ORDER_RAR,
FI_ORDER_RAW, FI_ORDER_RAS, FI_ORDER_WAR, FI_ORDER_WAW, FI_ORDER_WAS,
FI_ORDER_SAR, FI_ORDER_SAW, FI_ORDER_SAS, FI_ORDER_RMA_RAR,
FI_ORDER_RMA_RAW, FI_ORDER_RMA_WAR, FI_ORDER_RMA_WAW,
FI_ORDER_ATOMIC_RAR, FI_ORDER_ATOMIC_RAW, FI_ORDER_ATOMIC_WAR, and
FI_ORDER_ATOMIC_WAW.

## comp_order - Completion Ordering

This field is provided for version 1 compatibility and should be set to
0.

**Deprecated**

Completion ordering refers to the order in which completed requests are
written into the completion queue. Supported completion order values
are:

*FI_ORDER_DATA* (deprecated)
:   When set, this bit indicates that received data is written into
    memory in order. Data ordering applies to memory accessed as part of
    a single operation and between operations if message ordering is
    guaranteed.

*FI_ORDER_NONE* (deprecated)
:   No ordering is defined for completed operations. Requests submitted
    to the transmit context may complete in any order.

*FI_ORDER_STRICT* (deprecated)
:   Requests complete in the order in which they are submitted to the
    transmit context.

## total_buffered_recv

This field is provided for version 1 compatibility and should be set to
0.

## size

The size of the receive context. The mapping of the size value to
resources is provider specific, but it is directly related to the number
of command entries allocated for the endpoint. A smaller size value
consumes fewer hardware and software resources, while a larger size
allows queuing more transmit requests.

While the size attribute guides the size of underlying endpoint receive
queue, there is not necessarily a one-to-one mapping between a receive
operation and a queue entry. A single receive operation may consume
multiple queue entries; for example, one per scatter-gather entry.
Additionally, the size field is intended to guide the allocation of the
endpoint's receive context. Specifically, for connectionless endpoints,
there may be lower-level queues use to track communication on a per peer
basis. The sizes of any lower-level queues may only be significantly
smaller than the endpoint's receive size, in order to reduce resource
utilization.

## iov_limit

This is the maximum number of IO vectors (scatter-gather elements) that
a single posted operating may reference.

# SCALABLE ENDPOINTS

A scalable endpoint is a communication portal that supports multiple
transmit and receive contexts. Scalable endpoints are loosely modeled
after the networking concept of transmit/receive side scaling, also
known as multi-queue. Support for scalable endpoints is domain specific.
Scalable endpoints may improve the performance of multi-threaded and
parallel applications, by allowing threads to access independent
transmit and receive queues. A scalable endpoint has a single transport
level address, which can reduce the memory requirements needed to store
remote addressing data, versus using standard endpoints. Scalable
endpoints cannot be used directly for communication operations, and
require the application to explicitly create transmit and receive
contexts as described below.

## fi_tx_context

Transmit contexts are independent transmit queues. Ordering and
synchronization between contexts are not defined. Conceptually a
transmit context behaves similar to a send-only endpoint. A transmit
context may be configured with fewer capabilities than the base endpoint
and with different attributes (such as ordering requirements and inject
size) than other contexts associated with the same scalable endpoint.
Each transmit context has its own completion queue. The number of
transmit contexts associated with an endpoint is specified during
endpoint creation.

The fi_tx_context call is used to retrieve a specific context,
identified by an index (see above for details on transmit context
attributes). Providers may dynamically allocate contexts when
fi_tx_context is called, or may statically create all contexts when
fi_endpoint is invoked. By default, a transmit context inherits the
properties of its associated endpoint. However, applications may request
context specific attributes through the attr parameter. Support for per
transmit context attributes is provider specific and not guaranteed.
Providers will return the actual attributes assigned to the context
through the attr parameter, if provided.

## fi_rx_context

Receive contexts are independent receive queues for receiving incoming
data. Ordering and synchronization between contexts are not guaranteed.
Conceptually a receive context behaves similar to a receive-only
endpoint. A receive context may be configured with fewer capabilities
than the base endpoint and with different attributes (such as ordering
requirements and inject size) than other contexts associated with the
same scalable endpoint. Each receive context has its own completion
queue. The number of receive contexts associated with an endpoint is
specified during endpoint creation.

Receive contexts are often associated with steering flows, that specify
which incoming packets targeting a scalable endpoint to process.
However, receive contexts may be targeted directly by the initiator, if
supported by the underlying protocol. Such contexts are referred to as
'named'. Support for named contexts must be indicated by setting the
caps FI_NAMED_RX_CTX capability when the corresponding endpoint is
created. Support for named receive contexts is coordinated with address
vectors. See fi_av(3) and fi_rx_addr(3).

The fi_rx_context call is used to retrieve a specific context,
identified by an index (see above for details on receive context
attributes). Providers may dynamically allocate contexts when
fi_rx_context is called, or may statically create all contexts when
fi_endpoint is invoked. By default, a receive context inherits the
properties of its associated endpoint. However, applications may request
context specific attributes through the attr parameter. Support for per
receive context attributes is provider specific and not guaranteed.
Providers will return the actual attributes assigned to the context
through the attr parameter, if provided.

# SHARED CONTEXTS

Shared contexts are transmit and receive contexts explicitly shared
among one or more endpoints. A shareable context allows an application
to use a single dedicated provider resource among multiple transport
addressable endpoints. This can greatly reduce the resources needed to
manage communication over multiple endpoints by multiplexing transmit
and/or receive processing, with the potential cost of serializing access
across multiple endpoints. Support for shareable contexts is domain
specific.

Conceptually, shareable transmit contexts are transmit queues that may
be accessed by many endpoints. The use of a shared transmit context is
mostly opaque to an application. Applications must allocate and bind
shared transmit contexts to endpoints, but operations are posted
directly to the endpoint. Shared transmit contexts are not associated
with completion queues or counters. Completed operations are posted to
the CQs bound to the endpoint. An endpoint may only be associated with a
single shared transmit context.

Unlike shared transmit contexts, applications interact directly with
shared receive contexts. Users post receive buffers directly to a shared
receive context, with the buffers usable by any endpoint bound to the
shared receive context. Shared receive contexts are not associated with
completion queues or counters. Completed receive operations are posted
to the CQs bound to the endpoint. An endpoint may only be associated
with a single receive context, and all connectionless endpoints
associated with a shared receive context must also share the same
address vector.

Endpoints associated with a shared transmit context may use dedicated
receive contexts, and vice-versa. Or an endpoint may use shared transmit
and receive contexts. And there is no requirement that the same group of
endpoints sharing a context of one type also share the context of an
alternate type. Furthermore, an endpoint may use a shared context of one
type, but a scalable set of contexts of the alternate type.

## fi_stx_context

This call is used to open a shareable transmit context (see above for
details on the transmit context attributes). Endpoints associated with a
shared transmit context must use a subset of the transmit context's
attributes. Note that this is the reverse of the requirement for
transmit contexts for scalable endpoints.

## fi_srx_context

This allocates a shareable receive context (see above for details on the
receive context attributes). Endpoints associated with a shared receive
context must use a subset of the receive context's attributes. Note that
this is the reverse of the requirement for receive contexts for scalable
endpoints.

# SOCKET ENDPOINTS

The following feature and description should be considered experimental.
Until the experimental tag is removed, the interfaces, semantics, and
data structures associated with socket endpoints may change between
library versions.

This section applies to endpoints of type FI_EP_SOCK_STREAM and
FI_EP_SOCK_DGRAM, commonly referred to as socket endpoints.

Socket endpoints are defined with semantics that allow them to more
easily be adopted by developers familiar with the UNIX socket API, or by
middleware that exposes the socket API, while still taking advantage of
high-performance hardware features.

The key difference between socket endpoints and other active endpoints
are socket endpoints use synchronous data transfers. Buffers passed into
send and receive operations revert to the control of the application
upon returning from the function call. As a result, no data transfer
completions are reported to the application, and socket endpoints are
not associated with completion queues or counters.

Socket endpoints support a subset of message operations: fi_send,
fi_sendv, fi_sendmsg, fi_recv, fi_recvv, fi_recvmsg, and fi_inject.
Because data transfers are synchronous, the return value from send and
receive operations indicate the number of bytes transferred on success,
or a negative value on error, including -FI_EAGAIN if the endpoint
cannot send or receive any data because of full or empty queues,
respectively.

Socket endpoints are associated with event queues and address vectors,
and process connection management events asynchronously, similar to
other endpoints. Unlike UNIX sockets, socket endpoint must still be
declared as either active or passive.

Socket endpoints behave like non-blocking sockets. In order to support
select and poll semantics, active socket endpoints are associated with a
file descriptor that is signaled whenever the endpoint is ready to send
and/or receive data. The file descriptor may be retrieved using
fi_control.

# OPERATION FLAGS

Operation flags are obtained by OR-ing the following flags together.
Operation flags define the default flags applied to an endpoint's data
transfer operations, where a flags parameter is not available. Data
transfer operations that take flags as input override the op_flags value
of transmit or receive context attributes of an endpoint.

*FI_COMMIT_COMPLETE*
:   Indicates that a completion should not be generated (locally or at
    the peer) until the result of an operation have been made
    persistent. See [`fi_cq`(3)](fi_cq.3.html) for additional details on
    completion semantics.

*FI_COMPLETION*
:   Indicates that a completion queue entry should be written for data
    transfer operations. This flag only applies to operations issued on
    an endpoint that was bound to a completion queue with the
    FI_SELECTIVE_COMPLETION flag set, otherwise, it is ignored. See the
    fi_ep_bind section above for more detail.

*FI_DELIVERY_COMPLETE*
:   Indicates that a completion should be generated when the operation
    has been processed by the destination endpoint(s). See
    [`fi_cq`(3)](fi_cq.3.html) for additional details on completion
    semantics.

*FI_INJECT*
:   Indicates that all outbound data buffers should be returned to the
    user's control immediately after a data transfer call returns, even
    if the operation is handled asynchronously. This may require that
    the provider copy the data into a local buffer and transfer out of
    that buffer. A provider can limit the total amount of send data that
    may be buffered and/or the size of a single send that can use this
    flag. This limit is indicated using inject_size (see inject_size
    above).

*FI_INJECT_COMPLETE*
:   Indicates that a completion should be generated when the source
    buffer(s) may be reused. See [`fi_cq`(3)](fi_cq.3.html) for
    additional details on completion semantics.

*FI_MULTICAST*
:   Indicates that data transfers will target multicast addresses by
    default. Any fi_addr_t passed into a data transfer operation will be
    treated as a multicast address.

*FI_MULTI_RECV*
:   Applies to posted receive operations. This flag allows the user to
    post a single buffer that will receive multiple incoming messages.
    Received messages will be packed into the receive buffer until the
    buffer has been consumed. Use of this flag may cause a single posted
    receive operation to generate multiple completions as messages are
    placed into the buffer. The placement of received data into the
    buffer may be subjected to provider specific alignment restrictions.
    The buffer will be released by the provider when the available
    buffer space falls below the specified minimum (see
    FI_OPT_MIN_MULTI_RECV).

*FI_TRANSMIT_COMPLETE*
:   Indicates that a completion should be generated when the transmit
    operation has completed relative to the local provider. See
    [`fi_cq`(3)](fi_cq.3.html) for additional details on completion
    semantics.

# NOTES

Users should call fi_close to release all resources allocated to the
fabric endpoint.

Endpoints allocated with the FI_CONTEXT or FI_CONTEXT2 mode bits set
must typically provide struct fi_context(2) as their per operation
context parameter. (See fi_getinfo.3 for details.) However, when
FI_SELECTIVE_COMPLETION is enabled to suppress CQ completion entries,
and an operation is initiated without the FI_COMPLETION flag set, then
the context parameter is ignored. An application does not need to pass
in a valid struct fi_context(2) into such data transfers.

Operations that complete in error that are not associated with valid
operational context will use the endpoint context in any error reporting
structures.

Although applications typically associate individual completions with
either completion queues or counters, an endpoint can be attached to
both a counter and completion queue. When combined with using selective
completions, this allows an application to use counters to track
successful completions, with a CQ used to report errors. Operations that
complete with an error increment the error counter and generate a CQ
completion event.

As mentioned in fi_getinfo(3), the ep_attr structure can be used to
query providers that support various endpoint attributes. fi_getinfo can
return provider info structures that can support the minimal set of
requirements (such that the application maintains correctness). However,
it can also return provider info structures that exceed application
requirements. As an example, consider an application requesting no
msg_order. The resulting output from fi_getinfo may have all the
ordering bits set. The application can reset the ordering bits it does
not require before creating the endpoint. The provider is free to
implement a stricter ordering than is required by the application.

# RETURN VALUES

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. For fi_cancel, a return value of 0 indicates that the
cancel request was submitted for processing, a return value of
-FI_EAGAIN indicates that the request could not be submitted and that it
should be retried once progress has been made. For fi_setopt/fi_getopt,
a return value of -FI_ENOPROTOOPT indicates the provider does not
support the requested option.

Fabric errno values are defined in `rdma/fi_errno.h`.

# ERRORS

*-FI_EDOMAIN*
:   A resource domain was not bound to the endpoint or an attempt was
    made to bind multiple domains.

*-FI_ENOCQ*
:   The endpoint has not been configured with necessary completion
    queue.

*-FI_EOPBADSTATE*
:   The endpoint's state does not permit the requested operation.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cq`(3)](fi_cq.3.html)
[`fi_msg`(3)](fi_msg.3.html), [`fi_tagged`(3)](fi_tagged.3.html),
[`fi_rma`(3)](fi_rma.3.html) [`fi_peer`(3)](fi_peer.3.html)

{% include JB/setup %}

# NAME

fi_eq - Event queue operations

fi_eq_open / fi_close
:   Open/close an event queue

fi_control
:   Control operation of EQ

fi_eq_read / fi_eq_readerr
:   Read an event from an event queue

fi_eq_write
:   Writes an event to an event queue

fi_eq_sread
:   A synchronous (blocking) read of an event queue

fi_eq_strerror
:   Converts provider specific error information into a printable string

# SYNOPSIS

``` c
#include <rdma/fi_domain.h>

int fi_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
    struct fid_eq **eq, void *context);

int fi_close(struct fid *eq);

int fi_control(struct fid *eq, int command, void *arg);

ssize_t fi_eq_read(struct fid_eq *eq, uint32_t *event,
    void *buf, size_t len, uint64_t flags);

ssize_t fi_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *buf,
    uint64_t flags);

ssize_t fi_eq_write(struct fid_eq *eq, uint32_t event,
    const void *buf, size_t len, uint64_t flags);

ssize_t fi_eq_sread(struct fid_eq *eq, uint32_t *event,
    void *buf, size_t len, int timeout, uint64_t flags);

const char * fi_eq_strerror(struct fid_eq *eq, int prov_errno,
      const void *err_data, char *buf, size_t len);
```

# ARGUMENTS

*fabric*
:   Opened fabric descriptor

*eq*
:   Event queue

*attr*
:   Event queue attributes

*context*
:   User specified context associated with the event queue.

*event*
:   Reported event

*buf*
:   For read calls, the data buffer to write events into. For write
    calls, an event to insert into the event queue. For fi_eq_strerror,
    an optional buffer that receives printable error information.

*len*
:   Length of data buffer

*flags*
:   Additional flags to apply to the operation

*command*
:   Command of control operation to perform on EQ.

*arg*
:   Optional control argument

*prov_errno*
:   Provider specific error value

*err_data*
:   Provider specific error data related to a completion

*timeout*
:   Timeout specified in milliseconds

# DESCRIPTION

Event queues are used to report events associated with control
operations. They are associated with memory registration, address
vectors, connection management, and fabric and domain level events.
Reported events are either associated with a requested operation or
affiliated with a call that registers for specific types of events, such
as listening for connection requests.

## fi_eq_open

fi_eq_open allocates a new event queue.

The properties and behavior of an event queue are defined by
`struct fi_eq_attr`.

``` c
struct fi_eq_attr {
    size_t               size;      /* # entries for EQ */
    uint64_t             flags;     /* operation flags */
    enum fi_wait_obj     wait_obj;  /* requested wait object */
    int                  signaling_vector; /* interrupt affinity */
    struct fid_wait     *wait_set;  /* optional wait set, deprecated */
};
```

*size*
:   Specifies the minimum size of an event queue.

*flags*
:   Flags that control the configuration of the EQ.

\- *FI_WRITE*
:   Indicates that the application requires support for inserting user
    events into the EQ. If this flag is set, then the fi_eq_write
    operation must be supported by the provider. If the FI_WRITE flag is
    not set, then the application may not invoke fi_eq_write.

\- *FI_AFFINITY*
:   Indicates that the signaling_vector field (see below) is valid.

*wait_obj*
:   EQ's may be associated with a specific wait object. Wait objects
    allow applications to block until the wait object is signaled,
    indicating that an event is available to be read. Users may use
    fi_control to retrieve the underlying wait object associated with an
    EQ, in order to use it in other system calls. The following values
    may be used to specify the type of wait object associated with an
    EQ:

\- *FI_WAIT_NONE*
:   Used to indicate that the user will not block (wait) for events on
    the EQ. When FI_WAIT_NONE is specified, the application may not call
    fi_eq_sread. This is the default is no wait object is specified.

\- *FI_WAIT_UNSPEC*
:   Specifies that the user will only wait on the EQ using fabric
    interface calls, such as fi_eq_sread. In this case, the underlying
    provider may select the most appropriate or highest performing wait
    object available, including custom wait mechanisms. Applications
    that select FI_WAIT_UNSPEC are not guaranteed to retrieve the
    underlying wait object.

\- *FI_WAIT_SET* (deprecated)
:   Indicates that the event queue should use a wait set object to wait
    for events. If specified, the wait_set field must reference an
    existing wait set object.

\- *FI_WAIT_FD*
:   Indicates that the EQ should use a file descriptor as its wait
    mechanism. A file descriptor wait object must be usable in select,
    poll, and epoll routines. However, a provider may signal an FD wait
    object by marking it as readable or with an error.

\- *FI_WAIT_MUTEX_COND* (deprecated)
:   Specifies that the EQ should use a pthread mutex and cond variable
    as a wait object.

\- *FI_WAIT_YIELD*
:   Indicates that the EQ will wait without a wait object but instead
    yield on every wait. Allows usage of fi_eq_sread through a spin.

*signaling_vector*
:   If the FI_AFFINITY flag is set, this indicates the logical cpu
    number (0..max cpu - 1) that interrupts associated with the EQ
    should target. This field should be treated as a hint to the
    provider and may be ignored if the provider does not support
    interrupt affinity.

*wait_set* (deprecated)
:   If wait_obj is FI_WAIT_SET, this field references a wait object to
    which the event queue should attach. When an event is inserted into
    the event queue, the corresponding wait set will be signaled if all
    necessary conditions are met. The use of a wait_set enables an
    optimized method of waiting for events across multiple event queues.
    This field is ignored if wait_obj is not FI_WAIT_SET.

## fi_close

The fi_close call releases all resources associated with an event queue.
Any events which remain on the EQ when it is closed are lost.

The EQ must not be bound to any other objects prior to being closed,
otherwise the call will return -FI_EBUSY.

## fi_control

The fi_control call is used to access provider or implementation
specific details of the event queue. Access to the EQ should be
serialized across all calls when fi_control is invoked, as it may
redirect the implementation of EQ operations. The following control
commands are usable with an EQ.

*FI_GETWAIT (void \*\*)*
:   This command allows the user to retrieve the low-level wait object
    associated with the EQ. The format of the wait-object is specified
    during EQ creation, through the EQ attributes. The fi_control arg
    parameter should be an address where a pointer to the returned wait
    object will be written. This should be an 'int \*' for FI_WAIT_FD,
    or 'struct fi_mutex_cond' for FI_WAIT_MUTEX_COND (deprecated).

``` c
struct fi_mutex_cond {
    pthread_mutex_t     *mutex;
    pthread_cond_t      *cond;
};
```

## fi_eq_read

The fi_eq_read operations performs a non-blocking read of event data
from the EQ. The format of the event data is based on the type of event
retrieved from the EQ, with all events starting with a struct
fi_eq_entry header. At most one event will be returned per EQ read
operation. The number of bytes successfully read from the EQ is returned
from the read. The FI_PEEK flag may be used to indicate that event data
should be read from the EQ without being consumed. A subsequent read
without the FI_PEEK flag would then remove the event from the EQ.

The following types of events may be reported to an EQ, along with
information regarding the format associated with each event.

*Asynchronous Control Operations*
:   Asynchronous control operations are basic requests that simply need
    to generate an event to indicate that they have completed. These
    include the following types of events: memory registration, address
    vector resolution, and multicast joins.

Control requests report their completion by inserting a
`struct   fi_eq_entry` into the EQ. The format of this structure is:

``` c
struct fi_eq_entry {
    fid_t            fid;        /* fid associated with request */
    void            *context;    /* operation context */
    uint64_t         data;       /* completion-specific data */
};
```

For the completion of basic asynchronous control operations, the
returned event will indicate the operation that has completed, and the
fid will reference the fabric descriptor associated with the event. For
memory registration, this will be an FI_MR_COMPLETE event and the
fid_mr. Address resolution will reference an FI_AV_COMPLETE event and
fid_av. Multicast joins will report an FI_JOIN_COMPLETE and fid_mc. The
context field will be set to the context specified as part of the
operation, if available, otherwise the context will be associated with
the fabric descriptor. The data field will be set as described in the
man page for the corresponding object type (e.g., see
[`fi_av`(3)](fi_av.3.html) for a description of how asynchronous address
vector insertions are completed).

*Connection Notification*
:   Connection notifications are connection management notifications
    used to setup or tear down connections between endpoints. There are
    three connection notification events: FI_CONNREQ, FI_CONNECTED, and
    FI_SHUTDOWN. Connection notifications are reported using
    `struct   fi_eq_cm_entry`:

``` c
struct fi_eq_cm_entry {
    fid_t            fid;        /* fid associated with request */
    struct fi_info  *info;       /* endpoint information */
    uint8_t         data[];     /* app connection data */
};
```

A connection request (FI_CONNREQ) event indicates that a remote endpoint
wishes to establish a new connection to a listening, or passive,
endpoint. The fid is the passive endpoint. Information regarding the
requested, active endpoint's capabilities and attributes are available
from the info field. The application is responsible for freeing this
structure by calling fi_freeinfo when it is no longer needed. The
fi_info connreq field will reference the connection request associated
with this event. To accept a connection, an endpoint must first be
created by passing an fi_info structure referencing this connreq field
to fi_endpoint(). This endpoint is then passed to fi_accept() to
complete the acceptance of the connection attempt. Creating the endpoint
is most easily accomplished by passing the fi_info returned as part of
the CM event into fi_endpoint(). If the connection is to be rejected,
the connreq is passed to fi_reject().

Any application data exchanged as part of the connection request is
placed beyond the fi_eq_cm_entry structure. The amount of data available
is application dependent and limited to the buffer space provided by the
application when fi_eq_read is called. The amount of returned data may
be calculated using the return value to fi_eq_read. Note that the amount
of returned data is limited by the underlying connection protocol, and
the length of any data returned may include protocol padding. As a
result, the returned length may be larger than that specified by the
connecting peer.

If a connection request has been accepted, an FI_CONNECTED event will be
generated on both sides of the connection. The active side -- one that
called fi_connect() -- may receive user data as part of the FI_CONNECTED
event. The user data is passed to the connection manager on the passive
side through the fi_accept call. User data is not provided with an
FI_CONNECTED event on the listening side of the connection.

Notification that a remote peer has disconnected from an active endpoint
is done through the FI_SHUTDOWN event. Shutdown notification uses struct
fi_eq_cm_entry as declared above. The fid field for a shutdown
notification refers to the active endpoint's fid_ep.

*Asynchronous Error Notification*
:   Asynchronous errors are used to report problems with fabric
    resources. Reported errors may be fatal or transient, based on the
    error, and result in the resource becoming disabled. Disabled
    resources will fail operations submitted against them until they are
    explicitly re-enabled by the application.

Asynchronous errors may be reported for completion queues and endpoints
of all types. CQ errors can result when resource management has been
disabled, and the provider has detected a queue overrun. Endpoint errors
may be result of numerous actions, but are often associated with a
failed operation. Operations may fail because of buffer overruns,
invalid permissions, incorrect memory access keys, network routing
failures, network reach-ability issues, etc.

Asynchronous errors are reported using struct fi_eq_err_entry, as
defined below. The fabric descriptor (fid) associated with the error is
provided as part of the error data. An error code is also available to
determine the cause of the error.

## fi_eq_sread

The fi_eq_sread call is the blocking (or synchronous) equivalent to
fi_eq_read. It behaves is similar to the non-blocking call, with the
exception that the calls will not return until either an event has been
read from the EQ or an error or timeout occurs. Specifying a negative
timeout means an infinite timeout.

Threads blocking in this function will return to the caller if they are
signaled by some external source. This is true even if the timeout has
not occurred or was specified as infinite.

It is invalid for applications to call this function if the EQ has been
configured with a wait object of FI_WAIT_NONE or FI_WAIT_SET.

## fi_eq_readerr

The read error function, fi_eq_readerr, retrieves information regarding
any asynchronous operation which has completed with an unexpected error.
fi_eq_readerr is a non-blocking call, returning immediately whether an
error completion was found or not.

EQs are optimized to report operations which have completed
successfully. Operations which fail are reported 'out of band'. Such
operations are retrieved using the fi_eq_readerr function. When an
operation that completes with an unexpected error is inserted into an
EQ, it is placed into a temporary error queue. Attempting to read from
an EQ while an item is in the error queue results in an FI_EAVAIL
failure. Applications may use this return code to determine when to call
fi_eq_readerr.

Error information is reported to the user through struct
fi_eq_err_entry. The format of this structure is defined below.

``` c
struct fi_eq_err_entry {
    fid_t            fid;        /* fid associated with error */
    void            *context;    /* operation context */
    uint64_t         data;       /* completion-specific data */
    int              err;        /* positive error code */
    int              prov_errno; /* provider error code */
    void            *err_data;   /* additional error data */
    size_t           err_data_size; /* size of err_data */
};
```

The fid will reference the fabric descriptor associated with the event.
For memory registration, this will be the fid_mr, address resolution
will reference a fid_av, and CM events will refer to a fid_ep. The
context field will be set to the context specified as part of the
operation.

The data field will be set as described in the man page for the
corresponding object type (e.g., see [`fi_av`(3)](fi_av.3.html) for a
description of how asynchronous address vector insertions are
completed).

The general reason for the error is provided through the err field.
Provider or operational specific error information may also be available
through the prov_errno and err_data fields. Users may call
fi_eq_strerror to convert provider specific error information into a
printable string for debugging purposes.

On input, err_data_size indicates the size of the err_data buffer in
bytes. On output, err_data_size will be set to the number of bytes
copied to the err_data buffer. The err_data information is typically
used with fi_eq_strerror to provide details about the type of error that
occurred.

For compatibility purposes, if err_data_size is 0 on input, or the
fabric was opened with release \< 1.5, err_data will be set to a data
buffer owned by the provider. The contents of the buffer will remain
valid until a subsequent read call against the EQ. Applications must
serialize access to the EQ when processing errors to ensure that the
buffer referenced by err_data does not change.

# EVENT FIELDS

The EQ entry data structures share many of the same fields. The meanings
are the same or similar for all EQ structure formats, with specific
details described below.

*fid*
:   This corresponds to the fabric descriptor associated with the event.
    The type of fid depends on the event being reported. For FI_CONNREQ
    this will be the fid of the passive endpoint. FI_CONNECTED and
    FI_SHUTDOWN will reference the active endpoint. FI_MR_COMPLETE and
    FI_AV_COMPLETE will refer to the MR or AV fabric descriptor,
    respectively. FI_JOIN_COMPLETE will point to the multicast
    descriptor returned as part of the join operation. Applications can
    use fid-\>context value to retrieve the context associated with the
    fabric descriptor.

*context*
:   The context value is set to the context parameter specified with the
    operation that generated the event. If no context parameter is
    associated with the operation, this field will be NULL.

*data*
:   Data is an operation specific value or set of bytes. For connection
    events, data is application data exchanged as part of the connection
    protocol.

*err*
:   This err code is a positive fabric errno associated with an event.
    The err value indicates the general reason for an error, if one
    occurred. See fi_errno.3 for a list of possible error codes.

*prov_errno*
:   On an error, prov_errno may contain a provider specific error code.
    The use of this field and its meaning is provider specific. It is
    intended to be used as a debugging aid. See fi_eq_strerror for
    additional details on converting this error value into a human
    readable string.

*err_data*
:   On an error, err_data may reference a provider specific amount of
    data associated with an error. The use of this field and its meaning
    is provider specific. It is intended to be used as a debugging aid.
    See fi_eq_strerror for additional details on converting this error
    data into a human readable string.

*err_data_size*
:   On input, err_data_size indicates the size of the err_data buffer in
    bytes. On output, err_data_size will be set to the number of bytes
    copied to the err_data buffer. The err_data information is typically
    used with fi_eq_strerror to provide details about the type of error
    that occurred.

For compatibility purposes, if err_data_size is 0 on input, or the
fabric was opened with release \< 1.5, err_data will be set to a data
buffer owned by the provider. The contents of the buffer will remain
valid until a subsequent read call against the EQ. Applications must
serialize access to the EQ when processing errors to ensure that the
buffer referenced by err_data does no change.

# NOTES

If an event queue has been overrun, it will be placed into an 'overrun'
state. Write operations against an overrun EQ will fail with
-FI_EOVERRUN. Read operations will continue to return any valid,
non-corrupted events, if available. After all valid events have been
retrieved, any attempt to read the EQ will result in it returning an
FI_EOVERRUN error event. Overrun event queues are considered fatal and
may not be used to report additional events once the overrun occurs.

# RETURN VALUES

fi_eq_open
:   Returns 0 on success. On error, a negative value corresponding to
    fabric errno is returned.

fi_eq_read / fi_eq_readerr
:   On success, returns the number of bytes read from the event queue.
    On error, a negative value corresponding to fabric errno is
    returned. If no data is available to be read from the event queue,
    -FI_EAGAIN is returned.

fi_eq_sread
:   On success, returns the number of bytes read from the event queue.
    On error, a negative value corresponding to fabric errno is
    returned. If the timeout expires or the calling thread is signaled
    and no data is available to be read from the event queue, -FI_EAGAIN
    is returned.

fi_eq_write
:   On success, returns the number of bytes written to the event queue.
    On error, a negative value corresponding to fabric errno is
    returned.

fi_eq_strerror
:   Returns a character string interpretation of the provider specific
    error returned with a completion.

Fabric errno values are defined in `rdma/fi_errno.h`.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cntr`(3)](fi_cntr.3.html),
[`fi_poll`(3)](fi_poll.3.html)

{% include JB/setup %}

# NAME

fi_errno - fabric errors

fi_strerror - Convert fabric error into a printable string

# SYNOPSIS

``` c
#include <rdma/fi_errno.h>

const char *fi_strerror(int errno);
```

# ERRORS

*FI_ENOENT*
:   No such file or directory.

*FI_EIO*
:   I/O error

*FI_E2BIG*
:   Argument list too long.

*FI_EBADF*
:   Bad file number.

*FI_EAGAIN*
:   Try again.

*FI_ENOMEM*
:   Out of memory.

*FI_EACCES*
:   Permission denied.

*FI_EBUSY*
:   Device or resource busy

*FI_ENODEV*
:   No such device

*FI_EINVAL*
:   Invalid argument

*FI_EMFILE*
:   Too many open files

*FI_ENOSPC*
:   No space left on device

*FI_ENOSYS*
:   Function not implemented

*FI_EWOULDBLOCK*
:   Operation would block

*FI_ENOMSG*
:   No message of desired type

*FI_ENODATA*
:   No data available

*FI_EOVERFLOW*
:   Value too large for defined data type

*FI_EMSGSIZE*
:   Message too long

*FI_ENOPROTOOPT*
:   Protocol not available

*FI_EOPNOTSUPP*
:   Operation not supported on transport endpoint

*FI_EADDRINUSE*
:   Address already in use

*FI_EADDRNOTAVAIL*
:   Cannot assign requested address

*FI_ENETDOWN*
:   Network is down

*FI_ENETUNREACH*
:   Network is unreachable

*FI_ECONNABORTED*
:   Software caused connection abort

*FI_ECONNRESET*
:   Connection reset by peer

*FI_ENOBUFS*
:   No buffer space available

*FI_EISCONN*
:   Transport endpoint is already connected

*FI_ENOTCONN*
:   Transport endpoint is not connected

*FI_ESHUTDOWN*
:   Cannot send after transport endpoint shutdown

*FI_ETIMEDOUT*
:   Operation timed out

*FI_ECONNREFUSED*
:   Connection refused

*FI_EHOSTDOWN*
:   Host is down

*FI_EHOSTUNREACH*
:   No route to host

*FI_EALREADY*
:   Operation already in progress

*FI_EINPROGRESS*
:   Operation now in progress

*FI_EREMOTEIO*
:   Remote I/O error

*FI_ECANCELED*
:   Operation Canceled

*FI_ENOKEY*
:   Required key not available

*FI_EKEYREJECTED*
:   Key was rejected by service

*FI_EOTHER*
:   Unspecified error

*FI_ETOOSMALL*
:   Provided buffer is too small

*FI_EOPBADSTATE*
:   Operation not permitted in current state

*FI_EAVAIL*
:   Error available

*FI_EBADFLAGS*
:   Flags not supported

*FI_ENOEQ*
:   Missing or unavailable event queue

*FI_EDOMAIN*
:   Invalid resource domain

*FI_ENOCQ*
:   Missing or unavailable completion queue

*FI_ECRC*
:   CRC error

*FI_ETRUNC*
:   Truncation error

*FI_ENOKEY*
:   Required key not available

*FI_ENOAV*
:   Missing or unavailable address vector

*FI_EOVERRUN*
:   Queue has been overrun

*FI_ENORX*
:   Receiver not ready, no receive buffers available

*FI_ENOMR*
:   Memory registration limit exceeded

*FI_EFIREWALLADDR*
:   Host address unreachable due to firewall

# SEE ALSO

[`fabric`(7)](fabric.7.html)

{% include JB/setup %}

# NAME

fi_fabric - Fabric network operations

fi_fabric / fi_close
:   Open / close a fabric network

fi_tostr / fi_tostr_r
:   Convert fabric attributes, flags, and capabilities to printable
    string

# SYNOPSIS

``` c
#include <rdma/fabric.h>

int fi_fabric(struct fi_fabric_attr *attr,
    struct fid_fabric **fabric, void *context);

int fi_close(struct fid *fabric);

char * fi_tostr(const void *data, enum fi_type datatype);

char * fi_tostr_r(char *buf, size_t len, const void *data,
    enum fi_type datatype);
```

# ARGUMENTS

*attr*
:   Attributes of fabric to open.

*fabric*
:   Fabric network

*context*
:   User specified context associated with the opened object. This
    context is returned as part of any associated asynchronous event.

*buf*
:   Output buffer to write string.

*len*
:   Size in bytes of memory referenced by buf.

*data*
:   Input data to convert into a string. The format of data is
    determined by the datatype parameter.

*datatype*
:   Indicates the data to convert to a printable string.

# DESCRIPTION

A fabric identifier is used to reference opened fabric resources and
library related objects.

The fabric network represents a collection of hardware and software
resources that access a single physical or virtual network. All network
ports on a system that can communicate with each other through their
attached networks belong to the same fabric. A fabric network shares
network addresses and can span multiple providers. An application must
open a fabric network prior to allocating other network resources, such
as communication endpoints.

## fi_fabric

Opens a fabric network provider. The attributes of the fabric provider
are specified through the open call, and may be obtained by calling
fi_getinfo.

## fi_close

The fi_close call is used to release all resources associated with a
fabric object. All items associated with the opened fabric must be
released prior to calling fi_close.

## fi_tostr / fi_tostr_r

Converts fabric interface attributes, capabilities, flags, and enum
values into a printable string. The data parameter accepts a pointer to
the attribute or value(s) to display, with the datatype parameter
indicating the type of data referenced by the data parameter. Valid
values for the datatype are listed below, along with the corresponding
datatype or field value.

*FI_TYPE_INFO*
:   struct fi_info, including all substructures and fields

*FI_TYPE_EP_TYPE*
:   struct fi_info::type field

*FI_TYPE_EP_CAP*
:   struct fi_info::ep_cap field

*FI_TYPE_OP_FLAGS*
:   struct fi_info::op_flags field, or general uint64_t flags

*FI_TYPE_ADDR_FORMAT*
:   struct fi_info::addr_format field

*FI_TYPE_TX_ATTR*
:   struct fi_tx_attr

*FI_TYPE_RX_ATTR*
:   struct fi_rx_attr

*FI_TYPE_EP_ATTR*
:   struct fi_ep_attr

*FI_TYPE_DOMAIN_ATTR*
:   struct fi_domain_attr

*FI_TYPE_FABRIC_ATTR*
:   struct fi_fabric_attr

*FI_TYPE_THREADING*
:   enum fi_threading

*FI_TYPE_PROGRESS*
:   enum fi_progress

*FI_TYPE_PROTOCOL*
:   struct fi_ep_attr::protocol field

*FI_TYPE_MSG_ORDER*
:   struct fi_ep_attr::msg_order field

*FI_TYPE_MODE*
:   struct fi_info::mode field

*FI_TYPE_AV_TYPE*
:   enum fi_av_type

*FI_TYPE_ATOMIC_TYPE*
:   enum fi_datatype

*FI_TYPE_ATOMIC_OP*
:   enum fi_op

*FI_TYPE_VERSION*
:   Returns the library version of libfabric in string form. The data
    parameter is ignored.

*FI_TYPE_EQ_EVENT*
:   uint32_t event parameter returned from fi_eq_read(). See `fi_eq(3)`
    for a list of known values.

*FI_TYPE_CQ_EVENT_FLAGS*
:   uint64_t flags field in fi_cq_xxx_entry structures. See `fi_cq(3)`
    for valid flags.

*FI_TYPE_MR_MODE*
:   struct fi_domain_attr::mr_mode flags

*FI_TYPE_OP_TYPE*
:   enum fi_op_type

*FI_TYPE_FID*
:   struct fid \*

*FI_TYPE_HMEM_IFACE*
:   enum fi_hmem_iface \*

*FI_TYPE_CQ_FORMAT*
:   enum fi_cq_format

*FI_TYPE_LOG_LEVEL*
:   enum fi_log_level

*FI_TYPE_LOG_SUBSYS*
:   enum fi_log_subsys

fi_tostr() will return a pointer to an internal libfabric buffer that
should not be modified, and will be overwritten the next time fi_tostr()
is invoked. fi_tostr() is not thread safe.

The fi_tostr_r() function is a re-entrant and thread safe version of
fi_tostr(). It writes the string into a buffer provided by the caller.
fi_tostr_r() returns the start of the caller's buffer.

# NOTES

The following resources are associated with fabric domains: access
domains, passive endpoints, and CM event queues.

# FABRIC ATTRIBUTES

The fi_fabric_attr structure defines the set of attributes associated
with a fabric and a fabric provider.

``` c
struct fi_fabric_attr {
    struct fid_fabric *fabric;
    char              *name;
    char              *prov_name;
    uint32_t          prov_version;
    uint32_t          api_version;
};
```

## fabric

On input to fi_getinfo, a user may set this to an opened fabric instance
to restrict output to the given fabric. On output from fi_getinfo, if no
fabric was specified, but the user has an opened instance of the named
fabric, this will reference the first opened instance. If no instance
has been opened, this field will be NULL.

The fabric instance returned by fi_getinfo should only be considered
valid if the application does not close any fabric instances from
another thread while fi_getinfo is being processed.

## name

A fabric identifier.

## prov_name - Provider Name

The name of the underlying fabric provider.

To request an utility provider layered over a specific core provider,
both the provider names have to be specified using ";" as delimiter.

e.g. "ofi_rxm;verbs" or "verbs;ofi_rxm"

For debugging and administrative purposes, environment variables can be
used to control which fabric providers will be registered with
libfabric. Specifying "FI_PROVIDER=foo,bar" will allow any providers
with the names "foo" or "bar" to be registered. Similarly, specifying
"FI_PROVIDER=\^foo,bar" will prevent any providers with the names "foo"
or "bar" from being registered. Providers which are not registered will
not appear in fi_getinfo results. Applications which need a specific set
of providers should implement their own filtering of fi_getinfo's
results rather than relying on these environment variables in a
production setting.

## prov_version - Provider Version

Version information for the fabric provider, in a major.minor format.
The use of the FI_MAJOR() and FI_MINOR() version macros may be used to
extract the major and minor version data. See `fi_version(3)`.

In case of an utility provider layered over a core provider, the version
would always refer to that of the utility provider.

## api_version

The interface version requested by the application. This value
corresponds to the version parameter passed into `fi_getinfo(3)`.

# RETURN VALUE

Returns FI_SUCCESS on success. On error, a negative value corresponding
to fabric errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# ERRORS

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_eq`(3)](fi_eq.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html)

{% include JB/setup %}

# NAME

fi_getinfo, fi_freeinfo - Obtain / free fabric interface information

fi_allocinfo, fi_dupinfo - Allocate / duplicate an fi_info structure

# SYNOPSIS

``` c
#include <rdma/fabric.h>

int fi_getinfo(int version, const char *node, const char *service,
        uint64_t flags, const struct fi_info *hints, struct fi_info **info);

void fi_freeinfo(struct fi_info *info);

struct fi_info *fi_allocinfo(void);

struct fi_info *fi_dupinfo(const struct fi_info *info);
```

# ARGUMENTS

*version*
:   Interface version requested by application.

*node*
:   Optional, name or fabric address to resolve.

*service*
:   Optional, service name or port number of address.

*flags*
:   Operation flags for the fi_getinfo call.

*hints*
:   Reference to an fi_info structure that specifies criteria for
    selecting the returned fabric information. The fi_info hints
    structure must be allocated using either fi_allocinfo() or
    fi_dupinfo().

*info*
:   A pointer to a linked list of fi_info structures containing response
    information.

# DESCRIPTION

The fi_getinfo() call is used to discover what communication features
are available in the system, as well as how they might best be used by
an application. The call is loosely modeled on getaddrinfo().
fi_getinfo() permits an application to exchange information between an
application and the libfabric providers regarding its required set of
communication. It provides the ability to access complex network
details, balanced between being expressive but also simple to use.

fi_getinfo returns information about available fabric services for
reaching a specified node or service, subject to any provided hints.
Callers may specify NULL for node, service, and hints in order to
retrieve information about what providers are available and their
optimal usage models. If no matching fabric information is available,
info will be set to NULL and the call will return -FI_ENODATA.

Based on the input hints, node, and service parameters, a list of fabric
domains and endpoints will be returned. Each fi_info structure will
describe an endpoint that meets the application's specified
communication criteria. Each endpoint will be associated with a domain.
Applications can restrict the number of returned endpoints by including
additional criteria in their search hints. Relaxing or eliminating input
hints will increase the number and type of endpoints that are available.
Providers that return multiple endpoints to a single fi_getinfo call
should return the endpoints that are highest performing first. Providers
may indicate that an endpoint and domain can support additional
capabilities than those requested by the user only if such support will
not adversely affect application performance or security.

The version parameter is used by the application to request the desired
version of the interfaces. The version determines the format of all data
structures used by any of the fabric interfaces. Applications should use
the FI_VERSION(major, minor) macro to indicate the version, with
hard-coded integer values for the major and minor values. The
FI_MAJOR_VERSION and FI_MINOR_VERSION enum values defined in fabric.h
specify the latest version of the installed library. However, it is
recommended that the integer values for FI_MAJOR_VERSION and
FI_MINOR_VERSION be used, rather than referencing the enum types in
order to ensure compatibility with future versions of the library. This
protects against the application being built from source against a newer
version of the library that introduces new fields to data structures,
which would not be initialized by the application.

Node, service, or hints may be provided, with any combination being
supported. If node is provided, fi_getinfo will attempt to resolve the
fabric address to the given node. If node is not given, fi_getinfo will
attempt to resolve the fabric addressing information based on the
provided hints. Node is commonly used to provide a network address (such
as an IP address) or hostname. Service is usually associated with a
transport address (such as a TCP port number). Node and service
parameters may be mapped by providers to native fabric addresses.
Applications may also pass in an FI_ADDR_STR formatted address (see
format details below) as the node parameter. In such cases, the service
parameter must be NULL.

The hints parameter, if provided, may be used to limit the resulting
output as indicated below. As a general rule, specifying a non-zero
value for input hints indicates that a provider must support the
requested value or fail the operation with -FI_ENODATA. With the
exception of mode bits, hints that are set to zero are treated as a
wildcard. A zeroed hint value results in providers either returning a
default value or a value that works best for their implementation. Mode
bits that are set to zero indicate the application does not support any
modes.

The caller must call fi_freeinfo to release fi_info structures returned
by this call.

The fi_allocinfo call will allocate and zero an fi_info structure and
all related substructures. The fi_dupinfo will duplicate a single
fi_info structure and all the substructures within it.

# FI_INFO

``` c
struct fi_info {
    struct fi_info        *next;
    uint64_t              caps;
    uint64_t              mode;
    uint32_t              addr_format;
    size_t                src_addrlen;
    size_t                dest_addrlen;
    void                  *src_addr;
    void                  *dest_addr;
    fid_t                 handle;
    struct fi_tx_attr     *tx_attr;
    struct fi_rx_attr     *rx_attr;
    struct fi_ep_attr     *ep_attr;
    struct fi_domain_attr *domain_attr;
    struct fi_fabric_attr *fabric_attr;
    struct fid_nic        *nic;
};
```

*next*
:   Pointer to the next fi_info structure in the list. Will be NULL if
    no more structures exist.

*caps - fabric interface capabilities*
:   If specified, indicates the desired capabilities of the fabric
    interfaces. Supported capabilities are listed in the *Capabilities*
    section below.

*mode*
:   Operational modes supported by the application. See the *Mode*
    section below.

*addr_format - address format*
:   If specified, indicates the format of addresses referenced by the
    fabric interfaces and data structures. Supported formats are listed
    in the *Addressing formats* section below.

*src_addrlen - source address length*
:   Indicates the length of the source address. This value must be \> 0
    if *src_addr* is non-NULL. This field will be ignored in hints if
    FI_SOURCE flag is set, or *src_addr* is NULL.

*dest_addrlen - destination address length*
:   Indicates the length of the destination address. This value must be
    \> 0 if *dest_addr* is non-NULL. This field will be ignored in hints
    unless the node and service parameters are NULL or FI_SOURCE flag is
    set, or if *dst_addr* is NULL.

*src_addr - source address*
:   If specified, indicates the source address. This field will be
    ignored in hints if FI_SOURCE flag is set. On output a provider
    shall return an address that corresponds to the indicated fabric,
    domain, node, and/or service fields. The format of the address is
    indicated by the returned *addr_format* field. Note that any
    returned address is only used when opening a local endpoint. The
    address is not guaranteed to be usable by a peer process.

*dest_addr - destination address*
:   If specified, indicates the destination address. This field will be
    ignored in hints unless the node and service parameters are NULL or
    FI_SOURCE flag is set. If FI_SOURCE is not specified, on output a
    provider shall return an address the corresponds to the indicated
    node and/or service fields, relative to the fabric and domain. Note
    that any returned address is only usable locally.

*handle - provider context handle*
:   The use of this field is operation specific. If hints-\>handle is
    set to struct fid_pep, the hints-\>handle will be copied to
    info-\>handle on output from fi_getinfo. Other values of
    hints-\>handle will be handled in a provider specific manner. The
    fi_info::handle field is also used by fi_endpoint() and fi_reject()
    calls when processing connection requests or to inherit another
    endpoint's attributes. See [`fi_eq`(3)](fi_eq.3.html),
    [`fi_reject`(3)](fi_reject.3.html), and
    [`fi_endpoint`(3)](fi_endpoint.3.html). The info-\>handle field will
    be ignored by fi_dupinfo and fi_freeinfo.

*tx_attr - transmit context attributes*
:   Optionally supplied transmit context attributes. Transmit context
    attributes may be specified and returned as part of fi_getinfo. When
    provided as hints, requested values of struct fi_tx_ctx_attr should
    be set. On output, the actual transmit context attributes that can
    be provided will be returned. Output values will be greater than or
    equal to the requested input values.

*rx_attr - receive context attributes*
:   Optionally supplied receive context attributes. Receive context
    attributes may be specified and returned as part of fi_getinfo. When
    provided as hints, requested values of struct fi_rx_ctx_attr should
    be set. On output, the actual receive context attributes that can be
    provided will be returned. Output values will be greater than or or
    equal to the requested input values.

*ep_attr - endpoint attributes*
:   Optionally supplied endpoint attributes. Endpoint attributes may be
    specified and returned as part of fi_getinfo. When provided as
    hints, requested values of struct fi_ep_attr should be set. On
    output, the actual endpoint attributes that can be provided will be
    returned. Output values will be greater than or equal to requested
    input values. See [`fi_endpoint`(3)](fi_endpoint.3.html) for
    details.

*domain_attr - domain attributes*
:   Optionally supplied domain attributes. Domain attributes may be
    specified and returned as part of fi_getinfo. When provided as
    hints, requested values of struct fi_domain_attr should be set. On
    output, the actual domain attributes that can be provided will be
    returned. Output values will be greater than or equal to requested
    input values. See [`fi_domain`(3)](fi_domain.3.html) for details.

*fabric_attr - fabric attributes*
:   Optionally supplied fabric attributes. Fabric attributes may be
    specified and returned as part of fi_getinfo. When provided as
    hints, requested values of struct fi_fabric_attr should be set. On
    output, the actual fabric attributes that can be provided will be
    returned. See [`fi_fabric`(3)](fi_fabric.3.html) for details.

*nic - network interface details*
:   Optional attributes related to the hardware NIC associated with the
    specified fabric, domain, and endpoint data. This field is only
    valid for providers where the corresponding attributes are closely
    associated with a hardware NIC. See [`fi_nic`(3)](fi_nic.3.html) for
    details.

# CAPABILITIES

Interface capabilities are obtained by OR-ing the following flags
together. If capabilities in the hint parameter are set to 0, the
underlying provider will return the set of capabilities which are
supported. Otherwise, providers will return data matching the specified
set of capabilities. Providers may indicate support for additional
capabilities beyond those requested when the use of expanded
capabilities will not adversely affect performance or expose the
application to communication beyond that which was requested.
Applications may use this feature to request a minimal set of
requirements, then check the returned capabilities to enable additional
optimizations.

*FI_ATOMIC*
:   Specifies that the endpoint supports some set of atomic operations.
    Endpoints supporting this capability support operations defined by
    struct fi_ops_atomic. In the absence of any relevant flags,
    FI_ATOMIC implies the ability to initiate and be the target of
    remote atomic reads and writes. Applications can use the FI_READ,
    FI_WRITE, FI_REMOTE_READ, and FI_REMOTE_WRITE flags to restrict the
    types of atomic operations supported by an endpoint.

*FI_AV_USER_ID*
:   Requests that the provider support the association of a user
    specified identifier with each address vector (AV) address. User
    identifiers are returned with completion data in place of the AV
    address. See [`fi_domain`(3)](fi_domain.3.html) and
    [`fi_av`(3)](fi_av.3.html) for more details.

*FI_COLLECTIVE*
:   Requests support for collective operations. Endpoints that support
    this capability support the collective operations defined in
    [`fi_collective`(3)](fi_collective.3.html).

*FI_DIRECTED_RECV*
:   Requests that the communication endpoint use the source address of
    an incoming message when matching it with a receive buffer. If this
    capability is not set, then the src_addr parameter for msg and
    tagged receive operations is ignored.

*FI_TAGGED_DIRECTED_RECV*
:   Similar to FI_DIRECTED_RECV, but only applies to tagged receive
    operations.

*FI_EXACT_DIRECTED_RECV*
:   Similar to FI_DIRECTED_RECV, but requires the source address to be
    exact, i.e., FI_ADDR_UNSPEC is not allowed. This capability can be
    used alone, or in conjunction with FI_DIRECTED_RECV or
    FI_TAGGED_DIRECTED_RECV as a modifier to disallow FI_ADDR_UNSPEC
    being used as the source address.

*FI_FENCE*
:   Indicates that the endpoint support the FI_FENCE flag on data
    transfer operations. Support requires tracking that all previous
    transmit requests to a specified remote endpoint complete prior to
    initiating the fenced operation. Fenced operations are often used to
    enforce ordering between operations that are not otherwise
    guaranteed by the underlying provider or protocol.

*FI_HMEM*
:   Specifies that the endpoint should support transfers to and from
    device memory.

*FI_LOCAL_COMM*
:   Indicates that the endpoint support host local communication. This
    flag may be used in conjunction with FI_REMOTE_COMM to indicate that
    local and remote communication are required. If neither
    FI_LOCAL_COMM or FI_REMOTE_COMM are specified, then the provider
    will indicate support for the configuration that minimally affects
    performance. Providers that set FI_LOCAL_COMM but not
    FI_REMOTE_COMM, for example a shared memory provider, may only be
    used to communication between processes on the same system.

*FI_MSG*
:   Specifies that an endpoint should support sending and receiving
    messages or datagrams. Message capabilities imply support for send
    and/or receive queues. Endpoints supporting this capability support
    operations defined by struct fi_ops_msg.

The caps may be used to specify or restrict the type of messaging
operations that are supported. In the absence of any relevant flags,
FI_MSG implies the ability to send and receive messages. Applications
can use the FI_SEND and FI_RECV flags to optimize an endpoint as
send-only or receive-only.

*FI_MULTICAST*
:   Indicates that the endpoint support multicast data transfers. This
    capability must be paired with FI_MSG. Applications can use FI_SEND
    and FI_RECV to optimize multicast as send-only or receive-only.

*FI_MULTI_RECV*
:   Specifies that the endpoint must support the FI_MULTI_RECV flag when
    posting receive buffers.

*FI_TAGGED_MULTI_RECV*
:   Specifies that the endpoint must support the FI_MULTI_RECV flag when
    posting tagged receive buffers.

*FI_NAMED_RX_CTX*
:   Requests that endpoints which support multiple receive contexts
    allow an initiator to target (or name) a specific receive context as
    part of a data transfer operation.

*FI_PEER*
:   Specifies that the provider must support being used as a peer
    provider in the peer API flow. The provider must support importing
    owner_ops when opening a CQ, counter, and shared receive queue.

*FI_READ*
:   Indicates that the user requires an endpoint capable of initiating
    reads against remote memory regions. This flag requires that FI_RMA
    and/or FI_ATOMIC be set.

*FI_RECV*
:   Indicates that the user requires an endpoint capable of receiving
    message data transfers. Message transfers include base message
    operations as well as tagged message functionality.

*FI_REMOTE_COMM*
:   Indicates that the endpoint support communication with endpoints
    located at remote nodes (across the fabric). See FI_LOCAL_COMM for
    additional details. Providers that set FI_REMOTE_COMM but not
    FI_LOCAL_COMM, for example NICs that lack loopback support, cannot
    be used to communicate with processes on the same system.

*FI_REMOTE_READ*
:   Indicates that the user requires an endpoint capable of receiving
    read memory operations from remote endpoints. This flag requires
    that FI_RMA and/or FI_ATOMIC be set.

*FI_REMOTE_WRITE*
:   Indicates that the user requires an endpoint capable of receiving
    write memory operations from remote endpoints. This flag requires
    that FI_RMA and/or FI_ATOMIC be set.

*FI_RMA*
:   Specifies that the endpoint should support RMA read and write
    operations. Endpoints supporting this capability support operations
    defined by struct fi_ops_rma. In the absence of any relevant flags,
    FI_RMA implies the ability to initiate and be the target of remote
    memory reads and writes. Applications can use the FI_READ, FI_WRITE,
    FI_REMOTE_READ, and FI_REMOTE_WRITE flags to restrict the types of
    RMA operations supported by an endpoint.

*FI_RMA_EVENT*
:   Requests that an endpoint support the generation of completion
    events when it is the target of an RMA and/or atomic operation. This
    flag requires that FI_REMOTE_READ and/or FI_REMOTE_WRITE be enabled
    on the endpoint.

*FI_RMA_PMEM*
:   Indicates that the provider is 'persistent memory aware' and
    supports RMA operations to and from persistent memory. Persistent
    memory aware providers must support registration of memory that is
    backed by non- volatile memory, RMA transfers to/from persistent
    memory, and enhanced completion semantics. This flag requires that
    FI_RMA be set. This capability is experimental.

*FI_SEND*
:   Indicates that the user requires an endpoint capable of sending
    message data transfers. Message transfers include base message
    operations as well as tagged message functionality.

*FI_SHARED_AV*
:   Requests or indicates support for address vectors which may be
    shared among multiple processes.

*FI_SOURCE*
:   Requests that the endpoint return source addressing data as part of
    its completion data. This capability only applies to connectionless
    endpoints. Note that returning source address information may
    require that the provider perform address translation and/or look-up
    based on data available in the underlying protocol in order to
    provide the requested data, which may adversely affect performance.
    The performance impact may be greater for address vectors of type
    FI_AV_TABLE.

*FI_SOURCE_ERR*
:   Must be paired with FI_SOURCE. When specified, this requests that
    raw source addressing data be returned as part of completion data
    for any address that has not been inserted into the local address
    vector. Use of this capability may require the provider to validate
    incoming source address data against addresses stored in the local
    address vector, which may adversely affect performance.

*FI_TAGGED*
:   Specifies that the endpoint should handle tagged message transfers.
    Tagged message transfers associate a user-specified key or tag with
    each message that is used for matching purposes at the remote side.
    Endpoints supporting this capability support operations defined by
    struct fi_ops_tagged. In the absence of any relevant flags,
    FI_TAGGED implies the ability to send and receive tagged messages.
    Applications can use the FI_SEND and FI_RECV flags to optimize an
    endpoint as send-only or receive-only.

*FI_TRIGGER*
:   Indicates that the endpoint should support triggered operations.
    Endpoints support this capability must meet the usage model as
    described by [`fi_trigger`(3)](fi_trigger.3.html).

*FI_WRITE*
:   Indicates that the user requires an endpoint capable of initiating
    writes against remote memory regions. This flag requires that FI_RMA
    and/or FI_ATOMIC be set.

*FI_XPU*
:   Specifies that the endpoint should support transfers that may be
    initiated from heterogenous computation devices, such as GPUs. This
    flag requires that FI_TRIGGER be set. For additional details on XPU
    triggers see [`fi_trigger`(3)](fi_trigger.3.html).

Capabilities may be grouped into three general categories: primary,
secondary, and primary modifiers. Primary capabilities must explicitly
be requested by an application, and a provider must enable support for
only those primary capabilities which were selected. Primary modifiers
are used to limit a primary capability, such as restricting an endpoint
to being send-only. If no modifiers are specified for an applicable
capability, all relevant modifiers are assumed. See above definitions
for details.

Secondary capabilities may optionally be requested by an application. If
requested, a provider must support the capability or fail the fi_getinfo
request (FI_ENODATA). A provider may optionally report non-selected
secondary capabilities if doing so would not compromise performance or
security.

Primary capabilities: FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC,
FI_MULTICAST, FI_NAMED_RX_CTX, FI_DIRECTED_RECV,
FI_TAGGED_DIRECTED_RECV, FI_HMEM, FI_COLLECTIVE, FI_XPU, FI_AV_USER_ID,
FI_PEER

Primary modifiers: FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ,
FI_REMOTE_WRITE

Secondary capabilities: FI_MULTI_RECV, FI_TAGGED_MULTI_RECV, FI_SOURCE,
FI_RMA_EVENT, FI_SHARED_AV, FI_TRIGGER, FI_FENCE, FI_LOCAL_COMM,
FI_REMOTE_COMM, FI_SOURCE_ERR, FI_RMA_PMEM.

# MODE

The operational mode bits are used to convey requirements that an
application must adhere to when using the fabric interfaces. Modes
specify optimal ways of accessing the reported endpoint or domain.
Applications that are designed to support a specific mode of operation
may see improved performance when that mode is desired by the provider.
It is recommended that providers support applications that disable any
provider preferred modes.

On input to fi_getinfo, applications set the mode bits that they
support. On output, providers will clear mode bits that are not
necessary to achieve high-performance. Mode bits that remain set
indicate application requirements for using the fabric interfaces
created using the returned fi_info. The set of modes are listed below.
If a NULL hints structure is provided, then the provider's supported set
of modes will be returned in the info structure(s).

*FI_ASYNC_IOV*
:   Applications can reference multiple data buffers as part of a single
    operation through the use of IO vectors (SGEs). Typically, the
    contents of an IO vector are copied by the provider into an internal
    buffer area, or directly to the underlying hardware. However, when a
    large number of IOV entries are supported, IOV buffering may have a
    negative impact on performance and memory consumption. The
    FI_ASYNC_IOV mode indicates that the application must provide the
    buffering needed for the IO vectors. When set, an application must
    not modify an IO vector of length \> 1, including any related memory
    descriptor array, until the associated operation has completed.

*FI_CONTEXT*
:   Specifies that the provider requires that applications use struct
    fi_context as their per operation context parameter for operations
    that generated full completions. This structure should be treated as
    opaque to the application. For performance reasons, this structure
    must be allocated by the user, but may be used by the fabric
    provider to track the operation. Typically, users embed struct
    fi_context within their own context structure. The struct fi_context
    must remain valid until the corresponding operation completes or is
    successfully canceled. As such, fi_context should NOT be allocated
    on the stack. Doing so is likely to result in stack corruption that
    will be difficult to debug. Users should not update or interpret the
    fields in this structure, or reuse it until the original operation
    has completed. If an operation does not generate a completion
    (i.e. the endpoint was configured with FI_SELECTIVE_COMPLETION and
    the operation was not initiated with the FI_COMPLETION flag) then
    the context parameter is ignored by the fabric provider. The
    structure is specified in rdma/fabric.h.

*FI_CONTEXT2*
:   This bit is similar to FI_CONTEXT, but doubles the provider's
    requirement on the size of the per context structure. When set, this
    specifies that the provider requires that applications use struct
    fi_context2 as their per operation context parameter. Or,
    optionally, an application can provide an array of two fi_context
    structures (e.g. struct fi_context\[2\]) instead. The requirements
    for using struct fi_context2 are identical as defined for FI_CONTEXT
    above.

*FI_LOCAL_MR* (deprecated)
:   The provider is optimized around having applications register memory
    for locally accessed data buffers. Data buffers used in send and
    receive operations and as the source buffer for RMA and atomic
    operations must be registered by the application for access domains
    opened with this capability. This flag is defined for compatibility
    and is ignored if the application version is 1.5 or later and the
    domain mr_mode is set to anything other than FI_MR_BASIC or
    FI_MR_SCALABLE. See the domain attribute mr_mode
    [`fi_domain`(3)](fi_domain.3.html) and [`fi_mr`(3)](fi_mr.3.html).

*FI_MSG_PREFIX*
:   Message prefix mode indicates that an application will provide
    buffer space in front of all message send and receive buffers for
    use by the provider. Typically, the provider uses this space to
    implement a protocol, with the protocol headers being written into
    the prefix area. The contents of the prefix space should be treated
    as opaque. The use of FI_MSG_PREFIX may improve application
    performance over certain providers by reducing the number of IO
    vectors referenced by underlying hardware and eliminating provider
    buffer allocation.

FI_MSG_PREFIX only applies to send and receive operations, including
tagged sends and receives. RMA and atomics do not require the
application to provide prefix buffers. Prefix buffer space must be
provided with all sends and receives, regardless of the size of the
transfer or other transfer options. The ownership of prefix buffers is
treated the same as the corresponding message buffers, but the size of
the prefix buffer is not counted toward any message limits, including
inject.

Applications that support prefix mode must supply buffer space before
their own message data. The size of space that must be provided is
specified by the msg_prefix_size endpoint attribute. Providers are
required to define a msg_prefix_size that is a multiple of 8 bytes.
Additionally, applications may receive provider generated packets that
do not contain application data. Such received messages will indicate a
transfer size of that is equal to or smaller than msg_prefix_size.

The buffer pointer given to all send and receive operations must point
to the start of the prefix region of the buffer (as opposed to the
payload). For scatter-gather send/recv operations, the prefix buffer
must be a contiguous region, though it may or may not be directly
adjacent to the payload portion of the buffer.

*FI_RX_CQ_DATA*
:   This mode bit only applies to data transfers that set
    FI_REMOTE_CQ_DATA. When set, a data transfer that carries remote CQ
    data will consume a receive buffer at the target. This is true even
    for operations that would normally not consume posted receive
    buffers, such as RMA write operations.

# ADDRESSING FORMATS

Multiple fabric interfaces take as input either a source or destination
address parameter. This includes struct fi_info (src_addr and
dest_addr), CM calls (getname, getpeer, connect, join, and leave), and
AV calls (insert, lookup, and straddr). The fi_info addr_format field
indicates the expected address format for these operations.

A provider may support one or more of the following addressing formats.
In some cases, a selected addressing format may need to be translated or
mapped into an address which is native to the fabric. See
[`fi_av`(3)](fi_av.3.html).

*FI_ADDR_EFA*
:   Address is an Amazon Elastic Fabric Adapter (EFA) proprietary
    format.

*FI_ADDR_PSMX2*
:   Address is an Intel proprietary format used with their Performance
    Scaled Messaging protocol version 2.

*FI_ADDR_PSMX3*
:   Address is an Intel proprietary format used with their Performance
    Scaled Messaging protocol version 3.

*FI_ADDR_STR*
:   Address is a formatted character string. The length and content of
    the string is address and/or provider specific, but in general
    follows a URI model:

```{=html}
<!-- -->
```
    address_format[://[node][:[service][/[field3]...][?[key=value][&k2=v2]...]]]

Examples: - fi_sockaddr://10.31.6.12:7471 -
fi_sockaddr_in6://\[fe80::6:12\]:7471 -
fi_sockaddr://10.31.6.12:7471?qos=3

Since the string formatted address does not contain any provider
information, the prov_name field of the fabric attribute structure
should be used to filter by provider if necessary.

*FI_FORMAT_UNSPEC*
:   FI_FORMAT_UNSPEC indicates that a provider specific address format
    should be selected. Provider specific addresses may be protocol
    specific or a vendor proprietary format. Applications that select
    FI_FORMAT_UNSPEC should be prepared to treat returned addressing
    data as opaque. FI_FORMAT_UNSPEC targets apps which make use of an
    out of band address exchange. Applications which use
    FI_FORMAT_UNSPEC may use fi_getname() to obtain a provider specific
    address assigned to an allocated endpoint.

*FI_SOCKADDR*
:   Address is of type sockaddr. The specific socket address format will
    be determined at run time by interfaces examining the sa_family
    field.

*FI_SOCKADDR_IB*
:   Address is of type sockaddr_ib (defined in Linux kernel source)

*FI_SOCKADDR_IN*
:   Address is of type sockaddr_in (IPv4).

*FI_SOCKADDR_IN6*
:   Address is of type sockaddr_in6 (IPv6).

# FLAGS

The operation of the fi_getinfo call may be controlled through the use
of input flags. Valid flags include the following.

*FI_NUMERICHOST*
:   Indicates that the node parameter is a numeric string representation
    of a fabric address, such as a dotted decimal IP address. Use of
    this flag will suppress any lengthy name resolution protocol.

*FI_PROV_ATTR_ONLY*
:   Indicates that the caller is only querying for what providers are
    potentially available. All providers will return exactly one fi_info
    struct, regardless of whether that provider is usable on the current
    platform or not. The returned fi_info struct will contain default
    values for all members, with the exception of fabric_attr. The
    fabric_attr member will have the prov_name and prov_version values
    filled in.

*FI_SOURCE*
:   Indicates that the node and service parameters specify the local
    source address to associate with an endpoint. If specified, either
    the node and/or service parameter must be non-NULL. This flag is
    often used with passive endpoints.

*FI_RESCAN*
:   Indicates that the provider should rescan available network
    interfaces. This operation may be computationally expensive.

# RETURN VALUE

fi_getinfo() returns 0 on success. On error, fi_getinfo() returns a
negative value corresponding to fabric errno. Fabric errno values are
defined in `rdma/fi_errno.h`.

fi_allocinfo() returns a pointer to a new fi_info structure on success,
or NULL on error. fi_dupinfo() duplicates a single fi_info structure and
all the substructures within it, returning a pointer to the new fi_info
structure on success, or NULL on error. Both calls require that the
returned fi_info structure be freed via fi_freeinfo().

# ERRORS

*FI_EBADFLAGS*
:   The specified endpoint or domain capability or operation flags are
    invalid.

*FI_ENODATA*
:   Indicates that no providers could be found which support the
    requested fabric information.

*FI_ENOMEM*
:   Indicates that there was insufficient memory to complete the
    operation.

*FI_ENOSYS*
:   Indicates that requested version is newer than the library being
    used.

# NOTES

Various libfabric calls, including fi_getinfo, take a struct fi_info as
input. Applications must use libfabric allocated fi_info structures. A
zeroed struct fi_info can be allocated using fi_allocinfo, which may
then be initialized by the user. A struct fi_info may be copied for
modification using the fi_dupinfo() call.

If hints are provided, the operation will be controlled by the values
that are supplied in the various fields (see section on *fi_info*).
Applications that require specific communication interfaces, domains,
capabilities or other requirements, can specify them using fields in
*hints*. Libfabric returns a linked list in *info* that points to a list
of matching interfaces. *info* is set to NULL if there are no
communication interfaces or none match the input hints.

If node is provided, fi_getinfo will attempt to resolve the fabric
address to the given node. If node is not provided, fi_getinfo will
attempt to resolve the fabric addressing information based on the
provided hints. The caller must call fi_freeinfo to release fi_info
structures returned by fi_getinfo.

If neither node, service or hints are provided, then fi_getinfo simply
returns the list all available communication interfaces.

Multiple threads may call `fi_getinfo` simultaneously, without any
requirement for serialization.

# SEE ALSO

[`fi_open`(3)](fi_open.3.html), [`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_nic`(3)](fi_nic.3.html)
[`fi_trigger`(3)](fi_trigger.3.html)

{% include JB/setup %}

# NAME

fi_msg - Message data transfer operations

fi_recv / fi_recvv / fi_recvmsg
:   Post a buffer to receive an incoming message

fi_send / fi_sendv / fi_sendmsg fi_inject / fi_senddata : Initiate an
operation to send a message

# SYNOPSIS

``` c
#include <rdma/fi_endpoint.h>

ssize_t fi_recv(struct fid_ep *ep, void * buf, size_t len,
    void *desc, fi_addr_t src_addr, void *context);

ssize_t fi_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
    size_t count, fi_addr_t src_addr, void *context);

ssize_t fi_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
    uint64_t flags);

ssize_t fi_send(struct fid_ep *ep, const void *buf, size_t len,
    void *desc, fi_addr_t dest_addr, void *context);

ssize_t fi_sendv(struct fid_ep *ep, const struct iovec *iov,
    void **desc, size_t count, fi_addr_t dest_addr, void *context);

ssize_t fi_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
    uint64_t flags);

ssize_t fi_inject(struct fid_ep *ep, const void *buf, size_t len,
    fi_addr_t dest_addr);

ssize_t fi_senddata(struct fid_ep *ep, const void *buf, size_t len,
    void *desc, uint64_t data, fi_addr_t dest_addr, void *context);

ssize_t fi_injectdata(struct fid_ep *ep, const void *buf, size_t len,
    uint64_t data, fi_addr_t dest_addr);
```

# ARGUMENTS

*ep*
:   Fabric endpoint on which to initiate send or post receive buffer.

*buf*
:   Data buffer to send or receive.

*len*
:   Length of data buffer to send or receive, specified in bytes. Valid
    transfers are from 0 bytes up to the endpoint's max_msg_size.

*iov*
:   Vectored data buffer.

*count*
:   Count of vectored data entries.

*desc*
:   Descriptor associated with the data buffer. See
    [`fi_mr`(3)](fi_mr.3.html).

*data*
:   Remote CQ data to transfer with the sent message.

*dest_addr*
:   Destination address for connectionless transfers. Ignored for
    connected endpoints.

*src_addr*
:   Applies only to connectionless endpoints configured with the
    FI_DIRECTED_RECV. For all other endpoint configurations, src_addr is
    ignored. src_addr defines the source address to receive from. By
    default, the src_addr is treated as a source endpoint address
    (i.e. fi_addr_t returned from fi_av_insert / fi_av_insertsvc /
    fi_av_remove). If the FI_AUTH_KEY flag is specified with fi_recvmsg,
    src_addr is treated as a source authorization key (i.e. fi_addr_t
    returned from fi_av_insert_auth_key). If set to FI_ADDR_UNSPEC, any
    source address may match.

*msg*
:   Message descriptor for send and receive operations.

*flags*
:   Additional flags to apply for the send or receive operation.

*context*
:   User specified pointer to associate with the operation. This
    parameter is ignored if the operation will not generate a successful
    completion, unless an op flag specifies the context parameter be
    used for required input.

# DESCRIPTION

The send functions -- fi_send, fi_sendv, fi_sendmsg, fi_inject, and
fi_senddata -- are used to transmit a message from one endpoint to
another endpoint. The main difference between send functions are the
number and type of parameters that they accept as input. Otherwise, they
perform the same general function. Messages sent using fi_msg operations
are received by a remote endpoint into a buffer posted to receive such
messages.

The receive functions -- fi_recv, fi_recvv, fi_recvmsg -- post a data
buffer to an endpoint to receive inbound messages. Similar to the send
operations, receive operations operate asynchronously. Users should not
touch the posted data buffer(s) until the receive operation has
completed.

An endpoint must be enabled before an application can post send or
receive operations to it. For connected endpoints, receive buffers may
be posted prior to connect or accept being called on the endpoint. This
ensures that buffers are available to receive incoming data immediately
after the connection has been established.

Completed message operations are reported to the user through one or
more event collectors associated with the endpoint. Users provide
context which are associated with each operation, and is returned to the
user as part of the event completion. See fi_cq for completion event
details.

## fi_send

The call fi_send transfers the data contained in the user-specified data
buffer to a remote endpoint, with message boundaries being maintained.

## fi_sendv

The fi_sendv call adds support for a scatter-gather list to fi_send. The
fi_sendv transfers the set of data buffers referenced by the iov
parameter to a remote endpoint as a single message.

## fi_sendmsg

The fi_sendmsg call supports data transfers over both connected and
connectionless endpoints, with the ability to control the send operation
per call through the use of flags. The fi_sendmsg function takes a
`struct fi_msg` as input.

``` c
struct fi_msg {
    const struct iovec *msg_iov; /* scatter-gather array */
    void               **desc;   /* local request descriptors */
    size_t             iov_count;/* # elements in iov */
    fi_addr_t          addr;     /* optional endpoint address */
    void               *context; /* user-defined context */
    uint64_t           data;     /* optional message data */
};
```

## fi_inject

The send inject call is an optimized version of fi_send with the
following characteristics. The data buffer is available for reuse
immediately on return from the call, and no CQ entry will be written if
the transfer completes successfully.

Conceptually, this means that the fi_inject function behaves as if the
FI_INJECT transfer flag were set, selective completions are enabled, and
the FI_COMPLETION flag is not specified. Note that the CQ entry will be
suppressed even if the default behavior of the endpoint is to write CQ
entries for all successful completions. See the flags discussion below
for more details. The requested message size that can be used with
fi_inject is limited by inject_size.

If FI_HMEM is enabled, the fi_inject call can only accept buffer with
iface equal to FI_HMEM_SYSTEM if the provider requires the FI_MR_HMEM
mr_mode. This limitation applies to all the fi\_\*inject\* calls and
does not affect how inject_size is reported.

## fi_senddata

The send data call is similar to fi_send, but allows for the sending of
remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the transfer.

## fi_injectdata

The inject data call is similar to fi_inject, but allows for the sending
of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the transfer.

## fi_recv

The fi_recv call posts a data buffer to the receive queue of the
corresponding endpoint. Posted receives are searched in the order in
which they were posted in order to match sends. Message boundaries are
maintained. The order in which the receives complete is dependent on the
endpoint type and protocol. For connectionless endpoints, the src_addr
parameter can be used to indicate that a buffer should be posted to
receive incoming data from a specific remote endpoint.

## fi_recvv

The fi_recvv call adds support for a scatter-gather list to fi_recv. The
fi_recvv posts the set of data buffers referenced by the iov parameter
to a receive incoming data.

## fi_recvmsg

The fi_recvmsg call supports posting buffers over both connected and
connectionless endpoints, with the ability to control the receive
operation per call through the use of flags. The fi_recvmsg function
takes a struct fi_msg as input.

# FLAGS

The fi_recvmsg and fi_sendmsg calls allow the user to specify flags
which can change the default message handling of the endpoint. Flags
specified with fi_recvmsg / fi_sendmsg override most flags previously
configured with the endpoint, except where noted (see fi_endpoint.3).
The following list of flags are usable with fi_recvmsg and/or
fi_sendmsg.

*FI_REMOTE_CQ_DATA*
:   Applies to fi_sendmsg. Indicates that remote CQ data is available
    and should be sent as part of the request. See fi_getinfo for
    additional details on FI_REMOTE_CQ_DATA. This flag is implicitly set
    for fi_senddata and fi_injectdata.

*FI_COMPLETION*
:   Indicates that a completion entry should be generated for the
    specified operation. The endpoint must be bound to a completion
    queue with FI_SELECTIVE_COMPLETION that corresponds to the specified
    operation, or this flag is ignored.

*FI_MORE*
:   Indicates that the user has additional requests that will
    immediately be posted after the current call returns. Use of this
    flag may improve performance by enabling the provider to optimize
    its access to the fabric hardware.

*FI_INJECT*
:   Applies to fi_sendmsg. Indicates that the outbound data buffer
    should be returned to user immediately after the send call returns,
    even if the operation is handled asynchronously. This may require
    that the underlying provider implementation copy the data into a
    local buffer and transfer out of that buffer. This flag can only be
    used with messages smaller than inject_size.

*FI_MULTI_RECV*
:   Applies to posted receive operations. This flag allows the user to
    post a single buffer that will receive multiple incoming messages.
    Received messages will be packed into the receive buffer until the
    buffer has been consumed. Use of this flag may cause a single posted
    receive operation to generate multiple events as messages are placed
    into the buffer. The placement of received data into the buffer may
    be subjected to provider specific alignment restrictions.

The buffer will be released by the provider when the available buffer
space falls below the specified minimum (see FI_OPT_MIN_MULTI_RECV).
Note that an entry to the associated receive completion queue will
always be generated when the buffer has been consumed, even if other
receive completions have been suppressed (i.e. the Rx context has been
configured for FI_SELECTIVE_COMPLETION). See the FI_MULTI_RECV
completion flag [`fi_cq`(3)](fi_cq.3.html).

*FI_INJECT_COMPLETE*
:   Applies to fi_sendmsg. Indicates that a completion should be
    generated when the source buffer(s) may be reused.

*FI_TRANSMIT_COMPLETE*
:   Applies to fi_sendmsg and fi_recvmsg. For sends, indicates that a
    completion should not be generated until the operation has been
    successfully transmitted and is no longer being tracked by the
    provider. For receive operations, indicates that a completion may be
    generated as soon as the message has been processed by the local
    provider, even if the message data may not be visible to all
    processing elements. See [`fi_cq`(3)](fi_cq.3.html) for target side
    completion semantics.

*FI_DELIVERY_COMPLETE*
:   Applies to fi_sendmsg. Indicates that a completion should be
    generated when the operation has been processed by the destination.

*FI_FENCE*
:   Applies to transmits. Indicates that the requested operation, also
    known as the fenced operation, and any operation posted after the
    fenced operation will be deferred until all previous operations
    targeting the same peer endpoint have completed. Operations posted
    after the fencing will see and/or replace the results of any
    operations initiated prior to the fenced operation.

The ordering of operations starting at the posting of the fenced
operation (inclusive) to the posting of a subsequent fenced operation
(exclusive) is controlled by the endpoint's ordering semantics.

*FI_MULTICAST*
:   Applies to transmits. This flag indicates that the address specified
    as the data transfer destination is a multicast address. This flag
    must be used in all multicast transfers, in conjunction with a
    multicast fi_addr_t.

*FI_AUTH_KEY*
:   Only valid with domains configured with FI_AV_AUTH_KEY and
    connectionless endpoints configured with FI_DIRECTED_RECV. When used
    with fi_recvmsg, this flag denotes that the src_addr is an
    authorization key fi_addr_t instead of an endpoint fi_addr_t.

# NOTES

If an endpoint has been configured with FI_MSG_PREFIX, the application
must include buffer space of size msg_prefix_size, as specified by the
endpoint attributes. The prefix buffer must occur at the start of the
data referenced by the buf parameter, or be referenced by the first IO
vector. Message prefix space cannot be split between multiple IO
vectors. The size of the prefix buffer should be included as part of the
total buffer length.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

See the discussion below for details handling FI_EAGAIN.

# ERRORS

*-FI_EAGAIN*
:   Indicates that the underlying provider currently lacks the resources
    needed to initiate the requested operation. The reasons for a
    provider returning FI_EAGAIN are varied. However, common reasons
    include insufficient internal buffering or full processing queues.

Insufficient internal buffering is often associated with operations that
use FI_INJECT. In such cases, additional buffering may become available
as posted operations complete.

Full processing queues may be a temporary state related to local
processing (for example, a large message is being transferred), or may
be the result of flow control. In the latter case, the queues may remain
blocked until additional resources are made available at the remote side
of the transfer.

In all cases, the operation may be retried after additional resources
become available. When using FI_PROGRESS_MANUAL, the application must
check for transmit and receive completions after receiving FI_EAGAIN as
a return value, independent of the operation which failed. This is also
strongly recommended when using FI_PROGRESS_AUTO, as acknowledgements or
flow control messages may need to be processed in order to resume
execution.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cq`(3)](fi_cq.3.html)

{% include JB/setup %}

# NAME

fi_mr - Memory region operations

fi_mr_reg / fi_mr_regv / fi_mr_regattr
:   Register local memory buffers for direct fabric access

fi_close
:   Deregister registered memory buffers.

fi_mr_desc
:   Return a local descriptor associated with a registered memory region

fi_mr_key
:   Return the remote key needed to access a registered memory region

fi_mr_raw_attr
:   Return raw memory region attributes.

fi_mr_map_raw
:   Converts a raw memory region key into a key that is usable for data
    transfer operations.

fi_mr_unmap_key
:   Releases a previously mapped raw memory region key.

fi_mr_bind
:   Associate a registered memory region with a completion counter or an
    endpoint.

fi_mr_refresh
:   Updates the memory pages associated with a memory region.

fi_mr_enable
:   Enables a memory region for use.

fi_hmem_ze_device
:   Returns an hmem device identifier for a level zero driver and
    device.

# SYNOPSIS

``` c
#include <rdma/fi_domain.h>

int fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
    uint64_t access, uint64_t offset, uint64_t requested_key,
    uint64_t flags, struct fid_mr **mr, void *context);

int fi_mr_regv(struct fid_domain *domain, const struct iovec * iov,
    size_t count, uint64_t access, uint64_t offset, uint64_t requested_key,
    uint64_t flags, struct fid_mr **mr, void *context);

int fi_mr_regattr(struct fid_domain *domain, const struct fi_mr_attr *attr,
    uint64_t flags, struct fid_mr **mr);

int fi_close(struct fid *mr);

void * fi_mr_desc(struct fid_mr *mr);

uint64_t fi_mr_key(struct fid_mr *mr);

int fi_mr_raw_attr(struct fid_mr *mr, uint64_t *base_addr,
    uint8_t *raw_key, size_t *key_size, uint64_t flags);

int fi_mr_map_raw(struct fid_domain *domain, uint64_t base_addr,
    uint8_t *raw_key, size_t key_size, uint64_t *key, uint64_t flags);

int fi_mr_unmap_key(struct fid_domain *domain, uint64_t key);

int fi_mr_bind(struct fid_mr *mr, struct fid *bfid, uint64_t flags);

int fi_mr_refresh(struct fid_mr *mr, const struct iovec *iov,
    size_t count, uint64_t flags);

int fi_mr_enable(struct fid_mr *mr);

int fi_hmem_ze_device(int driver_index, int device_index);
```

# ARGUMENTS

*domain*
:   Resource domain

*mr*
:   Memory region

*bfid*
:   Fabric identifier of an associated resource.

*context*
:   User specified context associated with the memory region.

*buf*
:   Memory buffer to register with the fabric hardware.

*len*
:   Length of memory buffer to register. Must be \> 0.

*iov*
:   Vectored memory buffer.

*count*
:   Count of vectored buffer entries.

*access*
:   Memory access permissions associated with registration

*offset*
:   Optional specified offset for accessing specified registered
    buffers. This parameter is reserved for future use and must be 0.

*requested_key*
:   Requested remote key associated with registered buffers. Parameter
    is ignored if FI_MR_PROV_KEY flag is set in the domain mr_mode bits.
    Parameter may be ignored if remote access permissions are not asked
    for.

*attr*
:   Memory region attributes

*flags*
:   Additional flags to apply to the operation.

# DESCRIPTION

Registered memory regions associate memory buffers with permissions
granted for access by fabric resources. A memory buffer must be
registered with a resource domain before it can be used as the target of
a remote RMA or atomic data transfer. Additionally, a fabric provider
may require that data buffers be registered before being used in local
transfers. Memory registration restrictions are controlled using a
separate set of mode bits, specified through the domain attributes
(mr_mode field). Each mr_mode bit requires that an application take
specific steps in order to use memory buffers with libfabric interfaces.

As a special case, a new memory region can be created from an existing
memory region. Such a new memory region is called a sub-MR, and the
existing memory region is called the base MR. Sub-MRs may be used to
shared hardware resources, such as virtual to physical address
translations and page pinning. This can improve performance when
creating and destroying sub-regions that need different access rights.
The base MR itself can also be a sub-MR, allowing for a hierarchy of
memory regions.

The following apply to memory registration.

*Default Memory Registration*
:   If no mr_mode bits are set, the default behaviors describe below are
    followed. Historically, these defaults were collectively referred to
    as scalable memory registration. The default requirements are
    outlined below, followed by definitions of how each mr_mode bit
    alters the definition.

Compatibility: For library versions 1.4 and earlier, this was indicated
by setting mr_mode to FI_MR_SCALABLE and the fi_info mode bit
FI_LOCAL_MR to 0. FI_MR_SCALABLE and FI_LOCAL_MR were deprecated in
libfabric version 1.5, though they are supported for backwards
compatibility purposes.

For security, memory registration is required for data buffers that are
accessed directly by a peer process. For example, registration is
required for RMA target buffers (read or written to), and those accessed
by atomic or collective operations.

By default, registration occurs on virtual address ranges. Because
registration refers to address ranges, rather than allocated data
buffers, the address ranges do not need to map to data buffers allocated
by the application at the time the registration call is made. That is,
an application can register any range of addresses in their virtual
address space, whether or not those addresses are backed by physical
pages or have been allocated.

Note that physical pages must back addresses prior to the addresses
being accessed as part of a data transfer operation, or the data
transfers will fail. Additionally, depending on the operation, this
could result in the local process receiving a segmentation fault for
accessing invalid memory.

Once registered, the resulting memory regions are accessible by peers
starting at a base address of 0. That is, the target address that is
specified is a byte offset into the registered region.

The application also selects the access key associated with the MR. The
key size is restricted to a maximum of 8 bytes.

With scalable registration, locally accessed data buffers are not
registered. This includes source buffers for all transmit operations --
sends, tagged sends, RMA, and atomics -- as well as buffers posted for
receive and tagged receive operations.

Although the default memory registration behavior is convenient for
application developers, it is difficult to implement in hardware.
Attempts to hide the hardware requirements from the application often
results in significant and unacceptable impacts to performance. The
following mr_mode bits are provided as input into fi_getinfo. If a
provider requires the behavior defined for an mr_mode bit, it will leave
the bit set on output to fi_getinfo. Otherwise, the provider can clear
the bit to indicate that the behavior is not needed.

By setting an mr_mode bit, the application has agreed to adjust its
behavior as indicated. Importantly, applications that choose to support
an mr_mode must be prepared to handle the case where the mr_mode is not
required. A provider will clear an mr_mode bit if it is not needed.

*FI_MR_LOCAL*
:   When the FI_MR_LOCAL mode bit is set, applications must register all
    data buffers that will be accessed by the local hardware and provide
    a valid desc parameter into applicable data transfer operations.

When FI_MR_LOCAL is unset, applications are not required to register
data buffers before using them for local operations (e.g. send and
receive data buffers). Prior to libfabric 1.22, the desc parameter was
ignored. In libfabric 1.22 and later, the desc parameter must be either
valid or NULL. This behavior allows applications to optionally pass in a
valid desc parameter. If the desc parameter is NULL, any required local
memory registration will be handled by the provider.

A provider may hide local registration requirements from applications by
making use of an internal registration cache or similar mechanisms. Such
mechanisms, however, may negatively impact performance for some
applications, notably those which manage their own network buffers. In
order to support as broad range of applications as possible, without
unduly affecting their performance, applications that wish to manage
their own local memory registrations may do so by using the memory
registration calls and passing in a valid desc parameter.

Note: the FI_MR_LOCAL mr_mode bit replaces the FI_LOCAL_MR fi_info mode
bit. When FI_MR_LOCAL is set, FI_LOCAL_MR is ignored.

*FI_MR_RAW*
:   Raw memory regions are used to support providers with keys larger
    than 64-bits or require setup at the peer. When the FI_MR_RAW bit is
    set, applications must use fi_mr_raw_attr() locally and
    fi_mr_map_raw() at the peer before targeting a memory region as part
    of any data transfer request.

*FI_MR_VIRT_ADDR*
:   The FI_MR_VIRT_ADDR bit indicates that the provider references
    memory regions by virtual address, rather than a 0-based offset.
    Peers that target memory regions registered with FI_MR_VIRT_ADDR
    specify the destination memory buffer using the target's virtual
    address, with any offset into the region specified as virtual
    address + offset. Support of this bit typically implies that peers
    must exchange addressing data prior to initiating any RMA or atomic
    operation.

For memory regions that are registered using FI_MR_DMABUF, the starting
'virtual address' of the DMA-buf region is obtained by adding the offset
field to the base_addr field of struct fi_mr_dmabuf that was specified
through the registration call.

*FI_MR_ALLOCATED*
:   When set, all registered memory regions must be backed by physical
    memory pages at the time the registration call is made. In addition,
    applications must not perform operations which may result in the
    underlying virtual address to physical page mapping to change
    (e.g. calling free() against an allocated MR). Failing to adhere to
    this may result in the virtual address pointing to one set of
    physical pages while the MR points to another set of physical pages.

When unset, registered memory regions need not be backed by physical
memory pages at the time the registration call is made. In addition, the
underlying virtual address to physical page mapping is allowed to
change, and the provider will ensure the corresponding MR is updated
accordingly. This behavior enables application use-cases where memory
may be frequently freed and reallocated or system memory migrating
to/from device memory.

When unset, the application is responsible for ensuring that a
registered memory region references valid physical pages while a data
transfer is active against it, or the data transfer may fail.
Application changes to the virtual address range must be coordinated
with network traffic to or from that range.

If unset and FI_HMEM is supported, the ability for the virtual address
to physical address mapping to change extends to HMEM interfaces as
well. If a provider cannot support a virtual address to physical address
changing for a given HMEM interface, the provider should support a
reasonable fallback or the operation should fail.

*FI_MR_PROV_KEY*
:   This memory region mode indicates that the provider does not support
    application requested MR keys. MR keys are returned by the provider.
    Applications that support FI_MR_PROV_KEY can obtain the provider key
    using fi_mr_key(), unless FI_MR_RAW is also set. The returned key
    should then be exchanged with peers prior to initiating an RMA or
    atomic operation.

*FI_MR_MMU_NOTIFY*
:   FI_MR_MMU_NOTIFY is typically set by providers that support memory
    registration against memory regions that are not necessarily backed
    by allocated physical pages at the time the memory registration
    occurs. (That is, FI_MR_ALLOCATED is typically 0). However, such
    providers require that applications notify the provider prior to the
    MR being accessed as part of a data transfer operation. This
    notification informs the provider that all necessary physical pages
    now back the region. The notification is necessary for providers
    that cannot hook directly into the operating system page tables or
    memory management unit. See fi_mr_refresh() for notification
    details.

*FI_MR_RMA_EVENT*
:   This mode bit indicates that the provider must configure memory
    regions that are associated with RMA events prior to their use. This
    includes all memory regions that are associated with completion
    counters. When set, applications must indicate if a memory region
    will be associated with a completion counter as part of the region's
    creation. This is done by passing in the FI_RMA_EVENT flag to the
    memory registration call.

Such memory regions will be created in a disabled state and must be
associated with all completion counters prior to being enabled. To
enable a memory region, the application must call fi_mr_enable(). After
calling fi_mr_enable(), no further resource bindings may be made to the
memory region.

*FI_MR_ENDPOINT*
:   This mode bit indicates that the provider associates memory regions
    with endpoints rather than domains. Memory regions that are
    registered with the provider are created in a disabled state and
    must be bound to an endpoint prior to being enabled. To bind the MR
    with an endpoint, the application must use fi_mr_bind(). To enable
    the memory region, the application must call fi_mr_enable().

*FI_MR_HMEM*
:   This mode bit is associated with the FI_HMEM capability.

If FI_MR_HMEM is set, the application must register buffers that were
allocated using a device call and provide a valid desc parameter into
applicable data transfer operations even if they are only used for local
operations (e.g. send and receive data buffers). Device memory must be
registered using the fi_mr_regattr call, with the iface and device
fields filled out.

If FI_MR_HMEM is unset, the application need not register device buffers
for local operations. In addition, fi_mr_regattr is not required to be
used for device memory registration. It is the responsibility of the
provider to discover the appropriate device memory registration
attributes, if applicable.

Similar to if FI_MR_LOCAL is unset, if FI_MR_HMEM is unset, applications
may optionally pass in a valid desc parameter. If the desc parameter is
NULL, any required local memory registration will be handled by the
provider.

If FI_MR_HMEM is set, but FI_MR_LOCAL is unset, only device buffers must
be registered when used locally. In this case, the desc parameter passed
into data transfer operations must either be valid or NULL. Similarly,
if FI_MR_LOCAL is set, but FI_MR_HMEM is not, the desc parameter must
either be valid or NULL.

*FI_MR_COLLECTIVE*
:   This bit is associated with the FI_COLLECTIVE capability.

If FI_MR_COLLECTIVE is set, the provider requires that memory regions
used in collection operations must explicitly be registered for use with
collective calls. This requires registering regions passed to collective
calls using the FI_COLLECTIVE flag.

If FI_MR_COLLECTIVE is unset, memory registration for collection
operations is optional. Applications may optionally pass in a valid desc
parameter. If the desc parameter is NULL, any required local memory
registration will be handled by the provider.

*Basic Memory Registration* (deprecated)
:   Basic memory registration was deprecated in libfabric version 1.5,
    but is supported for backwards compatibility. Basic memory
    registration is indicated by setting mr_mode equal to FI_MR_BASIC.
    FI_MR_BASIC must be set alone and not paired with mr_mode bits.
    Unlike other mr_mode bits, if FI_MR_BASIC is set on input to
    fi_getinfo(), it will not be cleared by the provider. That is,
    setting mr_mode equal to FI_MR_BASIC forces basic registration if
    the provider supports it.

The behavior of basic registration is equivalent to requiring the
following mr_mode bits: FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, and
FI_MR_PROV_KEY. Additionally, providers that support basic registration
usually require the (deprecated) fi_info mode bit FI_LOCAL_MR, which was
incorporated into the FI_MR_LOCAL mr_mode bit.

The registrations functions -- fi_mr_reg, fi_mr_regv, and fi_mr_regattr
-- are used to register one or more memory regions with fabric
resources. The main difference between registration functions are the
number and type of parameters that they accept as input. Otherwise, they
perform the same general function.

**Deprecated** : By default, memory registration completes
synchronously. I.e. the registration call will not return until the
registration has completed. Memory registration can complete
asynchronous by binding the resource domain to an event queue using the
FI_REG_MR flag. See fi_domain_bind. When memory registration is
asynchronous, in order to avoid a race condition between the
registration call returning and the corresponding reading of the event
from the EQ, the mr output parameter will be written before any event
associated with the operation may be read by the application. An
asynchronous event will not be generated unless the registration call
returns success (0).

## fi_mr_reg

The fi_mr_reg call registers the user-specified memory buffer with the
resource domain. The buffer is enabled for access by the fabric hardware
based on the provided access permissions. See the access field
description for memory region attributes below.

Registered memory is associated with a local memory descriptor and,
optionally, a remote memory key. A memory descriptor is a provider
specific identifier associated with registered memory. Memory
descriptors often map to hardware specific indices or keys associated
with the memory region. Remote memory keys provide limited protection
against unwanted access by a remote node. Remote accesses to a memory
region must provide the key associated with the registration.

Because MR keys must be provided by a remote process, an application can
use the requested_key parameter to indicate that a specific key value be
returned. Support for user requested keys is provider specific and is
determined by the FI_MR_PROV_KEY flag value in the mr_mode domain
attribute.

Remote RMA and atomic operations indicate the location within a
registered memory region by specifying an address. The location is
referenced by adding the offset to either the base virtual address of
the buffer or to 0, depending on the mr_mode.

The offset parameter is reserved for future use and must be 0.

For asynchronous memory registration requests, the result will be
reported to the user through an event queue associated with the resource
domain. If successful, the allocated memory region structure will be
returned to the user through the mr parameter. The mr address must
remain valid until the registration operation completes. The context
specified with the registration request is returned with the completion
event.

For domains opened with FI_AV_AUTH_KEY, fi_mr_reg is not supported and
fi_mr_regattr must be used.

## fi_mr_regv

The fi_mr_regv call adds support for a scatter-gather list to fi_mr_reg.
Multiple memory buffers are registered as a single memory region.
Otherwise, the operation is the same.

For domains opened with FI_AV_AUTH_KEY, fi_mr_regv is not supported and
fi_mr_regattr must be used.

## fi_mr_regattr

The fi_mr_regattr call is a more generic, extensible registration call
that allows the user to specify the registration request using a struct
fi_mr_attr (defined below).

## fi_close

Fi_close is used to release all resources associated with a registering
a memory region. Once deregistered, further access to the registered
memory is not guaranteed. Active or queued operations that reference a
memory region being closed may attempt to access an invalid memory
region and fail. After an MR is closed, any new operations targeting the
closed MR will also fail. Applications are responsible for ensuring that
a MR is no longer needed prior to closing it. Note that accesses to a
closed MR from a remote peer will result in an error at the peer. The
state of the local endpoint will be unaffected.

For MRs that are associated with an endpoint (FI_MR_ENDPOINT flag is
set), the MR must be closed before the endpoint. If resources are still
associated with the MR when attempting to close, the call will return
-FI_EBUSY.

If a memory registration cache is used, the behavior of fi_close may be
affected. More information on the memory registration cache is in the
MEMORY REGISTRATION CACHE section.

## fi_mr_desc

Obtains the local memory descriptor associated with a MR. The memory
registration must have completed successfully before invoking this call.

## fi_mr_key

Returns the remote protection key associated with a MR. The memory
registration must have completed successfully before invoking this. The
returned key may be used in data transfer operations at a peer. If the
FI_MR_RAW mode bit has been set for the domain, then the memory key must
be obtained using the fi_mr_raw_key function instead. A return value of
FI_KEY_NOTAVAIL will be returned if the registration has not completed
or a raw memory key is required.

## fi_mr_raw_attr

Returns the raw, remote protection key and base address associated with
a MR. The memory registration must have completed successfully before
invoking this routine. Use of this call is required if the FI_MR_RAW
mode bit has been set by the provider; however, it is safe to use this
call with any memory region.

On input, the key_size parameter should indicate the size of the raw_key
buffer. If the actual key is larger than what can fit into the buffer,
it will return -FI_ETOOSMALL. On output, key_size is set to the size of
the buffer needed to store the key, which may be larger than the input
value. The needed key_size can also be obtained through the mr_key_size
domain attribute (fi_domain_attr) field.

A raw key must be mapped by a peer before it can be used in data
transfer operations. See fi_mr_map_raw below.

## fi_mr_map_raw

Raw protection keys must be mapped to a usable key value before they can
be used for data transfer operations. The mapping is done by the peer
that initiates the RMA or atomic operation. The mapping function takes
as input the raw key and its size, and returns the mapped key. Use of
the fi_mr_map_raw function is required if the peer has the FI_MR_RAW
mode bit set, but this routine may be called on any valid key. All
mapped keys must be freed by calling fi_mr_unmap_key when access to the
peer memory region is no longer necessary.

## fi_mr_unmap_key

This call releases any resources that may have been allocated as part of
mapping a raw memory key. All mapped keys must be freed before the
corresponding domain is closed.

## fi_mr_bind

The fi_mr_bind function associates a memory region with a counter or
endpoint. Counter bindings are needed by providers that support the
generation of completions based on fabric operations. Endpoint bindings
are needed if the provider associates memory regions with endpoints (see
FI_MR_ENDPOINT).

When binding with a counter, the type of events tracked against the
memory region is based on the bitwise OR of the following flags.

*FI_REMOTE_WRITE*
:   Generates an event whenever a remote RMA write or atomic operation
    modifies the memory region. Use of this flag requires that the
    endpoint through which the MR is accessed be created with the
    FI_RMA_EVENT capability.

When binding the memory region to an endpoint, flags should be 0.

## fi_mr_refresh

The use of this call is to notify the provider of any change to the
physical pages backing a registered memory region. This call must be
supported by providers requiring FI_MR_MMU_NOTIFY and may optionally be
supported by providers not requiring FI_MR_ALLOCATED.

This call informs the provider that the page table entries associated
with the region may have been modified, and the provider should verify
and update the registered region accordingly. The iov parameter is
optional and may be used to specify which portions of the registered
region requires updating.

Providers are only guaranteed to update the specified address ranges.
Failing to update a range will result in an error being returned.

When FI_MR_MMU_NOTIFY is set, the refresh operation has the effect of
disabling and re-enabling access to the registered region. Any
operations from peers that attempt to access the region will fail while
the refresh is occurring. Additionally, attempts to access the region by
the local process through libfabric APIs may result in a page fault or
other fatal operation.

When FI_MR_ALLOCATED is unset, -FI_ENOSYS will be returned if a provider
does not support fi_mr_refresh. If supported, the provider will
atomically update physical pages of the MR associated with the user
specified address ranges. The MR will remain enabled during this time.

Note: FI_MR_MMU_NOTIFY set behavior takes precedence over
FI_MR_ALLOCATED unset behavior.

The fi_mr_refresh call is only needed if the physical pages might have
been updated after the memory region was created.

## fi_mr_enable

The enable call is used with memory registration associated with the
FI_MR_RMA_EVENT mode bit. Memory regions created in the disabled state
must be explicitly enabled after being fully configured by the
application. Any resource bindings to the MR must be done prior to
enabling the MR.

# MEMORY REGION ATTRIBUTES

Memory regions are created using the following attributes. The struct
fi_mr_attr is passed into fi_mr_regattr, but individual fields also
apply to other memory registration calls, with the fields passed
directly into calls as function parameters.

``` c
struct fi_mr_attr {
    union {
        const struct iovec *mr_iov;
        const struct fi_mr_dmabuf *dmabuf;
    };
    size_t             iov_count;
    uint64_t           access;
    uint64_t           offset;
    uint64_t           requested_key;
    void               *context;
    size_t             auth_key_size;
    uint8_t            *auth_key;
    enum fi_hmem_iface iface;
    union {
        uint64_t       reserved;
        int            cuda;
        int            ze
        int            neuron;
        int            synapseai;
    } device;
    void               *hmem_data;
    size_t             page_size;
    const struct fid_mr *base_mr;
    size_t             sub_mr_cnt;
};

struct fi_mr_auth_key {
    struct fid_av *av;
    fi_addr_t     src_addr;
};
```

## mr_iov

This is an IO vector of virtual addresses and their length that
represent a single memory region. The number of entries in the iovec is
specified by iov_count.

## dmabuf

DMA-buf registrations are used to share device memory between a given
device and the fabric NIC and does not require that the device memory be
mmap'ed into the virtual address space of the calling process.

This structure references a DMA-buf backed device memory region. This
field is only usable if the application has successfully requested
support for FI_HMEM and the FI_MR_DMABUF flag is passed into the memory
registration call. DMA-buf regions are file-based references to device
memory. Such regions are identified through the struct fi_mr_dmabuf.

``` c
struct fi_mr_dmabuf {
    int      fd;
    uint64_t offset;
    size_t   len;
    void     *base_addr;
};
```

The fd is the file descriptor associated with the DMA-buf region. The
offset is the offset into the region where the memory registration
should begin. And len is the size of the region to register, starting at
the offset. The base_addr is the page-aligned starting virtual address
of the memory region allocated by the DMA-buf. If a base virtual address
is not available (because, for example, the calling process has not
mapped the memory region into its address space), base_addr can be set
to NULL.

The selection of dmabuf over the mr_iov field is controlled by
specifying the FI_MR_DMABUF flag.

## iov_count

The number of entries in the mr_iov array. The maximum number of memory
buffers that may be associated with a single memory region is specified
as the mr_iov_limit domain attribute. See `fi_domain(3)`.

## access

Indicates the type of *operations* that the local or a peer endpoint may
perform on registered memory region. Supported access permissions are
the bitwise OR of the following flags:

*FI_SEND*
:   The memory buffer may be used in outgoing message data transfers.
    This includes fi_msg and fi_tagged send operations, as well as
    fi_collective operations.

*FI_RECV*
:   The memory buffer may be used to receive inbound message transfers.
    This includes fi_msg and fi_tagged receive operations, as well as
    fi_collective operations.

*FI_READ*
:   The memory buffer may be used as the result buffer for RMA read and
    atomic operations on the initiator side. Note that from the
    viewpoint of the application, the memory buffer is being written
    into by the network.

*FI_WRITE*
:   The memory buffer may be used as the source buffer for RMA write and
    atomic operations on the initiator side. Note that from the
    viewpoint of the application, the endpoint is reading from the
    memory buffer and copying the data onto the network.

*FI_REMOTE_READ*
:   The memory buffer may be used as the source buffer of an RMA read
    operation on the target side. The contents of the memory buffer are
    not modified by such operations.

*FI_REMOTE_WRITE*
:   The memory buffer may be used as the target buffer of an RMA write
    or atomic operation. The contents of the memory buffer may be
    modified as a result of such operations.

*FI_COLLECTIVE*
:   This flag provides an explicit indication that the memory buffer may
    be used with collective operations. Use of this flag is required if
    the FI_MR_COLLECTIVE mr_mode bit has been set on the domain. This
    flag should be paired with FI_SEND and/or FI_RECV

Note that some providers may not enforce fine grained access
permissions. For example, a memory region registered for FI_WRITE access
may also behave as if FI_SEND were specified as well. Relaxed
enforcement of such access is permitted, though not guaranteed, provided
security is maintained.

## offset

The offset field is reserved for future use and must be 0.

## requested_key

An application specified access key associated with the memory region.
The MR key must be provided by a remote process when performing RMA or
atomic operations to a memory region. Applications can use the
requested_key field to indicate that a specific key be used by the
provider. This allows applications to use well known key values, which
can avoid applications needing to exchange and store keys. Support for
user requested keys is provider specific and is determined by the the
FI_MR_PROV_KEY flag in the mr_mode domain attribute field. Depending on
the provider, the user requested key may be ignored if the memory region
is for local access only. A provider may be unable to do so if the
hardware supports user requested keys and the same key is used for both
local and remote access.

## context

Application context associated with asynchronous memory registration
operations. This value is returned as part of any asynchronous event
associated with the registration. This field is ignored for synchronous
registration calls.

## auth_key_size

The size of key referenced by the auth_key field in bytes, or 0 if no
authorization key is given. This field is ignored unless the fabric is
opened with API version 1.5 or greater.

If the domain is opened with FI_AV_AUTH_KEY, auth_key_size must equal
`sizeof(struct fi_mr_auth_key)`.

## auth_key

Indicates the key to associate with this memory registration.
Authorization keys are used to limit communication between endpoints.
Only peer endpoints that are programmed to use the same authorization
key may access the memory region. The domain authorization key will be
used if the auth_key_size provided is 0. This field is ignored unless
the fabric is opened with API version 1.5 or greater.

If the domain is opened with FI_AV_AUTH_KEY, auth_key must point to a
user-defined `struct fi_mr_auth_key`.

## iface

Indicates the software interfaces used by the application to allocate
and manage the memory region. This field is ignored unless the
application has requested the FI_HMEM capability.

*FI_HMEM_SYSTEM*
:   Uses standard operating system calls and libraries, such as malloc,
    calloc, realloc, mmap, and free. When iface is set to
    FI_HMEM_SYSTEM, the device field (described below) is ignored.

*FI_HMEM_CUDA*
:   Uses Nvidia CUDA interfaces such as cuMemAlloc, cuMemAllocHost,
    cuMemAllocManaged, cuMemFree, cudaMalloc, cudaFree.

*FI_HMEM_ROCR*
:   Uses AMD ROCR interfaces such as hsa_memory_allocate and
    hsa_memory_free.

*FI_HMEM_ZE*
:   Uses oneAPI Level Zero interfaces such as zeDriverAllocSharedMem,
    zeDriverFreeMem.

*FI_HMEM_NEURON*
:   Uses the AWS Neuron SDK to support AWS Trainium devices.

*FI_HMEM_SYNAPSEAI*
:   Uses the SynapseAI API to support Habana Gaudi devices.

## device

Reserved 64 bits for device identifier if using non-standard HMEM
interface. This field is ignore unless the iface field is valid.
Otherwise, the device field is determined by the value specified through
iface.

*cuda*
:   For FI_HMEM_CUDA, this is equivalent to CUdevice (int).

*ze*
:   For FI_HMEM_ZE, this is equivalent to the index of the device in the
    ze_device_handle_t array. If there is only a single level zero
    driver present, an application may set this directly. However, it is
    recommended that this value be set using the fi_hmem_ze_device()
    macro, which will encode the driver index with the device.

*neuron*
:   For FI_HMEM_NEURON, the device identifier for AWS Trainium devices.

*synapseai*
:   For FI_HMEM_SYNAPSEAI, the device identifier for Habana Gaudi
    hardware.

## hmem_data

The hmem_data field is reserved for future use and must be null.

## page_size

Page size allows applications to optionally provide a hint at what the
optimal page size is for the an MR allocation. Typically, providers can
select the optimal page size. In cases where VA range has zero pages
backing it, which is supported with FI_MR_ALLOCATED unset, the provider
may not know the optimal page size during registration. Rather than use
a less efficient page size, this attribute allows applications to
specify the page size to be used.

If page size is zero, provider will select the page size.

If non-zero, page size must be supported by OS. If a specific page size
is specified for a memory region during creation, all pages later
associated with the region must be of the given size. Attaching a memory
page of a different size to a region may result in failed transfers to
or from the region.

Providers may choose to ignore page size. This will result in a provider
selected page size always being used.

## base_mr

If non-NULL, create a sub-MR from an existing memory region specified by
the base_mr field.

The sub-MR must be fully contained within the base MR; however, the
sub-MR has its own authorization keys and access rights. The following
attributes are inherited from the base MR, and as a result, are ignored
when creating the sub-MR:

iface, device, hmem_data, page_size

The sub-MR should hold a reference to the base MR. When fi_close is
called on the base MR, the call would fail if there are any outstanding
sub-MRs.

The base_mr field must be NULL if the FI_MR_DMABUF flag is set.

## sub_mr_cnt

The number of sub-MRs expected to be created from the memory region.
This value is not a limit. Instead, it is a hint to the provider to
allow provider specific optimization for sub-MR creation. For example,
the provider may reserve access keys or pre-allocation fid_mr objects.
The provider may ignore this hint.

## fi_hmem_ze_device

Returns an hmem device identifier for a level zero \<driver, device\>
tuple. The output of this call should be used to set
fi_mr_attr::device.ze for FI_HMEM_ZE interfaces. The driver and device
index values represent their 0-based positions in arrays returned from
zeDriverGet and zeDeviceGet, respectively.

## av

For memory registration being allocated against a domain configured with
FI_AV_AUTH_KEY, av is used to define the fid_av which contains the
authorization keys to be associated with the memory region. If the
domain is also opened with FI_MR_ENDPOINT, the specified AV must be the
same AV bound to the endpoint.

By default, the memory region will be associated with all authorization
keys in the AV.

## addr

If the domain was opened with FI_DIRECTED_RECV, addr can be used to
limit the memory region to a specific fi_addr_t, including fi_addr_t's
return from `fi_av_insert_auth_key`.

# NOTES

Direct access to an application's memory by a remote peer requires that
the application register the targeted memory buffer(s). This is
typically done by calling one of the fi_mr_reg\* routines. For
FI_MR_PROV_KEY, the provider will return a key that must be used by the
peer when accessing the memory region. The application is responsible
for transferring this key to the peer. If FI_MR_RAW mode has been set,
the key must be retrieved using the fi_mr_raw_attr function.

FI_MR_RAW allows support for providers that require more than 8-bytes
for their protection keys or need additional setup before a key can be
used for transfers. After a raw key has been retrieved, it must be
exchanged with the remote peer. The peer must use fi_mr_map_raw to
convert the raw key into a usable 64-bit key. The mapping must be done
even if the raw key is 64-bits or smaller.

The raw key support functions are usable with all registered memory
regions, even if FI_MR_RAW has not been set. It is recommended that
portable applications target using those interfaces; however, their use
does carry extra message and memory footprint overhead, making it less
desirable for highly scalable apps.

There may be cases where device peer to peer support should not be used
or cannot be used, such as when the PCIe ACS configuration does not
permit the transfer. The FI_HMEM_DISABLE_P2P environment variable can be
set to notify Libfabric that peer to peer transactions should not be
used. The provider may choose to perform a copy instead, or will fail
support for FI_HMEM if the provider is unable to do that.

# FLAGS

The follow flag may be specified to any memory registration call.

*FI_RMA_EVENT*
:   This flag indicates that the specified memory region will be
    associated with a completion counter used to count RMA operations
    that access the MR.

*FI_RMA_PMEM*
:   This flag indicates that the underlying memory region is backed by
    persistent memory and will be used in RMA operations. It must be
    specified if persistent completion semantics or persistent data
    transfers are required when accessing the registered region.

*FI_HMEM_DEVICE_ONLY*
:   This flag indicates that the memory is only accessible by a device.
    Which device is specified by the fi_mr_attr fields iface and device.
    This refers to memory regions that were allocated using a device API
    AllocDevice call (as opposed to using the host allocation or
    unified/shared memory allocation). This flag is only usable for
    domains opened with FI_HMEM capability support.

*FI_HMEM_HOST_ALLOC*
:   This flag indicates that the memory is owned by the host only.
    Whether it can be accessed by the device is implementation
    dependent. The fi_mr_attr field iface is still used to identify the
    device API, but the field device is ignored. This refers to memory
    regions that were allocated using a device API AllocHost call (as
    opposed to using malloc-like host allocation, unified/shared memory
    allocation, or AllocDevice). This flag is only usable for domains
    opened with FI_HMEM capability support.

*FI_MR_DMABUF*
:   This flag indicates that the memory region to registered is a
    DMA-buf backed region. When set, the region is specified through the
    dmabuf field of the fi_mr_attr structure. This flag is only usable
    for domains opened with FI_HMEM capability support.

*FI_MR_SINGLE_USE*
:   This flag indicates that the memory region is only used for a single
    operation. After the operation is complete, the key associated with
    the memory region is automatically invalidated and can no longer be
    used for remote access.

*FI_AUTH_KEY*
:   Only valid with domains configured with FI_AV_AUTH_KEY. When used
    with fi_mr_regattr, this flag denotes that the
    fi_mr_auth_key::src_addr field contains an authorization key
    fi_addr_t (i.e. fi_addr_t returned from fi_av_insert_auth_key)
    instead of an endpoint fi_addr_t (i.e. fi_addr_t return from
    fi_av_insert / fi_av_insertsvc / fi_av_remove).

# MEMORY DOMAINS

Memory domains identify the physical separation of memory which may or
may not be accessible through the same virtual address space.
Traditionally, applications only dealt with a single memory domain, that
of host memory tightly coupled with the system CPUs. With the
introduction of device and non-uniform memory subsystems, applications
often need to be aware of which memory domain a particular virtual
address maps to.

As a general rule, separate physical devices can be considered to have
their own memory domains. For example, a NIC may have user accessible
memory, and would be considered a separate memory domain from memory on
a GPU. Both the NIC and GPU memory domains are separate from host system
memory. Individual GPUs or computation accelerators may have distinct
memory domains, or may be connected in such a way (e.g. a GPU specific
fabric) that all GPUs would belong to the same memory domain.
Unfortunately, identifying memory domains is specific to each system and
its physical and/or virtual configuration.

Understanding memory domains in heterogenous memory environments is
important as it can impact data ordering and visibility as viewed by an
application. It is also important to understand which memory domain an
application is most tightly coupled to. In most cases, applications are
tightly coupled to host memory. However, an application running directly
on a GPU or NIC may be more tightly coupled to memory associated with
those devices.

Memory regions are often associated with a single memory domain. The
domain is often indicated by the fi_mr_attr iface and device fields.
Though it is possible for physical pages backing a virtual memory region
to migrate between memory domains based on access patterns. For example,
the physical pages referenced by a virtual address range could migrate
between host memory and GPU memory, depending on which computational
unit is actively using it.

See the [`fi_endpoint`(3)](fi_endpoint.3.html) and
[`fi_cq`(3)](fi_cq.3.html) man pages for addition discussion on message,
data, and completion ordering semantics, including the impact of memory
domains.

# RETURN VALUES

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned.

Fabric errno values are defined in `rdma/fi_errno.h`.

# ERRORS

*-FI_ENOKEY*
:   The requested_key is already in use.

*-FI_EKEYREJECTED*
:   The requested_key is not available. They key may be out of the range
    supported by the provider, or the provider may not support
    user-requested memory registration keys.

*-FI_ENOSYS*
:   Returned by fi_mr_bind if the provider does not support reporting
    events based on access to registered memory regions.

*-FI_EBADFLAGS*
:   Returned if the specified flags are not supported by the provider.

# MEMORY REGISTRATION CACHE

Many hardware NICs accessed by libfabric require that data buffers be
registered with the hardware while the hardware accesses it. This
ensures that the virtual to physical address mappings for those buffers
do not change while the transfer is occurring. The performance impact of
registering memory regions can be significant. As a result, some
providers make use of a registration cache, particularly when working
with applications that are unable to manage their own network buffers. A
registration cache avoids the overhead of registering and unregistering
a data buffer with each transfer.

If a registration cache is going to be used for host and device memory,
the device must support unified virtual addressing. If the device does
not support unified virtual addressing, either an additional
registration cache is required to track this device memory, or device
memory cannot be cached.

As a general rule, if hardware requires the FI_MR_LOCAL mode bit
described above, but this is not supported by the application, a memory
registration cache *may* be in use. The following environment variables
may be used to configure registration caches.

*FI_MR_CACHE_MAX_SIZE*
:   This defines the total number of bytes for all memory regions that
    may be tracked by the cache. If not set, the cache has no limit on
    how many bytes may be registered and cached. Setting this will
    reduce the amount of memory that is not actively being used as part
    of a data transfer that is registered with a provider. By default,
    the cache size is unlimited.

*FI_MR_CACHE_MAX_COUNT*
:   This defines the total number of memory regions that may be
    registered with the cache. If not set, a default limit is chosen.
    Setting this will reduce the number of regions that are registered,
    regardless of their size, which are not actively being used as part
    of a data transfer. Setting this to zero will disable registration
    caching.

*FI_MR_CACHE_MONITOR*
:   The cache monitor is responsible for detecting system memory
    (FI_HMEM_SYSTEM) changes made between the virtual addresses used by
    an application and the underlying physical pages. Valid monitor
    options are: userfaultfd, memhooks, kdreg2, and disabled. Selecting
    disabled will turn off the registration cache. Userfaultfd is a
    Linux kernel feature used to report virtual to physical address
    mapping changes to user space. Memhooks operates by intercepting
    relevant memory allocation and deallocation calls which may result
    in the mappings changing, such as malloc, mmap, free, etc. Note that
    memhooks operates at the elf linker layer, and does not use glibc
    memory hooks. Kdreg2 is supplied as a loadable Linux kernel module.

*FI_MR_CUDA_CACHE_MONITOR_ENABLED*
:   The CUDA cache monitor is responsible for detecting CUDA device
    memory (FI_HMEM_CUDA) changes made between the device virtual
    addresses used by an application and the underlying device physical
    pages. Valid monitor options are: 0 or 1. Note that the CUDA memory
    monitor requires a CUDA toolkit version with unified virtual
    addressing enabled.

*FI_MR_ROCR_CACHE_MONITOR_ENABLED*
:   The ROCR cache monitor is responsible for detecting ROCR device
    memory (FI_HMEM_ROCR) changes made between the device virtual
    addresses used by an application and the underlying device physical
    pages. Valid monitor options are: 0 or 1. Note that the ROCR memory
    monitor requires a ROCR version with unified virtual addressing
    enabled.

*FI_MR_ZE_CACHE_MONITOR_ENABLED*
:   The ZE cache monitor is responsible for detecting oneAPI Level Zero
    device memory (FI_HMEM_ZE) changes made between the device virtual
    addresses used by an application and the underlying device physical
    pages. Valid monitor options are: 0 or 1.

More direct access to the internal registration cache is possible
through the fi_open() call, using the "mr_cache" service name. Once
opened, custom memory monitors may be installed. A memory monitor is a
component of the cache responsible for detecting changes in virtual to
physical address mappings. Some level of control over the cache is
possible through the above mentioned environment variables.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_rma`(3)](fi_rma.3.html),
[`fi_msg`(3)](fi_msg.3.html), [`fi_atomic`(3)](fi_atomic.3.html)

{% include JB/setup %}

# NAME

fi_nic - Fabric network interface card attributes

# NETWORK INTERFACE CARD ATTRIBUTES

The fid_nic structure defines attributes for a struct fi_info that is
directly associated with underlying networking hardware and may be
returned directly from calling [`fi_getinfo`(3)](fi_getinfo.3.html). The
format of fid_nic and the related substructures are defined below.

Note that not all fields of all structures may be available. Unavailable
or fields that are not applicable to the indicated device will be set to
NULL or 0.

``` c
struct fid_nic {
    struct fid             fid;
    struct fi_device_attr *device_attr;
    struct fi_bus_attr    *bus_attr;
    struct fi_link_attr   *link_attr;
    void                  *prov_attr;
};

struct fi_device_attr {
    char *name;
    char *device_id;
    char *device_version;
    char *vendor_id;
    char *driver;
    char *firmware;
};

struct fi_pci_attr {
    uint16_t domain_id;
    uint8_t  bus_id;
    uint8_t  device_id;
    uint8_t  function_id;
};

struct fi_bus_attr {
    enum fi_bus_type       bus_type;
    union {
        struct fi_pci_attr pci;
    } attr;
};

struct fi_link_attr {
    char               *address;
    size_t             mtu;
    size_t             speed;
    enum fi_link_state state;
    char               *network_type;
};
```

## Device Attributes

Device attributes are used to identify the specific virtual or hardware
NIC associated with an fi_info structure.

*name*
:   The operating system name associated with the device. This may be a
    logical network interface name (e.g. eth0 or eno1) or an absolute
    filename.

*device_id*
:   This is a vendor specific identifier for the device or product.

*device_version*
:   Indicates the version of the device.

*vendor_id*
:   Indicates the name of the vendor that distributes the NIC.

*driver*
:   The name of the driver associated with the device

*firmware*
:   The device's firmware version.

## Bus Attributes

The bus attributes are used to identify the physical location of the NIC
in the system.

*bus_type*
:   Indicates the type of system bus where the NIC is located. Valid
    values are FI_BUS_PCI or FI_BUS_UNKNOWN.

*attr.pci.domain_id*
:   The domain where the PCI bus is located. Valid only if bus_type is
    FI_BUS_PCI.

*attr.pci.bus_id*
:   The PCI bus identifier where the device is located. Valid only if
    bus_type is FI_BUS_PCI.

*attr.pci.device_id*
:   The identifier on the PCI bus where the device is located. Valid
    only if bus_type is FI_BUS_PCI.

*attr.pci.function_id*
:   The function on the device being referenced. Valid only if bus_type
    is FI_BUS_PCI.

## Link Attributes

Link attributes describe low-level details about the network connection
into the fabric.

*address*
:   The primary link-level address associated with the NIC, such as a
    MAC address. If multiple addresses are available, only one will be
    reported.

*mtu*
:   The maximum transfer unit of link level frames or packets, in bytes.

*speed*
:   The active link data rate, given in bits per second.

*state*
:   The current physical port state. Possible values are
    FI_LINK_UNKNOWN, FI_LINK_DOWN, and FI_LINK_UP, to indicate if the
    port state is unknown or not applicable (unknown), inactive (down),
    or active (up).

*network_type*
:   Specifies the type of network interface currently active, such as
    Ethernet or InfiniBand.

## Provider Attributes

Provider attributes reference provider specific details of the device.
These attributes are both provider and device specific. The attributes
can be interpreted by [`fi_tostr`(3)](fi_tostr.3.html). Applications may
also use the other attribute fields, such as related fi_fabric_attr:
prov_name field, to determine an appropriate structure to cast the
attributes. The format and definition of this field is outside the scope
of the libfabric core framework, but may be available as part of a
provider specific header file included with libfabric package.

# NOTES

The fid_nic structure is returned as part of a call to
[`fi_getinfo`(3)](fi_getinfo.3.html). It is automatically freed as part
of calling [`fi_freeinfo`(3)](fi_freeinfo.3.html)

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_export_fid / fi_import_fid
:   Share a fabric object between different providers or resources

struct fid_peer_av
:   An address vector sharable between independent providers

struct fid_peer_av_set
:   An AV set sharable between independent providers

struct fid_peer_cq
:   A completion queue that may be shared between independent providers

struct fid_peer_cntr
:   A counter that may be shared between independent providers

struct fid_peer_srx
:   A shared receive context that may be shared between independent
    providers

# SYNOPSIS

``` c
#include <rdma/fabric.h>
#include <rdma/fi_ext.h>
#include <rdma/providers/fi_peer.h>

int fi_export_fid(struct fid *fid, uint64_t flags,
    struct fid **expfid, void *context);

int fi_import_fid(struct fid *fid, struct fid *expfid, uint64_t flags);
```

# ARGUMENTS

*fid*
:   Returned fabric identifier for opened object.

*expfid*
:   Exported fabric object that may be shared with another provider.

*flags*
:   Control flags for the operation.

\*context:
:   User defined context that will be associated with a fabric object.

# DESCRIPTION

NOTICE: The peer APIs describe by this man page are developmental and
may change between libfabric versions. The data structures and API
definitions should not be considered stable between versions. Providers
being used as peers must target the same libfabric version.

Functions defined in this man page are typically used by providers to
communicate with other providers, known as peer providers, or by other
libraries to communicate with the libfabric core, known as peer
libraries. Most middleware and applications should not need to access
this functionality, as the documentation mainly targets provider
developers.

Peer providers are a way for independently developed providers to be
used together in a tight fashion, such that layering overhead and
duplicate provider functionality can be avoided. Peer providers are
linked by having one provider export specific functionality to another.
This is done by having one provider export a sharable fabric object
(fid), which is imported by one or more peer providers.

As an example, a provider which uses TCP to communicate with remote
peers may wish to use the shared memory provider to communicate with
local peers. To remove layering overhead, the TCP based provider may
export its completion queue and shared receive context and import those
into the shared memory provider.

The general mechanisms used to share fabric objects between peer
providers are similar, independent from the object being shared.
However, because the goal of using peer providers is to avoid overhead,
providers must be explicitly written to support the peer provider
mechanisms.

When importing any shared fabric object into a peer, the owner will
create a separate fid_peer\_\* for each peer provider it intends to
import into. The owner will pass this unique fid_peer\_\* into each peer
through the context parameter of the init call for the resource
(i.e. fi_cq_open, fi_srx_context, fi_cntr_open, etc). The
fi_peer\_**context will indicate the owner-allocated fid_peer** for the
peer to use but is temporary for the init call and may not be accessed
by the peer after initialization. The peer will set just the peer_ops of
the owner-allocated fid and save a reference to the imported
fid_peer\_\* for use in the peer API flow. The peer will allocate its
own fid for internal uses and return that fid to the owner through the
regular fid parameter of the init call (as if it were just another
opened resource). The owner is responsible for saving the returned peer
fid from the open call in order to close it later (or to drive progress
in the case of the cq_fid).

There are two peer provider models. In the example listed above, both
peers are full providers in their own right and usable in a stand-alone
fashion. In a second model, one of the peers is known as an offload
provider. An offload provider implements a subset of the libfabric API
and targets the use of specific acceleration hardware. For example,
network switches may support collective operations, such as barrier or
broadcast. An offload provider may be written specifically to leverage
this capability; however, such a provider is not usable for general
purposes. As a result, an offload provider is paired with a main peer
provider.

# PEER AV

The peer AV allows the sharing of addressing metadata between providers.
It specifically targets the use case of having a main provider paired
with an offload provider, where the offload provider leverages the
communication that has already been established through the main
provider. In other situations, such as that mentioned above pairing a
tcp provider with a shared memory provider, each peer will likely have
their own AV that is not shared.

The setup for a peer AV is similar to the setup for a shared CQ,
described below. The owner of the AV creates a fid_peer_av object that
links back to its actual fid_av. The fid_peer_av is then imported by the
offload provider.

Peer AVs are configured by the owner calling the peer's fi_av_open()
call, passing in the FI_PEER flag, and pointing the context parameter to
struct fi_peer_av_context.

The data structures to support peer AVs are:

``` c
struct fid_peer_av;

struct fi_ops_av_owner {
    size_t  size;
    int (*query)(struct fid_peer_av *av, struct fi_av_attr *attr);
    fi_addr_t (*ep_addr)(struct fid_peer_av *av, struct fid_ep *ep);
};

struct fid_peer_av {
    struct fid fid;
    struct fi_ops_av_owner *owner_ops;
};

struct fi_peer_av_context {
    size_t size;
    struct fid_peer_av *av;
};
```

## fi_ops_av_owner::query()

This call returns current attributes for the peer AV. The owner sets the
fields of the input struct fi_av_attr based on the current state of the
AV for return to the caller.

## fi_ops_av_owner::ep_addr()

This lookup function returns the fi_addr of the address associated with
the given local endpoint. If the address of the local endpoint has not
been inserted into the AV, the function should return FI_ADDR_NOTAVAIL.

# PEER AV SET

The peer AV set allows the sharing of collective addressing data between
providers. It specifically targets the use case pairing a main provider
with a collective offload provider. The setup for a peer AV set is
similar to a shared CQ, described below. The owner of the AV set creates
a fid_peer_av_set object that links back to its fid_av_set. The
fid_peer_av_set is imported by the offload provider.

Peer AV sets are configured by the owner calling the peer's
fi_av_set_open() call, passing in the FI_PEER_AV flag, and pointing the
context parameter to struct fi_peer_av_set_context.

The data structures to support peer AV sets are:

``` c
struct fi_ops_av_set_owner {
    size_t  size;
    int (*members)(struct fid_peer_av_set *av, fi_addr_t *addr,
               size_t *count);
};

struct fid_peer_av_set {
    struct fid fid;
    struct fi_ops_av_set_owner *owner_ops;
};

struct fi_peer_av_set_context {
    size_t size;
    struct fi_peer_av_set *av_set;
};
```

## fi_ops_peer_av_owner::members

This call returns an array of AV addresses that are members of the AV
set. The size of the array is specified through the count parameter. On
return, count is set to the number of addresses in the AV set. If the
input count value is too small, the function returns -FI_ETOOSMALL.
Otherwise, the function returns an array of fi_addr values.

# PEER CQ

The peer CQ defines a mechanism by which a peer provider may insert
completions into the CQ owned by another provider. This avoids the
overhead of the libfabric user needing to access multiple CQs.

To setup a peer CQ, a provider creates a fid_peer_cq object, which links
back to the provider's actual fid_cq. The fid_peer_cq object is then
imported by a peer provider. The fid_peer_cq defines callbacks that the
providers use to communicate with each other. The provider that
allocates the fid_peer_cq is known as the owner, with the other provider
referred to as the peer. An owner may setup peer relationships with
multiple providers.

Peer CQs are configured by the owner calling the peer's fi_cq_open()
call. The owner passes in the FI_PEER flag to fi_cq_open(). When FI_PEER
is specified, the context parameter passed into fi_cq_open() must
reference a struct fi_peer_cq_context. Providers that do not support
peer CQs must fail the fi_cq_open() call with -FI_EINVAL (indicating an
invalid flag). The fid_peer_cq referenced by struct fi_peer_cq_context
must remain valid until the peer's CQ is closed.

The data structures to support peer CQs are defined as follows:

``` c
struct fi_ops_cq_owner {
    size_t  size;
    ssize_t (*write)(struct fid_peer_cq *cq, void *context, uint64_t flags,
        size_t len, void *buf, uint64_t data, uint64_t tag, fi_addr_t src);
    ssize_t (*writeerr)(struct fid_peer_cq *cq,
        const struct fi_cq_err_entry *err_entry);
};

struct fid_peer_cq {
    struct fid fid;
    struct fi_ops_cq_owner *owner_ops;
};

struct fi_peer_cq_context {
    size_t size;
    struct fid_peer_cq *cq;
};
```

For struct fid_peer_cq, the owner initializes the fid and owner_ops
fields. struct fi_ops_cq_owner is used by the peer to communicate with
the owning provider.

If manual progress is needed on the peer CQ, the owner should drive
progress by using the fi_cq_read() function with the buf parameter set
to NULL and count equal 0. The peer provider should set other functions
that attempt to read the peer's CQ (i.e. fi_cq_readerr, fi_cq_sread,
etc.) to return -FI_ENOSYS.

## fi_ops_cq_owner::write()

This call directs the owner to insert new completions into the CQ. The
fi_cq_attr::format field, along with other related attributes,
determines which input parameters are valid. Parameters that are not
reported as part of a completion are ignored by the owner, and should be
set to 0, NULL, or other appropriate value by the user. For example, if
source addressing is not returned with a completion, then the src
parameter should be set to FI_ADDR_NOTAVAIL and ignored on input.

The owner is responsible for locking, event signaling, and handling CQ
overflow. Data passed through the write callback is relative to the
user. For example, the fi_addr_t is relative to the peer's AV. The owner
is responsible for converting the address if source addressing is
needed.

(TBD: should CQ overflow push back to the user for flow control? Do we
need backoff / resume callbacks in ops_cq_user?)

## fi_ops_cq_owner::writeerr()

The behavior of this call is similar to the write() ops. It inserts a
completion indicating that a data transfer has failed into the CQ.

## EXAMPLE PEER CQ SETUP

The above description defines the generic mechanism for sharing CQs
between providers. This section outlines one possible implementation to
demonstrate the use of the APIs. In the example, provider A uses
provider B as a peer for data transfers targeting endpoints on the local
node.

    1. Provider A is configured to use provider B as a peer.  This may be coded
       into provider A or set through an environment variable.
    2. The application calls:
       fi_cq_open(domain_a, attr, &cq_a, app_context)
    3. Provider A allocates cq_a and automatically configures it to be used
       as a peer cq.
    4. Provider A takes these steps:
       allocate peer_cq and reference cq_a
       set peer_cq_context->cq = peer_cq
       set attr_b.flags |= FI_PEER
       fi_cq_open(domain_b, attr_b, &cq_b, peer_cq_context)
    5. Provider B allocates a cq, but configures it such that all completions
       are written to the peer_cq.  The cq ops to read from the cq are
       set to enosys calls.
    6. Provider B inserts its own callbacks into the peer_cq object.  It
       creates a reference between the peer_cq object and its own cq.

# PEER COUNTER

The peer counter defines a mechanism by which a peer provider may
increment value or error into the counter owned by another provider.

The setup of a peer counter is similar to the setup for a peer CQ
outlined above. The owner's counter object is imported directly into the
peer.

The data structures to support peer counters are defined as follows:

``` c
struct fi_ops_cntr_owner {
    size_t size;
    void (*inc)(struct fid_peer_cntr *cntr);
    void (*incerr)(struct fid_peer_cntr *cntr);
};

struct fid_peer_cntr {
    struct fid fid;
    struct fi_ops_cntr_owner *owner_ops;
};

struct fi_peer_cntr_context {
    size_t size;
    struct fid_peer_cntr *cntr;
};
```

Similar to the peer CQ, if manual progress is needed on the peer
counter, the owner should drive progress by using the fi_cntr_read() and
the fi_cntr_read() should do nothing but progress the peer cntr. The
peer provider should set other functions that attempt to access the
peer's cntr (i.e. fi_cntr_readerr, fi_cntr_set, etc.) to return
-FI_ENOSYS.

## fi_ops_cntr_owner::inc()

This call directs the owner to increment the value of the cntr.

## fi_ops_cntr_owner::incerr()

The behavior of this call is similar to the inc() ops. It increments the
error of the cntr indicating that a data transfer has failed into the
cntr.

# PEER DOMAIN

The peer domain allows a provider to access the operations of a domain
object of its peer. For example, an offload provider can use a peer
domain to register memory buffers with the main provider.

The setup of a peer domain is similar to the setup for a peer CQ outline
above. The owner's domain object is imported directly into the peer.

Peer domains are configured by the owner calling the peer's fi_domain2()
call. The owner passes in the FI_PEER flag to fi_domain2(). When FI_PEER
is specified, the context parameter passed into fi_domain2() must
reference a struct fi_peer_domain_context. Providers that do not support
peer domains must fail the fi_domain2() call with -FI_EINVAL. The
fid_domain referenced by struct fi_peer_domain_context must remain valid
until the peer's domain is closed.

The data structures to support peer domains are defined as follows:

``` c
struct fi_peer_domain_context {
    size_t size;
    struct fid_domain *domain;
};
```

# PEER EQ

The peer EQ defines a mechanism by which a peer provider may insert
events into the EQ owned by another provider. This avoids the overhead
of the libfabric user needing to access multiple EQs.

The setup of a peer EQ is similar to the setup for a peer CQ outline
above. The owner's EQ object is imported directly into the peer
provider.

Peer EQs are configured by the owner calling the peer's fi_eq_open()
call. The owner passes in the FI_PEER flag to fi_eq_open(). When FI_PEER
is specified, the context parameter passed into fi_eq_open() must
reference a struct fi_peer_eq_context. Providers that do not support
peer EQs must fail the fi_eq_open() call with -FI_EINVAL (indicating an
invalid flag). The fid_eq referenced by struct fi_peer_eq_context must
remain valid until the peer's EQ is closed.

The data structures to support peer EQs are defined as follows:

``` c
struct fi_peer_eq_context {
    size_t size;
    struct fid_eq *eq;
};
```

# PEER SRX

The peer SRX defines a mechanism by which peer providers may share a
common shared receive context. This avoids the overhead of having
separate receive queues, can eliminate memory copies, and ensures
correct application level message ordering.

The setup of a peer SRX is similar to the setup for a peer CQ outlined
above. A fid_peer_srx object links the owner of the SRX with the peer
provider. Peer SRXs are configured by the owner calling the peer's
fi_srx_context() call with the FI_PEER flag set. The context parameter
passed to fi_srx_context() must be a struct fi_peer_srx_context.

The owner provider initializes all elements of the fid_peer_srx and
referenced structures (fi_ops_srx_owner and fi_ops_srx_peer), with the
exception of the fi_ops_srx_peer callback functions. Those must be
initialized by the peer provider prior to returning from the
fi_srx_contex() call and are used by the owner to control peer actions.

The data structures to support peer SRXs are defined as follows:

    struct fid_peer_srx;

    /* Castable to dlist_entry */
    struct fi_peer_rx_entry {
        struct fi_peer_rx_entry *next;
        struct fi_peer_rx_entry *prev;
        struct fi_peer_srx *srx;
        fi_addr_t addr;
        size_t msg_size;
        uint64_t tag;
        uint64_t cq_data;
        uint64_t flags;
        void *context;
        size_t count;
        void **desc;
        void *peer_context;
        void *owner_context;
        struct iovec *iov;
    };

    struct fi_peer_match_attr {
        fi_addr_t addr;
        size_t msg_size;
        uint64_t tag;
    };

    struct fi_ops_srx_owner {
        size_t size;
        int (*get_msg)(struct fid_peer_srx *srx,
                       struct fi_peer_match_attr *attr,
                       struct fi_peer_rx_entry **entry);
        int (*get_tag)(struct fid_peer_srx *srx,
                       struct fi_peer_match_attr *attr,
                       uint64_t tag, struct fi_peer_rx_entry **entry);
        int (*queue_msg)(struct fi_peer_rx_entry *entry);
        int (*queue_tag)(struct fi_peer_rx_entry *entry);
        void (*foreach_unspec_addr)(struct fid_peer_srx *srx,
                      fi_addr_t (*get_addr)(struct fi_peer_rx_entry *));

        void (*free_entry)(struct fi_peer_rx_entry *entry);
    };

    struct fi_ops_srx_peer {
        size_t size;
        int (*start_msg)(struct fi_peer_rx_entry *entry);
        int (*start_tag)(struct fi_peer_rx_entry *entry);
        int (*discard_msg)(struct fi_peer_rx_entry *entry);
        int (*discard_tag)(struct fi_peer_rx_entry *entry);
    };

    struct fid_peer_srx {
        struct fid_ep ep_fid;
        struct fi_ops_srx_owner *owner_ops;
        struct fi_ops_srx_peer *peer_ops;
    };

    struct fi_peer_srx_context {
        size_t size;
        struct fid_peer_srx *srx;
    };

The ownership of structure field values and callback functions is
similar to those defined for peer CQs, relative to owner versus peer
ops.

The owner is responsible for acquiring any necessary locks before
anything that could result in peer callbacks. The following functions
are progress level functions: get_msg(), get_tag(), queue_msg(),
queue_tag(), free_entry(), start_msg(), start_tag(), discard_msg(),
discard_tag(). If needed, it is the owner's responsibility to acquire
the appropriate lock prior to calling into a peer's fi_cq_read(), or
similar, function that drives progress.

The following functions are domain level functions:
foreach_unspec_addr(). This function is used outside of message progress
flow (i.e. during fi_av_insert()). The owner of the srx is responsible
for acquiring the same lock, if needed.

## fi_peer_rx_entry

fi_peer_rx_entry defines a common receive entry for use between the
owner and peer. The entry is allocated and set by the owner and passed
between owner and peer to communicate details of the application-posted
receive entry. All fields are initialized by the owner, except in the
unexpected message case where the peer can initialize any extra
available data before queuing the message with the owner. The
peer_context and owner_context fields are only modifiable by the peer
and owner, respectively, to store extra provider-specific information.

## fi_ops_srx_owner::get_msg() / get_tag()

These calls are invoked by the peer provider to obtain the receive
buffer(s) where an incoming message should be placed. The peer provider
will pass in the relevant fields to request a matching rx_entry from the
owner. If source addressing is required, the addr will be passed in;
otherwise, the address will be set to FI_ADDR_NOT_AVAIL. The msg_size
field indicates the received message size. This field may be needed by
the owner when handling FI_MULTI_RECV or FI_PEEK. The owner will set the
peer_entry-\>msg_size field on get_msg/tag() for the owner and peer to
use later, if needed. This field will be set on both the expected and
unexpected paths. The returned rx_entry-\>iov returned from the owner
refers to the full size of the posted receive passed to the peer. The
peer provider is responsible for checking that an incoming message fits
within the provided buffer space and generating truncation errors. The
tag parameter is only used for tagged messages but must be set to 0 for
the non-tagged cases. An fi_peer_rx_entry is allocated by the owner,
whether or not a match was found. If a match was found, the owner will
return FI_SUCCESS and the rx_entry will be filled in with the known
receive fields for the peer to process accordingly. This includes the
information that was passed into the calls as well as the
rx_entry-\>flags with either FI_MSG \| FI_RECV (for get_msg()) or
FI_TAGGED \| FI_RECV (for get_tag()). The peer provider is responsible
for completing with any other flags, if needed. If no match was found,
the owner will return -FI_ENOENT; the rx_entry will still be valid but
will not match to an existing posted receive. When the peer gets
FI_ENOENT, it should allocate whatever resources it needs to process the
message later (on start_msg/tag) and set the rx_entry-\>peer_context
appropriately, followed by a call to the owner's queue_msg/tag. The get
and queue calls should be serialized. When the owner gets a matching
receive for the queued unexpected message, it will call the peer's start
function to notify the peer of the updated rx_entry (or the peer's
discard function if the message is to be discarded)

# fi_ops_srx_owner::queue_msg() / queue_tag()

Called by the peer to queue an incoming unexpected message to the srx.
Once it gets queued by the peer, the owner is responsible for starting
it once it gets matched to a receive buffer, or discard it if needed.

## fi_ops_srx_owner::foreach_unspec_addr()

Called by the peer when any addressing updates have occurred with the
peer. This triggers the owner to iterate over any entries whose address
is still unknown and call the inputed get_addr function on each to
retrieve updated address information.

# fi_ops_srx_owner:: free_entry()

Called by the peer when it is completely done using an owner-allocated
peer entry.

## fi_ops_srx_peer::start_msg() / start_tag()

These calls indicate that an asynchronous get_msg() or get_tag() has
completed and a buffer is now available to receive the message. Control
of the fi_peer_rx_entry is returned to the peer provider and has been
initialized for receiving the incoming message.

## fi_ops_srx_peer::discard_msg() / discard_tag()

Indicates that the message and data associated with the specified
fi_peer_rx_entry should be discarded. This often indicates that the
application has canceled or discarded the receive operation. No
completion should be generated by the peer provider for a discarded
message. Control of the fi_peer_rx_entry is returned to the peer
provider.

## EXAMPLE PEER SRX SETUP

The above description defines the generic mechanism for sharing SRXs
between providers. This section outlines one possible implementation to
demonstrate the use of the APIs. In the example, provider A uses
provider B as a peer for data transfers targeting endpoints on the local
node.

    1. Provider A is configured to use provider B as a peer.  This may be coded
       into provider A or set through an environment variable.
    2. The application calls:
       fi_srx_context(domain_a, attr, &srx_a, app_context)
    3. Provider A allocates srx_a and automatically configures it to be used
       as a peer srx.
    4. Provider A takes these steps:
       allocate peer_srx and reference srx_a
       set peer_srx_context->srx = peer_srx
       set attr_b.flags |= FI_PEER
       fi_srx_context(domain_b, attr_b, &srx_b, peer_srx_context)
    5. Provider B allocates an srx, but configures it such that all receive
       buffers are obtained from the peer_srx.  The srx ops to post receives are
       set to enosys calls.
    6. Provider B inserts its own callbacks into the peer_srx object.  It
       creates a reference between the peer_srx object and its own srx.

## EXAMPLE PEER SRX RECEIVE FLOW

The following outlines shows simplified, example software flows for
receive message handling using a peer SRX. The first flow demonstrates
the case where a receive buffer is waiting when the message arrives.

    1. Application calls fi_recv() / fi_trecv() on owner.
    2. Owner queues the receive buffer.
    3. A message is received by the peer provider.
    4. The peer calls owner->get_msg() / get_tag().
    5. The owner removes the queued receive buffer and returns it to
       the peer.  The get entry call will complete with FI_SUCCESS.
    6. When the peer finishes processing the message and completes it on its own
       CQ, the peer will call free_entry to free the entry with the owner.

The second case below shows the flow when a message arrives before the
application has posted the matching receive buffer.

    1. A message is received by the peer provider.
    2. The peer calls owner->get_msg() / get_tag(). If the incoming address is
       FI_ADDR_UNSPEC, the owner cannot match this message to a receive posted with
       FI_DIRECTED_RECV and can only match to receives posted with FI_ADDR_UNSPEC.
    3. The owner fails to find a matching receive buffer.
    4. The owner allocates a rx_entry with any known fields and returns -FI_ENOENT.
    5. The peer allocates any resources needed to handle the asynchronous processing
       and sets peer_context accordingly, calling the owner's queue
       function when ready to queue the unexpected message from the peer.
    6. The application calls fi_recv() / fi_trecv() on owner, posting the
       matching receive buffer.
    7. The owner matches the receive with the queued message on the peer. Note that
       the owner cannot match a directed receive with an unexpected message whose
       address is unknown.
    8. The owner removes the queued request, fills in the rest of the known fields
       and calls the peer->start_msg() / start_tag() function.
    9. When the peer finishes processing the message and completes it on its own
       CQ, the peer will call free_entry to free the entry with the owner.

Whenever a peer's addressing is updated (e.g. via fi_av_insert()), it
needs to call the owner's foreach_unspec_addr() call to trigger any
necessary updating of unknown entries. The owner is expected to iterate
over any necessary entries and call the inputed get_addr() function on
each one in order to get updated addressing information. Once the
address is known, the owner can proceed to receive directed receives
into those entries.

# fi_export_fid / fi_import_fid

The fi_export_fid function is reserved for future use.

The fi_import_fid call may be used to import a fabric object created and
owned by the libfabric user. This allows upper level libraries or the
application to override or define low-level libfabric behavior. Details
on specific uses of fi_import_fid are outside the scope of this
documentation.

# FI_PEER_TRANSFER

Providers frequently send control messages to their remote counterparts
as part of their wire protocol. For example, a provider may send an ACK
message to guarantee reliable delivery of a message or to meet a
requested completion semantic. When two or more providers are
coordinating as peers, it can be more efficient if control messages for
both peer providers go over the same transport. In some cases, such as
when one of the peers is an offload provider, it may even be required.
Peer transfers define the mechanism by which such communication occurs.

Peer transfers enable one peer to send and receive data transfers over
its associated peer. Providers that require this functionality indicate
this by setting the FI_PEER_TRANSFER flag as a mode bit,
i.e. fi_info::mode.

To use such a provider as a peer, the main, or owner, provider must
setup peer transfers by opening a peer transfer endpoint and accepting
transfers with this flag set. Setup of peer transfers involves the
following data structures:

``` c
struct fi_ops_transfer_peer {
    size_t size;
    ssize_t (*complete)(struct fid_ep *ep, struct fi_cq_tagged_entry *buf,
            fi_addr_t *src_addr);
    ssize_t (*comperr)(struct fid_ep *ep, struct fi_cq_err_entry *buf);
};

struct fi_peer_transfer_context {
    size_t size;
    struct fi_info *info;
    struct fid_ep *ep;
    struct fi_ops_transfer_peer *peer_ops;
};
```

Peer transfer contexts form a virtual link between endpoints allocated
on each of the peer providers. The setup of a peer transfer context
occurs through the fi_endpoint() API. The main provider calls
fi_endpoint() with the FI_PEER_TRANSFER mode bit set in the info
parameter, and the context parameter must reference the struct
fi_peer_transfer_context defined above.

The size field indicates the size of struct fi_peer_transfer_context
being passed to the peer. This is used for backward compatibility. The
info field is optional. If given, it defines the attributes of the main
provider's objects. It may be used to report the capabilities and
restrictions on peer transfers, such as whether memory registration is
required, maximum message sizes, data and completion ordering semantics,
and so forth. If the importing provider cannot meet these restrictions,
it must fail the fi_endpoint() call.

The peer_ops field contains callbacks from the main provider into the
peer and is used to report the completion (success or failure) of peer
initiated data transfers. The callback functions defined in struct
fi_ops_transfer_peer must be set by the peer provider before returning
from the fi_endpoint() call. Actions that the peer provider can take
from within the completion callbacks are most unrestricted, and can
include any of the following types of operations: initiation of
additional data transfers, writing events to the owner's CQ or EQ, and
memory registration/deregistration. The owner must ensure that deadlock
cannot occur prior to invoking the peer's callback should the peer
invoke any of these operations. Further, the owner must avoid recursive
calls into the completion callbacks.

# RETURN VALUE

Returns FI_SUCCESS on success. On error, a negative value corresponding
to fabric errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# SEE ALSO

[`fi_provider`(7)](fi_provider.7.html),
[`fi_provider`(3)](fi_provider.3.html), [`fi_cq`(3)](fi_cq.3.html),

{% include JB/setup %}

# NAME

fi_poll - Polling and wait set operations (deprecated)

fi_poll_open / fi_close (deprecated)
:   Open/close a polling set

fi_poll_add / fi_poll_del (deprecated)
:   Add/remove a completion queue or counter to/from a poll set.

fi_poll (deprecated)
:   Poll for progress and events across multiple completion queues and
    counters.

fi_wait_open / fi_close (deprecated)
:   Open/close a wait set

fi_wait (deprecated)
:   Waits for one or more wait objects in a set to be signaled.

fi_trywait
:   Indicate when it is safe to block on wait objects using native OS
    calls.

fi_control
:   Control wait set operation or attributes.

# SYNOPSIS

``` c
#include <rdma/fi_domain.h>

int fi_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
    struct fid_poll **pollset);

int fi_close(struct fid *pollset);

int fi_poll_add(struct fid_poll *pollset, struct fid *event_fid,
    uint64_t flags);

int fi_poll_del(struct fid_poll *pollset, struct fid *event_fid,
    uint64_t flags);

int fi_poll(struct fid_poll *pollset, void **context, int count);

int fi_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
    struct fid_wait **waitset);

int fi_close(struct fid *waitset);

int fi_wait(struct fid_wait *waitset, int timeout);

int fi_trywait(struct fid_fabric *fabric, struct fid **fids, size_t count);

int fi_control(struct fid *waitset, int command, void *arg);
```

# ARGUMENTS

*fabric*
:   Fabric provider

*domain*
:   Resource domain

*pollset*
:   Event poll set

*waitset*
:   Wait object set

*attr*
:   Poll or wait set attributes

*context*
:   On success, an array of user context values associated with
    completion queues or counters.

*fids*
:   An array of fabric descriptors, each one associated with a native
    wait object.

*count*
:   Number of entries in context or fids array.

*timeout*
:   Time to wait for a signal, in milliseconds.

*command*
:   Command of control operation to perform on the wait set.

*arg*
:   Optional control argument.

# DESCRIPTION

## fi_poll_open (deprecated)

fi_poll_open creates a new polling set. A poll set enables an optimized
method for progressing asynchronous operations across multiple
completion queues and counters and checking for their completions.

A poll set is defined with the following attributes.

``` c
struct fi_poll_attr {
    uint64_t             flags;     /* operation flags */
};
```

*flags*
:   Flags that set the default operation of the poll set. The use of
    this field is reserved and must be set to 0 by the caller.

## fi_close (deprecated)

The fi_close call releases all resources associated with a poll set. The
poll set must not be associated with any other resources prior to being
closed, otherwise the call will return -FI_EBUSY.

## fi_poll_add (deprecated)

Associates a completion queue or counter with a poll set.

## fi_poll_del (deprecated)

Removes a completion queue or counter from a poll set.

## fi_poll (deprecated)

Progresses all completion queues and counters associated with a poll set
and checks for events. If events might have occurred, contexts
associated with the completion queues and/or counters are returned.
Completion queues will return their context if they are not empty. The
context associated with a counter will be returned if the counter's
success value or error value have changed since the last time fi_poll,
fi_cntr_set, or fi_cntr_add were called. The number of contexts is
limited to the size of the context array, indicated by the count
parameter.

Note that fi_poll only indicates that events might be available. In some
cases, providers may consume such events internally, to drive progress,
for example. This can result in fi_poll returning false positives.
Applications should drive their progress based on the results of reading
events from a completion queue or reading counter values. The fi_poll
function will always return all completion queues and counters that do
have new events.

## fi_wait_open (deprecated)

fi_wait_open allocates a new wait set. A wait set enables an optimized
method of waiting for events across multiple completion queues and
counters. Where possible, a wait set uses a single underlying wait
object that is signaled when a specified condition occurs on an
associated completion queue or counter.

The properties and behavior of a wait set are defined by struct
fi_wait_attr.

``` c
struct fi_wait_attr {
    enum fi_wait_obj     wait_obj;  /* requested wait object */
    uint64_t             flags;     /* operation flags */
};
```

*wait_obj*
:   Wait sets are associated with specific wait object(s). Wait objects
    allow applications to block until the wait object is signaled,
    indicating that an event is available to be read. The following
    values may be used to specify the type of wait object associated
    with a wait set: FI_WAIT_UNSPEC, FI_WAIT_FD, FI_WAIT_MUTEX_COND
    (deprecated), and FI_WAIT_YIELD.

\- *FI_WAIT_UNSPEC*
:   Specifies that the user will only wait on the wait set using fabric
    interface calls, such as fi_wait. In this case, the underlying
    provider may select the most appropriate or highest performing wait
    object available, including custom wait mechanisms. Applications
    that select FI_WAIT_UNSPEC are not guaranteed to retrieve the
    underlying wait object.

\- *FI_WAIT_FD*
:   Indicates that the wait set should use a single file descriptor as
    its wait mechanism, as exposed to the application. Internally, this
    may require the use of epoll in order to support waiting on a single
    file descriptor. File descriptor wait objects must be usable in the
    POSIX select(2) and poll(2), and Linux epoll(7) routines (if
    available). Provider signal an FD wait object by marking it as
    readable or with an error.

\- *FI_WAIT_MUTEX_COND* (deprecated)
:   Specifies that the wait set should use a pthread mutex and cond
    variable as a wait object.

\- *FI_WAIT_POLLFD*
:   This option is similar to FI_WAIT_FD, but allows the wait mechanism
    to use multiple file descriptors as its wait mechanism, as viewed by
    the application. The use of FI_WAIT_POLLFD can eliminate the need to
    use epoll to abstract away needing to check multiple file
    descriptors when waiting for events. The file descriptors must be
    usable in the POSIX select(2) and poll(2) routines, and match
    directly to being used with poll. See the NOTES section below for
    details on using pollfd.

\- *FI_WAIT_YIELD*
:   Indicates that the wait set will wait without a wait object but
    instead yield on every wait.

*flags*
:   Flags that set the default operation of the wait set. The use of
    this field is reserved and must be set to 0 by the caller.

## fi_close (deprecated)

The fi_close call releases all resources associated with a wait set. The
wait set must not be bound to any other opened resources prior to being
closed, otherwise the call will return -FI_EBUSY.

## fi_wait (deprecated)

Waits on a wait set until one or more of its underlying wait objects is
signaled.

## fi_trywait

The fi_trywait call was introduced in libfabric version 1.3. The
behavior of using native wait objects without the use of fi_trywait is
provider specific and should be considered non-deterministic.

The fi_trywait() call is used in conjunction with native operating
system calls to block on wait objects, such as file descriptors. The
application must call fi_trywait and obtain a return value of FI_SUCCESS
prior to blocking on a native wait object. Failure to do so may result
in the wait object not being signaled, and the application not observing
the desired events. The following pseudo-code demonstrates the use of
fi_trywait in conjunction with the OS select(2) call.

``` c
fi_control(&cq->fid, FI_GETWAIT, (void *) &fd);
FD_ZERO(&fds);
FD_SET(fd, &fds);

while (1) {
    if (fi_trywait(&cq, 1) == FI_SUCCESS)
        select(fd + 1, &fds, NULL, &fds, &timeout);

    do {
        ret = fi_cq_read(cq, &comp, 1);
    } while (ret > 0);
}
```

fi_trywait() will return FI_SUCCESS if it is safe to block on the wait
object(s) corresponding to the fabric descriptor(s), or -FI_EAGAIN if
there are events queued on the fabric descriptor or if blocking could
hang the application.

The call takes an array of fabric descriptors. For each wait object that
will be passed to the native wait routine, the corresponding fabric
descriptor should first be passed to fi_trywait. All fabric descriptors
passed into a single fi_trywait call must make use of the same
underlying wait object type.

The following types of fabric descriptors may be passed into fi_trywait:
event queues, completion queues, counters, and wait sets. Applications
that wish to use native wait calls should select specific wait objects
when allocating such resources. For example, by setting the item's
creation attribute wait_obj value to FI_WAIT_FD.

In the case the wait object to check belongs to a wait set, only the
wait set itself needs to be passed into fi_trywait. The fabric resources
associated with the wait set do not.

On receiving a return value of -FI_EAGAIN from fi_trywait, an
application should read all queued completions and events, and call
fi_trywait again before attempting to block. Applications can make use
of a fabric poll set to identify completion queues and counters that may
require processing.

## fi_control

The fi_control call is used to access provider or implementation
specific details of a fids that support blocking calls, such as wait
sets, completion queues, counters, and event queues. Access to the wait
set or fid should be serialized across all calls when fi_control is
invoked, as it may redirect the implementation of wait set operations.
The following control commands are usable with a wait set or fid.

*FI_GETWAIT (void \*\*)*
:   This command allows the user to retrieve the low-level wait object
    associated with a wait set or fid. The format of the wait set is
    specified during wait set creation, through the wait set attributes.
    The fi_control arg parameter should be an address where a pointer to
    the returned wait object will be written. This should be an 'int \*'
    for FI_WAIT_FD, 'struct fi_mutex_cond' for FI_WAIT_MUTEX_COND
    (deprecated), or 'struct fi_wait_pollfd' for FI_WAIT_POLLFD. Support
    for FI_GETWAIT is provider specific.

*FI_GETWAITOBJ (enum fi_wait_obj \*)*
:   This command returns the type of wait object associated with a wait
    set or fid.

# RETURN VALUES

Returns FI_SUCCESS on success. On error, a negative value corresponding
to fabric errno is returned.

Fabric errno values are defined in `rdma/fi_errno.h`.

fi_poll
:   On success, if events are available, returns the number of entries
    written to the context array.

# NOTES

In many situations, blocking calls may need to wait on signals sent to a
number of file descriptors. For example, this is the case for socket
based providers, such as tcp and udp, as well as utility providers such
as multi-rail. For simplicity, when epoll is available, it can be used
to limit the number of file descriptors that an application must
monitor. The use of epoll may also be required in order to support
FI_WAIT_FD.

However, in order to support waiting on multiple file descriptors on
systems where epoll support is not available, or where epoll performance
may negatively impact performance, FI_WAIT_POLLFD provides this
mechanism. A significant different between using POLLFD versus FD wait
objects is that with FI_WAIT_POLLFD, the file descriptors may change
dynamically. As an example, the file descriptors associated with a
completion queues' wait set may change as endpoint associations with the
CQ are added and removed.

Struct fi_wait_pollfd is used to retrieve all file descriptors for fids
using FI_WAIT_POLLFD to support blocking calls.

``` c
struct fi_wait_pollfd {
    uint64_t      change_index;
    size_t        nfds;
    struct pollfd *fd;
};
```

*change_index*
:   The change_index may be used to determine if there have been any
    changes to the file descriptor list. Anytime a file descriptor is
    added, removed, or its events are updated, this field is incremented
    by the provider. Applications wishing to wait on file descriptors
    directly should cache the change_index value. Before blocking on
    file descriptor events, the app should use fi_control() to retrieve
    the current change_index and compare that against its cached value.
    If the values differ, then the app should update its file descriptor
    list prior to blocking.

*nfds*
:   On input to fi_control(), this indicates the number of entries in
    the struct pollfd \* array. On output, this will be set to the
    number of entries needed to store the current number of file
    descriptors. If the input value is smaller than the output value,
    fi_control() will return the error -FI_ETOOSMALL. Note that setting
    nfds = 0 allows an efficient way of checking the change_index.

*fd*
:   This points to an array of struct pollfd entries. The number of
    entries is specified through the nfds field. If the number of needed
    entries is less than or equal to the number of entries available,
    the struct pollfd array will be filled out with a list of file
    descriptors and corresponding events that can be used in the
    select(2) and poll(2) calls.

The change_index is updated only when the file descriptors associated
with the pollfd file set has changed. Checking the change_index is an
additional step needed when working with FI_WAIT_POLLFD wait objects
directly. The use of the fi_trywait() function is still required if
accessing wait objects directly.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cntr`(3)](fi_cntr.3.html),
[`fi_eq`(3)](fi_eq.3.html)

{% include JB/setup %}

# NAME

fi_prov_ini - External provider entry point

fi_param_define / fi_param_get
:   Register and retrieve environment variables with the libfabric core

fi_log_enabled / fi_log_ready / fi_log
:   Control and output debug logging information.

fi_open / fi_import / fi_close
:   Open and import a named library object

fi_import_log
:   Import new logging callbacks

# SYNOPSIS

``` c
#include <rdma/fabric.h>
#include <rdma/prov/fi_prov.h>

struct fi_provider* fi_prov_ini(void);

int fi_param_define(const struct fi_provider *provider, const char *param_name,
    enum fi_param_type type, const char *help_string_fmt, ...);

int fi_param_get_str(struct fi_provider *provider, const char *param_name,
    char **value);

int fi_param_get_int(struct fi_provider *provider, const char *param_name,
    int *value);

int fi_param_get_bool(struct fi_provider *provider, const char *param_name,
    int *value);

int fi_param_get_size_t(struct fi_provider *provider, const char *param_name,
    size_t *value);
```

``` c
#include <rdma/fabric.h>
#include <rdma/prov/fi_prov.h>
#include <rdma/prov/fi_log.h>

int fi_log_enabled(const struct fi_provider *prov, enum fi_log_level level,
    enum fi_log_subsys subsys);

int fi_log_ready(const struct fi_provider *prov, enum fi_log_level level,
    enum fi_log_subsys subsys, uint64_t *showtime);

void fi_log(const struct fi_provider *prov, enum fi_log_level level,
    enum fi_log_subsys subsys, const char *func, int line,
    const char *fmt, ...);
```

``` c
#include <rdma/fabric.h>

int fi_open(uint32_t version, const char *name, void *attr,
    size_t attr_len, uint64_t flags, struct fid **fid, void *context);

static inline int fi_import(uint32_t version, const char *name, void *attr,
                size_t attr_len, uint64_t flags, struct fid *fid,
                void *context);

int fi_close(struct fid *fid);
```

``` c
#include <rdma/fabric.h>
#include <rdma/fi_ext.h>

static inline int fi_import_log(uint32_t version, uint64_t flags,
                struct fid_logging *log_fid);
```

# ARGUMENTS

*provider*
:   Reference to the provider.

*version*
:   API version requested by application.

*name*
:   Well-known name of the library object to open.

*attr*
:   Optional attributes of object to open.

*attr_len*
:   Size of any attribute structure passed to fi_open. Should be 0 if no
    attributes are give.

*fid*
:   Returned fabric identifier for opened object.

# DESCRIPTION

A fabric provider implements the application facing software interfaces
needed to access network specific protocols, drivers, and hardware. The
interfaces and structures defined by this man page are exported by the
libfabric library, but are targeted for provider implementations, rather
than for direct use by most applications.

Integrated providers are those built directly into the libfabric library
itself. External providers are loaded dynamically by libfabric at
initialization time. External providers must be in a standard library
path or in the libfabric library search path as specified by environment
variable. Additionally, external providers must be named with the suffix
"-fi.so" at the end of the name.

Named objects are special purpose resources which are accessible
directly to applications. They may be used to enhance or modify the
behavior of library core. For details, see the fi_open call below.

## fi_prov_ini

This entry point must be defined by external providers. On loading,
libfabric will invoke fi_prov_ini() to retrieve the provider's
fi_provider structure. Additional interactions between the libfabric
core and the provider will be through the interfaces defined by that
struct.

## fi_param_define

Defines a configuration parameter for use by a specified provider. The
help_string and param_name arguments must be non-NULL, help_string must
additionally be non-empty. They are copied internally and may be freed
after calling fi_param_define.

## fi_param_get

Gets the value of a configuration parameter previously defined using
fi_param_define(). The value comes from the environment variable name of
the form FI\_`<provider_name>`{=html}\_`<param_name>`{=html}, all
converted to upper case.

If the parameter was previously defined and the user set a value,
FI_SUCCESS is returned and (\*value) points to the retrieved value.

If the parameter name was previously defined, but the user did not set a
value, -FI_ENODATA is returned and the value of (\*value) is unchanged.

If the parameter name was not previously defined via fi_param_define(),
-FI_ENOENT will be returned and the value of (\*value) is unchanged.

If the value in the environment is not valid for the parameter type,
-FI_EINVAL will be returned and the value of (\*value) is unchanged.

## fi_log_enabled / fi_log_ready / fi_log

These functions control debug and informational logging output.
Providers typically access these functions through the FI_LOG and
related macros in fi_log.h and do not call these function directly.

## fi_open

Open a library resource using a well-known name. This feature allows
applications and providers a mechanism which can be used to modify or
enhance core library services and behavior. The details are specific
based on the requested object name. Most applications will not need this
level of control.

The library API version known to the application should be provided
through the version parameter. The use of attributes is object
dependent. If required, attributes should be provided through the attr
parameter, with attr_len set to the size of the referenced attribute
structure. The following is a list of published names, along with
descriptions of the service or resource to which they correspond.

*mr_cache*
:   The mr_cache object references the internal memory registration
    cache used by the different providers. Additional information on the
    cache is available in the `fi_mr(3)` man page.

*logging*
:   The logging object references the internal logging subsystem used by
    the different providers. Once opened, custom logging callbacks may
    be installed. Can be opened only once and only the last import is
    used if imported multiple times.

## fi_import

This helper function is a combination of `fi_open` and `fi_import_fid`.
It may be used to import a fabric object created and owned by the
libfabric user. This allows the upper level libraries or the application
to override or define low-level libfabric behavior.

## fi_import_log

Helper function to override the low-level libfabric's logging system
with new callback functions.

``` c
struct fi_ops_log {
    size_t size;
    int (*enabled)(const struct fi_provider *prov, enum fi_log_level level,
               enum fi_log_subsys subsys, uint64_t flags);
    int (*ready)(const struct fi_provider *prov, enum fi_log_level level,
             enum fi_log_subsys subsys, uint64_t flags, uint64_t *showtime);
    void (*log)(const struct fi_provider *prov, enum fi_log_level level,
            enum fi_log_subsys subsys, const char *func, int line,
            const char *msg);
};

struct fid_logging {
    struct fid          fid;
    struct fi_ops_log   *ops;
};
```

# PROVIDER INTERFACE

The fi_provider structure defines entry points for the libfabric core to
use to access the provider. All other calls into a provider are through
function pointers associated with allocated objects.

``` c
struct fi_provider {
    uint32_t version;
    uint32_t fi_version;
    struct fi_context context;
    const char *name;
    int (*getinfo)(uint32_t version, const char *node, const char *service,
            uint64_t flags, const struct fi_info *hints,
            struct fi_info **info);
    int (*fabric)(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
            void *context);
    void    (*cleanup)(void);
};
```

## version

The provider version. For providers integrated with the library, this is
often the same as the library version.

## fi_version

The library interface version that the provider was implemented against.
The provider's fi_version must be greater than or equal to an
application's requested api version for the application to use the
provider. It is a provider's responsibility to support older versions of
the api if it wishes to supports legacy applications. For integrated
providers

# RETURN VALUE

Returns FI_SUCCESS on success. On error, a negative value corresponding
to fabric errno is returned. Fabric errno values are defined in
`rdma/fi_errno.h`.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_getinfo`(3)](fi_getinfo.3.html)
[`fi_mr`(3)](fi_mr.3.html),

{% include JB/setup %}

# NAME

fi_rma - Remote memory access operations

fi_read / fi_readv / fi_readmsg
:   Initiates a read from remote memory

fi_write / fi_writev / fi_writemsg fi_inject_write / fi_writedata :
Initiate a write to remote memory

# SYNOPSIS

``` c
#include <rdma/fi_rma.h>

ssize_t fi_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
    fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context);

ssize_t fi_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
    size_t count, fi_addr_t src_addr, uint64_t addr, uint64_t key,
    void *context);

ssize_t fi_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
    uint64_t flags);

ssize_t fi_write(struct fid_ep *ep, const void *buf, size_t len,
    void *desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
    void *context);

ssize_t fi_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
    size_t count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
    void *context);

ssize_t fi_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
    uint64_t flags);

ssize_t fi_inject_write(struct fid_ep *ep, const void *buf, size_t len,
    fi_addr_t dest_addr, uint64_t addr, uint64_t key);

ssize_t fi_writedata(struct fid_ep *ep, const void *buf, size_t len,
    void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t addr,
    uint64_t key, void *context);

ssize_t fi_inject_writedata(struct fid_ep *ep, const void *buf, size_t len,
    uint64_t data, fi_addr_t dest_addr, uint64_t addr, uint64_t key);
```

# ARGUMENTS

*ep*
:   Fabric endpoint on which to initiate read or write operation.

*buf*
:   Local data buffer to read into (read target) or write from (write
    source)

*len*
:   Length of data to read or write, specified in bytes. Valid transfers
    are from 0 bytes up to the endpoint's max_msg_size.

*iov*
:   Vectored data buffer.

*count*
:   Count of vectored data entries.

*addr*
:   Address of remote memory to access. This will be the virtual address
    of the remote region in the case of FI_MR_BASIC, or the offset from
    the starting address in the case of FI_MR_SCALABLE.

*key*
:   Protection key associated with the remote memory.

*desc*
:   Descriptor associated with the local data buffer See
    [`fi_mr`(3)](fi_mr.3.html).

*data*
:   Remote CQ data to transfer with the operation.

*dest_addr*
:   Destination address for connectionless write transfers. Ignored for
    connected endpoints.

*src_addr*
:   Source address to read from for connectionless transfers. Ignored
    for connected endpoints.

*msg*
:   Message descriptor for read and write operations.

*flags*
:   Additional flags to apply for the read or write operation.

*context*
:   User specified pointer to associate with the operation. This
    parameter is ignored if the operation will not generate a successful
    completion, unless an op flag specifies the context parameter be
    used for required input.

# DESCRIPTION

RMA (remote memory access) operations are used to transfer data directly
between a local data buffer and a remote data buffer. RMA transfers
occur on a byte level granularity, and no message boundaries are
maintained.

The write functions -- fi_write, fi_writev, fi_writemsg,
fi_inject_write, and fi_writedata -- are used to transmit data into a
remote memory buffer. The main difference between write functions are
the number and type of parameters that they accept as input. Otherwise,
they perform the same general function.

The read functions -- fi_read, fi_readv, and fi_readmsg -- are used to
transfer data from a remote memory region into local data buffer(s).
Similar to the write operations, read operations operate asynchronously.
Users should not touch the posted data buffer(s) until the read
operation has completed.

Completed RMA operations are reported to the user through one or more
completion queues associated with the endpoint. Users provide context
which are associated with each operation, and is returned to the user as
part of the completion. See fi_cq for completion event details.

By default, the remote endpoint does not generate an event or notify the
user when a memory region has been accessed by an RMA read or write
operation. However, immediate data may be associated with an RMA write
operation. RMA writes with immediate data will generate a completion
entry at the remote endpoint, so that the immediate data may be
delivered.

## fi_write

The call fi_write transfers the data contained in the user-specified
data buffer to a remote memory region.

## fi_writev

The fi_writev call adds support for a scatter-gather list to fi_write.
The fi_writev transfers the set of data buffers referenced by the iov
parameter to the remote memory region.

## fi_writemsg

The fi_writemsg call supports data transfers over both connected and
connectionless endpoints, with the ability to control the write
operation per call through the use of flags. The fi_writemsg function
takes a struct fi_msg_rma as input.

``` c
struct fi_msg_rma {
    const struct iovec *msg_iov;     /* local scatter-gather array */
    void               **desc;       /* operation descriptor */
    size_t             iov_count;    /* # elements in msg_iov */
    fi_addr_t          addr;        /* optional endpoint address */
    const struct fi_rma_iov *rma_iov;/* remote SGL */
    size_t             rma_iov_count;/* # elements in rma_iov */
    void               *context;     /* user-defined context */
    uint64_t           data;         /* optional immediate data */
};

struct fi_rma_iov {
    uint64_t           addr;         /* target RMA address */
    size_t             len;          /* size of target buffer */
    uint64_t           key;          /* access key */
};
```

## fi_inject_write

The write inject call is an optimized version of fi_write. It provides
similar completion semantics as fi_inject [`fi_msg`(3)](fi_msg.3.html).

## fi_writedata

The write data call is similar to fi_write, but allows for the sending
of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the transfer.

## fi_inject_writedata

The inject write data call is similar to fi_inject_write, but allows for
the sending of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of
the transfer.

## fi_read

The fi_read call requests that the remote endpoint transfer data from
the remote memory region into the local data buffer.

## fi_readv

The fi_readv call adds support for a scatter-gather list to fi_read. The
fi_readv transfers data from the remote memory region into the set of
data buffers referenced by the iov parameter.

## fi_readmsg

The fi_readmsg call supports data transfers over both connected and
connectionless endpoints, with the ability to control the read operation
per call through the use of flags. The fi_readmsg function takes a
struct fi_msg_rma as input.

# FLAGS

The fi_readmsg and fi_writemsg calls allow the user to specify flags
which can change the default data transfer operation. Flags specified
with fi_readmsg / fi_writemsg override most flags previously configured
with the endpoint, except where noted (see fi_endpoint.3). The following
list of flags are usable with fi_readmsg and/or fi_writemsg.

*FI_REMOTE_CQ_DATA*
:   Applies to fi_writemsg. Indicates that remote CQ data is available
    and should be sent as part of the request. See fi_getinfo for
    additional details on FI_REMOTE_CQ_DATA. This flag is implicitly set
    for fi_writedata and fi_inject_writedata.

*FI_COMPLETION*
:   Indicates that a completion entry should be generated for the
    specified operation. The endpoint must be bound to a completion
    queue with FI_SELECTIVE_COMPLETION that corresponds to the specified
    operation, or this flag is ignored.

*FI_MORE*
:   Indicates that the user has additional requests that will
    immediately be posted after the current call returns. Use of this
    flag may improve performance by enabling the provider to optimize
    its access to the fabric hardware.

*FI_INJECT*
:   Applies to fi_writemsg. Indicates that the outbound data buffer
    should be returned to user immediately after the write call returns,
    even if the operation is handled asynchronously. This may require
    that the underlying provider implementation copy the data into a
    local buffer and transfer out of that buffer. This flag can only be
    used with messages smaller than inject_size.

*FI_INJECT_COMPLETE*
:   Applies to fi_writemsg. Indicates that a completion should be
    generated when the source buffer(s) may be reused.

*FI_TRANSMIT_COMPLETE*
:   Applies to fi_writemsg. Indicates that a completion should not be
    generated until the operation has been successfully transmitted and
    is no longer being tracked by the provider.

*FI_DELIVERY_COMPLETE*
:   Applies to fi_writemsg. Indicates that a completion should be
    generated when the operation has been processed by the destination.

*FI_COMMIT_COMPLETE*
:   Applies to fi_writemsg when targeting persistent memory regions.
    Indicates that a completion should be generated only after the
    result of the operation has been made durable.

*FI_FENCE*
:   Applies to transmits. Indicates that the requested operation, also
    known as the fenced operation, and any operation posted after the
    fenced operation will be deferred until all previous operations
    targeting the same peer endpoint have completed. Operations posted
    after the fencing will see and/or replace the results of any
    operations initiated prior to the fenced operation.

The ordering of operations starting at the posting of the fenced
operation (inclusive) to the posting of a subsequent fenced operation
(exclusive) is controlled by the endpoint's ordering semantics.

# RETURN VALUE

Returns 0 on success. On error, a negative value corresponding to fabric
errno is returned. Fabric errno values are defined in `rdma/fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
:   See [`fi_msg`(3)](fi_msg.3.html) for a detailed description of
    handling FI_EAGAIN.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cq`(3)](fi_cq.3.html)

{% include JB/setup %}

# NAME

fi_tagged - Tagged data transfer operations

fi_trecv / fi_trecvv / fi_trecvmsg
:   Post a buffer to receive an incoming message

fi_tsend / fi_tsendv / fi_tsendmsg / fi_tinject / fi_tsenddata
:   Initiate an operation to send a message

# SYNOPSIS

``` c
#include <rdma/fi_tagged.h>

ssize_t fi_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
    fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context);

ssize_t fi_trecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
    size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
    void *context);

ssize_t fi_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
    uint64_t flags);

ssize_t fi_tsend(struct fid_ep *ep, const void *buf, size_t len,
    void *desc, fi_addr_t dest_addr, uint64_t tag, void *context);

ssize_t fi_tsendv(struct fid_ep *ep, const struct iovec *iov,
    void **desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
    void *context);

ssize_t fi_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
    uint64_t flags);

ssize_t fi_tinject(struct fid_ep *ep, const void *buf, size_t len,
    fi_addr_t dest_addr, uint64_t tag);

ssize_t fi_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
    void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t tag,
    void *context);

ssize_t fi_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
    uint64_t data, fi_addr_t dest_addr, uint64_t tag);
```

# ARGUMENTS

*fid*
:   Fabric endpoint on which to initiate tagged communication operation.

*buf*
:   Data buffer to send or receive.

*len*
:   Length of data buffer to send or receive, specified in bytes. Valid
    transfers are from 0 bytes up to the endpoint's max_msg_size.

*iov*
:   Vectored data buffer.

*count*
:   Count of vectored data entries.

*tag*
:   Tag associated with the message.

*ignore*
:   Mask of bits to ignore applied to the tag for receive operations.

*desc*
:   Memory descriptor associated with the data buffer. See
    [`fi_mr`(3)](fi_mr.3.html).

*data*
:   Remote CQ data to transfer with the sent data.

*dest_addr*
:   Destination address for connectionless transfers. Ignored for
    connected endpoints.

*src_addr*
:   Applies only to connectionless endpoints configured with the
    FI_DIRECTED_RECV. For all other endpoint configurations, src_addr is
    ignored. src_addr defines the source address to receive from. By
    default, the src_addr is treated as a source endpoint address
    (i.e. fi_addr_t returned from fi_av_insert / fi_av_insertsvc /
    fi_av_remove). If the FI_AUTH_KEY flag is specified with
    fi_trecvmsg, src_addr is treated as a source authorization key
    (i.e. fi_addr_t returned from fi_av_insert_auth_key). If set to
    FI_ADDR_UNSPEC, any source address may match.

*msg*
:   Message descriptor for send and receive operations.

*flags*
:   Additional flags to apply for the send or receive operation.

*context*
:   User specified pointer to associate with the operation. This
    parameter is ignored if the operation will not generate a successful
    completion, unless an op flag specifies the context parameter be
    used for required input.

# DESCRIPTION

Tagged messages are data transfers which carry a key or tag with the
message buffer. The tag is used at the receiving endpoint to match the
incoming message with a corresponding receive buffer. Message tags match
when the receive buffer tag is the same as the send buffer tag with the
ignored bits masked out. This can be stated as:

``` c
send_tag & ~ignore == recv_tag & ~ignore

or

send_tag | ignore == recv_tag | ignore
```

In general, message tags are checked against receive buffers in the
order in which messages have been posted to the endpoint. See the
ordering discussion below for more details.

The send functions -- fi_tsend, fi_tsendv, fi_tsendmsg, fi_tinject, and
fi_tsenddata -- are used to transmit a tagged message from one endpoint
to another endpoint. The main difference between send functions are the
number and type of parameters that they accept as input. Otherwise, they
perform the same general function.

The receive functions -- fi_trecv, fi_trecvv, fi_recvmsg -- post a data
buffer to an endpoint to receive inbound tagged messages. Similar to the
send operations, receive operations operate asynchronously. Users should
not touch the posted data buffer(s) until the receive operation has
completed. Posted receive buffers are matched with inbound send messages
based on the tags associated with the send and receive buffers.

An endpoint must be enabled before an application can post send or
receive operations to it. For connected endpoints, receive buffers may
be posted prior to connect or accept being called on the endpoint. This
ensures that buffers are available to receive incoming data immediately
after the connection has been established.

Completed message operations are reported to the user through one or
more event collectors associated with the endpoint. Users provide
context which are associated with each operation, and is returned to the
user as part of the event completion. See fi_cq for completion event
details.

## fi_tsend

The call fi_tsend transfers the data contained in the user-specified
data buffer to a remote endpoint, with message boundaries being
maintained. The local endpoint must be connected to a remote endpoint or
destination before fi_tsend is called. Unless the endpoint has been
configured differently, the data buffer passed into fi_tsend must not be
touched by the application until the fi_tsend call completes
asynchronously.

## fi_tsendv

The fi_tsendv call adds support for a scatter-gather list to fi_tsend.
The fi_sendv transfers the set of data buffers referenced by the iov
parameter to a remote endpoint as a single message.

## fi_tsendmsg

The fi_tsendmsg call supports data transfers over both connected and
connectionless endpoints, with the ability to control the send operation
per call through the use of flags. The fi_tsendmsg function takes a
struct fi_msg_tagged as input.

``` c
struct fi_msg_tagged {
    const struct iovec *msg_iov; /* scatter-gather array */
    void               *desc;    /* data descriptor */
    size_t             iov_count;/* # elements in msg_iov */
    fi_addr_t          addr;    /* optional endpoint address */
    uint64_t           tag;      /* tag associated with message */
    uint64_t           ignore;   /* mask applied to tag for receives */
    void               *context; /* user-defined context */
    uint64_t           data;     /* optional immediate data */
};
```

## fi_tinject

The tagged inject call is an optimized version of fi_tsend. It provides
similar completion semantics as fi_inject [`fi_msg`(3)](fi_msg.3.html).

## fi_tsenddata

The tagged send data call is similar to fi_tsend, but allows for the
sending of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the
transfer.

## fi_tinjectdata

The tagged inject data call is similar to fi_tinject, but allows for the
sending of remote CQ data (see FI_REMOTE_CQ_DATA flag) as part of the
transfer.

## fi_trecv

The fi_trecv call posts a data buffer to the receive queue of the
corresponding endpoint. Posted receives are searched in the order in
which they were posted in order to match sends. Message boundaries are
maintained. The order in which the receives complete is dependent on the
endpoint type and protocol.

## fi_trecvv

The fi_trecvv call adds support for a scatter-gather list to fi_trecv.
The fi_trecvv posts the set of data buffers referenced by the iov
parameter to a receive incoming data.

## fi_trecvmsg

The fi_trecvmsg call supports posting buffers over both connected and
connectionless endpoints, with the ability to control the receive
operation per call through the use of flags. The fi_trecvmsg function
takes a struct fi_msg_tagged as input.

# FLAGS

The fi_trecvmsg and fi_tsendmsg calls allow the user to specify flags
which can change the default message handling of the endpoint. Flags
specified with fi_trecvmsg / fi_tsendmsg override most flags previously
configured with the endpoint, except where noted (see fi_endpoint). The
following list of flags are usable with fi_trecvmsg and/or fi_tsendmsg.

*FI_REMOTE_CQ_DATA*
:   Applies to fi_tsendmsg. Indicates that remote CQ data is available
    and should be sent as part of the request. See fi_getinfo for
    additional details on FI_REMOTE_CQ_DATA. This flag is implicitly set
    for fi_tsenddata and fi_tinjectdata.

*FI_COMPLETION*
:   Indicates that a completion entry should be generated for the
    specified operation. The endpoint must be bound to a completion
    queue with FI_SELECTIVE_COMPLETION that corresponds to the specified
    operation, or this flag is ignored.

*FI_MORE*
:   Indicates that the user has additional requests that will
    immediately be posted after the current call returns. Use of this
    flag may improve performance by enabling the provider to optimize
    its access to the fabric hardware.

*FI_INJECT*
:   Applies to fi_tsendmsg. Indicates that the outbound data buffer
    should be returned to user immediately after the send call returns,
    even if the operation is handled asynchronously. This may require
    that the underlying provider implementation copy the data into a
    local buffer and transfer out of that buffer. This flag can only be
    used with messages smaller than inject_size.

*FI_MULTI_RECV*
:   Applies to posted tagged receive operations when the
    FI_TAGGED_MULTI_RECV capability is enabled. This flag allows the
    user to post a single tagged receive buffer that will receive
    multiple incoming messages. Received messages will be packed into
    the receive buffer until the buffer has been consumed. Use of this
    flag may cause a single posted receive operation to generate
    multiple events as messages are placed into the buffer. The
    placement of received data into the buffer may be subjected to
    provider specific alignment restrictions.

The buffer will be released by the provider when the available buffer
space falls below the specified minimum (see FI_OPT_MIN_MULTI_RECV).
Note that an entry to the associated receive completion queue will
always be generated when the buffer has been consumed, even if other
receive completions have been suppressed (i.e. the Rx context has been
configured for FI_SELECTIVE_COMPLETION). See the FI_MULTI_RECV
completion flag [`fi_cq`(3)](fi_cq.3.html).

*FI_INJECT_COMPLETE*
:   Applies to fi_tsendmsg. Indicates that a completion should be
    generated when the source buffer(s) may be reused.

*FI_TRANSMIT_COMPLETE*
:   Applies to fi_tsendmsg. Indicates that a completion should not be
    generated until the operation has been successfully transmitted and
    is no longer being tracked by the provider.

*FI_MATCH_COMPLETE*
:   Applies to fi_tsendmsg. Indicates that a completion should be
    generated only after the message has either been matched with a
    tagged buffer or was discarded by the target application.

*FI_FENCE*
:   Applies to transmits. Indicates that the requested operation, also
    known as the fenced operation, and any operation posted after the
    fenced operation will be deferred until all previous operations
    targeting the same peer endpoint have completed. Operations posted
    after the fencing will see and/or replace the results of any
    operations initiated prior to the fenced operation.

The ordering of operations starting at the posting of the fenced
operation (inclusive) to the posting of a subsequent fenced operation
(exclusive) is controlled by the endpoint's ordering semantics.

*FI_AUTH_KEY*
:   Only valid with domains configured with FI_AV_AUTH_KEY and
    connectionless endpoints configured with FI_DIRECTED_RECV or
    FI_TAGGED_DIRECTED_RECV. When used with fi_trecvmsg, this flag
    denotes that the src_addr is an authorization key fi_addr_t instead
    of an endpoint fi_addr_t.

The following flags may be used with fi_trecvmsg.

*FI_PEEK*
:   The peek flag may be used to see if a specified message has arrived.
    A peek request is often useful on endpoints that have provider
    allocated buffering enabled. Unlike standard receive operations, a
    receive operation with the FI_PEEK flag set does not remain queued
    with the provider after the peek completes successfully. The peek
    operation operates asynchronously, and the results of the peek
    operation are available in the completion queue associated with the
    endpoint. If no message is found matching the tags specified in the
    peek request, then a completion queue error entry with err field set
    to FI_ENOMSG will be available.

If a peek request locates a matching message, the operation will
complete successfully. The returned completion data will indicate the
meta-data associated with the message, such as the message length,
completion flags, available CQ data, tag, and source address. The data
available is subject to the completion entry format (e.g. struct
fi_cq_tagged_entry).

*FI_CLAIM*
:   If this flag is used in conjunction with FI_PEEK, it indicates if
    the peek request completes successfully -- indicating that a
    matching message was located -- the message is claimed by caller.
    Claimed messages can only be retrieved using a subsequent, paired
    receive operation with the FI_CLAIM flag set. A receive operation
    with the FI_CLAIM flag set, but FI_PEEK not set is used to retrieve
    a previously claimed message.

In order to use the FI_CLAIM flag, an application must supply a struct
fi_context structure as the context for the receive operation. The same
fi_context structure used for an FI_PEEK + FI_CLAIM operation must be
used by the paired FI_CLAIM request.

*FI_DISCARD*
:   This flag may be used in conjunction with either FI_PEEK or
    FI_CLAIM. If this flag is used in conjunction with FI_PEEK, it
    indicates if the peek request completes successfully -- indicating
    that a matching message was located -- the message is discarded by
    the provider, as the data is not needed by the application. This
    flag may also be used in conjunction with FI_CLAIM in order to
    discard a message previously claimed using an FI_PEEK + FI_CLAIM
    request.

If this flag is set, the input buffer(s) and length parameters are
ignored.

# RETURN VALUE

The tagged send and receive calls return 0 on success. On error, a
negative value corresponding to fabric *errno * is returned. Fabric
errno values are defined in `fi_errno.h`.

# ERRORS

*-FI_EAGAIN*
:   See [`fi_msg`(3)](fi_msg.3.html) for a detailed description of
    handling FI_EAGAIN.

*-FI_EINVAL*
:   Indicates that an invalid argument was supplied by the user.

*-FI_EOTHER*
:   Indicates that an unspecified error occurred.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_domain`(3)](fi_domain.3.html), [`fi_cq`(3)](fi_cq.3.html)

{% include JB/setup %}

# NAME

fi_trigger - Triggered operations

# SYNOPSIS

``` c
#include <rdma/fi_trigger.h>
```

# DESCRIPTION

Triggered operations allow an application to queue a data transfer
request that is deferred until a specified condition is met. A typical
use is to send a message only after receiving all input data. Triggered
operations can help reduce the latency needed to initiate a transfer by
removing the need to return control back to an application prior to the
data transfer starting.

An endpoint must be created with the FI_TRIGGER capability in order for
triggered operations to be specified. A triggered operation is requested
by specifying the FI_TRIGGER flag as part of the operation. Such an
endpoint is referred to as a trigger-able endpoint.

Any data transfer operation is potentially trigger-able, subject to
provider constraints. Trigger-able endpoints are initialized such that
only those interfaces supported by the provider which are trigger-able
are available.

Triggered operations require that applications use struct
fi_triggered_context as their per operation context parameter, or if the
provider requires the FI_CONTEXT2 mode, struct fi_trigger_context2. The
use of struct fi_triggered_context\[2\] replaces struct fi_context\[2\],
if required by the provider. Although struct fi_triggered_context\[2\]
is not opaque to the application, the contents of the structure may be
modified by the provider once it has been submitted as an operation.
This structure has similar requirements as struct fi_context\[2\]. It
must be allocated by the application and remain valid until the
corresponding operation completes or is successfully canceled.

Struct fi_triggered_context\[2\] is used to specify the condition that
must be met before the triggered data transfer is initiated. If the
condition is met when the request is made, then the data transfer may be
initiated immediately. The format of struct fi_triggered_context\[2\] is
described below.

``` c
struct fi_triggered_context {
    enum fi_trigger_event event_type;   /* trigger type */
    union {
        struct fi_trigger_threshold threshold;
        struct fi_trigger_xpu xpu;
        void *internal[3]; /* reserved */
    } trigger;
};

struct fi_triggered_context2 {
    enum fi_trigger_event event_type;   /* trigger type */
    union {
        struct fi_trigger_threshold threshold;
        struct fi_trigger_xpu xpu;
        void *internal[7]; /* reserved */
    } trigger;
};
```

The triggered context indicates the type of event assigned to the
trigger, along with a union of trigger details that is based on the
event type.

# COMPLETION BASED TRIGGERS

Completion based triggers defer a data transfer until one or more
related data transfers complete. For example, a send operation may be
deferred until a receive operation completes, indicating that the data
to be transferred is now available.

The following trigger event related to completion based transfers is
defined.

*FI_TRIGGER_THRESHOLD*
:   This indicates that the data transfer operation will be deferred
    until an event counter crosses an application specified threshold
    value. The threshold is specified using struct fi_trigger_threshold:

``` c
struct fi_trigger_threshold {
    struct fid_cntr *cntr; /* event counter to check */
    size_t threshold;      /* threshold value */
};
```

Threshold operations are triggered in the order of the threshold values.
This is true even if the counter increments by a value greater than 1.
If two triggered operations have the same threshold, they will be
triggered in the order in which they were submitted to the endpoint.

# DEFERRED WORK QUEUES

The following feature and description are enhancements to triggered
operation support.

The deferred work queue interface is designed as primitive constructs
that can be used to implement application-level collective operations.
They are a more advanced form of triggered operation. They allow an
application to queue operations to a deferred work queue that is
associated with the domain. Note that the deferred work queue is a
conceptual construct, rather than an implementation requirement.
Deferred work requests consist of three main components: an event or
condition that must first be met, an operation to perform, and a
completion notification.

Because deferred work requests are posted directly to the domain, they
can support a broader set of conditions and operations. Deferred work
requests are submitted using struct fi_deferred_work. That structure,
along with the corresponding operation structures (referenced through
the op union) used to describe the work must remain valid until the
operation completes or is canceled. The format of the deferred work
request is as follows:

``` c
struct fi_deferred_work {
    struct fi_context2    context;

    uint64_t              threshold;
    struct fid_cntr       *triggering_cntr;
    struct fid_cntr       *completion_cntr;

    enum fi_trigger_op    op_type;

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
remain on the queue until the triggering counter (success plus error
counter values) has reached the indicated threshold. If the triggering
condition has already been met at the time the work request is queued,
the operation will be initiated immediately.

On the completion of a deferred data transfer, the specified completion
counter will be incremented by one. Note that deferred counter
operations do not update the completion counter; only the counter
specified through the fi_op_cntr is modified. The completion_cntr field
must be NULL for counter operations.

Because deferred work targets support of collective communication
operations, posted work requests do not generate any completions at the
endpoint by default. For example, completed operations are not written
to the EP's completion queue or update the EP counter (unless the EP
counter is explicitly referenced as the completion_cntr). An application
may request EP completions by specifying the FI_COMPLETION flag as part
of the operation.

It is the responsibility of the application to detect and handle
situations that occur which could result in a deferred work request's
condition not being met. For example, if a work request is dependent
upon the successful completion of a data transfer operation, which
fails, then the application must cancel the work request.

To submit a deferred work request, applications should use the domain's
fi_control function with command FI_QUEUE_WORK and struct
fi_deferred_work as the fi_control arg parameter. To cancel a deferred
work request, use fi_control with command FI_CANCEL_WORK and the
corresponding struct fi_deferred_work to cancel. The fi_control command
FI_FLUSH_WORK will cancel all queued work requests. FI_FLUSH_WORK may be
used to flush all work queued to the domain, or may be used to cancel
all requests waiting on a specific triggering_cntr.

Deferred work requests are not acted upon by the provider until the
associated event has occurred; although, certain validation checks may
still occur when a request is submitted. Referenced data buffers are not
read or otherwise accessed. But the provider may validate fabric
objects, such as endpoints and counters, and that input parameters fall
within supported ranges. If a specific request is not supported by the
provider, it will fail the operation with -FI_ENOSYS.

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html), [`fi_mr`(3)](fi_mr.3.html),
[`fi_alias`(3)](fi_alias.3.html), [`fi_cntr`(3)](fi_cntr.3.html)

{% include JB/setup %}

# NAME

fi_version - Version of the library interfaces

# SYNOPSIS

``` c
#include <rdma/fabric.h>

uint32_t fi_version(void);

FI_MAJOR(version)

FI_MINOR(version)
```

# DESCRIPTION

This call returns the current version of the library interfaces. The
version includes major and minor numbers. These may be extracted from
the returned value using the FI_MAJOR() and FI_MINOR() macros.

# NOTES

The library may support older versions of the interfaces.

# RETURN VALUE

Returns the current library version. The upper 16-bits of the version
correspond to the major number, and the lower 16-bits correspond with
the minor number.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_efa - The Amazon Elastic Fabric Adapter (EFA) Provider

# OVERVIEW

The EFA provider supports the Elastic Fabric Adapter (EFA) device on
Amazon EC2. EFA provides reliable and unreliable datagram send/receive
with direct hardware access from userspace (OS bypass). For reliable
datagram (RDM) EP type, it supports two fabric names: `efa` and
`efa-direct`. The `efa` fabric implements a set of [wire
protocols](https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md)
to support more capabilities and features beyond the EFA device
capabilities. The `efa-direct` fabric, on the contrary, offloads all
libfabric data plane calls to the device directly without wire
protocols. Compared to the `efa` fabric, the `efa-direct` fabric
supports fewer capabilities and has more mode requirements for
applications. But it provides a fast path to hand off application
requests to the device. More details and difference between the two
fabrics will be presented below.

# SUPPORTED FEATURES

The following features are supported:

*Endpoint types*
:   The provider supports endpoint type *FI_EP_DGRAM*, and *FI_EP_RDM*
    on a new Scalable (unordered) Reliable Datagram protocol (SRD). SRD
    provides support for reliable datagrams and more complete error
    handling than typically seen with other Reliable Datagram (RD)
    implementations.

*RDM Endpoint capabilities*
:   For the `efa` fabric, the following data transfer interfaces are
    supported: *FI_MSG*, *FI_TAGGED*, *FI_SEND*, *FI_RECV*, *FI_RMA*,
    *FI_WRITE*, *FI_READ*, *FI_ATOMIC*, *FI_DIRECTED_RECV*,
    *FI_MULTI_RECV*, and *FI_SOURCE*. It provides SAS guarantees for
    data operations, and does not have a maximum message size (for all
    operations). For the `efa-direct` fabric, it supports *FI_MSG*,
    *FI_SEND*, *FI_RECV*, *FI_RMA*, *FI_WRITE*, *FI_READ*, and
    *FI_SOURCE*. As mentioned earlier, it doesn't provide SAS
    guarantees, and has different maximum message sizes for different
    operations. For MSG operations, the maximum message size is the MTU
    size of the efa device (approximately 8KiB). For RMA operations, the
    maximum message size is the maximum RDMA size of the EFA device. The
    exact values of these sizes can be queried by the `fi_getopt` API
    with option names `FI_OPT_MAX_MSG_SIZE` and `FI_OPT_MAX_RMA_SIZE`

*DGRAM Endpoint capabilities*
:   The DGRAM endpoint only supports *FI_MSG* capability with a maximum
    message size of the MTU of the underlying hardware (approximately 8
    KiB).

*Address vectors*
:   The provider supports *FI_AV_TABLE*. *FI_AV_MAP* was deprecated in
    Libfabric 2.x. Applications can still use *FI_AV_MAP* to create an
    address vector. But the EFA provider implementation will print a
    warning and switch to *FI_AV_TABLE*. *FI_EVENT* is unsupported.

*Completion events*
:   The provider supports *FI_CQ_FORMAT_CONTEXT*, *FI_CQ_FORMAT_MSG*,
    and *FI_CQ_FORMAT_DATA*. *FI_CQ_FORMAT_TAGGED* is supported on the
    `efa` fabric of RDM endpoint. Wait objects are not currently
    supported.

*Modes*
:   The provider requires the use of *FI_MSG_PREFIX* when running over
    the DGRAM endpoint. And it requires the use of *FI_CONTEXT2* mode
    for DGRAM endpoint and the `efa-direct` fabric of RDM endpoint. The
    `efa` fabric of RDM endpoint doesn't have these requirements.

*Memory registration modes*
:   The `efa` fabric of RDM endpoint does not require memory
    registration for send and receive operations, i.e. it does not
    require *FI_MR_LOCAL*. Applications may specify *FI_MR_LOCAL* in the
    MR mode flags in order to use descriptors provided by the
    application. The `efa-direct` fabric of *FI_EP_RDM* endpint and the
    *FI_EP_DGRAM* endpoint only supports *FI_MR_LOCAL*.

*Progress*
:   RDM and DGRAM endpoints support *FI_PROGRESS_MANUAL*. EFA
    erroneously claims the support for *FI_PROGRESS_AUTO*, despite not
    properly supporting automatic progress. Unfortunately, some
    Libfabric consumers also ask for *FI_PROGRESS_AUTO* when they only
    require *FI_PROGRESS_MANUAL*, and fixing this bug would break those
    applications. This will be fixed in a future version of the EFA
    provider by adding proper support for *FI_PROGRESS_AUTO*.

*Threading*
:   Both RDM and DGRAM endpoints supports *FI_THREAD_SAFE*.

# LIMITATIONS

## Completion events

-   Synchronous CQ read is not supported.
-   Wait objects are not currently supported.

## RMA operations

-   Completion events for RMA targets (*FI_RMA_EVENT*) is not supported.
-   For the `efa-direct` fabric, the target side of RMA operation must
    insert the initiator side's address into AV before the RMA operation
    is kicked off, due to a current device limitation. The same
    limitation applies to the `efa` fabric when the
    `FI_OPT_EFA_HOMOGENEOUS_PEERS` option is set as `true`.

## [Zero-copy receive mode](https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#48-user-receive-qp-feature--request-and-zero-copy-receive)

-   Zero-copy receive mode can be enabled only if SHM transfer is
    disabled.
-   Unless the application explicitly disables P2P, e.g. via
    FI_HMEM_P2P_DISABLED, zero-copy receive can be enabled only if
    available FI_HMEM devices all have P2P support.

## `fi_cancel` support

-   `fi_cancel` is only supported in the non-zero-copy-receive mode of
    the `efa` fabric. It's not supported in `efa-direct`, DGRAM
    endpoint, and the zero-copy receive mode of the `efa` fabric in RDM
    endpoint.

When using FI_HMEM for AWS Neuron or Habana SynapseAI buffers, the
provider requires peer to peer transaction support between the EFA and
the FI_HMEM device. Therefore, the FI_HMEM_P2P_DISABLED option is not
supported by the EFA provider for AWS Neuron or Habana SynapseAI.

# PROVIDER SPECIFIC ENDPOINT LEVEL OPTION

*FI_OPT_EFA_RNR_RETRY*
:   Defines the number of RNR retry. The application can use it to reset
    RNR retry counter via the call to fi_setopt. Note that this option
    must be set before the endpoint is enabled. Otherwise, the call will
    fail. Also note that this option only applies to RDM endpoint.

*FI_OPT_EFA_EMULATED_READ, FI_OPT_EFA_EMULATED_WRITE, FI_OPT_EFA_EMULATED_ATOMICS - bool*
:   These options only apply to the fi_getopt() call. They are used to
    query the EFA provider to determine if the endpoint is emulating
    Read, Write, and Atomic operations (return value is true), or if
    these operations are assisted by hardware support (return value is
    false).

*FI_OPT_EFA_USE_DEVICE_RDMA - bool*
:   This option only applies to the fi_setopt() call. Only available if
    the application selects a libfabric API version \>= 1.18. This
    option allows an application to change libfabric's behavior with
    respect to RDMA transfers. Note that there is also an environment
    variable FI_EFA_USE_DEVICE_RDMA which the user may set as well. If
    the environment variable and the argument provided with this
    variable are in conflict, then fi_setopt will return -FI_EINVAL, and
    the environment variable will be respected. If the hardware does not
    support RDMA and the argument is true, then fi_setopt will return
    -FI_EOPNOTSUPP. If the application uses API version \< 1.18, the
    argument is ignored and fi_setopt returns -FI_ENOPROTOOPT. The
    default behavior for RDMA transfers depends on API version. For API
    \>= 1.18 RDMA is enabled by default on any hardware which supports
    it. For API\<1.18, RDMA is enabled by default only on certain newer
    hardware revisions.

*FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES - bool*
:   This option only applies to the fi_setopt() call. It is used to
    force the endpoint to use in-order send/recv operation for each 128
    bytes aligned block. Enabling the option will guarantee data inside
    each 128 bytes aligned block being sent and received in order, it
    will also guarantee data to be delivered to the receive buffer only
    once. If endpoint is not able to support this feature, it will
    return -FI_EOPNOTSUPP for the call to fi_setopt().

*FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES - bool*
:   This option only applies to the fi_setopt() call. It is used to set
    the endpoint to use in-order RDMA write operation for each 128 bytes
    aligned block. Enabling the option will guarantee data inside each
    128 bytes aligned block being written in order, it will also
    guarantee data to be delivered to the target buffer only once. If
    endpoint is not able to support this feature, it will return
    -FI_EOPNOTSUPP for the call to fi_setopt().

*FI_OPT_EFA_HOMOGENEOUS_PEERS - bool*
:   This option only applies to the fi_setopt() call for RDM endpoints
    on efa fabric. RDM endpoints on efa-direct fabric are unaffected by
    this option. When set to true, it indicates all peers are
    homogeneous, meaning they run on the same platform, use the same
    software versions, and share identical capabilities. It accelerates
    the initial communication setup as interoperability between peers is
    guaranteed. When set to true, the target side of a RMA operation
    must insert the initiator side's address into AV before the RMA
    operation is kicked off, due to a current device limitation. The
    default value is false.

# PROVIDER SPECIFIC DOMAIN OPS

The efa provider exports extensions for operations that are not provided
by the standard libfabric interface. These extensions are available via
the "`fi_ext_efa.h`" header file.

## Domain Operation Extension

Domain operation extension is obtained by calling `fi_open_ops` (see
[`fi_domain(3)`](fi_domain.3.html))

``` c
int fi_open_ops(struct fid *domain, const char *name, uint64_t flags,
    void **ops, void *context);
```

Requesting `FI_EFA_DOMAIN_OPS` in `name` returns `ops` as the pointer to
the function table `fi_efa_ops_domain` defined as follows:

``` c
struct fi_efa_ops_domain {
    int (*query_mr)(struct fid_mr *mr, struct fi_efa_mr_attr *mr_attr);
};
```

### query_mr

This op queries an existing memory registration as input, and outputs
the efa specific mr attribute which is defined as follows

``` c
struct fi_efa_mr_attr {
    uint16_t ic_id_validity;
    uint16_t recv_ic_id;
    uint16_t rdma_read_ic_id;
    uint16_t rdma_recv_ic_id;
};
```

*ic_id_validity*

:   Validity mask of interconnect id fields. Currently the following
    bits are supported in the mask:

    FI_EFA_MR_ATTR_RECV_IC_ID: recv_ic_id has a valid value.

    FI_EFA_MR_ATTR_RDMA_READ_IC_ID: rdma_read_ic_id has a valid value.

    FI_EFA_MR_ATTR_RDMA_RECV_IC_ID: rdma_recv_ic_id has a valid value.

*recv_ic_id*
:   Physical interconnect used by the device to reach the MR for receive
    operation. It is only valid when `ic_id_validity` has the
    `FI_EFA_MR_ATTR_RECV_IC_ID` bit.

*rdma_read_ic_id*
:   Physical interconnect used by the device to reach the MR for RDMA
    read operation. It is only valid when `ic_id_validity` has the
    `FI_EFA_MR_ATTR_RDMA_READ_IC_ID` bit.

*rdma_recv_ic_id*
:   Physical interconnect used by the device to reach the MR for RDMA
    write receive. It is only valid when `ic_id_validity` has the
    `FI_EFA_MR_ATTR_RDMA_RECV_IC_ID` bit.

#### Return value

**query_mr()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).

To enable GPU Direct Async (GDA), which allows the GPU to interact
directly with the NIC, request `FI_EFA_GDA_OPS` in the `name` parameter.
This returns `ops` as a pointer to the function table `fi_efa_ops_gda`
defined as follows:

``` c
struct fi_efa_ops_gda {
    int (*query_addr)(struct fid_ep *ep_fid, fi_addr_t addr, uint16_t *ahn,
              uint16_t *remote_qpn, uint32_t *remote_qkey);
    int (*query_qp_wqs)(struct fid_ep *ep_fid, struct fi_efa_wq_attr *sq_attr, struct fi_efa_wq_attr *rq_attr);
    int (*query_cq)(struct fid_cq *cq_fid, struct fi_efa_cq_attr *cq_attr);
    int (*cq_open_ext)(struct fid_domain *domain_fid,
               struct fi_cq_attr *attr,
               struct fi_efa_cq_init_attr *efa_cq_init_attr,
               struct fid_cq **cq_fid, void *context);
    uint64_t (*get_mr_lkey)(struct fid_mr *mr);
};
```

### query_addr

This op queries the following address information for a given endpoint
and destination address.

*ahn*
:   Address handle number.

*remote_qpn*
:   Remote queue pair Number.

*remote_qkey*
:   qkey for the remote queue pair.

#### Return value

**query_addr()** returns FI_SUCCESS on success, or -FI_EINVAL on
failure.

### query_qp_wqs

This op queries EFA specific Queue Pair work queue attributes for a
given endpoint. It retrieves the send queue attributes in sq_attr and
receive queue attributes in rq_attr, which is defined as follows.

``` c
struct fi_efa_wq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
    uint32_t *doorbell;
    uint32_t max_batch;
};
```

*buffer*
:   Queue buffer.

*entry_size*
:   Size of each entry in the queue.

*num_entries*
:   Maximal number of entries in the queue.

*doorbell*
:   Queue doorbell.

*max_batch*
:   Maximum batch size for queue submissions.

#### Return value

**query_qp_wqs()** returns 0 on success, or the value of errno on
failure (which indicates the failure reason).

### query_cq

This op queries EFA specific Completion Queue attributes for a given cq.

``` c
struct fi_efa_cq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
};
```

*buffer*
:   Completion queue buffer.

*entry_size*
:   Size of each completion queue entry.

*num_entries*
:   Maximal number of entries in the completion queue.

#### Return value

**query_cq()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).

### cq_open_ext

This op creates a completion queue with external memory provided via
dmabuf. The memory can be passed by supplying the following struct.

``` c
struct fi_efa_cq_init_attr {
    uint64_t flags;
    struct {
        uint8_t *buffer;
        uint64_t length;
        uint64_t offset;
        uint32_t fd;
    } ext_mem_dmabuf;
};
```

*flags*

:   A bitwise OR of the various values described below.

    FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF: create CQ with external memory
    provided via dmabuf.

*ext_mem_dmabuf*

:   Structure containing information about external memory when using
    FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF flag.

    *buffer*
    :   Pointer to the memory mapped in the process's virtual address
        space. The field is optional, but if not provided, the use of CQ
        poll interfaces should be avoided.

    *length*
    :   Length of the memory region to use.

    *offset*
    :   Offset within the dmabuf.

    *fd*
    :   File descriptor of the dmabuf.

#### Return value

**cq_open_ext()** returns 0 on success, or the value of errno on failure
(which indicates the failure reason).

### get_mr_lkey

Returns the local memory translation key associated with a MR. The
memory registration must have completed successfully before invoking
this.

*lkey*
:   local memory translation key used by TX/RX buffer descriptor.

#### Return value

**get_mr_lkey()** returns lkey on success, or FI_KEY_NOTAVAIL if the
registration has not completed.

# Traffic Class (tclass) in EFA

To prioritize the messages from a given endpoint, user can specify
`fi_info->tx_attr->tclass = FI_TC_LOW_LATENCY` in the fi_endpoint() call
to set the service level in rdma-core. All other tclass values will be
ignored.

# RUNTIME PARAMETERS

*FI_EFA_IFACE*
:   A comma-delimited list of EFA device, i.e. NIC, names that should be
    visible to the application. This paramater can be used to
    include/exclude NICs to enforce process affinity based on the
    hardware topology. The default value is "all" which allows all
    available NICs to be discovered.

*FI_EFA_TX_SIZE*
:   Maximum number of transmit operations before the provider returns
    -FI_EAGAIN. For only the RDM endpoint, this parameter will cause
    transmit operations to be queued when this value is set higher than
    the default and the transmit queue is full.

*FI_EFA_RX_SIZE*
:   Maximum number of receive operations before the provider returns
    -FI_EAGAIN.

# RUNTIME PARAMETERS SPECIFIC TO RDM ENDPOINT

These OFI runtime parameters apply only to the RDM endpoint.

*FI_EFA_RX_WINDOW_SIZE*
:   Maximum number of MTU-sized messages that can be in flight from any
    single endpoint as part of long message data transfer.

*FI_EFA_TX_QUEUE_SIZE*
:   Depth of transmit queue opened with the NIC. This may not be set to
    a value greater than what the NIC supports.

*FI_EFA_RECVWIN_SIZE*
:   Size of out of order reorder buffer (in messages). Messages received
    out of this window will result in an error.

*FI_EFA_CQ_SIZE*
:   Size of any cq created, in number of entries.

*FI_EFA_MR_CACHE_ENABLE*
:   Enables using the mr cache and in-line registration instead of a
    bounce buffer for iov's larger than max_memcpy_size. Defaults to
    true. When disabled, only uses a bounce buffer

*FI_EFA_MR_MAX_CACHED_COUNT*
:   Sets the maximum number of memory registrations that can be cached
    at any time.

*FI_EFA_MR_MAX_CACHED_SIZE*
:   Sets the maximum amount of memory that cached memory registrations
    can hold onto at any time.

*FI_EFA_MAX_MEMCPY_SIZE*
:   Threshold size switch between using memory copy into a
    pre-registered bounce buffer and memory registration on the user
    buffer.

*FI_EFA_MTU_SIZE*
:   Overrides the default MTU size of the device.

*FI_EFA_RX_COPY_UNEXP*
:   Enables the use of a separate pool of bounce-buffers to copy
    unexpected messages out of the pre-posted receive buffers.

*FI_EFA_RX_COPY_OOO*
:   Enables the use of a separate pool of bounce-buffers to copy
    out-of-order RTS packets out of the pre-posted receive buffers.

*FI_EFA_MAX_TIMEOUT*
:   Maximum timeout (us) for backoff to a peer after a receiver not
    ready error.

*FI_EFA_TIMEOUT_INTERVAL*
:   Time interval (us) for the base timeout to use for exponential
    backoff to a peer after a receiver not ready error.

*FI_EFA_ENABLE_SHM_TRANSFER*
:   Enable SHM provider to provide the communication across all
    intra-node processes. SHM transfer will be disabled in the case
    where
    [`ptrace protection`](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection)
    is turned on. You can turn it off to enable shm transfer.

FI_EFA_ENABLE_SHM_TRANSFER is parsed during the fi_domain call and is
related to the FI_OPT_SHARED_MEMORY_PERMITTED endpoint option. If
FI_EFA_ENABLE_SHM_TRANSFER is set to true, the
FI_OPT_SHARED_MEMORY_PERMITTED endpoint option overrides
FI_EFA_ENABLE_SHM_TRANSFER. If FI_EFA_ENABLE_SHM_TRANSFER is set to
false, but the FI_OPT_SHARED_MEMORY_PERMITTED is set to true, the
FI_OPT_SHARED_MEMORY_PERMITTED setopt call will fail with -FI_EINVAL.

*FI_EFA_SHM_AV_SIZE*
:   Defines the maximum number of entries in SHM provider's address
    vector.

*FI_EFA_SHM_MAX_MEDIUM_SIZE*
:   Defines the switch point between small/medium message and large
    message. The message larger than this switch point will be
    transferred with large message protocol. NOTE: This parameter is now
    deprecated.

*FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE*
:   The maximum size for inter EFA messages to be sent by using medium
    message protocol. Messages which can fit in one packet will be sent
    as eager message. Messages whose sizes are smaller than this value
    will be sent using medium message protocol. Other messages will be
    sent using CTS based long message protocol.

*FI_EFA_FORK_SAFE*
:   Enable fork() support. This may have a small performance impact and
    should only be set when required. Applications that require to
    register regions backed by huge pages and also require fork support
    are not supported.

*FI_EFA_RUNT_SIZE*
:   The maximum number of bytes that will be eagerly sent by inflight
    messages uses runting read message protocol (Default 307200).

*FI_EFA_INTER_MIN_READ_MESSAGE_SIZE*
:   The minimum message size in bytes for inter EFA read message
    protocol. If instance support RDMA read, messages whose size is
    larger than this value will be sent by read message protocol.
    (Default 1048576).

*FI_EFA_INTER_MIN_READ_WRITE_SIZE*
:   The mimimum message size for emulated inter EFA write to use read
    write protocol. If firmware support RDMA read, and
    FI_EFA_USE_DEVICE_RDMA is 1, write requests whose size is larger
    than this value will use the read write protocol (Default 65536). If
    the firmware supports RDMA write, device RDMA write will always be
    used.

*FI_EFA_USE_DEVICE_RDMA*
:   Specify whether to require or ignore RDMA features of the EFA
    device. - When set to 1/true/yes/on, all RDMA features of the EFA
    device are used. But if EFA device does not support RDMA and
    FI_EFA_USE_DEVICE_RDMA is set to 1/true/yes/on, user's application
    is aborted and a warning message is printed. - When set to
    0/false/no/off, libfabric will emulate all fi_rma operations instead
    of offloading them to the EFA network device. Libfabric will not use
    device RDMA to implement send/receive operations. - If not set, RDMA
    operations will occur when available based on RDMA device
    ID/version.

*FI_EFA_USE_HUGE_PAGE*
:   Specify Whether EFA provider can use huge page memory for internal
    buffer. Using huge page memory has a small performance advantage,
    but can cause system to run out of huge page memory. By default, EFA
    provider will use huge page unless FI_EFA_FORK_SAFE is set to
    1/on/true.

*FI_EFA_USE_ZCPY_RX*
:   Enables the use of application's receive buffers in place of
    bounce-buffers when feasible. (Default: 1). Setting this environment
    variable to 0 can disable this feature. Explicitly setting this
    variable to 1 does not guarantee this feature due to other
    requirements. See
    https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md
    for details.

*FI_EFA_USE_UNSOLICITED_WRITE_RECV*
:   Use device's unsolicited write recv functionality when it's
    available. (Default: 1). Setting this environment variable to 0 can
    disable this feature.

*FI_EFA_INTERNAL_RX_REFILL_THRESHOLD*
:   The threshold that EFA provider will refill the internal rx pkt
    pool. (Default: 8). When the number of internal rx pkts to post is
    lower than this threshold, the refill will be skipped.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_hook - The Hook Fabric Provider Utility

# OVERVIEW

The hooking provider is a utility function that can intercept calls to
any provider. The hook provider is always available, but has zero impact
on calls unless enabled. It is useful for providing performance data on
selected calls or debugging information.

# SUPPORTED FEATURES

Hooking support is enabled through the FI_HOOK environment variable. To
enable hooking, FI_HOOK must be set to the name of one or more of the
available hooking providers. When multiple hooks are specified, the
names must be separated by a semi-colon. To obtain a list of hooking
providers available on the current system, one can use the fi_info
utility with the '--env' command line option. Hooking providers are
usually identified by 'hook' appearing in the provider name.

Known hooking providers include the following:

*ofi_hook_perf*
:   This hooks 'fast path' data operation calls. Performance data is
    captured on call entrance and exit, in order to provide an average
    of how long each call takes to complete. See the PERFORMANCE HOOKS
    section for available performance data.

*ofi_hook_trace*
:   This hooks most of API calls for fabric communications. The APIs and
    their runtime parameters are logged to provide detail trace
    information for debugging. See the TRACE HOOKS section for the APIs
    enabled for trace.

*ofi_hook_profile*
:   This hooks data operation calls, cq operation calls and mr
    registration calls. The API calls and the amount of data being
    operated are accumulated, to provide a view of APIs' usages and a
    histogram of data size operated in a workload execution. See the
    PROFILE HOOKS section for the report in the detail.

*ofi_hook_monitor*
:   Similar to *ofi_hook_profile*, this hooks data operation calls, cq
    operation calls and mr registration calls. The API calls and the
    amount of data being operated are accumulated and made available for
    export via an external sampler through a shared communication file.
    See the MONITOR HOOKS section for more details.

# PERFORMANCE HOOKS

The hook provider allows capturing inline performance data by accessing
the CPU Performance Management Unit (PMU). PMU data is only available on
Linux systems. Additionally, access to PMU data may be restricted to
privileged (super-user) applications.

Performance data is captured for critical data transfer calls: fi_msg,
fi_rma, fi_tagged, fi_cq, and fi_cntr. Captured data is displayed as
logged data using the FI_LOG_LEVEL trace level. Performance data is
logged when the associated fabric is destroyed.

The environment variable FI_PERF_CNTR is used to identify which
performance counter is tracked. The following counters are available:

*cpu_cycles*
:   Counts the number of CPU cycles each function takes to complete.

*cpu_instr*
:   Counts the number of CPU instructions each function takes to
    complete. This is the default performance counter if none is
    specified.

# TRACE HOOKS

This hook provider allows tracing each API call and its runtime
parameters. It is enabled by setting FI_HOOK to "trace".

The trace data include the provider's name, API function, and
input/output parameter values. The APIs enabled for tracing include the
following:

*data operation calls*
:   This include fi_msg, fi_rma, fi_tagged all data operations. The
    trace data contains the data buffer, data size being operated, data,
    tags, and flags when applicable.

*cq operation calls*
:   This includes fi_cq_read, fi_cq_sread, fi_cq_strerror and all cq
    operations. The trace data contains the cq entries based on cq
    format.

*cm operation calls*
:   This includes fi_getname, fi_setname, fi_getpeer, fi_connect and all
    cm operations. The trace data contains the target address.

*resource creation calls*
:   This includes fi_av_open, fi_cq_open, fi_endpoing, fi_cntr_open and
    fi_mr operations. The trace data contains the corresponding
    attributes used for resource creation.

The trace data is logged after API is invoked using the FI_LOG_LEVEL
trace level

# PROFILE HOOKS

This hook provider allows capturing data operation calls and the amount
of data being operated. It is enabled by setting FI_HOOK to "profile".

The provider counts the API invoked and accumulates the data size each
API call operates. For data and cq operations, instead of accumulating
all data together, it breaks down the data size into size buckets and
accumulates the amount of data in the corresponding bucket based on the
size of the data operated. For mr registration operations, it breaks
down memory registered per HMEM iface. At the end when the associated
fabric is destroyed, the provider generates a profile report.

The report contains the usage of data operation APIs, the amount of data
received in each CQ format and the amount of memory registered for rma
operations if any exist. In addition, the data operation APIs are
grouped into 4 groups based on the nature of the operations, message
send (fi_sendXXX, fi_tsendXXX), message receive (fi_recvXXX,
fi_trecvXXX), rma read (fi_readXXX) and rma write (fi_writeXXX) to
present the percentage usage of each API.

The report is in a table format which has APIs invoked in rows, and the
columns contain the following fields:

*API*
:   The API calls are invoked.

*Size*
:   Data size bucket that at least one API call operates data in that
    size bucket. The pre-defined size buckets (in Byte) are \[0-64\]
    \[64-512\] \[512-1K\] \[1K-4K\] \[4K-64K\] \[64K-256K\] \[256K-1M\]
    \[1M-4M\] \[4M-UP\].

*Count*
:   Count of the API calls.

*Amount*
:   Amount of data the API operated.

*% Count*
:   Percentage of the API calls over the total API calls in the same
    data operation group.

*% Amount*
:   Percentage of the amount of data from the API over the total amount
    of data operated in the same data operation group.

The report is logged using the FI_LOG_LEVEL trace level.

# MONITOR HOOKS

This hook provider builds on the "profile" hook provider and provides
continuous readout capabilities useful for monitoring libfabric
applications. It is enabled by setting FI_HOOK to "monitor".

Similar to the "profile" hook provider, this provider counts the number
of invoked API calls and accumulates the amount of data handled by each
API call. Refer to the documentation on the "profile" hook for more
information.

Data export is facilitated using a communication file on the filesystem,
which is created for each hooked libfabric provider. The monitor hook
expects to be run on a tmpfs. If available and unless otherwise
specified, files will be created under the tmpfs `/dev/shm`.

A sampler can periodically read this file to extract the gathered data.
Every N intercepted API calls or "ticks", the provider will check
whether data export has been requested by the sampler. If so, the
currently gathered counter data is copied to the file and the sampler
informed about the new data. The provider-local counter data is then
cleared. Each sample contains the counter delta to the previous sample.

Communication files will be created at path
`$FI_OFI_HOOK_MONITOR_BASEPATH/<uid>/<hostname>` and will have the name:
`<ppid>_<pid>_<sequential id>_<job id>_<provider name>`. `ppid` and
`pid` are taken from the perspective of the monitored application. In a
batched environment running SLURM, `job id` is set to the SLURM job ID,
otherwise it is set to 0.

See [`fi_mon_sampler`(1)](fi_mon_sampler.1.html) for documentation on
how to use the monitor provider sampler.

## CONFIGURATION

The "monitor" hook provider exposes several runtime options via
environment variables:

*FI_OFI_HOOK_MONITOR_BASEPATH*
:   String to basepath for communication files. (default: /dev/shm/ofi)

*FI_OFI_HOOK_MONITOR_DIR_MODE*
:   POSIX mode/permission for directories in basepath. (default: 01700)

*FI_OFI_HOOK_MONITOR_FILE_MODE*
:   POSIX mode/permission for communication files. (default: 0600)

*FI_OFI_HOOK_MONITOR_TICK_MAX*
:   Number of API calls before communication files are checked for data
    request. (default: 1024)

*FI_OFI_HOOK_MONITOR_LINGER*
:   Whether communication files should linger after termination.
    (default: 0) This is useful to allow the sampler to read the last
    counter data even if the libfabric application has already
    terminated. Note: Using this option without a sampler results in
    files cluttering FI_OFI_HOOK_MONITOR_BASEPATH. Make sure to either
    run a sampler or clean these files manually.

# LIMITATIONS

Hooking functionality is not available for providers built using the
FI_FABRIC_DIRECT feature. That is, directly linking to a provider
prevents hooking.

The hooking provider does not work with triggered operations.
Application that use FI_TRIGGER operations that attempt to hook calls
will likely crash.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html)

{% include JB/setup %}

# NAME

fi_mrail - The Multi-Rail Utility Provider

# OVERVIEW

The mrail provider (ofi_mrail) is an utility provider that layers over
an underlying provider to enable the use of multiple network ports
(rails). This increases the total available bandwidth of an underlying
provider. The current status of mrail provider is experimental - not all
libfabric features are supported and performance is not guaranteed.

# REQUIREMENTS

## Requirements for underlying provider

mrail provider requires the underlying provider to support the following
capabilities / modes:

-   Buffered receive (FI_BUFFERED_RECV)

-   FI_SOURCE

-   FI_AV_TABLE

## Requirements for applications

Applications need to: \* Support FI_MR_RAW MR mode bit to make use of
FI_RMA capability. \* Set FI_OFI_MRAIL_ADDR env variable (see RUNTIME
PARAMETERS section below).

# SUPPORTED FEATURES

*Endpoint types*
:   The provider supports only *FI_EP_RDM*.

*Endpoint capabilities*
:   The following data transfer interface is supported: *FI_MSG*,
    *FI_TAGGED*, *FI_RMA*.

\# LIMITATIONS

:   Limitations of the underlying provider may show up as that of mrail
    provider.

:   mrail provider doesn't allow pass-through of any mode bits to the
    underlying provider.

## Unsupported features

The following are the major libfabric features that are not supported.
Any other feature not listed in "Supported features" can be assumed as
unsupported.

-   FI_ATOMIC

-   Scalable endpoints

-   Shared contexts

-   FABRIC_DIRECT

-   Multicast

-   Triggered operations

# FUNCTIONALITY OVERVIEW

For messages (FI_MSG, FI_TAGGED), the provider uses different policies
to send messages over one or more rails based on message size (See
*FI_OFI_MRIAL_CONFIG* in the RUNTIME PARAMETERS section). Ordering is
guaranteed through the use of sequence numbers.

For RMA, the data is striped equally across all rails.

# RUNTIME PARAMETERS

The ofi_mrail provider checks for the following environment variables.

*FI_OFI_MRAIL_ADDR*
:   Comma delimited list of individual rail addresses. Each address can
    be an address in FI_ADDR_STR format, a host name, an IP address, or
    a netdev interface name.

*FI_OFI_MRAIL_ADDR_STRC*
:   Deprecated. Replaced by *FI_OFI_MRAIL_ADDR*.

*FI_OFI_MRAIL_CONFIG*
:   Comma separated list of `<max_size>:<policy>` pairs, sorted in
    ascending order of `<max_size>`. Each pair indicated the rail
    sharing policy to be used for messages up to the size `<max_size>`
    and not covered by all previous pairs. The value of `<policy>` can
    be *fixed* (a fixed rail is used), *round-robin* (one rail per
    message, selected in round-robin fashion), or *striping* (striping
    across all the rails). The default configuration is
    `16384:fixed,ULONG_MAX:striping`. The value ULONG_MAX can be input
    as -1.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{%include JB/setup %}

# NAME

fi_opx - The Omni-Path Express Fabric Provider

# OVERVIEW

The *opx* provider is a native libfabric provider suitable for use with
Omni-Path fabrics. OPX features great scalability and performance when
running libfabric-enabled message layers. OPX requires 3 additonal
external development libraries to build: libuuid, libnuma, and the Linux
kernel headers.

# SUPPORTED FEATURES

The OPX provider supports most features defined for the libfabric API.

Key features include:

Endpoint types
:   The Omni-Path HFI hardware is connectionless and reliable. The OPX
    provider only supports the *FI_EP_RDM* endpoint type.

Capabilities
:   Supported capabilities include *FI_MSG*, *FI_RMA, *FI_TAGGED*,
    *FI_ATOMIC*, *FI_SOURCE*, *FI_SEND*, *FI_RECV*, *FI_MULTI_RECV*,
    *FI_DIRECTED_RECV*, *FI_SOURCE\*.

Notes on *FI_DIRECTED_RECV* capability: The immediate data which is sent
within the "senddata" call to support *FI_DIRECTED_RECV* for OPX must be
exactly 4 bytes, which OPX uses to completely identify the source
address to an exascale-level number of ranks for tag matching on the
recv and can be managed within the MU packet. Therefore the domain
attribute "cq_data_size" is set to 4 which is the OFI standard minimum.

Modes
:   Two modes are defined: *FI_CONTEXT2* and *FI_ASYNC_IOV*. The OPX
    provider requires *FI_CONTEXT2*.

Additional features
:   Supported additional features include *FABRIC_DIRECT* and
    *counters*.

Progress
:   *FI_PROGRESS_MANUAL* and *FI_PROGRESS_AUTO* are supported, for best
    performance, use *FI_PROGRESS_MANUAL* when possible.
    *FI_PROGRESS_AUTO* will spawn 1 thread per CQ.

Address vector
:   *FI_AV_MAP* and *FI_AV_TABLE* are both supported. *FI_AV_MAP* is
    default.

Memory registration modes
:   Only *FI_MR_SCALABLE* is supported.

# UNSUPPORTED FEATURES

Endpoint types
:   Unsupported endpoint types include *FI_EP_DGRAM* and *FI_EP_MSG*.

Capabilities
:   The OPX provider does not support *FI_RMA_EVENT* and *FI_TRIGGER*
    capabilities.

# LIMITATIONS

OPX supports the following MPI versions:

Intel MPI from Parallel Studio 2020, update 4. Intel MPI from OneAPI
2021, update 3. Open MPI 4.1.2a1 (Older version of Open MPI will not
work). MPICH 3.4.2 and later.

Usage:

If using with OpenMPI 4.1.x, disable UCX and openib transports. OPX is
not compatible with Open MPI 4.1.x PML/BTL.

# CONFIGURATION OPTIONS

*OPX_AV*
:   OPX supports the option of setting the AV mode to use in a build. 3
    settings are supported: - table - map - runtime

Using table or map will only allow OPX to use FI_AV_TABLE or FI_AV_MAP.
Using runtime will allow OPX to use either AV mode depending on what the
application requests. Specifying map or table however may lead to a
slight performance improvement depending on the application.

To change OPX_AV, add OPX_AV=table, OPX_AV=map, or OPX_AV=runtime to the
configure command. For example, to create a new build with
OPX_AV=table:\
OPX_AV=table ./configure\
make install\
\
There is no way to change OPX_AV after it is set. If OPX_AV is not set
in the configure, the default value is runtime.

# RUNTIME PARAMETERS

*FI_OPX_UUID*
:   OPX requires a unique ID for each job. In order for all processes in
    a job to communicate with each other, they require to use the same
    UUID. This variable can be set with FI_OPX_UUID=\${RANDOM} The
    default UUID is 00112233445566778899aabbccddeeff.

*FI_OPX_FORCE_CPUAFFINITY*
:   Boolean (1/0, on/off, true/false, yes/no). Causes the thread to bind
    itself to the cpu core it is running on. Defaults to "No"

*FI_OPX_RELIABILITY_SERVICE_USEC_MAX*
:   Integer. This setting controls how frequently the reliability/replay
    function will issue PING requests to a remote connection. Reducing
    this value may improve performance at the expense of increased
    traffic on the OPX fabric. Default setting is 500.

*FI_OPX_RELIABILITY_SERVICE_MAX_OUTSTANDING_BYTES*
:   Integer. This setting controls the maximum number of bytes allowed
    to be in-flight (sent but un-ACK'd by receiver) per reliability flow
    (one-way communication between two endpoints).

Valid values are in the range of 8192-150,994,944 (8KB-144MB),
inclusive.

Default setting is 7,340,032 (7MB).

*FI_OPX_RELIABILITY_SERVICE_PRE_ACK_RATE*
:   Integer. This setting controls how frequently a receiving rank will
    send ACKs for packets it has received without being prompted through
    a PING request. A non-zero value N tells the receiving rank to send
    an ACK for the last N packets every Nth packet. Used in conjunction
    with an increased value for FI_OPX_RELIABILITY_SERVICE_USEC_MAX may
    improve performance.

Valid values are 0 (disabled) and powers of 2 in the range of 1-32,768,
inclusive.

Default setting is 64.

*FI_OPX_RELIABILITY_MAX_UNCONGESTED_PINGS*
:   Integer. This setting controls how many PING requests the
    reliability/replay function will issue per iteration of
    FI_OPX_RELIABILITY_SERVICE_USEC_MAX in situations with less
    contending outgoing traffic from the HFI. Default setting is 128.
    Range of valid values is 1-65535.

*FI_OPX_RELIABILITY_MAX_CONGESTED_PINGS*
:   Integer. This setting controls how many PING requests the
    reliability/replay function will issue per iteration of
    FI_OPX_RELIABILITY_SERVICE_USEC_MAX in situations with more
    contending, outgoing traffic from the HFI. Default setting is 4.
    Range of valid values is 1-65535.

*FI_OPX_SELINUX*
:   Boolean (1/0, on/off, true/false, yes/no). Set to true if you're
    running a security-enhanced Linux. This enables updating the Jkey
    used based on system settings. Defaults to "No"

*FI_OPX_HFI_SELECT*
:   String. Controls how OPX chooses which HFI to use when opening a
    context. Has two forms: - `<hfi-unit>` Force OPX provider to use
    `hfi-unit`. - `<selector1>[,<selector2>[,...,<selectorN>]]` Select
    HFI based on first matching `selector`

Where `selector` is one of the following forms: - `default` to use the
default logic - `fixed:<hfi-unit>` to fix to one `hfi-unit` -
`<selector-type>:<hfi-unit>:<selector-data>`

The above fields have the following meaning: - `selector-type` The
selector criteria the caller opening the context is evaluated against. -
`hfi-unit` The HFI to use if the caller matches the selector. -
`selector-data` Data the caller must match (e.g. NUMA node ID).

Where `selector-type` is one of the following: - `numa` True when caller
is local to the NUMA node ID given by `selector-data`. - `core` True
when caller is local to the CPU core given by `selector-data`.

And `selector-data` is one of the following: - `value` The specific
value to match - `<range-start>-<range-end>` Matches with any value in
that range

In the second form, when opening a context, OPX uses the `hfi-unit` of
the first-matching selector. Selectors are evaluated left-to-right. OPX
will return an error if the caller does not match any selector.

In either form, it is an error if the specified or selected HFI is not
in the Active state. In this case, OPX will return an error and
execution will not continue.

With this option, it is possible to cause OPX to try to open more
contexts on an HFI than there are free contexts on that HFI. In this
case, one or more of the context-opening calls will fail and OPX will
return an error. For the second form, as which HFI is selected depends
on properties of the caller, deterministic HFI selection requires
deterministic caller properties. E.g. for the `numa` selector, if the
caller can migrate between NUMA domains, then HFI selection will not be
deterministic.

The logic used will always be the first valid in a selector list. For
example, `default` and `fixed` will match all callers, so if either are
in the beginning of a selector list, you will only use `fixed` or
`default` regardles of if there are any more selectors.

Examples: - `FI_OPX_HFI_SELECT=0` all callers will open contexts on HFI
0. - `FI_OPX_HFI_SELECT=1` all callers will open contexts on HFI 1. -
`FI_OPX_HFI_SELECT=numa:0:0,numa:1:1,numa:0:2,numa:1:3` callers local to
NUMA nodes 0 and 2 will use HFI 0, callers local to NUMA domains 1 and 3
will use HFI 1. - `FI_OPX_HFI_SELECT=numa:0:0-3,default` callers local
to NUMA nodes 0 thru 3 (including 0 and 3) will use HFI 0, and all else
will use default selection logic. - `FI_OPX_HFI_SELECT=core:1:0,fixed:0`
callers local to CPU core 0 will use HFI 1, and all others will use HFI
0. - `FI_OPX_HFI_SELECT=default,core:1:0` all callers will use default
HFI selection logic.

*FI_OPX_PORT*
:   Integer. HFI1 port number. If the specified port is not available, a
    default active port will be selected. Special value 0 indicates any
    available port. Defaults to port 1 on OPA100 and any port on CN5000.

*FI_OPX_DELIVERY_COMPLETION_THRESHOLD*
:   Integer. Will be deprecated. Please use
    FI_OPX_SDMA_BOUNCE_BUF_THRESHOLD.

*FI_OPX_SDMA_BOUNCE_BUF_THRESHOLD*
:   Integer. The maximum message length in bytes that will be copied to
    the SDMA bounce buffer. For messages larger than this threshold, the
    send will not be completed until receiver has ACKed. Value must be
    between 16385 and 2147483646. Defaults to 16385.

*FI_OPX_SDMA_DISABLE*
:   Boolean (1/0, on/off, true/false, yes/no). Disables SDMA offload
    hardware. Default is 0.

*FI_OPX_MAX_PKT_SIZE*
:   Integer. Set the maximum packet size which must be less than or
    equal to the driver's MTU (Maximum Transmission Unit) size. Valid
    values: 2048, 4096, 8192, 10240. Default is set to 10240 for
    libraries built on CN5000 systems and set to 8192 for libraries
    built on OPA100 systems.

*FI_OPX_SDMA_MIN_PAYLOAD_BYTES*
:   Integer. The minimum length in bytes where SDMA will be used. For
    messages smaller than this threshold, the send will be completed
    using PIO. Value must be between 64 and 2147483646. Defaults to
    16385.

*FI_OPX_SDMA_MAX_WRITEVS_PER_CYCLE*
:   Integer. The maximum number of times writev will be called during a
    single poll cycle. Value must be between 1 and 1024. Defaults to 1.

*FI_OPX_SDMA_MAX_IOVS_PER_WRITEV*
:   Integer. The maximum number of IOVs passed to each writev call.
    Value must be between 3 and 128. Defaults to 64.

*FI_OPX_SDMA_MAX_PKTS*
:   Integer. The maximum number of packets transmitted per SDMA request
    when expected receive (TID) is NOT being used. Value must be between
    1 and 128. Defaults to 32.

*FI_OPX_SDMA_MAX_PKTS_TID*
:   Integer. The maximum number of packets transmitted per SDMA request
    when expected receive (TID) is being used. Value must be between 1
    and 512. Defaults to 64.

*FI_OPX_TID_MIN_PAYLOAD_BYTES*
:   Integer. The minimum length in bytes where TID (Expected Receive)
    will be used. For messages smaller than this threshold, the send
    will be completed using Eager Receive. Value must be between 4096
    and 2147483646. Defaults to 4096.

*FI_OPX_RZV_MIN_PAYLOAD_BYTES*
:   Integer. The minimum length in bytes where rendezvous will be used.
    For messages smaller than this threshold, the send will first try to
    be completed using eager or multi-packet eager. Value must be
    between 64 and 65536. Defaults to 16385.

*FI_OPX_MP_EAGER_DISABLE*
:   Boolean (1/0, on/off, true/false, yes/no). Disables multi-packet
    eager. Defaults to 0.

*FI_OPX_TID_DISABLE*
:   Boolean (1/0, on/off, true/false, yes/no). Disables using Token ID
    (TID). Defaults to 0.

*FI_OPX_EXPECTED_RECEIVE_ENABLE*
:   Deprecated. Use FI_OPX_TID_DISABLE instead.

*FI_OPX_PROG_AFFINITY*
:   String. This sets the affinity to be used for any progress threads.
    Set as a colon-separated triplet as `start:end:stride`, where stride
    controls the interval between selected cores. For example, `1:5:2`
    will have cores 1, 3, and 5 as valid cores for progress threads. By
    default no affinity is set.

*FI_OPX_AUTO_PROGRESS_INTERVAL_USEC*
:   Deprecated/ignored. Auto progress threads are now interrupt-driven
    and only poll when data is available.

*FI_OPX_PKEY*
:   Integer. Partition key, a 2 byte positive integer. Default is the
    Pkey in the index 0 of the Pkey table of the unit and port on which
    context is created.

*FI_OPX_SL*
:   Integer. Service Level. This will also determine Service Class and
    Virtual Lane. Default is 0

*FI_OPX_GPU_IPC_INTRANODE*
:   Boolean (0/1, on/off, true/false, yes/no). This setting controls
    whether IPC will be used to facilitate GPU to GPU intranode copies
    over PCIe, NVLINK, or xGMI. When this is turned off, GPU data will
    be copied to the host before being copied to another GPU which is
    slower than using IPC. This only has an effect with HMEM enabled
    builds of OPX. Defaults to on.

*FI_OPX_DEV_REG_SEND_THRESHOLD*
:   Integer. The individual packet threshold where lengths above do not
    use a device registered copy when sending data from GPU. The default
    threshold is 4096. This has no meaning if Libfabric was not
    configured with GDRCopy or ROCR support.

*FI_OPX_DEV_REG_RECV_THRESHOLD*
:   Integer. The individual packet threshold where lengths above do not
    use a device registered copy when receiving data into GPU. The
    default threshold is 8192. This has no meaning if Libfabric was not
    configured with GDRCopy or ROCR support.

*FI_OPX_MIXED_NETWORK*
:   Boolean (1/0, on/off, true/false, yes/no). Indicates that the
    network requires OPA100 support. Set to 0 if OPA100 support is not
    needed. Default is 1.

*FI_OPX_ROUTE_CONTROL*
:   Integer. Specify the route control for each packet type. The format
    is -
    `<inject packet type value>:<eager packet type value>:<multi-packet eager packet type value>:<dput packet type value>:<rendezvous control packet value>:<rendezvous data packet value>`.

Each value can range from 0-7. 0-3 is used for in-order and 4-7 is used
for out-of-order. If Token ID (TID) is enabled the out-of-order route
controls are disabled.

Default is `0:0:0:0:0:0` on OPA100 and `4:4:4:4:0:4` on CN5000.

*FI_OPX_SHM_ENABLE*
:   Boolean (1/0, on/off, true/false, yes/no). Enables shm across all
    ports and hfi units on the node. Setting it to NO disables shm
    except peers with same lid and same hfi1 (loopback). Defaults to:
    "YES"

*FI_OPX_LINK_DOWN_WAIT_TIME_MAX_SEC*
:   Integer. The maximum time in seconds to wait for a link to come back
    up. Default is 70 seconds.

*FI_OPX_MMAP_GUARD*
:   Boolean (0/1, on/off, true/false, yes/no). Enable guards around
    OPX/HFI mmaps. When enabled, this will cause a segfault when mmapped
    memory is illegally accessed through buffer overruns or underruns.
    Default is false.

*FI_OPX_CONTEXT_SHARING*
:   Boolean (1/0, on/off, true/false, yes/no). Enables context sharing
    in OPX. Defaults to FALSE (1 HFI context per endpoint).

*FI_OPX_ENDPOINTS_PER_HFI_CONTEXT*
:   Integer. Specify how many endpoints should share a single HFI
    context. Valid values are from 2 to 8. Default is to determine
    optimal value based on the number of contexts available on the
    system and number of processors online. Only applicable if context
    sharing is enabled. Otherwise this value is ignored.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(7)](fi_getinfo.7.html),

{% include JB/setup %}

# NAME

fi_psm2 - The PSM2 Fabric Provider

# OVERVIEW

The *psm2* provider runs over the PSM 2.x interface that is supported by
the Intel Omni-Path Fabric. PSM 2.x has all the PSM 1.x features plus a
set of new functions with enhanced capabilities. Since PSM 1.x and PSM
2.x are not ABI compatible the *psm2* provider only works with PSM 2.x
and doesn't support Intel TrueScale Fabric.

# LIMITATIONS

The *psm2* provider doesn't support all the features defined in the
libfabric API. Here are some of the limitations:

Endpoint types
:   Only support non-connection based types *FI_DGRAM* and *FI_RDM*

Endpoint capabilities
:   Endpoints can support any combination of data transfer capabilities
    *FI_TAGGED*, *FI_MSG*, *FI_ATOMICS*, and *FI_RMA*. These
    capabilities can be further refined by *FI_SEND*, *FI_RECV*,
    *FI_READ*, *FI_WRITE*, *FI_REMOTE_READ*, and *FI_REMOTE_WRITE* to
    limit the direction of operations.

*FI_MULTI_RECV* is supported for non-tagged message queue only.

Scalable endpoints are supported if the underlying PSM2 library supports
multiple endpoints. This condition must be satisfied both when the
provider is built and when the provider is used. See the *Scalable
endpoints* section for more information.

Other supported capabilities include *FI_TRIGGER*, *FI_REMOTE_CQ_DATA*,
*FI_RMA_EVENT*, *FI_SOURCE*, and *FI_SOURCE_ERR*. Furthermore,
*FI_NAMED_RX_CTX* is supported when scalable endpoints are enabled.

Modes
:   *FI_CONTEXT* is required for the *FI_TAGGED* and *FI_MSG*
    capabilities. That means, any request belonging to these two
    categories that generates a completion must pass as the operation
    context a valid pointer to type *struct fi_context*, and the space
    referenced by the pointer must remain untouched until the request
    has completed. If none of *FI_TAGGED* and *FI_MSG* is asked for, the
    *FI_CONTEXT* mode is not required.

Progress
:   The *psm2* provider requires manual progress. The application is
    expected to call *fi_cq_read* or *fi_cntr_read* function from time
    to time when no other libfabric function is called to ensure
    progress is made in a timely manner. The provider does support auto
    progress mode. However, the performance can be significantly
    impacted if the application purely depends on the provider to make
    auto progress.

Scalable endpoints
:   Scalable endpoints support depends on the multi-EP feature of the
    *PSM2* library. If the *PSM2* library supports this feature, the
    availability is further controlled by an environment variable
    *PSM2_MULTI_EP*. The *psm2* provider automatically sets this
    variable to 1 if it is not set. The feature can be disabled
    explicitly by setting *PSM2_MULTI_EP* to 0.

When creating a scalable endpoint, the exact number of contexts
requested should be set in the "fi_info" structure passed to the
*fi_scalable_ep* function. This number should be set in
"fi_info-\>ep_attr-\>tx_ctx_cnt" or "fi_info-\>ep_attr-\>rx_ctx_cnt" or
both, whichever greater is used. The *psm2* provider allocates all
requested contexts upfront when the scalable endpoint is created. The
same context is used for both Tx and Rx.

For optimal performance, it is advised to avoid having multiple threads
accessing the same context, either directly by posting
send/recv/read/write request, or indirectly by polling associated
completion queues or counters.

Using the scalable endpoint as a whole in communication functions is not
supported. Instead, individual tx context or rx context of the scalable
endpoint should be used. Similarly, using the address of the scalable
endpoint as the source address or destination address doesn't
collectively address all the tx/rx contexts. It addresses only the first
tx/rx context, instead.

Shared Tx contexts
:   In order to achieve the purpose of saving PSM context by using
    shared Tx context, the endpoints bound to the shared Tx contexts
    need to be Tx only. The reason is that Rx capability always requires
    a PSM context, which can also be automatically used for Tx. As the
    result, allocating a shared Tx context for Rx capable endpoints
    actually consumes one extra context instead of saving some.

Unsupported features
:   These features are unsupported: connection management, passive
    endpoint, and shared receive context.

# RUNTIME PARAMETERS

The *psm2* provider checks for the following environment variables:

*FI_PSM2_UUID*
:   PSM requires that each job has a unique ID (UUID). All the processes
    in the same job need to use the same UUID in order to be able to
    talk to each other. The PSM reference manual advises to keep UUID
    unique to each job. In practice, it generally works fine to reuse
    UUID as long as (1) no two jobs with the same UUID are running at
    the same time; and (2) previous jobs with the same UUID have exited
    normally. If running into "resource busy" or "connection failure"
    issues with unknown reason, it is advisable to manually set the UUID
    to a value different from the default.

The default UUID is 00FF00FF-0000-0000-0000-00FF0F0F00FF.

It is possible to create endpoints with UUID different from the one set
here. To achieve that, set 'info-\>ep_attr-\>auth_key' to the uuid value
and 'info-\>ep_attr-\>auth_key_size' to its size (16 bytes) when calling
fi_endpoint() or fi_scalable_ep(). It is still true that an endpoint can
only communicate with endpoints with the same UUID.

*FI_PSM2_NAME_SERVER*
:   The *psm2* provider has a simple built-in name server that can be
    used to resolve an IP address or host name into a transport address
    needed by the *fi_av_insert* call. The main purpose of this name
    server is to allow simple client-server type applications (such as
    those in *fabtests*) to be written purely with libfabric, without
    using any out-of-band communication mechanism. For such
    applications, the server would run first to allow endpoints be
    created and registered with the name server, and then the client
    would call *fi_getinfo* with the *node* parameter set to the IP
    address or host name of the server. The resulting *fi_info*
    structure would have the transport address of the endpoint created
    by the server in the *dest_addr* field. Optionally the *service*
    parameter can be used in addition to *node*. Notice that the
    *service* number is interpreted by the provider and is not a TCP/IP
    port number.

The name server is on by default. It can be turned off by setting the
variable to 0. This may save a small amount of resource since a separate
thread is created when the name server is on.

The provider detects OpenMPI and MPICH runs and changes the default
setting to off.

*FI_PSM2_TAGGED_RMA*
:   The RMA functions are implemented on top of the PSM Active Message
    functions. The Active Message functions have limit on the size of
    data can be transferred in a single message. Large transfers can be
    divided into small chunks and be pipe-lined. However, the bandwidth
    is sub-optimal by doing this way.

The *psm2* provider use PSM tag-matching message queue functions to
achieve higher bandwidth for large size RMA. It takes advantage of the
extra tag bits available in PSM2 to separate the RMA traffic from the
regular tagged message queue.

The option is on by default. To turn it off set the variable to 0.

*FI_PSM2_DELAY*
:   Time (seconds) to sleep before closing PSM endpoints. This is a
    workaround for a bug in some versions of PSM library.

The default setting is 0.

*FI_PSM2_TIMEOUT*
:   Timeout (seconds) for gracefully closing PSM endpoints. A forced
    closing will be issued if timeout expires.

The default setting is 5.

*FI_PSM2_CONN_TIMEOUT*
:   Timeout (seconds) for establishing connection between two PSM
    endpoints.

The default setting is 5.

*FI_PSM2_PROG_INTERVAL*
:   When auto progress is enabled (asked via the hints to *fi_getinfo*),
    a progress thread is created to make progress calls from time to
    time. This option set the interval (microseconds) between progress
    calls.

The default setting is 1 if affinity is set, or 1000 if not. See
*FI_PSM2_PROG_AFFINITY*.

*FI_PSM2_PROG_AFFINITY*
:   When set, specify the set of CPU cores to set the progress thread
    affinity to. The format is
    `<start>[:<end>[:<stride>]][,<start>[:<end>[:<stride>]]]*`, where
    each triplet `<start>:<end>:<stride>` defines a block of core_ids.
    Both `<start>` and `<end>` can be either the `core_id` (when \>=0)
    or `core_id - num_cores` (when \<0).

By default affinity is not set.

*FI_PSM2_INJECT_SIZE*
:   Maximum message size allowed for fi_inject and fi_tinject calls.
    This is an experimental feature to allow some applications to
    override default inject size limitation. When the inject size is
    larger than the default value, some inject calls might block.

The default setting is 64.

*FI_PSM2_LOCK_LEVEL*
:   When set, dictate the level of locking being used by the provider.
    Level 2 means all locks are enabled. Level 1 disables some locks and
    is suitable for runs that limit the access to each PSM2 context to a
    single thread. Level 0 disables all locks and thus is only suitable
    for single threaded runs.

To use level 0 or level 1, wait object and auto progress mode cannot be
used because they introduce internal threads that may break the
conditions needed for these levels.

The default setting is 2.

*FI_PSM2_LAZY_CONN*
:   There are two strategies on when to establish connections between
    the PSM2 endpoints that OFI endpoints are built on top of. In eager
    connection mode, connections are established when addresses are
    inserted into the address vector. In lazy connection mode,
    connections are established when addresses are used the first time
    in communication. Eager connection mode has slightly lower critical
    path overhead but lazy connection mode scales better.

This option controls how the two connection modes are used. When set to
1, lazy connection mode is always used. When set to 0, eager connection
mode is used when required conditions are all met and lazy connection
mode is used otherwise. The conditions for eager connection mode are:
(1) multiple endpoint (and scalable endpoint) support is disabled by
explicitly setting PSM2_MULTI_EP=0; and (2) the address vector type is
FI_AV_MAP.

The default setting is 0.

*FI_PSM2_DISCONNECT*
:   The provider has a mechanism to automatically send disconnection
    notifications to all connected peers before the local endpoint is
    closed. As the response, the peers call *psm2_ep_disconnect* to
    clean up the connection state at their side. This allows the same
    PSM2 epid be used by different dynamically started processes
    (clients) to communicate with the same peer (server). This
    mechanism, however, introduce extra overhead to the finalization
    phase. For applications that never reuse epids within the same
    session such overhead is unnecessary.

This option controls whether the automatic disconnection notification
mechanism should be enabled. For client-server application mentioned
above, the client side should set this option to 1, but the server
should set it to 0.

The default setting is 0.

*FI_PSM2_TAG_LAYOUT*
:   Select how the 96-bit PSM2 tag bits are organized. Currently three
    choices are available: *tag60* means 32-4-60 partitioning for CQ
    data, internal protocol flags, and application tag. *tag64* means
    4-28-64 partitioning for internal protocol flags, CQ data, and
    application tag. *auto* means to choose either *tag60* or *tag64*
    based on the hints passed to fi_getinfo -- *tag60* is used if remote
    CQ data support is requested explicitly, either by passing non-zero
    value via *hints-\>domain_attr-\>cq_data_size* or by including
    *FI_REMOTE_CQ_DATA* in *hints-\>caps*, otherwise *tag64* is used. If
    *tag64* is the result of automatic selection, *fi_getinfo* also
    returns a second instance of the provider with *tag60* layout.

The default setting is *auto*.

Notice that if the provider is compiled with macro *PSMX2_TAG_LAYOUT*
defined to 1 (means *tag60*) or 2 (means *tag64*), the choice is fixed
at compile time and this runtime option will be disabled.

# PSM2 EXTENSIONS

The *psm2* provider supports limited low level parameter setting through
the fi_set_val() and fi_get_val() functions. Currently the following
parameters can be set via the domain fid:

-   

    FI_PSM2_DISCONNECT \*
    :   Overwite the global runtime parameter *FI_PSM2_DISCONNECT* for
        this domain. See the *RUNTIME PARAMETERS* section for details.

Valid parameter names are defined in the header file
*rdma/fi_ext_psm2.h*.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_psm3`(7)](fi_psm3.7.html),

{% include JB/setup %}

# NAME

fi_psm3 - The PSM3 Fabric Provider

# OVERVIEW

The *psm3* provider implements a Performance Scaled Messaging capability
which supports most verbs UD and sockets devices. Additional features
and optimizations can be enabled when running over Intel's E810 Ethernet
NICs and/or using Intel's rendezvous kernel module
([`rv`](https://github.com/intel/iefs-kernel-updates)). PSM 3.x fully
integrates the OFI provider and the underlying PSM3
protocols/implementation and only exports the OFI APIs.

# SUPPORTED FEATURES

The *psm3* provider supports a subset of all the features defined in the
libfabric API.

Endpoint types
:   Supports non-connection based types *FI_DGRAM* and *FI_RDM*.

Endpoint capabilities
:   Endpoints can support any combination of data transfer capabilities
    *FI_TAGGED*, *FI_MSG*, *FI_ATOMICS*, and *FI_RMA*. These
    capabilities can be further refined by *FI_SEND*, *FI_RECV*,
    *FI_READ*, *FI_WRITE*, *FI_REMOTE_READ*, and *FI_REMOTE_WRITE* to
    limit the direction of operations.

*FI_MULTI_RECV* is supported for non-tagged message queue only.

Scalable endpoints are supported if the underlying PSM3 library supports
multiple endpoints. This condition must be satisfied both when the
provider is built and when the provider is used. See the *Scalable
endpoints* section for more information.

Other supported capabilities include *FI_TRIGGER*, *FI_REMOTE_CQ_DATA*,
*FI_RMA_EVENT*, *FI_SOURCE*, and *FI_SOURCE_ERR*. Furthermore,
*FI_NAMED_RX_CTX* is supported when scalable endpoints are enabled.

Modes
:   *FI_CONTEXT* is required for the *FI_TAGGED* and *FI_MSG*
    capabilities. That means, any request belonging to these two
    categories that generates a completion must pass as the operation
    context a valid pointer to type *struct fi_context*, and the space
    referenced by the pointer must remain untouched until the request
    has completed. If none of *FI_TAGGED* and *FI_MSG* is asked for, the
    *FI_CONTEXT* mode is not required.

Progress
:   The *psm3* provider performs optimal with manual progress. By
    default, the application is expected to call *fi_cq_read* or
    *fi_cntr_read* function from time to time when no other libfabric
    function is called to ensure progress is made in a timely manner.
    The provider does support auto progress mode. However, the
    performance can be significantly impacted if the application purely
    depends on the provider to make auto progress.

Scalable endpoints
:   Scalable endpoints support depends on the multi-EP feature of the
    *PSM3* library. If the *PSM3* library supports this feature, the
    availability is further controlled by an environment variable
    *PSM3_MULTI_EP*. The *psm3* provider automatically sets this
    variable to 1 if it is not set. The feature can be disabled
    explicitly by setting *PSM3_MULTI_EP* to 0.

When creating a scalable endpoint, the exact number of contexts
requested should be set in the "fi_info" structure passed to the
*fi_scalable_ep* function. This number should be set in
"fi_info-\>ep_attr-\>tx_ctx_cnt" or "fi_info-\>ep_attr-\>rx_ctx_cnt" or
both, whichever greater is used. The *psm3* provider allocates all
requested contexts upfront when the scalable endpoint is created. The
same context is used for both Tx and Rx.

For optimal performance, it is advised to avoid having multiple threads
accessing the same context, either directly by posting
send/recv/read/write request, or indirectly by polling associated
completion queues or counters.

Using the scalable endpoint as a whole in communication functions is not
supported. Instead, individual tx context or rx context of the scalable
endpoint should be used. Similarly, using the address of the scalable
endpoint as the source address or destination address doesn't
collectively address all the tx/rx contexts. It addresses only the first
tx/rx context, instead.

# LIMITATIONS

The *psm3* provider doesn't support all the features defined in the
libfabric API. Here are some of the limitations not listed above:

Unsupported features
:   These features are unsupported: connection management, passive
    endpoint, and shared receive context.

# RUNTIME PARAMETERS

The *psm3* provider checks for the following environment variables:

*FI_PSM3_UUID*
:   PSM requires that each job has a unique ID (UUID). All the processes
    in the same job need to use the same UUID in order to be able to
    talk to each other. The PSM reference manual advises to keep UUID
    unique to each job. In practice, it generally works fine to reuse
    UUID as long as (1) no two jobs with the same UUID are running at
    the same time; and (2) previous jobs with the same UUID have exited
    normally. If running into "resource busy" or "connection failure"
    issues with unknown reason, it is advisable to manually set the UUID
    to a value different from the default.

The default UUID is 00FF00FF-0000-0000-0000-00FF0F0F00FF.

It is possible to create endpoints with UUID different from the one set
here. To achieve that, set 'info-\>ep_attr-\>auth_key' to the uuid value
and 'info-\>ep_attr-\>auth_key_size' to its size (16 bytes) when calling
fi_endpoint() or fi_scalable_ep(). It is still true that an endpoint can
only communicate with endpoints with the same UUID.

*FI_PSM3_NAME_SERVER*
:   The *psm3* provider has a simple built-in name server that can be
    used to resolve an IP address or host name into a transport address
    needed by the *fi_av_insert* call. The main purpose of this name
    server is to allow simple client-server type applications (such as
    those in *fabtests*) to be written purely with libfabric, without
    using any out-of-band communication mechanism. For such
    applications, the server would run first to allow endpoints be
    created and registered with the name server, and then the client
    would call *fi_getinfo* with the *node* parameter set to the IP
    address or host name of the server. The resulting *fi_info*
    structure would have the transport address of the endpoint created
    by the server in the *dest_addr* field. Optionally the *service*
    parameter can be used in addition to *node*. Notice that the
    *service* number is interpreted by the provider and is not a TCP/IP
    port number.

The name server is on by default. It can be turned off by setting the
variable to 0. This may save a small amount of resource since a separate
thread is created when the name server is on.

The provider detects OpenMPI and MPICH runs and changes the default
setting to off.

*FI_PSM3_TAGGED_RMA*
:   The RMA functions are implemented on top of the PSM Active Message
    functions. The Active Message functions have limit on the size of
    data can be transferred in a single message. Large transfers can be
    divided into small chunks and be pipe-lined. However, the bandwidth
    is sub-optimal by doing this way.

The *psm3* provider use PSM tag-matching message queue functions to
achieve higher bandwidth for large size RMA. It takes advantage of the
extra tag bits available in PSM3 to separate the RMA traffic from the
regular tagged message queue.

The option is on by default. To turn it off set the variable to 0.

*FI_PSM3_DELAY*
:   Time (seconds) to sleep before closing PSM endpoints. This is a
    workaround for a bug in some versions of PSM library.

The default setting is 0.

*FI_PSM3_TIMEOUT*
:   Timeout (seconds) for gracefully closing PSM endpoints. A forced
    closing will be issued if timeout expires.

The default setting is 5.

*FI_PSM3_CONN_TIMEOUT*
:   Timeout (seconds) for establishing connection between two PSM
    endpoints.

The default setting is 5.

*FI_PSM3_PROG_INTERVAL*
:   When auto progress is enabled (asked via the hints to *fi_getinfo*),
    a progress thread is created to make progress calls from time to
    time. This option set the interval (microseconds) between progress
    calls.

The default setting is 1 if affinity is set, or 1000 if not. See
*FI_PSM3_PROG_AFFINITY*.

*FI_PSM3_PROG_AFFINITY*
:   When set, specify the set of CPU cores to set the progress thread
    affinity to. The format is
    `<start>[:<end>[:<stride>]][,<start>[:<end>[:<stride>]]]*`, where
    each triplet `<start>:<end>:<stride>` defines a block of core_ids.
    Both `<start>` and `<end>` can be either the `core_id` (when \>=0)
    or `core_id - num_cores` (when \<0).

By default affinity is not set.

*FI_PSM3_INJECT_SIZE*
:   Maximum message size allowed for fi_inject and fi_tinject calls.
    This is an experimental feature to allow some applications to
    override default inject size limitation. When the inject size is
    larger than the default value, some inject calls might block.

The default setting is 64.

*FI_PSM3_LOCK_LEVEL*
:   When set, dictate the level of locking being used by the provider.
    Level 2 means all locks are enabled. Level 1 disables some locks and
    is suitable for runs that limit the access to each PSM3 context to a
    single thread. Level 0 disables all locks and thus is only suitable
    for single threaded runs.

To use level 0 or level 1, wait object and auto progress mode cannot be
used because they introduce internal threads that may break the
conditions needed for these levels.

The default setting is 2.

*FI_PSM3_LAZY_CONN*
:   There are two strategies on when to establish connections between
    the PSM3 endpoints that OFI endpoints are built on top of. In eager
    connection mode, connections are established when addresses are
    inserted into the address vector. In lazy connection mode,
    connections are established when addresses are used the first time
    in communication. Eager connection mode has slightly lower critical
    path overhead but lazy connection mode scales better.

This option controls how the two connection modes are used. When set to
1, lazy connection mode is always used. When set to 0, eager connection
mode is used when required conditions are all met and lazy connection
mode is used otherwise. The conditions for eager connection mode are:
(1) multiple endpoint (and scalable endpoint) support is disabled by
explicitly setting PSM3_MULTI_EP=0; and (2) the address vector type is
FI_AV_MAP.

The default setting is 0.

*FI_PSM3_DISCONNECT*
:   The provider has a mechanism to automatically send disconnection
    notifications to all connected peers before the local endpoint is
    closed. As the response, the peers call *psm3_ep_disconnect* to
    clean up the connection state at their side. This allows the same
    PSM3 epid be used by different dynamically started processes
    (clients) to communicate with the same peer (server). This
    mechanism, however, introduce extra overhead to the finalization
    phase. For applications that never reuse epids within the same
    session such overhead is unnecessary.

This option controls whether the automatic disconnection notification
mechanism should be enabled. For client-server application mentioned
above, the client side should set this option to 1, but the server
should set it to 0.

The default setting is 0.

*FI_PSM3_TAG_LAYOUT*
:   Select how the 96-bit PSM3 tag bits are organized. Currently three
    choices are available: *tag60* means 32-4-60 partitioning for CQ
    data, internal protocol flags, and application tag. *tag64* means
    4-28-64 partitioning for internal protocol flags, CQ data, and
    application tag. *auto* means to choose either *tag60* or *tag64*
    based on the hints passed to fi_getinfo -- *tag60* is used if remote
    CQ data support is requested explicitly, either by passing non-zero
    value via *hints-\>domain_attr-\>cq_data_size* or by including
    *FI_REMOTE_CQ_DATA* in *hints-\>caps*, otherwise *tag64* is used. If
    *tag64* is the result of automatic selection, *fi_getinfo* also
    returns a second instance of the provider with *tag60* layout.

The default setting is *auto*.

Notice that if the provider is compiled with macro *PSMX3_TAG_LAYOUT*
defined to 1 (means *tag60*) or 2 (means *tag64*), the choice is fixed
at compile time and this runtime option will be disabled.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_psm2`(7)](fi_psm2.7.html),

{% include JB/setup %}

# NAME

fi_rxd - The RxD (RDM over DGRAM) Utility Provider

# OVERVIEW

The RxD provider is a utility provider that supports RDM endpoints
emulated over a base DGRAM provider.

# SUPPORTED FEATURES

The RxD provider currently supports *FI_MSG* capabilities.

*Endpoint types*
:   The provider supports only endpoint type *FI_EP_RDM*.

*Endpoint capabilities* : The following data transfer interface is
supported: *fi_msg*.

*Modes*
:   The provider does not require the use of any mode bits but supports
    core DGRAM providers that require FI_CONTEXT and FI_MSG_PREFIX.

*Progress*
:   The RxD provider only supports *FI_PROGRESS_MANUAL*.

# LIMITATIONS

The RxD provider has hard-coded maximums for supported queue sizes and
data transfers. Some of these limits are set based on the selected base
DGRAM provider.

No support for multi-recv.

No support for counters.

The RxD provider is still under development and is not extensively
tested.

# RUNTIME PARAMETERS

The *rxd* provider checks for the following environment variables:

*FI_OFI_RXD_SPIN_COUNT*
:   Number of times to read the core provider's CQ for a segment
    completion before trying to progress sends. Default is 1000.

*FI_OFI_RXD_RETRY*
:   Toggles retrying of packets and assumes reliability of individual
    packets and will reassemble all received packets. Retrying is turned
    on by default.

*FI_OFI_RXD_MAX_PEERS*
:   Maximum number of peers the provider should prepare to track.
    Default: 1024

*FI_OFI_RXD_MAX_UNACKED*
:   Maximum number of packets (per peer) to send at a time. Default: 128

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_rxm - The RxM (RDM over MSG) Utility Provider

# OVERVIEW

The RxM provider (ofi_rxm) is an utility provider that supports
FI_EP_RDM type endpoint emulated over FI_EP_MSG type endpoint(s) of an
underlying core provider. FI_EP_RDM endpoints have a reliable datagram
interface and RxM emulates this by hiding the connection management of
underlying FI_EP_MSG endpoints from the user. Additionally, RxM can hide
memory registration requirement from a core provider like verbs if the
apps don't support it.

# REQUIREMENTS

## Requirements for core provider

RxM provider requires the core provider to support the following
features:

-   MSG endpoints (FI_EP_MSG)

-   RMA read/write (FI_RMA) - Used for implementing rendezvous protocol
    for large messages.

-   FI_OPT_CM_DATA_SIZE of at least 24 bytes.

## Requirements for applications

Since RxM emulates RDM endpoints by hiding connection management and
connections are established only on-demand (when app tries to send
data), the first several data transfer calls would return EAGAIN.
Applications should be aware of this and retry until the operation
succeeds.

If an application has chosen manual progress for data progress, it
should also read the CQ so that the connection establishment progresses.
Not doing so would result in a stall. See also the ERRORS section in
fi_msg(3).

# SUPPORTED FEATURES

The RxM provider currently supports *FI_MSG*, *FI_TAGGED*, *FI_RMA* and
*FI_ATOMIC* capabilities.

*Endpoint types*
:   The provider supports only *FI_EP_RDM*.

*Endpoint capabilities*
:   The following data transfer interface is supported: *FI_MSG*,
    *FI_TAGGED*, *FI_RMA*, *FI_ATOMIC*.

*Progress*
:   The RxM provider supports both *FI_PROGRESS_MANUAL* and
    *FI_PROGRESS_AUTO*. Manual progress in general has better connection
    scale-up and lower CPU utilization since there's no separate
    auto-progress thread.

*Addressing Formats*
:   FI_SOCKADDR, FI_SOCKADDR_IN

*Memory Region*
:   FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, FI_MR_PROV_KEY MR mode bits would
    be required from the app in case the core provider requires it.

# LIMITATIONS

When using RxM provider, some limitations from the underlying MSG
provider could also show up. Please refer to the corresponding MSG
provider man pages to find about those limitations.

## Unsupported features

RxM provider does not support the following features:

-   op_flags: FI_FENCE.

-   Scalable endpoints

-   Shared contexts

-   FABRIC_DIRECT

-   FI_MR_SCALABLE

-   Authorization keys

-   Application error data buffers

-   Multicast

-   FI_SYNC_ERR

-   Reporting unknown source addr data as part of completions

-   Triggered operations

## Progress limitations

When sending large messages, an app doing an sread or waiting on the CQ
file descriptor may not get a completion when reading the CQ after being
woken up from the wait. The app has to do sread or wait on the file
descriptor again. This is needed because RxM uses a rendezvous protocol
for large message sends. An app would get woken up from waiting on CQ fd
when rendezvous protocol request completes but it would have to wait
again to get an ACK from the receiver indicating completion of large
message transfer by remote RMA read.

## FI_ATOMIC limitations

The FI_ATOMIC capability will only be listed in the fi_info if the
fi_info hints parameter specifies FI_ATOMIC. If FI_ATOMIC is requested,
message order FI_ORDER_RAR, FI_ORDER_RAW, FI_ORDER_WAR, FI_ORDER_WAW,
FI_ORDER_SAR, and FI_ORDER_SAW can not be supported.

## Miscellaneous limitations

-   RxM protocol peers should have same endian-ness otherwise
    connections won't successfully complete. This enables better
    performance at run-time as byte order translations are avoided.

# RUNTIME PARAMETERS

The ofi_rxm provider checks for the following environment variables.

*FI_OFI_RXM_BUFFER_SIZE*
:   Defines the transmit buffer size / inject size. Messages of size
    less than or equal to this would be transmitted via an eager
    protocol and messages greater in size would be transmitted via a
    rendezvous or SAR (Segmentation And Reassembly) protocol. Transmit
    data would be copied up to this size (default: \~16k).

*FI_OFI_RXM_COMP_PER_PROGRESS*
:   Defines the maximum number of MSG provider CQ entries (default: 1)
    that would be read per progress (RxM CQ read).

*FI_OFI_RXM_ENABLE_DYN_RBUF*
:   Enables support for dynamic receive buffering, if available by the
    message endpoint provider. This feature allows direct placement of
    received message data into application buffers, bypassing RxM bounce
    buffers. This feature targets providers that provide internal
    network buffering, such as the tcp provider. (default: false)

*FI_OFI_RXM_SAR_LIMIT*
:   Set this environment variable to control the RxM SAR (Segmentation
    And Reassembly) protocol. Messages of size greater than this
    (default: 128 Kb) would be transmitted via rendezvous protocol.

*FI_OFI_RXM_USE_SRX*
:   Set this to 1 to use shared receive context from MSG provider, or 0
    to disable using shared receive context. Shared receive contexts
    reduce overall memory usage, but may increase in message latency. If
    not set, verbs will not use shared receive contexts by default, but
    the tcp provider will.

*FI_OFI_RXM_TX_SIZE*
:   Defines default TX context size (default: 1024)

*FI_OFI_RXM_RX_SIZE*
:   Defines default RX context size (default: 1024)

*FI_OFI_RXM_MSG_TX_SIZE*
:   Defines FI_EP_MSG TX size that would be requested (default: 128).

*FI_OFI_RXM_MSG_RX_SIZE*
:   Defines FI_EP_MSG RX size that would be requested (default: 128).

*FI_UNIVERSE_SIZE*
:   Defines the expected number of ranks / peers an endpoint would
    communicate with (default: 256).

*FI_OFI_RXM_CM_PROGRESS_INTERVAL*
:   Defines the duration of time in microseconds between calls to RxM CM
    progression functions when using manual progress. Higher values may
    provide less noise for calls to fi_cq read functions, but may
    increase connection setup time (default: 10000)

*FI_OFI_RXM_CQ_EQ_FAIRNESS*
:   Defines the maximum number of message provider CQ entries that can
    be consecutively read across progress calls without checking to see
    if the CM progress interval has been reached (default: 128)

*FI_OFI_RXM_DETECT_HMEM_IFACE*
:   Set this to 1 to allow automatic detection of HMEM iface of user
    buffers when such information is not supplied. This feature allows
    such buffers be copied or registered (e.g. in Rendezvous) internally
    by RxM. Note that no extra memory registration is performed with
    this option. (default: false)

# Tuning

## Bandwidth

To optimize for bandwidth, ensure you use higher values than default for
FI_OFI_RXM_TX_SIZE, FI_OFI_RXM_RX_SIZE, FI_OFI_RXM_MSG_TX_SIZE,
FI_OFI_RXM_MSG_RX_SIZE subject to memory limits of the system and the tx
and rx sizes supported by the MSG provider.

FI_OFI_RXM_SAR_LIMIT is another knob that can be experimented with to
optimze for bandwidth.

## Memory

To conserve memory, ensure FI_UNIVERSE_SIZE set to what is required.
Similarly check that FI_OFI_RXM_TX_SIZE, FI_OFI_RXM_RX_SIZE,
FI_OFI_RXM_MSG_TX_SIZE and FI_OFI_RXM_MSG_RX_SIZE env variables are set
to only required values.

# NOTES

The data transfer API may return -FI_EAGAIN during on-demand connection
setup of the core provider FI_MSG_EP. See [`fi_msg`(3)](fi_msg.3.html)
for a detailed description of handling FI_EAGAIN.

# Troubleshooting / Known issues

If an RxM endpoint is expected to communicate with more peers than the
default value of FI_UNIVERSE_SIZE (256) CQ overruns can happen. To avoid
this set a higher value for FI_UNIVERSE_SIZE. CQ overrun can make a MSG
endpoint unusable.

At higher \# of ranks, there may be connection errors due to a node
running out of memory. The workaround is to use shared receive contexts
for the MSG provider (FI_OFI_RXM_USE_SRX=1) or reduce eager message size
(FI_OFI_RXM_BUFFER_SIZE) and MSG provider TX/RX queue sizes
(FI_OFI_RXM_MSG_TX_SIZE / FI_OFI_RXM_MSG_RX_SIZE).

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_shm - The SHM Fabric Provider

# OVERVIEW

The SHM provider is a complete provider that can be used on Linux
systems supporting shared memory and process_vm_readv/process_vm_writev
calls. The provider is intended to provide high-performance
communication between processes on the same system.

# SUPPORTED FEATURES

The SHM provider offers the following support:

*Endpoint types*
:   The provider supports only endpoint type *FI_EP_RDM*.

*Endpoint capabilities*
:   Endpoints can support any combination of the following data transfer
    capabilities: *FI_MSG*, *FI_TAGGED*, *FI_RMA*, amd *FI_ATOMICS*.
    These capabilities can be further defined by *FI_SEND*, *FI_RECV*,
    *FI_READ*, *FI_WRITE*, *FI_REMOTE_READ*, and *FI_REMOTE_WRITE* to
    limit the direction of operations.

*Modes*
:   The provider does not require the use of any mode bits.

*Progress*
:   The SHM provider supports *FI_PROGRESS_MANUAL*. Receive side data
    buffers are not modified outside of completion processing routines.
    The provider processes messages using three different methods, based
    on the size of the message. For messages smaller than 4096 bytes, tx
    completions are generated immediately after the send. For larger
    messages, tx completions are not generated until the receiving side
    has processed the message.

*Address Format*
:   The SHM provider uses the address format FI_ADDR_STR, which follows
    the general format pattern "\[prefix\]://[addr](#addr)". The
    application can provide addresses through the node or hints
    parameter. As long as the address is in a valid FI_ADDR_STR format
    (contains "://"), the address will be used as is. If the application
    input is incorrectly formatted or no input was provided, the SHM
    provider will resolve it according to the following SHM provider
    standards:

(flags & FI_SOURCE) ? src_addr : dest_addr = - if (node && service) :
"fi_ns://node:service" - if (service) : "fi_ns://service" - if (node &&
!service) : "fi_shm://node" - if (!node && !service) : "fi_shm://PID"

!(flags & FI_SOURCE) - src_addr = "fi_shm://PID"

In other words, if the application provides a source and/or destination
address in an acceptable FI_ADDR_STR format (contains "://"), the call
to util_getinfo will successfully fill in src_addr and dest_addr with
the provided input. If the input is not in an ADDR_STR format, the
shared memory provider will then create a proper FI_ADDR_STR address
with either the "fi_ns://" (node/service format) or "fi_shm://" (shm
format) prefixes signaling whether the addr is a "unique" address and
does or does not need an extra endpoint name identifier appended in
order to make it unique. For the shared memory provider, we assume that
the service (with or without a node) is enough to make it unique, but a
node alone is not sufficient. If only a node is provided, the
"fi_shm://" prefix is used to signify that it is not a unique address.
If no node or service are provided (and in the case of setting the src
address without FI_SOURCE and no hints), the process ID will be used as
a default address. On endpoint creation, if the src_addr has the
"fi_shm://" prefix, the provider will append ":\[uid\]:\[ep_idx\]" as a
unique endpoint name (essentially, in place of a service). In the case
of the "fi_ns://" prefix (or any other prefix if one was provided by the
application), no supplemental information is required to make it unique
and it will remain with only the application-defined address. Note that
the actual endpoint name will not include the FI_ADDR_STR "\*://" prefix
since it cannot be included in any shared memory region names. The
provider will strip off the prefix before setting the endpoint name. As
a result, the addresses "fi_prefix1://my_node:my_service" and
"fi_prefix2://my_node:my_service" would result in endpoints and regions
of the same name. The application can also override the endpoint name
after creating an endpoint using setname() without any address format
restrictions.

*Msg flags* The provider supports the following msg flags: - FI_CLAIM -
FI_COMPLETION - FI_DELIVERY_COMPLETE - FI_DISCARD - FI_INJECT -
FI_MULTI_RECV - FI_PEEK - FI_REMOTE_CQ_DATA

*MR registration mode* The provider can optimize RMA calls if the
application supports FI_MR_VIRT_ADDR. Otherwise, no extra MR modes are
required. If FI_HMEM support is requested, the provider will require
FI_MR_HMEM.

*Atomic operations* The provider supports all combinations of datatype
and operations as long as the message is less than 4096 bytes (or 2048
for compare operations).

# DSA

Intel Data Streaming Accelerator (DSA) is an integrated accelerator in
Intel Xeon processors starting with the Sapphire Rapids generation. One
of the capabilities of DSA is to offload memory copy operations from the
CPU. A system may have one or more DSA devices. Each DSA device may have
one or more work queues. The DSA specification can be found
[here](https://www.intel.com/content/www/us/en/develop/articles/intel-data-streaming-accelerator-architecture-specification.html).

The SAR protocol of the SHM provider is enabled to take advantage of DSA
to offload memory copy operations into and out of SAR buffers in shared
memory regions. To fully take advantage of the DSA offload capability,
memory copy operations are performed asynchronously. The copy initiator
thread constructs the DSA commands and submits to work queues. A copy
operation may consist of more than one DSA command. In such a case,
commands are spread across all available work queues in round robin
fashion. The progress thread checks for DSA command completions. If the
copy command successfully completes, it then notifies the peer to
consume the data. If DSA encounters a page fault during command
execution, the page fault is reported via completion records. In such a
case, the progress thread accesses the page to resolve the page fault
and resubmits the command after adjusting for partial completions. One
of the benefits of making memory copy operations asynchronous is that
now data transfers between different target endpoints can be initiated
in parallel. Use of Intel DSA in SAR protocol is disabled by default and
can be enabled using an environment variable. Note that CMA must be
disabled, e.g. FI_SHM_DISABLE_CMA=0, in order for DSA to be used. See
the RUNTIME PARAMETERS section.

Compiling with DSA capabilities depends on the accel-config library
which can be found [here](https://github.com/intel/idxd-config). Running
with DSA requires using Linux Kernel 5.19.0-rc3 or later.

DSA devices need to be setup just once before runtime. [This
configuration
file](https://github.com/intel/idxd-config/blob/stable/contrib/configs/os_profile.conf)
can be used as a template with the accel-config utility to configure the
DSA devices.

# LIMITATIONS

The SHM provider has hard-coded maximums for supported queue sizes and
data transfers. These values are reflected in the related fabric
attribute structures.

# RUNTIME PARAMETERS

The *shm* provider checks for the following environment variables:

*FI_SHM_SAR_THRESHOLD*
:   Maximum message size to use segmentation protocol before switching
    to mmap (only valid when CMA is not available). Default: SIZE_MAX
    (18446744073709551615)

*FI_SHM_TX_SIZE*
:   Maximum number of outstanding tx operations. Default 1024

*FI_SHM_RX_SIZE*
:   Maximum number of outstanding rx operations. Default 1024

*FI_SHM_DISABLE_CMA*
:   Manually disables CMA. Default false

*FI_SHM_USE_DSA_SAR*
:   Enables memory copy offload to Intel DSA SAR protocol. Default false

*FI_SHM_USE_XPMEM*
:   SHM can use SAR, CMA or XPMEM for host memory transfers. If
    FI_SHM_USE_XPMEM is set to 1, the provider will select XPMEM over
    CMA if XPMEM is available. Otherwise, if neither CMA nor XPMEM are
    available SHM shall default to the SAR protocol. Default 0

*FI_XPMEM_MEMCPY_CHUNKSIZE*
:   The maximum size which will be used with a single memcpy call. XPMEM
    copy performance improves when buffers are divided into smaller
    chunks. This environment variable is provided to fine tune
    performance on different systems. Default 262144

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_sockets - The Sockets Fabric Provider

# OVERVIEW

The sockets provider is being deprecated in favor of the tcp, udp, and
utility providers. Further work on the sockets provider will be minimal.
Most applications should instead use the tcp provider instead.

The sockets provider is a general purpose provider that can be used on
any system that supports TCP sockets. The provider is not intended to
provide performance improvements over regular TCP sockets, but rather to
allow developers to write, test, and debug application code even on
platforms that do not have high-performance fabric hardware. The sockets
provider supports all libfabric provider requirements and interfaces.

# SUPPORTED FEATURES

The sockets provider supports all the features defined for the libfabric
API. Key features include:

*Endpoint types*
:   The provider supports all endpoint types: *FI_EP_MSG*, *FI_EP_RDM*,
    and *FI_EP_DGRAM*.

*Endpoint capabilities*
:   The following data transfer interface is supported for a all
    endpoint types: *fi_msg*. Additionally, these interfaces are
    supported for reliable endpoints (*FI_EP_MSG* and *FI_EP_RDM*):
    *fi_tagged*, *fi_atomic*, and *fi_rma*.

*Modes*
:   The sockets provider supports all operational modes including
    *FI_CONTEXT* and *FI_MSG_PREFIX*.

*Progress*
:   Sockets provider supports both *FI_PROGRESS_AUTO* and
    *FI_PROGRESS_MANUAL*, with a default set to auto. When progress is
    set to auto, a background thread runs to ensure that progress is
    made for asynchronous requests.

# LIMITATIONS

Sockets provider attempts to emulate the entire API set, including all
defined options. In order to support development on a wide range of
systems, it is implemented over TCP sockets. As a result, the
performance numbers are lower compared to other providers implemented
over high-speed fabric, and lower than what an application might see
implementing to sockets directly.

Does not support FI_ADDR_STR address format.

# RUNTIME PARAMETERS

The sockets provider checks for the following environment variables -

*FI_SOCKETS_PE_WAITTIME*
:   An integer value that specifies how many milliseconds to spin while
    waiting for progress in *FI_PROGRESS_AUTO* mode.

*FI_SOCKETS_CONN_TIMEOUT*
:   An integer value that specifies how many milliseconds to wait for
    one connection establishment.

*FI_SOCKETS_MAX_CONN_RETRY*
:   An integer value that specifies the number of socket connection
    retries before reporting as failure.

*FI_SOCKETS_DEF_CONN_MAP_SZ*
:   An integer to specify the default connection map size.

*FI_SOCKETS_DEF_AV_SZ*
:   An integer to specify the default address vector size.

*FI_SOCKETS_DEF_CQ_SZ*
:   An integer to specify the default completion queue size.

*FI_SOCKETS_DEF_EQ_SZ*
:   An integer to specify the default event queue size.

*FI_SOCKETS_DGRAM_DROP_RATE*
:   An integer value to specify the drop rate of dgram frame when
    endpoint is *FI_EP_DGRAM*. This is for debugging purpose only.

*FI_SOCKETS_PE_AFFINITY*
:   If specified, progress thread is bound to the indicated range(s) of
    Linux virtual processor ID(s). This option is currently not
    supported on OS X. The usage is -
    id_start\[-id_end\[:stride\]\]\[,\].

*FI_SOCKETS_KEEPALIVE_ENABLE*
:   A boolean to enable the keepalive support.

*FI_SOCKETS_KEEPALIVE_TIME*
:   An integer to specify the idle time in seconds before sending the
    first keepalive probe. Only relevant if
    *FI_SOCKETS_KEEPALIVE_ENABLE* is enabled.

*FI_SOCKETS_KEEPALIVE_INTVL*
:   An integer to specify the time in seconds between individual
    keepalive probes. Only relevant if *FI_SOCKETS_KEEPALIVE_ENABLE* is
    enabled.

*FI_SOCKETS_KEEPALIVE_PROBES*
:   An integer to specify the maximum number of keepalive probes sent
    before dropping the connection. Only relevant if
    *FI_SOCKETS_KEEPALIVE_ENABLE* is enabled.

*FI_SOCKETS_IFACE*
:   The prefix or the name of the network interface (default: any)

# LARGE SCALE JOBS

For large scale runs one can use these environment variables to set the
default parameters e.g. size of the address vector(AV), completion queue
(CQ), connection map etc. that satisfies the requirement of the
particular benchmark. The recommended parameters for large scale runs
are *FI_SOCKETS_MAX_CONN_RETRY*, *FI_SOCKETS_DEF_CONN_MAP_SZ*,
*FI_SOCKETS_DEF_AV_SZ*, *FI_SOCKETS_DEF_CQ_SZ*, *FI_SOCKETS_DEF_EQ_SZ*.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_tcp - Provider that runs over TCP/IP

# OVERVIEW

The tcp provider is usable on all operation systems supported by
libfabric. It runs over TCP (SOCK_STREAM) sockets, and includes the
ability to leverage operation specific features, such as support for
zero-copy and io_uring. The provider implements a custom protocol over
TCP/IP needed to support the libfabric communication APIs.

# SUPPORTED FEATURES

The following features are supported

*Endpoint types*
:   *FI_EP_MSG*
:   *FI_EP_RDM*

*Endpoint capabilities*
:   *FI_MSG*, *FI_RMA*, *FI_TAGGED*, *FI_RMA_PMEM*, *FI_RMA_EVENT*,
    *FI_MULTI_RECV*, *FI_DIRECTED_RECV*

*Shared Rx Context*
:   The tcp provider supports shared receive context

# RUNTIME PARAMETERS

The tcp provider may be configured using several environment variables.
A subset of supported variables is defined below. For a full list, use
the fi_info utility application. For example, 'fi_info -g tcp' will show
all environment variables defined for the tcp provider.

*FI_TCP_IFACE*
:   A specific network interface can be requested with this variable

*FI_TCP_PORT_LOW_RANGE/FI_TCP_PORT_HIGH_RANGE*
:   These variables are used to set the range of ports to be used by the
    tcp provider for its passive endpoint creation. This is useful where
    only a range of ports are allowed by firewall for tcp connections.

*FI_TCP_TX_SIZE*
:   Transmit context size. This is the number of transmit requests that
    an application may post to the provider before receiving -FI_EAGAIN.
    Default: 256 for msg endpoints, 64k for rdm.

*FI_TCP_RX_SIZE*
:   Receive context size. This is the number of receive buffers that the
    application may post to the provider. Default: 256 for msg
    endpoints, 64k for rdm.

*FI_TCP_MAX_INJECT*
:   Maximum size of inject messages and the maximum size of an
    unexpected message that may be buffered at the receiver. Default 128
    bytes.

*FI_TCP_STAGING_SBUF_SIZE*
:   Size of buffer used to coalesce iovec's or send requests before
    posting to the kernel. The staging buffer is used when the socket is
    busy and cannot accept new data. In that case, the data can be
    queued in the staging buffer until the socket resumes sending. This
    optimizes transfering a series of back-to-back small messages to the
    same target. Default: 9000 bytes. Set to 0 to disable.

*FI_TCP_PREFETCH_RBUF_SIZE*
:   Size of the buffer used to prefetch received data from the kernel.
    When starting to receive a new message, the provider will request
    that the kernel fill the prefetch buffer and process received data
    from there. This reduces the number of kernel calls needed to
    receive a series of small messages. Default: 9000 bytes. Set to 0 to
    disable.

*FI_TCP_ZEROCOPY_SIZE*
:   Lower threshold where zero copy transfers will be used, if supported
    by the platform, set to -1 to disable. Default: disabled.

*FI_TCP_TRACE_MSG*
:   If enabled, will log transport message information on all sent and
    received messages. Must be paired with FI_LOG_LEVEL=trace to print
    the message details.

*FI_TCP_IO_URING*
:   Uses io_uring for socket operations if available, rather than going
    through the standard socket APIs (i.e. connect, accept, send, recv).
    Default: disabled.

# CONTROL OPERATIONS

The tcp provider supports the following control operations (see
[`fi_control`(3)](fi_control.3.html)):

*FI_GET_FD*
:   Retrieve the underlying socket file descriptor associated with an
    active endpoint. The argument must point to an integer where the
    descriptor will be stored. This allows applications to tune socket
    options not exposed through the libfabric API (SO_SNDBUF, SO_RCVBUF,
    SO_BUSY_POLL, etc).

# NOTES

The tcp provider supports both msg and rdm endpoints directly. Support
for rdm endpoints is available starting at libfabric version v1.18.0,
and comes from the merge back of the net provider found in libfabric
versions v1.16 and v1.17. For compatibility with older libfabric
versions, the tcp provider may be paired with the ofi_rxm provider as an
alternative solution for rdm endpoint support. It is recommended that
applications that do not need wire compatibility with older versions of
libfabric use the rdm endpoint support directly from the tcp provider.
This will provide the best performance.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_ucx - The UCX Fabric Provider

# OVERVIEW

The *ucx* provider runs over the UCX library that is currently supported
by the NVIDIA Infiniband fabrics. The *ucx* provider makes use of UCX
tag matching API in order to implement a limited set of the libfabric
data transfer APIs.

Supported UCP API version: 1.0

# LIMITATIONS

The *ucx* provider doesn't support all the features defined in the
libfabric API. Here are some of the limitations:

Endpoint types
:   The only supported type is *FI_EP_RDM*.

Endpoint capabilities
:   Endpoints support data transfer capabilities *FI_MSG*, *FI_TAGGED*,
    *FI_RMA* and *FI_MULTI_RECV*.

Threading
:   The supported threading mode is *FI_THREAD_DOMAIN*, i.e. the *ucx*
    provider is not thread safe.

# RUNTIME PARAMETERS

*FI_UCX_CONFIG*
:   The path to the UCX configuration file (default: none).

*FI_UCX_TINJECT_LIMIT*
:   Maximal tinject message size (default: 1024).

*FI_UCX_NS_ENABLE*
:   Enforce usage of name server functionality for UCX provider
    (default: disabled).

*FI_UCX_NS_PORT*
:   UCX provider's name server port (default: 12345).

*FI_UCX_NS_IFACE*
:   IPv4 network interface for UCX provider's name server (default:
    any).

*FI_UCX_CHECK_REQ_LEAK*
:   Check request leak (default: disabled).

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),

{% include JB/setup %}

# NAME

fi_udp - The UDP Fabric Provider

# OVERVIEW

The UDP provider is a basic provider that can be used on any system that
supports UDP sockets. The provider is not intended to provide
performance improvements over regular TCP sockets, but rather to allow
application and provider developers to write, test, and debug their
code. The UDP provider forms the foundation of a utility provider that
enables the implementation of libfabric features over any hardware.

# SUPPORTED FEATURES

The UDP provider supports a minimal set of features useful for sending
and receiving datagram messages over an unreliable endpoint.

*Endpoint types*
:   The provider supports only endpoint type *FI_EP_DGRAM*.

*Endpoint capabilities*
:   The following data transfer interface is supported: *fi_msg*. The
    provider supports standard unicast datagram transfers, as well as
    multicast operations.

*Modes*
:   The provider does not require the use of any mode bits.

*Progress*
:   The UDP provider supports both *FI_PROGRESS_AUTO* and
    *FI_PROGRESS_MANUAL*, with a default set to auto. However, receive
    side data buffers are not modified outside of completion processing
    routines.

# LIMITATIONS

The UDP provider has hard-coded maximums for supported queue sizes and
data transfers. These values are reflected in the related fabric
attribute structures

EPs must be bound to both RX and TX CQs.

No support for selective completions or multi-recv.

No support for counters.

# RUNTIME PARAMETERS

No runtime parameters are currently defined.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(3)](fi_getinfo.3.html)

{% include JB/setup %}

# NAME

fi_usnic - The usNIC Fabric Provider

# OVERVIEW

The *usnic* provider is designed to run over the Cisco VIC (virtualized
NIC) hardware on Cisco UCS servers. It utilizes the Cisco usNIC
(userspace NIC) capabilities of the VIC to enable ultra low latency and
other offload capabilities on Ethernet networks.

# RELEASE NOTES

-   The *usnic* libfabric provider requires the use of the "libnl"
    library.
    -   There are two versions of libnl generally available: v1 and v3;
        the usnic provider can use either version.
    -   If you are building libfabric/the usnic provider from source,
        you will need to have the libnl header files available (e.g., if
        you are installing libnl from RPM or other packaging system,
        install the "-devel" versions of the package).
    -   If you have libnl (either v1 or v3) installed in a non-standard
        location (e.g., not in /usr/lib or /usr/lib64), you may need to
        tell libfabric's configure where to find libnl via the
        `--with-libnl=DIR` command line option (where DIR is the
        installation prefix of the libnl package).
-   The most common way to use the libfabric usnic provider is via an
    MPI implementation that uses libfabric (and the usnic provider) as a
    lower layer transport. MPI applications do not need to know anything
    about libfabric or usnic in this use case -- the MPI implementation
    hides all these details from the application.
-   If you are writing applications directly to the libfabric API:
    -   *FI_EP_DGRAM* endpoints are the best supported method of
        utilizing the usNIC interface. Specifically, the *FI_EP_DGRAM*
        endpoint type has been extensively tested as the underlying
        layer for Open MPI's *usnic* BTL.
    -   *FI_EP_MSG* and *FI_EP_RDM* endpoints are implemented, but are
        only lightly tested. It is likely that there are still some bugs
        in these endpoint types. In particular, there are known bugs in
        RDM support in the presence of congestion or packet loss (issue
        1621). RMA is not yet supported.
    -   [`fi_provider`(7)](fi_provider.7.html) lists requirements for
        all providers. The following limitations exist in the *usnic*
        provider:
        -   multicast operations are not supported on *FI_EP_DGRAM* and
            *FI_EP_RDM* endpoints.
        -   *FI_EP_MSG* endpoints only support connect, accept, and
            getname CM operations.
        -   Passive endpoints only support listen, setname, and getname
            CM operations.
        -   *FI_EP_DGRAM* endpoints support `fi_sendmsg()` and
            `fi_recvmsg()`, but some flags are ignored. `fi_sendmsg()`
            supports `FI_INJECT` and `FI_COMPLETION`. `fi_recvmsg()`
            supports `FI_MORE`.
        -   Address vectors only support `FI_AV_MAP`.
        -   No counters are supported.
        -   The tag matching interface is not supported.
        -   *FI_MSG_PREFIX* is only supported on *FI_EP_DGRAM* and usage
            is limited to releases 1.1 and beyond.
        -   fi_control with FI_GETWAIT may only be used on CQs that have
            been bound to an endpoint. If fi_control is used on an
            unbound CQ, it will return -FI_EOPBADSTATE.
        -   There is limited support for data returned as part of an
            erroneous asynchronous operation. EQs will return error data
            for CM operations, CQs do not support returning error data.
        -   As of 1.5, usNIC supports fi_mr_regv, and fi_mr_regattr.
            Support is limited to a single iov.
        -   Atomic operations are not supported.
    -   Resource management is not supported. The application is
        responsible for resource protection.
    -   The usnic libfabric provider supports extensions that provide
        information and functionality beyond the standard libfabric
        interface. See the "USNIC EXTENSIONS" section, below.

# USNIC EXTENSIONS

The usnic libfabric provider exports extensions for additional VIC,
usNIC, and Ethernet capabilities not provided by the standard libfabric
interface.

These extensions are available via the "fi_ext_usnic.h" header file.

## Fabric Extension: getinfo

Version 2 of the "fabric getinfo" extension was introduced in Libfabric
release v1.3.0 and can be used to retrieve IP and SR-IOV information
about a usNIC device obtained from the
[`fi_getinfo`(3)](fi_getinfo.3.html) function.

The "fabric getinfo" extension is obtained by calling `fi_open_ops` and
requesting `FI_USNIC_FABRIC_OPS_1` to get the usNIC fabric extension
operations. The `getinfo` function accepts a version parameter that can
be used to select different versions of the extension. The information
returned by the "fabric getinfo" extension is accessible through a
`fi_usnic_info` struct that uses a version tagged union. The accessed
union member must correspond with the requested version. It is
recommended that applications explicitly request a version rather than
using the header provided `FI_EXT_USNIC_INFO_VERSION`. Although there is
a version 1 of the extension, its use is discouraged, and it may not be
available in future releases.

### Compatibility issues

The addition of version 2 of the extension caused an alignment issue
that could lead to invalid data in the v1 portion of the structure. This
means that the alignment difference manifests when an application using
v1 of the extension is compiled with Libfabric v1.1.x or v1.2.x, but
then runs with Libfabric.so that is v1.3.x or higher (and vice versa).

The v1.4.0 release of Libfabric introduced a padding field to explicitly
maintain compatibility with the v1.3.0 release. If the issue is
encountered, then it is recommended that you upgrade to a release
containing version 2 of the extension, or recompile with a patched
version of an older release.

``` c
#include <rdma/fi_ext_usnic.h>

struct fi_usnic_info {
    uint32_t ui_version;
    uint8_t ui_pad0[4];
    union {
        struct fi_usnic_info_v1 v1;
        struct fi_usnic_info_v2 v2;
    } ui;
} __attribute__((packed));

int getinfo(uint32_t version, struct fid_fabric *fabric,
        struct fi_usnic_info *info);
```

*version*
:   Version of getinfo to be used

*fabric*
:   Fabric descriptor

*info*
:   Upon successful return, this parameter will contain information
    about the fabric.

-   Version 2

``` c
struct fi_usnic_cap {
    const char *uc_capability;
    int uc_present;
} __attribute__((packed));

struct fi_usnic_info_v2 {
    uint32_t        ui_link_speed;
    uint32_t        ui_netmask_be;
    char            ui_ifname[IFNAMSIZ];
    unsigned        ui_num_vf;
    unsigned        ui_qp_per_vf;
    unsigned        ui_cq_per_vf;

    char            ui_devname[FI_EXT_USNIC_MAX_DEVNAME];
    uint8_t         ui_mac_addr[6];

    uint8_t         ui_pad0[2];

    uint32_t        ui_ipaddr_be;
    uint32_t        ui_prefixlen;
    uint32_t        ui_mtu;
    uint8_t         ui_link_up;

    uint8_t         ui_pad1[3];

    uint32_t        ui_vendor_id;
    uint32_t        ui_vendor_part_id;
    uint32_t        ui_device_id;
    char            ui_firmware[64];

    unsigned        ui_intr_per_vf;
    unsigned        ui_max_cq;
    unsigned        ui_max_qp;

    unsigned        ui_max_cqe;
    unsigned        ui_max_send_credits;
    unsigned        ui_max_recv_credits;

    const char      *ui_nicname;
    const char      *ui_pid;

    struct fi_usnic_cap **ui_caps;
} __attribute__((packed));
```

-   Version 1

``` c
struct fi_usnic_info_v1 {
    uint32_t ui_link_speed;
    uint32_t ui_netmask_be;
    char ui_ifname[IFNAMSIZ];

    uint32_t ui_num_vf;
    uint32_t ui_qp_per_vf;
    uint32_t ui_cq_per_vf;
} __attribute__((packed));
```

Version 1 of the "fabric getinfo" extension can be used by explicitly
requesting it in the call to `getinfo` and accessing the `v1` portion of
the `fi_usnic_info.ui` union. Use of version 1 is not recommended and it
may be removed from future releases.

The following is an example of how to utilize version 2 of the usnic
"fabric getinfo" extension.

``` c
#include <stdio.h>
#include <rdma/fabric.h>

/* The usNIC extensions are all in the
   rdma/fi_ext_usnic.h header */
#include <rdma/fi_ext_usnic.h>

int main(int argc, char *argv[]) {
    struct fi_info *info;
    struct fi_info *info_list;
    struct fi_info hints = {0};
    struct fi_ep_attr ep_attr = {0};
    struct fi_fabric_attr fabric_attr = {0};

    fabric_attr.prov_name = "usnic";
    ep_attr.type = FI_EP_DGRAM;

    hints.caps = FI_MSG;
    hints.mode = FI_LOCAL_MR | FI_MSG_PREFIX;
    hints.addr_format = FI_SOCKADDR;
    hints.ep_attr = &ep_attr;
    hints.fabric_attr = &fabric_attr;

    /* Find all usnic providers */
    fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, &hints, &info_list);

    for (info = info_list; NULL != info; info = info->next) {
        /* Open the fabric on the interface */
        struct fid_fabric *fabric;
        fi_fabric(info->fabric_attr, &fabric, NULL);

        /* Pass FI_USNIC_FABRIC_OPS_1 to get usnic ops
           on the fabric */
        struct fi_usnic_ops_fabric *usnic_fabric_ops;
        fi_open_ops(&fabric->fid, FI_USNIC_FABRIC_OPS_1, 0,
                (void **) &usnic_fabric_ops, NULL);

        /* Now use the returned usnic ops structure to call
           usnic extensions.  The following extension queries
           some IP and SR-IOV characteristics about the
           usNIC device. */
        struct fi_usnic_info usnic_info;

        /* Explicitly request version 2. */
        usnic_fabric_ops->getinfo(2, fabric, &usnic_info);

        printf("Fabric interface %s is %s:\n"
               "\tNetmask:  0x%08x\n\tLink speed: %d\n"
               "\tSR-IOV VFs: %d\n\tQPs per SR-IOV VF: %d\n"
               "\tCQs per SR-IOV VF: %d\n",
               info->fabric_attr->name,
               usnic_info.ui.v2.ui_ifname,
               usnic_info.ui.v2.ui_netmask_be,
               usnic_info.ui.v2.ui_link_speed,
               usnic_info.ui.v2.ui_num_vf,
               usnic_info.ui.v2.ui_qp_per_vf,
               usnic_info.ui.v2.ui_cq_per_vf);

        fi_close(&fabric->fid);
    }

    fi_freeinfo(info_list);
    return 0;
}
```

## Adress Vector Extension: get_distance

The "address vector get_distance" extension was introduced in Libfabric
release v1.0.0 and can be used to retrieve the network distance of an
address.

The "get_distance" extension is obtained by calling `fi_open_ops` and
requesting `FI_USNIC_AV_OPS_1` to get the usNIC address vector extension
operations.

``` c
int get_distance(struct fid_av *av, void *addr, int *metric);
```

*av*
:   Address vector

*addr*
:   Destination address

*metric*
:   On output this will contain `-1` if the destination host is
    unreachable, `0` is the destination host is locally connected, and
    `1` otherwise.

See fi_ext_usnic.h for more details.

# VERSION DIFFERENCES

## New naming convention for fabric/domain starting with libfabric v1.4

The release of libfabric v1.4 introduced a new naming convention for
fabric and domain. However the usNIC provider remains backward
compatible with applications supporting the old scheme and decides which
one to use based on the version passed to `fi_getinfo`:

-   When `FI_VERSION(1,4)` or higher is used:
    -   fabric name is the network address with the CIDR notation (i.e.,
        `a.b.c.d/e`)
    -   domain name is the usNIC Linux interface name (i.e., `usnic_X`)
-   When a lower version number is used, like `FI_VERSION(1, 3)`, it
    follows the same behavior the usNIC provider exhibited in libfabric
    \<= v1.3:
    -   fabric name is the usNIC Linux interface name (i.e., `usnic_X`)
    -   domain name is `NULL`

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_open_ops`(3)](fi_open_ops.3.html),
[`fi_provider`(7)](fi_provider.7.html),

{% include JB/setup %}

# NAME

fi_verbs - The Verbs Fabric Provider

# OVERVIEW

The verbs provider enables applications using OFI to be run over any
verbs hardware (Infiniband, iWarp, etc). It uses the Linux Verbs API for
network transport and provides a translation of OFI calls to appropriate
verbs API calls. It uses librdmacm for communication management and
libibverbs for other control and data transfer operations.

# REQUIREMENTS

To successfully build and install verbs provider as part of libfabric,
it needs the following packages: \* libibverbs \* libibverbs-devel \*
librdmacm \* librdmacm-devel

You may also want to look into any OS specific instructions for enabling
RDMA. e.g. RHEL has instructions on their documentation for enabling
RDMA.

The IPoIB interface should be configured with a valid IP address. This
is a requirement from librdmacm.

# SUPPORTED FEATURES

The verbs provider supports a subset of OFI features.

### Endpoint types

FI_EP_MSG, FI_EP_DGRAM (beta), FI_EP_RDM.

FI_EP_RDM is supported via OFI RxM and RxD utility providers which are
layered on top of verbs. To the app, the provider name string would
appear as "verbs;ofi_rxm" or "verbs;ofi_rxd". Please refer the man pages
for RxM (fi_rxm.7) and RxD (fi_rxd.7) to know about the capabilities and
limitations for the FI_EP_RDM endpoint.

### Endpoint capabilities and features

#### MSG endpoints

FI_MSG, FI_RMA, FI_ATOMIC and shared receive contexts.

#### DGRAM endpoints

FI_MSG

### Modes

Verbs provider requires applications to support the following modes:

#### FI_EP_MSG endpoint type

-   FI_MR_LOCAL mr mode.

-   FI_RX_CQ_DATA for applications that want to use RMA. Applications
    must take responsibility of posting receives for any incoming CQ
    data.

### Addressing Formats

Supported addressing formats include \* MSG and RDM (internal -
deprecated) EPs support: FI_SOCKADDR, FI_SOCKADDR_IN, FI_SOCKADDR_IN6,
FI_SOCKADDR_IB \* DGRAM supports: FI_ADDR_IB_UD

### Progress

Verbs provider supports FI_PROGRESS_AUTO: Asynchronous operations make
forward progress automatically.

### Operation flags

Verbs provider supports FI_INJECT, FI_COMPLETION, FI_REMOTE_CQ_DATA,
FI_TRANSMIT_COMPLETE.

### Msg Ordering

Verbs provider support the following message ordering:

-   Read after Read

-   Read after Write

-   Read after Send

-   Write after Write

-   Write after Send

-   Send after Write

-   Send after Send

### Fork

Verbs provider does not provide fork safety by default. Fork safety can
be requested by setting IBV_FORK_SAFE, or RDMAV_FORK_SAFE. If the system
configuration supports the use of huge pages, it is recommended to set
RDMAV_HUGEPAGES_SAFE. See ibv_fork_init(3) for additional details.

### Memory Registration Cache

The verbs provider uses the common memory registration cache
functionality that's part of libfabric utility code. This speeds up
memory registration calls from applications by caching registrations of
frequently used memory regions. Please refer to fi_mr(3): Memory
Registration Cache section for more details.

# LIMITATIONS

### Memory Regions

Only FI_MR_BASIC mode is supported. Adding regions via s/g list is
supported only up to a s/g list size of 1. No support for binding memory
regions to a counter.

### Wait objects

Only FI_WAIT_FD wait object is supported only for FI_EP_MSG endpoint
type. Wait sets are not supported.

### Resource Management

Application has to make sure CQs are not overrun as this cannot be
detected by the provider.

### Unsupported Features

The following features are not supported in verbs provider:

#### Unsupported Capabilities

FI_NAMED_RX_CTX, FI_DIRECTED_RECV, FI_TRIGGER, FI_RMA_EVENT

#### Other unsupported features

Scalable endpoints, FABRIC_DIRECT

#### Unsupported features specific to MSG endpoints

-   Counters, FI_SOURCE, FI_TAGGED, FI_PEEK, FI_CLAIM, fi_cancel,
    fi_ep_alias, shared TX context, cq_readfrom operations.
-   Completion flags are not reported if a request posted to an endpoint
    completes in error.

### Fork

The support for fork in the provider has the following limitations:

-   Fabric resources like endpoint, CQ, EQ, etc. should not be used in
    the forked process.
-   The memory registered using fi_mr_reg has to be page aligned since
    ibv_reg_mr marks the entire page that a memory region belongs to as
    not to be re-mapped when the process is forked (MADV_DONTFORK).

### XRC Transport

The XRC transport is intended to be used when layered with the RXM
provider and requires the use of shared receive contexts. See
[`fi_rxm`(7)](fi_rxm.7.html). To enable XRC, the following environment
variables must usually be set: FI_VERBS_PREFER_XRC and
FI_OFI_RXM_USE_SRX.

### Atomics

Mellanox hardware has limited support for atomics on little-endian
machines as the result buffer will be delivered back to the caller in
big-endian, requiring the caller to handle the conversion back into
little-endian for use. This limitation is exposed in OFI as well which
uses the verbs atomic support directly. Use of atomics on Mellanox cards
on little-endian machines is allowed but users should make note of this
verbs limitation and do any conversion necessary.

# RUNTIME PARAMETERS

The verbs provider checks for the following environment variables.

### Common variables:

*FI_VERBS_TX_SIZE*
:   Default maximum tx context size (default: 384)

*FI_VERBS_RX_SIZE*
:   Default maximum rx context size (default: 384)

*FI_VERBS_TX_IOV_LIMIT*
:   Default maximum tx iov_limit (default: 4). Note: RDM (internal -
    deprecated) EP type supports only 1

*FI_VERBS_RX_IOV_LIMIT*
:   Default maximum rx iov_limit (default: 4). Note: RDM (internal -
    deprecated) EP type supports only 1

*FI_VERBS_INLINE_SIZE*
:   Maximum inline size for the verbs device. Actual inline size
    returned may be different depending on device capability. This value
    will be returned by fi_info as the inject size for the application
    to use. Set to 0 for the maximum device inline size to be used.
    (default: 256).

*FI_VERBS_MIN_RNR_TIMER*
:   Set min_rnr_timer QP attribute (0 - 31) (default: 12)

*FI_VERBS_CQREAD_BUNCH_SIZE*
:   The number of entries to be read from the verbs completion queue at
    a time (default: 8).

*FI_VERBS_PREFER_XRC*
:   Prioritize XRC transport fi_info before RC transport fi_info
    (default: 0, RC fi_info will be before XRC fi_info)

*FI_VERBS_GID_IDX*
:   The GID index to use (default: 0)

*FI_VERBS_DEVICE_NAME*
:   Specify a specific verbs device to use by name

*FI_VERBS_USE_DMABUF*
:   If supported, try to use ibv_reg_dmabuf_mr first to register
    dmabuf-based buffers. Set it to "no" to always use ibv_reg_mr which
    can be helpful for testing the functionality of the dmabuf_peer_mem
    hooking provider and the corresponding kernel driver. (default: yes)

### Variables specific to MSG endpoints

*FI_VERBS_IFACE*
:   The prefix or the full name of the network interface associated with
    the verbs device (default: ib)

### Variables specific to DGRAM endpoints

*FI_VERBS_DGRAM_USE_NAME_SERVER*
:   The option that enables/disables OFI Name Server thread. The NS
    thread is used to resolve IP-addresses to provider specific
    addresses (default: 1, if "OMPI_COMM_WORLD_RANK" and "PMI_RANK"
    environment variables aren't defined)

*FI_VERBS_NAME_SERVER_PORT*
:   The port on which Name Server thread listens incoming connections
    and requests (default: 5678)

### Environment variables notes

The fi_info utility would give the up-to-date information on environment
variables: fi_info -p verbs -e

# Troubleshooting / Known issues

### fi_getinfo returns -FI_ENODATA

-   Set FI_LOG_LEVEL=info or FI_LOG_LEVEL=debug (if debug build of
    libfabric is available) and check if there any errors because of
    incorrect input parameters to fi_getinfo.
-   Check if "fi_info -p verbs" is successful. If that fails the
    following checklist may help in ensuring that the RDMA verbs stack
    is functional:
    -   If libfabric was compiled, check if verbs provider was built.
        Building verbs provider would be skipped if its dependencies
        (listed in requirements) aren't available on the system.
    -   Verify verbs device is functional:
        -   Does ibv_rc_pingpong (available in libibverbs) test work?
            -   Does ibv_devinfo (available in libibverbs) show the
                device with PORT_ACTIVE status?
                -   Check if Subnet Manager (SM) is running on the
                    switch or on one of the nodes in the cluster.
                -   Is the cable connected?
    -   Verify librdmacm is functional:
        -   Does ucmatose test (available in librdmacm) work?
        -   Is the IPoIB interface (e.g. ib0) up and configured with a
            valid IP address?

### Other issues

When running an app over verbs provider with Valgrind, there may be
reports of memory leak in functions from dependent libraries
(e.g. libibverbs, librdmacm). These leaks are safe to ignore.

The provider protects CQ overruns that may happen because more TX
operations were posted to endpoints than CQ size. On the receive side,
it isn't expected to overrun the CQ. In case it happens the application
developer should take care not to post excess receives without draining
the CQ. CQ overruns can make the MSG endpoints unusable.

# SEE ALSO

[`fabric`(7)](fabric.7.html), [`fi_provider`(7)](fi_provider.7.html),

{% include JB/setup %}

# NAME

fi_info - Simple utility to query for fabric interfaces

# SYNOPSIS

     fi_info [OPTIONS]

# DESCRIPTION

The fi_info utility can be used to query for available fabric
interfaces. The utility supports filtering based on a number of options
such as endpoint type, provider name, or supported modes. Additionally,
fi_info can also be used to discover the environment variables that can
be used to tune provider specific parameters. If no filters are
specified, then all available fabric interfaces for all providers and
endpoint types will be returned.

# OPTIONS

## Filtering

*-n, --node=\<NAME\>*
:   Node name or address used to filter interfaces. Only interfaces
    which can reach the given node or address will respond.

*-P, --port=\<PORT\>*
:   Port number used to filter interfaces.

*-c, --caps=\<CAP1\|CAP2\>..*
:   Pipe separated list of capabilities used to filter interfaces. Only
    interfaces supporting all of the given capabilities will respond.
    For more information on capabilities, see fi_getinfo(3).

*-m, --mode=\<MOD1\|MOD2\>..*
:   Pipe separated list of modes used to filter interfaces. Only
    interfaces supporting all of the given modes will respond. For more
    information on, modes see fi_getinfo(3).

*-t, --ep_type=\<EPTYPE\>*
:   Specifies the type of fabric interface communication desired. For
    example, specifying FI_EP_DGRAM would return only interfaces which
    support unreliable datagram. For more information on endpoint types,
    see fi_endpoint(3).

*-a, --addr_format=\<FMT\>*
:   Filter fabric interfaces by their address format. For example,
    specifying FI_SOCKADDR_IN would return only interfaces which use
    sockaddr_in structures for addressing. For more information on
    address formats, see fi_getinfo(3).

*-p, --provider=\<PROV\>*
:   Filter fabric interfaces by the provider implementation. For a list
    of providers, see the `--list` option.

*-d, --domain=\<DOMAIN\>*
:   Filter interfaces to only those with the given domain name.

*-f, --fabric=\<FABRIC\>*
:   Filter interfaces to only those with the given fabric name.

## Discovery

*-e, --env*
:   List libfabric related environment variables which can be used to
    enable extra configuration or tuning.

*-g \[filter\]*
:   Same as -e option, with output limited to environment variables
    containing filter as a substring.

*-l, --list*
:   List available libfabric providers.

*-v, --verbose*
:   By default, fi_info will display a summary of each of the interfaces
    discovered. If the verbose option is enabled, then all of the
    contents of the fi_info structure are displayed. For more
    information on the data contained in the fi_info structure, see
    fi_getinfo(3).

*--version*
:   Display versioning information.

# USAGE EXAMPLES

    $ fi_info -p verbs -t FI_EP_DGRAM

This will respond with all fabric interfaces that use endpoint type
FI_EP_DGRAM with the verbs provider.

    fi_info -c 'FI_MSG|FI_READ|FI_RMA'

This will respond with all fabric interfaces that can support
FI_MSG\|FI_READ\|FI_RMA capabilities.

# OUTPUT

By default fi_info will output a summary of the fabric interfaces
discovered:

    $ ./fi_info -p verbs -t FI_EP_DGRAM
    provider: verbs
        fabric: IB-0xfe80000000000000
        domain: mlx5_0-dgram
        version: 116.0
        type: FI_EP_DGRAM
        protocol: FI_PROTO_IB_UD

    $ ./fi_info -p tcp
    provider: tcp
        fabric: 192.168.7.0/24
        domain: eth0
        version: 116.0
        type: FI_EP_MSG
        protocol: FI_PROTO_SOCK_TCP

To see the full fi_info structure, specify the `-v` option.

    fi_info:
        caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_RMA_EVENT, FI_SOURCE, FI_DIRECTED_RECV ]
        mode: [  ]
        addr_format: FI_ADDR_IB_UD
        src_addrlen: 32
        dest_addrlen: 0
        src_addr: fi_addr_ib_ud://:::0/0/0/0
        dest_addr: (null)
        handle: (nil)
        fi_tx_attr:
            caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_SEND ]
            mode: [  ]
            op_flags: [  ]
            msg_order: [ FI_ORDER_RAR, FI_ORDER_RAW, FI_ORDER_RAS, FI_ORDER_WAW, FI_ORDER_WAS, FI_ORDER_SAW, FI_ORDER_SAS, FI_ORDER_RMA_RAR, FI_ORDER_RMA_RAW, FI_ORDER_RMA_WAW, FI_ORDER_ATOMIC_RAR, FI_ORDER_ATOMIC_RAW, FI_ORDER_ATOMIC_WAR, FI_ORDER_ATOMIC_WAW ]
            inject_size: 3840
            size: 1024
            iov_limit: 4
            rma_iov_limit: 4
            tclass: 0x0
        fi_rx_attr:
            caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_RECV, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_RMA_EVENT, FI_SOURCE, FI_DIRECTED_RECV ]
            mode: [  ]
            op_flags: [  ]
            msg_order: [ FI_ORDER_RAR, FI_ORDER_RAW, FI_ORDER_RAS, FI_ORDER_WAW, FI_ORDER_WAS, FI_ORDER_SAW, FI_ORDER_SAS, FI_ORDER_RMA_RAR, FI_ORDER_RMA_RAW, FI_ORDER_RMA_WAW, FI_ORDER_ATOMIC_RAR, FI_ORDER_ATOMIC_RAW, FI_ORDER_ATOMIC_WAR, FI_ORDER_ATOMIC_WAW ]
            size: 1024
            iov_limit: 4
        fi_ep_attr:
            type: FI_EP_RDM
            protocol: FI_PROTO_RXD
            protocol_version: 1
            max_msg_size: 18446744073709551615
            msg_prefix_size: 0
            max_order_raw_size: 18446744073709551615
            max_order_war_size: 0
            max_order_waw_size: 18446744073709551615
            mem_tag_format: 0xaaaaaaaaaaaaaaaa
            tx_ctx_cnt: 1
            rx_ctx_cnt: 1
            auth_key_size: 0
        fi_domain_attr:
            domain: 0x0
            name: mlx5_0-dgram
            threading: FI_THREAD_SAFE
            progress: FI_PROGRESS_MANUAL
            resource_mgmt: FI_RM_ENABLED
            av_type: FI_AV_UNSPEC
            mr_mode: [  ]
            mr_key_size: 8
            cq_data_size: 8
            cq_cnt: 128
            ep_cnt: 128
            tx_ctx_cnt: 1
            rx_ctx_cnt: 1
            max_ep_tx_ctx: 1
            max_ep_rx_ctx: 1
            max_ep_stx_ctx: 0
            max_ep_srx_ctx: 0
            cntr_cnt: 0
            mr_iov_limit: 1
            caps: [  ]
            mode: [  ]
            auth_key_size: 0
            max_err_data: 0
            mr_cnt: 0
            tclass: 0x0
        fi_fabric_attr:
            name: IB-0xfe80000000000000
            prov_name: verbs;ofi_rxd
            prov_version: 116.0
            api_version: 1.16
        nic:
            fi_device_attr:
                name: mlx5_0
                device_id: 0x101b
                device_version: 0
                vendor_id: 0x02c9
                driver: (null)
                firmware: 20.33.1048
            fi_bus_attr:
                bus_type: FI_BUS_UNKNOWN
            fi_link_attr:
                address: (null)
                mtu: 4096
                speed: 0
                state: FI_LINK_UP
                network_type: InfiniBand

To see libfabric related environment variables `-e` option.

    $ ./fi_info -e
    # FI_LOG_INTERVAL: Integer
    # Delay in ms between rate limited log messages (default 2000)

    # FI_LOG_LEVEL: String
    # Specify logging level: warn, trace, info, debug (default: warn)

    # FI_LOG_PROV: String
    # Specify specific provider to log (default: all)

    # FI_PROVIDER: String
    # Only use specified provider (default: all available)

To see libfabric related environment variables with substring use `-g`
option.

    $ ./fi_info -g tcp
    # FI_OFI_RXM_DEF_TCP_WAIT_OBJ: String
    # ofi_rxm: See def_wait_obj for description.  If set, this overrides the def_wait_obj when running over the tcp provider.  See def_wait_obj for valid values. (default: UNSPEC, tcp provider will select).

    # FI_TCP_IFACE: String
    # tcp: Specify interface name

    # FI_TCP_PORT_LOW_RANGE: Integer
    # tcp: define port low range

    # FI_TCP_PORT_HIGH_RANGE: Integer
    # tcp: define port high range

    # FI_TCP_TX_SIZE: size_t
    # tcp: define default tx context size (default: 256)

    # FI_TCP_RX_SIZE: size_t
    # tcp: define default rx context size (default: 256)

    # FI_TCP_NODELAY: Boolean (0/1, on/off, true/false, yes/no)
    # tcp: overrides default TCP_NODELAY socket setting

    # FI_TCP_STAGING_SBUF_SIZE: Integer
    # tcp: size of buffer used to coalesce iovec's or send requests before posting to the kernel, set to 0 to disable

    # FI_TCP_PREFETCH_RBUF_SIZE: Integer
    # tcp: size of buffer used to prefetch received data from the kernel, set to 0 to disable

    # FI_TCP_ZEROCOPY_SIZE: size_t
    # tcp: lower threshold where zero copy transfers will be used, if supported by the platform, set to -1 to disable (default: 18446744073709551615)

# SEE ALSO

[`fi_getinfo(3)`](fi_getinfo.3.html),
[`fi_endpoint(3)`](fi_endpoint.3.html)

{% include JB/setup %}

# NAME

fi_pingpong - Quick and simple pingpong test for libfabric

# SYNOPSIS

     fi_pingpong [OPTIONS]                      start server
     fi_pingpong [OPTIONS] <server address>     connect to server

# DESCRIPTION

fi_pingpong is a pingpong test for the core feature of the libfabric
library: transmitting data between two processes. fi_pingpong also
displays aggregated statistics after each test run, and can additionally
verify data integrity upon receipt.

By default, the datagram (FI_EP_DGRAM) endpoint is used for the test,
unless otherwise specified via `-e`.

# HOW TO RUN TESTS

Two copies of the program must be launched: first, one copy must be
launched as the server. Second, another copy is launched with the
address of the server.

As a client-server test, each have the following usage model:

## Start the server

    server$ fi_pingpong

## Start the client

    client$ fi_pingpong <server address>

# OPTIONS

The server and client must be able to communicate properly for the
fi_pingpong utility to function. If any of the `-e`, `-I`, `-S`, or `-p`
options are used, then they must be specified on the invocation for both
the server and the client process. If the `-d` option is specified on
the server, then the client will select the appropriate domain if no
hint is provided on the client side. If the `-d` option is specified on
the client, then it must also be specified on the server. If both the
server and client specify the `-d` option and the given domains cannot
communicate, then the application will fail.

## Control Messaging

*-B \<src_port\>*
:   The non-default source port number of the control socket. If this is
    not provided then the server will bind to port 47592 by default and
    the client will allow the port to be selected automatically.

*-P \<dest_port\>*
:   The non-default destination port number of the control socket. If
    this is not provided then the client will connect to 47592 by
    default. The server ignores this option.

## Fabric Filtering

*-p \<provider_name\>*
:   The name of the underlying fabric provider (e.g., sockets, psm3,
    usnic, etc.). If a provider is not specified via the -p switch, the
    test will pick one from the list of available providers (as returned
    by fi_getinfo(3)).

*-e \<endpoint\>*
:   The type of endpoint to be used for data messaging between the two
    processes. Supported values are dgram, rdm, and msg. For more
    information on endpoint types, see fi_endpoint(3).

*-d \<domain\>*
:   The name of the specific domain to be used.

*-s \<source address\>*
:   Address to corresponding domain. Required in multi-adapter
    environment.

## Test Options

*-I \<iter\>*
:   The number of iterations of the test will run.

*-S \<msg_size\>*
:   The specific size of the message in bytes the test will use or 'all'
    to run all the default sizes.

*-c*
:   Activate data integrity checks at the receiver (note: this will
    degrade performance).

## Utility

*-v*
:   Activate output debugging (warning: highly verbose)

*-h*
:   Displays help output for the pingpong test.

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

-   usNIC provider
-   1000 iterations
-   1024 bytes message size
-   server node as 192.168.0.123

## A longer test

### Server:

`server$ fi_pingpong -p usnic -I 10000 -S all`

### Client:

`client$ fi_pingpong -p usnic -I 10000 -S all 192.168.0.123`

# DEFAULTS

There is no default provider; if a provider is not specified via the
`-p` switch, the test will pick one from the list of available providers
(as returned by fi_getinfo(3)).

If no endpoint type is specified, 'dgram' is used.

The default tested sizes are: 64, 256, 1024, 4096, 65536, and 1048576.
The test will only test sizes that are within the selected endpoints
maximum message size boundary.

# OUTPUT

Each test generates data messages which are accounted for. Specifically,
the displayed statistics at the end are :

-   *bytes* : number of bytes per message sent
-   *#sent* : number of messages (ping) sent from the client to the
    server
-   *#ack* : number of replies (pong) of the server received by the
    client
-   *total* : amount of memory exchanged between the processes
-   *time* : duration of this single test
-   *MB/sec* : throughput computed from *total* and *time*
-   *usec/xfer* : average time for transferring a message outbound (ping
    or pong) in microseconds
-   *Mxfers/sec* : average amount of transfers of message outbound per
    second

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html) [`fabric`(7)](fabric.7.html),

{% include JB/setup %}

# NAME

fi_strerror - display libfabric error strings

# SYNOPSIS

    fi_strerror FI_ERROR_CODE

# DESCRIPTION

Display the error string for the given numeric `FI_ERROR_CODE`.
`FI_ERROR_CODE` may be a hexadecimal, octal, or decimal constant.
Although the [`fi_strerror`(3)](fi_errno.3.html) library function only
accepts positive error values, for convenience this utility accepts both
positive and negative error values.

This is primarily a convenience tool for developers.

# SEE ALSO

[`fabric`(7)](fabric.7.html) [`fi_errno`(3)](fi_errno.3.html)

{% include JB/setup %}

# NAME

Fabtests

# SYNOPSIS

Fabtests is a set of examples for fabric providers that demonstrates
various features of libfabric- high-performance fabric software library.

# OVERVIEW

Libfabric defines sets of interface that fabric providers can support.
The purpose of Fabtests examples is to demonstrate some of the major
features. The goal is to familiarize users with different
functionalities libfabric offers and how to use them. Although most
tests report performance numbers, they are designed to test
functionality and not performance. The exception are the benchmarks and
ubertest.

The tests are divided into the following categories. Except the unit
tests all of them are client-server tests. Not all providers will
support each test.

The test names try to indicate the type of functionality each test is
verifying. Although some tests work with any endpoint type, many are
restricted to verifying a single endpoint type. These tests typically
include the endpoint type as part of the test name, such as dgram, msg,
or rdm.

## Functional

These tests are a mix of very basic functionality tests that show major
features of libfabric.

*fi_av_xfer*
:   Tests communication for connectionless endpoints, as addresses are
    inserted and removed from the local address vector.

*fi_cm_data*
:   Verifies exchanging CM data as part of connecting endpoints.

*fi_cq_data*
:   Tranfers messages with CQ data.

*fi_dgram*
:   A basic datagram endpoint example.

*fi_inj_complete*
:   Sends messages using the FI_INJECT_COMPLETE operation flag.

*fi_mcast*
:   A simple multicast test.

*fi_msg*
:   A basic message endpoint example.

*fi_msg_epoll*
:   Transfers messages with completion queues configured to use file
    descriptors as wait objects. The file descriptors are retrieved by
    the program and used directly with the Linux epoll API.

*fi_msg_sockets*
:   Verifies that the address assigned to a passive endpoint can be
    transitioned to an active endpoint. This is required applications
    that need socket API semantics over RDMA implementations
    (e.g. rsockets).

*fi_multi_ep*
:   Performs data transfers over multiple endpoints in parallel.

*fi_multi_mr*
:   Issues RMA write operations to multiple memory regions, using
    completion counters of inbound writes as the notification mechanism.

*fi_rdm*
:   A basic RDM endpoint example.

*fi_rdm_atomic*
:   Test and verifies atomic operations over an RDM endpoint.

*fi_rdm_deferred_wq*
:   Test triggered operations and deferred work queue support.

*fi_rdm_multi_domain*
:   Performs data transfers over multiple endpoints, with each endpoint
    belonging to a different opened domain.

*fi_rdm_multi_recv*
:   Transfers multiple messages over an RDM endpoint that are received
    into a single buffer, posted using the FI_MULTI_RECV flag.

*fi_rdm_rma_event*
:   An RMA write example over an RDM endpoint that uses RMA events to
    notify the peer that the RMA transfer has completed.

*fi_rdm_rma_trigger*
:   A basic example of queuing an RMA write operation that is initiated
    upon the firing of a triggering completion. Works with RDM
    endpoints.

*fi_rdm_shared_av*
:   Spawns child processes to verify basic functionality of using a
    shared address vector with RDM endpoints.

*fi_rdm_stress*
:   A multi-process, multi-threaded stress test of RDM endpoints
    handling transfer errors.

*fi_rdm_tagged_peek*
:   Basic test of using the FI_PEEK operation flag with tagged messages.
    Works with RDM endpoints.

*fi_recv_cancel*
:   Tests canceling posted receives for tagged messages.

*fi_resmgmt_test*
:   Tests the resource management enabled feature. This verifies that
    the provider prevents applications from overrunning local and remote
    command queues and completion queues. This corresponds to setting
    the domain attribute resource_mgmt to FI_RM_ENABLED.

*fi_scalable_ep*
:   Performs data transfers over scalable endpoints, endpoints
    associated with multiple transmit and receive contexts.

*fi_shared_ctx*
:   Performs data transfers between multiple endpoints, where the
    endpoints share transmit and/or receive contexts.

*fi_unexpected_msg*
:   Tests the send and receive handling of unexpected tagged messages.

*fi_unmap_mem*
:   Tests data transfers where the transmit buffer is mmapped and
    unmapped between each transfer, but the virtual address of the
    transmit buffer tries to remain the same. This test is used to
    validate the correct behavior of memory registration caches.

*fi_flood*
:   The test performs a one-sided transfer by utilizing Bulk Memory
    Region (MR) registration and flooding the receiver with unexpected
    messages. This is followed by sequential MR registration transfers,
    which force the MR cache to evict the least recently used MRs before
    making new transfers. An optional sleep time can be enabled on the
    receiving side to allow the sender to get ahead of the receiver.

*fi_rdm_multi_client*
:   Tests a persistent server communicating with multiple clients, one
    at a time, in sequence.

## Benchmarks

The client and the server exchange messages in either a ping-pong
manner, for pingpong named tests, or transfer messages one-way, for bw
named tests. These tests can transfer various messages sizes, with
controls over which features are used by the test, and report
performance numbers. The tests are structured based on the benchmarks
provided by OSU MPI. They are not guaranteed to provide the best latency
or bandwidth performance numbers a given provider or system may achieve.

*fi_dgram_pingpong*
:   Latency test for datagram endpoints

*fi_msg_bw*
:   Message transfer bandwidth test for connected (MSG) endpoints.

*fi_msg_pingpong*
:   Message transfer latency test for connected (MSG) endpoints.

*fi_rdm_cntr_pingpong*
:   Message transfer latency test for reliable-datagram (RDM) endpoints
    that uses counters as the completion mechanism.

*fi_rdm_pingpong*
:   Message transfer latency test for reliable-datagram (RDM) endpoints.

*fi_rdm_tagged_bw*
:   Tagged message bandwidth test for reliable-datagram (RDM) endpoints.

*fi_rdm_tagged_pingpong*
:   Tagged message latency test for reliable-datagram (RDM) endpoints.

*fi_rma_bw*
:   An RMA read and write bandwidth test for reliable (MSG and RDM)
    endpoints.

*fi_rma_pingpong*
:   An RMA write and writedata latency test for reliable-datagram (RDM)
    endpoints.

## Unit

These are simple one-sided unit tests that validate basic behavior of
the API. Because these are single system tests that do not perform data
transfers their testing scope is limited.

*fi_av_test*
:   Verify address vector interfaces.

*fi_cntr_test*
:   Tests counter creation and destruction.

*fi_cq_test*
:   Tests completion queue creation and destruction.

*fi_dom_test*
:   Tests domain creation and destruction.

*fi_eq_test*
:   Tests event queue creation, destruction, and capabilities.

*fi_getinfo_test*
:   Tests provider response to fi_getinfo calls with varying hints.

*fi_mr_test*
:   Tests memory registration.

*fi_mr_cache_evict*
:   Tests provider MR cache eviction capabilities.

## Multinode

This test runs a series of tests over multiple formats and patterns to
help validate at scale. The patterns are an all to all, one to all, all
to one and a ring. The tests also run across multiple capabilities, such
as messages, rma, atomics, and tagged messages. Currently, there is no
option to run these capabilities and patterns independently, however the
test is short enough to be all run at once.

## Ubertest

This is a comprehensive latency, bandwidth, and functionality test that
can handle a variety of test configurations. The test is able to run a
large number of tests by iterating over a large number of test
variables. As a result, a full ubertest run can take a significant
amount of time. Because ubertest iterates over input variables, it
relies on a test configuration file for control, rather than extensive
command line options that are used by other fabtests. A configuration
file must be constructed for each provider. Example test configurations
are at test_configs.

*fi_ubertest*
:   This test takes a configure file as input. The file contains a list
    of variables and their values to iterate over. The test will run a
    set of latency, bandwidth, and functionality tests over a given
    provider. It will perform one execution for every possible
    combination of all variables. For example, if there are 8 test
    variables, with 6 having 2 possible values and 2 having 3 possible
    values, ubertest will execute 576 total iterations of each test.

# EFA provider specific tests

Beyond libfabric defined functionalities, EFA provider defines its
specific features/functionalities. These EFA provider specific fabtests
show users how to correctly use them.

*fi_efa_rnr_read_cq_error*
:   This test modifies the RNR retry count (rnr_retry) to 0 via
    fi_setopt, and then runs a simple program to test if the error cq
    entry (with error FI_ENORX) can be read by the application, if RNR
    happens.

*fi_efa_rnr_queue_resend*
:   This test modifies the RNR retry count (rnr_retry) to 0 via
    fi_setopt, and then tests RNR queue/re-send logic for different
    packet types. To run the test, one needs to use `-c` option to
    specify the category of packet types.

## Component tests

These stand-alone tests don't test libfabric functionalities. Instead,
they test some components that libfabric depend on. They are not called
by runfabtests.sh, either, and don't follow the fabtests coventions for
naming, config file, and command line options.

### Dmabuf RDMA tests

These tests check the functionality or performance of dmabuf based GPU
RDMA mechanism. They use oneAPI level-zero API to allocate buffer from
device memory, get dmabuf handle, and perform some device memory related
operations. Run with the *-h* option to see all available options for
each of the tests.

*xe_rdmabwe*
:   This Verbs test measures the bandwidth of RDMA operations. It runs
    in client-server mode. It has options to choose buffer location,
    test type (write, read, send/recv), device unit(s), NIC unit(s),
    message size, and the number of iterations per message size.

*fi_xe_rdmabw*
:   This test is similar to *xe_rdmabw*, but uses libfabric instead of
    Verbs.

*xe_mr_reg*
:   This Verbs test tries to register a buffer with the RDMA NIC.

*fi_xe_mr_reg*
:   This test is similar to *xe_mr_reg*, but uses libfabric instead of
    Verbs.

*xe_memcopy*
:   This test measures the performance of memory copy operations between
    buffers. It has options for buffer locations, as well as memory
    copying methods to use (memcpy, mmap + memcpy, copy with device
    command queue, etc).

### Other component tests

*sock_test*
:   This client-server test establishes socket connections and tests the
    functionality of select/poll/epoll with different set sizes.

## Config file options

The following keys and respective key values may be used in the config
file.

*prov_name*
:   Identify the provider(s) to test. E.g. udp, tcp, verbs,
    ofi_rxm;verbs, ofi_rxd;udp.

*test_type*
:   FT_TEST_LATENCY, FT_TEST_BANDWIDTH, FT_TEST_UNIT

*test_class*
:   FT_CAP_MSG, FT_CAP_TAGGED, FT_CAP_RMA, FT_CAP_ATOMIC

*class_function*
:   For FT_CAP_MSG and FT_CAP_TAGGED: FT_FUNC_SEND, FT_FUNC_SENDV,
    FT_FUNC_SENDMSG, FT_FUNC_INJECT, FT_FUNC_INJECTDATA,
    FT_FUNC_SENDDATA

For FT_CAP_RMA: FT_FUNC_WRITE, FT_FUNC_WRITEV, FT_FUNC_WRITEMSG,
FT_FUNC_WRITEDATA, FT_FUNC_INJECT_WRITE, FT_FUNC_INJECT_WRITEDATA,
FT_FUNC_READ, FT_FUNC_READV, FT_FUNC_READMSG

For FT_CAP_ATOMIC: FT_FUNC_ATOMIC, FT_FUNC_ATOMICV, FT_FUNC_ATOMICMSG,
FT_FUNC_INJECT_ATOMIC, FT_FUNC_FETCH_ATOMIC, FT_FUNC_FETCH_ATOMICV,
FT_FUNC_FETCH_ATOMICMSG, FT_FUNC_COMPARE_ATOMIC,
FT_FUNC_COMPARE_ATOMICV, FT_FUNC_COMPARE_ATOMICMSG

*constant_caps - values OR'ed together*
:   FI_RMA, FI_MSG, FI_SEND, FI_RECV, FI_READ, FI_WRITE, FI_REMOTE_READ,
    FI_REMOTE_WRITE, FI_TAGGED, FI_DIRECTED_RECV

*mode - values OR'ed together*
:   FI_CONTEXT, FI_CONTEXT2, FI_RX_CQ_DATA

*ep_type*
:   FI_EP_MSG, FI_EP_DGRAM, FI_EP_RDM

*comp_type*
:   FT_COMP_QUEUE, FT_COMP_CNTR, FT_COMP_ALL

*av_type*
:   FI_AV_MAP, FI_AV_TABLE

*eq_wait_obj*
:   FI_WAIT_NONE, FI_WAIT_UNSPEC, FI_WAIT_FD, FI_WAIT_MUTEX_COND

*cq_wait_obj*
:   FI_WAIT_NONE, FI_WAIT_UNSPEC, FI_WAIT_FD, FI_WAIT_MUTEX_COND

*cntr_wait_obj*
:   FI_WAIT_NONE, FI_WAIT_UNSPEC, FI_WAIT_FD, FI_WAIT_MUTEX_COND

*threading*
:   FI_THREAD_UNSPEC, FI_THREAD_SAFE, FI_THREAD_DOMAIN,
    FI_THREAD_COMPLETION

*progress*
:   FI_PROGRESS_MANUAL, FI_PROGRESS_AUTO, FI_PROGRESS_UNSPEC

*mr_mode*
:   (Values OR'ed together) FI_MR_LOCAL, FI_MR_VIRT_ADDR,
    FI_MR_ALLOCATED, FI_MR_PROV_KEY

*op*
:   For FT_CAP_ATOMIC: FI_MIN, FI_MAX, FI_SUM, FI_PROD, FI_LOR, FI_LAND,
    FI_BOR, FI_BAND, FI_LXOR, FI_BXOR, FI_ATOMIC_READ, FI_ATOMIC_WRITE,
    FI_CSWAP, FI_CSWAP_NE, FI_CSWAP_LE, FI_CSWAP_LT, FI_CSWAP_GE,
    FI_CSWAP_GT, FI_MSWAP

*datatype*
:   For FT_CAP_ATOMIC: FI_INT8, FI_UINT8, FI_INT16, FI_UINT16, FI_INT32,
    FI_UINT32, FI_INT64, FI_UINT64, FI_FLOAT, FI_DOUBLE,
    FI_FLOAT_COMPLEX, FI_DOUBLE_COMPLEX, FI_LONG_DOUBLE,
    FI_LONG_DOUBLE_COMPLEX

*msg_flags - values OR'ed together*
:   For FT_FUNC\_\[SEND,WRITE,READ,ATOMIC\]MSG: FI_REMOTE_CQ_DATA,
    FI_COMPLETION

*rx_cq_bind_flags - values OR'ed together*
:   FI_SELECTIVE_COMPLETION

*tx_cq_bind_flags - values OR'ed together*
:   FI_SELECTIVE_COMPLETION

*rx_op_flags - values OR'ed together*
:   FI_COMPLETION

*tx_op_flags - values OR'ed together*
:   FI_COMPLETION

*test_flags - values OR'ed together*
:   FT_FLAG_QUICKTEST

# HOW TO RUN TESTS

(1) Fabtests requires that libfabric be installed on the system, and at
    least one provider be usable.

(2) Install fabtests on the system. By default all the test executables
    are installed in /usr/bin directory unless specified otherwise.

(3) All the client-server tests have the following usage model:

    fi\_`<testname>`{=html} [OPTIONS](#options) start server
    fi\_`<testname>`{=html} `<host>`{=html} connect to server

# COMMAND LINE OPTIONS

Tests share command line options where appropriate. The following
command line options are available for one or more test. To see which
options apply for a given test, you can use the '-h' help option to see
the list available for that test.

*-h*
:   Displays help output for the test.

*-f `<fabric>`{=html}*
:   Restrict test to the specified fabric name.

*-d `<domain>`{=html}*
:   Restrict test to the specified domain name.

*-p `<provider>`{=html}*
:   Restrict test to the specified provider name.

*-e `<ep_type>`{=html}*
:   Use the specified endpoint type for the test. Valid options are msg,
    dgram, and rdm. The default endpoint type is rdm.

*-D `<device_name>`{=html}*
:   Allocate data buffers on the specified device, rather than in host
    memory. Valid options are ze, cuda and synapseai.

\*-a
:   The name of a shared address vector. This option only applies to
    tests that support shared address vectors.

*-B `<src_port>`{=html}*
:   Specifies the port number of the local endpoint, overriding the
    default.

*-C `<num_connections>`{=html}*
:   Specifies the number of simultaneous connections or communication
    endpoints to the server.

*-P `<dst_port>`{=html}*
:   Specifies the port number of the peer endpoint, overriding the
    default.

\*-s
:   Specifies the address of the local endpoint.

\*-F `<address_format>`{=html}
:   Specifies the address format.

\*-K
:   Fork a child process after initializing endpoint.

*-b\[=oob_port\]*
:   Enables out-of-band (via sockets) address exchange and test
    synchronization. A port for the out-of-band connection may be
    specified as part of this option to override the default. When
    specified, the input src_addr and dst_addr values are relative to
    the OOB socket connection, unless the -O option is also specified.

*-E\[=oob_port\]*
:   Enables out-of-band (via sockets) address exchange only. A port for
    the out-of-band connection may be specified as part of this option
    to override the default. Cannot be used together with the '-b'
    option. When specified, the input src_addr and dst_addr values are
    relative to the OOB socket connection, unless the -O option is also
    specified.

*-U*
:   Run fabtests with FI_DELIVERY_COMPLETE.

*-I `<number>`{=html}*
:   Number of data transfer iterations.

*-Q*
:   Associated any EQ with the domain, rather than directly with the EP.

*-w `<number>`{=html}*
:   Number of warm-up data transfer iterations.

*-S `<size>`{=html}*
:   Data transfer size or 'all' for a full range of sizes. By default a
    select number of sizes will be tested.

*-l*
:   If specified, the starting address of transmit and receive buffers
    will be aligned along a page boundary.

*-m*
:   Use machine readable output. This is useful for post-processing the
    test output with scripts.

*-t `<comp_type>`{=html}*
:   Specify the type of completion mechanism to use. Valid values are
    queue and counter. The default is to use completion queues.

*-c `<comp_method>`{=html}*
:   Indicate the type of processing to use checking for completed
    operations. Valid values are spin, sread, and fd. The default is to
    busy wait (spin) until the desired operation has completed. The
    sread option indicates that the application will invoke a blocking
    read call in libfabric, such as fi_cq_sread. Fd indicates that the
    application will retrieve the native operating system wait object
    (file descriptor) and use either poll() or select() to block until
    the fd has been signaled, prior to checking for completions.

*-o `<op>`{=html}*
:   For RMA based tests, specify the type of RMA operation to perform.
    Valid values are read, write, and writedata. Write operations are
    the default. For message based, tests, specify whether msg (default)
    or tagged transfers will be used.

*-M `<mcast_addr>`{=html}*
:   For multicast tests, specifies the address of the multicast group to
    join.

*-u `<test_config_file>`{=html}*
:   Specify the input file to use for test control. This is specified at
    the client for fi_ubertest and fi_rdm_stress and controls the
    behavior of the testing.

*-v*
:   Add data verification check to data transfers.

*-O `<addr>`{=html}*
:   Specify the out of band address to use, mainly useful if the address
    is not an IP address. If given, the src_addr and dst_addr address
    parameters will be passed through to the libfabric provider for
    interpretation.

# USAGE EXAMPLES

## A simple example

    run server: <test_name> -p <provider_name> -s <source_addr>
        e.g.    fi_msg_rma -p sockets -s 192.168.0.123
    run client: <test_name> <server_addr> -p <provider_name>
        e.g.    fi_msg_rma 192.168.0.123 -p sockets

## An example with various options

    run server: fi_rdm_atomic -p psm3 -s 192.168.0.123 -I 1000 -S 1024
    run client: fi_rdm_atomic 192.168.0.123 -p psm3 -I 1000 -S 1024

This will run "fi_rdm_atomic" for all atomic operations with

    - PSM3 provider
    - 1000 iterations
    - 1024 bytes message size
    - server node as 123.168.0.123

## Run multinode tests

    Server and clients are invoked with the same command:
        fi_multinode -n <number of processes> -s <server_addr> -C <mode>

    A process on the server must be started before any of the clients can be started
    succesfully. -C lists the mode that the tests will run in. Currently the options are

for rma and msg. If not provided, the test will default to msg.

## Run fi_rdm_stress

run server: fi_rdm_stress run client: fi_rdm_stress -u
fabtests/test_configs/rdm_stress/stress.json 127.0.0.1

## Run fi_ubertest

    run server: fi_ubertest
    run client: fi_ubertest -u fabtests/test_configs/tcp/all.test 127.0.0.1

This will run "fi_ubertest" with

    - tcp provider
    - configurations defined in fabtests/test_configs/tcp/all.test
    - server running on the same node

Usable config files are provided in
fabtests/test_configs/`<provider_name>`{=html}.

For more usage options: fi_ubertest -h

## Run the whole fabtests suite

A runscript scripts/runfabtests.sh is provided that runs all the tests
in fabtests and reports the number of pass/fail/notrun.

    Usage: runfabtests.sh [OPTIONS] [provider] [host] [client]

By default if none of the options are provided, it runs all the tests
using

    - sockets provider
    - 127.0.0.1 as both server and client address
    - for small number of optiond and iterations

Various options can be used to choose provider, subset tests to run,
level of verbosity etc.

    runfabtests.sh -vvv -t all psm3 192.168.0.123 192.168.0.124

This will run all fabtests using

    - psm3 provider
    - for different options and larger iterations
    - server node as 192.168.0.123 and client node as 192.168.0.124
    - print test output for all the tests

For detailed usage options: runfabtests.sh -h
