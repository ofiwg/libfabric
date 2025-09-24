# Libfabric Examples

The examples in this directory provide standalone libfabric examples for
developers as starting points for writing OFI programs.  Each example includes
detailed comments that explain every OFI call and its purpose.

A single Makefile is provided for simplicity. Set the environment variable
`LIBFABRIC_PREFIX` to specify where to find the libfabric library and headers.

## Examples

### fi_example_rdm (rdm.c)

This example uses `FI_EP_RDM` (reliable, unconnected) with a simple server and
client to execute a one-directional send and receive.

#### Server
```bash
FI_PROVIDER=<prov> ./fi_example_rdm
```

#### Client
```bash
FI_PROVIDER=<prov> ./fi_example_rdm <server_addr>
```

> **Note:** The server and client must use the same provider.  The server
address provided to the client must match the provider/interface being used.
Specifying `FI_PROVIDER` is optional; libfabric will select the fastest
interface available.

### fi_example_msg (msg.c)

This example uses `FI_EP_MSG` (reliable, connected) with a simple server and
client to execute send and receive.

#### Server
```bash
FI_PROVIDER=<prov> ./fi_example_msg
```

#### Client
```bash
FI_PROVIDER=<prov> ./fi_example_msg <server_addr>
```

> **Note:** The server and client must use the same provider.  The server
address provided to the client must match the provider/interface being used.
Specifying `FI_PROVIDER` is optional; libfabric will select the fastest
interface available.

### fi_example_rdm_tagged (rdm_tagged.c)

This example uses `FI_EP_RDM` (reliable, unconnected) with a simple server and
client to showcase tagged messages.

#### Server
```bash
FI_PROVIDER=<prov> ./fi_example_rdm_tagged
```

#### Client
```bash
FI_PROVIDER=<prov> ./fi_example_rdm_tagged <server_addr>
```
### fi_example_rdm_rma (rdm_rma.c)

This example uses `FI_EP_RDM` (reliable, unconnected) with a simple server and
client to showcase remote memory access.

#### Server
```bash
FI_PROVIDER=<prov> ./fi_example_rdm_rma
```

#### Client
```bash
FI_PROVIDER=<prov> ./fi_example_rdm_rma <server_addr>

### fi_example_tcp_socket (tcp_socket.c)

This example uses tcp sockets with a simple server and client to showcase send and receive messages.

#### Server
```bash
./fi_example_tcp_socket -s <server_addr>
```

#### Client
```bash
./fi_example_tcp_socket -s <server_addr> -c
