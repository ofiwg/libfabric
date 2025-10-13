Hi all,
This patch-set introduces minimal Ultra Ethernet driver infrastructure and
the lowest Ultra Ethernet sublayer - the Packet Delivery Sublayer (PDS),
which underpins the entire communication model of the Ultra Ethernet
Transport[1] (UET). Ultra Ethernet is a new RDMA transport designed for
efficient AI and HPC communication. The specifications are still being
ironed out and first public versions should be available soon. As there
isn't any UET hardware available yet, we introduce a software device model
which implements the lowest sublayer of the spec - PDS. The code is still
in early stages and experimental, aiming to start a discussion on the
kernel implementation and to show how we plan to organize it.

The PDS is responsible for establishing dynamic connections between Fabric
Endpoints (FEPs) called Packet Delivery Contexts (PDCs), packet
reliability, ordering, duplicate elimination and congestion management.
The PDS packet ordering is defined by a mode which can be one of:
 - Reliable, Ordered Delivery (ROD)
 - Reliable, Unordered Delivery (RUD)
 - Reliable, Unordered Delivery for Idempotent Operations (RUDI)
 - Unreliable, Unordered Delivery (UUD)

This set implements RUD mode of communication with Packet Sequence
Number (PSN) tracking, retransmits, idle timeouts, coalescing and selective
ACKs. It adds support for generating and processing Request, ACK, NACK and
Control packet types. Communication is done over UDP, so all Ultra Ethernet
headers are on top of UDP packets. Packets are tracked by Packet Sequence
Numbers (PSNs) uniquely assigned within a PDC, the PSN window sizes are
currently static.

In this RFC all of the code is under a single kernel module in
drivers/ultraeth/ and guarded by a new kconfig option CONFIG_ULTRAETH. The
plan is to have that split into core Ultra Ethernet module (ultraeth.ko)
which is responsible for managing the UET contexts, jobs and all other
common/generic UET configuration, and the software UET device model
(uecon.ko) which implements the UET protocols for communication in software
(e.g. the PDS will be a part of uecon) and is represented by a UDP tunnel
network device. Note that there are critical missing pieces that will be
present when we send the first version such as:
 - Ultra Ethernet specs will be publicly available
 - missing UET sublayers critical for communication
 - more complete user API
 - kernel UET device API
 - memory management
 - IPv6

The last patch is a hack which adds a custom character device used to test
communication and basic PDS functionality, for the first version of this set
we would rather extend and re-use some of the Infiniband infrastructure.

This set will also be used to better illustrate the UET code and concepts
for the "Networking For AI BoF"[2] at the upcoming Netdev 0x19 conference
in Zagreb, Croatia.

Thank you,
 Nik

[1] https://ultraethernet.org/
[2] https://netdevconf.info/0x19/sessions/bof/networking-for-ai-bof.html


Alex Badea (1):
  HACK: drivers: ultraeth: add char device

Nikolay Aleksandrov (12):
  drivers: ultraeth: add initial skeleton and kconfig option
  drivers: ultraeth: add context support
  drivers: ultraeth: add new genl family
  drivers: ultraeth: add job support
  drivers: ultraeth: add tunnel udp device support
  drivers: ultraeth: add initial PDS infrastructure
  drivers: ultraeth: add request and ack receive support
  drivers: ultraeth: add request transmit support
  drivers: ultraeth: add support for coalescing ack
  drivers: ultraeth: add sack support
  drivers: ultraeth: add nack support
  drivers: ultraeth: add initiator and target idle timeout support

 Documentation/netlink/specs/rt_link.yaml  |   14 +
 Documentation/netlink/specs/ultraeth.yaml |  218 ++++
 drivers/Kconfig                           |    2 +
 drivers/Makefile                          |    1 +
 drivers/ultraeth/Kconfig                  |   11 +
 drivers/ultraeth/Makefile                 |    4 +
 drivers/ultraeth/uecon.c                  |  324 ++++++
 drivers/ultraeth/uet_chardev.c            |  264 +++++
 drivers/ultraeth/uet_context.c            |  274 +++++
 drivers/ultraeth/uet_job.c                |  456 +++++++++
 drivers/ultraeth/uet_main.c               |   41 +
 drivers/ultraeth/uet_netlink.c            |  113 +++
 drivers/ultraeth/uet_netlink.h            |   29 +
 drivers/ultraeth/uet_pdc.c                | 1122 +++++++++++++++++++++
 drivers/ultraeth/uet_pds.c                |  481 +++++++++
 include/net/ultraeth/uecon.h              |   28 +
 include/net/ultraeth/uet_chardev.h        |   11 +
 include/net/ultraeth/uet_context.h        |   47 +
 include/net/ultraeth/uet_job.h            |   80 ++
 include/net/ultraeth/uet_pdc.h            |  170 ++++
 include/net/ultraeth/uet_pds.h            |  110 ++
 include/uapi/linux/if_link.h              |    8 +
 include/uapi/linux/ultraeth.h             |  536 ++++++++++
 include/uapi/linux/ultraeth_nl.h          |  116 +++
 24 files changed, 4460 insertions(+)
 create mode 100644 Documentation/netlink/specs/ultraeth.yaml
 create mode 100644 drivers/ultraeth/Kconfig
 create mode 100644 drivers/ultraeth/Makefile
 create mode 100644 drivers/ultraeth/uecon.c
 create mode 100644 drivers/ultraeth/uet_chardev.c
 create mode 100644 drivers/ultraeth/uet_context.c
 create mode 100644 drivers/ultraeth/uet_job.c
 create mode 100644 drivers/ultraeth/uet_main.c
 create mode 100644 drivers/ultraeth/uet_netlink.c
 create mode 100644 drivers/ultraeth/uet_netlink.h
 create mode 100644 drivers/ultraeth/uet_pdc.c
 create mode 100644 drivers/ultraeth/uet_pds.c
 create mode 100644 include/net/ultraeth/uecon.h
 create mode 100644 include/net/ultraeth/uet_chardev.h
 create mode 100644 include/net/ultraeth/uet_context.h
 create mode 100644 include/net/ultraeth/uet_job.h
 create mode 100644 include/net/ultraeth/uet_pdc.h
 create mode 100644 include/net/ultraeth/uet_pds.h
 create mode 100644 include/uapi/linux/ultraeth.h
 create mode 100644 include/uapi/linux/ultraeth_nl.h
 