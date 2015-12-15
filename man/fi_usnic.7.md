---
layout: page
title: fi_usnic(7)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

The usNIC Fabric Provider

# OVERVIEW

The *usnic* provider is designed to run over the Cisco VIC
(virtualized NIC) hardware on Cisco UCS servers.  It utilizes the
Cisco usNIC (userspace NIC) capabilities of the VIC to enable ultra
low latency and other offload capabilities on Ethernet networks.

# RELEASE NOTES

* The *usnic* libfabric provider requires the use of the "libnl"
  library.
    - There are two versions of libnl generally available: v1 and v3;
      the usnic provider can use either version.
    - If you are building libfabric/the usnic provider from source, you
      will need to have the libnl header files available (e.g., if you
      are installing libnl from RPM or other packaging system, install
      the "-devel" versions of the package).
    - If you have libnl (either v1 or v3) installed in a non-standard
      location (e.g., not in /usr/lib or /usr/lib64), you may need to
      tell libfabric's configure where to find libnl via the
      `--with-libnl=DIR` command line option (where DIR is the
      installation prefix of the libnl package).
* The most common way to use the libfabric usnic provider is via an
  MPI implementation that uses libfabric (and the usnic provider) as a
  lower layer transport.  MPI applications do not need to know
  anything about libfabric or usnic in this use case -- the MPI
  implementation hides all these details from the application.
* If you are writing applications directly to the libfabric API:
    - *FI_EP_DGRAM* endpoints are the best supported method of utilizing
      the usNIC interface.  Specifically, the *FI_EP_DGRAM* endpoint
      type has been extensively tested as the underlying layer for Open
      MPI's *usnic* BTL.
    - *FI_EP_MSG* and *FI_EP_RDM* endpoints are implemented, but are
      only lightly tested.  It is likely that there are still some bugs
      in these endpoint types.  RMA is not yet supported.
    - [`fi_provider`(7)](fi_provider.7.html) lists requirements for all
      providers.  The following limitations exist in the *usnic*
      provider:
        * multicast operations are not supported on *FI_EP_DGRAM* and
          *FI_EP_RDM* endpoints.
        * *FI_EP_MSG* endpoints only support connect, accept, and getname
          CM operations.
        * Passive endpoints only support listen, setname, and getname CM
          operations.
        * *FI_EP_DGRAM* endpoints support `fi_sendmsg()` and
          `fi_recvmsg()`, but some flags are ignored.  `fi_sendmsg()`
          supports `FI_INJECT` and `FI_COMPLETION`.  `fi_recvmsg()`
          supports `FI_MORE`.
        * Address vectors only support `FI_AV_MAP`.
        * No counters are supported.
        * The tag matching interface is not supported.
        * *FI_MSG_PREFIX* is only supported on *FI_EP_DGRAM* and usage
          is limited to releases 1.1 and beyond.
    - The usnic libfabric provider supports extensions that provide
      information and functionality beyond the standard libfabric
      interface.  See the "USNIC EXTENSIONS" section, below.

# USNIC EXTENSIONS

The usnic libfabric provider exports extensions for additional VIC,
usNIC, and Ethernet capabilities not provided by the standard
libfabric interface.

These extensions are available via the "fi_ext_usnic.h" header file.

The following is an example of how to utilize the usnic "fabric getinfo"
extension, which returns IP and SR-IOV information about a usNIC
interface obtained from the [`fi_getinfo`(3)](fi_getinfo.3.html)
function.

{% highlight c %}
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
        usnic_fabric_ops->getinfo(FI_EXT_USNIC_INFO_VERSION,
                                  fabric, &usnic_info);

        printf("Fabric interface %s is %s:\n"
               "\tNetmask:  0x%08x\n\tLink speed: %d\n"
               "\tSR-IOV VFs: %d\n\tQPs per SR-IOV VF: %d\n"
               "\tCQs per SR-IOV VF: %d\n",
               info->fabric_attr->name,
               usnic_info.ui.v1.ui_ifname,
               usnic_info.ui.v1.ui_netmask_be,
               usnic_info.ui.v1.ui_link_speed,
               usnic_info.ui.v1.ui_num_vf,
               usnic_info.ui.v1.ui_qp_per_vf,
               usnic_info.ui.v1.ui_cq_per_vf);

        fi_close(&fabric->fid);
    }

    fi_freeinfo(info_list);
    return 0;
}
{% endhighlight %}

Note that other usnic extensions are defined for other fabric objects.
The second argument to [`fi_open_ops`(3)](fi_open_ops.3.html) is used
to identify both the fid type and the extension family.  For example,
*FI_USNIC_AV_OPS_1* can be used in conjunction with an `fi_av` fid
to obtain usnic extensions for address vectors.

See fi_ext_usnic.h for more details.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_open_ops`(3)](fi_open_ops.3.html),
[`fi_provider`(7)](fi_provider.7.html),
