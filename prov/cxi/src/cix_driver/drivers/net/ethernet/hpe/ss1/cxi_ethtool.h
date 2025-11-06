/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2018-2020,2022,2024 Hewlett Packard Enterprise Development LP */

/* ethtool private flags for the Cassini Ethernet driver */
#define CXI_ETH_PF_INTERNAL_LOOPBACK         BIT(0)  /* internal loopback, tx to rx */
#define CXI_ETH_PF_EXTERNAL_LOOPBACK         BIT(1)  /* external loopback, rx to tx */
#define CXI_ETH_PF_LLR                       BIT(2)
#define CXI_ETH_PF_PRECODING                 BIT(3)
#define CXI_ETH_PF_IFG_HPC                   BIT(4)
#define CXI_ETH_PF_ROCE_OPT                  BIT(5)  /* RoCE Cassini Optimizations */
#define CXI_ETH_PF_IGNORE_ALIGN              BIT(6)  /* ignore align interrupt */
#define CXI_ETH_PF_DISABLE_PML_RECOVERY      BIT(7)  /* disable pml recovery */
#define CXI_ETH_PF_LINKTRAIN                 BIT(8)
#define CXI_ETH_PF_CK_SPEED                  BIT(9)
#define CXI_ETH_PF_REMOTE_FAULT_RECOVERY     BIT(10) /* enable pml recovery for remote faults */
#define CXI_ETH_PF_USE_UNSUPPORTED_CABLE     BIT(11) /* allow link up with unsupported cable */
#define CXI_ETH_PF_FEC_MONITOR               BIT(12) /* Turn FEC Monitor on/off */
#define CXI_ETH_PF_ALD                       BIT(13) /* control auto lane degrade */
#define CXI_ETH_PF_IGNORE_MEDIA_ERROR        BIT(14) /* ignore media error */
#define CXI_ETH_PF_USE_SUPPORTED_SS200_CABLE BIT(15) /* allow link up with supported ss200 cable */

#define PRIV_FLAGS_COUNT 16
#define LOOPBACK_MODE (CXI_ETH_PF_INTERNAL_LOOPBACK | CXI_ETH_PF_EXTERNAL_LOOPBACK)
