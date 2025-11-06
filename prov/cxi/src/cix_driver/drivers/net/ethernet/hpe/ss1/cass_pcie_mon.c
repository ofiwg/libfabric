// SPDX-License-Identifier: GPL-2.0
/* Copyright 2022 Hewlett Packard Enterprise Development LP */


/* Monitoring of vendor specific PCIe counters */

#include "cass_core.h"

#define NUM_LANES 16

/* Options for EVENT_COUNTER_CONTROL_REG register. */
#define COUNTERS_CLEAR_ONE  (0x01 << 0)
#define COUNTERS_CLEAR_ALL  (0x03 << 0)
#define COUNTERS_ENABLE_ALL (0x07 << 2)

struct pcie_cntrs_group {
	u8 first;
	u8 last;
	const char *names[];
};

static const struct pcie_cntrs_group group_0 = {
	0, 10,
	{ "RX Elastic Buffer Overflow Error",
	  "RX Elastic Buffer Underflow Error",
	  "Decode Error",
	  "Disparity Error",
	  "SKP OS Parity Error",
	  "SYNC Header Error",
	  "RX valid de-assertion",
	  "CTL SKP OS Parity Error",
	  "1st Retimer Parity Error",
	  "2nd Retimer Parity Error",
	  "Margin CRC and Parity Error"
	}
};

static const struct pcie_cntrs_group group_1 = {
	5, 10,
	{ "Detect EI Infer",
	  "Receiver Error",
	  "RX Recovery Request",
	  "N_FTS Timeout",
	  "Framing Error",
	  "Deskew Error"
	}
};

static const struct pcie_cntrs_group group_2 = {
	0, 7,
	{ "BAD TLP",
	  "LCRC Error",
	  "BAD DLLP",
	  "Replay Number Rollover",
	  "Replay Timeout",
	  "RX Nak DLLP",
	  "TX Nak DLLP",
	  "Retry LLP"
	}
};

static const struct pcie_cntrs_group group_3 = {
	0, 5,
	{ "FC Timeout",
	  "Poisoned TLP",
	  "ECRC Error",
	  "Unsupported Request",
	  "Completer Abort",
	  "Completion Timeout"
	}
};

static const struct pcie_cntrs_group group_4 = {
	0, 1,
	{ "EBUF SKP Add",
	  "EBUF SKP Del"
	}
};

static const struct pcie_cntrs_group group_5 = {
	0, 13,
	{ "L0 to Recovery Entry",
	  "L1 to Recovery Entry",
	  "TX L0s Entry",
	  "RX L0s Entry",
	  "ASPM L1 Reject",
	  "L1 Entry",
	  "L1 CPM",
	  "L1.1 Entry",
	  "L1.2 Entry",
	  "L1 Short Duration",
	  "L1.1 Abort",
	  "L2 Entry",
	  "Speed Change",
	  "Link Width Change",
	}
};

static const struct pcie_cntrs_group group_6 = {
	0, 6,
	{ "TX Ack DLLP",
	  "TX Update FC DLLP",
	  "RX ACK DLLP",
	  "RX Update FC DLLP",
	  "RX Nullified TLP",
	  "TX Nullified TLP",
	  "RX Duplicate TLP"
	}
};

static const struct pcie_cntrs_group group_7 = {
	0, 23,
	{ "TX Memory Write",
	  "TX Memory Read",
	  "TX Configuration Write",
	  "TX Configuration Read",
	  "TX IO Write",
	  "TX IO Read",
	  "TX Completion w/o data",
	  "TX Completion w/ data",
	  "TX Message TLP",
	  "TX Atomic",
	  "TX TLP with Prefix",
	  "RX Memory Write",
	  "RX Memory Read",
	  "RX Configuration Write",
	  "RX Configuration Read",
	  "RX IO Write",
	  "RX IO Read",
	  "RX Completion without data",
	  "RX Completion w data",
	  "RX Message TLP",
	  "RX Atomic",
	  "RX TLP with Prefix",
	  "TX CCIX TLP",
	  "RX CCIX TLP"
	}
};

static const struct pcie_cntrs_group *groups[8] = {
	&group_0, &group_1, &group_2, &group_3,
	&group_4, &group_5, &group_6, &group_7
};

/* Enable and clear counters for all lanes. */
static void enable_pcie_counters(struct cass_dev *hw)
{
	unsigned int lane;

	for (lane = 0; lane < NUM_LANES; lane++) {
		u32 val = lane << 8 | COUNTERS_ENABLE_ALL | COUNTERS_CLEAR_ALL;

		pci_write_config_dword(hw->cdev.pdev,
				       hw->pcie_mon.event_counter_control, val);
	}
}

/* Clear a single lane */
static void clear_counter_events(struct cass_dev *hw, u8 lane)
{
	u32 val = (lane << 8) | COUNTERS_CLEAR_ALL;

	pci_write_config_dword(hw->cdev.pdev,
			       hw->pcie_mon.event_counter_control, val);
}

/* Check whether events in a particular group are above an acceptable
 * threshold.
 */
static int check_group(struct cass_dev *hw, u8 group, int first, int last,
			unsigned int acceptable)
{
	u8 event;
	int err = 0;

	for (event = first; event <= last; event++) {
		u32 val = (group << 24) | (event << 16);

		pci_write_config_dword(hw->cdev.pdev,
				       hw->pcie_mon.event_counter_control, val);
		pci_read_config_dword(hw->cdev.pdev,
				      hw->pcie_mon.event_counter_data, &val);

		err += val;

		if (val > acceptable) {
			const struct pcie_cntrs_group *ginfo = groups[group];

			/* TODO: something that wouldn't spam logs */
			cxidev_err(&hw->cdev, "PCIe error: %s=%u\n",
				   ginfo->names[event - ginfo->first], val);
		}
	}

	return err;
}

/* Monitor groups with correctable and uncorrectable errors.
 */
static void pcie_monitoring_task(struct work_struct *work)
{
	struct cass_dev *hw =
		container_of(work, struct cass_dev, pcie_mon.task.work);

	/* Uncorrectable errors. None acceptable. */
	hw->pcie_mon.uncorr_err += check_group(hw, 3, 0, 5, 0);

	/* Correctable errors. */
	if (hw->pcie_mon.corr_err_min) {
		hw->pcie_mon.corr_err += check_group(hw, 1, 7, 7,
						     hw->pcie_mon.corr_err_min);
		hw->pcie_mon.corr_err += check_group(hw, 2, 0, 7,
						     hw->pcie_mon.corr_err_min);
		hw->pcie_mon.corr_err += check_group(hw, 5, 0, 0,
						     hw->pcie_mon.corr_err_min);
	}

	clear_counter_events(hw, 0);

	mod_delayed_work(system_wq, &hw->pcie_mon.task, 60 * HZ);
}

void start_pcie_monitoring(struct cass_dev *hw)
{
	int pos;

	/* Get Vendor Specific Information block */
	pos = 0;
	while ((pos = pci_find_next_ext_capability(hw->cdev.pdev, pos,
						   PCI_EXT_CAP_ID_VNDR))) {
		u32 header;

		pci_read_config_dword(hw->cdev.pdev,
				      pos + PCI_VNDR_HEADER, &header);

		if (PCI_VNDR_HEADER_ID(header) == 2 &&
		    PCI_VNDR_HEADER_REV(header) == 4 &&
		    PCI_VNDR_HEADER_LEN(header) == 0x100) {
			hw->pcie_mon.event_counter_control = pos + 8;
			hw->pcie_mon.event_counter_data = pos + 12;
			break;
		}
	}

	if (!hw->pcie_mon.event_counter_control) {
		cxidev_err(&hw->cdev, "PCIe monitoring is not available\n");
		return;
	}

	/* Set the number of acceptable correctable errors per
	 * minute depending on the link speed.
	 */
	switch (hw->pcie_link_speed) {
	case PCIE_SPEED_2_5GT:
		hw->pcie_mon.corr_err_min = 5;
		break;
	case PCIE_SPEED_5_0GT:
		hw->pcie_mon.corr_err_min = 10;
		break;
	case PCIE_SPEED_8_0GT:
		hw->pcie_mon.corr_err_min = 15;
		break;
	case PCIE_SPEED_16_0GT:
		hw->pcie_mon.corr_err_min = 31;
		break;
	case CASS_SPEED_20_0GT:
		hw->pcie_mon.corr_err_min = 40;
		break;
	case CASS_SPEED_25_0GT:
		hw->pcie_mon.corr_err_min = 48;
		break;
	case CASS_SPEED_32_0GT:
		hw->pcie_mon.corr_err_min = 61;
		break;
	}

	enable_pcie_counters(hw);

	INIT_DELAYED_WORK(&hw->pcie_mon.task, pcie_monitoring_task);
	mod_delayed_work(system_wq, &hw->pcie_mon.task, 60 * HZ);
}

void stop_pcie_monitoring(struct cass_dev *hw)
{
	if (!hw->pcie_mon.event_counter_control)
		return;

	cancel_delayed_work_sync(&hw->pcie_mon.task);
}
